import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

from phone_texture.pipeline import (FACES, batch_files, process_batch, prepare, apply,
                                    separate_paths, write_json, deskew_image, guided_frame)
from texture_alignment.align_textures import digest
from texture_alignment.blender_scene_guard import fingerprint


def sources(root):
    folder = root / 'SKU_01'
    folder.mkdir(parents=True)
    for face in FACES:
        im = Image.new('RGBA', (400, 80))
        ImageDraw.Draw(im).rounded_rectangle((10, 10, 389, 69), radius=10, fill=(50, 80, 120, 255))
        if face in ('left', 'right'):
            im = im.transpose(Image.Transpose.ROTATE_90)
        im.save(folder / f'{face}.png')
    return root


class UnifiedTests(unittest.TestCase):
    def test_reviewed_geometry_preserves_source_pixels_and_side_orientation(self):
        pixels=np.random.default_rng(5).integers(0,256,(60,400,4),dtype=np.uint8)
        pixels[:,:,3]=0
        pixels[10:50,10:390,3]=255
        result, info=guided_frame(Image.fromarray(pixels),'left',
                                  {'crop_xyxy':[10,10,390,50],'rotation_ccw_degrees':0})
        np.testing.assert_array_equal(np.array(result),np.rot90(pixels[10:50,10:390]))
        self.assertEqual(info['resampling_passes'],0)

    def test_changed_source_invalidates_reviewed_frame_guides(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=sources(root/'input');guide=root/'guide.json'
            write_json(guide,{'SKU_01':{'top':{'source_sha256':'stale','crop_xyxy':[0,0,400,80],'rotation_ccw_degrees':0}}})
            with self.assertRaisesRegex(ValueError,'裁切标定不一致'):
                process_batch(source,root/'output',root/'process',mode='standardize',frame_guides=guide)
            self.assertFalse((root/'output').exists())
            self.assertFalse((root/'process').exists())

    def test_deskew_delivers_six_faces_and_preserves_unselected_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = sources(root / 'input')
            hashes = {p.name: digest(p) for p in (source / 'SKU_01').iterdir()}
            process_batch(source, root / 'output', root / 'process', mode='deskew', faces=['top','bottom'])
            self.assertFalse((root / 'output').exists())
            from geometry_helpers import review_fixture
            from phone_texture.geometry_review import approve
            approve(root/'process',review_fixture(root/'process'),root/'output')
            self.assertEqual(sorted(p.name for p in (root / 'output').iterdir()), ['SKU_01'])
            for face in ('left','right','front','back'):
                self.assertEqual(hashes[face+'.png'], digest(root / 'output/SKU_01' / (face+'.png')))
            self.assertTrue((root / 'process/overview.png').exists())
            self.assertEqual(hashes, {p.name: digest(p) for p in (source / 'SKU_01').iterdir()})

    def test_side_quarter_turns_preserve_orientation(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = sources(Path(tmp) / 'input')
            for face in ('left','right'):
                im = Image.open(source / 'SKU_01' / (face+'.png'))
                result, _ = deskew_image(im, face)
                self.assertGreater(result.height, result.width)
                np.testing.assert_array_equal(np.array(result), np.array(im.crop(im.getbbox())))

    def test_failures_do_not_publish_partial_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=sources(root/'input')
            with patch('phone_texture.pipeline.deskew_image', side_effect=ValueError('bad rails')):
                with self.assertRaisesRegex(ValueError,'bad rails'):
                    process_batch(source,root/'output',root/'process',mode='deskew')
            self.assertFalse((root/'output').exists())
            self.assertFalse(json.loads((root/'process/report.json').read_text())['success'])

    def test_nested_paths_and_missing_faces_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=sources(root/'input')
            with self.assertRaises(ValueError):
                separate_paths(source,source/'process')
            (source/'SKU_01/back.png').unlink()
            with self.assertRaisesRegex(ValueError,'缺少六面'):
                batch_files(source)

    def test_alignment_uses_per_sku_scene_guard_and_clean_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=sources(root/'input'); work=root/'work'
            prepare(source,work,reference='SKU_01')
            config=json.loads((work/'landmarks.json').read_text())
            config['points']['SKU_01']={side:[30,40,300,310] for side in ('left','right')}
            cfg=root/'calibration.json'; write_json(cfg,config)
            with self.assertRaisesRegex(ValueError,'targets_by_sku'):
                apply(work,cfg,root/'output',root/'result-process')
            scene=root/'scene.json'; record={'vertices':[1,2,3]}
            write_json(scene,{'payload':record,'sha256':fingerprint(record)})
            config['scene_record']=str(scene)
            back=source/'SKU_01/back.png'
            config['reference_files']=[{'path':str(p),'sha256':digest(p)} for p in (back,scene)]
            config['targets_by_sku']=config['points']
            write_json(cfg,config)
            apply(work,cfg,root/'output',root/'result-process')
            for face in FACES:
                original=Image.open(source/'SKU_01'/f'{face}.png')
                result=Image.open(root/'output/SKU_01'/f'{face}.png')
                np.testing.assert_array_equal(np.array(original),np.array(result))
            self.assertEqual(len(list((root/'output').rglob('*'))),7)
            self.assertTrue((root/'result-process/overview.png').exists())
            back.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                apply(work,cfg,root/'new-output',root/'new-process')
            self.assertFalse((root/'new-output').exists())
