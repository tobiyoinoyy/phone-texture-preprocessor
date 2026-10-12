import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from texture_alignment.align_textures import apply, prepare, warp_rows, image_size, target_points
from texture_alignment.seam_targets import transfer_row


class MappingTests(unittest.TestCase):
    def test_uv_seam_keeps_unmatched_knots_and_pixel_center_convention(self):
        # Endpoints alone imply v=.5; the actual side has its own interior knot.
        b=[[0,0],[1,1]];s=[[0,0],[.5,.7],[1,1]]
        self.assertAlmostEqual(transfer_row(49.5,100,200,b,s),139.5)
        self.assertAlmostEqual(transfer_row(49.5,100,200,[[0,1],[1,0]],s),139.5)
        with self.assertRaisesRegex(ValueError,'outside reviewed seam'):
            transfer_row(101,100,200,b,s)
    def test_per_sku_targets_do_not_fall_back_to_reference_sku(self):
        config={'reference':'sku0','points':{'sku0':{'left':[10,20]}},
                'targets_by_sku':{'sku1':{'left':[12,24]}}}
        self.assertEqual(target_points(config,'sku1','left'),[12,24])
        with self.assertRaisesRegex(ValueError,'缺少该 SKU'):
            target_points(config,'sku0','left')
        config.pop('targets_by_sku')
        self.assertEqual(target_points(config,'sku1','left'),[10,20])
    def test_integer_landmarks_endpoints_and_dimensions(self):
        pixels = np.zeros((101, 7, 4), dtype=np.uint8)
        pixels[..., 3] = 255
        pixels[0, :, :3] = 18
        pixels[-1, :, :3] = 222
        pixels[25, :, 0] = 255
        pixels[75, :, 1] = 255
        result = np.array(warp_rows(Image.fromarray(pixels), [25, 75], [30, 60]))
        self.assertEqual(result.shape, pixels.shape)
        np.testing.assert_array_equal(result[30], pixels[25])
        np.testing.assert_array_equal(result[60], pixels[75])
        np.testing.assert_array_equal(result[[0, -1]], pixels[[0, -1]])

    def test_identity_is_byte_exact_including_transparent_rgb(self):
        pixels = np.random.default_rng(7).integers(0, 256, (101, 12, 4), dtype=np.uint8)
        result = np.array(warp_rows(Image.fromarray(pixels), [30, 60], [30, 60]))
        np.testing.assert_array_equal(result, pixels)

    def test_no_black_halo_at_transparency(self):
        pixels = np.zeros((11, 3, 4), dtype=np.uint8)
        pixels[5] = [240, 120, 60, 255]
        result = np.array(warp_rows(Image.fromarray(pixels), [5], [6]))
        translucent = result[(result[..., 3] > 0) & (result[..., 3] < 255)]
        self.assertGreater(len(translucent), 0)
        np.testing.assert_array_equal(translucent[:, :3], np.tile([240, 120, 60], (len(translucent), 1)))

    def test_invalid_points_rejected(self):
        im = Image.new('RGBA', (10, 100))
        for bad in ([], [10, 10], [60, 30], [0, 20], [20, 99], [float('nan')]):
            with self.assertRaises(ValueError):
                warp_rows(im, bad, [30, 60])
        with self.assertRaises(ValueError):
            warp_rows(im, [20], [30, 60])

    def test_sloped_antenna_and_fixed_button_together(self):
        a = np.zeros((101, 11, 4),dtype=np.uint8)
        a[...,3] = 255
        x = np.arange(11)
        a[20+x,x] = [255,0,0,255]
        a[60,:] = [0,255,0,255]
        anchors = np.stack([20+x,np.full(11,60)])
        b = np.asarray(warp_rows(Image.fromarray(a),anchors,[25,60]))
        np.testing.assert_array_equal(b[25],np.tile([255,0,0,255],(11,1)))
        np.testing.assert_array_equal(b[60],a[60])
        np.testing.assert_array_equal(b[[0,-1]],a[[0,-1]])
        self.assertEqual(b.shape,a.shape)

    def test_full_batch_pipeline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sources = root / 'input'
            work = root / 'work'
            before = {}
            for sku, size in [('sku0', (15, 101)), ('sku1', (20, 121))]:
                (sources / sku).mkdir(parents=True)
                for side in ('front', 'back', 'top', 'bottom'):
                    path = sources / sku / f'{side}.png'
                    Image.new('RGBA', (33, 55), (12, 34, 56, 123)).save(path)
                    before[path] = path.read_bytes()
                for side in ('left', 'right'):
                    path = sources / sku / f'{side}.png'
                    Image.new('RGBA', size, (230, 130, 80, 255)).save(path)
                    before[path] = path.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(argparse.Namespace(input=sources, output=work, reference='sku0', height=None, width=None))
            manifest = json.loads((work / 'manifest.json').read_text())
            self.assertEqual(len(manifest['images']), 4)
            for item in manifest['images']:
                self.assertEqual(image_size(work / item['file']), (15, 101))
            config = json.loads((work / 'landmarks.json').read_text())
            config['points'] = {'sku0': {'left': [25, 40, 50, 70], 'right': [30, 60]},
                                'sku1': {'left': [27, 42, 51, 71], 'right': [35, 65]}}
            config['targets_by_sku'] = {'sku0':{'left':[25,40,50,70],'right':[30,60]},
                                        'sku1':{'left':[23,40,50,74],'right':[29,64]}}
            (work / 'landmarks.json').write_text(json.dumps(config))
            with contextlib.redirect_stdout(io.StringIO()):
                apply(argparse.Namespace(work=work, landmarks=work / 'landmarks.json', output=None))
            for item in manifest['images']:
                with Image.open(work / 'aligned' / item['sku'] / f'{item["side"]}.png') as result:
                    self.assertEqual(result.size, (15, 101))
                    self.assertEqual(result.mode, 'RGBA')
            for path, content in before.items():
                self.assertEqual(path.read_bytes(), content)
                if path.stem in ('front', 'back', 'top', 'bottom'):
                    self.assertEqual((work / 'aligned' / path.parent.name / path.name).read_bytes(), content)
            self.assertTrue((work / 'aligned' / 'review-left-1.jpg').exists())
            report=json.loads((work/'aligned/alignment-report.json').read_text())
            for row in report['images']:
                self.assertEqual(row['target_y'],config['targets_by_sku'][row['sku']][row['side']])
            with self.assertRaises(ValueError):
                apply(argparse.Namespace(work=work, landmarks=work / 'landmarks.json', output=None))
            config['reference_files']=[{'path':str(sources/'sku0/back.png'),'sha256':'obsolete'}]
            (work/'landmarks.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'UV 基准文件已变化'):
                apply(argparse.Namespace(work=work,landmarks=work/'landmarks.json',output=root/'blocked'))
            self.assertFalse((root/'blocked').exists())


if __name__ == '__main__':
    unittest.main()
