"""Continue independent standard-model stages for a clearly identified review set."""
from pathlib import Path
import shutil

import cv2
from PIL import Image, ImageOps

from . import iphone17, iphone17_buttons, iphone17_port, iphone17_structure as structure
from . import pipeline, workspace as w
from .auto_geometry import frame_geometry
from phone_normalizer.api import Normalizer, NormalizerConfig


def geometry_candidates(case, review, manifest, sku, notes):
    records = case/'work'/'geometry-records'
    evidence_path = records/'report.json'
    rows = w.read(evidence_path)['images'] if evidence_path.exists() else []
    rows = {r['face']: r for r in rows if r['sku'] == sku}
    files = {}
    states = {}
    with Normalizer(NormalizerConfig(remove_support=False, deskew=False, tight_crop=False)) as normalizer:
        for face in pipeline.FACES:
            source = manifest['skus'][sku][face]
            staged = records/'staged'/sku/(face+'.png')
            record = rows.get(face, {})
            if (staged.exists() and record.get('source_sha256') == source['sha256']
                    and record.get('output_sha256') == w.sha(staged)):
                files[face], states[face] = staged, 'geometry'
                continue
            raw = ImageOps.exif_transpose(Image.open(source['snapshot'])).convert('RGB')
            cache = records/'mattes'/(face+'.png')
            identity = dict(source_sha256=source['sha256'], max_side=1536 if face in ('front','back') else 1024,
                            model='ZhengPeng7/BiRefNet_dynamic')
            if cache.exists() and cache.with_suffix('.json').exists() and w.read(cache.with_suffix('.json')) == identity:
                matte = Image.open(cache).convert('RGBA')
            else:
                matte = iphone17.segment_face(normalizer, raw, identity['max_side'])
            path = review/'geometry'/(face+'.png')
            try:
                image, _ = iphone17.geometry_face(matte, raw, face, None)
                state = 'geometry'
            except ValueError as exc:
                notes.append(dict(face=face, stage='geometry', error=str(exc)))
                image, state = matte, 'source_matte'
                if face == 'bottom':
                    try:
                        image, _ = frame_geometry(matte, face, source_image=raw, separate_support_edge=False)
                        state = 'frame_without_paper_check'
                    except ValueError:
                        pass
            image.save(path)
            files[face], states[face] = path, state
    return files, states


def side_material(source, side):
    image = Image.open(source).convert('RGBA')
    rows, _, _ = structure.antenna_rows(image, [int(image.width*.4), int(image.width*.6)])
    return {side: dict(antenna_lines=structure.per_column_boundaries(image, rows),
                       hardware_boxes=structure.hardware(image, side))}


def continue_review(case, sku, review_dir=None):
    import torch
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    manifest = w.read(case/'work/intake.json')
    if set(manifest['skus']) != {sku}:
        raise ValueError('标准版续接的原图与机台不符')
    w.check_sources(manifest)
    review = review_dir if review_dir is not None else case/'independent-review'
    pipeline.fresh(review)
    review.mkdir()
    for name in ('geometry','lighting','candidates','checks'):
        (review/name).mkdir()
    notes = []
    files, stages = geometry_candidates(case, review, manifest, sku, notes)
    stages = dict(stages)
    selected = dict(files)
    for face in pipeline.FACES:
        if stages[face] != 'geometry':
            continue
        try:
            materials = side_material(files[face], face) if face in ('left','right') else {}
            check = iphone17.light_face(files[face], face, review, review/'lighting', materials)
            w.save(review/'checks'/(face+'-lighting.json'), check)
            selected[face] = review/'lighting'/(face+'.png')
            stages[face] = 'lighting'
        except (ValueError,RuntimeError,cv2.error) as exc:
            notes.append(dict(stage='lighting', face=face, error=str(exc)))
            continue
        if face in ('left','right') and stages['back'] != 'source_matte':
            try:
                spec = structure.propose(files, sides=(face,))[face]
                aligned, proof = structure.apply(Image.open(selected[face]), spec)
                w.save(review/'checks'/(face+'-antenna.json'), proof)
                selected[face] = review/'checks'/(face+'-antenna.png')
                aligned.save(selected[face])
                stages[face] = 'antenna'
                buttons = iphone17_buttons.propose(files['back'], selected[face], face, case/'work/geometry-records/report.json')
                image, checks = iphone17_buttons.apply(aligned, buttons)
                w.save(review/'checks'/(face+'-buttons.json'), checks)
                selected[face] = review/'checks'/(face+'-buttons.png')
                image.save(selected[face])
                stages[face] = 'antenna_and_buttons'
            except (ValueError,RuntimeError,cv2.error) as exc:
                notes.append(dict(stage='buttons' if stages[face]=='antenna' else 'antenna', face=face, error=str(exc)))
        elif face in ('left','right'):
            notes.append(dict(stage='antenna',face=face,error='Rear geometry unavailable for alignment'))
        elif face == 'bottom':
            try:
                mapping, proof = iphone17_port.mapping(Image.open(files[face]))
                image = iphone17_port.apply(Image.open(selected[face]), mapping)
                port = iphone17_port.measure(image)['port']
                error = float(port[0]-(image.width-1)/2)
                if abs(error)>2.5:
                    raise ValueError('Final rendered USB-C centre failed independent measurement')
                w.save(review/'checks'/'bottom-registration.json', dict(proof, final_rendered_center_error_px=error))
                selected[face] = review/'checks'/'bottom.png'
                image.save(selected[face])
                stages[face] = 'bottom_registration'
            except (ValueError,RuntimeError,cv2.error) as exc:
                notes.append(dict(stage='bottom_registration', face=face, error=str(exc)))
    hashes = {}
    for face, source in selected.items():
        dest = review/'candidates'/(face+'.png')
        shutil.copy2(source, dest)
        hashes[face] = w.sha(dest)
        if hashes[face] != w.sha(source):
            raise ValueError('候选图复制校验失败')
    w.check_sources(manifest)
    report = dict(sku=sku, model='iphone17', face_processing=stages, blocked_checks=notes,
                  candidate=str(review/'candidates'), final_sha256=hashes,
                  visual_review_passed=False, model_uv_verified=False)
    w.save(review/'review.json', report)
    return report
