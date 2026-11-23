"""Continue independent standard-model stages for a clearly identified review set."""
from pathlib import Path
import shutil

import cv2
from PIL import Image, ImageOps

from . import iphone17, iphone17_buttons, iphone17_port, iphone17_structure as structure
from . import pipeline, workspace as w
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
                from .geometry_continuation import recover_plate
                try:
                    image, proof = recover_plate(matte, face, exc)
                except ValueError:
                    image, state = matte, 'source_matte'
                else:
                    notes[-1]['candidate_status'] = 'degraded_repeated_stand_contact'
                    w.save(review/'checks'/(face+'-geometry-contact.json'), proof)
                    state = 'contact_candidate'
            image.save(path)
            files[face], states[face] = path, state
    return files, states


def side_material(source, side):
    image = Image.open(source).convert('RGBA')
    notes = []
    try:
        rows, _, _ = structure.antenna_rows(image, [int(image.width*.4), int(image.width*.6)])
        lines = structure.per_column_boundaries(image, rows)
    except ValueError:
        lines = []
        notes.append('天线边缘不清，按实拍边框整体调光')
    try:
        boxes = structure.hardware(image, side)
    except ValueError:
        boxes = []
        notes.append('按键轮廓不清，按实拍边框整体调光')
    return {side: dict(antenna_lines=lines, hardware_boxes=boxes, material_notes=notes)}


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
    degraded_lighting = {}
    files, stages = geometry_candidates(case, review, manifest, sku, notes)
    stages = dict(stages)
    selected = dict(files)
    frame_alignment = {}
    geometry_states = dict(stages)
    for face in pipeline.FACES:
        if stages[face] not in ('geometry', 'contact_candidate'):
            continue
        try:
            materials = side_material(files[face], face) if face in ('left','right') else {}
            check = iphone17.light_face(files[face], face, review, review/'lighting', materials)
            w.save(review/'checks'/(face+'-lighting.json'), check)
            selected[face] = review/'lighting'/(face+'.png')
            stages[face] = 'lighting'
        except (ValueError,RuntimeError,cv2.error) as exc:
            notes.append(dict(stage='lighting', face=face, error=str(exc)))
            from .lighting_continuation import recover
            fallback = review/'degraded-lighting'/(face+'.png')
            try:
                evidence = recover(files[face], face, fallback, review/'material-inputs'/face)
                w.save(review/'checks'/(face+'-degraded-lighting.json'), evidence)
            except (ValueError, RuntimeError, cv2.error) as retry:
                notes[-1]['lighting_fallback_error'] = str(retry)
                continue
            notes[-1]['degraded_lighting'] = True
            degraded_lighting[face] = evidence
            selected[face] = fallback
            stages[face] = 'degraded_lighting'
        if face in ('left','right') and not any(n.get('face') == 'back' and
                                               n.get('stage') == 'geometry' for n in notes):
            try:
                spec = structure.propose(files, sides=(face,))[face]
                aligned, proof = structure.apply(Image.open(selected[face]), spec)
                w.save(review/'checks'/(face+'-antenna.json'), proof)
                selected[face] = review/'checks'/(face+'-antenna.png')
                aligned.save(selected[face])
                frame_alignment[face] = (spec,proof)
                stages[face] = 'antenna'
                buttons = iphone17_buttons.propose(files['back'], selected[face], face, case/'work/geometry-records/report.json')
                image, checks = iphone17_buttons.apply(aligned, buttons)
                w.save(review/'checks'/(face+'-buttons.json'), checks)
                selected[face] = review/'checks'/(face+'-buttons.png')
                image.save(selected[face])
                stages[face] = 'antenna_and_buttons'
            except iphone17_buttons.SourceEvidenceError:
                raise
            except (ValueError,RuntimeError,cv2.error) as exc:
                notes.append(dict(stage='buttons' if stages[face]=='antenna' else 'antenna', face=face, error=str(exc)))
        elif face in ('left','right'):
            notes.append(dict(stage='antenna',face=face,error='Rear geometry unavailable for alignment'))
        elif face == 'bottom':
            try:
                mapping, proof = iphone17_port.mapping(Image.open(files[face]))
                image = iphone17_port.apply(Image.open(selected[face]), mapping)
                bottom_proof = proof
                selected[face] = review/'checks'/'bottom.png'
                image.save(selected[face])
                frame_alignment[face] = mapping
                stages[face] = 'bottom_registration'
            except (ValueError,RuntimeError,cv2.error) as exc:
                notes.append(dict(stage='bottom_registration', face=face, error=str(exc)))
    from .iphone17_model_finish import FRAMES, finish_frames
    frame_finish = None
    try:
        unavailable = [f for f in FRAMES if geometry_states[f] != 'geometry']
        if unavailable:
            raise ValueError('接缝缺少已确认的边框几何：' + '、'.join(unavailable))
        for f in FRAMES:
            if not (review/'material-inputs'/f/'main_surface.png').is_file():
                raise ValueError('接缝缺少本机边框金属蒙版：' + f)
        finished = review/'frame-finish'
        frame_finish = finish_frames(selected,review,finished,review/'frame-seam-records',
                                     review_alignment=frame_alignment)
        selected.update({f:finished/(f+'.png') for f in FRAMES})
    except ValueError as exc:
        notes.append(dict(stage='seam',error=str(exc)))
    from .iphone17_back_finish import finish_back
    camera_finish = None
    back_edges = None
    try:
        if geometry_states['back'] != 'geometry':
            raise ValueError('背图轮廓尚未确认，无法校验双镜位置和顶边')
        aligned_back, camera_finish = finish_back(Image.open(selected['back']))
        back_edges = camera_finish.get('canvas_edge_normalization')
        back_path = review/'checks'/'back-finish.png'
        aligned_back.save(back_path)
        camera_finish['output_sha256'] = w.sha(back_path)
        selected['back'] = back_path
    except ValueError as exc:
        notes.append(dict(face='back', stage='camera_registration', error=str(exc)))
        # Canvas margins can be corrected independently of lens acceptance.
        from .back_geometry import normalize_back_edges
        try:
            edge_candidate, back_edges = normalize_back_edges(Image.open(selected['back']))
            edge_path = review/'checks'/'back-canvas-edges.png'
            edge_candidate.save(edge_path)
            selected['back'] = edge_path
        except ValueError as edge_error:
            notes.append(dict(face='back', stage='back_canvas_edges', error=str(edge_error)))
    hashes = {}
    for face, source in selected.items():
        dest = review/'candidates'/(face+'.png')
        shutil.copy2(source, dest)
        hashes[face] = w.sha(dest)
        if hashes[face] != w.sha(source):
            raise ValueError('候选图复制校验失败')
    if stages['bottom']=='bottom_registration':
        try:
            iphone17_port.verify_export(review/'candidates'/'bottom.png',bottom_proof)
            w.save(review/'checks'/'bottom-registration.json',bottom_proof)
        except ValueError as exc:
            notes.append(dict(stage='bottom_registration',face='bottom',error=str(exc)))
            stages['bottom']='bottom_registration_unverified'
    w.check_sources(manifest)
    warnings = [dict(code=n['stage']+'_position_retained',face=n['face'],error=n['error'],
                     message=('天线' if n['stage']=='antenna' else '按键')+'保留实拍位置')
                for n in notes if n['stage'] in ('buttons','antenna')]
    notes = [n for n in notes if n['stage'] not in ('buttons','antenna')]
    report = dict(item_effect_warnings=warnings,sku=sku, model='iphone17', face_processing=stages, blocked_checks=notes,
                  candidate=str(review/'candidates'), final_sha256=hashes,
                  degraded_lighting=degraded_lighting,model_frame_finish=frame_finish,
                  model_camera_registration=camera_finish,
                  back_canvas_edge_normalization=back_edges,
                  degraded_geometry=[n['face'] for n in notes if n.get('candidate_status')],
                  visual_review_passed=False, model_uv_verified=False)
    w.save(review/'review.json', report)
    return report
