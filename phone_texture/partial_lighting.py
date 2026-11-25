"""Continue verified faces after a different face can't be cropped safely."""
import contextlib
from pathlib import Path

import cv2

from . import auto_material, lighting, pipeline, shadow_lift_profile, workspace as w
from .automatic import measure_masks
from .geometry_review import write
from .lighting_continuation import recover
from .source_preservation import require_material_boundary


def finish(case, sku, manifest, geometry, files, failures, light_profile, protection_policy, progress):
    out = case/'partial-lighting'
    out.mkdir()
    stages = {}
    blocks = list(failures)
    for face in pipeline.FACES:
        if face not in files:
            stages[face] = dict(lighting='unprocessed', selected_stage='unavailable_geometry')
            continue
        progress('lighting_partial', face, 6+len(stages))
        source = files[face]
        row = next(r for r in geometry['images'] if r['face']==face)
        if w.sha(source) != row['output_sha256'] or row['source_sha256'] != manifest['skus'][sku][face]['sha256']:
            raise ValueError('Partial lighting source does not match the current crop')
        assets = case/'work/partial-material-inputs'/face
        planned = case/'work/partial-lighting-attempts'/face
        planned.mkdir(parents=True)
        standard_error = None
        try:
            job, _, allowed, paper = auto_material.propose_job(
                source, face, assets, companions=files, protection_policy=protection_policy)
            require_material_boundary(job)
            job, _ = shadow_lift_profile.apply(job, face, assets, light_profile)
            job.update(output_dir=str(planned), records_dir=str(case/'work/partial-lighting-records'/face))
            spec = planned/'job.json'
            write(spec, job)
            with (planned/'render.log').open('w') as log, contextlib.redirect_stdout(log):
                lighting.run('aluminum', spec)
            lit = planned/(face+'.png')
            proof = measure_masks(source, lit, allowed, paper)
            # A valid no-op should be recorded, not reported as a repair.
            target = out/(face+'.png')
            import shutil
            shutil.copy2(lit, target)
            if w.sha(lit) != w.sha(target):
                raise RuntimeError('Partial lighting changed during candidate export')
            status = 'effect_warning' if proof['changed_pixels'] else 'checked_unchanged'
            stages[face] = dict(lighting=status, selected_stage='lighting',
                                lighting_sha256=w.sha(target), lighting_file=str(target),
                                effect_scope='not_full_device_verified', pixel_checks=proof)
            continue
        except (ValueError, RuntimeError, cv2.error) as error:
            standard_error = str(error)
        target = out/(face+'.png')
        try:
            proof = recover(source, face, target, assets, metal_back=True,
                                      review_camera=light_profile == 'accepted-shadow-lift-v2')
        except (ValueError, RuntimeError, cv2.error) as error:
            stages[face] = dict(lighting='blocked', selected_stage='geometry', lighting_error=standard_error,
                                lighting_fallback_error=str(error))
            blocks.append(dict(stage='lighting', face=face, error=standard_error, fallback_error=str(error)))
        else:
            stages[face] = dict(lighting='degraded', selected_stage='degraded_lighting',
                                lighting_sha256=w.sha(target), lighting_file=str(target), degraded_lighting=proof)
            blocks.append(dict(stage='lighting', face=face, error=standard_error, degraded_lighting=True))
    w.check_sources(manifest)
    return dict(status='partial_candidates_needs_review', six_faces_generated=False, six_faces_processed=False,
                error='; '.join(x['face']+': '+x['error'] for x in failures),
                candidate=None, partial_lighting_directory=str(out), per_face_stages=stages,
                stage_status=dict(geometry='partial', lighting='partial', structure='blocked', seam='blocked'),
                blocked_checks=blocks, review_warnings=[dict(face=x['face'],code=x['stage']+'_blocked',message=x['error']) for x in blocks],
                face_processing={f: r['selected_stage'] for f,r in stages.items()}, source_hashes_unchanged=True)
