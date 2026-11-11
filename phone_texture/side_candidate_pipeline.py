"""Current-job side review continuation; no saved experiment selection."""
import copy
from pathlib import Path
import traceback

from . import workspace as w
from .geometry_review import write


def binding(path, root):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise RuntimeError('Current side evidence escaped the current case')
    return dict(path=str(path), sha256=w.sha(path))


def report(value, root, key):
    if not isinstance(value, dict):
        value = w.read(Path(value))
    path = value.get(key) or value.get('result_path')
    if not path:
        raise RuntimeError('Current side stage did not persist its report')
    bound = binding(path, root)
    saved = w.read(Path(path))
    extras = set(value) - set(saved)
    allowed = {'result_path', 'result_sha256', 'observation_path', 'observation_sha256'}
    if (not extras <= allowed or any(value.get(name) != item for name, item in saved.items())
            or any(value[name] != bound['path'] for name in extras if name.endswith('_path'))
            or any(value[name] != bound['sha256'] for name in extras if name.endswith('_sha256'))):
        raise RuntimeError('Returned side stage differs from its persisted report')
    return value, bound


def normal_lighting(source, candidate, face, checks, case, folder, sku):
    """Expose already completed current-job checks without rendering again."""
    from .automatic import implementation_manifest
    if checks.get('source_sha256') != w.sha(source) or checks.get('output_sha256') != w.sha(candidate):
        raise RuntimeError('Current normal lighting no longer matches its pixel checks')
    repo = Path(__file__).resolve().parents[1]
    inputs = {str(Path(p).resolve()): w.sha(p) for p in
              (source, candidate, case/'work/lighting-checks.json', case/'work/lighting-jobs.json')}
    path = folder/'normal-lighting-evidence.json'
    jobs = w.read(case/'work/lighting-jobs.json')
    preparation = Path(jobs[sku][face]).resolve(strict=True)
    binding(preparation, case)
    inputs[str(preparation)] = w.sha(preparation)
    result = dict(version='current_stage_lighting_v1', face=face,
        source=str(source), source_sha256=w.sha(source), candidate=str(candidate),
        candidate_sha256=w.sha(candidate), source_geometry_preserved=True,
        eligible_for_visual_review=True, hard_failures=[], inputs_unchanged=True,
        implementation_unchanged=True, normal_stage_checks_complete=True,
        checks=copy.deepcopy(checks), input_sha256=inputs,
        code_sha256={str(repo/name): digest for name, digest in implementation_manifest(repo).items()},
        strict_structure_verified=False, material_classification_complete=False,
        quality_checks_complete=False, visual_approved=False, model_uv_verified=False,
        selected_for_delivery=False, result_path=str(path),
        artifacts=dict(result=str(path), preparation=str(preparation), candidate=str(candidate)),
        lighting_rendered_in_current_job=True, additional_render_performed=False)
    write(path, result)
    return result


def continue_sides(case, sku, geometry, geometry_files, raw_evidence, reference,
                   selected, selected_hashes, per_face, checks, progress, verify_sources):
    """Try each side once and retain independently checked successful stages.

    Runtime/I/O/integrity errors propagate. A typed local evidence rejection can
    leave a selected earlier stage intact, with the failure recorded explicitly.
    """
    from . import side_candidate_structure as structure
    from . import side_candidate_lighting as illumination

    case = Path(case).resolve()
    root = case/'work/side-candidate-review'
    root.mkdir(parents=True, exist_ok=False)
    geometry_path = root/'current-geometry.json'
    write(geometry_path, geometry)
    outcome = dict(schema='current_side_review/v1', requested=True,
        execution_mode='fresh_current_sources', faces={},
        strict_structure_verified=False, material_classification_complete=False,
        quality_checks_complete=False, visual_approved=False, model_uv_verified=False,
        selected_for_delivery=False, saved_candidates_reused=False,
        geometry_changed=False, seam_executed=False)
    light_reports, structure_reports, failures = {}, {}, []
    for face in ('left', 'right'):
        verify_sources()
        progress('alignment', face, 12)
        folder = root/face
        folder.mkdir()
        item = dict(selected_lighting=False, selected_structure=False,
                    previous_stage=per_face[face]['selected_stage'])
        outcome['faces'][face] = item
        phase = 'observation'
        try:
            if face not in raw_evidence:
                item.update(state='unavailable', reason='Current source projection was not available')
                failures.append(dict(stage='side_review_observation', face=face,
                    kind='unsupported_source_evidence', error=item['reason']))
                continue
            observed = structure.observe_current(geometry_files[face], face, folder/'observation',
                raw_context=raw_evidence[face], geometry_files=geometry_files,
                geometry_report=geometry_path, body_reference=reference, sku=sku)
            observed, item['observation'] = report(observed, case, 'observation_path')
            phase = 'lighting'
            if face in checks:
                lighting_result = normal_lighting(geometry_files[face], selected[face],
                    face, checks[face], case, folder, sku)
                lighting_via = 'normal_stage_current_job'
            else:
                lighting_result = illumination.run(geometry_files[face], face, folder/'lighting',
                    endpoint_report=Path(item['observation']['path']), protection_policy='none',
                    lighting_profile='accepted-shadow-lift-v2')
                lighting_via = 'new_review_render'
            lighting_result, light_bound = report(lighting_result, case, 'result_path')
            if (lighting_result.get('eligible_for_visual_review') is not True or
                    lighting_result.get('hard_failures') != []):
                raise RuntimeError('A rejected lighting candidate was returned as selectable')
            light_path = Path(lighting_result['candidate'])
            binding(light_path, case)
            if (lighting_result['source_sha256'] != w.sha(geometry_files[face]) or
                    lighting_result['candidate_sha256'] != w.sha(light_path)):
                raise RuntimeError('Current lighting candidate identity differs')
            item['lighting'] = dict(light_bound, executed=True, via=lighting_via)
            item['selected_lighting'] = True
            light_reports[face] = lighting_result
            selected[face] = light_path
            selected_hashes[face] = lighting_result['candidate_sha256']
            per_face[face].update(lighting_before_side_review=per_face[face]['lighting'],
                lighting='review_applied' if lighting_via == 'new_review_render' else per_face[face]['lighting'],
                selected_stage='lighting_review' if lighting_via == 'new_review_render' else 'lighting',
                lighting_sha256=lighting_result['candidate_sha256'], lighting_file=str(light_path),
                illumination_assumption_review=lighting_via == 'new_review_render')
            phase = 'structure'
            transformed = structure.generate(lighting_result, observed, folder/'structure')
            transformed, structural_bound = report(transformed, case, 'result_path')
            if (transformed.get('eligible_for_visual_review') is not True or
                    transformed.get('hard_failures') != []):
                raise RuntimeError('A rejected structure candidate was returned as selectable')
            candidate = Path(transformed['candidate'])
            binding(candidate, case)
            if (transformed['source_sha256'] != lighting_result['candidate_sha256'] or
                    transformed['candidate_sha256'] != w.sha(candidate)):
                raise RuntimeError('Current structure candidate identity differs')
            item['structure'] = dict(structural_bound, executed=True)
            item.update(selected_structure=True, state='applied_needs_review')
            structure_reports[face] = transformed
            selected[face], selected_hashes[face] = candidate, transformed['candidate_sha256']
            reference_review = copy.deepcopy(transformed.get('reference_review') or
                                             observed.get('reference_review') or {})
            per_face[face].update(structure='review_required', seam='blocked', selected_stage='structure_review',
                structure_operation_applied=True, strict_structure_verified=False,
                strict_endpoint_evidence_complete=False, material_classification_complete=False,
                structure_review_required=True, reference_review_required=True,
                source_reference_status=reference_review.get('original_status'),
                review_evidence=dict(lighting_result=light_bound, structure_result=structural_bound,
                    observation=item['observation'], reference_review=reference_review,
                    quality_checks_complete=False, visual_approved=False, model_uv_verified=False,
                    selected_for_delivery=False, selected_for_review_export=True))
        except ValueError as error:
            # Helpers expose local rejections only after verifying their inputs.
            rejected = getattr(error, 'result', None)
            rejected_path = getattr(error, 'result_path', None)
            if (not isinstance(rejected, dict) or rejected.get('inputs_unchanged') is not True or
                    rejected.get('implementation_unchanged') is not True or not rejected_path):
                raise
            if ('result_path' in rejected and rejected['result_path'] != str(rejected_path)):
                raise RuntimeError('Rejected stage report path differs')
            rejected, rejected_binding = report(dict(rejected, result_path=str(rejected_path)), case, 'result_path')
            if rejected.get('eligible_for_visual_review') is not False or not rejected.get('hard_failures'):
                raise RuntimeError('Local stage rejection lacks its persisted failed checks')
            verify_sources()
            entry = dict(stage='side_review_'+phase, face=face, kind='review_candidate_rejected',
                error=str(error), traceback=traceback.format_exc(), evidence=rejected_binding,
                retained_stage=per_face[face]['selected_stage'])
            failures.append(entry)
            item.update(state='candidate_rejected', rejection=entry)
        verify_sources()
        write(root/'report.json', outcome)
    outcome['complete'] = all(outcome['faces'][f]['selected_structure'] for f in ('left', 'right'))
    outcome['result_path'] = str(root/'report.json')
    write(root/'report.json', outcome)
    return outcome, light_reports, structure_reports, failures
