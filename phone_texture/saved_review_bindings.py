"""Read-only binding validator for saved illumination and v2/v3 side reviews.

No image imports, rendering, inference, mutation, or implicit failure fallback.
The caller supplies audit.file/audit.json and must recheck its recorded bytes
after copying. Passing means eligible for review assembly, never acceptance.
"""
import copy
import math
from pathlib import Path


FACES = ('front', 'back', 'left', 'right', 'top', 'bottom')
V2 = 'canvas_connected_nonzero_alpha_compatibility_review_v2'
V3 = 'row_alpha_mass_center_contour_sampling_review_v3'
FALSE_FLAGS = ('material_classification_complete', 'strict_endpoint_evidence_complete',
               'quality_checks_complete', 'visual_approved', 'selected_for_delivery',
               'model_uv_verified', 'six_faces_processed', 'seam_executed')
STRUCTURE_FALSE = ('structural_alignment_certified', 'strict_structure_verified', 'production_integrated')
TOLERANCE = 2.5  # Existing structure_review measurement tolerance, not a new limit.


def _require(value, message):
    if not value:
        raise ValueError(message)


def _sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _path(value):
    _require(isinstance(value, str) and Path(value).is_absolute(), 'Evidence paths must be absolute')
    return str(Path(value).resolve())


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value), 'Missing/nonfinite numeric evidence')
    return value


def _flags(record, false=(), true=()):
    for name in false:
        _require(record.get(name) is False, 'Missing or elevated review flag: ' + name)
    for name in true:
        _require(record.get(name) is True, 'Missing required recorded check: ' + name)


def _zeros(record, names):
    for name in names:
        _require(type(record.get(name)) is int and record[name] == 0, 'Nonzero/missing integrity check: ' + name)


def _entry_record(entry, field, audit):
    descriptor = entry.get(field)
    _require(isinstance(descriptor, dict) and _sha(descriptor.get('sha256')),
             'Explicit path/SHA descriptor required: ' + field)
    path = _path(descriptor.get('path'))
    value = audit.json(path, descriptor['sha256'])
    _require(isinstance(value, dict), 'Evidence must be a JSON object: ' + field)
    return path, descriptor['sha256'], value


def _bindings(mapping, audit):
    _require(isinstance(mapping, dict) and mapping, 'Nonempty input bindings required')
    normalized = {}
    for path, expected in mapping.items():
        path = _path(path)
        _require(_sha(expected), 'Invalid input SHA')
        _require(path not in normalized or normalized[path] == expected, 'Conflicting input aliases')
        audit.file(path, expected)
        normalized[path] = expected
    return normalized


def _bound(mapping, path, expected=None):
    path = _path(path)
    value = mapping.get(path)
    _require(_sha(value) and (expected is None or value == expected), 'Missing/wrong source binding: ' + path)
    return value


def _artifact(record, name, audit):
    path = _path(record['artifacts'][name])
    expected = record['artifact_sha256'].get(path)
    _require(_sha(expected), 'Artifact hash missing: ' + name)
    audit.file(path, expected)
    return path, expected


def _geometry(baseline, face):
    values = baseline.get('_geometry_sha256')
    value = values.get(face) if isinstance(values, dict) else baseline['per_face_stages'][face]['source_sha256']
    _require(_sha(value), 'Baseline geometry SHA missing: ' + face)
    return value


def _key(record, key, face):
    _require(record.get('key') == key and record.get('face') == face, 'Evidence belongs to another key/face')


def _verify_lighting(key, face, entry, baseline, audit):
    path, digest, record = _entry_record(entry, 'lighting_result', audit)
    _key(record, key, face)
    _require(record.get('state') == 'illumination_assumption_review'
             and record.get('execution_state') == 'checks_finished' and record.get('hard_failures') == [],
             'Only completed successful illumination-assumption reviews are supported')
    _flags(record, FALSE_FLAGS + ('structure_executed', 'segmentation_executed', 'structural_alignment_applied'),
           ('render_completed', 'candidate_generated', 'eligible_for_visual_review',
            'source_geometry_preserved', 'inputs_unchanged', 'lab_dependencies_unchanged'))
    inputs = _bindings(record['input_sha256'], audit)
    _bindings(record['lab_dependency_sha256'], audit)
    integrity = record['integrity']
    _require(isinstance(integrity, dict) and
             {'inputs', 'lab_dependencies', 'prepared_assets', 'frozen_r6',
              'user_service_code', 'selected_six_sources'} <= set(integrity),
             'Lighting final integrity categories are missing')
    for check in integrity.values():
        _require(isinstance(check, dict) and check.get('unchanged') is True
                 and check.get('issues', []) == [], 'Lighting final integrity is incomplete')
    identity = record['source_identity_before']
    _require(identity == record['source_identity_after'] and identity.get('six_hashes_verified') is True,
             'Lighting raw source identities changed or were not verified')
    _require(_path(identity['source']) == _path(baseline['source']), 'Lighting uses another baseline source folder')
    selected = baseline.get('_selected_source_hashes')
    _require(isinstance(selected, dict) and len(selected) == 6, 'Six selected raw-source hashes are required')
    raw_hashes = {f + '.jpg': selected.get(f + '.jpg', selected.get(f)) for f in FACES}
    _require(all(_sha(h) for h in raw_hashes.values()) and identity['hashes'] == raw_hashes,
             'Lighting raw sources differ from selected six-source identity')
    for name, expected in raw_hashes.items():
        audit.file(str(Path(baseline['source']) / name), expected)
    lighting, lighting_sha = _artifact(record, 'candidate', audit)
    before, before_sha = _artifact(record, 'before', audit)
    geometry_sha = _geometry(baseline, face)
    _require(before_sha == geometry_sha, 'Lighting before image differs from original geometry')
    checks = record['checks']
    required = {'source_contract', 'original_masks', 'original_and_tangent_effects', 'field_and_hardware_diagnostics'}
    _require(isinstance(checks, dict) and required <= set(checks), 'Required lighting check family missing')
    _flags(checks['source_contract'], true=('source_and_alpha_contract_passed',))
    _zeros(checks['source_contract'], ('alpha_changed', 'outside_allowed_changed'))
    masks = checks['original_masks']
    _zeros(masks, ('alpha_changed', 'outside_material_changed'))
    _require(masks['source_sha256'] == geometry_sha and masks['output_sha256'] == lighting_sha,
             'Lighting measured source/output hashes do not bind selected PNG')
    if 'render_candidate_sha256' in record:
        _require(record['render_candidate_sha256'] == lighting_sha, 'Render candidate hash mismatch')
    prep = record['preparation']
    prep_path = _path(record['artifacts']['preparation'])
    _require(audit.json(prep_path) == prep, 'Embedded lighting preparation differs from saved preparation')
    assets = _bindings(prep['prepared_assets_sha256'], audit)
    _bindings(prep['input_sha256'], audit)
    _require(prep['source_sha256'] == geometry_sha, 'Lighting preparation uses another geometry')
    audit.file(_path(prep['source']), geometry_sha)
    scope_path = str(Path(prep_path).parent / 'scope.json')
    scope = audit.json(scope_path, _bound(assets, scope_path))
    _require(scope['source_sha256'] == geometry_sha, 'Lighting scope source mismatch')
    _flags(scope, false=('material_classification_complete', 'structural_alignment_certified'))
    assumption = scope['illumination_assumption_review']
    _require(isinstance(assumption, dict) and assumption.get('policy') == 'metal_fit_shared_low_frequency_L_review_v1',
             'Lighting assumption scope is missing or unknown')
    _flags(assumption, ('material_classification_complete', 'strict_endpoint_evidence_complete',
                        'structural_alignment_applied', 'quality_checks_complete'), ('source_geometry_unchanged',))
    zones, effects = scope['local_checks'], checks['original_and_tangent_effects']
    _require(isinstance(zones, dict) and zones and isinstance(effects, dict) and set(zones) == set(effects),
             'Lighting scoped effect checks are absent/incomplete')
    for name, zone in zones.items():
        effect = effects[name]
        _flags(effect, true=('effect_check_passed',))
        audit.file(_path(zone['mask']), effect['witness_sha256'])
        for source_limit, measured, reported in (
                ('max_added_step_L', 'max_added_step_L', 'limit_L'),
                ('max_pixel_added_step_L', 'max_pixel_added_step_L', 'pixel_limit_L')):
            if source_limit in zone:
                limit = _number(zone[source_limit])
                _require(_number(effect[reported]) == limit and _number(effect[measured]) <= limit
                         and _number(effect['pair_count']) > 0, 'Incomplete/failed scoped effect: ' + name)
    return path, digest, record, lighting, lighting_sha, inputs, assumption


def _centers(result, preparation, version):
    plan = preparation['mapping']
    _require(plan.get('policy') == 'integer_band_center_translation_fixed_alpha_review_v1'
             and plan.get('alpha_policy') == 'immutable_source_alpha', 'Unknown longitudinal mapping policy')
    rows = preparation['center_before_rows'] if version == V2 else preparation['source_central_rows']
    bands, checks = plan['bands'], result['checks']['final_central_measurements']
    _require(isinstance(bands, list) and bands and len(rows) == 2 * len(bands)
             and len(checks) == len(bands), 'Missing central measurement coverage')
    if version == V2:
        _require(preparation['measurement_tolerance'] == TOLERANCE, 'Changed structure measurement tolerance')
    for i, (band, report) in enumerate(zip(bands, checks)):
        measured = report['measured_rows']
        _require(report['band'] == i and len(measured) == 2 and len(band['target_pair']) == 2,
                 'Central measurements are misassociated')
        lo, hi = (_number(x) for x in measured)
        source_width = _number(rows[2*i+1]) - _number(rows[2*i])
        error = (lo + hi - sum(_number(x) for x in band['target_pair'])) / 2
        _require(hi > lo and source_width > 0 and abs(error) <= TOLERANCE
                 and abs(hi - lo - source_width) <= TOLERANCE
                 and math.isclose(_number(report['center_error_px']), error, abs_tol=1e-9)
                 and math.isclose(_number(report['final_width']), hi-lo, abs_tol=1e-9)
                 and math.isclose(_number(report['source_width']), source_width, abs_tol=1e-9),
                 'Recorded center/width checks do not match their source and targets')
        for j, value in enumerate((lo, hi)):
            _require(abs(value - (_number(rows[2*i+j]) + _number(band['delta']))) <= TOLERANCE,
                     'Measured central edge differs from declared longitudinal translation')


def validate_side(key, face, entry, baseline, audit):
    """Validate pinned saved records; retain review flags and original warnings.

    Explicit lighting-only strategy: structure_policy='lighting_only' with no
    structure_result. Failed/unknown structure records never trigger fallback.
    """
    _require(face in ('left', 'right') and isinstance(key, str) and len(key.split('/')) == 2,
             'A model/SKU key and left/right face are required')
    _require(baseline['sku'] == key.split('/')[1]
             and Path(baseline['source']).parts[-2:] == tuple(key.split('/')),
             'R6 baseline belongs to another key')
    lp, lsha, light, lighting, lighting_sha, light_inputs, assumption = _verify_lighting(key, face, entry, baseline, audit)
    evidence = {'lighting_result': {'path': lp, 'sha256': lsha},
                'lighting_state': light['state'], 'selected_for_review_export': True,
                'quality_checks_complete': False, 'visual_approved': False,
                'selected_for_delivery': False, 'model_uv_verified': False}
    warnings = [{'kind': 'illumination_assumption_review', 'face': face,
                 'original_assumption': copy.deepcopy(assumption),
                 'check_coverage': copy.deepcopy(light.get('check_coverage'))}]
    history = {'baseline_per_face_stage': copy.deepcopy(baseline['per_face_stages'][face]),
               'baseline_blocked_checks': copy.deepcopy(baseline.get('blocked_checks', []))}
    base_result = dict(selected_path=lighting, selected_sha256=lighting_sha,
        lighting_path=lighting, lighting_sha256=lighting_sha, structure_applied=False,
        review_evidence=evidence, warnings=warnings, source_reference_status=None,
        reference_review_required=True, lighting_checks=copy.deepcopy(light['checks']), historical_failures=history)
    if entry.get('structure_result') is None:
        _require(entry.get('structure_policy') == 'lighting_only',
                 'Missing structure requires explicit lighting_only policy')
        evidence['structure_state'] = 'not_selected_explicit_lighting_only'
        base_result['source_reference_status'] = copy.deepcopy(baseline.get('source_body_reference'))
        warnings.append({'kind': 'structure_not_applied', 'face': face})
        return base_result
    _require(entry.get('structure_policy', 'validated_candidate') == 'validated_candidate',
             'Unsupported or conflicting structure selection policy')
    sp, ssha, result = _entry_record(entry, 'structure_result', audit)
    pp, psha, prep = _entry_record(entry, 'preparation', audit)
    _key(result, key, face); _key(prep, key, face)
    _flags(result, FALSE_FLAGS + STRUCTURE_FALSE,
           ('remap_executed', 'candidate_generated', 'eligible_for_visual_review'))
    _require(result.get('hard_failures') == [] and result['final_integrity'].get('unchanged') is True,
             'Structure has hard failures or incomplete final integrity')
    version = result.get('review_policy')
    _require(version in (V2, V3), 'Only saved successful v2/v3 structure schemas are supported')
    prep_inputs = _bindings(prep['input_sha256'], audit)
    _require(result['final_integrity'].get('files') ==
             len(result['input_sha256'] if version == V3 else prep['input_sha256']),
             'Structure final integrity does not cover its recorded input bindings')
    _bound(prep_inputs, lp, lsha)
    _bound(prep_inputs, lighting, lighting_sha)
    _require(_path(prep['source']) == lighting and prep['source_sha256'] == lighting_sha,
             'Structure source is not the selected illumination candidate')
    _bound(prep_inputs, prep['geometry_source'], _geometry(baseline, face))
    audit.file(_path(prep['geometry_source']), _geometry(baseline, face))
    prior_checks = prep['prior_effect_checks'] if version == V2 else prep['prior_lighting_checks']
    _require(prior_checks == light['checks'], 'Structure lost or changed prior illumination checks')
    back = prep['same_device_back'] if version == V2 else prep['own_back']
    _bound(prep_inputs, back, _geometry(baseline, 'back'))
    audit.file(_path(back), _geometry(baseline, 'back'))
    if version == V2:
        _require(result['execution_state'] == 'candidate_generated_with_explicit_alpha_review'
                 and result['state'] == 'band_center_boundary_alpha_review'
                 and result['structure_operation'] == 'band_center_translation_review', 'Wrong v2 execution state')
        _require(Path(pp).parent == Path(sp).parent, 'V2 preparation and result must share their saved run')
        attempt_path = str(Path(pp).parent / 'remap-attempt.json')
        attempt = audit.json(attempt_path)
        _require(attempt['preparation_sha256'] == psha, 'V2 remap attempt does not bind preparation')
        _flags(result, true=('source_and_mapping_identical_to_v1',))
        _flags(prep, true=('source_and_mapping_identical_to_v1', 'opacity_thresholds_unchanged',
                           'unknown_endpoints_still_unknown', 'illumination_assumption_review_retained'))
        proof = result['checks']['mapping_and_alpha_review']
        _flags(proof, true=('all_nonzero_weight_donors_checked', 'source_rgb_rigid_inside_integer_core', 'single_resampling'))
        _zeros(proof, ('alpha_zero_donors_used', 'internal_conflicts_permitted'))
        reference = copy.deepcopy(prep.get('reference_review', {}))
        _require(isinstance(reference, dict), 'Malformed optional complete reference review')
        reference.update({name: copy.deepcopy(prep[name]) for name in
                          ('back_reference_status', 'reference_certified', 'reference_review_required')})
        failures = [path for path, digest in prep_inputs.items() if digest == prep['v1_failure_sha256']]
        _require(len(failures) == 1, 'V2 historical rejection is not uniquely bound')
        prior = audit.json(failures[0], prep['v1_failure_sha256'])
        _key(prior, key, face)
        _require(prior.get('candidate_generated') is False and prior.get('hard_failures'),
                 'V2 historical rejection evidence is incomplete')
        history['historical_v1_rejection'] = copy.deepcopy(prior['hard_failures'])
        evidence['remap_attempt'] = attempt_path
    else:
        _require(result['execution_state'] in ('candidate_generated_needs_contour_and_visual_review',
                                               'candidate_generated_needs_contour_and_reference_review')
                 and result['state'] == 'contour_relative_sampling_assumption_review'
                 and result['structure_operation'] == 'band_center_contour_relative_sampling_review', 'Wrong v3 execution state')
        inputs = _bindings(result['input_sha256'], audit)
        _bound(inputs, pp, psha); _bound(inputs, lp, lsha); _bound(inputs, lighting, lighting_sha)
        if 'source_preparation' in result:
            source_preparation = result['source_preparation']
            _require(isinstance(source_preparation, dict) and _path(source_preparation['path']) == pp
                     and source_preparation['sha256'] == psha
                     and _path(source_preparation['source']) == lighting
                     and source_preparation['source_sha256'] == lighting_sha,
                     'V3 explicit preparation does not bind the supplied source evidence')
            for name in ('own_back', 'bound_back_rows', 'target_rows',
                         'source_central_rows', 'source_central_columns'):
                _require(source_preparation[name] == prep[name], 'V3 preparation field changed: ' + name)
        _flags(result, ('source_lighting_reprocessed', 'back_reference_recomputed', 'longitudinal_mapping_changed',
                        'whole_core_rgb_rigid', 'legacy_same_x_contract_reclassified_as_passed'), ('horizontal_sampling_changed',))
        _require(result['prepared_longitudinal_mapping'] == prep['mapping'], 'V3 changed its prepared longitudinal plan')
        _flags(prep, true=('source_lighting_assumption_retained',))
        proof = result['checks']['contour_sampling']
        _flags(proof, ('whole_core_rgb_rigid', 'source_rgb_rigid_inside_integer_core',
                       'all_source_pixels_retained_claimed', 'strict_2d_invertibility_claimed',
                       'legacy_same_x_contract_reclassified_as_passed', 'full_map_same_for_all_columns',
                       'area_conservation_claimed', 'actual_xy_donor_original_thresholds_passed'),
               ('all_actual_positive_xy_donors_checked', 'horizontal_map_nondecreasing',
                'same_y_map_for_all_columns', 'horizontal_resampling_applied'))
        _zeros(proof, ('alpha_zero_actual_donors_used', 'internal_opacity_conflicts_permitted'))
        _require(_number(proof['actual_xy_donor_contributions']) > 0, 'No actual xy donor evidence')
        _artifact(result, 'sampling_coordinates', audit)
        _require(result['reference_review'] == prep['reference_review'], 'V3 reference review was reinterpreted')
        reference = copy.deepcopy(result['reference_review'])
        if result['execution_state'] == 'candidate_generated_needs_contour_and_reference_review':
            _require('source_preparation' in result
                     and reference.get('original_status') == 'source_tail_applicability_unknown'
                     and 'original_applicable' in reference
                     and reference.get('original_applicable') is None
                     and reference.get('original_unknown_preserved') is True,
                     'Reference-review state must preserve the explicit unknown reference')
            _flags(result, false=('target_derivation_repeated',))
            _require(result['source_identity_before'] == result['source_identity_after']
                     == light['source_identity_before'], 'V3 raw source identity changed')
        prior_path = str(Path(pp).parent / 'result.json')
        prior = audit.json(prior_path, _bound(inputs, prior_path))
        _key(prior, key, face)
        _require(prior.get('candidate_generated') is False and prior.get('hard_failures')
                 and result.get('historical_same_x_rejection') == prior['hard_failures'],
                 'V3 historical same-x rejection is missing or changed')
        history['historical_same_x_rejection'] = copy.deepcopy(prior['hard_failures'])
    _require(proof['policy'] == version, 'Structure check policy does not match result')
    _zeros(proof, ('alpha_changed', 'fixed_rgba_changed', 'transparent_rgba_changed'))
    _flags(result, false=('original_opacity_contract_passed',), true=('opacity_review_required',))
    _flags(proof, false=('original_opacity_contract_passed', 'strict_structure_verified',
                         'visual_approved', 'selected_for_delivery'), true=('review_required',))
    _require(isinstance(proof.get('opacity_review_events'), list) and proof['opacity_review_events'],
             'Explicit opacity review events missing')
    _centers(result, prep, version)
    _flags(result['checks']['shoulder_diagnostics'], ('acceptance_threshold_applied', 'automatic_effect_pass', 'visual_approved'),
           ('both_axes_measured',))
    _require(reference.get('reference_certified') is False and reference.get('reference_review_required') is True,
             'Uncertified reference review must remain explicit')
    selected, selected_sha = _artifact(result, 'candidate', audit)
    _require(Path(selected).parent == Path(sp).parent, 'Structure candidate is outside its saved run')
    evidence.update(structure_result={'path': sp, 'sha256': ssha}, preparation={'path': pp, 'sha256': psha},
                    structure_policy=version, structure_checks=copy.deepcopy(result['checks']),
                    remaining_checks=copy.deepcopy(result['remaining_checks']),
                    reference_review=copy.deepcopy(reference))
    warnings.append({'kind': version, 'face': face, 'original_opacity_contract_passed': False,
                     'opacity_review_required': True, 'remaining_checks': copy.deepcopy(result['remaining_checks'])})
    base_result.update(selected_path=selected, selected_sha256=selected_sha, structure_applied=True,
                       source_reference_status=reference, reference_review_required=True)
    return base_result
