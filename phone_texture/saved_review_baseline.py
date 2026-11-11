"""Bind a supported R6 partial baseline to its original six sources and outputs.

Only hashes and saved JSON are checked here. No image processing or new quality
approval is performed. The caller's Audit verifies every referenced file byte.
"""
from pathlib import Path

FACES = ('front', 'back', 'left', 'right', 'top', 'bottom')
LIT = ('front', 'back', 'top', 'bottom')
MODELS = {'iphone17pro': 'ip17pro', 'iphone17promax': 'ip17promax'}


def require(value, message):
    if not value:
        raise ValueError('Saved baseline: ' + message)


def digest(value):
    require(isinstance(value, str) and len(value) == 64
            and all(c in '0123456789abcdef' for c in value), 'invalid SHA-256')
    return value


def absolute(value):
    require(isinstance(value, str) and Path(value).is_absolute(), 'absolute evidence path required')
    return Path(value).resolve()


def document(audit, entry, field, required_path=None):
    value = entry.get(field)
    require(isinstance(value, dict) and set(value) == {'path', 'sha256'}, field + ' path/SHA missing')
    path = absolute(value['path'])
    if required_path is not None:
        require(path == required_path.resolve(), field + ' is outside the baseline case')
    result = audit.json(path, digest(value['sha256']))
    require(isinstance(result, dict), field + ' must be an object')
    return result


def zeros(record, fields, label):
    require(isinstance(record, dict), label + ' missing')
    for field in fields:
        require(type(record.get(field)) is int and record[field] == 0,
                label + ' nonzero/missing ' + field)


def validate_baseline(audit, entry, baseline, geometry_report):
    """Return original geometry SHA map; preserve all warning/quality states."""
    require(isinstance(entry, dict) and isinstance(baseline, dict), 'entry/baseline missing')
    sku, model = entry.get('sku'), entry.get('model')
    require(isinstance(sku, str) and sku not in ('', '.', '..') and Path(sku).name == sku
            and model in MODELS and entry.get('key') == MODELS[model] + '/' + sku,
            'invalid model/SKU key')
    require(document(audit, entry, 'baseline_result') == baseline, 'baseline differs from bound report')
    require(baseline.get('sku') == sku, 'baseline SKU mismatch')
    case = absolute(baseline.get('process')); source = absolute(baseline.get('source'))
    require(case.name == sku and tuple(source.parts[-2:]) == (MODELS[model], sku),
            'baseline source/case identity mismatch')
    options = entry.get('options')
    require(isinstance(options, dict) and options.get('model') == model
            and options.get('protection_policy') == 'none'
            and options.get('review_fallback') == 'preserve-source'
            and options.get('lighting_profile') == 'accepted-shadow-lift-v2'
            and options.get('bottom_alignment') == 'appearance'
            and options.get('source_body_reference_review') is True,
            'unsupported entry processing options')
    require(baseline.get('protection_policy') == options['protection_policy']
            and baseline.get('lighting_profile') == options['lighting_profile']
            and baseline.get('source_body_reference_review_enabled') is True,
            'baseline processing configuration differs from entry')
    require(baseline.get('status') == 'source_preserved_needs_review'
            and baseline.get('six_faces_generated') is True
            and baseline.get('source_hashes_unchanged') is True,
            'unsupported baseline generation state')
    for key in ('six_faces_processed', 'mandatory_program_checks_complete', 'quality_checks_complete',
                'visual_review_passed', 'model_uv_verified', 'structural_alignment_applied', 'seam_accepted'):
        require(baseline.get(key) is False, 'unsupported baseline flag ' + key)
    require(baseline.get('stage_status') == dict(geometry='complete', lighting='partial',
            structure='blocked', seam='blocked'), 'unsupported baseline stage schema')

    intake = document(audit, entry, 'baseline_intake', case/'work/intake.json')
    bound_geometry = document(audit, entry, 'geometry_report', case/'work/geometry-records/report.json')
    require(bound_geometry == geometry_report, 'geometry argument differs from bound report')
    require(absolute(intake.get('source_root')) == source, 'intake source differs from baseline')
    require(isinstance(intake.get('skus'), dict) and set(intake['skus']) == {sku}, 'intake SKU set differs')
    intake_faces = intake['skus'][sku]
    sources = entry.get('source_sha256')
    require(isinstance(sources, dict) and set(sources) == set(FACES)
            and isinstance(intake_faces, dict) and set(intake_faces) == set(FACES), 'six sources required')
    rows = geometry_report.get('images')
    require(geometry_report.get('candidates_complete') is True and isinstance(rows, list)
            and len(rows) == 6 and all(isinstance(item, dict) for item in rows), 'geometry rows incomplete')
    by_face = {item.get('face'): item for item in rows}
    require(set(by_face) == set(FACES), 'geometry faces duplicate or missing')
    staged = case/'work/geometry-records/staged'/sku
    geometry = {}
    for face in FACES:
        expected = digest(sources[face]); raw = intake_faces[face]; item = by_face[face]
        require(isinstance(raw, dict) and raw.get('sha256') == expected
                and item.get('sku') == sku and item.get('source_sha256') == expected,
                'raw/intake/geometry source mismatch: ' + face)
        raw_path, snapshot = absolute(raw.get('source')), absolute(raw.get('snapshot'))
        require(raw_path == source/(face+'.jpg')
                and snapshot == case/'work/input'/sku/(face+'.jpg')
                and absolute(item.get('source')) == snapshot,
                'raw or snapshot path belongs to another face/case: ' + face)
        audit.file(raw_path, expected); audit.file(snapshot, expected)
        geometry[face] = digest(item.get('output_sha256'))
        audit.file(staged/(face+'.png'), geometry[face])

    bottom = baseline.get('bottom_alignment')
    require(isinstance(bottom, dict) and bottom.get('model') == model
            and bottom.get('status') in ('applied_with_notice', 'kept_source_with_notice'),
            'unsupported bottom alignment state')
    require(bottom.get('applied') is (bottom['status'] == 'applied_with_notice')
            and bottom.get('source_geometry_unmodified') is True
            and bottom.get('source_hashes_unchanged') is True
            and bottom.get('companion_faces_byte_identical') is True,
            'bottom source/companion identity not verified')
    require(absolute(bottom.get('source')) == staged/'bottom.png'
            and bottom.get('source_sha256') == geometry['bottom'], 'bottom original source mismatch')
    selected_dir = absolute(bottom.get('selected_dir'))
    require(selected_dir.is_relative_to(case/'work/bottom-alignment'), 'bottom selected directory outside baseline')
    bottom_sha = digest(bottom.get('selected_sha256'))
    for face in FACES:
        audit.file(selected_dir/(face+'.png'), bottom_sha if face == 'bottom' else geometry[face])
    if bottom['status'] == 'kept_source_with_notice':
        require(bottom.get('fallback_byte_identical') is True and bottom_sha == geometry['bottom'],
                'kept bottom differs from original geometry')
    else:
        integrity = bottom.get('integrity', {})
        require(all(integrity.get(key) is True for key in ('canvas_unchanged', 'endcaps_exact',
                'source_sampling_in_bounds', 'finite_monotone_map')), 'applied bottom integrity incomplete')
        mapping = bottom.get('source_x_mapping')
        require(isinstance(mapping, dict) and set(mapping) == {'path', 'sha256'}, 'bottom mapping evidence missing')
        audit.file(absolute(mapping['path']), digest(mapping['sha256']))

    final = baseline.get('final_sha256'); per_face = baseline.get('per_face_stages')
    checks = baseline.get('lighting_checks'); canvas = baseline.get('final_canvas_checks')
    require(isinstance(final, dict) and set(final) == set(FACES)
            and isinstance(per_face, dict) and set(per_face) == set(FACES)
            and isinstance(checks, dict) and set(checks) == set(LIT)
            and isinstance(canvas, dict) and set(canvas) == set(FACES), 'baseline per-face records incomplete')
    for face in FACES:
        expected = digest(final[face]); info = per_face[face]
        input_sha = bottom_sha if face == 'bottom' else geometry[face]
        require(info.get('source_sha256') == input_sha, 'baseline geometry-to-stage input mismatch: ' + face)
        require(info.get('selected_sha256') == expected, 'selected/final mismatch: ' + face)
        selected_path = absolute(info.get('selected_stage_file'))
        require(selected_path == case/'work/lighting'/sku/(face+'.png'), 'selected stage belongs to another case: ' + face)
        audit.file(selected_path, expected)
        require(baseline.get('face_processing', {}).get(face) == info.get('selected_stage'),
                'face processing stage mismatch: ' + face)
        frame = canvas[face]
        zeros(frame, ('body_pixels_lost', 'paper_pixels_lost', 'removed_max_alpha'), 'final canvas ' + face)
        require(frame.get('retained_rgba_unchanged') is True and frame.get('resampled') is False
                and frame.get('source_size') == frame.get('output_size')
                and frame.get('removed_margins_ltrb') == [0, 0, 0, 0], 'unsupported final canvas transform: ' + face)
        if face in ('left', 'right'):
            require(info.get('selected_stage') == 'geometry' and info.get('lighting') == 'failed'
                    and expected == geometry[face], 'unsupported historical side stage: ' + face)
            continue
        require(info.get('selected_stage') == 'lighting'
                and info.get('lighting') in (('complete', 'effect_warning') if face == 'back' else ('complete',)),
                'unsupported historical lighting stage: ' + face)
        require(info.get('lighting_sha256') == expected, 'lighting/selected hash mismatch: ' + face)
        lighting_path = absolute(info.get('lighting_file'))
        require(lighting_path.is_relative_to(case/'work/lighting-attempts'), 'lighting artifact outside baseline')
        audit.file(lighting_path, expected)
        check = checks[face]
        require(check.get('source_sha256') == input_sha and check.get('output_sha256') == expected,
                'lighting source/output does not bind selected face: ' + face)
        zeros(check, ('alpha_changed', 'paper_changed', 'outside_material_changed'), 'lighting ' + face)
        no_zone = check.get('actual_no_zone_effects')
        zeros(no_zone, ('protected_zone_pixels', 'alpha_pixels_changed'), 'no-zone ' + face)
        require(no_zone.get('protection_policy') == 'none', 'lighting protection configuration differs: ' + face)
        if info.get('lighting') == 'effect_warning':
            require(any(item.get('face') == face and item.get('original_effect_check_passed') is False
                        for item in baseline.get('lighting_effect_warnings', [])), 'effect warning evidence missing')
    return geometry
