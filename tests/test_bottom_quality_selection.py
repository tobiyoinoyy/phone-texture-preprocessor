"""R7 routing controls, prepared statically; root alone may run them.

Uses the existing tiny stage fixtures. No real port fit, lighting algorithm,
model inference or camera sample is exercised by these controls.
"""
import copy
from collections import Counter
from pathlib import Path

import pytest
from phone_texture import bottom_alignment as bottom
from phone_texture import bottom_quality_selection as quality
from phone_texture import closed_paper_lighting as recipe
from phone_texture import resolved_bottom_material
from phone_texture import stage_processing as stages
from test_bottom_downstream_fallback import setup_attempt, assert_once_except_bottom, reject_attempts


def setup_quality(tmp_path, monkeypatch, *, original='pass', port_warning=True):
    c = setup_attempt(tmp_path, monkeypatch)
    c['quality_calls'] = []

    def source():
        c['quality_calls'].append('source')
        before = (copy.deepcopy(c['geometry']), dict(c['files']), copy.deepcopy(c['report']))
        path = bottom.original_source_for_quality_comparison(c['files'], c['geometry'],
            c['original_files'], c['original_geometry'], c['report'])
        assert before == (c['geometry'], c['files'], c['report'])
        return path

    def commit(path):
        c['quality_calls'].append('commit')
        return bottom.retain_source_after_quality_comparison(c['files'], c['geometry'],
            c['original_files'], c['original_geometry'], c['report'], c['process'], path)

    c['quality'] = dict(source=source, commit=commit)
    monkeypatch.setattr(stages.auto_material, 'lighting_scope',
        lambda source, face, *args: dict(local_checks={'actual_check': {}}) if face == 'bottom' else {})

    def effect(function, scope, source, candidate, *args):
        if 'actual_check' not in scope.get('local_checks', {}):
            return dict(passed=True)
        is_port = stages.w.sha(source) == c['candidate_sha']
        if not is_port and original == 'error':
            raise ValueError('Lighting correction discontinuity at bottom_paper_metal_horizontal: 1.60 L')
        if (is_port and port_warning) or (not is_port and original == 'warning'):
            return dict(original_effect_check_passed=False, passed=False,
                        status='effect_review_required', error='actual original effect warning')
        return dict(actual_check=dict(effect_check_passed=True))

    monkeypatch.setattr(stages, 'measure_effect', effect)
    return c


def run(c, *, enabled=True):
    return stages.process_stages(c['case'], 'sku', c['geometry'], c['files'], c['manifest'],
        'iphone17pro', 'none', 'accepted-shadow-lift-v2', lambda *a, **k: None, lambda *a: None,
        bottom_fallback=c['fallback'], bottom_quality=c['quality'] if enabled else None)


def audit(c):
    return stages.w.read(c['case']/'work/bottom-quality-selection/report.json')


def test_warning_compares_only_bottom_and_commits_actual_passing_original(tmp_path, monkeypatch):
    c = setup_quality(tmp_path, monkeypatch)
    result = run(c)
    assert_once_except_bottom(c, 2)
    assert c['quality_calls'] == ['source', 'commit'] and c['callbacks'] == []
    assert c['seam_calls'] == ['bottom-quality-original']
    assert result['per_face_stages']['bottom']['lighting'] == 'complete'
    assert result['per_face_stages']['bottom']['source_sha256'] == c['original_sha']['bottom']
    assert not result['lighting_effect_warnings'] and not result['blocked_checks']
    assert all(v == 'complete' for v in result['stage_status'].values())
    report = audit(c)
    assert report['selected'] == 'original'
    assert report['original']['original_effect_check_passed'] is True
    assert report['port']['original_effect_check_passed'] is False
    assert report['warning_reclassified_as_error'] is False
    assert stages.w.read(report['port']['checks'])['source_relative_effect_checks']['original_effect_check_passed'] is False
    quality.verify_attempt(report['port']); quality.verify_attempt(report['original'])
    assert c['report']['applied'] is False and c['report']['downstream_quality_selection']
    assert not c['report'].get('downstream_source_retry')
    assert c['report']['attempt_selected_sha256'] == c['candidate_sha']
    assert c['report']['selected_sha256'] == c['original_sha']['bottom']
    assert c['original_geometry'] == c['original_geometry_snapshot']
    for face in stages.pipeline.FACES:
        assert stages.w.sha(c['original_files'][face]) == c['original_sha'][face]
        assert stages.w.sha(c['files'][face]) == c['original_sha'][face]
    assert not (c['process']/'attempt-report.json').exists()  # No fake hard failure.


@pytest.mark.parametrize('original', ['warning', 'error'])
def test_original_nonpass_keeps_first_port_candidate_and_its_warning(tmp_path, monkeypatch, original):
    c = setup_quality(tmp_path, monkeypatch, original=original)
    before = copy.deepcopy(c['report'])
    result = run(c)
    assert_once_except_bottom(c, 2)
    assert c['quality_calls'] == ['source'] and c['callbacks'] == []
    assert c['seam_calls'] == ['bottom-port-candidate']
    assert c['report'] == before and c['report']['applied'] is True
    assert result['per_face_stages']['bottom']['source_sha256'] == c['candidate_sha']
    assert result['per_face_stages']['bottom']['lighting'] == 'effect_warning'
    assert result['status'] == 'full_stages_effect_warning_needs_review'
    assert len(result['lighting_effect_warnings']) == 1 and not result['blocked_checks']
    report = audit(c)
    assert report['selected'] == 'port' and report['original']['original_effect_check_passed'] is False
    quality.verify_attempt(report['port'])
    if original == 'error':
        assert report['original']['error_type'] == 'ValueError'
        assert report['original']['local_rejection_code'] == 'known_measured_effect_discontinuity'
        assert report['original']['diagnostic_sha256']
    assert not (c['process']/'source-quality-selected').exists()


@pytest.mark.parametrize('enabled,warning', [(True, False), (False, True)])
def test_no_comparison_without_warning_or_without_callback(tmp_path, monkeypatch, enabled, warning):
    c = setup_quality(tmp_path, monkeypatch, port_warning=warning)
    run(c, enabled=enabled)
    assert_once_except_bottom(c, 1)
    assert c['quality_calls'] == [] and c['callbacks'] == []
    assert not (c['case']/'work/bottom-quality-selection').exists()


def test_existing_hard_rejection_keeps_existing_source_retry(tmp_path, monkeypatch):
    c = setup_quality(tmp_path, monkeypatch)
    reject_attempts(monkeypatch, ['bottom-port-candidate'])
    result = run(c)
    assert_once_except_bottom(c, 2)
    assert len(c['callbacks']) == 1 and c['quality_calls'] == []
    assert c['report']['downstream_source_retry'] and c['seam_calls'] == ['bottom-source-fallback']
    assert result['per_face_stages']['bottom']['optional_bottom_candidate_rejection']['kind'] == 'material_classification'


@pytest.mark.parametrize('exception', [OSError, RuntimeError])
def test_trial_io_or_runtime_failure_is_not_hidden(tmp_path, monkeypatch, exception):
    c = setup_quality(tmp_path, monkeypatch)
    reject_attempts(monkeypatch, ['bottom-quality-original'], exception=exception)
    with pytest.raises(exception):
        run(c)
    assert c['quality_calls'] == ['source'] and c['report']['applied']


def test_trial_pixel_integrity_value_error_is_not_hidden(tmp_path, monkeypatch):
    c = setup_quality(tmp_path, monkeypatch)
    from phone_texture import automatic
    original = automatic.measure_masks

    def masks(source, candidate, *args):
        if candidate.parent.name == 'bottom-quality-original':
            raise ValueError('Output alpha changed')
        return original(source, candidate, *args)

    monkeypatch.setattr(automatic, 'measure_masks', masks)
    with pytest.raises(ValueError, match='alpha'):
        run(c)
    assert c['quality_calls'] == ['source'] and c['report']['applied']


def test_changed_trial_source_fails_closed_before_commit(tmp_path, monkeypatch):
    c = setup_quality(tmp_path, monkeypatch)
    original = stages.lighting.run

    def corrupt(engine, path):
        original(engine, path)
        if path.parent.name == 'bottom-quality-original':
            c['original_files']['bottom'].write_bytes(b'changed original source')
            raise ValueError('Trial locally failed after source mutation')

    monkeypatch.setattr(stages.lighting, 'run', corrupt)
    with pytest.raises(RuntimeError, match='comparison source changed'):
        run(c)
    assert c['quality_calls'] == ['source'] and c['report']['applied']


def test_changed_trial_input_asset_is_not_hidden_as_local_failure(tmp_path, monkeypatch):
    c = setup_quality(tmp_path, monkeypatch)
    original = stages.lighting.run

    def corrupt(engine, path):
        original(engine, path)
        if path.parent.name == 'bottom-quality-original':
            # job.json itself is part of the frozen input set.
            path.write_text('{}')
            raise ValueError('Local failure after changing the trial job')

    monkeypatch.setattr(stages.lighting, 'run', corrupt)
    with pytest.raises(RuntimeError, match='input assets changed'):
        run(c)
    assert c['quality_calls'] == ['source'] and c['report']['applied']


@pytest.mark.parametrize('phase,message', [
    ('material', 'Material landmarks belong to another source'),
    ('material', 'Bottom paper material boundary unresolved: Source contour material source identity differs'),
    ('scope', 'Lighting scope source mismatch'),
    ('scope', 'Invalid source-normal peer witness'),
    ('render', 'Material domain mask dimensions differ from source'),
    ('render', 'Unknown illumination field'),
    ('effect', 'Local output changed geometry'),
    ('effect', 'Lighting scope source mismatch'),
    ('effect', 'Unrecognized future ValueError'),
])
def test_unrecognized_or_integrity_value_errors_propagate_in_all_trial_phases(tmp_path, monkeypatch, phase, message):
    c = setup_quality(tmp_path, monkeypatch)
    if phase == 'material':
        old = stages.auto_material.propose_job
        def operation(source, face, folder, **kwargs):
            value = old(source, face, folder, **kwargs)
            if folder.name == 'bottom-quality-original': raise ValueError(message)
            return value
        monkeypatch.setattr(stages.auto_material, 'propose_job', operation)
    elif phase == 'scope':
        old = stages.auto_material.lighting_scope
        def operation(source, face, job, folder):
            if folder.parent.name == 'bottom-quality-original': raise ValueError(message)
            return old(source, face, job, folder)
        monkeypatch.setattr(stages.auto_material, 'lighting_scope', operation)
    elif phase == 'render':
        old = stages.lighting.run
        def operation(engine, path):
            value = old(engine, path)
            if path.parent.name == 'bottom-quality-original': raise ValueError(message)
            return value
        monkeypatch.setattr(stages.lighting, 'run', operation)
    else:
        old = stages.measure_effect
        def operation(function, scope, source, candidate, *args):
            if candidate.parent.name == 'bottom-quality-original': raise ValueError(message)
            return old(function, scope, source, candidate, *args)
        monkeypatch.setattr(stages, 'measure_effect', operation)
    with pytest.raises(ValueError, match=message):
        run(c)
    assert c['quality_calls'] == ['source'] and c['report']['applied']
    assert c['callbacks'] == []
    assert not (c['process']/'source-quality-selected').exists()


def test_known_material_support_rejection_keeps_port_without_faking_hard_error(tmp_path, monkeypatch):
    c = setup_quality(tmp_path, monkeypatch)
    old = stages.auto_material.propose_job
    def material(source, face, folder, **kwargs):
        if folder.name == 'bottom-quality-original':
            raise ValueError('Bottom paper material boundary unresolved: No supported source paper termination to apply')
        return old(source, face, folder, **kwargs)
    monkeypatch.setattr(stages.auto_material, 'propose_job', material)
    result = run(c)
    assert c['quality_calls'] == ['source'] and c['callbacks'] == []
    assert result['per_face_stages']['bottom']['lighting'] == 'effect_warning'
    assert audit(c)['original']['local_rejection_code'] == 'known_source_material_support_rejection'
    assert (c['case']/'work/material-inputs/bottom-quality-original/quality-trial-failure.json').is_file()


def test_scope_success_certificate_requires_all_normal_evidence():
    scope = dict(local_checks={'normal': dict(normal_peer_mask='bound-mask', max_normal_peer_step_L=1.25)})
    good = dict(normal=dict(effect_check_passed=True, normal_check_passed=True,
        normal_coverage=dict(complete=True, original_threshold_passed=True,
            normal_pairs=20, qualified_target_pairs=20, limit_L=1.25, max_observed_residual_L75=1.0)))
    assert quality.effect_fully_passed(scope, good) is True
    for mutate in ('missing_zone', 'incomplete', 'threshold_false', 'normal_false', 'wrapper_warning'):
        value = copy.deepcopy(good)
        if mutate == 'missing_zone': value.clear()
        elif mutate == 'incomplete': value['normal']['normal_coverage']['complete'] = False
        elif mutate == 'threshold_false': value['normal']['normal_coverage']['original_threshold_passed'] = False
        elif mutate == 'normal_false': value['normal']['normal_check_passed'] = False
        else: value['original_effect_check_passed'] = False
        assert quality.effect_fully_passed(scope, value) is False
    assert quality.effect_fully_passed({}, dict(passed=True)) is False


def test_support_error_classifier_accepts_only_exact_known_engine_message():
    message = ('Unsupported editable illumination: 64 pixels; bounds [4, 171, 113, 190]. '
               'Split the component and fit its own reviewed source samples; '
               'nearest-neighbor or median hole filling is not permitted.')
    assert quality.local_rejection_code(ValueError(message), 'render') == 'known_render_field_support_rejection'
    assert quality.local_rejection_code(ValueError(message), 'pixel_integrity') is None
    assert quality.local_rejection_code(ValueError('Unsupported editable illumination: source identity mismatch'), 'render') is None
    assert quality.local_rejection_code(ValueError(message+' source changed'), 'render') is None


def record_fixture(tmp_path):
    assets = tmp_path/'record-assets'
    assets.mkdir()
    source, candidate = tmp_path/'source.bin', tmp_path/'candidate.bin'
    source.write_bytes(b'checked source')
    candidate.write_bytes(b'checked output')
    scope = dict(local_checks={'actual_check': {}})
    stages.write(assets/'job.json', dict(source=str(source)))
    stages.write(assets/'lighting-scope.json', scope)
    checks = dict(source_sha256=stages.w.sha(source), output_sha256=stages.w.sha(candidate),
        source_relative_effect_checks=dict(actual_check=dict(effect_check_passed=True)))
    return source, candidate, assets, checks, scope


def test_attempt_records_and_verifies_pixel_checks_bound_to_actual_files(tmp_path):
    args = record_fixture(tmp_path)
    record = quality.record_attempt(*args)
    quality.verify_attempt(record)
    assert record['source_sha256'] == args[3]['source_sha256']
    assert record['candidate_sha256'] == args[3]['output_sha256']


@pytest.mark.parametrize('index', [0, 1])
def test_record_rejects_source_or_candidate_changed_since_pixel_checks(tmp_path, index):
    args = record_fixture(tmp_path)
    args[index].write_bytes(b'changed after measure_masks')
    with pytest.raises(RuntimeError, match='pixel checks do not bind'):
        quality.record_attempt(*args)
    assert not (args[2]/'selection-checks.json').exists()


@pytest.mark.parametrize('field', ['source_sha256', 'output_sha256'])
def test_verify_rejects_cross_binding_even_with_matching_checks_file_hash(tmp_path, field):
    args = record_fixture(tmp_path)
    record = quality.record_attempt(*args)
    checks = stages.w.read(record['checks'])
    checks[field] = '0'*64
    stages.write(Path(record['checks']), checks)
    # A matching checksum of the checks file is insufficient by itself.
    record['checks_sha256'] = stages.w.sha(record['checks'])
    with pytest.raises(RuntimeError, match='pixel checks differ'):
        quality.verify_attempt(record)


@pytest.mark.parametrize('kind', ['line', 'pixel', 'boundary'])
def test_certificate_rechecks_existing_numeric_limits_and_sampling(kind):
    zone = ({'max_added_step_L': 1.25} if kind == 'line' else
            {'max_pixel_added_step_L': 1.25} if kind == 'pixel' else
            {'boundary_peer_mask': 'bound', 'max_boundary_step_L': 2.0})
    values = ({'max_added_step_L': 1.0} if kind == 'line' else
              {'max_pixel_added_step_L': 1.0} if kind == 'pixel' else
              {'boundary_step_L99': 1.5, 'boundary_peak_L': 2.8})
    scope = {'local_checks': {'x': zone}}
    effect = {'x': dict(effect_check_passed=True, pair_count=20, **values)}
    assert quality.effect_fully_passed(scope, effect)
    for failure in ('too_few', 'over_limit', 'nonfinite', 'quantization'):
        changed = copy.deepcopy(effect)
        if failure == 'too_few': changed['x']['pair_count'] = 19
        elif failure == 'quantization': changed['x']['quantization_review'] = {'review_required': True}
        else: changed['x'][next(iter(values))] = 4.0 if failure == 'over_limit' else float('nan')
        assert not quality.effect_fully_passed(scope, changed)


@pytest.mark.parametrize('budget', [6, 17])
def test_original_closure_keeps_existing_recipe_not_a_forced_six_limit(monkeypatch, budget):
    view = dict(protection_policy='none', source_label_material_evidence=dict(
        material_boundary_resolution='original_source_closure'), domains=[dict(name='printed_label',
        strength=[.8, 0, 0], max_lab_shift=[budget, 0, 0], reference_boxes=[[1, 2, 3, 4]])])
    before = copy.deepcopy(view)
    monkeypatch.setattr(resolved_bottom_material, 'validate', lambda *a: pytest.fail('recipe route must not run'))
    assert recipe.apply(view, 'bottom') is None
    assert view == before


@pytest.mark.parametrize('resolution', ['source_contour_frame_domain', 'source_aligned_metal_window'])
def test_other_verified_resolution_enhancements_remain(monkeypatch, resolution):
    view = dict(protection_policy='none', source_label_material_evidence=dict(material_boundary_resolution=resolution),
        domains=[dict(name='printed_label', strength=[.8, 0, 0], max_lab_shift=[6, 0, 0],
                      reference_boxes=[[1, 2, 3, 4]]), dict(name='main_surface')])
    monkeypatch.setattr(resolved_bottom_material, 'validate', lambda *a: dict(resolution=resolution))
    proof = recipe.apply(view, 'bottom')
    assert proof is not None and view['domains'][0]['max_lab_shift'] == [24, 0, 0]
    assert view['correction_blends'][0]['sigma'] == 1.25
