"""Aesthetic warnings never skip mandatory source or continuity checks."""
import copy

import numpy as np
import pytest
from PIL import Image

from phone_texture import lighting_review as review
from phone_texture import shadow_lift_profile as profile


PROFILES = ('accepted-shadow-lift-v1', 'accepted-shadow-lift-v2')


def fixture(tmp_path):
    shape = (32, 40)
    rgba = np.full((*shape, 4), 100, np.uint8)
    rgba[..., 3] = 255
    source, output = tmp_path/'source.png', tmp_path/'output.png'
    Image.fromarray(rgba).save(source)
    Image.fromarray(rgba).save(output)
    allowed = np.ones(shape, bool)
    protected = np.zeros(shape, bool)

    def mask(name, values):
        path = tmp_path/(name+'.png')
        Image.fromarray(values.astype('uint8')*255).save(path)
        return str(path)

    full = mask('full', allowed)
    spec = dict(source_sha256=review.digest(source), action='aluminum',
        mode='regional', reason='Synthetic source-relative brightness goal',
        protection_policy='none', local_checks={
            'rear_shadow_top': dict(mask=full, min_L=80., max_L=100.,
                                    max_added_step_L=1.25, step_axis=0)})
    return spec, source, output, allowed, protected, rgba, mask


def measured(function, data, name='accepted-shadow-lift-v2'):
    spec, source, output, allowed, protected = data[:5]
    return profile.measure_effect(function, spec, source, output, allowed, protected, name)


def perimeter(spec, allowed, mask):
    yy, xx = np.indices(allowed.shape)
    regions = dict(top=(yy<3)&(xx>=15)&(xx<25),
        bottom=(yy>=29)&(xx>=15)&(xx<25),
        left=(xx<3)&(yy>=12)&(yy<20),
        right=(xx>=37)&(yy>=12)&(yy<20),
        top_left=(xx<3)&(yy<3), top_right=(xx>=37)&(yy<3),
        bottom_left=(xx<3)&(yy>=29), bottom_right=(xx>=37)&(yy>=29))
    spec.update(mode='perimeter', reference_basis='Synthetic source material',
        required_metal_mask=mask('required', allowed),
        zones={name:dict(mask=mask(name, area), min_L=80., max_L=100.)
               for name, area in regions.items()})


@pytest.mark.parametrize('name', PROFILES)
@pytest.mark.parametrize('too_bright', (False, True))
def test_real_local_threshold_failure_stays_false_with_full_measurements(tmp_path, name, too_bright):
    data = fixture(tmp_path)
    zone = data[0]['local_checks']['rear_shadow_top']
    if too_bright:
        zone.update(min_L=0., max_L=30.)
    result = measured(review.measure_local, data, name)
    assert result['passed'] is False
    assert result['original_effect_check_passed'] is False
    assert result['status'] == 'effect_warning_not_quality_passed'
    assert result['non_aesthetic_scope_checks_complete'] is True
    failure = result['effect_failures'][0]
    assert failure['check'] == 'rear_shadow_top'
    assert failure['kind'] == 'local_brightness'
    assert failure['min_L'] == zone['min_L'] and failure['max_L'] == zone['max_L']
    if too_bright:
        assert failure['measured_L'] > zone['max_L']
    else:
        assert failure['measured_L'] < zone['min_L']
    assert 'Local feature effect failed: rear_shadow_top: L=' in result['error']
    checked = result['effect_measurements']['rear_shadow_top']
    assert checked['effect_check_passed'] is False
    assert checked['max_added_step_L'] == 0
    assert checked['pair_count'] > 20
    assert checked['witness_sha256'] == review.digest(zone['mask'])
    with pytest.raises(review.EffectThresholdFailure, match='Local feature effect failed'):
        measured(review.measure_local, data, 'baseline')


def test_perimeter_and_all_local_brightness_failures_are_retained(tmp_path):
    data = fixture(tmp_path)
    spec, _, _, allowed, _, _, mask = data
    perimeter(spec, allowed, mask)
    spec['local_checks']['rear_shadow_bottom'] = copy.deepcopy(spec['local_checks']['rear_shadow_top'])
    result = measured(review.measure_perimeter, data)
    assert result['passed'] is False
    assert len(result['effect_failures']) == 10
    assert set(result['effect_measurements']) == set(review.ZONES)|{'rear_shadow_top', 'rear_shadow_bottom'}
    assert all(item['effect_check_passed'] is False for item in result['effect_measurements'].values())
    assert result['effect_measurements']['rear_shadow_bottom']['max_added_step_L'] == 0


@pytest.mark.parametrize('mode', ('regional', 'perimeter'))
@pytest.mark.parametrize('name', PROFILES)
def test_brightness_failure_never_hides_same_zone_continuity(tmp_path, mode, name):
    data = fixture(tmp_path)
    spec, _, output, allowed, _, rgba, mask = data
    if mode == 'perimeter':
        perimeter(spec, allowed, mask)
    rgba[16:, :, :3] += 40
    Image.fromarray(rgba).save(output)
    function = review.measure_perimeter if mode == 'perimeter' else review.measure_local
    with pytest.raises(ValueError, match='Lighting correction discontinuity') as caught:
        measured(function, data, name)
    assert not isinstance(caught.value, review.EffectThresholdFailure)


@pytest.mark.parametrize('fault', ('frozen', 'invalid_threshold', 'too_few_pairs'))
def test_earlier_dim_zone_never_hides_later_hard_gate(tmp_path, fault):
    data = fixture(tmp_path)
    spec, _, _, allowed, protected, _, mask = data
    first = np.zeros(allowed.shape, bool)
    first[:, :20] = True
    later = ~first
    spec['local_checks']['rear_shadow_top']['mask'] = mask('first', first)
    second = dict(mask=mask('later', later), min_L=0., max_L=100.)
    spec['local_checks']['later'] = second
    if fault == 'frozen':
        protected[20, 25] = True
        message = 'Local feature omitted or frozen: later'
    elif fault == 'invalid_threshold':
        second['min_L'] = float('nan')
        message = 'Invalid local feature threshold'
    else:
        tiny = np.zeros(allowed.shape, bool)
        tiny[0, 25] = True
        second.update(mask=mask('tiny', tiny), max_added_step_L=1.25)
        message = 'Too few same-material continuity pairs'
    with pytest.raises(ValueError, match=message) as caught:
        measured(review.measure_local, data)
    assert not isinstance(caught.value, review.EffectThresholdFailure)


@pytest.mark.parametrize('kind', ('normal_peer', 'boundary_peer', 'pixel'))
def test_brightness_failure_never_hides_other_continuity_gates(tmp_path, kind):
    data = fixture(tmp_path)
    spec, _, output, allowed, _, rgba, mask = data
    zone = spec['local_checks']['rear_shadow_top']
    zone.pop('max_added_step_L')
    left = np.zeros(allowed.shape, bool)
    left[:, :20] = True
    if kind == 'normal_peer':
        zone.update(mask=mask('left', left), normal_peer_mask=mask('right', ~left), max_normal_peer_step_L=1.25)
        rgba[16:, :20, :3] += 40
        message = 'Lighting normal peer discontinuity'
    elif kind == 'boundary_peer':
        zone.update(mask=mask('left', left), boundary_peer_mask=mask('right', ~left), max_boundary_step_L=2.)
        rgba[left, :3] += 40
        message = 'Lighting boundary discontinuity'
    else:
        zone['max_pixel_added_step_L'] = 1.25
        rgba[16, 20, :3] += 40
        message = 'Lighting pixel correction discontinuity'
    Image.fromarray(rgba).save(output)
    with pytest.raises(ValueError, match=message) as caught:
        measured(review.measure_local, data)
    assert not isinstance(caught.value, review.EffectThresholdFailure)


@pytest.mark.parametrize('fault', ('alpha', 'source_hash'))
def test_source_identity_and_alpha_remain_hard_failures(tmp_path, fault):
    data = fixture(tmp_path)
    spec, _, output, _, _, rgba, _ = data
    if fault == 'alpha':
        rgba[0, 0, 3] = 0
        Image.fromarray(rgba).save(output)
        message = 'Local output changed geometry'
    else:
        spec['source_sha256'] = '0'*64
        message = 'Lighting scope source mismatch'
    with pytest.raises(ValueError, match=message) as caught:
        measured(review.measure_local, data)
    assert not isinstance(caught.value, review.EffectThresholdFailure)


@pytest.mark.parametrize('message', (
    'Local feature effect failed: unaudited early error',
    'Local effect failed: unaudited early error',
    'Regional effect failed: unaudited early error',
    'Perimeter effect failed: unaudited early error',
    'Unsupported editable illumination: no source support',
    'Bottom paper material boundary unresolved: no source closure',
))
def test_strings_cannot_claim_all_mandatory_scope_checks_ran(message):
    def failure(*args):
        raise ValueError(message)
    with pytest.raises(ValueError) as caught:
        profile.measure_effect(failure, None, None, None, None, None, PROFILES[-1])
    assert str(caught.value) == message


def test_full_pass_retains_normal_measurement_return(tmp_path):
    data = fixture(tmp_path)
    data[0]['local_checks']['rear_shadow_top'].update(min_L=0., max_L=100.)
    result = measured(review.measure_local, data)
    assert 'status' not in result
    assert result['rear_shadow_top']['effect_check_passed'] is True
    assert result['rear_shadow_top']['max_added_step_L'] == 0


def test_stage_keeps_warned_candidate_without_claiming_quality_pass(tmp_path, monkeypatch):
    from phone_texture import stage_processing as stages
    from test_independent_stages import checkpoint, fake_stages, run_stages
    case, files, geometry, manifest = checkpoint(tmp_path)
    observed = fake_stages(monkeypatch)
    monkeypatch.setattr(stages, 'measure_effect', profile.measure_effect)

    def effect(scope, source, output, allowed, protected):
        if source.stem == 'back':
            raise review.EffectThresholdFailure([dict(kind='local_brightness', check='rear_shadow_top',
                measured_L=37.35, min_L=50., max_L=80.,
                error='Local feature effect failed: rear_shadow_top: L=37.35')],
                {'rear_shadow_top':dict(effect_check_passed=False, after_percentile_L=37.35)})
        return {'source_thresholds': {'effect_check_passed': True}}

    monkeypatch.setattr(stages.lighting_review, 'measure_local', effect)
    monkeypatch.setattr(stages.lighting_review, 'measure_perimeter', effect)
    result = run_stages(case, files, geometry, manifest)
    assert observed['lighting'] == list(stages.pipeline.FACES)
    assert result['stage_status'] == dict(geometry='complete', lighting='complete', structure='complete', seam='complete')
    assert result['per_face_stages']['back']['lighting'] == 'effect_warning'
    assert result['status'] == 'full_stages_effect_warning_needs_review'
    assert result['six_faces_processed'] is False
    assert result['mandatory_program_checks_complete'] is False
    assert result['quality_checks_complete'] is False
    assert result['visual_review_passed'] is False
    assert not result['blocked_checks']
    warning = result['lighting_effect_warnings'][0]
    assert warning['face'] == 'back'
    assert warning['passed'] is False and warning['original_effect_check_passed'] is False
    assert observed['structure_inputs']['back'] != stages.w.sha(files['back'])

