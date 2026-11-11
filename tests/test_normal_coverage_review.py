"""Missing measurement coverage is a review state, never a quality pass."""
import numpy as np
import pytest
from phone_texture import normal_review_state as normal
from phone_texture import shadow_lift_profile as profile
from phone_texture import web_app


def masks(peer_count):
    target = np.zeros((3, 60), bool)
    peer = target.copy()
    target[:2, :25] = True
    peer[:2, 30:30+peer_count] = True
    return target, peer


def test_empty_and_limited_coverage_are_not_passes():
    empty = np.zeros((3, 60), bool)
    zero = np.zeros_like(empty, dtype=float)
    result = normal.measure_normal(zero, empty, empty, 1.25)
    assert result['status'] == 'insufficient' and not result['complete']
    assert result['max_observed_residual_L75'] is None
    assert result['not_applicable_certified'] is False
    target, peer = masks(8)
    result = normal.measure_normal(zero, target, peer, 1.25)
    assert result['status'] == 'limited' and not result['complete']
    assert result['rows'][0]['witness_pairs'] == 25
    assert result['rows'][0]['peer_pairs'] == 8
    assert result['max_observed_residual_L75'] == 0


@pytest.mark.parametrize('peer_count', [8, 20])
def test_measurable_normal_regression_remains_hard_even_when_coverage_limited(peer_count):
    target, peer = masks(peer_count)
    delta = np.zeros(target.shape, float)
    delta[1, :25] = 4
    with pytest.raises(ValueError, match='normal peer discontinuity'):
        normal.measure_normal(delta, target, peer, 1.25)


def test_complete_normal_coverage_still_measures_the_original_limit():
    target, peer = masks(20)
    delta = np.zeros(target.shape, float)
    delta[1, :25] = .5
    result = normal.measure_normal(delta, target, peer, 1.25)
    assert result['complete'] and result['qualified_target_pairs'] == 25
    assert result['max_observed_residual_L75'] == .5


def test_missing_coverage_warning_does_not_claim_all_checks_completed():
    def unavailable(*args):
        raise normal.EffectCoverageUnavailable({'bottom': {'status': 'insufficient'}}, {}, [])
    result = profile.measure_effect(unavailable, None, None, None, None, None, profile.NAME)
    assert result['kind'] == 'normal_coverage_unavailable'
    assert not result['passed'] and not result['original_effect_check_passed']
    assert not result['non_aesthetic_scope_checks_complete']
    assert not result['normal_scope_checks_complete']
    assert not result['quality_checks_complete']
    with pytest.raises(normal.EffectCoverageUnavailable):
        profile.measure_effect(unavailable, None, None, None, None, None, 'baseline')


def test_web_does_not_describe_missing_normal_coverage_as_failed_brightness():
    row = dict(status='full_stages_effect_warning_needs_review', six_faces_generated=True,
        stage_status={stage: 'complete' for stage in ('geometry', 'lighting', 'structure', 'seam')},
        lighting_effect_warnings=[dict(face='bottom', kind='normal_coverage_unavailable', error='missing peers')])
    value = web_app.processing_state(row)
    text = str(value)
    assert '法向对照不足' in text and '局部提亮未达' not in text
