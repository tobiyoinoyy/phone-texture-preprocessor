"""Gamut safety must not reinstate a smooth photographic shadow."""
import numpy as np
from phone_texture import lighting


def test_smooth_shadow_lift_is_not_limited_to_highlights_displacement():
    engine = lighting.engine('aluminum')
    h, w = 40, 80
    source = np.zeros((h, w, 3), float)
    source[..., 0] = np.linspace(94, 34, w)
    # Independently planted source texture is never part of the correction.
    source[10:20, 55:57, 0] -= 4
    desired = source.copy()
    desired[..., 0] += np.linspace(1, 61, w)
    safe = desired.copy()
    safe[:, 0, 0] = source[:, 0, 0] + .5
    mask = np.ones((h, w), bool)
    old, _ = engine.continuous_gamut_limits(desired, safe, source, mask, .6)
    result, proof = engine.continuous_gamut_limits(
        desired, safe, source, mask, .6, follow_field=True)
    assert old[15, 60, 0] < desired[15, 60, 0] - 5
    np.testing.assert_array_equal(result[:, 1:], desired[:, 1:])
    np.testing.assert_array_equal(result[:, 0], safe[:, 0])
    assert result[15, 54, 0] - result[15, 55, 0] == 4
    assert proof['follows_fitted_correction_gradient']


def test_gradient_propagation_does_not_cross_paper_or_wrap_canvas():
    engine = lighting.engine('aluminum')
    source = np.zeros((12, 30, 3), float)
    source[..., 0] = 50
    desired = source.copy(); desired[..., 0] += 20
    safe = desired.copy(); safe[:, 0, 0] = 50
    mask = np.ones((12, 30), bool); mask[:, 10] = False
    result, _ = engine.continuous_gamut_limits(
        desired, safe, source, mask, .6, follow_field=True)
    np.testing.assert_array_equal(result[:, 10:], desired[:, 10:])
    assert result[6, 1, 0] < desired[6, 1, 0]
