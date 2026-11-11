"""DRAFT, NOT RUN. Existing engine behavior, not a changed acceptance gate."""
import numpy as np
import pytest
from phone_texture import lighting


def controls():
    masks = {'paper': np.zeros((40, 90), bool), 'metal': np.zeros((40, 90), bool)}
    masks['paper'][:, :40] = True
    masks['metal'][:, 40:] = True
    shift = np.zeros((40, 90, 3), float)
    shift[masks['paper'], 0] = 2.
    shift[masks['metal'], 0] = 24.
    source = np.zeros_like(shift); source[..., 0] = 45.
    config = dict(domains=['paper', 'metal'], boundary_domain='paper',
                  sigma=1.25, kind='source_material_transition')
    return shift, masks, source, config


def test_direct_blend_keeps_distinct_pure_material_platforms_and_texture():
    engine = lighting.engine('aluminum')
    shift, masks, source, config = controls()
    original_source = source.copy()
    result, proof = engine.blend_correction_boundary(shift, masks, config, source)
    # The seam is columns 39/40; integer support at sigma 1.25 is 36..43.
    np.testing.assert_array_equal(result[:, :36], shift[:, :36])
    np.testing.assert_array_equal(result[:, 44:], shift[:, 44:])
    np.testing.assert_array_equal(source, original_source)
    np.testing.assert_array_equal(result[..., 1:], shift[..., 1:])
    assert result[20, 39, 0] > 2.
    assert result[20, 40, 0] < 24.
    assert proof['gamut_continuity_pixels'] == 0
    assert np.all(result[..., 0] >= 2.-1e-12) and np.all(result[..., 0] <= 24.+1e-12)


def test_preblend_paper_budget_is_not_a_postblend_pixel_guarantee():
    engine = lighting.engine('aluminum')
    shift, masks, source, config = controls()
    result, _ = engine.blend_correction_boundary(shift, masks, config, source)
    assert np.max(shift[masks['paper'], 0]) <= 6.
    assert np.max(result[masks['paper'], 0]) > 6.
    # This is a documented risk witness, not approval of that pixel budget.


@pytest.mark.parametrize('update', [dict(sigma=2.01), dict(sigma=0),
    dict(sigma=float('nan')), dict(domains=['paper', 'metal', 'third']),
    dict(domains=['paper', 'paper']), dict(domains=['paper', 'unknown'])])
def test_invalid_source_transition_is_rejected(update):
    engine = lighting.engine('aluminum')
    shift, masks, source, config = controls(); config.update(update)
    with pytest.raises(ValueError):
        engine.blend_correction_boundary(shift, masks, config, source)


def test_nonneighbor_materials_do_not_create_an_invented_transition():
    engine = lighting.engine('aluminum')
    shift, masks, source, config = controls()
    masks['paper'][:, 38:40] = False
    masks['metal'][:, 40:42] = False
    with pytest.raises(ValueError):
        engine.blend_correction_boundary(shift, masks, config, source)
