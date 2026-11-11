"""Unmarked paper needs a physical closure, not a guessed rectangle."""
import numpy as np
import pytest

from phone_texture.auto_material import trim_strip_to_source_end


def source(end=95, slope=0., antialias=False):
    h, w = 140, 700
    yy, xx = np.indices((h, w))
    a = np.full((h, w, 4), 255, np.uint8)
    a[..., :3] = [25, 35, 58]
    sides = (xx >= 290) & (xx < 350)
    physical_end = end+slope*(xx-290)
    actual = sides & (yy < physical_end)
    a[..., :3][actual] = [220, 225, 230]
    if antialias:
        aa = sides & (yy == end)
        a[..., :3][aa] = [80, 90, 110]
        actual |= aa
    for y in range(12, 76, 10):
        a[y:y+2, 300:340, :3] = 40
    proposed = sides.copy()
    proof = {'source_strip_edges': [[0, 290], [0, 349]]}
    return a, proposed, actual, proof


@pytest.mark.parametrize('slope', [0., .08, -.08])
def test_source_closed_unmarked_strip_preserves_paper_and_releases_metal(slope):
    a, mask, actual, proof = source(slope=slope)
    before, original_mask = a.copy(), mask.copy()
    result, evidence = trim_strip_to_source_end(a, mask, proof)
    np.testing.assert_array_equal(result, actual)
    np.testing.assert_array_equal(a, before)
    np.testing.assert_array_equal(mask, original_mask)
    closure = evidence['unmarked_source_closure']
    assert closure['source_end_support'] >= .75
    assert len(closure['post_edge_witnesses']) == 3
    assert closure['lower_continuation_checked_rows']
    assert not evidence['bottom_source_audit']['certifies_complete_paper']


def test_source_antialias_and_print_are_preserved():
    a, mask, actual, proof = source(antialias=True)
    result, _ = trim_strip_to_source_end(a, mask, proof)
    np.testing.assert_array_equal(result, actual)
    assert result[95, 290:350].all()
    assert result[32:34, 300:340].all()


def test_cross_edge_does_not_require_extrapolation_outside_paper_columns():
    a, mask, actual, proof = source(slope=.08)
    # The measured paper ends inside the image; extending its edge to unrelated
    # far-away metal columns would leave the image, which is irrelevant.
    a = np.pad(a, ((0, 0), (0, 900), (0, 0)), mode='edge')
    mask = np.pad(mask, ((0, 0), (0, 900)))
    actual = np.pad(actual, ((0, 0), (0, 900)))
    result, _ = trim_strip_to_source_end(a, mask, proof)
    np.testing.assert_array_equal(result, actual)


def test_paper_continuing_to_canvas_cannot_be_closed_from_side_fits():
    a, mask, _, proof = source(end=140)
    with pytest.raises(ValueError, match='Bottom paper end unresolved'):
        trim_strip_to_source_end(a, mask, proof)


def test_a_common_metal_shadow_cannot_supply_paper_specific_edge():
    a, mask, _, proof = source(end=140)
    a[95:, :, :3] = np.maximum(a[95:, :, :3].astype(int)-20, 0)
    with pytest.raises(ValueError, match='Bottom paper end unresolved'):
        trim_strip_to_source_end(a, mask, proof)


def test_an_interior_printed_stroke_cannot_close_paper_margins():
    a, mask, _, proof = source(end=140)
    a[95:102, 299:341, :3] = [25, 35, 58]
    with pytest.raises(ValueError, match='Bottom paper end unresolved'):
        trim_strip_to_source_end(a, mask, proof)


def test_full_width_printed_band_rejected_when_paper_reappears_below():
    a, mask, _, proof = source(end=140)
    a[95:107, 290:350, :3] = [25, 35, 58]
    with pytest.raises(ValueError, match='continue below'):
        trim_strip_to_source_end(a, mask, proof)


def test_different_material_below_edge_is_not_certified_as_metal():
    a, mask, _, proof = source()
    a[95:, 290:350, :3] = [100, 110, 130]
    with pytest.raises(ValueError, match='broad source metal'):
        trim_strip_to_source_end(a, mask, proof)


def test_transparent_background_is_not_a_metal_witness():
    a, mask, _, proof = source()
    a[:, :290, 3] = 0
    a[:, 350:, 3] = 0
    with pytest.raises(ValueError, match='Bottom paper end unresolved'):
        trim_strip_to_source_end(a, mask, proof)


def contaminated_reference(end=95):
    a, mask, actual, proof = source(end=end)
    # Subtle paper on silver metal, with a dark port/bevel outside the nearby
    # lateral metal. The distant feature is not material noise at the paper.
    a[..., :3] = [170, 180, 190]
    a[..., :3][actual] = [195, 205, 215]
    for y in range(12, 76, 10):
        a[y:y+2, 300:340, :3] = 40
    a[90:95, 270:276, :3] = 25
    return a, mask, actual, proof


def test_nearby_metal_can_resolve_paper_when_wide_reference_contains_port():
    a, mask, actual, proof = contaminated_reference()
    before = a.copy()
    result, evidence = trim_strip_to_source_end(a, mask, proof)
    np.testing.assert_array_equal(result, actual)
    np.testing.assert_array_equal(a, before)
    closure = evidence['unmarked_source_closure']
    assert closure['preceding_margin_witness']['sampling'] == 'local_before_broad_after'
    assert min(closure['preceding_margin_witness']['margin_fractions']) >= .75
    assert len(closure['post_edge_witnesses']) == 3


def test_local_margin_evidence_cannot_bypass_different_post_edge_material():
    a, mask, _, proof = contaminated_reference()
    a[95:, 290:350, :3] = [100, 110, 130]
    with pytest.raises(ValueError, match='broad source metal'):
        trim_strip_to_source_end(a, mask, proof)


def test_local_margin_evidence_cannot_close_a_full_width_print_band():
    a, mask, _, proof = contaminated_reference(end=140)
    a[95:107, 290:350, :3] = [170, 180, 190]
    with pytest.raises(ValueError, match='continue below'):
        trim_strip_to_source_end(a, mask, proof)


def test_distant_dark_feature_cannot_hide_continuing_paper():
    a, mask, _, proof = contaminated_reference(end=140)
    a[95:107, 290:350, :3] = [170, 180, 190]
    a[90:, 270:276, :3] = 25
    with pytest.raises(ValueError, match='continue below'):
        trim_strip_to_source_end(a, mask, proof)


def test_distant_dark_feature_cannot_hide_a_different_terminal_material():
    a, mask, _, proof = contaminated_reference()
    a[90:, 270:276, :3] = 25
    a[95:, 290:350, :3] = [100, 110, 130]
    with pytest.raises(ValueError, match='source metal'):
        trim_strip_to_source_end(a, mask, proof)
