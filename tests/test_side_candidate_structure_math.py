"""Synthetic, static-prepared controls for contour-relative REVIEW transport.

No product images or frozen artifacts are used or modified. Test execution is
reserved for the parent's globally serialized window.
"""

import numpy as np
import pytest

from phone_texture.side_candidate_structure_math import (
    ReviewRejected,
    build_mapping,
    inverse_mass_coordinates,
    remap_array,
    row_mass_coordinates,
)


HEIGHT = 240
WIDTH = 20


def _plan():
    return build_mapping(
        HEIGHT, WIDTH,
        [
            dict(source_pair=[40, 46], observed_bounds=[39, 47], target_pair=[48, 54]),
            dict(source_pair=[194, 200], observed_bounds=[193, 201], target_pair=[186, 192]),
        ],
        [90, 150],
    )


def _rgba():
    yy, xx = np.indices((HEIGHT, WIDTH))
    rgba = np.empty((HEIGHT, WIDTH, 4), dtype=np.uint8)
    rgba[..., 0] = yy
    rgba[..., 1] = xx * 11
    rgba[..., 2] = (yy * 3 + xx * 5) % 251
    rgba[..., 3] = 0
    rgba[:, 2:18, 3] = 255
    return rgba


def _set_support(rgba, row, left, right):
    rgba[row, :, 3] = 0
    rgba[row, left:right, 3] = 255


def _assert_immutable(source, before, candidate):
    np.testing.assert_array_equal(source, before)
    assert candidate.shape == source.shape
    assert candidate.dtype == source.dtype
    np.testing.assert_array_equal(candidate[..., 3], source[..., 3])
    transparent = source[..., 3] == 0
    np.testing.assert_array_equal(candidate[transparent], source[transparent])
    np.testing.assert_array_equal(candidate[:16], source[:16])
    np.testing.assert_array_equal(candidate[224:], source[224:])
    np.testing.assert_array_equal(candidate[90:151], source[90:151])


def _assert_all_transport_donors_positive(source, transport):
    source_y = transport["source_y"]
    assert source_y.shape == (HEIGHT,)
    low_y = np.floor(source_y).astype(int)
    high_y = np.ceil(source_y).astype(int)
    fraction_y = source_y - low_y
    active = (np.abs(source_y - np.arange(HEIGHT)) > 1e-9)[:, None]
    active = active & (source[..., 3] > 0)
    np.testing.assert_array_equal(
        np.isfinite(transport["target_alpha_mass_u"]), active
    )
    for name, row_indices, used_y in (
        ("x_for_low_y", low_y, np.ones(HEIGHT, dtype=bool)),
        ("x_for_high_y", high_y, fraction_y > 0),
    ):
        coordinates = transport[name]
        assert coordinates.shape == (HEIGHT, WIDTH)
        assert coordinates.dtype == np.float64
        used = active & used_y[:, None]
        np.testing.assert_array_equal(np.isfinite(coordinates), used)
        yy, xx = np.nonzero(used)
        values = coordinates[yy, xx]
        left = np.floor(values).astype(int)
        right = np.ceil(values).astype(int)
        assert np.all((left >= 0) & (right < WIDTH))
        # ceil equals floor at exact centers: an unused alpha-zero neighbour
        # must not be mistaken for an actual contribution.
        assert np.all(source[row_indices[yy], left, 3] > 0)
        assert np.all(source[row_indices[yy], right, 3] > 0)


def test_uniform_mass_centers_invert_to_their_original_pixel_centers():
    alpha = np.array([0, 255, 255, 255, 0], dtype=np.uint8)
    before = alpha.copy()
    positive_x, u = row_mass_coordinates(alpha)
    np.testing.assert_array_equal(positive_x, [1, 2, 3])
    np.testing.assert_allclose(u, [1 / 6, 1 / 2, 5 / 6], rtol=0, atol=1e-12)
    source_x, evidence = inverse_mass_coordinates(alpha, u)
    np.testing.assert_allclose(source_x, positive_x, rtol=0, atol=1e-12)
    assert isinstance(evidence, dict)
    assert evidence["lower_extension_count"] == 0
    assert evidence["upper_extension_count"] == 0
    np.testing.assert_array_equal(alpha, before)


def test_nonuniform_mass_uses_alpha_weighted_centers():
    alpha = np.array([0, 64, 128, 64, 0], dtype=np.uint8)
    positive_x, u = row_mass_coordinates(alpha)
    np.testing.assert_array_equal(positive_x, [1, 2, 3])
    np.testing.assert_allclose(u, [0.125, 0.5, 0.875], rtol=0, atol=1e-12)
    assert np.all(np.diff(u) > 0)
    source_x, _ = inverse_mass_coordinates(alpha, u)
    np.testing.assert_allclose(source_x, positive_x, rtol=0, atol=1e-12)


def test_center_endpoint_extension_uses_only_positive_end_pixels():
    alpha = np.array([0, 0, 64, 128, 64, 0, 0], dtype=np.uint8)
    u = np.array([0.0, 0.125, 0.5, 0.875, 1.0])
    source_x, evidence = inverse_mass_coordinates(alpha, u)
    np.testing.assert_allclose(source_x, [2, 2, 3, 4, 4], rtol=0, atol=1e-12)
    assert np.all(alpha[np.floor(source_x).astype(int)] > 0)
    assert np.all(alpha[np.ceil(source_x).astype(int)] > 0)
    assert evidence["first_positive"] == 2
    assert evidence["last_positive"] == 4
    assert evidence["lower_extension_count"] == 1
    assert evidence["upper_extension_count"] == 1
    assert evidence["lower_u_range"] == [0.0, 0.0]
    assert evidence["upper_u_range"] == [1.0, 1.0]
    assert "constant" in evidence["endpoint_extension"]
    assert "not hole filling" in evidence["endpoint_extension"]


def test_one_positive_pixel_has_a_constant_positive_inverse():
    alpha = np.array([0, 0, 200, 0], dtype=np.uint8)
    positive_x, u = row_mass_coordinates(alpha)
    np.testing.assert_array_equal(positive_x, [2])
    np.testing.assert_allclose(u, [0.5], rtol=0, atol=1e-12)
    source_x, _ = inverse_mass_coordinates(alpha, np.array([0.0, 0.5, 1.0]))
    np.testing.assert_array_equal(source_x, [2.0, 2.0, 2.0])


@pytest.mark.parametrize(
    "values", [[], [0, 0, 0], [0, 255, 0, 255, 0]]
)
def test_empty_or_disconnected_positive_support_is_rejected(values):
    alpha = np.asarray(values, dtype=np.uint8)
    with pytest.raises(ReviewRejected):
        row_mass_coordinates(alpha)
    with pytest.raises(ReviewRejected):
        inverse_mass_coordinates(alpha, np.array([0.5]))


@pytest.mark.parametrize("dtype", [np.float64, np.int16, np.bool_])
def test_alpha_rows_require_uint8(dtype):
    alpha = np.array([0, 1, 1, 0], dtype=dtype)
    with pytest.raises(ReviewRejected):
        row_mass_coordinates(alpha)
    with pytest.raises(ReviewRejected):
        inverse_mass_coordinates(alpha, np.array([0.5]))


@pytest.mark.parametrize("u", [[-0.001], [1.001], [float("nan")], [float("inf")]])
def test_outside_or_nonfinite_mass_coordinate_is_not_clamped_or_filled(u):
    alpha = np.array([0, 255, 255, 0], dtype=np.uint8)
    with pytest.raises(ReviewRejected):
        inverse_mass_coordinates(alpha, np.asarray(u))


def test_unchanged_row_contours_preserve_horizontal_pixel_identity():
    source = _rgba()
    before = source.copy()
    candidate, proof = remap_array(source, _plan())
    _assert_immutable(source, before, candidate)
    np.testing.assert_array_equal(candidate[41:62, 2:18, :3], source[33:54, 2:18, :3])
    np.testing.assert_array_equal(candidate[179:200, 2:18, :3], source[187:208, 2:18, :3])
    assert proof["horizontal_resampling_applied"] is False
    assert proof["whole_core_rgb_rigid"] is False


def test_different_single_segment_contours_avoid_same_x_zero_donors():
    source = _rgba()
    _set_support(source, 42, 4, 16)
    before = source.copy()
    assert source[42, 2, 3] == 0 and source[50, 2, 3] == 255
    candidate, proof, transport = remap_array(source, _plan(), return_transport=True)
    _assert_immutable(source, before, candidate)
    _assert_all_transport_donors_positive(source, transport)
    # Target row 50 has 16 positive pixels; donor row 42 has 12. Endpoint
    # extension remains on positive x=4/15, not transparent same-x pixels.
    np.testing.assert_array_equal(candidate[50, 2, :3], source[42, 4, :3])
    np.testing.assert_array_equal(candidate[50, 17, :3], source[42, 15, :3])
    assert not np.array_equal(candidate[50, 2, :3], source[42, 2, :3])
    assert proof["horizontal_resampling_applied"] is True
    assert proof["whole_core_rgb_rigid"] is False
    assert proof["endpoint_extension_count"] > 0
    assert proof["all_actual_positive_xy_donors_checked"] is True
    assert proof["alpha_zero_actual_donors_used"] == 0


def test_each_used_y_donor_has_its_own_contour_inverse():
    source = _rgba()
    plan = _plan()
    # Target 69 lies on the noninteger upper shoulder, between rows 62 and 63.
    _set_support(source, 63, 5, 15)
    before = source.copy()
    candidate, proof, transport = remap_array(source, plan, return_transport=True)
    _assert_immutable(source, before, candidate)
    _assert_all_transport_donors_positive(source, transport)
    sy = float(transport["source_y"][69])
    assert 62 < sy < 62.5
    fraction = sy - 62
    assert transport["x_for_low_y"][69, 2] == 2
    assert transport["x_for_high_y"][69, 2] == 5
    assert transport["target_alpha_mass_u"][69, 2] == 1 / 32
    expected = np.rint(
        (1 - fraction) * source[62, 2, :3].astype(float)
        + fraction * source[63, 5, :3].astype(float)
    ).astype(np.uint8)
    np.testing.assert_array_equal(candidate[69, 2, :3], expected)
    assert proof["horizontal_resampling_applied"] is True
    assert proof["whole_core_rgb_rigid"] is False


def test_internal_transparent_hole_in_a_used_high_row_is_rejected():
    source = _rgba()
    source[63, 10, 3] = 0
    before = source.copy()
    with pytest.raises(ReviewRejected) as rejected:
        remap_array(source, _plan())
    assert rejected.value.evidence["row"] == 63
    np.testing.assert_array_equal(source, before)


def test_empty_consumed_donor_row_is_not_repaired_or_filled():
    source = _rgba()
    source[42, :, 3] = 0
    before = source.copy()
    with pytest.raises(ReviewRejected) as rejected:
        remap_array(source, _plan())
    assert rejected.value.evidence["row"] == 42
    np.testing.assert_array_equal(source, before)


def test_internal_nonzero_opacity_shortfall_remains_hard():
    source = _rgba()
    source[42, 7, 3] = 128
    before = source.copy()
    with pytest.raises(ReviewRejected):
        remap_array(source, _plan())
    np.testing.assert_array_equal(source, before)


def test_external_nonzero_opacity_shortfall_remains_explicit_review():
    source = _rgba()
    source[42, 2, 3] = 128
    before = source.copy()
    candidate, proof = remap_array(source, _plan())
    _assert_immutable(source, before, candidate)
    assert proof["review_required"] is True
    assert proof["opacity_review_events"]
    assert proof["alpha_zero_actual_donors_used"] == 0
