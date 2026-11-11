"""Current evidence branch controls; no product images or SKU allowlist."""
from phone_texture.side_candidate_structure_observation import reference_contract

from copy import deepcopy

import pytest

UNKNOWN_REASON = (
    "Exposed two-flank rail evidence required; unsupported geometry branch"
)

def _geometry():
    return {
        "method": "source_exposed_plate_rgb_alpha_consensus_v1",
        "source_size": [100, 100],
        "output_size": [20, 80],
        "rotation_ccw_degrees": 0.0,
        "canvas_trim_xyxy": [0, 0, 20, 80],
        "visible_canvas_trim": {
            "source_size": [20, 80],
            "crop_xyxy": [0, 0, 20, 80],
            "output_size": [20, 80],
        },
        "clear_bands": None,
        "bottom_rail": [0.0, 79.0],
        "alpha_lower_rail": [0.0, 79.0],
        "own_corner_evidence": {
            "method": "own_top_bottom_corner_curve_consensus_v1",
            "symmetric_body_assumption": True,
            "own_corner_support_fractions": [0.8, 0.9],
            "corner_residual_p95_px": 2.0,
            "raw_rgb_supported_fraction": 0.9,
            "samples": 50,
        },
        "axis_evidence": {"method": "independent_long_side_axis_v1"},
    }

def _reference(status="source_tail_applicability_unknown"):
    applicability = {
        "status": status,
        "applicable": None,
        "reason": UNKNOWN_REASON,
    }
    if status == "no_in_shaft_remote_extension_detected":
        applicability["applicable"] = False
        applicability.pop("reason")
    return {
        "selected_coordinate_mode": "canvas_pixel_centers",
        "status": status,
        "reference_supported": False,
        "reference_certified": False,
        "applicability": applicability,
    }

def _set_path(document, path, value):
    current = document
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value

def _assert_review_flags(result, status, unknown):
    assert result["reference_certified"] is False
    assert result["reference_review_required"] is True
    assert result["independent_bottom_rails_certified"] is False
    assert result["new_remote_tail_measurement_performed"] is False
    assert result["geometry_changed"] is False
    assert result["original_status"] == status
    assert result["original_unknown_preserved"] is unknown

def test_existing_no_extension_screen_still_requires_uncertified_review():
    reference = _reference("no_in_shaft_remote_extension_detected")
    geometry = _geometry()
    before = deepcopy((reference, geometry))
    result = reference_contract(reference, geometry)
    _assert_review_flags(result, "no_in_shaft_remote_extension_detected", False)
    assert result["mode"] == "existing_canvas_reference_review"
    assert result["original_applicable"] is False
    assert (reference, geometry) == before

def test_exact_unknown_branch_is_an_own_corner_assumption_not_a_certificate():
    reference = _reference()
    geometry = _geometry()
    before = deepcopy((reference, geometry))
    result = reference_contract(reference, geometry)
    _assert_review_flags(result, "source_tail_applicability_unknown", True)
    assert result["mode"] == "existing_canvas_own_corner_assumption_review"
    assert result["original_applicable"] is None
    assert result["duplicated_rail_is_not_independent_evidence"] is True
    assert result["corner_plane_used_to_generate_targets"] is False
    assert (reference, geometry) == before

@pytest.mark.parametrize(
    "status", ["no_in_shaft_remote_extension_detected", "source_tail_applicability_unknown"]
)
@pytest.mark.parametrize("coordinate", ["body_endpoints", None, "canvas_pixel_centers "])
def test_wrong_or_missing_coordinate_mode_is_rejected(status, coordinate):
    reference = _reference(status)
    reference["selected_coordinate_mode"] = coordinate
    with pytest.raises(ValueError):
        reference_contract(reference, _geometry())

@pytest.mark.parametrize("applicable", [None, True])
def test_no_extension_status_requires_explicit_false_applicability(applicable):
    reference = _reference("no_in_shaft_remote_extension_detected")
    reference["applicability"]["applicable"] = applicable
    with pytest.raises(ValueError):
        reference_contract(reference, _geometry())

@pytest.mark.parametrize(
    "path,value",
    [
        (("status",), "other_unknown"),
        (("applicability", "reason"), "Other geometry evidence is unavailable"),
        (("applicability", "reason"), UNKNOWN_REASON + " "),
        (("applicability", "reason"), None),
        (("applicability", "applicable"), False),
        (("applicability", "applicable"), True),
    ],
)
def test_unknown_exception_is_limited_to_its_exact_recorded_reason(path, value):
    reference = _reference()
    _set_path(reference, path, value)
    with pytest.raises(ValueError):
        reference_contract(reference, _geometry())

@pytest.mark.parametrize(
    "path,value",
    [
        (("method",), "another_geometry_branch"),
        (("clear_bands",), [[1, 2], [18, 19]]),
        (("own_corner_evidence", "method"), "another_corner_method"),
        (("own_corner_evidence", "symmetric_body_assumption"), False),
        (("own_corner_evidence", "own_corner_support_fractions"), [0.599, 0.9]),
        (("own_corner_evidence", "own_corner_support_fractions"), [0.9, 0.599]),
        (("own_corner_evidence", "own_corner_support_fractions"), [0.9]),
        (("own_corner_evidence", "own_corner_support_fractions"), [0.9, 0.9, 0.9]),
        (("own_corner_evidence", "corner_residual_p95_px"), 3.001),
        (("own_corner_evidence", "raw_rgb_supported_fraction"), 0.599),
        (("axis_evidence", "method"), "another_axis_method"),
        (("alpha_lower_rail",), [0.0, 79.0001]),
    ],
)
def test_own_corner_exception_rejects_missing_strength_or_different_branch(path, value):
    geometry = _geometry()
    _set_path(geometry, path, value)
    before = deepcopy(geometry)
    with pytest.raises(ValueError):
        reference_contract(_reference(), geometry)
    assert geometry == before

@pytest.mark.parametrize("missing", ["own_corner_evidence", "axis_evidence"])
def test_missing_own_corner_or_axis_evidence_is_rejected(missing):
    geometry = _geometry()
    del geometry[missing]
    with pytest.raises(ValueError):
        reference_contract(_reference(), geometry)

def test_exact_corner_support_and_residual_boundaries_remain_allowed_for_review():
    geometry = _geometry()
    geometry["own_corner_evidence"].update(
        own_corner_support_fractions=[0.6, 0.6],
        corner_residual_p95_px=3.0,
        raw_rgb_supported_fraction=0.6,
    )
    result = reference_contract(_reference(), geometry)
    _assert_review_flags(result, "source_tail_applicability_unknown", True)
    assert result["mode"] == "existing_canvas_own_corner_assumption_review"

def test_own_corner_reference_requires_recorded_rotation_angle():
    geometry = _geometry()
    del geometry["rotation_ccw_degrees"]
    with pytest.raises(ValueError):
        reference_contract(_reference(), geometry)

def test_visible_crop_cannot_escape_the_first_crop_canvas():
    geometry = _geometry()
    # Width/height still equal output_size. Only the intermediate-canvas
    # boundary is violated, so checking final dimensions alone is insufficient.
    geometry["visible_canvas_trim"]["crop_xyxy"] = [1, 0, 21, 80]
    with pytest.raises(ValueError):
        reference_contract(_reference(), geometry)
