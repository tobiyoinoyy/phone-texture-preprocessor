"""A resolved straight lower rail need not extrapolate an unresolved corner."""
import numpy as np
import pytest
from PIL import Image

from phone_texture.auto_geometry import frame_geometry
from phone_texture.top_frame import _calibrate_top, calibrate_top
from test_top_frame import scene


def flat_lower_scene():
    matte, raw, _, _ = scene()
    a = np.array(matte); rgb = np.array(raw)
    yy, xx = np.indices(a.shape[:2])
    distance = np.minimum(xx-50, 849-xx)
    # The lower corner resolves in five pixels. Its remaining plateau is
    # measured, rather than inventing a long exponential decay.
    top = 200 + 35*np.exp(-np.maximum(distance, 0)/10)
    bottom = 330 - np.maximum(0, 5-distance)*1.2
    body = (distance >= 0) & (yy >= top) & (yy < bottom)
    camera = (xx >= 120) & (xx < 770) & (yy >= 130) & (yy < 230)
    rgb[:] = 230; rgb[camera] = [90, 65, 40]; rgb[body] = [180, 100, 60]
    return Image.fromarray(np.dstack([rgb, (body|camera).astype('uint8')*255])), Image.fromarray(rgb)


def test_flat_plateau_recovers_without_inventing_corner_or_retaining_camera():
    matte, raw = flat_lower_scene()
    with pytest.raises(ValueError, match='TOP_CORNER_AMBIGUOUS'):
        _calibrate_top(np.array(matte), np.array(raw))
    _, proof = frame_geometry(matte, 'top', source_image=raw)
    assert proof['method'] == 'top_body_corners_and_exposed_lower_v2'
    assert proof['automatic_checks']['body_changed_pixels'] == 0
    assert proof['transform_checks']['camera_after_rotation_pixels'] == 0
    assert proof['transform_checks']['body_lost_after_crop_pixels'] == 0
    assert max(proof['transform_checks']['residual_rail_degrees']) <= .15


def test_accepted_corner_calibration_is_exactly_unchanged():
    matte, raw, *_ = scene()
    expected = _calibrate_top(np.array(matte), np.array(raw))
    actual = calibrate_top(np.array(matte), np.array(raw))
    assert all(np.array_equal(a, b) for a, b in zip(expected[:2], actual[:2]))
    assert expected[2] == actual[2]


@pytest.mark.parametrize('corruption', ['missing_rgb', 'one_band', 'threshold_drift', 'corner_protrusion'])
def test_plateau_fallback_rejects_missing_or_conflicting_independent_evidence(corruption):
    matte, raw = flat_lower_scene(); a = np.array(matte); rgb = np.array(raw)
    if corruption == 'missing_rgb':
        rgb[:] = 128
    elif corruption == 'one_band':
        a[328:340, 90:150, 3] = 255
    elif corruption == 'threshold_drift':
        a[330:335, 80:820, 3] = 100
    else:
        a[328:340, 825:840, 3] = 255
    with pytest.raises(ValueError):
        calibrate_top(a, rgb)
