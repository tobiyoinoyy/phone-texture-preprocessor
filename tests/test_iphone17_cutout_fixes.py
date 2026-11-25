"""Regression tests for iPhone 17 geometry cutout fixes (01, 02, 07)."""
import numpy as np
import pytest
from PIL import Image, ImageDraw
from scipy import ndimage

from phone_texture.auto_geometry import (
    check_front_support_residue,
    exposed_lower_plane,
    frame_geometry,
    plate_geometry,
)
from phone_texture.top_frame import _calibrate_top, calibrate_top, trim_top_side_button
from test_top_exposed_lower import flat_lower_scene
from test_top_frame import scene as top_scene


def test_trim_top_side_button_removes_extreme_end_protrusion_and_recovers_top():
    matte, raw, _, _ = top_scene()
    a = np.array(matte)
    rgb = np.array(raw)
    # Normal top frame without a side button is completely untouched.
    untouched, trim_none = trim_top_side_button(a, rgb)
    assert trim_none is None
    assert np.array_equal(untouched, a)

    # Add a 12px mid-height protruding side button on the right extreme end.
    a_btn = a.copy()
    rgb_btn = rgb.copy()
    a_btn[245:285, 850:862, :3] = [140, 100, 70]
    a_btn[245:285, 850:862, 3] = 255
    rgb_btn[245:285, 850:862] = [140, 100, 70]

    trimmed, proof = trim_top_side_button(a_btn, rgb_btn)
    assert proof is not None
    assert proof['trimmed_ends'][0]['side'] == 'right'
    assert proof['trimmed_ends'][0]['trimmed_width_px'] == 12
    assert not trimmed[:, 850:862, 3].any()

    out, geom_proof = frame_geometry(
        Image.fromarray(a_btn), 'top', source_image=Image.fromarray(rgb_btn)
    )
    assert geom_proof['side_button_trim']['trimmed_ends'][0]['side'] == 'right'
    assert geom_proof['automatic_checks']['body_changed_pixels'] == 0
    assert out.width <= 804


def test_exposed_lower_rail_avoids_small_outer_dent_but_rejects_stand_projection():
    matte, raw = flat_lower_scene()
    a = np.array(matte)
    rgb = np.array(raw)
    # Add a realistic 3px localized dent inside the default right band (88..95% of 800 = 754..810).
    a[329:333, 780:808, :3] = [180, 100, 60]
    a[329:333, 780:808, 3] = 255
    rgb[329:333, 780:808] = [180, 100, 60]

    upper, lower, proof = calibrate_top(a, rgb)
    assert proof['method'] == 'top_body_corners_and_exposed_lower_v2'
    assert abs(float(np.polyval(lower, 450)) - 330.0) <= 1.5

    # A 12px stand projection inside the same band must still be rejected.
    a_bad = np.array(matte)
    rgb_bad = np.array(raw)
    a_bad[329:342, 760:810, :3] = [180, 100, 60]
    a_bad[329:342, 760:810, 3] = 255
    rgb_bad[329:342, 760:810] = [180, 100, 60]
    with pytest.raises(ValueError):
        calibrate_top(a_bad, rgb_bad)


def test_taper_aware_own_corners_removes_front_white_stand_clips():
    h, w = 900, 500
    yy, xx = np.indices((h, w))
    # Build a gently tapered (10px over shaft) rounded front phone body.
    # Top width = 280 (x=110..390), bottom width = 294 (x=103..397), y=60..690, r=36.
    t = np.clip((yy - 60) / 630.0, 0.0, 1.0)
    left = 110.0 - 7.0 * t
    right = 390.0 + 7.0 * t
    r = 36.0
    cx = np.clip(xx, left + r, right - r)
    cy = np.clip(yy, 60.0 + r, 690.0 - r)
    body = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r

    # Attach bright white central stand + two protruding clips at x=20%..80%.
    stand = (yy >= 685) & (xx >= 165) & (xx < 335)
    clips = (yy >= 685) & (yy < 725) & (((xx >= 158) & (xx < 185)) | ((xx >= 315) & (xx < 342)))
    mask = body | stand | clips

    raw = np.full((h, w, 3), 232, np.uint8)
    raw[stand | clips] = 220
    raw[body] = 22
    # Bright thin metal bezel at bottom rim of body.
    raw[body & (yy >= 686)] = 155
    matte = Image.fromarray(np.dstack([raw, mask.astype(np.uint8) * 255]))

    out, proof = plate_geometry(
        matte, 'front', source_image=Image.fromarray(raw),
        reconstruct_paper_tail=False, prefer_front_closure=True,
    )
    check = check_front_support_residue(out)
    assert not check['bright_support_residue']
    assert out.height <= 635
    assert proof['own_corner_evidence']['method'] == 'own_top_bottom_corner_curve_consensus_taper_v1'


def test_exposed_lower_plane_rejects_steep_diagonal_slope_mismatch():
    h, w = 420, 400
    y, x = np.mgrid[:h, :w]
    # Level phone body (body_slope=0), but left side has a 20px lower stand step
    # creating a ~3.5 degree diagonal connecting left step to right phone bottom.
    diag_bottom = 350 - 0.062 * (x - 50)
    body = (x >= 50) & (x < 350) & (y >= 60) & (y < diag_bottom)
    stand = (x >= 170) & (x < 230) & (y >= 320)
    mask = body | stand
    raw = np.where(mask[..., None], np.array([90, 100, 110]), np.array([220, 220, 220])).astype(float)
    grad = np.linalg.norm(np.gradient(ndimage.gaussian_filter1d(raw, .8, axis=0), axis=0), axis=2)

    with pytest.raises(ValueError, match='two-sided'):
        exposed_lower_plane(mask, grad, 50, 350, 330, body_slope=0.0)


def test_back_no_tail_trims_sticker_stand_lip_and_bottom_corner_wedge():
    h, w = 800, 480
    yy, xx = np.indices((h, w))
    t = np.clip((yy - 50) / 590.0, 0.0, 1.0)
    left = 100.0 - 5.0 * t
    right = 380.0 + 5.0 * t
    r = 38.0
    cx = np.clip(xx, left + r, right - r)
    cy = np.clip(yy, 50.0 + r, 640.0 - r)
    body = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r

    # Central stand + triangular wedge tucked under bottom-left rounded corner (x=115..160)
    stand = (yy >= 620) & (xx >= 115) & (xx < 320)
    mask = body | stand

    raw = np.full((h, w, 3), 232, np.uint8)
    raw[stand] = 135
    raw[body] = 205
    # Dark outer rim along body silhouette
    eroded = ndimage.binary_erosion(body, iterations=3)
    raw[body & ~eroded] = 55
    # Paper mask bleeds 16px below bottom bezel into gray stand under sticker
    paper = np.zeros((h, w), bool)
    paper[560:656, 185:225] = True

    matte = Image.fromarray(np.dstack([raw, mask.astype(np.uint8) * 255]))
    stages = {}
    out, proof = plate_geometry(
        matte, 'back', paper=paper,
        stage_sink=lambda name, im: stages.update({name: im.copy()}),
        source_image=Image.fromarray(raw),
        reconstruct_paper_tail=False,
    )
    after = np.asarray(stages['after_support'])
    # Stand block under sticker (y=644..656, below 1px AA margin) must be completely removed.
    assert not after[644:660, 185:225, 3].any()
    # Stand wedge strictly outside the 2px rounded-corner AA border must be trimmed to 0.
    outer_body = ndimage.binary_dilation(body, iterations=2)
    wedge_area = (~outer_body) & (yy >= 625) & (yy <= 640) & (xx >= 115) & (xx < 135)
    assert after[..., 3][wedge_area].mean() < 10
    # Rounded phone body interior must remain intact.
    assert np.all(after[eroded, 3] == 255)
    assert out.height <= 594
