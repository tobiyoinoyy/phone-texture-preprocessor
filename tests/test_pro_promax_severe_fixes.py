"""Targeted regression tests for Batch 2 iPhone 17 Pro / Pro Max severe fixes."""
import json
import numpy as np
import pytest
from PIL import Image, ImageDraw
from scipy import ndimage


def test_rear_camera_glass_excludes_mid_contrast_lidar_on_bright_silver_back():
    from phone_texture.auto_material import rear_camera_glass

    h, w = 1000, 500
    a = np.full((h, w, 4), 255, dtype=np.uint8)
    a[..., :3] = 228  # Bright silver aluminum plateau.
    yy, xx = np.indices((h, w))
    # Three large dark lenses occupy ~15% of the upper 30% so p15 lands ~55.
    for cx, cy, r in ((95, 95, 50), (95, 225, 50), (225, 160, 50)):
        a[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r, :3] = 22
    # Compact circular LiDAR scanner at ~56 gray (just above p15, huge contrast vs 228 silver).
    lidar = (xx - 395) ** 2 + (yy - 220) ** 2 <= 24 * 24
    a[lidar, :3] = 56
    # Thin dark cast-shadow arc must still be rejected.
    arc = ((xx - 395) ** 2 + (yy - 95) ** 2 <= 25 * 25) & ((xx - 395) ** 2 + (yy - 95) ** 2 >= 21 * 21)
    a[arc, :3] = 45

    glass, proof = rear_camera_glass(a, a[..., 3] > 0)
    assert glass[lidar].mean() > 0.95
    assert len(proof["mid_contrast_accepted"]) == 1
    assert not glass[arc].any()


def test_dark_rear_mark_detects_dark_blue_emblem_with_adaptive_deficit_trio():
    from phone_texture.rear_lighting_features import dark_rear_mark

    h, w = 1000, 500
    a = np.full((h, w, 4), 255, dtype=np.uint8)
    # Dark anodized blue back (~74 gray) with dark emblem (~39 gray, contrast ~35).
    a[..., :3] = [56, 74, 102]
    yy, xx = np.indices((h, w))
    body = ((xx - 250) / 50.0) ** 2 + ((yy - 635) / 48.0) ** 2 <= 1.0
    leaf = ((xx - 262) / 13.0) ** 2 + ((yy - 568) / 14.0) ** 2 <= 1.0
    a[body | leaf, :3] = [28, 39, 55]
    panel = np.ones((h, w), dtype=bool)

    mask, proof = dark_rear_mark(a, panel)
    assert proof["detected"] is True
    assert proof["threshold_deficits"] == [15, 20, 25]
    assert mask[body].mean() > 0.95 and mask[leaf].mean() > 0.95

    # Faint low-contrast smudge (<24 gray contrast) must still be ignored.
    faint = a.copy()
    faint[body | leaf, :3] = [44, 60, 86]
    faint_mask, faint_proof = dark_rear_mark(faint, panel)
    assert faint_proof["detected"] is False
    assert not faint_mask.any()


def test_truncated_dark_control_rejected_and_lighting_hardware_recovers_sapphire_button():
    from phone_texture.side_material import _is_truncated_dark_control, lighting_hardware

    h, w = 1200, 120
    a = np.full((h, w, 4), 255, dtype=np.uint8)
    a[..., :3] = [165, 175, 185]  # Metallic rail.
    yy, xx = np.indices((h, w))

    def draw_capsule(arr, cx, y0, y1, r, color, rim=None):
        cap = (xx - cx) ** 2 + (yy - np.clip(yy, y0 + r, y1 - r)) ** 2 <= r * r
        arr[cap, :3] = color
        if rim is not None:
            ring = ndimage.binary_dilation(cap, iterations=2) & ~cap
            arr[ring, :3] = rim
        return cap

    # Power button (upper half): metallic anodized button with clear dark groove.
    draw_capsule(a, 60, 340, 485, 20, [160, 170, 180], rim=[70, 75, 80])
    # Camera control (lower half): dark sapphire with a specular streak splitting it.
    sapphire = draw_capsule(a, 60, 720, 875, 24, [45, 52, 62])
    # Add a small specular highlight spot inside the sapphire (lowers fill below 0.98).
    a[795:808, 45:75, :3] = [145, 155, 165]

    # A prematurely truncated box ending at y=830 inside dark sapphire must be flagged.
    assert _is_truncated_dark_control(a, [36, 720, 84, 830]) is True
    # The full capsule ending at y=875 transitions cleanly back to bright metal.
    assert _is_truncated_dark_control(a, [36, 720, 84, 875]) is False

    comps, _ = lighting_hardware(Image.fromarray(a), "right")
    assert len(comps) == 2
    # Lower component covers the full dark sapphire control (not truncated at y=800).
    lower_mask, lower_proof = comps[1]
    assert lower_proof["box"][1] <= 725 and lower_proof["box"][3] >= 870
    assert (lower_mask & sapphire).sum() / sapphire.sum() > 0.85


def test_frame_end_color_corrects_desaturated_blue_and_dim_silver_ends_while_preserving_good_silver():
    from phone_texture.frame_end_color import correct

    # 1. Deep-blue top frame with desaturated gray tips (tint ~6.0, loss ~3.0 -> previously skipped at 7.0).
    blue = np.full((80, 700, 4), [68, 88, 122, 255], dtype=np.uint8)
    blue[:, :60, :3] = [71, 79, 98]
    blue[:, -60:, :3] = [66, 74, 92]
    mask = np.ones((80, 700), dtype=bool)
    out_blue, rep_blue = correct(blue, mask)
    assert [r["status"] for r in rep_blue] == ["corrected", "corrected"]
    assert out_blue[40, 20, 2] > blue[40, 20, 2] + 5

    # 2. Silver frame with moderately dim end (loss ~6.0 -> previously skipped at 7.0).
    dim_silver = np.full((80, 700, 4), [226, 228, 229, 255], dtype=np.uint8)
    dim_silver[:, :60, :3] = [207, 209, 211]
    _, rep_dim = correct(dim_silver, mask)
    assert [r["status"] for r in rep_dim] == ["corrected", "within_tolerance"]

    # 3. Good bright silver with minor 2 L darkening stays within_tolerance.
    good_silver = np.full((80, 700, 4), [233, 238, 240, 255], dtype=np.uint8)
    good_silver[:, :50, :3] = [228, 233, 235]
    out_good, rep_good = correct(good_silver, mask)
    assert np.array_equal(out_good, good_silver)
    assert [r["status"] for r in rep_good] == ["within_tolerance", "within_tolerance"]


def test_frame_contact_recovers_bottom_with_curved_tip_and_adjacent_camera_lens_pair():
    from phone_texture.frame_contact import recover_frame

    size = (900, 550)
    body = Image.new("L", size)
    draw = ImageDraw.Draw(body)
    draw.rounded_rectangle((50, 180, 849, 279), radius=24, fill=255)
    body_mask = np.asarray(body) > 0
    yy, xx = np.indices(body_mask.shape)
    # Off-center pedestal on left side (x = 120..520).
    stand = (xx >= 120) & (xx <= 520) & (yy >= 278)
    # Two adjacent camera lenses protruding above the upper-left rail with a
    # narrow valley still above the rail (y=171 vs rail y=180).
    lens1 = (xx >= 145) & (xx <= 245) & (yy >= 160) & (yy <= 185)
    lens2 = (xx >= 275) & (xx <= 375) & (yy >= 159) & (yy <= 185)
    valley = (xx >= 245) & (xx <= 275) & (yy >= 174) & (yy <= 185)
    cameras = lens1 | lens2 | valley

    raw = np.full((*body_mask.shape, 3), 215, dtype=np.uint8)
    raw[stand] = 115
    raw[cameras] = 75
    raw[body_mask] = [205, 130, 85]
    rgba = np.dstack([raw, (body_mask | stand | cameras).astype(np.uint8) * 255])

    out, proof = recover_frame(Image.fromarray(rgba), Image.fromarray(raw), "bottom")
    assert proof["method"] == "source_supported_local_frame_contacts"
    assert proof["transform_checks"]["residual_projection_pixels"] == 0
    assert 90 <= out.height <= 115


def test_rear_panel_recovers_true_top_seam_below_strong_camera_plateau_bevel():
    from phone_texture.auto_material import rear_panel

    h, w = 1000, 480
    a = np.full((h, w, 4), 255, dtype=np.uint8)
    a[..., :3] = 150
    yy, xx = np.indices((h, w))
    # Strong camera plateau bevel near 0.28*h (y=280) with NO inset panel corners.
    a[279:282, 40:440, :3] = 40
    # True inset rear glass panel at y=335..955 with mild dark seam (~125 vs 150) & 4 rounded corners.
    left, right, top, bottom, r = 40, 440, 335, 955, 58
    inside = (xx >= left) & (xx <= right) & (yy >= top) & (yy <= bottom)
    for cx, cy in ((left + r, top + r), (right - r, top + r), (left + r, bottom - r), (right - r, bottom - r)):
        corner = ((xx < cx) if cx < w / 2 else (xx > cx)) & ((yy < cy) if cy < h / 2 else (yy > cy))
        inside &= ~corner | ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r)
    seam = inside & ~ndimage.binary_erosion(inside, iterations=2)
    a[seam, :3] = 115

    _, proof = rear_panel(Image.fromarray(a))
    assert abs(proof["bounds"][1] - top) <= 3
    assert all(c["supported"] for c in proof["corners"])


def test_back_paper_mask_excludes_semitransparent_bottom_fringe_whisker():
    from phone_texture.auto_material import paper_mask

    h, w = 800, 400
    a = np.full((h, w, 4), 255, dtype=np.uint8)
    a[..., :3] = [45, 60, 88]  # Dark blue back.
    # Printed white sticker near bottom-center.
    a[650:797, 170:220, :3] = 240
    a[670:780:6, 178:212:4, :3] = 20  # Printed ink.
    # Semi-transparent white studio-background fringe along the bottom 3 rows.
    a[797:800, 80:320, :3] = 235
    a[797:800, 80:320, 3] = 160

    mask, _ = paper_mask(Image.fromarray(a), "back", resolve_end=False)
    assert mask[700, 195]
    # Must not bleed horizontally along the semi-transparent bottom fringe!
    assert not mask[798, 100]
    assert not mask[798, 280]


def test_side_rail_post_sweep_prevents_gamut_cliff_discontinuity():
    from phone_texture.lighting_spot_review import side_rail

    h, w = 90, 720
    alpha = np.full((h, w), 255, dtype=np.uint8)
    main = np.ones((h, w), dtype=bool)
    source = np.full((h, w, 3), 75.0)
    # Bright specular highlight at x=320 pinned near L=99 (so max safe lift = 5 L).
    source[:20, 318:323, 0] = 94.0
    safe = source.copy()
    safe[..., 0] = np.minimum(99.0, source[..., 0] + 20.0)
    # Gamut-limited smooth field drops gently to 5 L around x=320.
    limited = safe.copy()
    xx = np.arange(w)
    ramp = np.minimum(20.0, 5.0 + 0.6 * np.maximum(0, np.abs(xx - 320) - 2))
    limited[:20, :, 0] = np.minimum(safe[:20, :, 0], source[:20, :, 0] + ramp[None, :])
    before = limited.copy()

    side_rail(source, limited, safe, alpha, main)
    added = limited[:20, :, 0] - source[:20, :, 0]
    # Adjacent added-L step along the rail must never spike to a multi-L cliff!
    assert float(np.abs(np.diff(added, axis=1)).max()) <= 1.05
    # Non-lifted regions remain unchanged.
    assert np.all(limited[..., 0] >= before[..., 0] - 1e-6)


def test_uv_seam_tolerates_short_terminal_transparent_corner_tip_but_rejects_uneditable_visible(tmp_path):
    from phone_texture.frame_seams import seam_engine

    eng = seam_engine()
    h, w = 100, 100
    a = np.full((h, w, 4), [140, 150, 165, 255], dtype=np.uint8)
    b = np.full((h, w, 4), [150, 160, 175, 255], dtype=np.uint8)
    # Corner tip (rows 0..3) is transparent on view b (~4% of the 100px seam).
    b[:4, :, 3] = 0
    mask_a = np.full((h, w), 255, dtype=np.uint8)
    mask_b = np.where(b[..., 3] > 0, 255, 0).astype(np.uint8)
    prot = np.zeros((h, w), dtype=np.uint8)

    for name, arr in (("a.png", a), ("b.png", b)):
        Image.fromarray(arr).save(tmp_path / name)
    Image.fromarray(mask_a).save(tmp_path / "ma.png")
    Image.fromarray(mask_b).save(tmp_path / "mb.png")
    Image.fromarray(prot).save(tmp_path / "prot.png")
    (tmp_path / "scene.json").write_text("{}")

    ys = np.linspace(0.5, 98.5, 13)
    chain = dict(
        names=["a", "b"],
        edges=[0],
        arc=np.linspace(0.0, 1.0, 13).tolist(),
        sides=[
            dict(name="a", image_size=[w, h], px_per_unit=100.0,
                 points=[[5.0, float(y)] for y in ys], normals=[[1.0, 0.0]] * 13),
            dict(name="b", image_size=[w, h], px_per_unit=100.0,
                 points=[[5.0, float(y)] for y in ys], normals=[[1.0, 0.0]] * 13),
        ],
    )
    (tmp_path / "chains.json").write_text(json.dumps(dict(chains=[chain])))
    job = dict(
        scene=str(tmp_path / "scene.json"),
        seam_chains=str(tmp_path / "chains.json"),
        views=dict(
            a=dict(source=str(tmp_path / "a.png"), editable_mask=str(tmp_path / "ma.png"), protected_mask=str(tmp_path / "prot.png")),
            b=dict(source=str(tmp_path / "b.png"), editable_mask=str(tmp_path / "mb.png"), protected_mask=str(tmp_path / "prot.png")),
        ),
        output_dir=str(tmp_path / "out1"),
        records_dir=str(tmp_path / "rec1"),
        width_units=0.20,
        sigma=2,
        guard_pixels=6,
        protection_feather=0,
        minimum_joint_coverage=0.98,
        maximum_p95_deltaE76=2.0,
    )
    jp = tmp_path / "job1.json"
    jp.write_text(json.dumps(job))
    report = eng.run(jp)
    assert report["seams"][0]["after_p95_deltaE76"] <= 2.0

    # Visible non-editable protected pixels (>2% of visible seam) must STILL fail closed!
    prot_bad = prot.copy()
    prot_bad[40:48, :] = 255
    Image.fromarray(prot_bad).save(tmp_path / "prot_bad.png")
    job2 = dict(job, output_dir=str(tmp_path / "out2"), records_dir=str(tmp_path / "rec2"))
    job2["views"]["b"]["protected_mask"] = str(tmp_path / "prot_bad.png")
    jp2 = tmp_path / "job2.json"
    jp2.write_text(json.dumps(job2))
    with pytest.raises(ValueError, match="jointly editable material coverage"):
        eng.run(jp2)


def test_antenna_rows_prefers_clean_recovered_dominant_band_over_weak_lower_candidate(monkeypatch):
    from phone_texture import antenna_material, auto_structure

    # Synthetic side strip with real dark antennas at [160..180] and [1820..1840],
    # plus a weaker bright band at [125..155] directly above band 0.
    h, w = 2000, 120
    a = np.full((h, w, 4), 255, dtype=np.uint8)
    a[..., :3] = 120
    a[125:155, :, :3] = 136  # Weak bright band (opposite polarity).
    a[160:180, :, :3] = 75   # Real dark antenna 0 (dominant!).
    a[1820:1840, :, :3] = 75  # Real dark antenna 1.

    orig_trace = antenna_material.trace_boundaries

    def fake_trace(image, rows, bands=(0, 1), *, recover_center=True, **kw):
        if bands == (0,):
            lo, hi = int(rows[0]), int(rows[1])
            if abs(lo - 160) <= 3 and abs(hi - 180) <= 3:
                if not recover_center:
                    raise ValueError("Opaque antenna endpoint lacks paired source evidence")
                return np.zeros((2, w)), dict(endpoint_shape_review_required=False, endpoint_shape_weak=[])
            if abs(lo - 125) <= 3 and abs(hi - 155) <= 3:
                return np.zeros((2, w)), dict(endpoint_shape_review_required=True, endpoint_shape_weak=[{"band": 0, "side": "left"}])
            raise ValueError("Insufficient central paired antenna support")
        return orig_trace(image, rows, bands=bands, recover_center=recover_center, **kw)

    monkeypatch.setattr(antenna_material, "trace_boundaries", fake_trace)
    rows, _, _ = auto_structure.antenna_rows(Image.fromarray(a), [48, 72], require_full_span=True)
    assert abs(rows[0] - 160) <= 2 and abs(rows[1] - 180) <= 2


def test_harmonize_back_rim_spots_includes_camera_surround_when_rear_shadow_unavailable(tmp_path):
    from phone_texture.lighting import engine

    alum = engine("aluminum")
    h, w = 400, 200
    a = np.full((h, w, 4), 255, dtype=np.uint8)
    a[..., :3] = [190, 115, 75]
    # Add a small dark spot straddling the rear_perimeter/camera_surround boundary (y=12..22).
    a[10:22, 95:105, :3] = [150, 85, 50]
    src = tmp_path / "back.png"
    Image.fromarray(a).save(src)
    perim = np.zeros((h, w), dtype=np.uint8)
    perim[:16, :] = 255
    perim[-16:, :] = 255
    perim[:, :16] = 255
    perim[:, -16:] = 255
    cam = np.zeros((h, w), dtype=np.uint8)
    cam[16:140, 16:184] = 255
    main = np.zeros((h, w), dtype=np.uint8)
    main[140:384, 16:184] = 255
    for name, m in (("perim.png", perim), ("cam.png", cam), ("main.png", main)):
        Image.fromarray(m).save(tmp_path / name)
    item = dict(
        name="back",
        source=str(src),
        source_sha256=alum.digest(src),
        protection_policy="none",
        camera_metal_layout="triple",
        reference_boxes=[[30, 180, 50, 200]],
        domains=[
            dict(name="rear_perimeter", mask=str(tmp_path / "perim.png"), reference_boxes=[[20, 4, 40, 12]], sigma=[10, 4], strength=[0.8, 0, 0]),
            dict(name="camera_surround", mask=str(tmp_path / "cam.png"), reference_boxes=[[30, 60, 50, 80]], sigma=[10, 10], strength=[0.8, 0, 0]),
            dict(name="main_surface", mask=str(tmp_path / "main.png"), reference_boxes=[[30, 180, 50, 200]], sigma=[10, 10], strength=[0.8, 0, 0]),
        ],
        correction_blends=[dict(domains=["rear_perimeter", "camera_surround", "main_surface"], boundary_domain="rear_perimeter", sigma=4.0)],
    )
    target, _ = alum.target_color({"views": [item]})
    dst, _, _, _ = alum.process_view(item, target, dict(color_mode="source_lightness"))
    dL = alum.rgb_to_lab(dst[..., :3])[..., 0] - alum.rgb_to_lab(a[..., :3])[..., 0]
    # No artificial jump across y=15 -> y=16 boundary.
    assert float(np.percentile(np.abs(dL[15, 20:180] - dL[16, 20:180]), 99)) < 2.0


def test_current_seam_readings_mirrors_terminal_transparent_corner_tip_tolerance(tmp_path):
    from phone_texture.iphone17_model_finish import _current_seam_readings

    h, w = 100, 100
    files, masks = {}, {}
    for face in ("left", "right", "top", "bottom"):
        arr = np.full((h, w, 4), [145, 155, 170, 255], dtype=np.uint8)
        if face == "right":
            # 4-row transparent corner tip (~4% of seam).
            arr[:4, :, 3] = 0
        p = tmp_path / f"{face}.png"
        Image.fromarray(arr).save(p)
        files[face] = p
        masks[face] = arr[..., 3] > 0

    ys = np.linspace(0.5, 98.5, 13)
    chain = dict(
        names=["top", "right"],
        edges=[0],
        arc=np.linspace(0.0, 1.0, 13).tolist(),
        sides=[
            dict(name="top", points=[[5.0, float(y)] for y in ys]),
            dict(name="right", points=[[5.0, float(y)] for y in ys]),
        ],
    )
    readings = _current_seam_readings(files, [chain], masks, sigma=2)
    assert len(readings) == 1
    assert readings[0]["after_p95_deltaE76"] < 0.5

    # Interior non-editable gap on visible pixels (> 2%) must still raise!
    masks_bad = dict(masks)
    m = masks["right"].copy()
    m[40:48, :] = False
    masks_bad["right"] = m
    with pytest.raises(ValueError, match="jointly editable photographed metal"):
        _current_seam_readings(files, [chain], masks_bad, sigma=2)


def test_back_lens_disks_per_slot_consensus_recovers_dark_body_lenses():
    import cv2
    from phone_texture.back_lens import _disks

    # Dark blue plateau (gray ~ 56) with three lenses whose asymmetric shadow halos
    # shift individual lenses slightly across coarse thresholds (breaking joint 1.5px).
    h, w = 1000, 500
    a = np.full((h, w, 4), 255, dtype=np.uint8)
    a[..., :3] = 56
    centres = [(98, 92, 46), (100, 224, 46), (225, 158, 47)]
    for cx, cy, r in centres:
        cv2.circle(a, (cx, cy), r, (18, 18, 18, 255), -1)
        # Soft dark shadow halo shifting slightly off-center at higher gray thresholds.
        cv2.circle(a, (cx + 4, cy), r + 3, (46, 46, 46, 255), 3)
    disks, count = _disks(a)
    assert disks is not None and count >= 3
    for got, (cx, cy, _) in zip(disks, centres):
        assert np.hypot(got[0] - cx, got[1] - cy) < 3.0


def test_rear_shadow_plateau_core_accepts_coherent_top_bottom_ledges():
    from phone_texture.rear_shadow import InsufficientRearShadowEvidence, _plateau_core

    h, w = 1000, 500
    yy, xx = np.indices((h, w))
    # Rounded-rectangle camera plateau [32..465, 32..268] with corner radius 40.
    r = 40.0
    cx = np.clip(xx, 32.0 + r, 465.0 - r)
    cy = np.clip(yy, 32.0 + r, 268.0 - r)
    inside = ((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r
    L = np.where(inside, 74.0, 82.0)
    # Soften bottom step in left 2/3 of the bottom check span (.57..80 of w = 285..400)
    # so 2 of 3 windows have ~0.85 L/px gradient while all 3 agree on exact row 268.
    L[(~inside) & (yy >= 268) & (xx >= 285) & (xx < 365)] = 76.2
    bounds, proof = _plateau_core(L)
    assert abs(bounds[3] - 268) <= 8

    # Completely flat featureless image must still fail closed.
    with pytest.raises(InsufficientRearShadowEvidence):
        _plateau_core(np.full((h, w), 80.0, dtype=float))


def test_camera_metal_align_plateau_shadows_lifts_masked_cast_shadow_only():
    from phone_texture.camera_metal import _align_plateau_shadows

    h, w = 360, 500
    yy, xx = np.indices((h, w))
    # Masked flat plateau excluding one circular camera lens at (120, 140), outer r=55.
    dist = np.hypot(xx - 120.0, yy - 140.0) - 55.0
    flat = (yy >= 20) & (yy < 330) & (xx >= 20) & (xx < 470) & (dist > 2.0)
    raw = np.zeros((h, w, 3), dtype=float)
    raw[..., 0] = 32.0
    raw[..., 1] = 4.0
    raw[..., 2] = -16.0
    lab = raw.copy()
    lab[..., 0] = 36.0
    # Add a coherent cast shadow crescent to the right of the lens (dist 3..26 px).
    shadow = flat & (dist > 2.5) & (dist < 26.0) & (xx > 165) & (np.abs(yy - 140) < 42)
    raw[shadow, 0] -= 9.0
    lab[shadow, 0] -= 8.0
    lab[shadow, 2] += 4.5  # Shadow desaturated/grayish shift.

    before_open = lab[flat & (xx > 320)].copy()
    before_lens = lab[~flat].copy()
    rep = _align_plateau_shadows(raw, lab, flat, [((120.0, 140.0, 48.0), dist, None, None)], None, scale=0.5)
    assert rep["status"] == "applied" and rep["pixels"] > 200
    # Shadow crescent aligned toward normal plateau lightness and blue chroma.
    assert float(np.median(lab[shadow, 0])) > 33.5
    assert float(np.median(lab[shadow, 2])) < -13.5
    # Open unshaded plateau and masked lens interior stay untouched.
    np.testing.assert_allclose(lab[flat & (xx > 320)], before_open, atol=0.1)
    np.testing.assert_array_equal(lab[~flat], before_lens)


def test_continuous_gamut_limits_robust_quantile_eliminates_grain_craters():
    from phone_texture.lighting import engine

    alum = engine("aluminum")
    h, w = 120, 160
    yy, xx = np.indices((h, w))
    mask = np.ones((h, w), dtype=bool)
    # Shadowed Cosmic Orange surface (L=62, a=35, b=33) with a few isolated 2x2
    # bright sandblasted aluminum glints (L=74) + a broad bright highlight block on the right.
    source = np.zeros((h, w, 3), dtype=float)
    source[:] = [62.0, 35.0, 33.0]
    for gy, gx in ((30, 35), (60, 55), (85, 40)):
        source[gy:gy + 2, gx:gx + 2, 0] = 74.0
    source[:, 110:, 0] = 74.0  # Genuine broad saturated highlight band.
    desired = source.copy()
    desired[..., 0] += 10.0
    safe, _ = alum.preserve_chroma_in_gamut(desired, source)

    legacy, _ = alum.continuous_gamut_limits(desired, safe, source, mask, 0.6)
    robust, proof = alum.continuous_gamut_limits(
        desired, safe, source, mask, 0.6, seed_quantile=15, seed_window=9
    )
    assert proof["method"] == "robust_quantile_gamut_cap_propagation"
    # Legacy punches a deep dark crater around the isolated glint at (60, 55).
    assert float(legacy[60, 62, 0] - source[60, 62, 0]) < 6.0
    # Robust quantile prevents isolated glints from cratering surrounding metal!
    assert float(robust[60, 62, 0] - source[60, 62, 0]) > 9.5
    # Genuine broad highlight band on the right still propagates smoothly!
    assert float(robust[60, 108, 0] - source[60, 108, 0]) < 4.5
    # Every pixel remains inside its own exact sRGB safe bound.
    assert np.all(robust[..., 0] <= safe[..., 0] + 1e-6)


def test_top_frame_isolated_white_ding_does_not_create_dark_diamond_halo(tmp_path):
    from phone_texture import auto_material, lighting

    h, w = 80, 600
    # Shaded Cosmic Orange top frame with a brighter right-side reference and one
    # isolated 3x3 white scuff/ding at (y=36..39, x=240..243).
    rgb = np.full((h, w, 3), [195, 118, 76], dtype=np.uint8)
    rgb[:, 420:540] = [232, 146, 102]
    rgb[36:39, 240:243] = [252, 250, 248]
    src = np.full((h, w, 4), 255, dtype=np.uint8)
    src[..., :3] = rgb
    path = tmp_path / "top.png"
    Image.fromarray(src).save(path)

    job, _, _, _ = auto_material.propose_job(path, "top", tmp_path / "materials")
    item = job["views"][0]
    item["domains"][0]["reference_boxes"] = [[460, 25, 467, 32]]
    eng = lighting.engine("aluminum")
    target, _ = eng.target_color(job)
    out, _, _, proof = eng.process_view(
        item, target, dict(job["defaults"], color_mode=job["color_mode"])
    )
    sL = eng.rgb_to_lab(src[..., :3])[..., 0]
    oL = eng.rgb_to_lab(out[..., :3])[..., 0]
    dL = oL - sL
    # Surrounding metal 8 px away from the 3x3 white ding must keep full lift
    # (no 40 px dark diamond halo pulled down by 0.2 L/px propagation!).
    far_lift = float(np.median(dL[25:55, 160:190]))
    near_lift = float(np.median(dL[34:41, 249:254]))
    assert far_lift - near_lift < 1.0
    assert proof["gamut_continuity"][0]["seed_quantile"] == 15.0
