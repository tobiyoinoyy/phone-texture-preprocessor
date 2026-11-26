"""Targeted regression tests for Batch 1 full-rerun QA fixes."""
import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw
from scipy import ndimage


def test_iphone17_rear_antennas_rejects_asymmetric_scuff_and_recovers_paired_stripes():
    from phone_texture.iphone17_structure import rear_antennas

    im = Image.new("RGBA", (988, 2058), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((8, 1, 979, 2056), radius=96, fill="#383838")
    d.rounded_rectangle((22, 14, 965, 2043), radius=90, fill="#dcdcdc")
    # Real symmetric plastic breaks on both rims at [212..226] and [1831..1845].
    for x0, x1 in ((8, 19), (968, 979)):
        for y0, y1 in ((212, 226), (1831, 1845)):
            d.rectangle((x0, y0, x1, y1), fill="#888888")
    # Brighter narrow scuff ONLY on the outer 3px of right-image edge (phone left)
    # at asymmetric y=1735..1743: tricks 3-track _rear_antennas, while paired
    # detectors find the true symmetric [1831..1845] stripe.
    d.rectangle((975, 1735, 979, 1743), fill="#eeeeee")

    actual = rear_antennas(im)
    for side in ("left", "right"):
        rows = actual[side]["rows"]
        assert abs(rows[0] - 211.5) < 1.5 and abs(rows[1] - 226.5) < 1.5
        assert abs(rows[2] - 1830.5) < 1.5 and abs(rows[3] - 1845.5) < 1.5

    # Extreme unmatched asymmetry with NO valid symmetric pair must fail closed.
    bad = Image.new("RGBA", (988, 2058), (0, 0, 0, 0))
    db = ImageDraw.Draw(bad)
    db.rounded_rectangle((8, 1, 979, 2056), radius=96, fill="#383838")
    db.rounded_rectangle((22, 14, 965, 2043), radius=90, fill="#dcdcdc")
    db.rectangle((968, 212, 979, 226), fill="#aaaaaa")
    db.rectangle((968, 1720, 979, 1728), fill="#aaaaaa")  # 105px off & 0.57x width
    with pytest.raises(ValueError, match="inconsistent|Rear rim antenna"):
        rear_antennas(bad, sides=("left",))


def test_rear_camera_glass_separates_dark_blue_plateau_merged_lenses():
    from phone_texture.auto_material import rear_camera_glass

    h, w = 1000, 500
    a = np.full((h, w, 4), 255, dtype=np.uint8)
    # Underexposed dark anodized blue back: plateau sits near 32 gray (below 35 floor!),
    # causing gray < 35 to flood the entire upper plateau into one oversized component.
    a[..., :3] = [24, 32, 46]
    a[:300, 20:480, :3] = [23, 31, 44]
    yy, xx = np.indices((h, w))
    lenses = [
        (xx - 105) ** 2 + (yy - 95) ** 2 <= 52 * 52,
        (xx - 230) ** 2 + (yy - 160) ** 2 <= 52 * 52,
        (xx - 105) ** 2 + (yy - 225) ** 2 <= 52 * 52,
    ]
    for disk in lenses:
        a[disk, :3] = [1, 2, 3]
    # Bottom-left lens has slightly brighter rim metal around it so it separates at 35,
    # while top-left and middle-right stay merged with the dark 31-gray plateau.
    bright_ring = ndimage.binary_dilation(lenses[2], iterations=10) & ~lenses[2]
    a[bright_ring, :3] = [38, 48, 66]
    lidar = (xx - 395) ** 2 + (yy - 220) ** 2 <= 22 * 22
    a[lidar, :3] = [0, 1, 2]
    # Thin dark cast-shadow arc must still be rejected.
    arc = ((xx - 395) ** 2 + (yy - 95) ** 2 <= 26 * 26) & ((xx - 395) ** 2 + (yy - 95) ** 2 >= 22 * 22)
    a[arc, :3] = [1, 1, 2]

    glass, proof = rear_camera_glass(a, a[..., 3] > 0)
    for disk in lenses:
        assert glass[disk].mean() > 0.95
    assert glass[lidar].mean() > 0.95
    assert not glass[arc].any()
    assert len(proof["dark_metal_recovered"]) >= 2


def test_blemished_fallback_rejects_truncated_dark_control_and_recovers_full_ccb():
    from phone_texture.side_material import hardware_materials, lighting_hardware

    h, w = 1400, 120
    a = np.full((h, w, 4), 255, dtype=np.uint8)
    a[..., :3] = [195, 130, 85]  # Orange aluminum rail.
    yy, xx = np.indices((h, w))

    def draw_capsule(arr, cx, y0, y1, r, color, rim=None):
        cap = (xx - cx) ** 2 + (yy - np.clip(yy, y0 + r, y1 - r)) ** 2 <= r * r
        arr[cap, :3] = color
        if rim is not None:
            ring = ndimage.binary_dilation(cap, iterations=2) & ~cap
            arr[ring, :3] = rim
        return cap

    # Power button (upper).
    draw_capsule(a, 60, 400, 560, 20, [190, 126, 82], rim=[85, 58, 38])
    # Full dark Camera Control Button (lower, 200px tall).
    ccb = draw_capsule(a, 60, 860, 1060, 22, [85, 65, 55])
    # Internal faint specular band + corner blemish dividing off a ~75px sub-box
    # inside the dark sapphire that would previously pass blemished_rim_fallback.
    a[910:913, 38:82, :3] = [150, 110, 80]
    a[985:988, 38:82, :3] = [150, 110, 80]
    a[910:920, 38:46, :3] = [145, 108, 78]

    comps, _ = lighting_hardware(Image.fromarray(a), "right")
    assert len(comps) == 2
    lower_mask, lower_proof = comps[1]
    assert lower_proof["box"][3] - lower_proof["box"][1] >= 170
    assert (lower_mask & ccb).sum() / ccb.sum() > 0.85
    # Strict hardware_materials must never return the truncated 75px fragment either.
    try:
        strict = hardware_materials(Image.fromarray(a), "right")
        assert strict[1][1]["box"][3] - strict[1][1]["box"][1] >= 170
    except ValueError:
        pass


def test_seeded_paper_rejects_distorted_qr_quad_and_recovers_square_qr(monkeypatch):
    from phone_texture.auto_material import seeded_paper

    h, w = 1000, 500
    a = np.full((h, w, 4), 255, dtype=np.uint8)
    a[..., :3] = [212, 215, 218]  # Pale silver glass.
    # Draw a white paper sticker with a square dark-ink QR block at x=190..238, y=850..898.
    a[780:995, 180:250, :3] = [245, 245, 244]
    rng = np.random.default_rng(0)
    qr = rng.integers(0, 2, size=(12, 12))
    for gy in range(12):
        for gx in range(12):
            if qr[gy, gx]:
                a[850 + gy * 4 : 854 + gy * 4, 190 + gx * 4 : 194 + gx * 4, :3] = 20

    # Simulate OpenCV QRCodeDetector returning a bogus 2:1 trapezoid on raw RGB/gray
    # (extending right into clean silver glass) while succeeding cleanly on binary input.
    class FakeQR:
        def detect(self, img):
            if img.ndim == 3 or (img.max() - img.min() < 250):
                # Bogus 2:1 trapezoid spilled right to x=310.
                y_off = int(h * 0.55) if img.ndim == 3 else int(h * 0.65)
                x_off = 0 if img.ndim == 3 else int(w * 0.2)
                pts = np.array(
                    [[[190 - x_off, 860 - y_off], [238 - x_off, 850 - y_off],
                      [310 - x_off, 898 - y_off], [220 - x_off, 898 - y_off]]],
                    dtype=np.float32,
                )
                return True, pts
            # Adaptive-thresholded binary image finds the true 48x48 square QR.
            y_off = int(h * 0.65)
            x_off = int(w * 0.2)
            pts = np.array(
                [[[190 - x_off, 850 - y_off], [238 - x_off, 850 - y_off],
                  [238 - x_off, 898 - y_off], [190 - x_off, 898 - y_off]]],
                dtype=np.float32,
            )
            return True, pts

    monkeypatch.setattr(cv2, "QRCodeDetector", FakeQR)
    mask, proof = seeded_paper(a, "back", resolve_end=False)
    assert mask is not None
    sb = proof["seed_box"]
    assert 0.8 <= (sb[2] - sb[0]) / max(1, sb[3] - sb[1]) <= 1.25
    assert sb[2] <= 242
    # Must NOT spill a sharp gray rectangle into clean silver glass to the right (x=275..310).
    assert mask[840:910, 275:310].mean() < 0.05


def test_front_geometry_recovers_secondary_hist_peak_and_trims_corner_wedge():
    from phone_texture.auto_geometry import _plate_geometry_strict

    h, w = 1000, 1000
    raw = np.full((h, w, 3), 230, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    # Dark phone front body at x=300..700, y=100..860 with rounded corners (r=56).
    im_body = Image.new("L", (w, h), 0)
    ImageDraw.Draw(im_body).rounded_rectangle((300, 100, 700, 860), radius=56, fill=255)
    body = np.asarray(im_body) > 0
    raw[body] = [25, 27, 32]
    # Bright white stand hanging down from y=860, with two wider flat white shelves at y=892
    # (making y=892 taller in bincount than the exposed phone gaps at y=860!) plus vertical
    # support columns across the clear-band regions reaching the bottom, and a corner wedge.
    stand = np.zeros((h, w), dtype=bool)
    stand[856:892, 345:405] = True
    stand[856:892, 595:665] = True  # Extends under the bottom-right rounded corner!
    stand[860:, 365:440] = True
    stand[860:, 560:635] = True
    # Notice stand RGB is pure white 226..230 (same as background -> weak RGB gradient at y=892).
    raw[stand & ~body] = 228
    alpha = np.where(body | stand, 255, 0).astype(np.uint8)
    matte = Image.fromarray(np.dstack([raw, alpha]))

    out, proof = _plate_geometry_strict(matte, "front", source_image=Image.fromarray(raw))
    assert proof["method"] == "local_stand_removal_from_exposed_plate_edges_v1"
    assert out.height <= 768  # True phone height is 761px; stand shelf (y=892) cleanly removed!
    out_a = np.asarray(out)
    # Bottom-right corner pixel outside the rounded bezel must be transparent (wedge trimmed).
    assert out_a[-3, -3, 3] == 0
    assert out_a[-10, -10, 3] == 0
