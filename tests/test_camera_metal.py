"""Cross-layout regressions for the source-bound back lighting stage."""

import numpy as np
import pytest

from phone_texture import camera_metal
from phone_texture.lighting import engine


def test_shared_field_includes_dark_mark_and_preserves_photographed_contrast():
    yy, xx = np.ogrid[:80, :100]
    dark = np.hypot(xx - 49, yy - 40) < 6
    mask = (xx > 10) & (xx < 90) & (yy > 10) & (yy < 70)
    source = np.zeros((80, 100, 3))
    source[..., 0] = 44 - dark * 8
    baseline = source.copy()
    baseline[..., 0] += 6
    fixed = baseline.copy()
    from phone_texture.camera_shared_field import apply

    def transfer(im):
        return apply(source[..., 0], im[..., 0], mask & ~dark, mask.astype(float), mask, 0.3)

    assert transfer(fixed) == 0
    assert np.array_equal(fixed, baseline)
    fixed[..., 0] -= dark * 7
    assert transfer(fixed) > 0
    assert abs(np.median(fixed[..., 0][dark] - baseline[..., 0][dark])) < 2
    assert np.array_equal(fixed[~mask], baseline[~mask])


@pytest.mark.parametrize("layout", ["dual", "triple"])
def test_flat_camera_and_glass_do_not_acquire_any_line(layout):
    source = np.zeros((410, 300, 4), np.uint8)
    source[:] = [103, 117, 148, 255]
    centres = [(66, 53), (66, 131)]
    if layout == "triple":
        centres.append((134, 92))
    yy, xx = np.ogrid[:410, :300]
    lenses = []
    for cx, cy in centres:
        r = 30.0
        source[np.hypot(xx - cx, yy - cy) < r] = [12, 12, 12, 255]
        lenses.append(
            dict(centre=[cx, cy, r], outer=[r + 5] * 360, inner=[r] * 360, supported=[True] * 360)
        )
    plan = dict(layout=layout, location="fixture", lenses=lenses)
    mask = camera_metal.dual_scope((410, 300), plan) if layout == "dual" else source[..., 0] > 20
    lab = engine("aluminum").rgb_to_lab(source[..., :3])
    result, active, report = camera_metal.apply(source, lab, lab, mask, plan)
    assert np.max(abs(result - lab[: len(result)])) < 1e-8
    assert not active[~mask[: len(active)]].any()
    assert all(v["status"] == "clear" for v in report["secondary_arc_guard"])
    assert report["flash_exterior_pixels"] == 0


def test_shared_field_recovers_both_sides_of_joint_from_photo():
    from phone_texture.camera_shared_field import apply

    yy, xx = np.ogrid[:150, :190]
    joint = (yy >= 72) & (yy <= 78)
    raw = np.broadcast_to(np.where(joint, 17.0, 44.0), (150, 190)).copy()
    glint = (xx - 90) ** 2 + (yy - 69) ** 2 < 5
    raw[glint] += 10
    field = 6 + 0.012 * xx
    normal = raw + field
    scope = np.hypot((xx - 92) / 50, (yy - 76) / 34) < 1
    bad = normal.copy()
    dark = np.exp(-(((xx - 92) / 10) ** 2) - ((yy - 75) / 20) ** 2)
    bad -= 15 * dark
    bad[glint] = normal[glint] + 5
    old = bad.copy()
    quiet = (abs(yy - 76) > 12) & ~scope
    edited = apply(raw, bad, quiet, scope.astype(float), scope, 0.65)
    assert edited > 150
    assert np.array_equal(bad[~scope], old[~scope])
    assert np.max(abs((bad - normal)[dark > 0.85])) < 0.7
    assert np.max(abs((bad - normal)[glint])) < 0.7


def test_saturated_metal_receives_largest_supported_part_of_correction():
    render = engine("aluminum")
    base = render.rgb_to_lab(np.array([[239,122,20],[139,146,161]],dtype=np.uint8))
    proposed = base + [[25,9,12],[2,0,0]]
    result,limited = render.bounded_lab_offset(base,proposed)
    assert limited==1
    assert base[0,0]<result[0,0]<proposed[0,0]
    assert np.array_equal(result[1],proposed[1])
    rgb=render.lab_to_linear_rgb(result)
    assert rgb.min()>=-1e-7 and rgb.max()<=1+1e-7
