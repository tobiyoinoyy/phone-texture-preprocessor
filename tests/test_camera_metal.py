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


def test_lens_cast_shadow_follows_adjacent_metal_and_preserves_the_joint():
    from phone_texture.lens_exterior import lift

    y, x = np.ogrid[:320, :320]
    distance = np.hypot(x-160, y-160) - 100
    angle = np.arctan2(y-160, x-160)
    outside = distance > 2
    normal = 48 + 0.006 * x + 0.006 * y
    texture = 0.5 * np.cos(y*2) * np.sin(x)
    shade = 18 * np.exp(-(angle/0.38)**2) * np.exp(-((distance-7)/5.5)**4)
    light = normal + texture - shade
    light[~outside] = 18
    lab = np.stack([light, 1+0.13*shade, -5-0.18*shade], axis=-1)
    raw = lab.copy()
    lab[..., 0] += 4
    before = lab.copy()
    edge = np.full(360, 100.)
    ok = np.ones(360, bool)
    lens = dict(centre=[160, 160, 90], outer=edge, supported=ok)
    proof = lift(raw, lab, outside, lens, edge, ok)
    assert proof["pixels"] > 100
    dark = (distance > 5) & (distance < 9) & (abs(angle) < 0.14)
    assert np.max(abs(lab[..., 0][dark] - normal[dark] - texture[dark] - 4)) < 2.5
    assert np.max(abs(lab[..., 1][dark] - 1)) < 0.5
    np.testing.assert_array_equal(before[~outside | (distance > 26)], lab[~outside | (distance > 26)])

    # A partial near-metal strip cannot borrow reference pixels from across a bevel.
    without_reference = outside & (distance < 15)
    unchanged = before.copy()
    assert lift(raw, unchanged, without_reference, lens, edge, ok)["pixels"] == 0
    np.testing.assert_array_equal(unchanged, before)


def test_gray_fallback_verifies_and_excludes_each_glass_independently(monkeypatch):
    from phone_texture import back_lens, back_geometry

    seeds = np.array([[50., 50., 25.], [50., 140., 25.], [130., 95., 25.]])
    detections = iter([(None, 6), (seeds, 8)])
    monkeypatch.setattr(back_lens, '_disks', lambda image: next(detections))

    def edge(image, trial, *, search_width):
        x, y, r = trial[0]
        if y == 140 and r < 25:
            return np.array([[x, y, 25.01 if search_width == 6 else 25.]])
        if x == 50 and y == 50:
            return np.array(trial)
        raise ValueError('ambiguous arc')

    monkeypatch.setattr(back_geometry, 'measure_glass_edges', edge)
    image = np.zeros((600, 350, 4), np.uint8)
    plan = camera_metal.measure(image, 'triple')
    assert plan['slots'] == [0, 1]
    assert [v['slot'] for v in plan['exclusions']] == [2]
    assert abs(plan['lenses'][1]['centre'][2] - 25.) < .01
    x0, y0, x1, y1 = plan['exclusions'][0]['box']
    assert x0 < 105 < 155 < x1 and y0 < 70 < 120 < y1


def test_independent_seeds_require_three_distinct_thresholds(monkeypatch):
    from phone_texture import back_lens

    # A shifting right-hand blob and two competing candidates on the lower
    # left cannot veto a stable upper-left disk or become lighting anchors.
    readings = [[(195., 182., 90.), (197., 440., 90.), (202., 441., 90.),
                 (441. + 5*i, 318., 92.)] for i in range(4)]
    monkeypatch.setattr(back_lens, '_disk_readings', lambda img: iter(readings))
    seeds, _ = camera_metal._independent_seeds(np.zeros((2080, 1000, 4), np.uint8))
    assert np.array_equal(seeds[0], [195., 182., 90.])
    assert seeds[1:] == [None, None]
