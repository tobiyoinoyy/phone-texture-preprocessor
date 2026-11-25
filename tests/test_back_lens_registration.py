"""Lens registration on the standard back canvas (2026-11-10 user decision)."""
import numpy as np
import pytest
from PIL import Image
from phone_texture import back_lens as bl
from phone_texture.back_canvas import STANDARD_BACK_SIZE


def synthetic_back(model, shift=(0., 0.), body=(90, 100, 180)):
    w, h = STANDARD_BACK_SIZE[model]
    a = np.zeros((h, w, 4), np.uint8); a[..., :3] = body; a[20:h-20, 20:w-20, 3] = 255
    yy, xx = np.mgrid[:h, :w]
    spec = bl.STANDARD[model]
    for cx, cy in np.asarray(spec['lenses']) * [w, h] + shift:
        ring = (xx-cx)**2 + (yy-cy)**2
        a[ring <= 110**2, :3] = (170, 175, 190)   # bezel ring
        a[ring <= 98**2, :3] = (12, 12, 14)       # lens glass
    fx, fy = np.asarray(spec['flash']) * [w, h]
    a[(xx-fx)**2 + (yy-fy)**2 <= 40**2, :3] = (230, 220, 200)
    return a


@pytest.mark.parametrize('model', ['iphone17pro', 'iphone17promax'])
def test_shifted_lenses_are_registered_and_rest_is_untouched(model):
    a = synthetic_back(model, shift=(-3.0, 2.5))
    out, proof = bl.register(a, model)
    assert proof['applied'] and proof['residual_px'] <= 1
    np.testing.assert_array_equal(out[..., 3], a[..., 3])
    h = a.shape[0]
    np.testing.assert_array_equal(out[int(.35*h):], a[int(.35*h):])


def test_aligned_back_is_left_byte_identical():
    a = synthetic_back('iphone17pro')
    out, proof = bl.register(a, 'iphone17pro')
    assert not proof['applied'] and proof['reason'] == 'already_aligned'
    assert out is a


def test_large_shift_is_not_guessed():
    a = synthetic_back('iphone17pro', shift=(0., 12.))
    out, proof = bl.register(a, 'iphone17pro')
    assert not proof['applied'] and proof['reason'] == 'shift_exceeds_limit'
    assert out is a


def test_missing_lenses_keep_the_image():
    a = synthetic_back('iphone17pro'); a[..., :3] = (90, 100, 180)
    out, proof = bl.register(a, 'iphone17pro')
    assert proof['reason'] == 'lenses_not_measured' and out is a


@pytest.mark.parametrize('model', ['iphone17pro', 'iphone17promax'])
def test_delivered_model_specific_back_has_a_level_rail_and_three_aligned_disks(tmp_path, model):
    from phone_texture.back_canvas import standardize
    from phone_texture.back_inspection import inspect_back
    source = synthetic_back(model, shift=(-2., 3.))
    h, width = source.shape[:2]
    y, x = np.ogrid[:h, :width]
    source[..., 3] = np.where(y >= 2 + np.rint(5*x/width), 255, 0)
    raw = tmp_path/'source.png'
    result = tmp_path/'back.png'
    Image.fromarray(source).save(raw)
    report = standardize(raw, result, model)
    check = inspect_back(result, model)
    assert report['back_ready_for_backfill'] and check['verified']
    assert len(check['distances_px']) == 3 and max(check['distances_px']) <= .30
    assert abs(report['top_rail']['before']['left_minus_right_px']) > 2
    assert abs(check['rail']['left_minus_right_px']) <= .5
    assert check['empty_canvas_rows'] == dict(top=0, bottom=0)
    assert report['canvas_edge_normalization']['verified']
    fx, fy = np.rint(np.array(bl.STANDARD[model]['flash'])*[width,h]).astype(int)
    actual = np.asarray(Image.open(result))
    np.testing.assert_array_equal(actual[fy-45:fy+45,fx-45:fx+45], source[fy-45:fy+45,fx-45:fx+45])


@pytest.mark.parametrize('model', ['iphone17pro', 'iphone17promax'])
def test_legacy_empty_outer_rows_fail_independent_back_acceptance(tmp_path, model):
    from phone_texture.back_inspection import inspect_back
    source = synthetic_back(model)
    source[..., 3] = 255
    source[:4, :, 3] = 0
    source[-2:, :, 3] = 0
    path = tmp_path/'legacy-back.png'
    Image.fromarray(source).save(path)
    check = inspect_back(path, model)
    assert not check['verified']
    assert check['empty_canvas_rows'] == dict(top=4, bottom=2)


def test_large_lens_shift_keeps_the_photographed_plateau_ledge_and_corner():
    from PIL import ImageDraw

    model = 'iphone17promax'
    a = synthetic_back(model, shift=(-1., -7.))
    h, w = a.shape[:2]
    # Two separate edges on the raised step, including the curved-to-straight transition.
    mask = Image.new('L', (w, h))
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle((68, 56, 1011, 589), radius=130, fill=255)
    core = np.asarray(mask) != 0
    body = np.all(a[..., :3] == [90, 100, 180], axis=-1)
    a[core & body, :3] = [190, 135, 83]
    outline = Image.new('L', (w, h))
    ImageDraw.Draw(outline).rounded_rectangle((68, 56, 1011, 589), radius=130, outline=255, width=3)
    edge = np.asarray(outline) != 0
    a[edge, :3] = [72, 48, 30]
    out, proof = bl.register(a, model)
    assert proof['verified'] and proof['applied'] and proof['residual_px'] <= bl.MIN_SHIFT_PX
    assert proof['plateau_boundary'] == 'source_measured'
    np.testing.assert_array_equal(out[edge | ~core], a[edge | ~core])


@pytest.mark.parametrize('retry_status', ['verified', 'radius_changed', 'position_unresolved'])
def test_one_unstable_glass_fit_retries_only_its_source_pixels(monkeypatch, retry_status):
    model = 'iphone17promax'
    photo = synthetic_back(model, shift=(-1.6, 1.8))
    expected, first_proof = bl.register(photo, model)
    assert first_proof['verified']
    measure = bl.glass_centres
    calls = []
    def unstable_lower_fit(image, model):
        disks, count = measure(image, model)
        calls.append(image.shape)
        if len(calls) == 2 or (len(calls) == 3 and retry_status == 'position_unresolved'):
            disks[1, 1] += .95
        if len(calls) == 3 and retry_status == 'radius_changed':
            disks[1, 2] += 1.
        return disks, count
    monkeypatch.setattr(bl, 'glass_centres', unstable_lower_fit)
    result, proof = bl.register(photo, model)
    assert len(calls) == 3
    assert proof['linear_retry_lenses'] == [1]
    if retry_status != 'verified':
        assert not proof['verified'] and result is photo
        return
    assert max(proof['per_lens_error_px']) <= bl.MIN_SHIFT_PX
    assert max(proof['retry_radius_drift_px']) <= bl.MIN_SHIFT_PX
    h, w = photo.shape[:2]
    cx, cy = np.array(bl.STANDARD[model]['lenses'][1]) * [w, h]
    y, x = np.ogrid[:h, :w]
    outside = np.hypot(x - cx, y - cy) > 1.31 * 99
    np.testing.assert_array_equal(result[outside], expected[outside])
    np.testing.assert_array_equal(result[..., 3], photo[..., 3])
    assert np.any(result != expected)
