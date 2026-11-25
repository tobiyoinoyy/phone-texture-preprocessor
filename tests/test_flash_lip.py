import numpy as np
import pytest

from phone_texture.flash_lip import correct


@pytest.mark.parametrize("width", [650, 1000])
def test_recover_flash_lip_from_photographed_metal(width):
    h = width * 2
    y, x = np.ogrid[:h // 5, :width]
    distance = np.hypot(x - 0.81 * width, y - 0.08 * h)
    radius = 0.045 * width
    scale = radius / 46
    offset = (distance - radius) / scale
    raw = 40 + .4 * np.sin(x / 3) - 4 * np.exp(-((offset - 6) / 2) ** 2)
    raw[offset < 0] = 85
    source = np.zeros((h, width, 4), dtype=np.uint8)
    source[:len(raw), :, :3] = (raw[..., None] * 2.5).astype(np.uint8)
    source[..., 3] = 255
    metal = offset > 1
    uniform = raw + 6
    current = uniform - 11 * np.exp(-((offset - 7) / 3) ** 2)
    before = current.copy()
    proof = correct(source, raw, current, metal)
    assert proof["status"] == "applied"
    near = (offset > 6) & (offset < 9)
    assert np.max(abs(current[near] - uniform[near])) < 0.3
    locked = (offset <= 1) | (offset > 24)
    np.testing.assert_array_equal(current[locked], before[locked])

    clean = uniform.copy()
    assert correct(source, raw, clean, metal)["changed_pixels"] == 0
    np.testing.assert_array_equal(clean, uniform)
    sparse = metal & (x < 0.81 * width)
    unchanged = before.copy()
    assert correct(source, raw, unchanged, sparse)["status"] == "insufficient_reference_metal"
    np.testing.assert_array_equal(unchanged, before)


def test_normal_back_lighting_corrects_flash_even_with_no_locatable_lenses(tmp_path):
    from PIL import Image
    from phone_texture.lighting import engine

    height, width = 1000, 500
    yy, xx = np.ogrid[:height, :width]
    radius = np.hypot(xx - 405, yy - 80)
    rgba = np.full((height, width, 4), 255, dtype=np.uint8)
    rgba[..., :3] = np.where(radius[..., None] < 22, 220, 90)
    photo = tmp_path / 'back.png'
    region = tmp_path / 'surround.png'
    Image.fromarray(rgba).save(photo)
    Image.new('L', (width, height), 255).save(region)
    render = engine('aluminum')
    view = dict(source=str(photo), source_sha256=render.digest(photo), name='back',
                domains=[dict(name='camera_surround', mask=str(region), sigma=[5, 5])])
    target = np.array([50, 0, 0])
    defaults = dict(color_mode='source_lightness')
    baseline = render.process_view(view, target, defaults)[0]
    fixed, _, _, report = render.process_view(dict(view, camera_metal_layout='triple'), target, defaults)
    assert report['camera_metal']['flash_lip']['status'] == 'applied'
    near = (radius > 24.5) & (radius < 26.5)
    far = (radius > 37) & (radius < 40)
    def contrast(image):
        return np.mean(image[..., 0][far]) - np.mean(image[..., 0][near])
    assert contrast(baseline) > 4
    assert abs(contrast(fixed)) < abs(contrast(baseline)) / 3
    np.testing.assert_array_equal(fixed[radius < 22], baseline[radius < 22])
