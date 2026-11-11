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
