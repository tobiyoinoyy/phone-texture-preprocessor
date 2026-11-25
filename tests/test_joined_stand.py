import numpy as np
import pytest
from PIL import Image

from phone_texture.joined_stand import cut_joined_stand


def joined_fixture():
    h, w = 730, 400
    x = np.arange(w)
    width = 280
    left = (x-60)/width
    right = (340-x)/width
    rail = 635+.025*(x-200)
    edge = rail - sum(5*width*np.maximum(.20-t, 0)**2.5 for t in (left, right))
    y = np.arange(h)[:, None]
    body = (x >= 60) & (x < 340) & (y < edge) & (y > 50)
    stand = (x > 109) & (x < 293) & (y > 617)
    raw = np.full((h, w, 3), 225, np.uint8)
    raw[stand] = 180
    raw[body] = [123, 60, 37]
    mask = body | stand
    return raw, body, mask, stand


def recover(raw, mask, paper=None):
    stages = {}
    matte = Image.fromarray(np.dstack((raw, mask.astype('uint8')*255)))
    cut_joined_stand(matte, mask, raw, paper, 60, 340, 630,
                     dict(rotation_ccw_degrees=0), stages.__setitem__)
    return np.asarray(stages['after_support'])


def test_source_supported_cut_keeps_body_and_round_ends():
    raw, body, mask, stand = joined_fixture()
    result = recover(raw, mask)
    np.testing.assert_array_equal(result[..., :3], raw)
    assert not np.any(result[..., 3][stand & (np.arange(len(raw))[:, None] > 644)])
    assert np.mean(result[..., 3][body] > 0) > .999
    np.testing.assert_array_equal(result[:, :105, 3], 255*body[:, :105])
    np.testing.assert_array_equal(result[:, 300:, 3], 255*body[:, 300:])


def test_neutral_body_or_unwitnessed_corner_is_not_cut():
    raw, body, mask, _ = joined_fixture()
    with pytest.raises(ValueError):
        recover(np.repeat(raw.mean(2).astype('uint8')[:, :, None], 3, axis=2), mask)
    mask[625:, 60:140] = True
    with pytest.raises(ValueError):
        recover(raw, mask)


def test_print_extending_past_rail_vetoes_cut_even_when_paper_seed_stops_early():
    raw, body, mask, _ = joined_fixture()
    paper = np.zeros_like(body)
    paper[550:631, 180:198] = True
    raw[550:650, 180:198] = 220
    raw[550:650, 186:189] = 45
    with pytest.raises(ValueError, match='label end'):
        recover(raw, mask, paper)
