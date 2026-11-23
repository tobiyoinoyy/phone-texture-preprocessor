"""Regression cases for empty optical samples and six-column download receipts."""
import hashlib

import numpy as np
import pytest

from phone_texture.auto_structure import windows
from phone_texture.face_perspective import _profile_peak
from phone_texture.web_app import FACES, sheet_column_directions


@pytest.mark.parametrize('pick', ['outer_low', 'outer_high'])
def test_positive_subthreshold_profile_has_no_optical_peak(pick):
    assert _profile_peak(np.array([.1, .3, .5]), 0, 3, pick) is None
    assert _profile_peak(np.array([.1, 1.5, .1]), 0, 3, pick) == 1


def test_integer_midpoint_is_owned_by_one_search_window():
    upper, lower = windows([24, 30], 90)
    assert upper[1] == lower[0] == 27


def test_column_receipt_accepts_unresolved_orientation_and_rejects_conflicts():
    hashes = {f: hashlib.sha256(f.encode()).hexdigest() for f in FACES}
    images = [dict(name=f + '.png', face=f, sha256=v) for f, v in hashes.items()]
    side = next(i for i in images if i['face'] == 'right')
    side.update(face=None, reason='疑似镜像或侧面方向不确定', score=.7,
                side_scores={'right': .72}, mirror_score=.65)
    assert sheet_column_directions(images, hashes)
    side['mirror_score'] = .75
    assert not sheet_column_directions(images, hashes)
    side.update(face='left')
    assert not sheet_column_directions(images, hashes)
    side.update(face=None, reason='方向识别不确定', score=0)
    assert not sheet_column_directions(images, hashes)
    side.update(face='right')
    assert not sheet_column_directions(images, dict(hashes, right=hashes['left']))
    assert not sheet_column_directions(images[:-1], hashes)


def test_filled_camera_control_requires_measurable_round_ends():
    from PIL import Image, ImageDraw
    from phone_texture.side_control_material import filled_control

    rail = Image.new('RGBA', (100, 1200), (160, 130, 120, 255))
    expected = (30, 640, 70, 860)
    button = rail.copy()
    ImageDraw.Draw(button).rounded_rectangle(expected, radius=20, fill=(60, 52, 49, 255))
    controls = filled_control(button, 450)
    assert controls
    for _, proof in controls:
        assert max(abs(np.array(proof['box']) - [30, 640, 71, 861])) <= 2
    ImageDraw.Draw(rail).rectangle(expected, fill=(60, 52, 49, 255))
    assert not filled_control(rail, 450)


def test_lighting_retry_only_handles_measured_metal_steps(tmp_path):
    from phone_texture.stage_processing import _continuity_candidate
    from phone_texture.shadow_lift_profile import NAME
    with pytest.raises(ValueError, match='source identity changed'):
        _continuity_candidate(ValueError('source identity changed'), {}, tmp_path/'job.json',
                              'left', None, {}, None, None, NAME)
    with pytest.raises(ValueError, match='correction discontinuity'):
        _continuity_candidate(ValueError('Lighting correction discontinuity at rear_button_left_0: 1.31 L'),
                              {}, tmp_path/'job.json', 'back', None, {}, None, None, 'baseline')
