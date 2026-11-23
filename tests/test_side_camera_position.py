import numpy as np
import pytest
from pathlib import Path
from PIL import Image,ImageDraw
from unittest.mock import patch

from phone_texture.side_camera_position import classify
from phone_texture import view_recognition as recognition


def side(camera='left'):
    image=Image.new('RGB',(800,800),(230,230,230))
    draw=ImageDraw.Draw(image)
    draw.rounded_rectangle((65,385,735,425),radius=8,fill=(100,105,115))
    if camera in ('left','both'):
        draw.rectangle((110,365,255,390),fill=(70,75,85))
    if camera in ('right','both'):
        draw.rectangle((545,365,690,390),fill=(70,75,85))
    return image


@pytest.mark.parametrize('end,face',[('left','right'),('right','left')])
def test_camera_end_classifies_sides_without_complete_buttons(end,face):
    proof=classify(np.asarray(side(end)))
    assert proof['face']==face
    assert len(proof['readings'])==3


@pytest.mark.parametrize('camera',['none','both'])
def test_missing_or_two_ended_projection_has_no_unique_direction(camera):
    assert classify(np.asarray(side(camera)))['face'] is None


def test_camera_direction_resolves_local_hardware_template_conflict(tmp_path):
    photo=tmp_path/'photo.jpg'
    side('left').save(photo)
    atlas=dict(face=None,suggested_face='right',scores=dict(left=.83,right=.91),
               reason='疑似镜像或侧面方向不确定')
    with patch.object(recognition,'_recognize_atlas',return_value=atlas):
        with patch('phone_texture.source_side_recognition.recognize_source_side') as hardware:
            result=recognition.recognize(photo)
    assert result['face']=='right'
    assert result['method']=='fixed_studio_pose_camera_end'
    hardware.assert_not_called()
