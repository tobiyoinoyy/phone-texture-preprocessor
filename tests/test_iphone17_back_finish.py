import numpy as np
import pytest
from PIL import Image, ImageDraw
from phone_texture.iphone17_back_finish import level_top, top_measure, finish_back, glass_edges
from phone_texture.iphone17_model_finish import PROFILE
from phone_texture import workspace as w


def test_corrects_both_lenses_and_slanted_top_on_the_final_canvas():
    camera = w.read(PROFILE)['camera']
    target = np.array(camera['centers_px'])
    image = Image.new('RGBA', tuple(camera['reference_size']))
    draw = ImageDraw.Draw(image)
    draw.polygon([(0,8),(987,1),(987,2057),(0,2057)], fill='#d6d6d6')
    for (x,y),radius in zip(target + [[4,-3],[-2,6]], camera['radii_px']):
        for pad,color in [(8,'#bbbbbb'),(0,'#181818')]:
            r = radius + pad
            draw.ellipse((x-r,y-r,x+r,y+r), fill=color)
    fixed, report = finish_back(image)
    measured = glass_edges(np.array(fixed))[:, :2]
    assert tuple(fixed.size) == tuple(camera['reference_size'])
    assert max(np.linalg.norm(measured-target, axis=1)) < .3
    assert abs(report['top_before']['left_minus_right_px']) > 3
    assert abs(top_measure(np.array(fixed)[...,3])['left_minus_right_px']) <= .5
    assert np.array_equal(np.asarray(image)[700:], np.asarray(fixed)[700:])


def test_discontinuous_top_is_rejected():
    image = Image.new('RGBA', (988,2058), 'white')
    ImageDraw.Draw(image).rectangle((240,0,660,28), fill=(0,0,0,0))
    with pytest.raises(ValueError,match='rail'):
        level_top(image)
