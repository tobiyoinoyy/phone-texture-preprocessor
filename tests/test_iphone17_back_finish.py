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
    assert np.array_equal(np.asarray(image)[700:-round(.059*image.width)], np.asarray(fixed)[700:-round(.059*image.width)])


def test_discontinuous_top_is_rejected():
    image = Image.new('RGBA', (988,2058), 'white')
    ImageDraw.Draw(image).rectangle((240,0,660,28), fill=(0,0,0,0))
    with pytest.raises(ValueError,match='rail'):
        level_top(image)


def test_back_edge_normalization_removes_empty_rows_without_moving_the_lenses():
    from phone_texture.back_geometry import normalize_back_edges

    image = Image.new('RGBA', (988,2058))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((1,4,986,2054),radius=70,fill='#d6d6d6')
    draw.ellipse((90,80,290,280),fill='#181818')
    before = np.asarray(image).copy()
    normalized, proof = normalize_back_edges(image)
    after = np.asarray(normalized)
    assert normalized.size == image.size
    assert after[0,:,3].any() and after[-1,:,3].any()
    np.testing.assert_array_equal(before[60:-60],after[60:-60])
    assert proof['central_pixels_unchanged']
    assert abs(top_measure(after[...,3])['left_minus_right_px'])<=.5


def test_soft_photographed_edge_retains_antialiasing_without_unequal_headroom():
    from phone_texture.back_geometry import normalize_back_edges

    a = np.full((2058,988,4),255,np.uint8)
    a[..., :3] = [130,150,170]
    a[:8,:494,3] = np.array([0,31,64,104,159,213,242,249])[:,None]
    a[:4,494:,3] = np.array([0,0,0,128])[:,None]
    image = Image.fromarray(a)
    fixed, proof = normalize_back_edges(image)
    actual = np.asarray(fixed)
    assert proof['top_target_edge_px'] < 1
    assert actual[0,:,3].any() and actual[-1,:,3].any()
    assert abs(proof['top']['left_minus_right_px']) <= .5
    assert proof['top']['central_span_p95_px'] <= 1.1
    np.testing.assert_array_equal(a[60:-60],actual[60:-60])
