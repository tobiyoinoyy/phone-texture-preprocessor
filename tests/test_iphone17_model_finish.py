import numpy as np
from PIL import Image, ImageDraw
import pytest

from phone_texture.iphone17_model_finish import register_camera, PROFILE
from phone_texture import workspace as w


def camera_back(displacements):
    model=w.read(PROFILE)['camera']
    im=Image.new('RGBA',tuple(model['reference_size']),(193,200,214,255))
    draw=ImageDraw.Draw(im)
    for (x,y),r,(dx,dy) in zip(model['centers_px'],model['radii_px'],displacements):
        x,y=x+dx,y+dy
        draw.ellipse((x-r-8,y-r-8,x+r+8,y+r+8),fill='#999999')
        draw.ellipse((x-r,y-r,x+r,y+r),fill='#171717')
    draw.ellipse((395,260,446,311),fill='#ffffff')
    return im


def test_standard_lenses_remeasure_after_warp_and_keep_flash_and_body():
    src=camera_back([(-6,9),(-4,12)])
    image,proof=register_camera(src)
    assert max(proof['model_reference_error_px'])<=.65
    a,b=np.asarray(src),np.asarray(image)
    assert np.array_equal(a[...,3],b[...,3]) and np.array_equal(a[:,370:],b[:,370:])
    assert np.array_equal(a[650:],b[650:]) and np.array_equal(a[:,:15],b[:,:15])


def test_single_lens_cannot_be_registered_as_standard_pair():
    with pytest.raises(ValueError,match='Two separate'):
        register_camera(camera_back([(0,0)]))


def test_profile_has_all_four_corresponding_material_boundaries():
    profile=w.read(PROFILE)
    assert {frozenset(c['names']) for c in profile['chains']}=={
        frozenset((a,b)) for a in ('top','bottom') for b in ('left','right')}
    assert all(c['arc'][-1]>.28 and np.all(np.diff(c['arc'])>0) for c in profile['chains'])
