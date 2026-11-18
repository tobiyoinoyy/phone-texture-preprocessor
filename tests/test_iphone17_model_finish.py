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
    assert all(np.isclose(c['arc'][-1],.2325932,atol=1e-6) and np.all(np.diff(c['arc'])>0) for c in profile['chains'])


def test_seam_measurement_uses_current_positions_and_rejects_unobserved_metal(tmp_path):
    from phone_texture.iphone17_model_finish import _current_seam_readings,FRAMES
    files={}
    masks={}
    for face in FRAMES:
        a=np.full((44,42,4),128,np.uint8)
        a[...,3]=255
        if face=='left':
            a[24:35,:,:3]=175
        files[face]=tmp_path/(face+'.png')
        Image.fromarray(a).save(files[face])
        masks[face]=np.ones(a.shape[:2],bool)
    def boundary(y):
        return dict(names=['top','left'],edges=[0],arc=[0.,1.],
                    sides=[dict(name=face,points=[[8,y],[33,y]]) for face in ('top','left')])
    readings=_current_seam_readings(files,[boundary(10),boundary(29)],masks,2)
    assert readings[0]['after_p95_deltaE76']==0 and readings[1]['after_p95_deltaE76']>2
    masks['left'][25:32,:]=False
    with pytest.raises(ValueError,match='jointly editable'):
        _current_seam_readings(files,[boundary(29)],masks,2)
