import numpy as np
import pytest

from phone_texture.auto_geometry import plate_axis,_plate_axis_strict


def image_pair(angle=0.,both_ragged=False):
    h,w=500,300;yy,xx=np.indices((h,w));slope=np.tan(np.deg2rad(angle))
    left=70+slope*(yy[:,0]-250);right=230+slope*(yy[:,0]-250)
    raw=np.full((h,w,3),220,np.uint8)
    inside=(xx>=left[:,None])&(xx<=right[:,None]);raw[inside]=35
    matte=np.zeros((h,w,4),np.uint8);matte[...,:3]=raw
    jagged=7*np.sin(np.arange(h)*.17)
    mask=(xx>=left[:,None]+jagged[:,None])&(xx<=right[:,None]+(jagged[:,None] if both_ragged else 0))
    matte[...,3]=mask.astype(np.uint8)*255
    return matte,raw,np.arange(30,470)


@pytest.mark.parametrize('angle',[0.,.7,-.7])
def test_one_unstable_matte_side_recovers_from_two_actual_source_edges(angle):
    matte,raw,rows=image_pair(angle)
    with pytest.raises(ValueError):_plate_axis_strict(matte,raw,rows)
    slope,proof=plate_axis(matte,raw,rows)
    assert abs(np.degrees(np.arctan(slope))+angle)<.08
    assert proof['method']=='one_opposite_silhouette_and_two_raw_sides_v1'
    assert proof['raw_source_residual_limit_px']==2.5
    assert not proof['source_RGB_edited'] and not proof['alpha_edited_for_axis']


def test_blank_source_cannot_use_an_alpha_proposal():
    matte,raw,rows=image_pair();raw[:]=120
    with pytest.raises(ValueError):plate_axis(matte,raw,rows)


def test_one_source_edge_is_not_two_independent_edges():
    matte,raw,rows=image_pair();raw[:,:150]=35
    with pytest.raises(ValueError):plate_axis(matte,raw,rows)


def test_two_unstable_silhouettes_do_not_supply_an_independent_anchor():
    matte,raw,rows=image_pair(both_ragged=True)
    with pytest.raises(ValueError):plate_axis(matte,raw,rows)


def test_wrong_source_angle_does_not_borrow_the_alpha_anchor():
    matte,_,rows=image_pair();_,raw,_=image_pair(2.)
    with pytest.raises(ValueError):plate_axis(matte,raw,rows)


@pytest.mark.parametrize('seed',range(20))
def test_unstructured_RGB_noise_is_not_a_body_axis(seed):
    matte,_,rows=image_pair();raw=np.random.default_rng(seed).integers(0,256,(500,300,3),dtype=np.uint8)
    with pytest.raises(ValueError):plate_axis(matte,raw,rows)
