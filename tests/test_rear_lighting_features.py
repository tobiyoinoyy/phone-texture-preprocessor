import numpy as np
import pytest
from phone_texture.rear_lighting_features import dark_rear_mark
from phone_texture.lighting import engine


def emblem_scene(scale=1):
    h,w=1000*scale,500*scale;yy,xx=np.indices((h,w));x=xx/scale;y=yy/scale
    a=np.full((h,w,4),255,np.uint8);a[...,:3]=100
    body=((x-250)/48)**2+((y-640)/46)**2<1
    body &= (x-299)**2+(y-630)**2>18**2
    leaf=((x-263)/12)**2+((y-580)/13)**2<1
    mark=body|leaf;a[mark,:3]=30
    panel=(y>350)&(y<950)&(x>25)&(x<475)
    return a,panel,mark


@pytest.mark.parametrize('scale',[1,2])
def test_source_mark_is_fit_exclusion_not_frozen_pixels(scale):
    a,panel,truth=emblem_scene(scale);before=a.copy()
    mask,proof=dark_rear_mark(a,panel)
    assert proof['detected'] and not proof['pixel_protection']
    assert mask[truth].all() and mask.sum()<truth.sum()*1.3
    np.testing.assert_array_equal(a,before)


def test_low_contrast_or_missing_leaf_does_not_invent_a_logo():
    a,panel,mark=emblem_scene();a[mark,:3]=90
    mask,proof=dark_rear_mark(a,panel)
    assert not mask.any() and not proof['detected']
    a,panel,_=emblem_scene();a[:600,:,:3]=100
    mask,proof=dark_rear_mark(a,panel)
    assert not mask.any() and not proof['detected']


def test_dark_albedo_does_not_create_halo_or_wash_out_feature():
    render=engine('aluminum');h=w=360;yy,xx=np.indices((h,w))
    illumination=43+xx*.012+yy*.009
    lab=np.zeros((h,w,3));lab[...,0]=illumination
    feature=(xx-180)**2+(yy-180)**2<=50**2
    lab[feature,0]-=24
    fit=~feature;editable=np.ones((h,w),bool)
    low,support,_=render.feature_excluded_field(lab,fit,editable,[12,12],feature)
    # Independent truth: an albedo edge must not affect inferred lighting.
    region=(xx-180)**2+(yy-180)**2<100**2
    assert np.max(abs(low[...,0][region]-illumination[region]))<.9
    corrected=lab[...,0]+(50-low[...,0])
    assert np.max(abs(corrected[region&~feature]-50))<.9
    assert np.max(abs(corrected[feature]-26))<.9
    assert support[editable].min()>=.008
    far=(xx<15)&(yy<15)
    normal,_,_=render.supported_field(lab,fit,far,[12,12])
    np.testing.assert_array_equal(low[far],normal[far])


def test_feature_field_rejects_unexcluded_feature_and_missing_source_support():
    render=engine('aluminum');lab=np.zeros((100,100,3));feature=np.zeros((100,100),bool)
    feature[35:65,35:65]=True
    with pytest.raises(ValueError,match='excluded source feature'):
        render.feature_excluded_field(lab,np.ones((100,100),bool),feature,[5,5],feature)
    with pytest.raises(ValueError,match='Unsupported'):
        render.feature_excluded_field(lab,np.zeros((100,100),bool),feature,[5,5],feature)
