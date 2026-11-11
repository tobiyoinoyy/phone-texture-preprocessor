import numpy as np
import pytest
from phone_texture.auto_geometry import plate_corner_contact_limit


def scene():
    h,w=500,300;yy,xx=np.indices((h,w));r=40.;left,right,top,bottom=50.,250.,30.,450.
    cx=np.clip(xx,left+r,right-r);cy=np.clip(yy,top+r,bottom-r)
    body=(xx-cx)**2+(yy-cy)**2<=r*r
    stand=(xx>=70)&(xx<170)&(yy>=420)
    raw=np.full((h,w,3),210,np.uint8);raw[stand]=60;raw[body]=140
    # Bright reflection on the actual body is not part of the support.
    reflection=body&(xx>90)&(xx<125)&(yy>438);raw[reflection]=235
    a=np.dstack([raw,np.where(body|stand,255,0).astype(np.uint8)])
    u=np.arange(50,251,dtype=float);horizontal=np.maximum(0,40-np.minimum(u-50,250-u))
    curve=40-np.sqrt(40**2-horizontal**2)
    proof=dict(axis_slope=0.,body_u=[left,right],bottom_v=bottom,
               upper_curve_u=u.tolist(),upper_curve_v=curve.tolist())
    support=np.zeros(w,bool);support[70:170]=True
    return a,raw,body,reflection,proof,support


def test_contact_follows_source_curve_preserving_round_body_and_reflection():
    a,raw,body,reflection,proof,support=scene()
    limit,record=plate_corner_contact_limit(a,raw,[0,450],proof,support)
    yy=np.arange(a.shape[0])[:,None];removed=(yy>limit[None,:]+1)&support[None,:]
    assert record['changed_columns']
    assert not np.any(removed&body)
    assert not np.any(removed&reflection)
    assert np.any(removed&(a[...,3]>0)&~body&(yy<450))
    assert record['corners'][0]['rgb_support_fraction']>=.6


def test_exposed_corner_is_not_recut_by_symmetric_shape_assumption():
    a,raw,body,reflection,proof,support=scene()
    # A shape discrepancy without a far stand projection is not authorization
    # to erase the actual segmented corner.
    limit,record=plate_corner_contact_limit(a,raw,[0,450],proof,np.zeros_like(support))
    assert record['changed_columns']==[]
    np.testing.assert_array_equal(limit,np.full(a.shape[1],450.))


def test_corner_contact_requires_independent_source_rgb():
    a,raw,body,reflection,proof,support=scene()
    with pytest.raises(ValueError,match='independent raw RGB'):
        plate_corner_contact_limit(a,np.full_like(raw,140),[0,450],proof,support)


def test_contact_does_not_trim_real_paper_tail():
    a,raw,body,reflection,proof,support=scene();paper=np.zeros(body.shape,bool)
    paper[430:480,70:96]=True
    limit,record=plate_corner_contact_limit(a,raw,[0,450],proof,support,paper)
    np.testing.assert_array_equal(limit[70:96],np.full(26,450.))
    assert not set(record['changed_columns'])&set(range(70,96))


def test_translucent_stand_fringe_is_removed_along_with_opaque_contact():
    a,raw,body,reflection,proof,support=scene()
    a[440:460,69,3]=32;support[69]=True
    limit,record=plate_corner_contact_limit(a,raw,[0,450],proof,support)
    # The fringe is outside the true curved body but above the old flat cut.
    assert not body[448,69] and a[448,69,3]==32
    assert 448>limit[69]+1
    assert 69 in record['changed_columns']
