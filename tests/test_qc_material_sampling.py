import numpy as np
import pytest
from scipy import ndimage
from phone_texture.lighting import engine
from phone_texture.side_material import refine_capsule_boundary


def test_thin_rim_uses_own_opaque_normal_samples_not_another_edges_shoulder():
    e=engine('aluminum');h,w=120,160
    alpha=np.full((h,w),255,np.uint8)
    distance=ndimage.distance_transform_edt(np.pad(alpha>0,1))[1:-1,1:-1]
    yy,xx=np.indices((h,w));editable=((distance<=8)&(yy<h-20))|(xx<8)|(xx>=w-8)|(yy>=h-2)
    # The dark glass below the thick side rim is not a material sample.
    editable[(yy>h-8)&(yy<h-2)&(xx>8)&(xx<w-9)]=False
    fit=editable.copy();lab=np.full((h,w,3),0.);lab[...,0]=45
    before=lab.copy()
    with pytest.raises(ValueError,match='Unsupported editable contour'):
        e.supported_contour_field(lab,fit,editable,alpha,[10,2],12)
    low,support,reliable=e.supported_contour_field(lab,fit,editable,alpha,[10,2],source_inset_limit=12)
    assert support[editable].min()>=.008
    np.testing.assert_allclose(low[editable,0],45)
    np.testing.assert_array_equal(lab,before)
    fit[-12:]=False
    with pytest.raises(ValueError,match='Unsupported editable contour'):
        e.supported_contour_field(lab,fit,editable,alpha,[10,2],source_inset_limit=12)


def test_bounded_capsule_refinement_requires_complete_source_edges():
    h,w=200,80;yy,xx=np.indices((h,w));x,y,bw,bh=25,40,30,120;r=(bw-1)/2
    source=(xx-x-r)**2+(yy-np.clip(yy,y+r,y+bh-1-r))**2<=r*r
    boundary=source&~ndimage.binary_erosion(source)
    distance=ndimage.distance_transform_edt(~boundary);alpha=np.full((h,w),255,np.uint8)
    original=ndimage.binary_dilation(source,iterations=1)
    result=refine_capsule_boundary(distance,alpha,[x-1,y-1,bw+2,bh+2],original,1.5)
    assert result is not None
    mask,proof=result
    assert proof['boundary_support']>=.95 and min(proof['edge_support'])>=.75
    broken=boundary.copy();broken[(xx<x+bw*.3)&(yy>y+bw)&(yy<y+bh-bw)]=False
    assert refine_capsule_boundary(ndimage.distance_transform_edt(~broken),alpha,[x-1,y-1,bw+2,bh+2],original,1.5) is None


def test_front_contour_field_is_invariant_to_transparent_canvas_margin():
    e=engine('aluminum');h,w=100,160;yy,xx=np.indices((h,w))
    alpha=np.where(yy>=2+((xx-80)/50)**2,255,0).astype(np.uint8)
    # Thin real metal reaches the tightly cropped bottom canvas edge.
    editable=(alpha>0)&((yy>h-3)|(xx<7)|(xx>w-8))
    fit=editable.copy();lab=np.zeros((h,w,3));lab[...,0]=35+xx/12+yy/25
    args=dict(sigma=[8,2],source_inset_limit=8)
    low,support,_=e.supported_contour_field(lab,fit,editable,alpha,**args)
    pad=20
    low_p,support_p,_=e.supported_contour_field(np.pad(lab,((pad,pad),(pad,pad),(0,0))),
        np.pad(fit,pad),np.pad(editable,pad),np.pad(alpha,pad),**args)
    # Identical original pixels must retain identical support and correction;
    # padding is transparent geometry, never invented image content.
    np.testing.assert_allclose(support[editable],support_p[pad:-pad,pad:-pad][editable],atol=1e-12)
    np.testing.assert_allclose(low[editable],low_p[pad:-pad,pad:-pad][editable],atol=1e-10)
    # Non-material RGB, including extreme glass values, cannot enter the
    # source estimate even when the normal/tangent window crosses that area.
    altered=lab.copy();altered[~fit]=[1e6,-1e6,1e6]
    low_b,support_b,_=e.supported_contour_field(altered,fit,editable,alpha,
        sigma=[8,2],source_inset_limit=8.2)
    np.testing.assert_allclose(low[editable],low_b[editable],atol=1e-10)
    np.testing.assert_allclose(support[editable],support_b[editable],atol=1e-12)
