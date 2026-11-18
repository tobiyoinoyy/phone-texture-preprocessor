import cv2
import numpy as np
import pytest
from PIL import Image

from phone_texture.iphone17_model_finish import _photographed_lens_rings
from phone_texture.iphone17_paper import measured_paper
from phone_texture.iphone17_structure import _warp_along_rail


def test_dark_lenses_require_two_closed_rings():
    w,h=1000,2050
    gray=np.full((550,380),28,np.uint8)
    alpha=np.full_like(gray,255)
    for y in (176,418):
        cv2.circle(gray,(188,y),107,115,-1,cv2.LINE_AA)
        cv2.circle(gray,(188,y),95,14,-1,cv2.LINE_AA)
    pair=_photographed_lens_rings(gray,alpha,w,h)
    assert np.max(abs(pair-np.array([[188,176,95],[188,418,95]])))<1
    gray[300:,185:210]=28
    with pytest.raises(ValueError,match='closed'):
        _photographed_lens_rings(gray,alpha,w,h)


def test_printed_paper_end_must_stop_on_metal():
    a=np.full((120,800,4),105,np.uint8)
    a[...,3]=255
    a[:94,350:424,:3]=235
    for y in range(8,78,12):
        a[y:y+5,365:402,:3]=40
    a[68:83,345:350,:3]=235
    proof={'seed_box':[360,18,408,87]}
    mask,_=measured_paper(a,proof)
    assert mask[80,347] and mask[50,375] and mask[90,417]
    assert not mask[80,339] and not mask[100,380]
    a[94:,350:424,:3]=235
    # A continuous body bevel behind a strip cannot be its lower end.
    a[90:93,:,:3]=65
    with pytest.raises(ValueError):
        measured_paper(a,proof)


def test_native_rail_transport_preserves_locked_rows_and_body():
    h,w=240,48
    yy,xx=np.indices((h,w))
    edge=np.interp(yy[:,0],[0,80,90,130,150,h-1],[3,3,5,5,3,3])
    a=np.zeros((h,w,4),np.uint8)
    a[...,:3]=np.where((yy<90)[...,None],70,150)
    a[...,3]=np.rint(255*np.minimum(np.clip(xx-edge[:,None]+.5,0,1),
                                    np.clip(w-3-xx,0,1))).astype(np.uint8)
    image=Image.fromarray(a)
    src=np.array([25,80,110,190])
    dst=np.array([25,92,122,190])
    result=np.asarray(_warp_along_rail(image,src,dst))
    assert np.array_equal(result[:25],a[:25])
    assert np.array_equal(result[190:],a[190:])
    assert not np.any((a[...,3]>=240)&(result[...,3]<128))


def test_dark_button_plane_needs_both_cap_edges_and_all_three_holdouts():
    from phone_texture.iphone17_buttons import _button_face_plane,button_edges
    gray=np.full((1980,110,3),145,np.uint8)
    boxes=[[32,y,79,y+n] for y,n in ((370,100),(540,160),(740,160))]
    for x,y,u,v in boxes:
        gray[y:v,x:u]=27
        gray[y+5:v-7,x+4:u-4]=175
    im=Image.fromarray(gray)
    sources=button_edges(im,boxes)
    targets=np.array([[y+5-.5,v-7-.5] for x,y,u,v in boxes])*.95+42
    _,_,proof=_button_face_plane(im,boxes,sources,targets)
    assert np.max(np.abs(proof['held_out_error_px']))<1
    inconsistent=targets.copy()
    inconsistent[-1]+=16
    with pytest.raises(ValueError,match='Held-out'):
        _button_face_plane(im,boxes,sources,inconsistent)
    x,y,u,v=boxes[0]
    gray[y:v,x:u]=27
    with pytest.raises(ValueError,match='cap edge'):
        _button_face_plane(Image.fromarray(gray),boxes,sources,targets)
