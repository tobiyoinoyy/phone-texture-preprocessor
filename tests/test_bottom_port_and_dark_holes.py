"""Rounded-rectangle port hit beside stickers; local-contrast holes on dark colours."""
import numpy as np
import pytest
from phone_texture import bottom_rectify as br
from port_source_fixture import fixture


def port_center(a):
    return br.measure(a,'ip17pro')['port'][0]


@pytest.mark.parametrize('model',['ip17pro','ip17promax'])
def test_sticker_beside_port_does_not_move_port(model):
    clean,_=fixture(model)
    labelled,_=fixture(model,label_edge=True)
    assert abs(br.measure(labelled,model)['port'][0]-br.measure(clean,model)['port'][0])<=.5
    _,proof,_,_=br.rectify(labelled,model)
    assert abs(proof['port_holdout_error_px'])<=1 and proof['hole_contrast']=='raw'


def test_sticker_over_port_end_is_rejected_not_miscorrected():
    labelled,_=fixture('ip17pro',label_occlusion=True)
    with pytest.raises(br.Rejected,match='port_independent_holdout_failed'):
        br.rectify(labelled,'ip17pro')


def dark_uneven(a):
    """Dark metal lit from one side: holes only ~20 levels darker than local metal."""
    h,w=a.shape[:2];out=a.copy();gray=a[...,0].astype(float)
    metal=np.linspace(50,130,w)[None,:].repeat(h,0)
    dark=gray<128
    value=np.where(dark,metal-20,metal+(gray-210)*.3)
    out[...,:3]=np.clip(value,0,255).astype('uint8')[...,None]
    return out


def test_dark_uneven_metal_falls_back_to_local_contrast():
    a,_=fixture('ip17pro');dark=dark_uneven(a)
    with pytest.raises(br.Rejected,match='independent_hole_consensus_failed'):
        br.measure(dark,'ip17pro')
    local=br.measure(dark,'ip17pro','local');raw=br.measure(a,'ip17pro')
    assert local['hole_contrast']=='local'
    assert np.abs(local['holes'][:,:2]-raw['holes'][:,:2]).max()<=1
    _,proof,_,_=br.rectify(dark,'ip17pro')
    assert proof['hole_contrast']=='local' and abs(proof['port_holdout_error_px'])<=1


def test_raw_contrast_success_never_uses_fallback(monkeypatch):
    a,_=fixture('ip17pro');calls=[]
    original=br.measure
    def spy(image,model,contrast='raw'):
        calls.append(contrast);return original(image,model,contrast)
    monkeypatch.setattr(br,'measure',spy)
    br.rectify(a,'ip17pro')
    assert set(calls)=={'raw'}


def test_nested_port_rim_with_sticker_keeps_concentric_centre():
    """Pro Max style: dark opening inside a lighter chamfer rim, sticker against the rim."""
    plain,_=fixture('ip17promax',inner_shift=0,outer_gray=235)
    labelled,_=fixture('ip17promax',inner_shift=0,outer_gray=235,label_edge=True)
    assert abs(br.measure(labelled,'ip17promax')['port'][0]-br.measure(plain,'ip17promax')['port'][0])<=1
