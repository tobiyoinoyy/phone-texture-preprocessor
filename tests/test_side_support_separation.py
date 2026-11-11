import numpy as np
from PIL import Image
from phone_texture import auto_geometry as ag


def fixture(monkeypatch, error=.2):
    a=np.zeros((80,240,4),np.uint8)
    a[20:41,20:220,:3]=120;a[20:41,20:220,3]=255
    a[40:70,100:140,:3]=80;a[40:70,100:140,3]=255
    # A real dent and soft upper corner are retained, not painted or expanded.
    a[25:28,70:72,:3]=30;a[20,20:24,3]=100
    p={'rgb_boundary_error_px':error,'rgb_boundary_residual_p95_px':.3}
    monkeypatch.setattr(ag,'frame_rails',lambda *args:(np.array([0.,20.]),np.array([0.,40.4]),p.copy()))
    return a


def test_separation_keeps_body_and_fractional_edge_without_stand_margin(monkeypatch):
    a=fixture(monkeypatch);before=a.copy();stages={}
    image,proof=ag.frame_geometry(Image.fromarray(a),'left',lambda n,v:stages.update({n:np.array(v)}),Image.fromarray(a[...,:3]))
    separated=stages['after_support']
    np.testing.assert_array_equal(a,before)
    np.testing.assert_array_equal(separated[...,:3],before[...,:3])
    np.testing.assert_array_equal(separated[:40],before[:40])
    assert np.all(separated[40,100:140,3]==102)
    assert not separated[41:,:,3].any()
    assert proof['support_edge_separation']['outward_allowance_px']==0
    assert proof['support_edge_separation']['phone_core_inward_offset_px']==0


def test_uncertain_boundary_is_recorded_and_not_cropped_inward(monkeypatch):
    a=fixture(monkeypatch,error=1.2);stages={}
    _,p=ag.frame_geometry(Image.fromarray(a),'right',lambda n,v:stages.update({n:np.array(v)}),Image.fromarray(a[...,:3]))
    assert p['support_edge_separation']['status']=='unresolved_source_alpha_disagreement'
    assert p['support_edge_separation']['outward_allowance_px']==1
    assert np.all(stages['after_support'][40,100:140,3]==255)


def test_existing_soft_boundary_is_never_made_more_opaque(monkeypatch):
    a=fixture(monkeypatch);a[40,100:140,3]=51;stages={}
    ag.frame_geometry(Image.fromarray(a),'left',lambda n,v:stages.update({n:np.array(v)}),Image.fromarray(a[...,:3]))
    assert np.all(stages['after_support'][40,100:140,3]==51)
    assert np.all(stages['after_support'][...,3]<=a[...,3])
