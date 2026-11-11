"""Recover real edges hidden by a measured illumination slope."""
import numpy as np
import pytest
from PIL import Image
from phone_texture.antenna_material import trace_boundaries
from phone_texture.photometric_recovery import recover
from phone_texture.source_antenna_evidence import create,create_photometric,load


def shadow_source():
    a=np.full((500,100,4),255,np.uint8);a[...,:3]=180
    a[42:51,:,:3]=90;a[450:459,:,:3]=90
    for lo,hi in [(42,51),(450,459)]:
        ys=np.arange(lo-22,hi+23);value=100+.8*(ys-(lo+hi)/2)
        value[(ys>=lo)&(ys<hi)]-=3
        a[ys,:12,:3]=np.clip(np.rint(value[:,None,None]),0,255).astype(np.uint8)
    return a


def test_source_recovery_retains_original_pixels_and_alpha():
    a=shadow_source();before=a.copy();im=Image.fromarray(a)
    with pytest.raises(ValueError):trace_boundaries(im,[42,51,450,459])
    reference,proof=recover(im,[42,51,450,459]);lines,_=trace_boundaries(reference,[42,51,450,459])
    assert proof['normalized_columns']
    assert np.max(abs(lines[:,0]-[41.5,50.5,449.5,458.5]))<2.5
    for row in proof['normalized_columns']:
        assert min(row['source_evidence']['edge_t'])>=1.5
        assert row['source_evidence']['joint_rectangular_t']>=3
        assert row['source_evidence']['opposing_agreement']>=.8
    np.testing.assert_array_equal(a,before)
    np.testing.assert_array_equal(np.asarray(reference)[...,3],a[...,3])


def test_observer_cannot_override_a_rejection():
    with pytest.raises(ValueError):
        trace_boundaries(Image.fromarray(shadow_source()),[42,51,450,459],_observer=lambda *args:np.zeros((4,100)))


def test_cached_analytical_evidence_is_bound_and_never_written_to_geometry(tmp_path):
    a=shadow_source();geometry=tmp_path/'geometry.png';Image.fromarray(a).save(geometry)
    original=tmp_path/'original.png';Image.fromarray(a[...,:3]).transpose(Image.Transpose.ROTATE_270).save(original)
    spec=dict(source_size=[500,100],rotation_ccw_degrees=0,canvas_trim_xyxy=[0,0,500,100],visible_canvas_trim={'crop_xyxy':[0,0,100,500]})
    context=create(original,geometry,spec,'left',tmp_path/'raw-reference.png');saved=geometry.read_bytes()
    _,bound=create_photometric(context,geometry,[42,51,450,459],'left')
    trace_boundaries(load(bound,geometry,'left'),[42,51,450,459])
    assert geometry.read_bytes()==saved
    with open(bound['photometric_path'],'ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='changed'):load(bound,geometry,'left')


@pytest.mark.parametrize('kind',['single_edge','same_sign','wrong_width'])
def test_background_model_cannot_fabricate_a_missing_edge(kind):
    a=np.full((500,100,4),255,np.uint8);a[...,:3]=180
    a[42:51,:,:3]=90;a[450:459,:,:3]=90;a[:250,:12,:3]=180
    if kind=='single_edge':a[42:250,:12,:3]=90
    elif kind=='same_sign':a[42:250,:12,:3]=120;a[51:250,:12,:3]=60
    else:a[42:61,:12,:3]=90
    with pytest.raises(ValueError):recover(Image.fromarray(a),[42,51,450,459])


def test_weak_endpoint_shape_is_flagged_not_relabelled_as_missing_terminal_columns(monkeypatch):
    from phone_texture import photometric_recovery as recovery
    a=np.full((500,100,4),255,np.uint8);a[...,:3]=180
    a[42:51,:,:3]=90;a[450:459,:,:3]=90
    rng=np.random.default_rng(3)
    a[:,:8,:3]=np.clip(np.linspace(60,85,500)[:,None,None]+
        rng.normal(0,4,(500,8,1)),0,255).astype(np.uint8)
    before=a.copy()
    def inappropriate_fit(*args,**kwargs):
        pytest.fail('A shape rejection must not invoke terminal illumination fitting')
    monkeypatch.setattr(recovery,'normalize',inappropriate_fit)
    # R10 (2026-11-06): a weak end strip no longer rejects; it is flagged for
    # review and its columns hold the nearest coherent pair.
    reference,proof=recovery.recover(Image.fromarray(a),[42,51,450,459])
    assert proof['method']=='ordinary_source_pair'
    from phone_texture.antenna_material import trace_boundaries
    _,gate=trace_boundaries(reference,[42,51,450,459])
    assert gate['endpoint_shape_review_required']
    assert 'left' in gate['endpoint_held_sides']
    for band in gate['bands']:
        assert set(range(8))<=set(band['endpoint_extended_columns'])
    np.testing.assert_array_equal(a,before)


@pytest.mark.parametrize('seed',[4,5,7,8,9,11,13,16,17])
def test_noisy_paired_peaks_remain_rejected_during_recovery(seed):
    a=np.full((500,100,4),255,np.uint8);a[...,:3]=180
    a[42:51,:,:3]=90;a[450:459,:,:3]=90
    rng=np.random.default_rng(seed)
    a[:,:8,:3]=np.clip(np.linspace(60,85,500)[:,None,None]+
        rng.normal(0,4,(500,8,1)),0,255).astype(np.uint8)
    # Recovery never turns noise into measured pairs (see endpoint-shape test).
    try:
        reference,_=recover(Image.fromarray(a),[42,51,450,459])
    except ValueError:
        return
    _,proof=trace_boundaries(reference,[42,51,450,459])
    for band in proof['bands']:
        assert set(range(8))<=set(band['endpoint_extended_columns'])
