"""Lighting material closure is separate from geometry and source texture."""
import numpy as np
import pytest
from PIL import Image
from pathlib import Path

from phone_texture import auto_material, lighting_review
from phone_texture.bottom_paper_closure import resolve_material_strip_end
from test_unmarked_paper_closure import source


def test_material_closure_releases_metal_without_changing_source_or_locator():
    a, locator, actual, proof = source(slope=.08, antialias=False)
    original, old_locator = a.copy(), locator.copy()
    material, evidence = resolve_material_strip_end(a, locator, proof)
    np.testing.assert_array_equal(material, actual)
    np.testing.assert_array_equal(a, original)
    np.testing.assert_array_equal(locator, old_locator)
    assert evidence['material_boundary_resolution'] == 'original_source_closure'


def test_delayed_metal_window_cannot_skip_reappearing_paper():
    a, locator, _, proof = source(end=140)
    # A full-width printed band mimics a transverse paper edge. Paper returns
    # before a later exposed-metal window, so window search must still reject.
    a[95:107, 290:350, :3] = [25, 35, 58]
    a[122:, 290:350, :3] = [25, 35, 58]
    with pytest.raises(ValueError, match='material boundary unresolved'):
        resolve_material_strip_end(a, locator, proof)


def test_no_cross_edge_is_explicitly_unresolved_not_a_certified_closure():
    a, locator, _, proof = source(end=140)
    result, evidence = resolve_material_strip_end(a, locator, proof)
    np.testing.assert_array_equal(result, locator)
    assert evidence['material_boundary_resolution'] == 'unresolved_no_source_cross_edge'


def test_marker_position_prediction_alone_does_not_certify_paper_to_canvas(monkeypatch):
    a, locator, _, proof = source(end=140)
    monkeypatch.setattr(auto_material, 'trim_strip_to_source_end',
        lambda a, mask, proof: (mask, dict(proof, source_paper_end='continues_to_face_canvas')))
    result, evidence = resolve_material_strip_end(a, locator, proof)
    np.testing.assert_array_equal(result, locator)
    assert evidence['material_boundary_resolution'] == 'unresolved_no_source_cross_edge'


def test_numeric_marker_and_non_rejection_audit_cannot_certify_different_tail_material(monkeypatch):
    a, locator, _, proof = source()
    a[95:,290:350,:3]=[100,110,130]
    proposed=locator.copy();proposed[95:]=False
    monkeypatch.setattr(auto_material,'trim_strip_to_source_end',
        lambda a,mask,proof:(proposed,dict(proof,source_paper_end=94,
            bottom_source_audit=dict(status='unresolved',certifies_complete_paper=False))))
    with pytest.raises(ValueError,match='independent full-width source material closure is required'):
        resolve_material_strip_end(a,locator,proof)


def test_numeric_marker_is_independently_resolved_from_original_source(monkeypatch):
    a, locator, actual, proof = source()
    # A marker-only proposal is deliberately too short. Independent source
    # margins/cross-edge/metal witnesses must produce the actual paper end.
    proposed=locator.copy();proposed[85:]=False
    monkeypatch.setattr(auto_material,'trim_strip_to_source_end',
        lambda a,mask,proof:(proposed,dict(proof,source_paper_end=84)))
    result,evidence=resolve_material_strip_end(a,locator,proof)
    np.testing.assert_array_equal(result,actual)
    assert evidence['material_boundary_resolution']=='source_aligned_metal_window'


def test_bottom_paper_is_lit_with_frame_metal_without_boundary_search(tmp_path, monkeypatch):
    a, locator, actual, _ = source()
    path=tmp_path/'bottom.png';Image.fromarray(a).save(path)
    import phone_texture.bottom_paper_closure as closure
    def no_boundary(*args, **kwargs):
        raise AssertionError('bottom paper boundary must not be searched')
    monkeypatch.setattr(closure, 'resolve_material_strip_end', no_boundary)
    job, proof, allowed, protected=auto_material.propose_job(path,'bottom',tmp_path/'materials',protection_policy='none')
    view=job['views'][0]
    assert not protected.any() and proof['paper']['method']=='bottom_paper_lit_with_frame_metal'
    assert [d['name'] for d in view['domains']]==['main_surface']
    assert np.all(allowed[locator&(a[...,3]>0)])
    fit=np.array(Image.open(view['domains'][0]['fit_mask']).convert('L'))>0
    assert not np.any(fit&locator)
    assert 'source_label_location_hint_mask' not in view
    scope=auto_material.lighting_scope(path,'bottom',job,tmp_path/'scope')
    assert not any(name.startswith('bottom_paper_') for name in scope['local_checks'])


def normal_fixture(tmp_path):
    h,w=32,80
    a=np.full((h,w,4),255,np.uint8)
    a[...,:3]=np.broadcast_to((180-np.arange(h)*4)[:,None,None],(h,w,3))
    source_path=tmp_path/'source.png';Image.fromarray(a).save(source_path)
    target=np.zeros((h,w),bool);target[:,30:50]=True
    peers=np.zeros((h,w),bool);peers[:,5:20]=True
    target_path=tmp_path/'target.png';Image.fromarray(target.astype('uint8')*255).save(target_path)
    peer_path=tmp_path/'peers.png';Image.fromarray(peers.astype('uint8')*255).save(peer_path)
    scope=dict(source_sha256=lighting_review.digest(source_path),action='aluminum',mode='regional',
        reason='Source-matched same-normal paper-tail metal',protection_policy='none',
        local_checks={'tail':dict(mask=str(target_path),normal_peer_mask=str(peer_path),
            min_L=0,max_L=100,percentile=10,max_normal_peer_step_L=1.25)})
    out=a.copy();out[...,:3]=200
    return source_path,a,out,scope,target,peers


def test_real_normal_gradient_is_shared_by_source_matched_metal(tmp_path):
    source_path,a,out,scope,target,peers=normal_fixture(tmp_path)
    output=tmp_path/'output.png';Image.fromarray(out).save(output)
    # Strong compensation of the original normal gradient is not a local seam.
    delta=lighting_review.lightness(out[...,:3])-lighting_review.lightness(a[...,:3])
    assert abs(np.diff(delta,axis=0)).max()>1.25
    allowed=np.ones(target.shape,bool)
    checks=lighting_review.measure_local(scope,source_path,output,allowed,~allowed)
    assert checks['tail']['max_normal_peer_step_L']==0


def test_new_normal_step_is_rejected_against_unchanged_same_row_peers(tmp_path):
    source_path,a,out,scope,target,peers=normal_fixture(tmp_path)
    out[16:,30:50,:3]=225
    output=tmp_path/'bad.png';Image.fromarray(out).save(output)
    allowed=np.ones(target.shape,bool)
    with pytest.raises(ValueError,match='Lighting normal peer discontinuity'):
        lighting_review.measure_local(scope,source_path,output,allowed,~allowed)


def test_normal_peer_shortage_is_a_failure_and_peer_mask_is_frozen(tmp_path):
    source_path,a,out,scope,target,peers=normal_fixture(tmp_path)
    output=tmp_path/'output.png';Image.fromarray(out).save(output)
    allowed=np.ones(target.shape,bool)
    peers[:,10:]=False
    peer_path=Path(scope['local_checks']['tail']['normal_peer_mask'])
    Image.fromarray(peers.astype('uint8')*255).save(peer_path)
    with pytest.raises(ValueError,match='Insufficient same-row normal metal peers'):
        lighting_review.measure_local(scope,source_path,output,allowed,~allowed)
    work=tmp_path/'work';work.mkdir()
    lighting_review.write(work/'geometry.json',{'sku':{'bottom':{'path':str(source_path),'sha256':lighting_review.digest(source_path)}}})
    lighting_review.write(work/'spec.json',{'sku':{'bottom':scope}})
    lighting_review.freeze_scope(work,work/'spec.json')
    frozen=lighting_review.read(work/'lighting-scope.json')
    assert str(peer_path.resolve()) in frozen['assets']
    peers[:,15]=True;Image.fromarray(peers.astype('uint8')*255).save(peer_path)
    with pytest.raises(ValueError,match='changed'):
        lighting_review.load_scope(work)

