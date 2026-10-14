"""Boundary defects must fail even with matching hashes/high confidence."""
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image, ImageDraw
from phone_texture.geometry_review import (guided_geometry,check_boundary_pixels,review_reasons,
    read_mask,verify_report,approve,digest,write,CHECKS)
from phone_texture.pipeline import process_batch
from geometry_helpers import approved_fixture,review_fixture


def rounded():
    im=Image.new('RGBA',(120,100))
    d=ImageDraw.Draw(im);d.rounded_rectangle((10,10,109,69),radius=15,fill=(130,80,20,255))
    d.rectangle((45,70,75,99),fill=(240,240,240,255))
    return im


def masks(tmp,im):
    remove=np.zeros((100,120),np.uint8);remove[70:,45:76]=255
    path=tmp/'remove.png';Image.fromarray(remove).save(path)
    return dict(rotation_ccw_degrees=0,support_remove_mask=dict(path=str(path),sha256=digest(path)))


def test_reviewed_rectangle_cannot_cut_round_corners():
    with pytest.raises(ValueError,match='intersects'):
        guided_geometry(rounded(),'top',dict(reviewed=True,crop_xyxy=[10,10,110,65],rotation_ccw_degrees=0))


def test_local_support_cut_preserves_original_body_and_complete_rounding(tmp_path):
    im=rounded();guide=masks(tmp_path,im);stages={}
    out,_=guided_geometry(im,'top',guide,lambda n,v:stages.update({n:v}))
    orig=np.array(im);before=np.array(stages['after_support'])
    np.testing.assert_array_equal(orig[:70],before[:70])
    fg=np.zeros((100,120),np.uint8);fg[64:67,23:97]=255
    bg=np.zeros_like(fg);bg[71:,45:76]=255
    assert check_boundary_pixels(stages['after_support'],fg,bg)['lost_body_pixels']==0
    assert out.size==(100,60)
    # One actual lost body pixel, or one alpha=1 support pixel, is a failure.
    for xy,alpha,reason in [((64,25),0,'lost_body'),((75,50),1,'residual_support')]:
        bad=before.copy();bad[xy][3]=alpha
        with pytest.raises(ValueError,match=reason):check_boundary_pixels(Image.fromarray(bad),fg,bg)


def test_paper_tail_conflicting_deletion_rejected(tmp_path):
    im=rounded();guide=masks(tmp_path,im)
    p=np.zeros((100,120),np.uint8);p[60:76,56:63]=255
    path=tmp_path/'paper.png';Image.fromarray(p).save(path)
    guide['protect_mask']=dict(path=str(path),sha256=digest(path))
    with pytest.raises(ValueError,match='protected paper'):guided_geometry(im,'back',guide)


def test_mask_replacement_invalidates_calibration(tmp_path):
    guide=masks(tmp_path,rounded());Image.new('L',(120,100),255).save(guide['support_remove_mask']['path'])
    with pytest.raises(ValueError,match='hash changed'):guided_geometry(rounded(),'top',guide)


def test_high_confidence_does_not_dismiss_support_or_ambiguity():
    reasons=review_reasons({'geometry':{'support_applied':True,'support_confidence':1.0,
        'body_evidence':{'review_reasons':['ambiguous_body_edges']}}})
    assert reasons==['ambiguous_body_edges','heuristic_support_cut']


def tiny_batch(tmp):
    folder=tmp/'images'/'A';folder.mkdir(parents=True)
    for face in ('front','back','left','right','top','bottom'):
        Image.new('RGBA',(16,32),(100,80,70,255)).save(folder/(face+'.png'))
    return folder.parent


@pytest.mark.parametrize('fault',['pending','missing_check','evidence','source','ambiguity_no_witnesses'])
def test_geometry_gate_rejects_bad_proof(tmp_path,fault):
    root=tiny_batch(tmp_path);p=approved_fixture(root,tmp_path);report=json.loads(p.read_text())
    if fault=='pending':report['geometry_status']='needs_review'
    if fault=='missing_check':report['geometry_review']['items']['A/top']['four_corners']=False
    if fault=='evidence':Path(report['images'][0]['geometry_evidence'][0]['path']).write_bytes(b'changed')
    if fault=='source':Path(report['images'][0]['source']).write_bytes(b'changed')
    if fault=='ambiguity_no_witnesses':report['images'][0]['review_reasons']=['ambiguous_body_edges']
    write(p,report)
    with pytest.raises(ValueError):verify_report(p)


def test_missing_geometry_blocks_lighting_checkpoint(tmp_path):
    from test_six_view_workflow import w
    root=tiny_batch(tmp_path);work=tmp_path/'work';w.intake(root,work)
    with pytest.raises(ValueError,match='Approved geometry'):w.checkpoint(work,root)
    assert not (work/'geometry.json').exists()


@pytest.mark.parametrize('fault',['stale_review','unreviewed','changed_output'])
def test_approval_failure_never_publishes(tmp_path,fault):
    from test_unified_pipeline import sources
    source=sources(tmp_path/'source');work=tmp_path/'work';out=tmp_path/'out'
    process_batch(source,out,work,mode='deskew',faces=['top','bottom'])
    assert not out.exists()
    p=review_fixture(work);review=json.loads(p.read_text())
    if fault=='stale_review':review['report_sha256']='stale'
    if fault=='unreviewed':review['items']['SKU_01/back']['complete_lower_edge']=False
    if fault=='changed_output':Image.new('RGBA',(20,20),'white').save(work/'staged/SKU_01/front.png')
    write(p,review)
    with pytest.raises(ValueError):approve(work,p,out)
    assert not out.exists()
    assert json.loads((work/'report.json').read_text())['geometry_status']=='needs_review'
