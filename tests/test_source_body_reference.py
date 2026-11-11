"""Small isolated controls; run only in the root's serialized guard."""
import copy
import hashlib

import numpy as np
from PIL import Image
import pytest

from phone_texture import source_body_reference as sbr
from phone_texture import structure_review as sr


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tmp_path, *, top_step=True, left_step=True):
    raw=np.full((260,200,3),200,np.uint8)
    raw[20:180,20:180]=70
    if not top_step:
        raw[:30,20:180]=70
    if not left_step:
        raw[160:200,40:80]=70
    alpha=np.zeros((260,200),np.uint8)
    alpha[20:180,20:180]=255
    # Central stand changes canvas extent, never physical rail evidence.
    alpha[180:250,90:110]=255
    raw_path=tmp_path/'raw.png';back_path=tmp_path/'back.png'
    Image.fromarray(raw).save(raw_path)
    Image.fromarray(np.dstack([raw,alpha])).save(back_path)
    geometry=dict(method=sbr.GEOMETRY_METHOD,source_size=[200,260],output_size=[200,260],
        rotation_ccw_degrees=0.,canvas_trim_xyxy=[0,0,200,260],
        visible_canvas_trim=dict(crop_xyxy=[0,0,200,260]),rgb_painted=False,
        body_x=[20,180],bottom_rail=[0.,180.],alpha_lower_rail=[0.,180.],
        clear_bands=[.18,.25],exposed_samples=50,lower_edge_residual_p95_px=1.,
        review_reasons=['heuristic_support_cut','complete_outline_requires_review'])
    item=dict(face='back',source_sha256=sha(raw_path),output_sha256=sha(back_path),geometry=geometry)
    return raw_path,back_path,item


def test_independent_body_ignores_central_extension_and_stays_review(tmp_path):
    raw,back,item=fixture(tmp_path)
    record=sbr.build(raw,back,item)
    assert record['body_y']==[20.,179.5]
    assert record['reference_supported'] and record['reference_review_required']
    assert not any(record[k] for k in ('reference_certified','remap_verified','visual_approved','publishable'))
    before=copy.deepcopy(record)
    assert sbr.targets(record,[35.,45.,150.,160.],[5.,240.],260)!=sbr.targets(record,[35.,45.,150.,160.],[7.,230.],260)
    assert record==before  # side coordinates cannot fit/change the source body
    assert sha(raw)==item['source_sha256'] and sha(back)==item['output_sha256']


@pytest.mark.parametrize('kwargs',[dict(top_step=False),dict(left_step=False)])
def test_top_and_each_native_flank_are_required(tmp_path,kwargs):
    raw,back,item=fixture(tmp_path,**kwargs)
    with pytest.raises(ValueError,match='top or two native flank'):
        sbr.build(raw,back,item)


def test_internal_line_with_no_source_transition_is_not_body_end(tmp_path):
    raw,back,item=fixture(tmp_path)
    item['geometry']['bottom_rail']=[0.,120.]
    item['geometry']['alpha_lower_rail']=[0.,120.]
    with pytest.raises(ValueError,match='top or two native flank'):
        sbr.build(raw,back,item)


def test_source_or_geometry_identity_change_rejects(tmp_path):
    raw,back,item=fixture(tmp_path)
    record=sbr.build(raw,back,item)
    wrong=tmp_path/'wrong.png';wrong.write_bytes(b'another capture')
    with pytest.raises(ValueError,match='another back'):
        sbr.validate(record,wrong)
    raw.write_bytes(raw.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='bound source changed'):
        sbr.validate(record,back)


@pytest.mark.parametrize('field,value',[
    ('canvas_trim_xyxy',[1,0,201,260]),('output_size',[199,260]),
    ('method','unverified_panel_line'),('alpha_lower_rail',[0.,160.])])
def test_wrong_mapping_or_unsupported_provenance_rejects(tmp_path,field,value):
    raw,back,item=fixture(tmp_path)
    item['geometry'][field]=value
    with pytest.raises(ValueError):
        sbr.build(raw,back,item)


def test_resealed_length_edit_cannot_replace_bound_raw_rail(tmp_path):
    raw,back,item=fixture(tmp_path)
    record=sbr.build(raw,back,item)
    record['body_y'][1]-=10
    record['proposal']['body_y']=record['body_y']
    record['evidence_sha256']=sbr._seal(record)
    with pytest.raises(ValueError,match='fixed source rail'):
        sbr.validate(record,back)


def test_review_record_cannot_claim_remap_pass(tmp_path):
    raw,back,item=fixture(tmp_path)
    record=sbr.build(raw,back,item)
    record['remap_verified']=True
    record['evidence_sha256']=sbr._seal(record)
    with pytest.raises(ValueError,match='must not claim final acceptance'):
        sbr.validate(record,back)


def test_edge_coordinates_have_no_half_pixel_shift():
    geometry=dict(source_size=[200,260],output_size=[180,220],rotation_ccw_degrees=0.,
        canvas_trim_xyxy=[10,20,190,240],visible_canvas_trim=dict(crop_xyxy=[0,0,180,220]))
    forward,inverse,proof=sbr.coordinate_map(geometry)
    assert forward(30.5,180.)==pytest.approx((20.,159.5))
    assert inverse(20.,159.5)==pytest.approx((30.,179.5))
    assert proof['maximum_roundtrip_error_px']<1e-8


def test_reference_gate_uses_same_mapping_and_keeps_pixel_detection(tmp_path,monkeypatch):
    raw,back,item=fixture(tmp_path)
    record=sbr.build(raw,back,item)
    rows=[35.,45.,150.,160.]
    ref=dict(face='back',sha256=sha(back),body_y=record['body_y'],antenna_rows=rows,
        x_range=[30,40],search_windows=[[30,40],[41,49],[146,154],[156,165]],
        signs=[1,-1,1,-1],source_body_reference=record)
    monkeypatch.setattr(sr,'detect',lambda *args:np.asarray(rows))
    assert sr.reference({'back':back},ref,260,[5.,240.],sbr.MODE)==pytest.approx(
        sbr.targets(record,rows,[5.,240.],260))
    monkeypatch.setattr(sr,'detect',lambda *args:np.asarray(rows)+5)
    with pytest.raises(ValueError,match='actual reference pixels'):
        sr.reference({'back':back},ref,260,[5.,240.],sbr.MODE)


def test_old_hardware_gate_still_rejects_even_supported_reference(tmp_path,monkeypatch):
    raw,back,item=fixture(tmp_path)
    record=sbr.build(raw,back,item)
    # source is merely a small synthetic geometry fixture; target overlaps the
    # independently supplied hardware range. No warp/render is run here.
    rows=[35.,45.,150.,160.]
    ref=dict(face='back',sha256=sha(back),body_y=record['body_y'],antenna_rows=rows,
        x_range=[30,40],search_windows=[[30,40],[41,49],[146,154],[156,165]],
        signs=[1,-1,1,-1],source_body_reference=record)
    cfg=dict(source_sha256=sha(back),body_y=[5.,240.],coordinate_mode=sbr.MODE,
        reference_review_required=True,source_lines=[35.,45.,90.,110.,150.,160.],
        antenna_indices=[0,1,4,5],back=ref,button_ranges=[[50,230]],button_basis='fixture')
    monkeypatch.setattr(sr,'detect',lambda *args:np.asarray(rows))
    with pytest.raises(ValueError,match='Button ranges'):
        sr.mapping({'left':back,'back':back},'left',cfg)
