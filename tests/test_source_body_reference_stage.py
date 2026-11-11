"""Isolated controls, prepared for the root's global serialized test window."""
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from phone_texture import source_body_reference as sbr
from phone_texture import source_body_reference_applicability as app
from phone_texture import stage_processing as stages
from phone_texture.web_app import processing_state
from test_source_body_reference import fixture, sha


def change_alpha(back,item,edit):
    image=np.array(Image.open(back))
    edit(image[...,3])
    Image.fromarray(image).save(back)
    item['output_sha256']=sha(back)


def test_screen_detects_real_remote_alpha_and_records_scope(tmp_path):
    raw,back,item=fixture(tmp_path)
    audit=app.inspect(raw,back,item)
    assert audit['applicable'] is True
    assert audit['remote_alpha128_pixel_count']==1160
    assert audit['remote_alpha128_bbox_xyxy']==[90,192,110,250]
    assert audit['policy']['distance_raw_pixels']==12
    assert audit['policy']['new_applicability_policy'] is True
    assert audit['policy']['quality_certification'] is False


@pytest.mark.parametrize('tail_alpha',[0,127])
def test_no_remote_opaque_extension_keeps_old_reference(tmp_path,tail_alpha):
    raw,back,item=fixture(tmp_path)
    change_alpha(back,item,lambda alpha:alpha.__setitem__((slice(180,None),slice(None)),tail_alpha))
    audit=app.inspect(raw,back,item)
    assert audit['applicable'] is False
    assert audit['status']=='no_in_shaft_remote_extension_detected'


def test_both_raw_planes_must_be_exceeded(tmp_path):
    raw,back,item=fixture(tmp_path)
    item['geometry']['alpha_lower_rail']=[0.,183.]
    def edit(alpha):
        alpha[180:]=0
        alpha[192:195,90:110]=255
    change_alpha(back,item,edit)
    audit=app.inspect(raw,back,item)
    assert audit['applicable'] is False
    assert audit['mapped_planes']['rgb_remote']==[0.,191.5]
    assert audit['mapped_planes']['alpha_remote']==[0.,194.5]


def test_raw_offset_is_mapped_before_rotation(tmp_path):
    raw,back,item=fixture(tmp_path)
    item['geometry']['rotation_ccw_degrees']=3.
    audit=app.inspect(raw,back,item)
    rgb=audit['mapped_planes']['rgb'];remote=audit['mapped_planes']['rgb_remote']
    assert remote[0]==pytest.approx(rgb[0])
    assert remote[1]-rgb[1]==pytest.approx(12/math.cos(math.radians(3.)))
    assert abs(remote[1]-rgb[1]-12)>.01


def test_outside_shaft_is_explicitly_out_of_screen_scope(tmp_path):
    raw,back,item=fixture(tmp_path)
    def edit(alpha):
        alpha[180:]=0
        alpha[200:240,0:8]=255
    change_alpha(back,item,edit)
    audit=app.inspect(raw,back,item)
    assert audit['applicable'] is False
    assert 'outside-shaft' in audit['policy']['scope']
    assert audit['status']!='no_tail'


def test_missing_rail_is_unknown_but_wrong_source_is_hard(tmp_path):
    raw,back,item=fixture(tmp_path)
    item['geometry'].pop('bottom_rail')
    assert app.inspect(raw,back,item)['applicable'] is None
    raw.write_bytes(raw.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='identity'):
        app.inspect(raw,back,item)


def stage_fixture(tmp_path):
    raw,back,item=fixture(tmp_path)
    case=tmp_path/'case';records=case/'work/geometry-records';records.mkdir(parents=True)
    report=records/'report.json'
    report.write_text(json.dumps(dict(candidates_complete=True,images=[item])))
    metadata=dict(raw_path=raw,back_path=back,geometry_report_path=report,geometry_item=item)
    manifest=dict(skus={'fixture':{'back':dict(sha256=sha(raw),snapshot=str(raw))}})
    return case,{'back':back},manifest,metadata


@pytest.mark.parametrize('applicable',[False,None])
def test_non_applicable_or_unknown_does_not_build_or_warn(tmp_path,monkeypatch,applicable):
    args=stage_fixture(tmp_path)
    monkeypatch.setattr(app,'inspect',lambda *a:dict(applicable=applicable,status='audit_only'))
    monkeypatch.setattr(sbr,'build',lambda *a:pytest.fail('Must not build without source-only applicability'))
    options,audit,warnings=stages.prepare_source_body_reference(args[0],'fixture',*args[1:])
    assert options==dict(coordinate_mode='canvas_pixel_centers')
    assert warnings==[] and audit['reference_review_required'] is False


def test_tail_with_insufficient_reference_falls_back_with_truthful_warning(tmp_path,monkeypatch):
    args=stage_fixture(tmp_path)
    monkeypatch.setattr(app,'inspect',lambda *a:dict(applicable=True,status='source_remote_tail_detected'))
    def failed(*args):raise ValueError('top native witness unavailable')
    monkeypatch.setattr(sbr,'build',failed)
    options,audit,warnings=stages.prepare_source_body_reference(args[0],'fixture',*args[1:])
    assert options==dict(coordinate_mode='canvas_pixel_centers')
    assert audit['reference_supported'] is False and audit['reference_review_required'] is True
    assert warnings[0]['kind']=='source_body_reference_unavailable'
    assert 'top native witness unavailable' in warnings[0]['error']


def test_supported_reference_built_once_saved_and_still_review(tmp_path,monkeypatch):
    args=stage_fixture(tmp_path);original=sbr.build;calls=[]
    def counted(*a):
        calls.append(a)
        return original(*a)
    monkeypatch.setattr(sbr,'build',counted)
    options,audit,warnings=stages.prepare_source_body_reference(args[0],'fixture',*args[1:])
    assert len(calls)==1 and options['coordinate_mode']==sbr.MODE
    assert options['body_reference']['evidence_sha256']==audit['source_body_reference_sha256']
    assert sha(Path(audit['evidence_path']))==audit['evidence_file_sha256']
    assert audit['reference_supported'] is True and audit['reference_review_required'] is True
    assert audit['remap_verified'] is False and audit['publishable'] is False
    assert warnings[0]['kind']=='source_body_reference_review'


def test_metadata_wrong_back_row_is_not_silent_unknown(tmp_path):
    args=stage_fixture(tmp_path)
    args[-1]['geometry_item']['geometry']['bottom_rail']=[0.,150.]
    with pytest.raises(ValueError,match='report row differs'):
        stages.prepare_source_body_reference(args[0],'fixture',*args[1:])


def test_full_stages_reference_warning_is_not_brightness_or_retry():
    warning=dict(face='back',kind='source_body_reference_review',code='source_body_reference_review',
        message='天线参考采用本机源图机身边界，最终位置待核对')
    done=dict(six_faces_generated=True,six_faces_processed=False,
        status='full_stages_effect_warning_needs_review',
        stage_status=dict.fromkeys(('geometry','lighting','structure','seam'),'complete'),
        item_effect_warnings=[warning],review_warnings=[warning],lighting_effect_warnings=[])
    result=processing_state(done)
    assert result['all_stages_complete'] and result['review_only'] and not result['retry_required']
    assert result['stage_summary']=='调光、天线对齐和接缝处理已完成'
    assert [v['code'] for v in result['review_warnings']]==['source_body_reference_review']
    assert '提亮' not in str(result['review_reasons'])
