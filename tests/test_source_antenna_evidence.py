import numpy as np
import pytest
from PIL import Image
from phone_texture.source_antenna_evidence import create,load


def fixture(tmp_path,face='left'):
    y,x=np.indices((100,100));rgb=np.stack([x*2,y*2,x+y],axis=-1).astype(np.uint8)
    source=tmp_path/'source.png';Image.fromarray(rgb).save(source)
    reference=Image.fromarray(rgb).crop((10,20,90,40)).transpose(Image.Transpose.ROTATE_90 if face=='left' else Image.Transpose.ROTATE_270)
    a=np.zeros((80,20,4),np.uint8);a[...,3]=250
    geometry_path=tmp_path/'geometry.png';Image.fromarray(a).save(geometry_path)
    spec=dict(source_size=[100,100],rotation_ccw_degrees=0,canvas_trim_xyxy=[10,20,90,40],
        visible_canvas_trim={'crop_xyxy':[0,0,20,80]})
    context=create(source,geometry_path,spec,face,tmp_path/'evidence.png')
    return context,geometry_path,np.array(reference)


@pytest.mark.parametrize('face',['left','right'])
def test_original_rgb_reference_has_exact_frozen_coordinates_and_alpha(tmp_path,face):
    ctx,path,expected=fixture(tmp_path,face);before=path.read_bytes()
    a=np.array(load(ctx,path,face))
    np.testing.assert_array_equal(a[...,:3],expected)
    np.testing.assert_array_equal(a[...,3],250)
    assert path.read_bytes()==before


@pytest.mark.parametrize('target',['raw_source','path'])
def test_changed_source_or_evidence_rejected(tmp_path,target):
    ctx,path,_=fixture(tmp_path);Image.new('RGBA',(20,80),(1,2,3,250)).save(ctx[target])
    with pytest.raises(ValueError,match='changed'):load(ctx,path,'left')


def test_changed_geometry_and_wrong_face_rejected(tmp_path):
    ctx,path,_=fixture(tmp_path)
    with pytest.raises(ValueError,match='another face'):load(ctx,path,'right')
    Image.new('RGBA',(20,80),(0,0,0,249)).save(path)
    with pytest.raises(ValueError,match='changed'):load(ctx,path,'left')


def test_source_size_mismatch_rejected(tmp_path):
    ctx,path,_=fixture(tmp_path);spec=dict(ctx['geometry'],source_size=[99,100])
    with pytest.raises(ValueError,match='dimensions'):create(ctx['raw_source'],path,spec,'left',tmp_path/'bad.png')


@pytest.mark.parametrize('source_width',[9,4,0])
def test_original_evidence_requires_its_own_pair_and_agrees_with_central_geometry(tmp_path,source_width):
    from phone_texture.auto_structure import source_antenna_rows
    actual=np.full((500,100,4),255,np.uint8);actual[...,:3]=180
    actual[42:51,:,:3]=90;actual[450:459,:,:3]=90
    original=actual[...,:3].copy();original[:]=180
    original[42:42+source_width]=90;original[450:450+source_width]=90
    raw=Image.fromarray(original).transpose(Image.Transpose.ROTATE_270)
    source=tmp_path/'raw.png';raw.save(source)
    actual[:,:12,:3]=80;geometry_path=tmp_path/'geometry.png';Image.fromarray(actual).save(geometry_path)
    spec=dict(source_size=[500,100],rotation_ccw_degrees=0,canvas_trim_xyxy=[0,0,500,100],
        visible_canvas_trim={'crop_xyxy':[0,0,100,500]})
    ctx=create(source,geometry_path,spec,'left',tmp_path/'evidence.png')
    if source_width==9:
        detection,rows,_,_,used=source_antenna_rows(Image.fromarray(actual),[40,60],geometry_path,ctx,'left')
        assert used==ctx and rows==[41,50,449,458]
    else:
        with pytest.raises(ValueError):source_antenna_rows(Image.fromarray(actual),[40,60],geometry_path,ctx,'left')
