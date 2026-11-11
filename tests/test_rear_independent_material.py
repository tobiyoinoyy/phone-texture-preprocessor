"""Rear material evidence does not certify or require side endpoint geometry."""
import numpy as np
import pytest
from PIL import Image
from phone_texture.auto_material import _independent_rear_material_reference
from phone_texture.geometry_review import digest


def rear_source(tmp_path):
    a=np.full((500,400,4),255,np.uint8);a[...,:3]=170
    a[42:54,:,:3]=65;a[450:462,:,:3]=65
    path=tmp_path/'back.png';Image.fromarray(a).save(path)
    return path,Image.open(path).convert('RGBA')


@pytest.mark.parametrize('side,edge',[('left',399),('right',0)])
def test_current_rear_three_window_consensus_is_bound_without_side_prior(tmp_path,side,edge):
    path,image=rear_source(tmp_path);before=path.read_bytes();pixels=image.tobytes()
    rows,columns,proof=_independent_rear_material_reference(image,side,edge,path)
    assert np.max(abs(np.array(rows)-[41.5,53.5,449.5,461.5]))<1
    assert proof['agreeing_columns']>=3
    assert proof['rear_source_sha256']==digest(path)
    assert proof['side_width_prior_used'] is False
    assert proof['structural_alignment_certified'] is False
    assert proof['material_reference_basis']=='current_rear_three_window_consensus'
    assert path.read_bytes()==before and image.tobytes()==pixels


def test_two_rear_windows_cannot_replace_required_three_window_consensus(tmp_path,monkeypatch):
    from phone_texture import auto_structure
    path,image=rear_source(tmp_path);original=auto_structure.antenna_rows;calls=[]
    def only_two_supported(image,columns,expected_width=None,require_full_span=False):
        calls.append((list(columns),expected_width))
        if len(calls)>2:raise ValueError('No supported antenna strip in both body ends')
        return original(image,columns,expected_width,require_full_span)
    monkeypatch.setattr(auto_structure,'antenna_rows',only_two_supported)
    with pytest.raises(ValueError,match='neighbouring-column consensus'):
        _independent_rear_material_reference(image,'left',399,path)
    assert len(calls)>=3 and all(prior is None for _,prior in calls)


def test_missing_rear_strip_is_not_inferred_from_other_views(tmp_path):
    path,image=rear_source(tmp_path);a=np.array(image);a[400:,:,:3]=170
    Image.fromarray(a).save(path);image=Image.open(path).convert('RGBA')
    with pytest.raises(ValueError,match='neighbouring-column consensus'):
        _independent_rear_material_reference(image,'left',399,path)
