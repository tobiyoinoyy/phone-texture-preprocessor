"""Only uniquely certified source material can resolve an old contour clash."""
import copy
import numpy as np
import pytest
from PIL import Image
from phone_texture import side_material as sm
from test_side_hardware_proposals import shape,canvas


def candidate(mask,box):
    return mask,dict(box=box,pixels=int(mask.sum()),boundary_support=1.,
        edge_support=[1.]*4,gap_radius=0)


def conflict_fixture():
    outer=candidate(shape(),[35,1100,90,1350])
    shifted=candidate(shape(x=38),[38,1100,93,1350])
    upper=candidate(shape(x=42,y=450,bw=41,bh=220),[42,450,83,670])
    image=Image.fromarray(canvas())
    with pytest.raises(ValueError,match='crossing') as error:
        sm._select_hardware_candidates([outer,shifted,upper],image.width)
    return image,[outer,shifted,upper],error.value


def test_uniquely_resolved_group_keeps_other_mask_and_proof_identity(monkeypatch):
    image,candidates,error=conflict_fixture();upper=candidates[2]
    originals=[(mask.copy(),copy.deepcopy(proof)) for mask,proof in candidates]
    before=image.tobytes();chosen=candidates[0][0].copy()
    accepted=dict(candidates[0][1],proposal_method='independent_signed_source_rails_and_caps_v1',
        accepted=True,independent_material={'accepted':True})
    def independent(source,face,existing=()):
        assert len(existing)==1 and existing[0][0] is upper[0] and existing[0][1] is upper[1]
        return [(chosen,accepted)],[accepted]
    monkeypatch.setattr(sm,'_independent_hardware_proposals',independent)
    monkeypatch.setattr(sm,'_hardware_material_step',lambda *args:{'accepted':False})
    result=sm._resolve_hardware_conflicts(image,'right',candidates,error)
    assert len(result)==2
    unchanged=next(item for item in result if item[0] is upper[0])
    assert unchanged[1] is upper[1]
    replacement=next(item for item in result if item[0] is chosen)
    audit=replacement[1]['hardware_conflict_resolution']
    assert audit['accepted'] and audit['acceptance_gates_unchanged']
    assert audit['original_conflict']==error.hardware_contour_conflict
    assert 'hardware_conflict_resolution' not in accepted
    for (mask,proof),(old_mask,old_proof) in zip(candidates,originals):
        np.testing.assert_array_equal(mask,old_mask);assert proof==old_proof
    assert image.tobytes()==before


def test_independently_certified_legacy_material_vetoes_different_boundary(monkeypatch):
    image,candidates,error=conflict_fixture()
    monkeypatch.setattr(sm,'_independent_hardware_proposals',
        lambda *args,**kwargs:([candidates[0]],[candidates[0][1]]))
    monkeypatch.setattr(sm,'_hardware_material_step',lambda *args:{'accepted':True})
    with pytest.raises(ValueError,match='crossing') as rejection:
        sm._resolve_hardware_conflicts(image,'right',candidates,error)
    review=rejection.value.hardware_contour_conflict['independent_disambiguation']
    assert 'independent material evidence' in review['reason']
    assert review['competing_legacy_indices']==[1]


def test_three_candidate_bridge_cannot_hide_conflicting_extremes(monkeypatch):
    image,candidates,error=conflict_fixture()
    independent=[candidate(shape(x=x),[x,1100,x+55,1350]) for x in (35,34,36)]
    monkeypatch.setattr(sm,'_independent_hardware_proposals',
        lambda *args,**kwargs:(independent,[proof for _,proof in independent]))
    monkeypatch.setattr(sm,'_hardware_material_step',lambda *args:{'accepted':False})
    with pytest.raises(ValueError,match='crossing') as rejection:
        sm._resolve_hardware_conflicts(image,'right',candidates,error)
    assert 'Competing independently' in rejection.value.hardware_contour_conflict['independent_disambiguation']['reason']


def test_no_independent_candidate_does_not_silently_keep_largest_hull(monkeypatch):
    image,candidates,error=conflict_fixture()
    monkeypatch.setattr(sm,'_independent_hardware_proposals',lambda *args,**kwargs:([],[]))
    with pytest.raises(ValueError,match='crossing') as rejection:
        sm._resolve_hardware_conflicts(image,'right',candidates,error)
    assert rejection.value.hardware_contour_conflict['independent_disambiguation']['independent_distinct_count']==0


def test_noise_extended_source_bbox_does_not_hide_supported_cap():
    a=canvas();outer=shape();a[outer,:3]=[42,52,67]
    # A narrow off-center source branch extends the contour box >0.6 widths
    # below the control, while the actual central cap remains visible.
    a[1340:1394,75:79,:3]=[42,52,67]
    image=Image.fromarray(a);before=image.tobytes()
    proposals,_=sm._independent_hardware_proposals(image,'right')
    unique=sm._distinct_hardware_proposals(proposals,1.875)
    assert len(unique)==1
    mask,proof=unique[0]
    assert np.count_nonzero(mask&outer)/np.count_nonzero(mask|outer)>.95
    assert proof['independent_material']['accepted']
    assert proof['boundary_support']>=.95 and min(proof['edge_support'])>=.75
    assert proof['source_component_iou']>=.90
    assert image.tobytes()==before


@pytest.mark.parametrize('kind',['missing_cap','rectangle','same_material_core'])
def test_wider_cap_search_does_not_accept_missing_or_false_material(kind):
    a=canvas();outer=shape();a[outer,:3]=[42,52,67]
    a[1340:1394,75:79,:3]=[42,52,67]
    if kind=='missing_cap':a[1090:1125,45:80,:3]=[145,165,195]
    elif kind=='rectangle':
        a=canvas();a[1100:1350,35:90,:3]=[42,52,67]
    else:
        core=shape(x=45,y=1110,bw=35,bh=230)
        a[core,:3]=[145,165,195]
    proposals,_=sm._independent_hardware_proposals(Image.fromarray(a),'right')
    assert proposals==[]
