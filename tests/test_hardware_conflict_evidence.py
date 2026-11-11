"""Conflict reports describe source candidates without waiving any gate."""
import copy
import numpy as np
import pytest
from phone_texture.side_material import _select_hardware_candidates


def candidate(x=30,width=40,top=40,height=130,gap=0):
    mask=np.zeros((220,100),bool)
    mask[top:top+height,x:x+width]=True
    proof=dict(box=[x,top,x+width,top+height],pixels=int(mask.sum()),
        boundary_support=.98,edge_support=[.98]*4,gap_radius=gap,contour_index=gap)
    return mask,proof


def test_crossing_candidates_still_rejected_with_original_pixel_witnesses():
    a=candidate();b=candidate(x=33,gap=1)
    masks=[a[0].copy(),b[0].copy()];proofs=copy.deepcopy([a[1],b[1]])
    with pytest.raises(ValueError,match='Ambiguous crossing side hardware contours') as exc:
        _select_hardware_candidates([a,b],100)
    evidence=exc.value.hardware_contour_conflict
    assert evidence['meaning'].endswith('not_confirmed_image_defect')
    assert evidence['tolerance_px']==1.5
    assert evidence['incoming_candidate']['gap_radius']==1
    comparison=evidence['comparisons'][0]
    assert comparison['maximum_incoming_distance_px']==3
    assert comparison['incoming_outside_tolerance_pixels']>0
    assert all(b[0][y,x] and not a[0][y,x] for x,y in comparison['source_witnesses_xy'])
    for before,after in zip(masks,[a[0],b[0]]):np.testing.assert_array_equal(before,after)
    assert [a[1],b[1]]==proofs


def test_duplicate_and_nested_contours_keep_original_outer_mask_and_proof():
    outer=candidate();duplicate=(outer[0].copy(),dict(outer[1],gap_radius=1))
    inner=candidate(x=35,width=30,top=45,height=120,gap=2)
    result=_select_hardware_candidates([outer,duplicate,inner],100)
    assert len(result)==1
    assert result[0][0] is outer[0]
    assert result[0][1] is outer[1]


def test_crossing_disagreement_is_not_hidden_by_better_boundary_support():
    a=candidate();b=candidate(x=33,gap=1);b[1]['boundary_support']=1.
    with pytest.raises(ValueError,match='Ambiguous crossing'):
        _select_hardware_candidates([a,b],100)
