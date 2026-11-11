"""A real tiny source highlight must not dim a surrounding shadowed rail."""
import numpy as np
import pytest
from phone_texture import lighting


@pytest.mark.parametrize('sign', [1, -1])
def test_local_source_feature_gamut_limit_does_not_create_broad_halo(sign):
    e=lighting.engine('aluminum')
    src=np.zeros((60, 30, 3));src[...,0]=40 if sign==1 else 60
    src[30, 20, 0]=99 if sign==1 else 1
    src[10:15, 15, 0]-=3  # independent real scratch
    desired=src.copy();desired[...,0]+=sign*40
    safe=desired.copy();safe[30,20,0]=src[30,20,0]+sign*.5
    mask=np.ones(src.shape[:2],bool)
    prior,_=e.continuous_gamut_limits(desired,safe,src,mask,.6,follow_field=True)
    result,proof=e.continuous_gamut_limits(desired,safe,src,mask,.6,
        follow_field=True,follow_source_edges=True)
    assert abs(prior[20,20,0]-desired[20,20,0])>20
    old_halo=np.count_nonzero(np.abs(prior[...,0]-safe[...,0])>1)
    new_halo=np.count_nonzero(np.abs(result[...,0]-safe[...,0])>1)
    assert new_halo < old_halo/4
    assert abs(result[20,20,0]-desired[20,20,0])<5
    np.testing.assert_array_equal(result[30,20],safe[30,20])
    assert result[12,14,0]-result[12,15,0]==3
    assert proof['follows_source_edges']


def test_feature_aware_cap_still_obeys_material_barriers_and_canvas_edges():
    e=lighting.engine('aluminum')
    src=np.zeros((12,30,3));src[...,0]=50
    desired=src.copy();desired[...,0]+=20
    safe=desired.copy();safe[:,0,0]=50
    mask=np.ones((12,30),bool);mask[:,10]=False
    result,_=e.continuous_gamut_limits(desired,safe,src,mask,.6,
        follow_field=True,follow_source_edges=True)
    np.testing.assert_array_equal(result[:,10:],desired[:,10:])
    assert result[6,1,0]<desired[6,1,0]


def test_ordinary_source_grain_does_not_relax_cap_continuity():
    e=lighting.engine('aluminum')
    src=np.zeros((30,30,3));src[...,0]=50
    src[::2,::2,0]+=.2
    desired=src.copy();desired[...,0]+=20
    safe=desired.copy();safe[0,0,0]=51
    mask=np.ones((30,30),bool)
    old,_=e.continuous_gamut_limits(desired,safe,src,mask,.6,follow_field=True)
    new,_=e.continuous_gamut_limits(desired,safe,src,mask,.6,
        follow_field=True,follow_source_edges=True)
    np.testing.assert_array_equal(new,old)
