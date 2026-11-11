"""RGB rounding can preserve a continuous field without filtering texture."""
import numpy as np
import pytest
from phone_texture import lighting
from phone_texture.lighting_review import lightness


def test_physical_srgb_white_is_exactly_100_L():
 value=lightness(np.array([[0,0,0],[254,254,254],[255,255,255]],np.uint8))
 assert value[0]==0 and value[-1]==100 and value[0]<value[1]<value[2]


def test_adjacent_corner_quantization_retains_fitted_field_and_source():
 e=lighting.engine('aluminum');a=np.full((8,20,3),180,np.uint8)
 lab=e.rgb_to_lab(a);lab[...,0]+=np.arange(8)[:,None]*1.24
 before=lab.copy();source=a.copy();mask=np.ones(a.shape[:2],bool);ordinary=e.lab_to_rgb(lab)
 result,proof=e.quantize_continuous_illumination(lab,a,mask,1.25)
 assert abs(np.diff(lightness(ordinary)-lightness(a),axis=0)).max()>1.25
 assert abs(np.diff(lightness(result)-lightness(a),axis=0)).max()<=1.25
 assert abs(result.astype(int)-ordinary.astype(int)).max()<=1
 np.testing.assert_array_equal(lab,before);np.testing.assert_array_equal(a,source)
 assert not proof['float_field_changed'] and proof['maximum_RGB_code_change']==1


def test_unlisted_pixels_remain_exact_and_material_holes_do_not_link():
 e=lighting.engine('aluminum');a=np.full((9,20,3),180,np.uint8)
 lab=e.rgb_to_lab(a);lab[...,0]+=np.arange(9)[:,None]*1.24
 mask=np.ones(a.shape[:2],bool);mask[3:6]=False;mask[:,10:]=False
 result,_=e.quantize_continuous_illumination(lab,a,mask,1.25)
 np.testing.assert_array_equal(result[~mask],e.lab_to_rgb(lab)[~mask])


def test_unrepresentable_step_is_never_repaired_by_altering_float_field():
 e=lighting.engine('aluminum');a=np.full((8,20,3),180,np.uint8)
 lab=e.rgb_to_lab(a);lab[4:,:,0]+=4;before=lab.copy()
 result,proof=e.quantize_continuous_illumination(lab,a,np.ones(a.shape[:2],bool),1.25)
 assert abs(np.diff(lightness(result)-lightness(a),axis=0)).max()>1.25
 assert proof['infeasible_restarts']>0
 np.testing.assert_array_equal(lab,before)


@pytest.mark.parametrize('step',[0,-1,float('nan'),6])
def test_invalid_quantization_budget_rejected(step):
 e=lighting.engine('aluminum');a=np.full((8,20,3),180,np.uint8)
 with pytest.raises(ValueError):e.quantize_continuous_illumination(e.rgb_to_lab(a),a,np.ones(a.shape[:2],bool),step)


def test_zero_correction_roundtrips_the_source_rgb():
 e=lighting.engine('aluminum');rng=np.random.default_rng(923);a=rng.integers(0,256,(18,20,3),dtype=np.uint8)
 result,proof=e.quantize_continuous_illumination(e.rgb_to_lab(a),a,np.ones(a.shape[:2],bool),1.25)
 np.testing.assert_array_equal(result,a);assert proof['changed_pixels']==0
