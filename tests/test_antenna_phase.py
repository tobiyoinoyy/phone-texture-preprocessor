import numpy as np
import pytest
from PIL import Image
from scipy.special import erf
from phone_texture.antenna_material import trace_boundaries, _refine_pair_phase


def test_continuous_source_edge_is_localized_below_integer_pixel_error():
    h,w=500,100;yy,xx=np.indices((h,w))
    upper=449.12+.006*xx;lower=458.12+.006*xx
    def step(y):return .5*(1+erf(y/(2**.5*.65)))
    signal=180-90*(step(yy-41.5)-step(yy-50.5))-90*(step(yy-upper)-step(yy-lower))
    a=np.full((h,w,4),255,np.uint8);a[...,:3]=np.rint(signal)[...,None]
    before=a.copy();lines,proof=trace_boundaries(Image.fromarray(a),[42,51,450,459])
    # Ground truth is an analytically sampled, sloping material boundary.
    # Its geometry is independent of the detector's gradient/peak fit.
    assert np.max(abs(lines[2,15:85]-upper[0,15:85]))<.16
    assert np.max(abs(lines[3,15:85]-lower[0,15:85]))<.16
    assert np.max(abs(lines[3]-lines[2]-9))<.1
    assert proof['bands'][0]['subpixel_phase'] is None
    assert proof['bands'][1]['subpixel_phase']['maximum_shift_px']<=.5
    np.testing.assert_array_equal(a,before)


def test_phase_cannot_supply_missing_or_rgb_only_support():
    gradient=np.zeros((30,3));gradient[8:11]=np.array([1,4,2])[:,None]
    gradient[18:21]=-np.array([1,4,2])[:,None]
    chosen=np.array([[9.,np.nan,9.],[19.,np.nan,19.]])
    result,proof=_refine_pair_phase(gradient,chosen,np.array([True,False,True]),np.array([1,0,0]))
    assert proof['columns']==[0]
    assert np.isnan(result[:,1]).all()
    np.testing.assert_array_equal(result[:,2],chosen[:,2])


@pytest.mark.parametrize('values',[(1,2,3),(2,2,2),(.1,.4,.2)])
def test_phase_rejects_nonpeak_flat_or_weak_pair(values):
    gradient=np.zeros((30,1));gradient[8:11,0]=values;gradient[18:21,0]=-np.array(values)
    chosen=np.array([[9.],[19.]])
    result,proof=_refine_pair_phase(gradient,chosen,np.array([True]),np.array([1]))
    np.testing.assert_array_equal(result,chosen)
    assert not proof['columns']
