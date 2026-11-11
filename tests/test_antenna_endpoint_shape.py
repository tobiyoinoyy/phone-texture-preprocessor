"""Weak source edges require a strip profile; noise peaks are insufficient."""
import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from phone_texture.antenna_material import trace_boundaries


def base():
    a=np.full((500,100,4),255,np.uint8)
    a[...,:3]=180
    a[42:51,:,:3]=90
    a[450:459,:,:3]=90
    return a


def test_quantized_soft_endpoint_requires_independent_original_evidence():
    a=base();weak=np.full((500,12,3),180.)
    weak[42:51]=178;weak[450:459]=178
    a[:,:12,:3]=np.rint(gaussian_filter1d(weak,2,axis=0)).astype(np.uint8)
    before=a.copy()
    with pytest.raises(ValueError,match='endpoint.*evidence'):
        trace_boundaries(Image.fromarray(a),[42,51,450,459])
    np.testing.assert_array_equal(a,before)


@pytest.mark.parametrize('seed',[3,4,5,7,8,9,11,13,16,17])
def test_endpoint_noise_does_not_become_a_source_strip(seed):
    a=base();rng=np.random.default_rng(seed)
    a[:,:8,:3]=np.clip(np.linspace(60,85,500)[:,None,None]+rng.normal(0,4,(500,8,1)),0,255).astype(np.uint8)
    # Noise is never measured as a strip: it is rejected, or (R10, dark end
    # cap within the bound) every noisy column only holds the last real pair.
    try:
        lines,proof=trace_boundaries(Image.fromarray(a),[42,51,450,459])
    except ValueError:
        return
    for band in proof['bands']:
        assert set(range(8))<=set(band['endpoint_extended_columns'])
    np.testing.assert_allclose(lines[:,:8],lines[:,[8]].repeat(8,axis=1),atol=1.)
