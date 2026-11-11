"""Adjacent source colour can prove both edges when their RGB mean cancels."""
import numpy as np
import pytest
from PIL import Image

from phone_texture.antenna_material import trace_boundaries


def photo():
    a=np.full((500,100,4),255,np.uint8)
    a[...,:3]=180
    a[42:51,:,:3]=90
    a[450:459,:,:3]=90
    return a


def test_consistent_material_colour_survives_mean_rgb_cancellation():
    a=photo()
    for x in range(85,100):
        weight=(x-85)/14
        metal=np.rint((1-weight)*np.array([180,180,180])+weight*np.array([120,60,90]))
        a[:,x,:3]=metal
        a[42:51,x,:3]=90
        a[450:459,x,:3]=90
    # The last column is genuinely invisible in mean RGB, but its material
    # colour direction is continuous with the adjacent source columns.
    assert np.mean(a[41,99,:3])==np.mean(a[45,99,:3])
    lines,proof=trace_boundaries(Image.fromarray(a),[42,51,450,459])
    assert abs(lines[0,-1]-41.5)<1.5
    assert abs(lines[3,-1]-458.5)<1.5
    for band in proof['bands']:
        evidence=next(e for e in band['rgb_continuation'] if e['column']==99)
        assert min(evidence['material_contrasts'])>=1
        assert evidence['projected_gradients'][0]>=.5
        assert evidence['projected_gradients'][1]<=-.5


def test_channel_noise_cannot_supply_an_unrelated_material_direction():
    a=photo();a[:,99,:3]=100
    a[42:51,99,:3]=[130,70,100]
    a[450:459,99,:3]=[130,70,100]
    # The noisy column is never measured from its channel noise; it is only
    # the bounded end-cap extension holding the last supported pair.
    lines,proof=trace_boundaries(Image.fromarray(a),[42,51,450,459])
    for band in proof['bands']:
        assert 99 in band['endpoint_extended_columns']
        assert all(e['column']!=99 for e in band['rgb_continuation'])
    np.testing.assert_allclose(lines[:,99],lines[:,98],atol=.5)


def test_curved_endpoint_follows_measured_pairs_without_global_window_widening():
    a=photo()
    for x in range(96,100):
        shift=(x-95)*2;a[:,x,:3]=180
        a[42+shift:51+shift,x,:3]=90
        a[450+shift:459+shift,x,:3]=90
    lines,proof=trace_boundaries(Image.fromarray(a),[42,51,450,459])
    assert lines[0,-1]>47
    for band in proof['bands']:
        endpoint=next(e for e in band['rgb_continuation'] if e['column']==99)
        assert 98 in endpoint['reference_columns']
        assert endpoint['rows'][1]-endpoint['rows'][0]==9


def test_continuation_never_jumps_over_a_missing_pair_to_a_distant_edge():
    a=photo();a[:,98,:3]=80
    # Column 99 still has two strong edges, but no measured pair connects it
    # through column 98 to the body; making a bridge would be extrapolation.
    a[:,99,:3]=180;a[48:57,99,:3]=90;a[456:465,99,:3]=90
    lines,proof=trace_boundaries(Image.fromarray(a),[42,51,450,459])
    # Columns 98-99 are held at the last measured pair, never bridged to the
    # displaced edges at rows 48/456.
    assert all({98,99}<=set(band['endpoint_extended_columns']) for band in proof['bands'])
    assert abs(lines[0,99]-41.5)<1.5 and abs(lines[2,99]-449.5)<1.5
