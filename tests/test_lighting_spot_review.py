import numpy as np
import pytest

from phone_texture.lighting_spot_review import camera_arc, side_rail


@pytest.mark.parametrize("vertical", [True, False])
def test_rail_releases_only_supported_excess_shadow(vertical):
    y, x = np.ogrid[:90, :720]
    alpha = np.full((90, 720), 255, np.uint8)
    main = np.ones_like(alpha, bool)
    main[4:12, 450:550] = False  # Independently classified hardware.
    source = np.full((90,720,3), 44.)
    safe = source.copy()
    safe[...,0] += 10
    limited = safe.copy()
    dark = (x-320)**2/20**2 + (y-6)**2/5**2 < 1
    limited[...,0] -= dark*19
    limited[~main,0] -= 8
    before = limited.copy()
    def view(a):
        return np.swapaxes(a,0,1) if vertical else a
    checks = side_rail(*(view(a) for a in (source,limited,safe,alpha,main)))
    assert checks[0]['hits_before'] > checks[0]['hits_after']
    assert checks[1]['status'] == 'clear'
    assert abs(limited[6,320,0]-safe[6,320,0]) < 3
    np.testing.assert_array_equal(limited[~dark],before[~dark])
    np.testing.assert_array_equal(limited[...,1:], before[...,1:])


def test_existing_shadow_and_uniform_light_pass_without_pixel_changes():
    source = np.full((50,360,3),40.)
    source[:9,126:145,0] -= 15
    current = source + [9,0,0]
    before = current.copy()
    ones = np.ones(source.shape[:2],bool)
    proof = side_rail(source,current,current.copy(),ones.astype('uint8')*255,ones)
    assert all(x['status']=='clear' for x in proof)
    np.testing.assert_array_equal(current,before)


@pytest.mark.parametrize('zone', ['ring','exterior'])
def test_camera_spot_triggers_on_new_contrast_and_preserves_other_material(zone):
    y,x=np.ogrid[:240,:240]
    radii=np.hypot(y-120,x-120)
    angle=np.arctan2(y-120,x-120)
    lens=dict(centre=[120,120,60],inner=[60]*360,outer=[72]*360,supported=[True]*360)
    mask=(radii>60)&(radii<72) if zone=='ring' else radii>74
    rr=66 if zone=='ring' else 79
    spot=mask & (abs(angle)<.10) & (abs(radii-rr)<3.5)
    raw=np.full((240,240,3),32.)
    candidate=raw+[5.,0,0]
    candidate[spot]+=[-12,8,1]
    previous=candidate.copy()
    proof=camera_arc(raw,candidate,mask,lens,zone)
    assert proof['hits_before']>proof['hits_after']
    assert abs(candidate[120,120+rr,0]-37)<4
    assert abs(candidate[120,120+rr,1]-32)<4
    np.testing.assert_array_equal(candidate[~mask],previous[~mask])
    # Already levelled source shadow is valid and must not be restored.
    raw[spot,0]-=10
    clean=np.full_like(candidate,36.)
    expected=clean.copy()
    assert camera_arc(raw,clean,mask,lens,zone)['status']=='clear'
    np.testing.assert_array_equal(clean,expected)
