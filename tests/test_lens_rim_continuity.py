import numpy as np

from phone_texture.lens_rim_continuity import review


def fixture():
    y, x = np.ogrid[:270,:270]
    r = np.hypot(x-135,y-135)
    theta = np.arctan2(y-135,x-135)
    metal = (r > 100) & (r < 125)
    inner = 24*np.exp(-.5*((r-110)/.9)**2)
    outer = 20*np.exp(-.5*((r-115)/.9)**2)
    image = np.zeros((270,270,3))
    image[...,0] = 44 - inner - outer
    lens = dict(centre=[135,135,98],outer=np.where(np.arange(360)<315,109,114))
    return image, lens, metal, r, theta, inner, outer


def test_two_alternating_lines_keep_their_own_source_contrast():
    raw,lens,metal,r,theta,first,second = fixture()
    candidate = raw + [7,0,0]
    sector = (theta > -1.05) & (theta < -.48)
    candidate[...,0] += np.where(theta < -.77,second,first)*sector*.85
    before = candidate.copy()
    proof = review(raw,candidate,metal,lens,metal & (r > 111))
    assert proof['weak_angles_before'] > 15
    assert proof['weak_angles_after'] < proof['weak_angles_before'] / 4
    for radius in [110,115]:
        local = sector & (abs(r-radius)<.4)
        assert np.median(abs(candidate[...,0][local]-raw[...,0][local]-7)) < 1.5
    np.testing.assert_array_equal(candidate[~metal],before[~metal])
    np.testing.assert_array_equal(candidate[...,1:],before[...,1:])


def test_real_gap_and_intact_double_rim_are_not_reconstructed():
    raw,lens,metal,r,theta,first,second = fixture()
    raw[...,0] += second*(abs(theta)<.35)  # This photographed arc really has a single line.
    candidate = raw + [5,1,0]
    before = candidate.copy()
    assert review(raw,candidate,metal,lens)['repaired_pixels'] == 0
    np.testing.assert_array_equal(candidate,before)
    candidate[...,0] += first*(abs(theta-.7)<.20)*.85
    review(raw,candidate,metal,lens)
    np.testing.assert_array_equal(candidate[abs(theta)<.20],before[abs(theta)<.20])


def test_already_lifted_shadow_on_connected_metal_stays_lifted():
    raw, lens, metal, radius, theta, inner, outer = fixture()
    # A shallow shadow runs outside a stronger, continuous physical seam.
    raw[..., 0] += .50 * outer
    lens['outer'] = np.full(360, 109.)
    candidate = raw + [4, 0, 0]
    candidate[..., 0] += .50 * outer * (abs(theta) < .45)
    before = candidate.copy()
    metal = (radius > 100) & (radius < 130)
    flat = metal & (radius > 112)
    result = review(raw, candidate, metal, lens, flat)
    assert result['excluded_shadow_angles'] > 20
    marked = (abs(theta) < .32) & (abs(radius - 115) < 2)
    np.testing.assert_array_equal(candidate[marked], before[marked])
    # If a bevel cuts off the reference, it cannot justify a shadow exclusion.
    partial = before.copy()
    uncertain = review(raw, partial, metal, lens, flat & (radius < 117))
    assert uncertain['excluded_shadow_angles'] == 0
