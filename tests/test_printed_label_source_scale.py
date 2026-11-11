"""Source-support behavior for an independently fitted narrow paper strip."""
import numpy as np
import pytest
from PIL import Image
from phone_texture import lighting
from phone_texture import shadow_lift_profile as profile


def paper_source():
    a = np.full((600, 600, 4), 255, np.uint8)
    a[..., :3] = [60, 100, 180]  # another material must not enter paper fit
    m = np.zeros(a.shape[:2], bool)
    m[100:400, 240:320] = True
    m[400, 220:340] = True  # connected soft paper terminal wider than opaque core
    yy, xx = np.indices(m.shape)
    a[m, :3] = (175 + yy[m, None]*.04).astype(np.uint8)
    a[400, 220:340, 3] = 100
    a[180:230, 270:274, :3] -= 65  # source print/scratch detail
    return a, m


def render(tmp_path, a, m, sigma):
    H = lighting.engine('aluminum')
    src, mask, fit = [tmp_path/(name+'.png') for name in ('source', 'mask', 'fit')]
    Image.fromarray(a).save(src)
    Image.fromarray(m.astype('uint8')*255).save(mask)
    Image.fromarray((m & (a[..., 3] >= 240)).astype('uint8')*255).save(fit)
    view = dict(source=str(src), protection_policy='none', reference_boxes=[[245, 110, 252, 117]],
        domains=[dict(name='printed_label', mask=str(mask), fit_mask=str(fit), sigma=sigma,
            reference_boxes=[[245, 110, 252, 117]], strength=[.8, 0, 0],
            max_lab_shift=[6, 0, 0], feather=0, correction_smooth=1.5,
            gamut_continuity_step_L=.2, gamut_continuity_follow_field=True)])
    target, _ = H.target_color({'views': [view]})
    return H.process_view(view, target, {'color_mode': 'source_lightness'})


def test_source_cross_section_resolves_connected_terminal_without_borrowing_material(tmp_path):
    a, m = paper_source()
    with pytest.raises(ValueError, match='Unsupported editable illumination'):
        render(tmp_path, a, m, [8, 4.2])
    sigma, proof = profile.printed_label_sigma(a, m, 'back')
    b, allowed, protected, info = render(tmp_path, a, m, sigma)
    assert proof['support_floor'] == .008
    assert info['domains'][0]['minimum_editable_support'] >= .008
    assert info['domains'][0]['editable_pixels'] == int(m.sum())
    assert not protected.any()
    np.testing.assert_array_equal(a[..., 3], b[..., 3])
    np.testing.assert_array_equal(a[~m], b[~m])
    assert np.any(b[400, 220:340, :3] != a[400, 220:340, :3])
    L = lighting.engine('aluminum').rgb_to_lab(b[..., :3])[..., 0]
    assert np.median(L[180:230, 270:274]) < np.median(L[180:230, 276:280])-20
    foreign = a.copy(); foreign[~m, :3] = [220, 40, 70]
    c, _, _, _ = render(tmp_path, foreign, m, sigma)
    np.testing.assert_array_equal(c[m], b[m])


@pytest.mark.parametrize('kind', ['disconnected', 'long_connected_tail', 'zero_opaque_fit', 'soft_bridge'])
def test_true_unrepresented_paper_remains_rejected(tmp_path, kind):
    a, m = paper_source()
    if kind == 'disconnected':
        m[20:25, 20:25] = True; a[20:25, 20:25, 3] = 100
    elif kind == 'long_connected_tail':
        m[400, 320:490] = True; a[400, 320:490, 3] = 100
    elif kind == 'soft_bridge':
        a[200:260, 240:320, 3] = 100
    else:
        a[m, 3] = 100
    sigma, _ = profile.printed_label_sigma(a, m, 'back')
    with pytest.raises(ValueError, match='Unsupported editable illumination|Insufficient reviewed material samples|Empty reference box'):
        render(tmp_path, a, m, sigma)


def test_other_faces_and_ambiguous_shape_keep_original_scale():
    a, m = paper_source()
    for face in ('front', 'top', 'bottom', 'left', 'right'):
        assert profile.printed_label_sigma(a, m, face)[0] == [8, 4.2]
    horizontal = np.zeros(m.shape, bool); horizontal[100:180, 100:500] = True
    assert profile.printed_label_sigma(a, horizontal, 'back')[0] == [8, 4.2]
