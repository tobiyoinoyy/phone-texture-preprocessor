"""R10: rebuilt paper transitions and the small contour-support tolerance."""
import numpy as np
import pytest

from phone_texture.label_transition_migration import migrate_label_blends, rebuild_label_blends
from phone_texture.lighting import engine


def _masks():
    h, w = 40, 40
    yy, xx = np.indices((h, w))
    old_paper = (yy >= 10) & (yy < 20) & (xx >= 10) & (xx < 20)
    paper = (yy >= 10) & (yy < 22) & (xx >= 10) & (xx < 20)   # refined: 2 rows taller
    metal = ~old_paper
    before = {'main_surface': metal & ~old_paper, 'label_component_0': old_paper}
    after = {'main_surface': metal & ~paper, 'printed_label': paper,
             'camera_surround': np.zeros((h, w), bool)}
    blends = [dict(domains=['rear_perimeter', 'camera_surround', 'main_surface'],
                   boundary_domain='rear_perimeter', sigma=7.9),
              dict(domains=['label_component_0', 'main_surface'], boundary_domain='label_component_0',
                   sigma=1.25, kind='source_material_transition')]
    return blends, before, after


def test_exact_migration_still_rejects_a_refined_paper():
    blends, before, after = _masks()
    with pytest.raises(ValueError, match='Replacement paper extends'):
        migrate_label_blends(blends, before, after, ['label_component_0'])


def test_rebuild_keeps_unrelated_blends_and_reattaches_paper_to_actual_neighbours():
    blends, _, after = _masks()
    result, proof = rebuild_label_blends(blends, after, ['label_component_0'], 'exact failed')
    assert result[0] == blends[0]
    assert result[1:] == [dict(domains=['printed_label', 'main_surface'], boundary_domain='printed_label',
                               sigma=1.25, kind='source_material_transition')]
    assert proof['peers'] == ['main_surface'] and proof['exact_migration_error'] == 'exact failed'
    assert not any('label_component_0' in b['domains'] for b in result)


def _rim(unsupported):
    alpha = np.zeros((80, 40), np.uint8); alpha[5:75, 5:35] = 255
    editable = (alpha > 0) & (np.indices(alpha.shape)[1] < 12)
    fit = editable.copy()
    ys, xs = np.nonzero(editable)
    # Remove fit support around a few isolated pixels far from other samples.
    lab = np.zeros((80, 40, 3)); lab[:] = [65, 0, 0]
    return lab, fit, editable, alpha


def test_contour_field_tolerates_a_few_isolated_unsupported_pixels():
    e = engine('aluminum')
    lab, fit, editable, alpha = _rim(0)
    stray = np.zeros_like(editable); stray[40, 30] = True; stray[60, 30] = True
    editable = editable | stray   # isolated pixels with no nearby fit samples
    low, support, reliable = e.supported_contour_field(lab, fit, editable, alpha, [8, 1.2])
    assert not reliable[stray].any() and reliable[editable & ~stray].all()
    assert np.isfinite(low).all() and np.all(low[stray] == 0)


def test_contour_field_still_rejects_large_unsupported_regions():
    e = engine('aluminum')
    lab, fit, editable, alpha = _rim(0)
    with pytest.raises(ValueError, match='Unsupported editable contour illumination'):
        e.supported_contour_field(lab, np.zeros_like(fit), editable, alpha, [8, 1.2])


def _panel_image(weak_corners=()):
    from PIL import Image
    h, w = 1000, 480
    a = np.full((h, w, 4), 255, np.uint8); a[..., :3] = 200
    yy, xx = np.indices((h, w))
    left, right, top, bottom, r = 40, 440, 320, 960, 60
    inside = (xx >= left) & (xx <= right) & (yy >= top) & (yy <= bottom)
    centres = {0: (left+r, top+r), 1: (right-r, top+r), 2: (left+r, bottom-r), 3: (right-r, bottom-r)}
    for cx, cy in centres.values():
        corner = (np.abs(xx-cx) > 0) & (((xx < cx) if cx < w/2 else (xx > cx)) & ((yy < cy) if cy < h/2 else (yy > cy)))
        inside &= ~corner | ((xx-cx)**2+(yy-cy)**2 <= r*r)
    from scipy import ndimage
    seam = inside & ~ndimage.binary_erosion(inside, iterations=2)
    a[seam, :3] = 120
    for i in weak_corners:   # erase the dark seam on that corner arc only
        cx, cy = centres[i]
        # Fade every seam pixel near the bounding-box corner, so no radius
        # (including the large-radius search tail) can find the dark line.
        bx = right if cx > w/2 else left; by = bottom if cy > h/2 else top
        zone = (xx-bx)**2 + (yy-by)**2 <= (r + int(w*.2) + 4)**2
        a[zone & seam, :3] = 200
    return Image.fromarray(a)


def test_rear_panel_borrows_mirrored_radius_for_one_weak_corner():
    from phone_texture.auto_material import rear_panel
    _, full = rear_panel(_panel_image())
    _, proof = rear_panel(_panel_image(weak_corners=(3,)))
    assert proof['corners'][3]['radius_source'] == 'mirrored_same_row_corner'
    assert proof['corners'][3]['radius'] == proof['corners'][2]['radius']
    assert abs(proof['corners'][3]['radius'] - full['corners'][3]['radius']) <= 3


def test_rear_panel_still_rejects_two_weak_corners():
    from phone_texture.auto_material import rear_panel
    with pytest.raises(ValueError, match='Rear panel corner lacks source seam support'):
        rear_panel(_panel_image(weak_corners=(0, 3)))


def test_gaussian_field_tolerates_a_few_isolated_pixels_only():
    e = engine('aluminum')
    h, w = 60, 60
    lab = np.zeros((h, w, 3)); lab[..., 0] = 50
    fit = np.zeros((h, w), bool); fit[10:50, 10:50] = True
    stray = np.zeros((h, w), bool); stray[0, [0, 30, 59]] = True; stray[59, 30] = True
    low, _, reliable = e.supported_field(lab, fit, fit | stray, [1, 1])
    assert not reliable[stray].any() and np.isfinite(low).all()
    many = np.zeros((h, w), bool); many[0:2, :] = True
    with pytest.raises(ValueError, match='Unsupported editable illumination'):
        e.supported_field(lab, fit, fit | many, [1, 1])


def test_contour_field_tolerates_soft_fringe_beside_supported_rim():
    e = engine('aluminum')
    alpha = np.zeros((80, 40), np.uint8); alpha[5:75, 5:35] = 255
    editable = (alpha > 0) & (np.indices(alpha.shape)[1] < 12)
    fit = editable.copy()
    lab = np.zeros((80, 40, 3)); lab[:] = [65, 0, 0]
    ring = np.zeros_like(editable); ring[10:70, 4] = True
    alpha[ring] = 60
    fit_without_ring = fit & ~ring
    e.supported_contour_field(lab, fit_without_ring, editable | ring, alpha, [8, 1.2])


def test_back_perspective_bottom_fallback_only_after_primary_fails(monkeypatch):
    from phone_texture import face_perspective as fp
    calls=[]
    real=fp._rectify
    def spy(a,face,fallback):
        calls.append(fallback)
        if not fallback:raise fp.Rejected('reference_edge_not_straight')
        return None,dict(action='unchanged_taper_within_noise',bottom_seam='outline_inset_fallback')
    monkeypatch.setattr(fp,'_rectify',spy)
    _,proof=fp.rectify(np.zeros((10,10,4),np.uint8),'back')
    assert calls==[False,True] and proof['bottom_seam']=='outline_inset_fallback'
    calls.clear()
    with pytest.raises(fp.Rejected):
        fp.rectify(np.zeros((10,10,4),np.uint8),'front')
    assert calls==[False]
