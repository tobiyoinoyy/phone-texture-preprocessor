import numpy as np
from phone_texture.auto_material import rear_camera_glass


def scene():
    a=np.full((600,400,4),255,np.uint8);a[...,:3]=180
    yy,xx=np.indices(a.shape[:2])
    lens=(xx-90)**2+(yy-85)**2<=30**2
    a[lens,:3]=20
    return a,lens


def test_interior_dark_lens_and_specular_hole_remain_glass():
    a,lens=scene();a[80:85,90:95,:3]=230
    before=a.copy();mask,proof=rear_camera_glass(a,a[...,3]>0)
    np.testing.assert_array_equal(mask,lens)
    np.testing.assert_array_equal(a,before)
    assert len(proof['accepted'])==1 and not proof['rejected_outer_fragments']


def test_round_bounding_box_of_outer_arc_is_not_camera_glass():
    a,lens=scene();yy,xx=np.indices(a.shape[:2])
    # A curved body silhouette can satisfy the historical lens box criteria.
    outside=(xx-315)**2+(yy-80)**2>65**2
    corner=(xx>=315)&(yy<=80)
    a[corner&outside,3]=0
    arc=corner&~outside&((xx-315)**2+(yy-80)**2>59**2)
    a[arc,:3]=15
    mask,proof=rear_camera_glass(a,a[...,3]>0)
    assert not np.any(mask&arc)
    assert np.all(mask[lens])
    assert proof['rejected_outer_fragments']


def test_soft_alpha_fragment_is_not_an_interior_camera():
    a,lens=scene();a[40:90,300:335,:3]=20;a[40:90,334,3]=200
    mask,proof=rear_camera_glass(a,a[...,3]>0)
    assert not mask[40:90,300:335].any()
    assert np.all(mask[lens])
    assert proof['rejected_outer_fragments'][0]['outer_neighbourhood_pixels']>0


def test_opaque_component_next_to_transparency_also_rejected():
    a,lens=scene();a[40:90,300:335,:3]=20;a[40:90,335,3]=0
    mask,proof=rear_camera_glass(a,a[...,3]>0)
    assert not mask[40:90,300:335].any()
    assert np.all(mask[lens])
    assert proof['rejected_outer_fragments']


def test_lens_outside_remaining_material_not_reintroduced():
    a,lens=scene();remaining=a[...,3]>0;remaining[lens]=False
    mask,_=rear_camera_glass(a,remaining)
    assert not mask.any()


def test_thin_closed_shadow_ring_is_not_filled_into_a_glass_disk():
    a,lens=scene();yy,xx=np.indices(a.shape[:2])
    radius=(xx-90)**2+(yy-85)**2
    ring=(radius>=38**2)&(radius<=41**2)
    a[ring,:3]=15
    mask,proof=rear_camera_glass(a,a[...,3]>0)
    np.testing.assert_array_equal(mask,lens)
    assert proof['rejected_shadow_arcs']


def test_separate_cast_shadow_crescent_stays_editable():
    a,lens=scene();yy,xx=np.indices(a.shape[:2])
    radius=(xx-90)**2+(yy-85)**2
    shadow=(radius>=33**2)&(radius<=42**2)&(xx>95)
    a[shadow,:3]=15
    mask,_=rear_camera_glass(a,a[...,3]>0)
    assert not mask[shadow].any()
    assert mask[lens].all()
