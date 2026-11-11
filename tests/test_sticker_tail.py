"""Shadowed printed paper remains paper; adjacent metal never becomes a box."""
import numpy as np
import pytest
from PIL import Image

from phone_texture.auto_material import paper_mask,source_paper_tail


def label_photo(slope=0.,end=600):
    h,w=600,500;a=np.full((h,w,4),255,np.uint8)
    shade=np.clip((np.arange(h)-565)/35,0,1)
    paper=225-105*shade
    a[...,:3]=np.array([90,110,160])[None,None,:]*(1-.5*shade[:,None,None])
    actual=np.zeros((h,w),bool);actual[250:400,220:280]=True
    for y in range(400,end):
        shift=int(round((y-400)*slope));actual[y,230+shift:270+shift]=True
    a[:,:,:3][actual]=np.repeat(paper[:,None],w,axis=1)[actual,None]
    for y in range(330,385,9):
        for x in range(230,270,9):a[y:y+4,x:x+4,:3]=40
    for y in range(410,end):
        shift=int(round((y-400)*slope))
        if y%9<7:
            a[y,238+shift:243+shift,:3]=30
            a[y,258+shift:262+shift,:3]=30
    return a,actual


@pytest.mark.parametrize('slope',[0.,.03])
def test_shadowed_tail_is_recovered_from_measured_sides_without_freezing_metal(slope):
    a,actual=label_photo(slope);before=a.copy();mask,proof=paper_mask(Image.fromarray(a),'back')
    # Interior blank paper, as well as dark printing, must be protected.
    for y in range(580,599):
        shift=int(round((y-400)*slope))
        assert mask[y,245+shift:255+shift].all()
        assert not mask[y,220+shift:226+shift].any()
        assert not mask[y,274+shift:280+shift].any()
    assert proof['source_paper_tail']['terminal_printed_rows']>=8
    np.testing.assert_array_equal(a,before)


def test_a_short_label_cannot_extrapolate_a_protected_rectangle_to_canvas_end():
    a,actual=label_photo(end=581)
    # Simulate an already-resolved seed; the real remaining bottom is metal.
    mask,proof=source_paper_tail(a,actual,{})
    np.testing.assert_array_equal(mask,actual)
    assert not mask[581:].any()
    assert proof['source_paper_tail']['status']=='source_closed_label'


def test_transparent_pixels_never_become_tail_protection():
    a,actual=label_photo();a[-2:,:,3]=0
    mask,_=paper_mask(Image.fromarray(a),'back')
    assert not mask[-2:].any()


def test_a_truncated_seed_is_not_evidence_that_the_source_paper_ends():
    a,actual=label_photo();seed=actual.copy();seed[581:]=False
    mask,proof=source_paper_tail(a,seed,{})
    assert mask[585:599,245:255].all()
    assert proof['source_paper_tail'].get('status')!='source_closed_label'


def test_pale_metal_grabcut_ink_only_seed_recovers_shadowed_blank_paper():
    a,actual=label_photo();shade=np.clip((np.arange(len(a))-565)/35,0,1)
    a[:,:,:3][~actual]=np.repeat((180-95*shade)[:,None],a.shape[1],axis=1)[~actual,None]
    seed=actual.copy();seed[578:] &= a[578:,:,:3].mean(axis=2)<80
    mask,proof=source_paper_tail(a,seed,{})
    assert mask[585:599,245:255].all()
    assert not mask[585:599,275:285].any()
    assert proof['source_paper_tail']['added_paper_pixels']>0


def test_missing_one_physical_side_is_rejected_despite_a_rectangular_seed():
    a,actual=label_photo()
    # Erase one source paper edge and its inside margin; a seed alone cannot
    # serve as independent support for the physical paper side.
    a[400:,218:245,:3]=a[400:,210:211,:3]
    with pytest.raises(ValueError,match='side|continuation'):
        source_paper_tail(a,actual,{})


def test_raw_blank_canvas_cannot_move_the_measured_body_contact_end():
    a,actual=label_photo();seed=actual.copy()
    seed[578:] &= a[578:,:,:3].mean(axis=2)<80
    expected,info=source_paper_tail(a,seed,{},physical_end=np.full(a.shape[1],600.))
    padded=np.pad(a,((0,160),(0,0),(0,0)));padded[600:,:,:3]=120
    padded_seed=np.pad(seed,((0,160),(0,0)))
    result,proof=source_paper_tail(padded,padded_seed,{},physical_end=np.full(a.shape[1],600.))
    np.testing.assert_array_equal(result[:600],expected)
    assert not result[600:].any()
    assert proof['source_paper_tail']['physical_end_source']=='independent_body_contact_plane'
    assert info['source_paper_tail']['terminal_printed_rows']==proof['source_paper_tail']['terminal_printed_rows']
