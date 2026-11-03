import numpy as np
import pytest

from phone_texture.lighting import engine


def test_sparse_source_coverage_preserves_supported_edge_correction():
    e=engine('aluminum');shape=(31,41)
    shift=np.zeros((*shape,3));shift[:]=[12,-3,2]
    reliable=np.ones(shape,bool);fit=np.zeros(shape,bool);fit[12:19,17:24]=True
    before=shift.copy()
    result,proof=e.smooth_material_correction(shift,reliable,fit,reliable,1.5,'source_fit_coverage')
    # A sparse reliable fit must not turn already supported outer pixels into
    # an implicit protected strip or introduce a density-shaped correction.
    np.testing.assert_allclose(result,shift,atol=1e-12)
    np.testing.assert_array_equal(shift,before)
    assert proof['zero_local_fit_pixels']>0
    assert proof['zero_local_fit_max_legacy_difference']==0


def test_background_mixed_seeds_do_not_vote_like_opaque_body_samples():
    e=engine('aluminum');h,w=40,61
    shift=np.zeros((h,w,3));shift[...,0]=5;shift[10:,:,0]=40
    reliable=np.ones((h,w),bool);fit=np.zeros((h,w),bool);fit[10:]=True
    legacy,_=e.smooth_material_correction(shift,reliable,fit,reliable,1.5)
    revised,_=e.smooth_material_correction(shift,reliable,fit,reliable,1.5,'source_fit_coverage')
    assert legacy[10,w//2,0]<revised[10,w//2,0]<40
    assert revised[...,0].min()>=5-1e-12 and revised[...,0].max()<=40+1e-12
    assert np.diff(revised[:,w//2,0]).min()>=-1e-12
    # No local source-fit support at the distant soft edge: retain the
    # existing supported field exactly, rather than silently replacing it.
    np.testing.assert_array_equal(revised[0],legacy[0])


def test_unsupported_samples_cannot_enter_the_source_coverage_field():
    e=engine('aluminum');shape=(40,60)
    reliable=np.zeros(shape,bool);reliable[8:32,8:52]=True
    fit=np.ones(shape,bool);shift=np.zeros((*shape,3));shift[...,0]=17
    expected,_=e.smooth_material_correction(shift,reliable,fit,reliable,1.5,'source_fit_coverage')
    shift[~reliable]=[1e6,-1e6,1e6]
    actual,_=e.smooth_material_correction(shift,reliable,fit,reliable,1.5,'source_fit_coverage')
    np.testing.assert_array_equal(actual[reliable],expected[reliable])


def test_source_coverage_rejects_missing_supported_field_and_unknown_mode():
    e=engine('aluminum');shape=(20,30)
    shift=np.ones((*shape,3));none=np.zeros(shape,bool);editable=~none
    with pytest.raises(ValueError,match='Unsupported source-fit'):
        e.smooth_material_correction(shift,none,none,editable,1.5,'source_fit_coverage')
    with pytest.raises(ValueError,match='Unknown correction smoothing'):
        e.smooth_material_correction(shift,editable,editable,editable,1.5,'unrecognized')
