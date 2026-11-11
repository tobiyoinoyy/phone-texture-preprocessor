import numpy as np
import pytest
from test_unmarked_paper_closure import source
from phone_texture.bottom_paper_closure import resolve_aligned_strip_end
from phone_texture.bottom_paper_audit import guard_bottom_paper_end

@pytest.mark.parametrize('slope',[-.13,-.07,.07,.13])
def test_source_edge_registration_preserves_sloped_paper_exactly(slope):
 a,mask,actual,proof=source(slope=slope);before=a.copy();original=mask.copy()
 result,evidence=resolve_aligned_strip_end(a,mask,proof)
 np.testing.assert_array_equal(result,actual)
 np.testing.assert_array_equal(a,before);np.testing.assert_array_equal(mask,original)
 assert not evidence['aligned_source_closure']['source_rgb_interpolated']
 guard_bottom_paper_end(a,result,evidence)

@pytest.mark.parametrize('slope',[-.12,.12])
def test_sloped_full_width_print_band_cannot_hide_continuing_paper(slope):
 a,mask,actual,proof=source(end=140);yy,xx=np.indices(mask.shape)
 stripe=mask&(yy>=95+slope*(xx-290))&(yy<107+slope*(xx-290));a[stripe,:3]=[25,35,58]
 with pytest.raises(ValueError,match='continue below'):
  resolve_aligned_strip_end(a,mask,proof)

def test_aligned_closure_still_requires_current_source_side_measurements():
 a,m,_,_=source(slope=.1)
 with pytest.raises(ValueError,match='independently measured paper sides'):
  resolve_aligned_strip_end(a,m,{})
