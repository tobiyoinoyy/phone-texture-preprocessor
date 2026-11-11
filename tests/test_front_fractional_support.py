import numpy as np
from phone_texture.auto_geometry import front_lower_closure,front_support_coverage
from test_front_source_closure import fixture


def test_exact_front_coverage_keeps_body_and_removes_full_contact_under_it():
 mask,raw,body,last=fixture();_,line,_,_,_,proof=front_lower_closure(mask,raw,80,321,last,mask.astype(np.uint8)*255)
 yy,xx=np.indices(mask.shape);limit=np.polyval(line,np.arange(mask.shape[1])+.5)[None,:]
 support=((xx[0]>=105)&(xx[0]<135))|((xx[0]>=270)&(xx[0]<300))
 original=mask.astype(np.uint8)*255;out=front_support_coverage(original,limit,support[None,:])
 assert np.array_equal(out[body],original[body])
 assert not out[(~body)&mask&(yy>=limit)].any()
 assert np.array_equal(out[:,~support],original[:,~support])


def test_fractional_source_boundary_does_not_add_an_opaque_outer_pixel():
 alpha=np.full((20,10),255,np.uint8);limit=np.full((1,10),10.5)
 out=front_support_coverage(alpha,limit,np.ones((1,10),bool))
 assert np.all(out[:10]==255) and np.all(out[10]==128) and not out[11:].any()
 assert np.all(alpha==255)
