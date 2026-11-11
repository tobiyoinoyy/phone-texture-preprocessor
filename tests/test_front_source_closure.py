import numpy as np
import pytest
from PIL import Image,ImageDraw
from phone_texture.auto_geometry import front_lower_closure,principal


def fixture():
 h,w=700,400;im=Image.new('L',(w,h));ImageDraw.Draw(im).rounded_rectangle((80,50,320,550),radius=30,fill=255);body=np.array(im)>0;yy,xx=np.indices((h,w));clips=(((xx>=105)&(xx<135))|((xx>=270)&(xx<300)))&(yy>=530)&(yy<590)
 mask=body|clips;raw=np.full((h,w,3),230,np.uint8);raw[(xx>=100)&(xx<305)&(yy>=545)]=210;raw[clips]=200;raw[body]=20;raw[body&(yy>=546)]=170
 widths=mask.sum(axis=1);wide=np.flatnonzero(widths>widths.max()*.8);return mask,raw,body,int(wide.max())


def test_front_source_alpha_keeps_all_known_body_pixels_despite_bright_rim():
 mask,raw,body,last=fixture();before=raw.copy();_,line,_,_,_,proof=front_lower_closure(mask,raw,80,321,last)
 yy,xx=np.indices(mask.shape);removed=yy>np.polyval(line,xx+.5)+1
 assert not np.any(removed&body)
 assert np.any(removed&mask&~body)
 assert np.array_equal(raw,before)
 assert proof['RGB_does_not_set_inward_cut']
 assert min(proof['dark_glass_closure_fractions'])>=.9


def test_front_alpha_without_actual_rgb_closure_is_rejected():
 mask,raw,body,last=fixture();raw[:]=20
 with pytest.raises(ValueError):front_lower_closure(mask,raw,80,321,last)


def test_source_boundary_shifted_away_from_alpha_is_not_confirmed():
 mask,raw,body,last=fixture();raw[520:]=230
 with pytest.raises(ValueError):front_lower_closure(mask,raw,80,321,last)


def test_narrow_support_cannot_define_front_body_end():
 mask,raw,body,last=fixture();mask[550:650,140:260]=True;raw[550:650,140:260]=20
 with pytest.raises(ValueError):front_lower_closure(mask,raw,80,321,last)
