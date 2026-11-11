import numpy as np
import pytest
from PIL import Image
from phone_texture.auto_geometry import frame_geometry
from phone_texture.top_frame import calibrate_top


def scene(lower_slope=0,fast_lower=False):
 h,w=450,900;yy,xx=np.mgrid[:h,:w];distance=np.minimum(xx-50,849-xx)
 top=200+35*np.exp(-np.maximum(distance,0)/10)
 bottom=330+lower_slope*(xx-450)-30*np.exp(-np.maximum(distance,0)/(2 if fast_lower else 8))
 body=(distance>=0)&(yy>=top)&(yy<bottom);camera=(xx>=120)&(xx<770)&(yy>=130)&(yy<230)
 raw=np.full((h,w,3),230,np.uint8);raw[camera]=[90,65,40];raw[body]=[180,100,60]
 return Image.fromarray(np.dstack([raw,(body|camera).astype(np.uint8)*255])),Image.fromarray(raw),body,camera&~body


def test_rail_check_uses_existing_postrotation_angle_not_fixed_resolution_pixels():
 matte,raw,body,camera=scene(.0051);matte=matte.resize((1800,900),Image.Resampling.NEAREST);raw=raw.resize((1800,900),Image.Resampling.NEAREST);stages={}
 out,p=frame_geometry(matte,'top',lambda n,i:stages.update({n:i}),raw)
 assert p['rail_agreement']['thickness_delta_px']>4
 assert max(p['transform_checks']['residual_rail_degrees'])<=.15
 assert p['automatic_checks']['body_changed_pixels']==0
 assert p['transform_checks']['camera_after_rotation_pixels']==0


def test_underresolved_lower_decay_requires_observed_flat_rail_and_raw_rgb():
 matte,raw,body,camera=scene(fast_lower=True)
 out,p=frame_geometry(matte,'top',source_image=raw)
 assert any(c['fast_lower_plateau'] for c in p['corners'])
 assert p['automatic_checks']['body_changed_pixels']==0
 with pytest.raises(ValueError):calibrate_top(np.array(matte),np.full_like(np.array(raw),128))


def test_diverging_rails_are_still_rejected_by_same_angle_limit():
 matte,raw,*_=scene(.01)
 with pytest.raises(ValueError,match='RAILS_DISAGREE'):calibrate_top(np.array(matte),np.array(raw))


def test_altered_lower_source_alpha_without_rgb_evidence_is_rejected():
 matte,raw,*_=scene(fast_lower=True);a=np.array(matte);a[328:340,825:840,3]=255
 with pytest.raises(ValueError):calibrate_top(a,np.array(raw))
