"""Bright frame grain must survive correction when RGB headroom runs out."""
import copy
import numpy as np
from PIL import Image
from phone_texture import auto_material, lighting


def test_top_frame_gamut_limit_preserves_faint_source_texture(tmp_path):
    h,w=70,700;yy,xx=np.indices((h,w))
    rgb=np.full((h,w,3),150.)
    rgb[:,60:150]=[235,238,244]
    rgb[:,60:150]+=np.where((xx[:,60:150]+yy[:,60:150])%2,2.,-2.)[...,None]
    rgb[:,260:440]=[245,247,252]
    source=np.full((h,w,4),255,np.uint8);source[...,:3]=rgb
    path=tmp_path/'source.png';Image.fromarray(source).save(path)
    job,_,allowed,paper=auto_material.propose_job(path,'top',tmp_path/'materials')
    # A genuine bright source reference, with faint grain elsewhere on the
    # same metal. Per-pixel gamut clamping previously erased that grain.
    item=job['views'][0];item['domains'][0]['reference_boxes']=[[300,20,307,27]]
    e=lighting.engine('aluminum');target,_=e.target_color(job)
    defaults=dict(job['defaults'],color_mode=job['color_mode'])
    result,_,_,proof=e.process_view(item,target,defaults)
    old=copy.deepcopy(item)
    for domain in old['domains']:domain.pop('gamut_continuity_step_L',None)
    clipped,_,_,_=e.process_view(old,target,defaults)
    pattern=(xx+yy)[20:50,85:125]%2
    def contrast(a):
        L=e.rgb_to_lab(a[...,:3])[20:50,85:125,0]
        return float(L[pattern==1].mean()-L[pattern==0].mean())
    assert contrast(clipped)<contrast(source)*.1
    assert contrast(result)>contrast(source)*.75
    np.testing.assert_array_equal(result[...,3],source[...,3])
    np.testing.assert_array_equal(result[~allowed|paper],source[~allowed|paper])
    assert proof['gamut_continuity'][0]['adjusted_pixels']>0
