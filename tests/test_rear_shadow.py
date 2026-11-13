"""Independent image formation controls for ring-shadow correction."""
import copy
import numpy as np
import pytest
from PIL import Image,ImageDraw
from phone_texture import rear_shadow
from phone_texture.auto_material import rear_camera_glass
from phone_texture.lighting import engine


def scene(shadow=True,scale=1):
    h,w=1500*scale,720*scale;yy,xx=np.indices((h,w));x=xx/scale;y=yy/scale
    image=Image.new('RGBA',(w,h),(65,65,65,255));draw=ImageDraw.Draw(image)
    draw.rounded_rectangle(tuple(int(v*scale) for v in (45,40,675,405)),radius=80*scale,fill=(180,180,180),outline=(20,20,20),width=3*scale)
    a=np.array(image);truth=np.zeros((h,w),bool);ring=np.zeros((h,w),bool)
    for cx,cy in [(130,120),(130,305),(315,215)]:
        radius=62.;distance=np.sqrt((x-cx)**2+(y-cy)**2)/radius
        if shadow:
            shade=45*np.exp(-((distance-1.30)/.13)**2)*np.clip((x-cx)/radius,0,1)
            region=(distance>1.18)&(distance<1.62)
            a[region,:3]=np.clip(a[region,:3].astype(float)-shade[region,None],0,255)
            truth|=(distance>1.24)&(distance<1.35)&(x>cx+radius*.9)
        for lo,hi,value in [(0,1,10),(1,1.11,120),(1.11,1.145,230),(1.145,1.18,25)]:
            region=(distance>=lo)&(distance<hi);a[region,:3]=value
        ring|=distance<1.18
    glass,_=rear_camera_glass(a,a[...,3]>0)
    material=(a[...,0]>65)&~glass
    # Include ring metal but keep the source plateau as the ordinary material.
    plate=np.zeros((h,w),bool);plate[45*scale:400*scale,50*scale:670*scale]=True
    material|=plate&ring&~glass
    return a,material,truth,ring


@pytest.mark.parametrize('scale',[1,2])
def test_measured_ring_shadow_improves_without_editing_hardware_or_outside(scale):
    a,m,truth,ring=scene(scale=scale);render=engine('aluminum')
    plan=rear_shadow.prepare(a,m,'source','material');assert plan['status']=='ready'
    L=render.rgb_to_lab(a[...,:3])[...,0]
    field,scope,proof=rear_shadow.correction(L,L,m,plan)
    assert np.median(field[truth])>4
    # The user now permits illumination correction of the dark OUTER joint.
    # The independently constructed glass, metal face and bright rim stay fixed.
    hardware=ring&(a[...,0]!=25)
    assert not scope[hardware].any()
    assert not field[~m].any()
    assert np.isfinite(field).all() and field.min()>=0 and field.max()<=35
    assert all(v['min_active_fit_support']>=.008 for v in proof['lenses'])


def test_uniform_plate_does_not_acquire_a_bright_halo():
    a,m,_,_=scene(shadow=False);plan=rear_shadow.prepare(a,m,'s','m')
    L=engine('aluminum').rgb_to_lab(a[...,:3])[...,0]
    field,_,_=rear_shadow.correction(L,L,m,plan)
    assert field.max()<.25


def test_narrow_joint_directional_darkening_is_reduced_without_editing_bright_rim():
    a,m,_,ring=scene(shadow=False)
    yy,xx=np.indices(m.shape)
    joint=np.zeros_like(m);dark=np.zeros_like(m)
    for cx,cy in [(130,120),(130,305),(315,215)]:
        radius=np.hypot(xx-cx,yy-cy)/62
        band=(radius>=1.145)&(radius<1.22)
        joint|=band;dark|=band&(xx>cx)
    a[joint,:3]=25
    m|=joint
    a[dark,:3]=5
    plan=rear_shadow.prepare(a,m,'s','m')
    L=engine('aluminum').rgb_to_lab(a[...,:3])[...,0]
    field,scope,proof=rear_shadow.correction(L,L,m,plan)
    assert np.percentile(field[dark],75)>2
    assert not scope[ring&~joint].any()
    assert all(x['narrow_joint_correction']['minimum_reference_angles']>=234 for x in proof['lenses'])


def test_absent_outer_rings_are_rejected_not_replaced_with_fixed_radii():
    a,m,_,ring=scene(shadow=False);glass,_=rear_camera_glass(a,a[...,3]>0)
    a[ring&~glass,:3]=180
    with pytest.raises(ValueError,match='ring edge'):
        rear_shadow.prepare(a,m,'s','m')


def test_missing_reference_is_a_failure():
    a,m,_,_=scene();plan=rear_shadow.prepare(a,m,'s','m')
    L=engine('aluminum').rgb_to_lab(a[...,:3])[...,0]
    with pytest.raises(ValueError,match='surrounding metal reference'):
        rear_shadow.correction(L,L,np.zeros_like(m),plan)


def test_two_cameras_are_outside_this_three_camera_profile():
    a,m,_,_=scene();a[100:300,220:420,:3]=180
    result=rear_shadow.prepare(a,m,'s','m')
    assert result['status']=='not_applicable'


@pytest.mark.parametrize('changed',['source','material'])
def test_engine_rejects_stale_shadow_bindings(tmp_path,changed):
    a=np.full((20,30,4),255,np.uint8);a[...,:3]=100
    source=tmp_path/'source.png';mask=tmp_path/'mask.png'
    Image.fromarray(a).save(source);Image.new('L',(30,20),255).save(mask)
    render=engine('aluminum');plan=dict(source_sha256=render.digest(source),material_sha256=render.digest(mask),material_mask=str(mask))
    plan[changed+'_sha256']='0'*64
    item=dict(name='back',source=str(source),domains=[dict(name='camera_surround',mask=str(mask),sigma=[3,3],strength=[0,0,0])],rear_shadow_correction=plan)
    with pytest.raises(ValueError,match='binding changed'):
        render.process_view(item,np.array([40,0,0]),dict(color_mode='source_lightness'))
