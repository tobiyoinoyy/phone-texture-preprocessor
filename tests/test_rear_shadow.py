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


@pytest.mark.parametrize('radius', [62., 105.])
def test_extra_lift_cannot_reveal_a_second_line_outside_the_true_rim(radius):
    y, x = np.ogrid[:330, :330]
    distance = np.hypot(x-165, y-165) - radius*1.14
    angle = np.rad2deg(np.arctan2(y-165, x-165))
    material = distance > 0
    lens = dict(ellipse=[165, 165, 2*radius, 2*radius, 0],
                hardware_radii=[1.14]*rear_shadow.ANGLES)
    # A single dark joint fades into the photographed metal. The artificial
    # lift introduces a second trough while leaving the photographed joint intact.
    raw = np.zeros((*material.shape, 3))
    raw[..., 0] = 34 - 22*np.exp(-np.maximum(distance, 0)/4.5)
    raw[..., 2] = -18 + 10*np.exp(-((distance-8)/3)**2)
    base = raw.copy()
    base[..., 0] += 5.
    untouched = base.copy()
    assert rear_shadow._restore_secondary_arc(base, raw, untouched, material, lens)['status'] == 'clear'
    assert np.array_equal(untouched, base)
    broad_arc = np.abs(angle) < 38
    fixed = base.copy()
    fixed[..., 0] += 21*np.exp(-((distance-5)/1.8)**2)*broad_arc
    original = fixed.copy()
    report = rear_shadow._restore_secondary_arc(base, raw, fixed, material, lens)
    assert report['status'] == 'guarded'
    assert report['reconstructed_angles'] > 20
    assert report['same_material_color_pixels'] > 0
    assert np.array_equal(fixed[distance <= 2], original[distance <= 2])
    assert np.array_equal(fixed[np.abs(angle) > 55], original[np.abs(angle) > 55])
    centre_arc = np.abs(angle) < 12
    crest, valley = [(distance > lo)&(distance < hi)&centre_arc for lo, hi in [(4, 6), (8, 10)]]
    assert np.median(fixed[crest, 0]) - np.median(fixed[valley, 0]) < 1.
    assert np.median(np.abs(fixed[valley, 2]+18)) < 2.5


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


def test_optional_shadow_rejection_keeps_verified_base_lighting(tmp_path, monkeypatch):
    a=np.full((35,45,4),255,np.uint8)
    a[...,:3]=np.linspace(44,128,45,dtype=np.uint8)[None,:,None]
    source=tmp_path/'source.png';mask=tmp_path/'mask.png'
    Image.fromarray(a).save(source);Image.new('L',(45,35),255).save(mask)
    render=engine('aluminum')
    view=dict(name='back',source=str(source),domains=[dict(name='camera_surround',mask=str(mask),sigma=[3,3])])
    defaults=dict(color_mode='source_lightness')
    baseline,_,_,_=render.process_view(view,np.array([60,0,0]),defaults)
    assert not np.array_equal(baseline,a)
    plan=dict(source_sha256=render.digest(source),material_sha256=render.digest(mask),material_mask=str(mask))
    def reject(*_):
        raise rear_shadow.InsufficientRearShadowEvidence('Rear shadow: narrow joint lacks source support')
    monkeypatch.setattr(rear_shadow,'correction',reject)
    result,_,_,report=render.process_view(dict(view,rear_shadow_correction=plan),np.array([60,0,0]),defaults)
    assert np.array_equal(result,baseline)
    assert report['rear_shadow_correction']['status'] == 'source_evidence_unavailable'


def test_one_unsupported_lens_preserves_its_base_and_other_corrections(monkeypatch):
    rgba, metal, _, _ = scene()
    plan = rear_shadow.prepare(rgba, metal, 'source', 'material')
    light = engine('aluminum').rgb_to_lab(rgba[..., :3])[..., 0]
    complete, _, _ = rear_shadow.correction(light, light, metal, plan)
    original = rear_shadow._lens_field
    lens = plan['lenses'][0]

    def reject_first(*args):
        if args[5] is lens:
            raise rear_shadow.InsufficientRearShadowEvidence('source shadow has unrecoverable black pixels')
        return original(*args)

    monkeypatch.setattr(rear_shadow, '_lens_field', reject_first)
    field, scope, proof = rear_shadow.correction(light, light, metal, plan)
    assert proof['status'] == 'partially_applied'
    assert [r['lens'] for r in proof['skipped_lenses']] == [0]
    cx, cy, major, minor, angle = lens['ellipse']
    yy, xx = np.indices(metal.shape)
    phi = np.deg2rad(angle)
    u = (np.cos(phi)*(xx-cx) + np.sin(phi)*(yy-cy)) / (major/2)
    v = (-np.sin(phi)*(xx-cx) + np.cos(phi)*(yy-cy)) / (minor/2)
    protected = u*u + v*v <= (max(lens['start_radii']) + .42)**2
    assert not scope[protected].any()
    assert scope.any() and np.array_equal(field[~protected], complete[~protected])


@pytest.mark.parametrize("saved_mask", [False, True])
@pytest.mark.parametrize("tint", [(1., 1., 1.), (.48, .59, .82), (1.20, .75, .50)])
def test_pro_fallback_reviews_cast_shadow_with_or_without_saved_plateau(tmp_path, saved_mask, tint):
    from phone_texture.lighting_continuation import recover

    source, metal, shade, hardware = scene()
    source[..., :3] = np.clip(np.rint(source[..., :3] * tint), 0, 255).astype('uint8')
    src = tmp_path/'source.png'
    out = tmp_path/'back.png'
    Image.fromarray(source).save(src)
    if saved_mask:
        Image.fromarray((metal * 255).astype('uint8')).save(tmp_path/'camera_surround.png')
    # Keep a genuinely paper-covered spot inside the otherwise editable shade.
    paper = np.zeros(shade.shape, bool)
    paper[115:125, 207:213] = True
    Image.fromarray((paper * 255).astype('uint8')).save(tmp_path/'paper.png')
    proof = recover(src, 'back', out, tmp_path, metal_back=True, review_camera=True)
    result = np.array(Image.open(out))
    assert proof['status'] == 'degraded_lighting'
    assert proof['camera_metal']['applied_pixels'] > 0
    assert proof['camera_metal']['measured_slots'] == [0, 1, 2]
    assert np.median(result[..., 0][shade & ~paper]) > np.median(source[..., 0][shade & ~paper])
    np.testing.assert_array_equal(result[hardware | paper], source[hardware | paper])
    np.testing.assert_array_equal(result[..., 3], source[..., 3])
