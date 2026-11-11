"""Behavioral controls for the accepted illumination target, not visual approval."""
import copy
import numpy as np
import pytest
from PIL import Image
from phone_texture import auto_material,lighting
from phone_texture.lighting_review import EffectThresholdFailure
from phone_texture import shadow_lift_profile as profile


def source_job(tmp_path):
    h,w=70,500
    y,x=np.indices((h,w))
    a=np.full((h,w,4),255,np.uint8)
    shade=110+55*np.cos((x-w/2)/w*np.pi)
    a[...,:3]=np.clip(shade[...,None]+np.array([0,5,10]),0,255)
    a[15:50,40:43,:3]-=8  # independent source blemish, not a fit target
    a[:2,:,3]=0
    path=tmp_path/'source.png';Image.fromarray(a).save(path)
    assets=tmp_path/'assets'
    job,_,_,_=auto_material.propose_job(path,'top',assets,protection_policy='none')
    return a,path,assets,job


def test_lift_remains_source_bound_and_never_blurs_or_moves_texture(tmp_path):
    a,path,assets,job=source_job(tmp_path)
    new,proof=profile.apply(job,'top',assets)
    engine=lighting.engine('aluminum');target,_=engine.target_color(new)
    b,allowed,protected,_=engine.process_view(new['views'][0],target,dict(new['defaults'],color_mode='source_lightness'))
    np.testing.assert_array_equal(a[...,3],b[...,3])
    np.testing.assert_array_equal(a[~allowed|protected],b[~allowed|protected])
    np.testing.assert_array_equal(a[a[...,3]==0],b[a[...,3]==0])
    assert proof['source_fit_support_threshold']==.008
    assert not proof['other_SKUs_user_visual_approved']
    # The planted scratch must remain darker than both same-row neighbors.
    L=engine.rgb_to_lab(b[...,:3])[...,0]
    assert np.median(L[20:45,40:43])<np.median(L[20:45,46:49])-1


def test_foreign_source_hash_is_rejected_before_profile_material_writes(tmp_path):
    _,_,assets,job=source_job(tmp_path);job['views'][0]['source_sha256']='0'*64
    with pytest.raises(ValueError,match='source identity'):profile.apply(job,'top',assets)
    assert not (assets/'main_surface-without-label.png').exists()


def test_paper_has_own_small_fit_and_is_not_frozen(tmp_path):
    a,path,assets,job=source_job(tmp_path);label=np.zeros(a.shape[:2],bool);label[:,225:270]=True
    a[label,:3]=200;a[20:45,235:240,:3]=35;Image.fromarray(a).save(path)
    engine=lighting.engine('aluminum');job['views'][0]['source_sha256']=engine.digest(path)
    label_path=assets/'paper-fit-hint.png';Image.fromarray(label.astype('uint8')*255).save(label_path);job['views'][0]['source_label_fit_exclusion_mask']=str(label_path)
    new,_=profile.apply(job,'top',assets);paper=next(d for d in new['views'][0]['domains'] if d['name']=='printed_label')
    assert paper['strength'][0]>0 and paper['max_lab_shift']==[6,0,0]
    assert np.array(Image.open(paper['mask'])).any()
    assert not np.array(Image.open(new['views'][0]['protected_mask'])).any()


@pytest.mark.parametrize('name', ['accepted-shadow-lift-v1', 'accepted-shadow-lift-v2'])
def test_aesthetic_warning_is_never_changed_to_a_pass_or_used_for_hard_failures(name):
    def aesthetic(*args):
        raise EffectThresholdFailure([dict(kind='perimeter_brightness', check='top',
            measured_L=35., min_L=50., max_L=80., error='Perimeter effect failed at top: control')],
            {'top':dict(effect_check_passed=False, after_L=35.)})
    result=profile.measure_effect(aesthetic,None,None,None,None,None,name)
    assert result['original_effect_check_passed'] is False and result['passed'] is False
    with pytest.raises(ValueError,match='Perimeter effect failed'):profile.measure_effect(aesthetic,None,None,None,None,None,'baseline')
    def mandatory(*args):raise ValueError('Unsupported editable illumination: control')
    with pytest.raises(ValueError,match='Unsupported editable'):profile.measure_effect(mandatory,None,None,None,None,None,name)


def test_existing_complete_paper_component_is_refit_without_a_false_failure(tmp_path):
    a,path,assets,job=source_job(tmp_path)
    label=np.zeros(a.shape[:2],bool);label[:,225:270]=True
    mp=assets/'already-paper.png';Image.fromarray(label.astype('uint8')*255).save(mp)
    metal=~label;metal_path=assets/'metal.png';Image.fromarray(metal.astype('uint8')*255).save(metal_path)
    view=job['views'][0];view['source_label_fit_exclusion_mask']=str(mp)
    view['domains'][0]['mask']=str(metal_path)
    view['domains'].append(dict(name='label_component_0',mask=str(mp),fit_mask=str(mp),reference_boxes=[[230,4,237,11]],sigma=[4,20],strength=[.72,0,0]))
    view['correction_blends']=[dict(domains=['label_component_0','main_surface'],boundary_domain='label_component_0',sigma=1.25,kind='source_material_transition')]
    new,proof=profile.apply(job,'top',assets)
    domains={d['name']:d for d in new['views'][0]['domains']}
    assert 'label_component_0' not in domains and 'printed_label' in domains
    assert new['views'][0]['correction_blends'] == [dict(
        domains=['printed_label', 'main_surface'], boundary_domain='printed_label',
        sigma=1.25, kind='source_material_transition')]
    migration = next(change['printed_label_transition_migration']
                     for change in proof['changes']
                     if 'printed_label_transition_migration' in change)
    assert migration['status'] == 'inherited_pair_migrated'
    assert migration['old_paper_components'] == ['label_component_0']
    assert migration['direct_support_pixels'] > 0
    assert migration['source_boundary_certified_here'] is False
    assert migration['final_rgb_support_guaranteed'] is False
    union=np.logical_or.reduce([np.array(Image.open(d['mask']))>0 for d in domains.values()])
    np.testing.assert_array_equal(union,a[...,3]>0)
    assert any(x.get('removed_redundant_paper_components')==['label_component_0'] for x in proof['changes'])


def test_back_perimeter_retains_generic_source_fit_and_render(tmp_path):
    a,path,assets,job=source_job(tmp_path)
    view=job['views'][0];view['name']='back'
    domain=view['domains'][0];domain.update(name='rear_perimeter',
        strength=[.92,0,0],max_lab_shift=[48,0,0])
    baseline=copy.deepcopy(domain)
    render=lighting.engine('aluminum');target,_=render.target_color(job)
    before,_,_,_=render.process_view(view,target,dict(job['defaults'],color_mode='source_lightness'))
    new,proof=profile.apply(job,'back',assets)
    actual=new['views'][0]['domains'][0]
    assert set(actual)==set(baseline)|{'acceptance_reference_boxes'}
    for key,value in baseline.items():
        if key=='mask' or key.endswith('_mask'):
            np.testing.assert_array_equal(np.array(Image.open(actual[key])),np.array(Image.open(value)))
        else:
            assert actual[key]==value
    v2_box,_=profile.reference(a,baseline,render)
    assert actual['acceptance_reference_boxes']==[v2_box]
    target,_=render.target_color(new)
    after,_,_,_=render.process_view(new['views'][0],target,dict(new['defaults'],color_mode='source_lightness'))
    np.testing.assert_array_equal(after,before)
    np.testing.assert_array_equal(np.array(Image.open(path).convert('RGBA')),a)
    assert any(change.get('retained_generic_source_contour_model') for change in proof['changes'])
    retained=next(change for change in proof['changes'] if change.get('retained_generic_source_contour_model'))
    assert retained['acceptance_reference_boxes']==[v2_box]
    assert retained['acceptance_thresholds_relaxed'] is False


def test_perimeter_retention_does_not_disable_other_back_material_lift(tmp_path):
    _,_,assets,job=source_job(tmp_path)
    job['views'][0]['name']='back'
    old=copy.deepcopy(job['views'][0]['domains'][0])
    new,_=profile.apply(job,'back',assets)
    main=new['views'][0]['domains'][0]
    assert main['name']=='main_surface'
    assert main['strength']==[.98,0,0] and main['max_lab_shift']==[60,0,0]
    assert main['strength']!=old['strength']


def test_retained_rear_keeps_original_v2_scope_thresholds_and_witnesses(tmp_path):
    # Distinct generic and V2 references expose a silent weakening of the
    # brightness gate even when the 1.25 L continuity gate stays unchanged.
    h,w=120,100
    y,x=np.indices((h,w))
    a=np.full((h,w,4),255,np.uint8)
    a[...,:3]=(70+160*x/(w-1)).astype(np.uint8)[...,None]
    path=tmp_path/'back.png';Image.fromarray(a).save(path)
    def mask(name,region):
        p=tmp_path/(name+'.png');Image.fromarray(region.astype(np.uint8)*255).save(p)
        return str(p)
    rear=(x<12)|(x>=w-12)|(y<12)|(y>=h-12)
    regions={'rear_perimeter':rear,'camera_surround':~rear&(y<40),
             'main_surface':~rear&(y>=40)}
    domains=[]
    for name,region in regions.items():
        mp=mask(name,region)
        domains.append(dict(name=name,mask=mp,fit_mask=mp,
            reference_boxes=[auto_material.reference_box(a,region)],
            strength=[.92,0,0],max_lab_shift=[48,0,0],sigma=[18,5],
            feather=0,correction_smooth=1.5))
    rear_domain=domains[0]
    rear_domain.update(field='contour',reference_profile='contour_depth_lift',
        source_body_contour={'sides':{'left':{'protrusion_rows':[[30,50]]},
                                      'right':{'protrusion_rows':[[70,90]]}}})
    render=lighting.engine('aluminum')
    job=dict(color_mode='source_lightness',views=[dict(name='back',source=str(path),
        source_sha256=render.digest(path),protection_policy='none',
        sticker_mask=mask('paper',np.zeros((h,w),bool)),domains=domains,
        correction_blends=[dict(domains=list(regions),boundary_domain='rear_perimeter',sigma=2)])])
    v2_box,_=profile.reference(a,rear_domain,render)
    from phone_texture.lighting_review import lightness
    def reference_L(box):
        x0,y0,x1,y1=box
        return float(np.median(lightness(a[y0:y1,x0:x1,:3])))
    assert reference_L(v2_box)>reference_L(rear_domain['reference_boxes'][0])+2
    strict,proof=profile.apply(job,'back',tmp_path)
    generated=strict['views'][0]['domains'][0]
    for key in ('reference_boxes','strength','max_lab_shift','sigma'):
        assert generated[key]==rear_domain[key]
    assert generated['acceptance_reference_boxes']==[v2_box]
    assert generated['reference_boxes']!=generated['acceptance_reference_boxes']

    # Reconstruct the original V2 rear proposal. The other V2 material domains
    # already match; this fixture has no label refinement or removed material.
    original_v2=copy.deepcopy(strict)
    original_v2['views'][0]['reference_boxes']=[v2_box]
    old_rear=original_v2['views'][0]['domains'][0]
    old_rear.pop('acceptance_reference_boxes')
    old_rear.update(reference_boxes=[v2_box],strength=[.98,0,0],max_lab_shift=[60,0,0])
    expected=auto_material.lighting_scope(path,'back',original_v2,tmp_path/'original-v2-scope')
    actual=auto_material.lighting_scope(path,'back',strict,tmp_path/'strict-scope')
    def canonical(value):
        if isinstance(value,dict):return {k:canonical(v) for k,v in value.items()}
        if isinstance(value,list):return [canonical(v) for v in value]
        if isinstance(value,str) and value.endswith('.png'):
            pixels=np.array(Image.open(value))
            return pixels.shape,pixels.dtype.str,pixels.tobytes()
        return value
    assert canonical(actual)==canonical(expected)
    assert actual['local_checks']['rear_button_left_0']['max_added_step_L']==1.25
    assert actual['local_checks']['rear_inner_boundary_left']['max_boundary_step_L']==2.

    # A negative control proves that the fallback-only reference would have
    # weakened both median and dark-shadow requirements in this source.
    weakened=copy.deepcopy(strict)
    weakened['views'][0]['domains'][0].pop('acceptance_reference_boxes')
    lowered=auto_material.lighting_scope(path,'back',weakened,tmp_path/'weakened-scope')
    assert any(actual['zones'][name]['min_L']>lowered['zones'][name]['min_L']
               for name in actual['zones'])
    assert any(actual['local_checks'][name]['min_L']>lowered['local_checks'][name]['min_L']
               for name in actual['local_checks'] if name.startswith('rear_shadow_'))
    retained=next(change for change in proof['changes'] if change.get('retained_generic_source_contour_model'))
    assert retained['parameters']['reference_boxes']==rear_domain['reference_boxes']
    assert retained['acceptance_reference_boxes']==[v2_box]
    assert retained['acceptance_thresholds_relaxed'] is False
