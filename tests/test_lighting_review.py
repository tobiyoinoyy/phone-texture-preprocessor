import copy
import json

import numpy as np
from PIL import Image
import pytest

from phone_texture import lighting_review as r
from test_six_view_workflow import setup, w


def fixture(tmp):
    _, work, geometry, plan = setup(tmp)
    a = np.array(Image.open(geometry/'back.png'))
    a[..., :3] = 40
    Image.fromarray(a).save(work/'source.png')
    a[..., :3] = 160
    Image.fromarray(a).save(work/'good.png')
    Image.new('L',(16,32),255).save(work/'metal.png')
    # Different source-coordinate witnesses, including small corner patches.
    boxes = [(5,0,10,2),(5,30,10,32),(0,8,2,24),(14,8,16,24),
             (0,0,3,3),(13,0,16,3),(0,29,3,32),(13,29,16,32)]
    zones = {}
    for name, box in zip(r.ZONES,boxes):
        m=Image.new('L',(16,32));m.paste(255,box);p=work/f'{name}.png';m.save(p)
        zones[name]={'mask':str(p),'min_L':50,'max_L':80}
    spec=dict(source_sha256=r.digest(work/'source.png'),action='aluminum',mode='perimeter',
              protection_policy='reviewed_regions',protection_basis='Synthetic task preserves defined materials',
              reason='Remove confirmed metal shadow rim',reference_basis='Reviewed same-material source sample',
              required_metal_mask=str(work/'metal.png'),zones=zones)
    return work, spec


def test_effect_accepts_good_rejects_old_dark_rim(tmp_path):
    work, spec=fixture(tmp_path);allowed=np.ones((32,16),bool)
    result=r.measure_perimeter(spec,work/'source.png',work/'good.png',allowed,~allowed)
    assert len(result)==8
    with pytest.raises(ValueError,match='effect failed'):
        r.measure_perimeter(spec,work/'source.png',work/'source.png',allowed,~allowed)


@pytest.mark.parametrize('fault',['corner','frozen','wrong_source','missing_zone','nan','empty','alpha'])
def test_perimeter_guards(tmp_path,fault):
    work,spec=fixture(tmp_path);allowed=np.ones((32,16),bool)
    if fault=='corner':
        a=r.pixels(work/'good.png');a[:3,:3,:3]=40;Image.fromarray(a).save(work/'good.png')
    elif fault=='frozen':allowed[0,0]=False
    elif fault=='wrong_source':spec['source_sha256']='old'
    elif fault=='missing_zone':del spec['zones']['bottom_right']
    elif fault=='nan':spec['zones']['top']['min_L']=float('nan')
    elif fault=='empty':Image.new('L',(16,32),0).save(work/'top.png')
    elif fault=='alpha':
        a=r.pixels(work/'good.png');a[0,0,3]=0;Image.fromarray(a).save(work/'good.png')
    with pytest.raises(ValueError):r.measure_perimeter(spec,work/'source.png',work/'good.png',allowed,~allowed)


def reviewed(work,spec):
    path=work/'qa'
    review=r.evidence_images(spec,work/'source.png',work/'good.png',path)
    review['checks']={c:True for c in r.CHECKS};r.write(path/'review.json',review)
    return path/'review.json'


@pytest.mark.parametrize('fault',['unchecked','stale_output','evidence_changed','missing_evidence','scope_changed'])
def test_visual_review_is_bound_to_actual_artifacts(tmp_path,fault):
    work,spec=fixture(tmp_path);p=reviewed(work,spec)
    r.check_review(spec,work/'source.png',work/'good.png',p)
    review=r.read(p)
    if fault=='unchecked':review['checks']['four_corners']=False
    elif fault=='stale_output':Image.new('RGBA',(16,32),'white').save(work/'good.png')
    elif fault=='evidence_changed':Image.new('RGB',(2,2),'white').save(next(iter(review['evidence'])))
    elif fault=='missing_evidence':review['evidence'].pop(next(iter(review['evidence'])))
    elif fault=='scope_changed':spec['zones']['top']['min_L']=10
    r.write(p,review)
    with pytest.raises(ValueError):r.check_review(spec,work/'source.png',work/'good.png',p)


def test_publish_cannot_silently_copy_required_front(tmp_path):
    _,work,geometry,plan=setup(tmp_path)
    spec={'SKU_01':{f:dict(action=e['action'],reason=e['reason'],mode='unchanged',
            source_sha256=w.sha(geometry/(f+'.png'))) for f,e in plan['SKU_01'].items()}}
    spec['SKU_01']['front'].update(action='aluminum',mode='regional',reason='Visible metal rim must be processed',protection_policy='reviewed_regions',protection_basis='Synthetic regional test')
    for face in ('front','back'):
        spec['SKU_01'][face]['perimeter_exemption']=dict(reason='user_scope_excludes_perimeter',
            basis='Synthetic regional-correction test',source_sha256=w.sha(geometry/(face+'.png')))
    r.write(work/'scope.json',spec);r.freeze_scope(work,work/'scope.json')
    w.save(work/'plan.json',plan)
    with pytest.raises(ValueError,match='frozen lighting scope'):w.publish(work,work/'plan.json',tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_publish_missing_scope_is_blocked(tmp_path):
    _,work,geometry,plan=setup(tmp_path);w.save(work/'plan.json',plan)
    with pytest.raises(ValueError,match='Frozen lighting scope'):w.publish(work,work/'plan.json',tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_scope_cannot_be_relaxed_after_freeze(tmp_path):
    work,spec=fixture(tmp_path)
    # Minimal baseline for this scope unit test, separate from approved publication.
    r.write(work/'geometry.json',{'SKU_01':{'back':{'path':str(work/'source.png'),'sha256':r.digest(work/'source.png')}}})
    r.write(work/'spec.json',{'SKU_01':{'back':spec}});r.freeze_scope(work,work/'spec.json')
    assert r.load_scope(work)['SKU_01']['back']['mode']=='perimeter'
    with pytest.raises(ValueError,match='already frozen'):r.freeze_scope(work,work/'spec.json')
    frozen=r.read(work/'lighting-scope.json');frozen['specs']['SKU_01']['back']['zones']['top']['min_L']=1
    r.write(work/'lighting-scope.json',frozen)
    with pytest.raises(ValueError,match='changed after review'):r.load_scope(work)


def perimeter_delivery(tmp):
    _,work,geometry,plan=setup(tmp)
    (tmp/'assets').mkdir()
    assets,perimeter=fixture(tmp/'assets')
    perimeter['source_sha256']=w.sha(geometry/'front.png')
    specs={'SKU_01':{f:dict(source_sha256=w.sha(geometry/(f+'.png')),action='copy',mode='unchanged',reason='Outside requested scope') for f in w.FACES}}
    specs['SKU_01']['back']['perimeter_exemption']=dict(reason='user_scope_excludes_perimeter',
        basis='Test task requests front only',source_sha256=w.sha(geometry/'back.png'))
    specs['SKU_01']['front']=perimeter
    r.write(work/'scope.json',specs);r.freeze_scope(work,work/'scope.json')
    a=r.pixels(geometry/'front.png');a[...,:3]=160;Image.fromarray(a).save(work/'front.png')
    Image.new('L',(16,32),255).save(work/'allowed.png');Image.new('L',(16,32),0).save(work/'protected.png')
    (work/'job.json').write_text('{}')
    entry=dict(action='aluminum',reason='Entire front metal perimeter',output=str(work/'front.png'),
               allowed_mask=str(work/'allowed.png'),protected_mask=str(work/'protected.png'),job=str(work/'job.json'))
    review=r.evidence_images(perimeter,geometry/'front.png',work/'front.png',work/'qa')
    review['checks']={c:True for c in r.CHECKS};r.write(work/'qa/review.json',review)
    entry['perimeter_review']=str(work/'qa/review.json');plan['SKU_01']['front']=entry
    return work,plan


def test_perimeter_publication_records_separate_effect_status(tmp_path):
    work,plan=perimeter_delivery(tmp_path);w.save(work/'plan.json',plan)
    w.publish(work,work/'plan.json',tmp_path/'out')
    report=r.read(work/'validation.json')
    assert report['geometry_status']=='approved'
    assert report['lighting_status']=='pixel_constraints_passed'
    assert report['perimeter_status']=='passed'
    assert len(report['skus']['SKU_01']['front']['perimeter_effect'])==8


@pytest.mark.parametrize('fault',['copied_pixels','missing_review','scope_mask_changed'])
def test_perimeter_failure_never_creates_delivery(tmp_path,fault):
    work,plan=perimeter_delivery(tmp_path)
    if fault=='copied_pixels':w.shutil.copy2(work/'geometry/SKU_01/front.png',work/'front.png')
    elif fault=='missing_review':plan['SKU_01']['front'].pop('perimeter_review')
    else:
        spec=r.load_scope(work)['SKU_01']['front'];Image.new('L',(16,32),255).save(spec['zones']['top']['mask'])
    w.save(work/'plan.json',plan)
    with pytest.raises(ValueError):w.publish(work,work/'plan.json',tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_aluminum_cannot_skip_front_rim_just_because_screen_is_protected(tmp_path):
    _,work,geometry,plan=setup(tmp_path)
    specs={'SKU_01':{f:dict(action='copy',mode='unchanged',reason='Protect glass',
              source_sha256=w.sha(geometry/(f+'.png'))) for f in w.FACES}}
    specs['SKU_01']['left'].update(action='aluminum',mode='regional',protection_policy='reviewed_regions',protection_basis='Synthetic test')
    r.write(work/'spec.json',specs)
    with pytest.raises(ValueError,match='explicit source-bound exemption'):
        r.freeze_scope(work,work/'spec.json')
    assert not (work/'lighting-scope.json').exists()


def test_regional_local_check_rejects_dark_patch_and_binds_mask(tmp_path):
    work, spec = fixture(tmp_path)
    spec.update(mode='regional', local_checks={'antenna-end': spec['zones']['bottom_left']})
    spec.pop('zones'); spec.pop('required_metal_mask')
    allowed = np.ones((32,16),bool)
    assert r.measure_local(spec,work/'source.png',work/'good.png',allowed,~allowed)
    with pytest.raises(ValueError,match='Local feature effect failed'):
        r.measure_local(spec,work/'source.png',work/'source.png',allowed,~allowed)
    r.write(work/'geometry.json',{'SKU_01':{'left':{'path':str(work/'source.png'),'sha256':r.digest(work/'source.png')}}})
    r.write(work/'scope.json',{'SKU_01':{'left':spec}});r.freeze_scope(work,work/'scope.json')
    review = reviewed(work,spec)
    r.check_review(spec,work/'source.png',work/'good.png',review)
    Image.new('L',(16,32),255).save(spec['local_checks']['antenna-end']['mask'])
    with pytest.raises(ValueError,match='changed after scope freeze'):r.load_scope(work)


@pytest.mark.parametrize('fault',['dark','missing_review','none'])
def test_regional_publication_checks_effect_and_visual_review(tmp_path,fault):
    _,work,geometry,plan=setup(tmp_path)
    specs={'SKU_01':{f:dict(action='copy',mode='unchanged',reason='Outside local task',source_sha256=w.sha(geometry/(f+'.png'))) for f in w.FACES}}
    for face in ('front','back'):
        specs['SKU_01'][face]['perimeter_exemption']=dict(reason='user_scope_excludes_perimeter',basis='Local antenna repair only',source_sha256=w.sha(geometry/(face+'.png')))
    Image.new('L',(16,32),255).save(work/'allowed.png');Image.new('L',(16,32),0).save(work/'protected.png')
    spec=specs['SKU_01']['left'];spec.update(action='aluminum',mode='regional',protection_policy='sticker_only',sticker_mask=str(work/'protected.png'),local_checks={'antenna-end':dict(mask=str(work/'allowed.png'),min_L=50,max_L=80)})
    r.write(work/'scope.json',specs);r.freeze_scope(work,work/'scope.json')
    a=r.pixels(geometry/'left.png');a[...,:3]=40 if fault=='dark' else 160;Image.fromarray(a).save(work/'left.png');(work/'job.json').write_text('{}')
    entry=dict(action='aluminum',reason='Local antenna repair',output=str(work/'left.png'),allowed_mask=str(work/'allowed.png'),protected_mask=str(work/'protected.png'),job=str(work/'job.json'))
    review=r.evidence_images(spec,geometry/'left.png',work/'left.png',work/'qa');review['checks']={c:True for c in r.CHECKS};r.write(work/'qa/review.json',review)
    if fault!='missing_review':entry['local_review']=str(work/'qa/review.json')
    plan['SKU_01']['left']=entry;w.save(work/'plan.json',plan)
    if fault=='none':
        w.publish(work,work/'plan.json',tmp_path/'out')
        assert r.read(work/'validation.json')['skus']['SKU_01']['left']['local_status']=='effect_and_visual_review_passed'
    else:
        with pytest.raises(ValueError,match='Local feature'):w.publish(work,work/'plan.json',tmp_path/'out')
        assert not (tmp_path/'out').exists()
