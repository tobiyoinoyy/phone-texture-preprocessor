"""Verify delivery boundaries using real PNG changes, not instruction text."""
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

MODULE = Path(__file__).resolve().parents[1]/'skills/phone-six-view-process/scripts/workspace.py'
spec = importlib.util.spec_from_file_location('workflow', MODULE)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


def setup(tmp):
    source = tmp/'source'
    source.mkdir()
    for f in w.FACES:
        Image.new('RGBA', (16, 32), (120, 100, 90, 255)).save(source/(f+'.png'))
    work = tmp/'process'
    w.intake(source, work, delivery_goal="photos", goal_reason="Test user requests lighting-only photos")
    geometry = work/'geometry'/'SKU_01'
    geometry.mkdir(parents=True)
    for f in w.FACES:
        w.shutil.copy2(source/(f+'.png'), geometry/(f+'.png'))
    from geometry_helpers import approved_fixture
    proof = approved_fixture(geometry.parent, work)
    w.checkpoint(work, geometry.parent, proof)
    plan = {'SKU_01': {f: {'action':'copy', 'reason':'Glass/no correction needed'} for f in w.FACES}}
    return source, work, geometry, plan


def scope(work, geometry, plan):
    spec = {'SKU_01': {f: dict(source_sha256=w.sha(geometry/(f+'.png')),
        action=e['action'], mode='unchanged' if e['action']=='copy' else 'regional',
        reason=e['reason'], protection_policy='reviewed_regions',
        protection_basis='Synthetic task explicitly retains protected features') for f,e in plan['SKU_01'].items()}}
    for face in ('front','back'):
        spec['SKU_01'][face]['perimeter_exemption'] = dict(reason='user_scope_excludes_perimeter',
            basis='Test task explicitly requests left only', source_sha256=w.sha(geometry/(face+'.png')))
    w.save(work/'scope-input.json', spec)
    w.freeze_scope(work, work/'scope-input.json')


def correction(work, geometry, plan):
    a = np.array(Image.open(geometry/'left.png'))
    a[10, 5, :3] += 10
    Image.fromarray(a).save(work/'left.png')
    allowed = np.zeros(a.shape[:2], dtype=np.uint8)
    allowed[10,5] = 255
    Image.fromarray(allowed).save(work/'allowed.png')
    Image.fromarray(np.zeros_like(allowed)).save(work/'protected.png')
    (work/'job.json').write_text('{}')
    plan['SKU_01']['left'] = dict(action='aluminum', reason='Metal lightness', output=str(work/'left.png'),
        allowed_mask=str(work/'allowed.png'), protected_mask=str(work/'protected.png'), job=str(work/'job.json'))


def test_valid_complete_six_face_delivery(tmp_path):
    source, work, geometry, plan = setup(tmp_path)
    correction(work, geometry, plan)
    scope(work, geometry, plan)
    w.save(work/'plan.json', plan)
    report = w.publish(work, work/'plan.json', tmp_path/'out')
    assert len(list((tmp_path/'out'/'SKU_01').glob('*.png'))) == 6
    assert report['SKU_01']['left']['changed_pixels'] == 1
    assert w.sha(tmp_path/'out'/'SKU_01'/'front.png') == w.sha(source/'front.png')


@pytest.mark.parametrize('failure', ['alpha', 'outside', 'protected', 'source', 'geometry', 'missing'])
def test_rejects_invalid_delivery_without_publishing(tmp_path, failure):
    source, work, geometry, plan = setup(tmp_path)
    correction(work, geometry, plan)
    scope(work, geometry, plan)
    if failure in ('alpha', 'outside'):
        a = np.array(Image.open(work/'left.png'))
        a[0,0,3 if failure == 'alpha' else 0] -= 10
        Image.fromarray(a).save(work/'left.png')
    elif failure == 'protected':
        Image.new('L', (16,32), 255).save(work/'protected.png')
    elif failure in ('source','geometry'):
        Image.new('RGBA', (16,32), 'white').save((source if failure == 'source' else geometry)/'back.png')
    else:
        del plan['SKU_01']['bottom']
    w.save(work/'plan.json', plan)
    with pytest.raises(ValueError):
        w.publish(work, work/'plan.json', tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_batch_selection_and_duplicate_face(tmp_path):
    src=tmp_path/'batch'
    for sku in ['A','B']:
        (src/sku).mkdir(parents=True)
        for face in w.FACES:
            Image.new('RGB',(16,32),'gray').save(src/sku/(face+'.jpg'))
    manifest=w.intake(src,tmp_path/'work',skus=['B'])
    assert set(manifest['skus'])=={'B'}
    Image.new('RGB',(16,32),'gray').save(src/'A'/'front.png')
    with pytest.raises(ValueError, match='Duplicate'):
        w.intake(src,tmp_path/'work2',skus=['A'])
    with pytest.raises(ValueError):
        w.intake(src,src/'nested',skus=['B'])


def boundary_fixture(work, geometry, plan):
    correction(work, geometry, plan)
    feature = np.zeros((32,16), dtype=np.uint8)
    feature[3:8, 4:9] = 255
    metal = np.zeros_like(feature)
    metal[8:11, 4:9] = 255
    Image.fromarray(feature).save(work/'paper.png')
    Image.fromarray(metal).save(work/'metal.png')
    Image.fromarray(feature).save(work/'protected.png')
    Image.fromarray(255-feature).save(work/'allowed.png')
    entry = plan['SKU_01']['left']
    entry.update(contains_sticker=True, boundary_checks=[{
        'source_sha256':w.sha(geometry/'left.png'), 'feature_mask':str(work/'paper.png'),
        'adjacent_metal_mask':str(work/'metal.png')}])
    return entry


@pytest.mark.parametrize('failure',['oversized','missing_paper','missing_review','stale_source'])
def test_paper_boundary_guard_rejects_wrong_protection(tmp_path, failure):
    _, work, geometry, plan = setup(tmp_path)
    entry = boundary_fixture(work, geometry, plan)
    scope(work, geometry, plan)
    if failure == 'oversized':
        # No changes inside this wrong protection: the old invariant alone passes.
        protected=np.array(Image.open(work/'protected.png'))
        protected[8:10,4:9]=255
        Image.fromarray(protected).save(work/'protected.png')
        allowed=np.array(Image.open(work/'allowed.png'))
        allowed[8:10,4:9]=0
        Image.fromarray(allowed).save(work/'allowed.png')
    elif failure == 'missing_paper':
        Image.new('L',(16,32),0).save(work/'protected.png')
    elif failure == 'missing_review':
        entry.pop('boundary_checks')
    else:
        entry['boundary_checks'][0]['source_sha256']='old-source'
    w.save(work/'plan.json',plan)
    with pytest.raises(ValueError):
        w.publish(work,work/'plan.json',tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_accurate_paper_boundary_publishes(tmp_path):
    _,work,geometry,plan=setup(tmp_path)
    boundary_fixture(work,geometry,plan)
    scope(work,geometry,plan)
    w.save(work/'plan.json',plan)
    report=w.publish(work,work/'plan.json',tmp_path/'out')
    checks=report['SKU_01']['left']['boundary_checks'][0]
    assert checks['frozen_metal_pixels']==0
    assert checks['exposed_feature_pixels']==0
