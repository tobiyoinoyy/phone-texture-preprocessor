"""Synthetic workflow checks only; no business-photo processing or AI calls."""
import copy
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from phone_texture import workflow as f, workspace as w, structure_review as s
from geometry_helpers import review_fixture
from test_structure_review import fixture


def batch(tmp):
    source, files, spec, specfile = fixture(tmp)
    work, output = tmp / 'work', tmp / 'delivery'
    f.create(source, work, output, input_kind='existing')
    assert f.run(work)['stage'] == 'geometry_review'
    config = w.read(work / 'workflow.json')
    config['geometry']['review'] = str(review_fixture(work / 'geometry-records'))
    f.save(work / 'workflow.json', config)
    assert f.run(work)['stage'] == 'lighting_scope'
    baseline = w.read(work / 'geometry.json')
    scope = {'A': {face: dict(source_sha256=item['sha256'], action='copy', mode='unchanged',
                              reason='Synthetic unchanged lighting fixture') for face, item in baseline['A'].items()}}
    w.save(work / 'scope.json', scope)
    jobs = {'A': {face: dict(action='copy') for face in w.FACES}}
    w.save(work / 'jobs.json', jobs)
    config['lighting'].update(scope='scope.json', jobs='jobs.json')
    f.save(work / 'workflow.json', config)
    return work, output, config, files, spec, specfile


def approve_structure(work, config, specfile):
    config['structure']['spec'] = str(specfile)
    f.save(work / 'workflow.json', config)
    assert f.run(work)['stage'] == 'structure_review'
    review = w.read(work / 'structure-records/review-template.json')
    review['reviewer'] = 'Synthetic integration test'
    for sides in review['items'].values():
        for side in sides:
            sides[side] = dict.fromkeys(s.CHECKS, True)
            sides[side]['notes'] = 'Known synthetic stripes and button range inspected in the fixture.'
    w.save(work / 'structure-review.json', review)
    config['structure']['review'] = 'structure-review.json'
    f.save(work / 'workflow.json', config)


def test_one_workflow_resumes_through_all_gates_and_keeps_sources(tmp_path):
    work, output, config, files, spec, specfile = batch(tmp_path)
    assert config['alignment_basis'] == 'back_image'
    assert f.run(work)['stage'] == 'structure'
    assert not output.exists()
    approve_structure(work, config, specfile)
    assert f.run(work)['status'] == 'completed'
    assert len(list((output / 'A').iterdir())) == 6
    assert f.run(work)['status'] == 'completed'
    for face in ('front', 'back', 'top', 'bottom'):
        assert w.sha(output / 'A' / (face + '.png')) == w.sha(files['A'][face])
    record = w.read(work / 'validation.json')['structure']['checks']['A']['left']
    assert record['alignment_basis'] == 'back_image'
    assert record['back_reference_sha256'] == w.sha(files['A']['back'])
    assert record['button_pixels_unchanged']
    image = s.pixels(output / 'A/left.png'); image[10, 1, 0] += 1
    Image.fromarray(image).save(output / 'A/left.png')
    with pytest.raises(ValueError, match='已发布文件'):
        f.run(work)


def test_concurrent_run_or_changed_contract_cannot_sneak_past_gates(tmp_path):
    work, output, config, files, spec, specfile = batch(tmp_path)
    with f.locked(work):
        with pytest.raises(ValueError, match='正在运行'):
            f.run(work)
    config['alignment_basis'] = 'front_back_images'
    f.save(work / 'workflow.json', config)
    with pytest.raises(ValueError, match='对齐基准'):
        f.run(work)
    assert not output.exists()


def test_explicit_front_back_override_is_not_the_back_default(tmp_path):
    work, output, config, files, spec, specfile = batch(tmp_path)
    spec['A']['left']['basis'] = 'front_back_images'
    w.save(specfile, spec)
    config['structure']['spec'] = str(specfile)
    f.save(work / 'workflow.json', config)
    with pytest.raises(ValueError, match='对齐基准'):
        f.run(work)
    assert not (work / 'structure-records').exists()


def test_each_sku_uses_its_own_back_by_default(tmp_path):
    source, files, spec, specfile = fixture(tmp_path)
    folder = source / 'B'; folder.mkdir()
    for face in w.FACES:
        image = s.pixels(files['A'][face])
        if face == 'back':
            image[..., :3] = 150; image[64:76, :, :3] = 70; image[184:196, :, :3] = 70
        Image.fromarray(image).save(folder / (face + '.png'))
    spec['B'] = copy.deepcopy(spec['A'])
    for side in ('left', 'right'):
        cfg = spec['B'][side]
        cfg['source_sha256'] = w.sha(folder / (side + '.png'))
        cfg['back'].update(sha256=w.sha(folder / 'back.png'), antenna_rows=[63.5, 75.5, 183.5, 195.5],
                           search_windows=[[58, 69], [70, 81], [178, 189], [190, 201]])
        for m in cfg['measurements'].values():
            m['search_windows'] = [[58, 69], [70, 81], [178, 189], [190, 201]]
    w.save(specfile, spec)
    report = s.prepare(source, specfile, tmp_path / 'aligned')
    a = report['checks']['A']['left']['checks']['center']['target_rows']
    b = report['checks']['B']['left']['checks']['center']['target_rows']
    assert np.allclose(np.array(b) - a, 4)


def test_changed_task_does_not_reuse_old_geometry(tmp_path):
    work, output, config, files, spec, specfile = batch(tmp_path)
    config['geometry']['device'] = 'cpu'
    f.save(work / 'workflow.json', config)
    with pytest.raises(ValueError, match='几何参数'):
        f.run(work)
    assert not output.exists()


def test_real_aluminum_engine_is_reachable_through_unified_workflow(tmp_path):
    work, output, config, files, spec, specfile = batch(tmp_path)
    baseline = w.read(work / 'geometry.json'); src = Path(baseline['A']['left']['path'])
    mask = np.zeros((256, 60), np.uint8)
    Image.fromarray(mask).save(work / 'no-sticker.png')
    scope = w.read(work / 'scope.json')
    scope['A']['left'].update(action='aluminum', mode='regional', protection_policy='sticker_only',
                              sticker_mask=str(work / 'no-sticker.png'))
    for face in ('front', 'back'):
        scope['A'][face]['perimeter_exemption'] = dict(reason='user_scope_excludes_perimeter',
            basis='Synthetic test only processes the side', source_sha256=baseline['A'][face]['sha256'])
    w.save(work / 'scope.json', scope)
    job = dict(defaults=dict(sigma=[3, 3]), views=[dict(name='left', source=str(src),
        reference_boxes=[[8, 82, 20, 88]], domains=[dict(name='metal', shapes=[dict(type='rectangle', xy=[0, 80, 59, 95])])])])
    w.save(work / 'aluminum-job.json', job)
    jobs = w.read(work / 'jobs.json'); jobs['A']['left'] = dict(action='aluminum', job='aluminum-job.json')
    w.save(work / 'jobs.json', jobs)
    assert f.run(work)['stage'] == 'structure'
    assert (work / 'lighting-records/A/left/workflow-proof.json').exists()
    assert f.run(work)['stage'] == 'structure'
    job['defaults']['sigma'] = [4, 4]; w.save(work / 'aluminum-job.json', job)
    with pytest.raises(ValueError, match='光照参数'):
        f.run(work)
    assert not output.exists()
