"""Dispatch must not turn a rejected saved continuation into fresh processing."""
from pathlib import Path

import pytest

from phone_texture import automatic, saved_review


def task(tmp_path, descriptor):
    return ('sku14', str(tmp_path/'source'), str(tmp_path/'case'), 'cpu', 1,
            'iphone17pro', 'none', 'preserve-source', 'accepted-shadow-lift-v2',
            None, 'appearance', True, descriptor)


def no_fresh_intake(*args, **kwargs):
    pytest.fail('Saved route reached normal pipeline intake')


def test_bound_saved_route_does_not_enter_normal_pipeline(tmp_path, monkeypatch):
    descriptor={'manifest_sha256':'a'*64}
    seen={}
    def restore(value, **kwargs):
        seen.update(kwargs)
        assert value == descriptor
        kwargs['case'].mkdir()
        return dict(sku=kwargs['sku'], status='source_preserved_needs_review',
            six_faces_generated=True, six_faces_processed=False,
            quality_checks_complete=False, visual_review_passed=False,
            model_uv_verified=False, structural_review_applied_faces=['left','right'])
    monkeypatch.setattr(saved_review,'process_saved',restore)
    monkeypatch.setattr(automatic.w,'intake',no_fresh_intake)
    result=automatic.process_one(task(tmp_path,descriptor))
    assert result['execution_route']=='saved_review'
    assert result['six_faces_generated'] and not result['six_faces_processed']
    assert result['quality_checks_complete'] is False and result['model_uv_verified'] is False
    assert result['worker_pid']>0 and result['worker_parent_pid']>0
    assert result['opencv_runtime']['effective_threads']==1
    assert seen['options']==dict(model='iphone17pro',protection_policy='none',
        review_fallback='preserve-source',lighting_profile='accepted-shadow-lift-v2',
        bottom_alignment='appearance',source_body_reference_review=True)


def test_rejected_saved_record_never_falls_back_to_segmentation(tmp_path,monkeypatch):
    def reject(*args,**kwargs):
        raise ValueError('Stale saved source/evidence')
    monkeypatch.setattr(saved_review,'process_saved',reject)
    monkeypatch.setattr(automatic.w,'intake',no_fresh_intake)
    result=automatic.process_one(task(tmp_path,{'present':True}))
    assert result['status']=='failed' and not result['six_faces_generated']
    assert result['execution_route']=='saved_review_rejected'
    assert result['saved_review_requested'] is True
    assert 'Stale saved' in result['error']
    assert not (tmp_path/'case/candidate').exists()


def test_existing_case_is_not_overwritten_by_saved_route(tmp_path,monkeypatch):
    case=tmp_path/'case';case.mkdir();sentinel=case/'result.json';sentinel.write_text('keep')
    monkeypatch.setattr(saved_review,'process_saved',lambda *a,**k: pytest.fail('Must reject before helper'))
    with pytest.raises(ValueError,match='already exists'):
        automatic.process_one(task(tmp_path,{'present':True}))
    assert sentinel.read_text()=='keep'


def fake_source(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();(source/'front.jpg').write_bytes(b'identity sentinel')
    monkeypatch.setattr(automatic.w,'face_files',lambda path:{f:Path(path)/(f+'.jpg') for f in automatic.pipeline.FACES})
    monkeypatch.setattr(automatic.concurrent.futures,'ProcessPoolExecutor',
        lambda *a,**k:pytest.fail('No worker may start after invalid review preflight'))
    return source


def test_changed_manifest_pin_rejects_before_work_or_workers(tmp_path,monkeypatch):
    source=fake_source(tmp_path,monkeypatch);manifest=tmp_path/'review.json';manifest.write_text('{}')
    monkeypatch.setattr(saved_review,'load_selected',lambda *a,**k:pytest.fail('Expected SHA must be checked first'))
    with pytest.raises(ValueError,match='manifest changed'):
        automatic.run(source,tmp_path/'work',tmp_path/'output',jobs=1,cpu_threads=1,
            model='iphone17pro',saved_review_manifest=manifest,saved_review_manifest_sha256='0'*64)
    assert not (tmp_path/'work').exists() and not (tmp_path/'output').exists()


def test_source_or_config_rejection_precedes_batch_creation(tmp_path,monkeypatch):
    source=fake_source(tmp_path,monkeypatch);manifest=tmp_path/'review.json';manifest.write_text('{}')
    def reject(path,sources,options,**kwargs):
        assert sources=={'SKU_01':source}
        assert kwargs['expected_manifest_sha256']==automatic.w.sha(manifest)
        raise ValueError('Saved review options/source mismatch')
    monkeypatch.setattr(saved_review,'load_selected',reject)
    with pytest.raises(ValueError,match='options/source mismatch'):
        automatic.run(source,tmp_path/'work',tmp_path/'output',jobs=1,cpu_threads=1,
            model='iphone17pro',saved_review_manifest=manifest)
    assert not (tmp_path/'work').exists() and not (tmp_path/'output').exists()


def test_orphan_manifest_sha_is_not_ignored(tmp_path):
    with pytest.raises(ValueError,match='requires a manifest path'):
        automatic.run(tmp_path/'missing',tmp_path/'work',tmp_path/'output',
            saved_review_manifest_sha256='a'*64)


def test_required_missing_geometry_stops_before_model_import(tmp_path,monkeypatch):
    import builtins
    from phone_texture import stage_processing
    monkeypatch.setattr(automatic.w,'intake',lambda *a,**k:{'skus':{'sku14':{}}})
    monkeypatch.setattr(stage_processing,'resume_geometry',lambda *a,**k:None)
    original_import=builtins.__import__
    def guarded_import(name,*args,**kwargs):
        if name in ('torch','phone_normalizer.api'):
            pytest.fail('Required-resume failure must precede model import')
        return original_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded_import)
    values=list(task(tmp_path,None));values[9]=str(tmp_path/'missing-checkpoint');values.append(True)
    result=automatic.process_one(tuple(values))
    assert result['status']=='failed' and not result['six_faces_generated']
    assert 'fresh segmentation was not started' in result['error']
