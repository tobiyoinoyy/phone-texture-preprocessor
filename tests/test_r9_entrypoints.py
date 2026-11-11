"""R9 metadata-only controls; real Web execution belongs to the root guard run."""
import copy
import json
import pytest
from phone_texture import cli, web_app as web
from phone_texture.candidate_manifest import candidate_manifest_item
from test_saved_review_entrypoints import setup_app, start, capture_auto, worker_job, capture_process
from test_saved_review_display import saved_review_row


def checkpoint(tmp_path):
    path = tmp_path/'checkpoint'; (path/'cases').mkdir(parents=True)
    return path


def test_auto_defaults_are_serial_and_review_disabled(monkeypatch, tmp_path):
    calls = capture_auto(monkeypatch)
    assert cli.main(['auto', str(tmp_path), '--model', 'iphone17pro']) == 2
    options = calls[0][1]
    assert options['jobs'] == options['cpu_threads'] == 1
    assert options['side_candidate_review'] is False


def test_auto_explicit_side_review_reaches_api(monkeypatch, tmp_path):
    calls = capture_auto(monkeypatch)
    assert cli.main(['auto', str(tmp_path), '--model', 'iphone17pro',
        '--side-candidate-review', '--source-body-reference-review']) == 2
    assert calls[0][1]['side_candidate_review'] is True


def test_cli_web_forwards_new_job_configuration(monkeypatch, tmp_path):
    calls = []; path = checkpoint(tmp_path)
    monkeypatch.setattr(web, 'main', lambda args: calls.append(args) or 0)
    assert cli.main(['web', '--no-open', '--side-candidate-review',
        '--resume-geometry-from', str(path), '--require-geometry-resume']) == 0
    assert '--side-candidate-review' in calls[0]
    assert '--require-geometry-resume' in calls[0]
    assert calls[0][calls[0].index('--resume-geometry-from')+1] == str(path)


def test_new_web_job_resumes_checkpoint_without_fake_retry(tmp_path, monkeypatch):
    app, _ = setup_app(tmp_path, monkeypatch); path = checkpoint(tmp_path)
    result = start(app, side_candidate_review=True, source_body_reference_review=True,
        resume_geometry_from=str(path), require_geometry_resume=True)
    persisted = web.read(app.registry/(result['id']+'.json'))
    assert 'parent_job_id' not in persisted
    assert persisted['side_candidate_review'] is True
    assert persisted['require_geometry_resume'] is True
    assert persisted['resume_geometry_from'] == str(path.resolve())
    assert persisted['cpu_threads'] == persisted['inference_slots'] == persisted['concurrency'] == 1


def test_new_web_job_service_defaults_remain_explicitly_disableable(tmp_path, monkeypatch):
    path = checkpoint(tmp_path)
    app, _ = setup_app(tmp_path, monkeypatch, side_candidate_review=True,
        require_geometry_resume=True, resume_geometry_from=path)
    result = start(app, side_candidate_review=False, require_geometry_resume=False,
                   resume_geometry_from=None)
    assert result['side_candidate_review'] is False
    assert result['resume_geometry_from'] is None
    assert result['require_geometry_resume'] is False


@pytest.mark.parametrize('field', ['side_candidate_review', 'require_geometry_resume'])
@pytest.mark.parametrize('invalid', [None, 1, 'true'])
def test_review_and_resume_booleans_not_coerced(tmp_path, monkeypatch, field, invalid):
    app, _ = setup_app(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='布尔'):
        start(app, **{field: invalid})
    assert not app.jobs


@pytest.mark.parametrize('path', [None, 'missing', 'empty'])
def test_required_checkpoint_missing_rejected_before_job(tmp_path, monkeypatch, path):
    app, _ = setup_app(tmp_path, monkeypatch)
    value = None if path is None else tmp_path/path
    if path == 'empty': value.mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        start(app, require_geometry_resume=True, resume_geometry_from=str(value) if value else None)
    assert not app.jobs


@pytest.mark.parametrize('value', [0, 9, 2.0, '3'])
def test_web_refuses_concurrency_outside_one_to_eight(tmp_path, monkeypatch, value):
    app, _ = setup_app(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='1–8'):
        app.start(dict(scan_id='scan', skus=['sku'], model='iphone17pro', concurrency=value))


@pytest.mark.parametrize('n,slots,experimental', [(3, '2', False), (6, '2', True)])
def test_worker_passes_requested_concurrency(tmp_path, monkeypatch, n, slots, experimental):
    app, _ = setup_app(tmp_path, monkeypatch)
    job = worker_job(app, tmp_path, concurrency=n)
    monkeypatch.delenv('PHONE_TEXTURE_GUARD_R9', raising=False)
    processes = capture_process(monkeypatch); app.run(job)
    command = processes[0][0]
    assert command[command.index('--jobs')+1] == str(n)
    assert command[command.index('--inference-slots')+1] == slots
    assert ('--experimental-concurrency' in command) is experimental


def test_worker_passes_serial_and_new_checkpoint_flags(tmp_path, monkeypatch):
    app, _ = setup_app(tmp_path, monkeypatch); path = checkpoint(tmp_path)
    job = worker_job(app, tmp_path, side_candidate_review=True,
        source_body_reference_review=True, require_geometry_resume=True,
        resume_geometry_from=str(path))
    monkeypatch.delenv('PHONE_TEXTURE_GUARD_R9', raising=False)
    processes = capture_process(monkeypatch); app.run(job)
    command = processes[0][0]
    for flag in ('--jobs', '--inference-slots', '--cpu-threads'):
        assert command[command.index(flag)+1] == '1'
    assert '--side-candidate-review' in command and '--require-geometry-resume' in command
    assert command[command.index('--resume-geometry-from')+1] == str(path)
    assert job['worker_command'] == command and job['worker_returncode'] == 2


def fresh_row():
    row = saved_review_row()
    for key in ('saved_candidate_review_assembly', 'assembly_mode', 'review_continuation_saved'):
        row.pop(key)
    row['source_hashes_unchanged'] = True
    row['side_candidate_review_complete'] = True
    row['current_side_review'] = dict(schema='current_side_review/v1',
        execution_mode='fresh_current_sources', faces={face:dict(
            observation=dict(path='/fresh/'+face+'/observation.json', sha256='b'*64),
            lighting=dict(path='/fresh/'+face+'/lighting.json', sha256='c'*64, executed=True),
            structure=dict(path='/fresh/'+face+'/structure.json', sha256='d'*64, executed=True),
            selected_lighting=True, selected_structure=True) for face in ('left', 'right')})
    return row


@pytest.mark.parametrize('lighting', ['review_applied', 'complete', 'effect_warning'])
def test_fresh_closed_review_has_honest_caption_and_no_default_retry(lighting):
    row = fresh_row(); row['per_face_stages']['left']['lighting'] = lighting
    state = web.processing_state(row)
    assert state['current_side_review_ready'] is True
    assert state['review_continuation_ready'] is True
    assert state['review_continuation_saved'] is False
    assert state['retry_required'] is state['all_stages_complete'] is state['review_only'] is False
    manifest = candidate_manifest_item(row)
    assert manifest['current_side_review'] == row['current_side_review']
    assert manifest['side_candidate_review_complete'] is True


@pytest.mark.parametrize('kind', ['missing_face', 'not_executed', 'selected_mismatch',
    'strict_true', 'source_changed', 'bad_hash', 'single_selected', 'seam_claimed'])
def test_fresh_incomplete_or_mislabelled_candidate_still_retryable(kind):
    row = fresh_row()
    if kind == 'missing_face': del row['current_side_review']['faces']['right']
    elif kind == 'not_executed': row['current_side_review']['faces']['right']['lighting']['executed'] = False
    elif kind == 'selected_mismatch': row['per_face_stages']['right']['selected_sha256'] = 'e'*64
    elif kind == 'strict_true': row['strict_structure_verified'] = True
    elif kind == 'source_changed': row['source_hashes_unchanged'] = False
    elif kind == 'bad_hash': row['current_side_review']['faces']['left']['structure']['sha256'] = 'bad'
    elif kind == 'single_selected': row['current_side_review']['faces']['right']['selected_structure'] = False
    elif kind == 'seam_claimed': row['stage_status']['seam'] = 'complete'
    state = web.processing_state(row)
    assert state['current_side_review_ready'] is False
    assert state['retry_required'] is True


def test_old_saved_review_still_distinct():
    state = web.processing_state(saved_review_row())
    assert state['review_continuation_ready'] is state['review_continuation_saved'] is True
    assert state['current_side_review_ready'] is False


def test_guarded_worker_passes_inherited_fd_and_records_child_identity(tmp_path, monkeypatch):
    app, _ = setup_app(tmp_path, monkeypatch)
    job = worker_job(app, tmp_path)
    monkeypatch.setattr(web, 'r9_guard_context', lambda: dict(run_id='test-run', lock_fd=55))
    monkeypatch.setattr(web, 'r9_worker_identity', lambda proc: dict(
        child_pid=222, child_create_time=123., child_pgid=222, child_sid=222))
    processes = capture_process(monkeypatch); app.run(job)
    assert processes[0][1]['pass_fds'] == (55,)
    assert processes[0][1]['start_new_session'] is True
    assert job['guard_run_id'] == 'test-run'
    assert job['child_pid'] == job['child_pgid'] == job['child_sid'] == 222
    assert job['child_create_time'] == 123.


def test_guard_pause_duration_is_excluded_with_matching_run(tmp_path, monkeypatch):
    owner = tmp_path/'owner.json'
    owner.write_text(json.dumps(dict(run_id='test', wrapper_create_time=100., paused_seconds=7.)))
    monkeypatch.setattr(web, 'R9_GUARD_OWNER', owner)
    assert web.r9_paused_seconds(dict(run_id='test', wrapper_create_time=100., paused_at_start=2.)) == 5.
    with pytest.raises(ValueError, match='owner changed'):
        web.r9_paused_seconds(dict(run_id='other', wrapper_create_time=100., paused_at_start=0.))


def test_retry_preserves_side_options_and_uses_serial_default(tmp_path, monkeypatch):
    app, scan = setup_app(tmp_path, monkeypatch)
    old_root = app.runs/'old'; (old_root/'work/cases/sku').mkdir(parents=True)
    web.save(old_root/'work/cases/sku/result.json', dict(sku='sku', status='failed'))
    app.jobs['old'] = dict(id='old', status='failed', source=scan['source'], model='iphone17pro',
        names=['sku'], concurrency=3, root=str(old_root), work=str(old_root/'work'),
        output=str(tmp_path/'exports/old-output'), side_candidate_review=True,
        source_body_reference_review=True, require_geometry_resume=True)
    monkeypatch.setattr(app, 'scan', lambda value: scan)
    new = app.retry(dict(id='old'))
    assert new['concurrency'] == 1
    assert new['side_candidate_review'] is new['require_geometry_resume'] is True
    assert new['source_body_reference_review'] is True
    assert new['parent_job_id'] == 'old'


@pytest.mark.parametrize('with_manifest', [False, True])
def test_start_rejects_incompatible_side_review_before_job(tmp_path, monkeypatch, with_manifest):
    app, _ = setup_app(tmp_path, monkeypatch)
    options = dict(side_candidate_review=True)
    if with_manifest:
        path = tmp_path/'registry.json'; path.write_text('{}')
        options.update(source_body_reference_review=True, saved_review_manifest=str(path))
    with pytest.raises(ValueError, match='新侧面复核'):
        start(app, **options)
    assert not app.jobs
