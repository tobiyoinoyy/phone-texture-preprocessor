"""Metadata/parser/command controls. No real child, server, inference or images."""
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from phone_texture import automatic, cli, frame_seams, web_app as web


def manifest_file(tmp_path):
    path = tmp_path/'saved-review.json'
    path.write_text(json.dumps(dict(schema='metadata_test_only')))
    return path


def capture_auto(monkeypatch):
    calls = []
    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return dict(status='review_candidates_needs_visual_review', processed=0,
                    generated=1, total=1, visual_approved=0)
    monkeypatch.setattr(automatic, 'run', run)
    return calls


def test_real_auto_parser_keeps_ordinary_defaults(monkeypatch, tmp_path):
    calls = capture_auto(monkeypatch)
    assert cli.main(['auto', str(tmp_path), '--model', 'iphone17pro']) == 2
    options = calls[0][1]
    assert options['source_body_reference_review'] is False
    assert options['require_geometry_resume'] is False
    assert options['saved_review_manifest'] is None
    assert options['saved_review_manifest_sha256'] is None
    assert options['protection_policy'] == 'none'


def test_real_auto_parser_passes_explicit_manifest_reference_and_pin(monkeypatch, tmp_path):
    path = manifest_file(tmp_path); digest = web.w.sha(path)
    calls = capture_auto(monkeypatch)
    assert cli.main(['auto', str(tmp_path), '--model', 'iphone17pro',
        '--source-body-reference-review', '--saved-review-manifest', str(path),
        '--saved-review-manifest-sha256', digest]) == 2
    options = calls[0][1]
    assert options['source_body_reference_review'] is True
    assert options['saved_review_manifest'] == path
    assert options['saved_review_manifest_sha256'] == digest


@pytest.mark.parametrize('kind', ['wrong', 'malformed', 'without_path'])
def test_cli_rejects_manifest_pin_before_auto_call(monkeypatch, tmp_path, kind):
    path = manifest_file(tmp_path); calls = capture_auto(monkeypatch)
    argv = ['auto', str(tmp_path), '--model', 'iphone17pro']
    if kind != 'without_path':
        argv += ['--saved-review-manifest', str(path)]
    argv += ['--saved-review-manifest-sha256', 'bad' if kind == 'malformed' else '0'*64]
    with pytest.raises(SystemExit) as error:
        cli.main(argv)
    assert error.value.code == 2
    assert calls == []


def test_web_public_parser_forwards_explicit_service_configuration(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(web, 'main', lambda args: calls.append(args) or 0)
    path = manifest_file(tmp_path); digest = web.w.sha(path)
    assert cli.main(['web', '--no-open', '--source-body-reference-review',
        '--saved-review-manifest', str(path), '--saved-review-manifest-sha256', digest]) == 0
    args = calls[0]
    assert '--source-body-reference-review' in args
    assert args[args.index('--saved-review-manifest')+1] == str(path)
    assert args[args.index('--saved-review-manifest-sha256')+1] == digest


def setup_app(tmp_path, monkeypatch, **options):
    app = web.App(tmp_path/'home', **options)
    source = tmp_path/'raw'; source.mkdir()
    files = {}
    for face in web.FACES:
        path = source/(face+'.jpg'); path.write_bytes(('raw '+face).encode()); files[face] = path
    scan = dict(id='scan', source=str(source), single=True, rows=[dict(name='sku', valid=True,
        recognition=dict(images=[dict(file=str(path), sha256=web.w.sha(path)) for path in files.values()]))])
    app.scans['scan'] = scan
    monkeypatch.setattr(web.w, 'face_files', lambda folder: files)
    monkeypatch.setattr(frame_seams, 'profile', lambda model: ({}, {}))
    monkeypatch.setattr(web, 'host_sample', lambda: dict(pressure=1, swap=0, available=10*1024**3))
    monkeypatch.setattr(web.shutil, 'disk_usage', lambda path: SimpleNamespace(free=10*1024**3))
    monkeypatch.setattr(web.threading.Thread, 'start', lambda self: None)
    return app, scan


def start(app, **options):
    return app.start(dict(scan_id='scan', skus=['sku'], model='iphone17pro', concurrency=1, **options))


def test_web_start_without_configuration_does_not_read_saved_experiments(tmp_path, monkeypatch):
    app, _ = setup_app(tmp_path, monkeypatch)
    result = start(app)
    assert result['source_body_reference_review'] is False
    assert result['saved_review_manifest'] is None
    assert result['saved_review_manifest_sha256'] is None


@pytest.mark.parametrize('override', [False, True])
def test_web_new_job_inherits_config_or_honors_explicit_disable(tmp_path, monkeypatch, override):
    path = manifest_file(tmp_path)
    app, _ = setup_app(tmp_path, monkeypatch,
        source_body_reference_review=True, saved_review_manifest=path)
    options = dict(source_body_reference_review=False, saved_review_manifest=None) if override else {}
    result = start(app, **options)
    saved = web.read(app.registry/(result['id']+'.json'))
    assert result['source_body_reference_review'] is (not override)
    assert saved['source_body_reference_review'] is (not override)
    assert saved['saved_review_manifest'] == (None if override else str(path.resolve()))
    assert saved['saved_review_manifest_sha256'] == (None if override else web.w.sha(path))


def test_web_explicit_request_pins_manifest(tmp_path, monkeypatch):
    path = manifest_file(tmp_path); app, _ = setup_app(tmp_path, monkeypatch)
    result = start(app, saved_review_manifest=str(path), source_body_reference_review=True)
    assert result['saved_review_manifest'] == str(path.resolve())
    assert result['saved_review_manifest_sha256'] == web.w.sha(path)
    assert result['source_body_reference_review'] is True


@pytest.mark.parametrize('invalid', [None, 1, 'true', [], {}])
def test_web_does_not_coerce_source_reference_booleans(tmp_path, monkeypatch, invalid):
    app, _ = setup_app(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='布尔'):
        start(app, source_body_reference_review=invalid)
    assert app.jobs == {}


def test_changed_configured_manifest_is_rejected_before_job_creation(tmp_path, monkeypatch):
    path = manifest_file(tmp_path)
    app, _ = setup_app(tmp_path, monkeypatch, saved_review_manifest=path)
    path.write_text('changed')
    with pytest.raises(ValueError, match='已改变'):
        start(app)
    assert app.jobs == {}


def test_retry_legacy_false_and_no_manifest_do_not_inherit_new_service_defaults(tmp_path, monkeypatch):
    path = manifest_file(tmp_path)
    app, scan = setup_app(tmp_path, monkeypatch,
        source_body_reference_review=True, saved_review_manifest=path)
    old_root = app.runs/'old'; (old_root/'work/cases/sku').mkdir(parents=True)
    web.save(old_root/'work/cases/sku/result.json', dict(sku='sku', status='failed'))
    old = dict(id='old', status='failed', source=scan['source'], model='iphone17pro', names=['sku'],
        concurrency=1, root=str(old_root), work=str(old_root/'work'), output=str(tmp_path/'exports/old-output'))
    app.jobs['old'] = old
    monkeypatch.setattr(app, 'scan', lambda value: scan)
    new = app.retry(dict(id='old'))
    assert new['source_body_reference_review'] is False
    assert new['saved_review_manifest'] is None
    assert new['saved_review_manifest_sha256'] is None
    assert new['parent_job_id'] == 'old'


def worker_job(app, tmp_path, **options):
    root = app.runs/'worker'; root.mkdir()
    job = dict(id='worker', root=str(root), work=str(root/'work'), output=str(tmp_path/'out'),
        source=str(tmp_path/'raw'), model='iphone17pro', names=['sku'], single=True,
        concurrency=1, status='starting')
    job.update(options); app.jobs['worker'] = job
    app.cancel_events['worker'] = threading.Event()
    return job


def capture_process(monkeypatch):
    calls = []
    class Process:
        returncode = 2
        def __init__(self, command, **options): calls.append((command, options))
        def poll(self): return 2
    monkeypatch.setattr(web.subprocess, 'Popen', Process)
    return calls


def test_actual_worker_passes_persisted_options_to_real_parser(tmp_path, monkeypatch):
    path = manifest_file(tmp_path); digest = web.w.sha(path)
    app, _ = setup_app(tmp_path, monkeypatch)
    job = worker_job(app, tmp_path, source_body_reference_review=True,
        saved_review_manifest=str(path), saved_review_manifest_sha256=digest,
        protection_policy='none', lighting_profile='accepted-shadow-lift-v2')
    processes = capture_process(monkeypatch)
    app.run(job)
    assert len(processes) == 1
    command = processes[0][0]
    assert command[:4] == [sys.executable, '-m', 'phone_texture', 'auto']
    assert command.count('--source-body-reference-review') == 1
    assert command[command.index('--saved-review-manifest')+1] == str(path)
    assert command[command.index('--saved-review-manifest-sha256')+1] == digest
    calls = capture_auto(monkeypatch)
    assert cli.main(command[3:]) == 2
    assert calls[0][1]['saved_review_manifest_sha256'] == digest
    assert calls[0][1]['source_body_reference_review'] is True


@pytest.mark.parametrize('failure', ['changed', 'unpinned', 'nonbool', 'wrong_profile'])
def test_worker_fails_before_child_for_invalid_persisted_configuration(tmp_path, monkeypatch, failure):
    path = manifest_file(tmp_path); digest = web.w.sha(path)
    app, _ = setup_app(tmp_path, monkeypatch)
    options = dict(source_body_reference_review=True, saved_review_manifest=str(path),
                   saved_review_manifest_sha256=digest)
    if failure == 'changed': path.write_text('changed')
    elif failure == 'unpinned': options.pop('saved_review_manifest_sha256')
    elif failure == 'nonbool': options['source_body_reference_review'] = 'true'
    elif failure == 'wrong_profile': options['lighting_profile'] = 'baseline'
    job = worker_job(app, tmp_path, **options); processes = capture_process(monkeypatch)
    app.run(job)
    assert processes == []
    assert job['status'] == 'failed'


def test_cli_explicit_required_resume_passes_true_without_changing_defaults(tmp_path, monkeypatch):
    calls = capture_auto(monkeypatch)
    assert cli.main(['auto', str(tmp_path), '--model', 'iphone17pro',
        '--resume-geometry-from', str(tmp_path/'checkpoint'), '--require-geometry-resume']) == 2
    assert calls[0][1]['require_geometry_resume'] is True
    assert calls[0][1]['resume_geometry_from'] == tmp_path/'checkpoint'
