"""Polling caches must observe changed files and preserve download verification."""
import hashlib
import os
from pathlib import Path

import pytest

from phone_texture import web_app as web
from test_review_bundle_web import fixture


def test_compact_snapshot_preserves_display_and_reuses_unchanged_reports(tmp_path, monkeypatch):
    app, job, report, row, bundle = fixture(tmp_path)
    app.package(job, report)
    path = Path(job['work'])/'cases'/row['sku']/'result.json'
    reads = []
    original = Path.read_text

    def observed_read(source, *args, **kwargs):
        if source == path:
            reads.append(source)
        return original(source, *args, **kwargs)

    full = app.snapshot(job)
    monkeypatch.setattr(Path, 'read_text', observed_read)
    compact = app.snapshot(job, compact=True)
    assert compact == app.snapshot(job, compact=True)
    assert len(reads) == 1
    for original_row, compact_row in zip(full['rows'], compact['rows']):
        expected = dict(original_row)
        expected.pop('bottom_alignment')
        expected['per_face_stages'] = {
            face: dict(lighting=details.get('lighting'))
            for face, details in expected['per_face_stages'].items()
        }
        assert compact_row == expected

    row.update(status='failed', six_faces_generated=False)
    web.save(path, row)
    states = {item['name']: item for item in app.snapshot(job, compact=True)['rows']}
    assert states[row['sku']]['status'] == 'failed'
    assert len(reads) == 2


def test_bundle_poll_cache_invalidates_and_download_always_rechecks(tmp_path, monkeypatch):
    app, job, report, row, path = fixture(tmp_path)
    app.package(job, report)
    original = web.w.sha
    hashes = []

    def observed_hash(source):
        if Path(source) == path:
            hashes.append(source)
        return original(source)

    monkeypatch.setattr(web.w, 'sha', observed_hash)
    app.snapshot(job, compact=True)
    app.snapshot(job, compact=True)
    assert len(hashes) == 1
    app.artifact(dict(job=job['id'], kind='bundle', sku=row['sku']))
    assert len(hashes) == 2

    # Same size and mtime must still invalidate the display cache via ctime.
    info = path.stat()
    path.write_bytes(b'x' * info.st_size)
    os.utime(path, ns=(info.st_atime_ns, info.st_mtime_ns))
    states = {item['name']: item for item in app.snapshot(job, compact=True)['rows']}
    assert not states[row['sku']]['portable_review_available']
    assert len(hashes) == 3
    with pytest.raises(ValueError):
        app.artifact(dict(job=job['id'], kind='bundle', sku=row['sku']))


def test_report_cache_recovers_from_missing_invalid_and_replaced_files(tmp_path):
    path = tmp_path/'result.json'
    assert web.read_status(path) == {}
    path.write_text('{')
    assert web.read_status(path) == {}
    web.save(path, dict(status='running'))
    assert web.read_status(path) == dict(status='running')
    web.save(path, dict(status='complete'))
    assert web.read_status(path) == dict(status='complete')
    path.unlink()
    assert web.read_status(path) == {}


def test_hash_streams_multiple_blocks_without_read_bytes(tmp_path, monkeypatch):
    payload = b'phone-texture' * 200000
    path = tmp_path/'archive.zip'
    path.write_bytes(payload)

    def reject_whole_file_read(*args, **kwargs):
        pytest.fail('Hashing must not load the whole file into memory')

    monkeypatch.setattr(Path, 'read_bytes', reject_whole_file_read)
    assert web.w.sha(path) == hashlib.sha256(payload).hexdigest()
