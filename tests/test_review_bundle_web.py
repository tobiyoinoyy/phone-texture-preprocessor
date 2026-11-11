"""Portable download admission controls; byte fixtures are not producer proofs."""
from pathlib import Path
import pytest
from phone_texture import web_app as web
from test_web_app import package_fixture


def fixture(tmp_path):
    app, job, report = package_fixture(tmp_path)
    row = report['attempts'][0]
    path = Path(job['output'])/row['sku']/'候选证据包.zip'
    path.write_bytes(b'controlled portable archive fixture')
    row['portable_review_package'] = dict(zip_sha256=web.w.sha(path))
    for item in report['attempts']:
        target = Path(job['work'])/'cases'/item['sku']
        target.mkdir(parents=True)
        web.save(target/'result.json', item)
    web.save(Path(job['work'])/'partial-candidate-manifest.json',
        dict(generated=2, total=2, visual_approved=0, items=report['attempts']))
    return app, job, report, row, path


def test_bundle_download_bound_to_actual_export_and_case(tmp_path):
    app, job, report, row, path = fixture(tmp_path)
    app.package(job, report)
    assert app.artifact(dict(job=job['id'], kind='bundle', sku=row['sku'])) == path
    states = {item['name']:item for item in app.snapshot(job)['rows']}
    assert states[row['sku']]['portable_review_available'] is True
    assert sum(item['portable_review_available'] for item in states.values()) == 1


@pytest.mark.parametrize('kind', ['missing', 'changed', 'failed', 'foreign'])
def test_bundle_cannot_bypass_failed_or_changed_export(tmp_path, kind):
    app, job, report, row, path = fixture(tmp_path)
    app.package(job, report)
    name = row['sku']
    if kind == 'missing': path.unlink()
    elif kind == 'changed': path.write_bytes(b'changed')
    elif kind == 'failed':
        row.update(status='failed', six_faces_generated=False)
        web.save(Path(job['work'])/'cases'/name/'result.json', row)
    else: name = '../unselected'
    with pytest.raises(ValueError):
        app.artifact(dict(job=job['id'], kind='bundle', sku=name))
    if kind != 'foreign':
        states = {item['name']:item for item in app.snapshot(job)['rows']}
        assert states[row['sku']]['portable_review_available'] is False


def test_missing_portable_package_fails_packaging(tmp_path):
    app, job, report, row, path = fixture(tmp_path)
    path.unlink()
    with pytest.raises(ValueError, match='复核记录包'):
        app.package(job, report)
    assert 'bundle:'+row['sku'] not in job.get('artifact_sha256', {})
