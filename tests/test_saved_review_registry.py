"""Registry dispatch controls; pixel/evidence validation is tested separately."""
import copy
import json
from pathlib import Path

import pytest

from phone_texture import saved_review as saved

ROOT=Path(__file__).resolve().parents[2]
# Regression against local saved-review records beside the repository;
# skipped where those private records are absent (e.g. a fresh clone).
if not (ROOT/'continuation-r8/saved-review-manifest.json').exists():
    pytest.skip('local saved-review records not present', allow_module_level=True)


def registry(tmp_path):
    data=json.loads((ROOT/'continuation-r8/saved-review-manifest.json').read_text())
    path=tmp_path/'manifest.json';path.write_text(json.dumps(data))
    return path,data,data['entries'][0]


def source_mock(monkeypatch,hashes):
    # No image reads in these routing tests; the real worker/E2E validates bytes.
    monkeypatch.setattr(saved,'_current_sources',lambda source:({},copy.deepcopy(hashes)))
    monkeypatch.setattr(saved,'_assemble',lambda *args: {})


def test_valid_matched_registry_is_pinned_without_writing_case(tmp_path,monkeypatch):
    path,data,entry=registry(tmp_path);source=tmp_path/'raw'
    source_mock(monkeypatch,entry['source_sha256'])
    values=saved.load_selected(path,{entry['sku']:source},entry['options'])
    descriptor=values[entry['sku']]
    assert descriptor['manifest_sha256']==saved.sha(path)
    assert descriptor['entry_sha256']==saved.object_sha(entry)
    assert not source.exists() and not (tmp_path/('.unwritten-review-'+entry['sku'])).exists()


def test_no_matching_sku_keeps_ordinary_route_without_reading_unrelated_sources(tmp_path,monkeypatch):
    path,data,entry=registry(tmp_path)
    monkeypatch.setattr(saved,'_current_sources',lambda *a:pytest.fail('Unrelated SKU read'))
    assert saved.load_selected(path,{'sku48':tmp_path/'raw'},entry['options'])=={}


def test_matched_sku_with_changed_source_is_not_treated_as_a_miss(tmp_path,monkeypatch):
    path,data,entry=registry(tmp_path);hashes=dict(entry['source_sha256'],front='0'*64)
    source_mock(monkeypatch,hashes)
    with pytest.raises(saved.SavedReviewError,match='six sources differ'):
        saved.load_selected(path,{entry['sku']:tmp_path/'raw'},entry['options'])


def test_same_capture_wrong_model_rejects_but_other_capture_is_unmatched(tmp_path,monkeypatch):
    path,data,entry=registry(tmp_path);options=dict(entry['options'],model='iphone17promax')
    source_mock(monkeypatch,entry['source_sha256'])
    with pytest.raises(saved.SavedReviewError,match='another requested model'):
        saved.load_selected(path,{entry['sku']:tmp_path/'raw'},options)
    source_mock(monkeypatch,dict(entry['source_sha256'],front='0'*64))
    assert saved.load_selected(path,{entry['sku']:tmp_path/'raw'},options)=={}


@pytest.mark.parametrize('change',[{'source_body_reference_review':False},{'bottom_alignment':'off'},
    {'lighting_profile':'baseline'},{'protection_policy':'sticker_only'}])
def test_matched_saved_options_cannot_silently_override_request(tmp_path,monkeypatch,change):
    path,data,entry=registry(tmp_path);source_mock(monkeypatch,entry['source_sha256'])
    with pytest.raises(saved.SavedReviewError,match='options differ'):
        saved.load_selected(path,{entry['sku']:tmp_path/'raw'},dict(entry['options'],**change))


@pytest.mark.parametrize('mutation',['schema','duplicate','unknown_flags'])
def test_registry_schema_and_identity_are_explicit(tmp_path,mutation):
    path,data,entry=registry(tmp_path)
    if mutation=='schema':data['schema']='unknown'
    elif mutation=='duplicate':data['entries'].append(copy.deepcopy(entry))
    else:data['entries'][0]['options']['source_body_reference_review']=1
    path.write_text(json.dumps(data))
    with pytest.raises(saved.SavedReviewError):saved._manifest(path)


def test_manifest_changed_after_preflight_rejects_before_worker_case(tmp_path,monkeypatch):
    path,data,entry=registry(tmp_path);source_mock(monkeypatch,entry['source_sha256'])
    source=tmp_path/'raw';case=tmp_path/'case'
    descriptor=saved.load_selected(path,{entry['sku']:source},entry['options'])[entry['sku']]
    data['note']='changed after queueing';path.write_text(json.dumps(data))
    with pytest.raises(saved.SavedReviewError,match='Stale source/evidence SHA'):
        saved.process_saved(descriptor,sku=entry['sku'],source=source,case=case,options=entry['options'])
    assert not case.exists()


def test_dry_copy_validates_source_without_creating_destination(tmp_path):
    source=tmp_path/'evidence.bin';source.write_bytes(b'bound evidence')
    target=tmp_path/'not-created/sub/candidate.bin';audit=saved.Audit()
    saved.copy_bound(audit,source,target,saved.sha(source));audit.finish()
    assert not target.parent.exists()
    with pytest.raises(saved.SavedReviewError,match='Stale source/evidence SHA'):
        saved.copy_bound(audit,source,target,'0'*64)
