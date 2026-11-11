"""Portable bottom-stage contracts; synthetic files and mocked downstream work.

Root runs these serially against the R5 candidate. No research fixture paths,
model inference, real photographs, port detection, lighting or UV processing.
"""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from phone_texture import automatic, bottom_alignment, pipeline, stage_processing


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_geometry(tmp_path):
    root = tmp_path/'approved-geometry'
    root.mkdir()
    files, entries = {}, []
    for index, face in enumerate(pipeline.FACES):
        path = root/(face+'.png')
        Image.new('RGBA', (240, 24), (100+index, 120, 140, 255)).save(path)
        files[face] = path
        entries.append(dict(sku='sku_test', face=face, output_sha256=digest(path),
            geometry=dict(output_size=[240, 24]), source=str(path), source_sha256=digest(path)))
    geometry = dict(candidates_complete=True, geometry_status='needs_review', images=entries,
        resume_provenance=dict(reused_geometry_only=True, visual_approval_added=False))
    return files, geometry


def identity_map(image):
    return np.broadcast_to(np.arange(image.shape[1], dtype='float32'), image.shape[:2]).copy()


def bounded_stub_candidate(image, model):
    """A deterministic center-only fixture change, not a port algorithm test."""
    out = image.copy()
    out[image.shape[0]//2, image.shape[1]//2, 0] += 1
    active = np.zeros(image.shape[:2], dtype=bool)
    active[image.shape[0]//2, image.shape[1]//2] = True
    return out, dict(action='corrected_needs_visual_review', port_width_change_fraction=0.), identity_map(image), active


def assert_originals_unchanged(files, before, geometry, serialized):
    assert {face: digest(path) for face, path in files.items()} == before
    assert json.dumps(geometry, sort_keys=True) == serialized


def test_prepare_off_is_exact_and_never_calls_rectify(tmp_path, monkeypatch):
    files, geometry = source_geometry(tmp_path)
    before = {f: digest(p) for f, p in files.items()}
    serialized = json.dumps(geometry, sort_keys=True)
    def forbidden(*args, **kwargs):
        pytest.fail('off mode must not attempt bottom detection or correction')
    monkeypatch.setattr(bottom_alignment, 'rectify', forbidden)
    selected, effective, report = bottom_alignment.prepare(files, geometry, 'iphone17pro', tmp_path/'stage', enabled=False)
    assert report['status'] == 'disabled' and not report['applied']
    assert report['blocks_generation'] is False and report['notices'] == []
    assert {f: digest(p) for f, p in selected.items()} == before
    stage_processing.geometry_identity(effective, selected, 'sku_test')
    assert_originals_unchanged(files, before, geometry, serialized)


@pytest.mark.parametrize('error_type', [bottom_alignment.Rejected, ValueError, RuntimeError, FloatingPointError])
def test_prepare_expected_failure_keeps_exact_source_and_notice(tmp_path, monkeypatch, error_type):
    files, geometry = source_geometry(tmp_path)
    before = {f: digest(p) for f, p in files.items()}
    serialized = json.dumps(geometry, sort_keys=True)
    def rejected(*args, **kwargs):
        raise error_type('source evidence is uncertain')
    monkeypatch.setattr(bottom_alignment, 'rectify', rejected)
    selected, effective, report = bottom_alignment.prepare(files, geometry, 'iphone17pro', tmp_path/'stage')
    assert report['status'] == 'kept_source_with_notice' and report['applied'] is False
    assert report['blocks_generation'] is False and report['fallback_byte_identical'] is True
    assert report['notices'] and 'uncertain' in report['reason']
    assert {f: digest(p) for f, p in selected.items()} == before
    stage_processing.geometry_identity(effective, selected, 'sku_test')
    assert_originals_unchanged(files, before, geometry, serialized)


def test_prepare_invalid_candidate_is_rejected_before_selection(tmp_path, monkeypatch):
    files, geometry = source_geometry(tmp_path)
    def bad_candidate(image, model):
        out, proof, mapping, active = bounded_stub_candidate(image, model)
        out[0, 0, 0] ^= 255  # Endcap pixels must remain exact.
        return out, proof, mapping, active
    monkeypatch.setattr(bottom_alignment, 'rectify', bad_candidate)
    selected, _, report = bottom_alignment.prepare(files, geometry, 'iphone17pro', tmp_path/'stage')
    assert report['status'] == 'kept_source_with_notice'
    assert report['reason'] == 'endcap_pixels_changed'
    assert all(digest(selected[f]) == digest(files[f]) for f in pipeline.FACES)


def test_prepare_new_bottom_rebinds_effective_geometry_and_keeps_five_faces(tmp_path, monkeypatch):
    files, geometry = source_geometry(tmp_path)
    before = {f: digest(p) for f, p in files.items()}
    serialized = json.dumps(geometry, sort_keys=True)
    monkeypatch.setattr(bottom_alignment, 'rectify', bounded_stub_candidate)
    selected, effective, report = bottom_alignment.prepare(files, geometry, 'iphone17pro', tmp_path/'stage')
    assert report['status'] == 'applied_with_notice' and report['applied'] is True
    assert report['physical_accuracy_verified'] is False and report['model_uv_verified'] is False
    assert digest(selected['bottom']) != before['bottom']
    assert all(digest(selected[f]) == before[f] for f in pipeline.FACES if f != 'bottom')
    old_bottom = next(v for v in geometry['images'] if v['face'] == 'bottom')
    new_bottom = next(v for v in effective['images'] if v['face'] == 'bottom')
    assert old_bottom['output_sha256'] == before['bottom']
    assert new_bottom['pre_bottom_alignment_sha256'] == before['bottom']
    assert new_bottom['output_sha256'] == digest(selected['bottom'])
    stage_processing.geometry_identity(effective, selected, 'sku_test')
    with pytest.raises(ValueError, match='source identity changed'):
        stage_processing.geometry_identity(geometry, selected, 'sku_test')
    assert_originals_unchanged(files, before, geometry, serialized)


def test_prepare_provenance_failure_is_not_a_soft_notice(tmp_path):
    files, geometry = source_geometry(tmp_path)
    next(v for v in geometry['images'] if v['face'] == 'bottom')['output_sha256'] = '0'*64
    with pytest.raises(ValueError, match='Geometry provenance mismatch'):
        bottom_alignment.prepare(files, geometry, 'iphone17pro', tmp_path/'stage', enabled=False)


def test_prepare_io_failure_is_not_a_soft_notice(tmp_path, monkeypatch):
    files, geometry = source_geometry(tmp_path)
    def denied_copy(*args, **kwargs):
        raise OSError('test destination is unavailable')
    monkeypatch.setattr(bottom_alignment.shutil, 'copy2', denied_copy)
    with pytest.raises(OSError, match='destination is unavailable'):
        bottom_alignment.prepare(files, geometry, 'iphone17pro', tmp_path/'stage', enabled=False)


@pytest.mark.parametrize('mode, reject', [('appearance', False), ('appearance', True), ('off', False)])
def test_process_one_keeps_resume_slot_and_calls_r4_independent_stages(tmp_path, monkeypatch, mode, reject):
    files, geometry = source_geometry(tmp_path)
    before = {f: digest(p) for f, p in files.items()}
    serialized = json.dumps(geometry, sort_keys=True)
    case, resume = tmp_path/'new-case', tmp_path/'previous-case'
    manifest = dict(skus={'sku_test': {f: dict(source=str(p), snapshot=str(p), sha256=digest(p)) for f, p in files.items()}})
    seen = {}
    def intake(source, work, *, single_name):
        Path(work).mkdir(parents=True)
        assert single_name == 'sku_test'
        return copy.deepcopy(manifest)
    def resume_geometry(old_case, new_case, sku, current_manifest):
        assert Path(old_case) == resume  # task[9] remains resume_case.
        assert Path(new_case) == case and sku == 'sku_test'
        assert current_manifest == manifest
        seen['resume_called'] = True
        return copy.deepcopy(geometry), dict(files)
    def no_fresh_inference(*args, **kwargs):
        raise AssertionError('A valid resumed checkpoint must not enter fresh inference')
    # Even if routing regresses, fail before importing/loading a real model.
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(set_num_threads=no_fresh_inference))
    monkeypatch.setattr(automatic.w, 'intake', intake)
    monkeypatch.setattr(stage_processing, 'resume_geometry', resume_geometry)
    def rectify(image, model):
        assert mode == 'appearance'  # task[10] controls only this optional stage.
        seen['rectify_called'] = True
        if reject:
            raise bottom_alignment.Rejected('port evidence uncertain')
        return bounded_stub_candidate(image, model)
    monkeypatch.setattr(bottom_alignment, 'rectify', rectify)
    downstream = dict(status='source_preserved_needs_review', six_faces_generated=True,
        six_faces_processed=False, lighting_complete=False, structural_alignment_applied=False,
        stage_status=dict(geometry='complete', lighting='partial', structure='blocked', seam='blocked'),
        face_processing={'bottom': {'lighting': 'complete', 'selected_stage': 'lighting'},
                         'back': {'lighting': 'failed', 'selected_stage': 'geometry'}},
        blocked_checks=[dict(stage='lighting', face='back', error='simulated independent face failure')],
        review_warnings=[dict(face='back', code='simulated_review')])
    def independent_stages(new_case, sku, effective, selected, current_manifest, model,
                           protection_policy, lighting_profile, progress, stage, *, allow_partial, bottom_fallback=None,
                           source_body_reference_review=False, source_body_reference_metadata=None, bottom_quality=None,
                           side_candidate_review=False):
        assert source_body_reference_review is False and side_candidate_review is False
        assert source_body_reference_review is False and source_body_reference_metadata is None
        if mode == 'appearance' and not reject:
            assert set(bottom_quality) == {'source', 'commit'}
            assert all(callable(value) for value in bottom_quality.values())
            assert digest(bottom_quality['source']()) == before['bottom']
        else:
            assert bottom_quality is None
        assert seen.get('resume_called') and allow_partial is True
        assert sku == 'sku_test' and model == 'iphone17pro' and protection_policy == 'none'
        assert lighting_profile == 'accepted-shadow-lift-v2' and current_manifest == manifest
        stage_processing.geometry_identity(effective, selected, sku)
        assert all(Path(selected[f]) != files[f] for f in pipeline.FACES)
        assert all(digest(selected[f]) == before[f] for f in pipeline.FACES if f != 'bottom')
        assert (digest(selected['bottom']) != before['bottom']) == (mode == 'appearance' and not reject)
        # New job/scopes must receive these effective sources, never old paths.
        seen['downstream_sources'] = {f: dict(path=str(p), sha256=digest(p)) for f, p in selected.items()}
        return copy.deepcopy(downstream)
    monkeypatch.setattr(stage_processing, 'process_stages', independent_stages)
    task = ('sku_test', str(tmp_path/'raw'), str(case), 'cpu', 1, 'iphone17pro',
            'none', 'preserve-source', 'accepted-shadow-lift-v2', str(resume), mode)
    result = automatic.process_one(task)
    assert seen.get('downstream_sources') and seen.get('rectify_called', False) == (mode == 'appearance')
    assert result['status'] == downstream['status'] and result['stage_status'] == downstream['stage_status']
    assert result['six_faces_generated'] is True and result['six_faces_processed'] is False
    assert result['blocked_checks'] == downstream['blocked_checks']
    assert result['face_processing'] == downstream['face_processing']
    assert result['bottom_alignment']['status'] == ('disabled' if mode == 'off' else 'kept_source_with_notice' if reject else 'applied_with_notice')
    assert result['opencv_runtime']['requested_native_threads'] == 0
    assert result['opencv_runtime']['effective_threads'] == 1
    assert_originals_unchanged(files, before, geometry, serialized)
