"""Routing/fallback controls; stage computations use explicit metadata doubles."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import phone_texture
from phone_texture import side_candidate_pipeline as route


def saved(folder, name, value):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/name
    value = dict(value, result_path=str(path))
    path.write_text(json.dumps(value))
    return value


def rejection(folder, face, phase):
    data = saved(folder, 'result.json', dict(face=face, inputs_unchanged=True,
        implementation_unchanged=True, eligible_for_visual_review=False,
        hard_failures=[dict(stage=phase, error='insufficient current evidence')]))
    error = ValueError('current evidence rejected')
    error.result = data
    error.result_path = data['result_path']
    return error


def environment(tmp_path, monkeypatch, reject=None, corrupt=False, integrity=False):
    case = tmp_path/'current-case'; case.mkdir()
    files = {}
    for face in ('front', 'back', 'left', 'right', 'top', 'bottom'):
        path = case/(face+'.png'); path.write_bytes(('geometry '+face).encode()); files[face] = path
    selected = dict(files)
    hashes = {face:route.w.sha(path) for face,path in files.items()}
    original_hashes = dict(hashes)
    rows = {face:dict(lighting='failed', selected_stage='geometry', source_sha256=hashes[face]) for face in files}
    calls = []

    def observe(source, face, folder, **kwargs):
        calls.append(('observation', face))
        if reject == ('observation', face):
            raise rejection(folder, face, 'observation')
        value = saved(folder, 'observation.json', dict(face=face, current=True))
        # The observer may return a bound convenience SHA not embedded in JSON.
        value['observation_path'] = value['result_path']
        value['observation_sha256'] = route.w.sha(Path(value['result_path']))
        return value

    def lighting(source, face, folder, **kwargs):
        calls.append(('lighting', face))
        if integrity and face == 'left':
            raise RuntimeError('bound input changed')
        if reject == ('lighting', face):
            raise rejection(folder, face, 'lighting')
        folder.mkdir(parents=True)
        candidate = folder/(face+'.png'); candidate.write_bytes(('light '+face).encode())
        return saved(folder, 'result.json', dict(face=face, source=str(source),
            source_sha256=route.w.sha(source), candidate=str(candidate),
            candidate_sha256='0'*64 if corrupt and face == 'left' else route.w.sha(candidate),
            eligible_for_visual_review=True, hard_failures=[]))

    def structure(light, observed, folder):
        face = observed['face']; calls.append(('structure', face))
        if reject == ('structure', face):
            raise rejection(folder, face, 'structure')
        folder.mkdir(parents=True)
        candidate = folder/(face+'.png'); candidate.write_bytes(('aligned '+face).encode())
        return saved(folder, 'result.json', dict(face=face, source=light['candidate'],
            source_sha256=light['candidate_sha256'], candidate=str(candidate),
            candidate_sha256=route.w.sha(candidate), eligible_for_visual_review=True, hard_failures=[],
            reference_review=dict(original_status='source_tail_applicability_unknown', original_applicable=None)))

    monkeypatch.setattr(phone_texture, 'side_candidate_structure',
        SimpleNamespace(observe_current=observe, generate=structure), raising=False)
    monkeypatch.setattr(phone_texture, 'side_candidate_lighting', SimpleNamespace(run=lighting), raising=False)
    data = dict(case=case, sku='unseen-sku', geometry=dict(images=[]), geometry_files=files,
        raw_evidence=dict(left={}, right={}), reference=dict(enabled=True), selected=selected,
        selected_hashes=hashes, per_face=rows, checks={}, progress=lambda *args:None,
        verify_sources=lambda:None)
    return data, calls, original_hashes


def test_current_sides_both_execute_and_remain_review(tmp_path, monkeypatch):
    data, calls, old = environment(tmp_path, monkeypatch)
    result, lighting, structure, failures = route.continue_sides(**data)
    assert result['complete'] is True and failures == []
    assert set(lighting) == set(structure) == {'left', 'right'}
    assert calls == [(phase, face) for face in ('left', 'right') for phase in ('observation', 'lighting', 'structure')]
    for face in ('left', 'right'):
        assert data['selected_hashes'][face] != old[face]
        assert data['per_face'][face]['selected_stage'] == 'structure_review'
        assert data['per_face'][face]['strict_structure_verified'] is False
        assert data['per_face'][face]['source_reference_status'] == 'source_tail_applicability_unknown'


@pytest.mark.parametrize('phase', ['observation', 'lighting'])
def test_local_rejection_retains_previous_side_and_continues_other(tmp_path, monkeypatch, phase):
    data, calls, old = environment(tmp_path, monkeypatch, reject=(phase, 'left'))
    result, lighting, structure, failures = route.continue_sides(**data)
    assert result['complete'] is False and len(failures) == 1
    assert data['selected_hashes']['left'] == old['left']
    assert data['per_face']['left']['selected_stage'] == 'geometry'
    assert set(structure) == {'right'}
    assert ('structure', 'left') not in calls


def test_structure_rejection_retains_new_verified_lighting(tmp_path, monkeypatch):
    data, calls, old = environment(tmp_path, monkeypatch, reject=('structure', 'left'))
    result, lighting, structure, failures = route.continue_sides(**data)
    assert result['complete'] is False and len(failures) == 1
    assert set(lighting) == {'left', 'right'} and set(structure) == {'right'}
    assert data['per_face']['left']['selected_stage'] == 'lighting_review'
    assert data['selected_hashes']['left'] == lighting['left']['candidate_sha256'] != old['left']


@pytest.mark.parametrize('kind', ['corrupt', 'integrity'])
def test_integrity_failure_is_not_a_local_fallback(tmp_path, monkeypatch, kind):
    data, calls, _ = environment(tmp_path, monkeypatch, **{kind:True})
    with pytest.raises(RuntimeError):
        route.continue_sides(**data)
    assert ('observation', 'right') not in calls


def test_missing_raw_projection_is_explicit_and_other_side_continues(tmp_path, monkeypatch):
    data, calls, old = environment(tmp_path, monkeypatch)
    data['raw_evidence'].pop('left')
    result, _, structure, failures = route.continue_sides(**data)
    assert result['faces']['left']['state'] == 'unavailable'
    assert result['complete'] is False and failures[0]['kind'] == 'unsupported_source_evidence'
    assert data['selected_hashes']['left'] == old['left'] and set(structure) == {'right'}


def test_report_rejects_mismatched_convenience_digest(tmp_path):
    record = saved(tmp_path, 'result.json', dict(current=True))
    record['result_sha256'] = '0'*64
    with pytest.raises(RuntimeError, match='differs'):
        route.report(record, tmp_path, 'result_path')
