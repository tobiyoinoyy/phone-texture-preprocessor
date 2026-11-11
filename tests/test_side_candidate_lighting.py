"""Small scope and failure-routing controls; root executes under its guard."""
import json
from pathlib import Path

import numpy as np
import pytest

from phone_texture import automatic, side_candidate_lighting as candidate
from phone_texture import side_candidate_material as material
from phone_texture.side_candidate_scope import build_scope


def observations(width=20, height=200):
    bands = {}
    for index, lo in enumerate((50., 140.)):
        columns = [dict(column=x, paired_source_supported=x not in (0, width-1),
            prediction_only=False, measured_pair=[lo, lo+10]) for x in range(width)]
        bands[str(index)] = dict(original_RGB_geometry_seeds=dict(observer_reached=True,
            width=width, height=height, columns=columns, selected_band_seeds=[lo, lo+10],
            endpoint_shape_diagnostic=[dict(would_reject=True, columns=[2])], actual_gate_passed=False))
    return bands


def test_unknown_columns_stay_local_and_are_never_labeled_antenna():
    source = np.full((200, 20, 4), 255, dtype=np.uint8)
    scope = build_scope(source, observations(), face='left', source_file_sha256='a'*64)
    assert scope['hold'][50, 0] and scope['hold'][50, 2] and scope['hold'][140, 19]
    assert not scope['hold'][0].any() and not scope['hold'][-1].any()
    assert not scope['antenna_masks']['antenna_0'][:, [0, 2, 19]].any()
    assert scope['antenna_masks']['antenna_0'][50:60, 3:19].all()
    assert not scope['metadata']['classification_complete']
    assert not scope['metadata']['strict_endpoint_evidence_complete']


def test_predictions_cannot_become_measured_material_samples():
    source = np.full((200, 20, 4), 255, dtype=np.uint8)
    bands = observations()
    bands['0']['original_RGB_geometry_seeds']['columns'][5]['prediction_only'] = True
    with pytest.raises(ValueError, match='prediction cannot label'):
        build_scope(source, bands, face='right', source_file_sha256='a'*64)


def test_scope_requires_unresolved_pixels_instead_of_replacing_normal_success():
    source = np.full((200, 20, 4), 255, dtype=np.uint8)
    bands = observations()
    for band in bands.values():
        witness = band['original_RGB_geometry_seeds']
        witness['endpoint_shape_diagnostic'] = []
        for column in witness['columns']:
            column['paired_source_supported'] = True
    with pytest.raises(ValueError, match='explicitly unresolved'):
        build_scope(source, bands, face='left', source_file_sha256='a'*64)


def bound_input(tmp_path, monkeypatch):
    # Not an image fixture: all controls below stop at the mocked proposal,
    # before a decoder, renderer, or field operation can run.
    source = tmp_path/'geometry.png'; source.write_bytes(b'bound current geometry')
    raw = tmp_path/'raw.png'; raw.write_bytes(b'bound raw geometry witness')
    original = tmp_path/'source.jpg'; original.write_bytes(b'bound raw photograph')
    module = Path(candidate.__file__).resolve(); repo = module.parents[1]
    monkeypatch.setattr(automatic, 'implementation_manifest',
        lambda root: {str(module.relative_to(repo)): candidate.sha(module)})
    context = dict(face='left', path=str(raw), sha256=candidate.sha(raw),
        geometry_sha256=candidate.sha(source), raw_source=str(original), raw_source_sha256=candidate.sha(original))
    report = dict(schema=candidate.OBSERVATION_SCHEMA, face='left', source_sha256=candidate.sha(source),
        processing_state='completed_diagnostic_only', observations_computed_from_current_source=True,
        historical_observations_reused=False, hard_failures=[],
        source_context=context, input_sha256={str(path):candidate.sha(path) for path in (source,raw,original)},
        code_sha256={str(module):candidate.sha(module)}, inputs_unchanged=True, code_unchanged=True)
    path = tmp_path/'observation.json'; path.write_text(json.dumps(report))
    return source, path, report


def test_source_verified_local_rejection_preserves_typed_failure_evidence(tmp_path, monkeypatch):
    source, report, _ = bound_input(tmp_path, monkeypatch)
    def reject(*args, **kwargs):
        raise ValueError('Measured material support is insufficient')
    monkeypatch.setattr(material, 'propose_job', reject)
    with pytest.raises(candidate.SideCandidateLightingFailure) as caught:
        candidate.run(source, 'left', tmp_path/'candidate', endpoint_report=report)
    result = caught.value.result
    assert result['input_dependencies_unchanged'] is True
    assert result['integrity_failure'] is False
    assert result['render_attempts'] == 0 and result['candidate'] is None
    assert result['eligible_for_visual_review'] is False
    assert all(result[key] is False for key in candidate.FALSE_FLAGS)
    assert Path(caught.value.result_path).is_file()


@pytest.mark.parametrize('kind', ['io', 'source_mutation', 'code_fault'])
def test_corruption_and_io_are_not_local_review_fallbacks(tmp_path, monkeypatch, kind):
    source, report, _ = bound_input(tmp_path, monkeypatch)
    def reject(*args, **kwargs):
        if kind == 'source_mutation':
            source.write_bytes(b'changed during processing')
            raise ValueError('Local detector also failed')
        if kind == 'io':
            raise OSError('Cannot read the candidate input')
        raise RuntimeError('Unexpected implementation failure')
    monkeypatch.setattr(material, 'propose_job', reject)
    with pytest.raises(candidate.SideCandidateIntegrityError) as caught:
        candidate.run(source, 'left', tmp_path/'candidate', endpoint_report=report)
    result = caught.value.result
    assert result['integrity_failure'] is True
    assert result['eligible_for_visual_review'] is False
    if kind == 'source_mutation':
        assert result['input_dependencies_unchanged'] is False


def test_old_saved_observation_schema_cannot_enter_fresh_generation(tmp_path, monkeypatch):
    source, report, value = bound_input(tmp_path, monkeypatch)
    value['schema'] = 'historical_diagnostic'
    report.write_text(json.dumps(value))
    calls = []
    monkeypatch.setattr(material, 'propose_job', lambda *a, **k: calls.append(a))
    with pytest.raises(candidate.SideCandidateIntegrityError) as caught:
        candidate.run(source, 'left', tmp_path/'candidate', endpoint_report=report)
    assert calls == []
    assert caught.value.result['render_attempts'] == 0


def test_material_adapter_rejects_ordinary_route_before_decoding(tmp_path):
    with pytest.raises(ValueError, match='current observations'):
        material.propose_job(tmp_path/'missing.png', 'front', tmp_path/'out',
                             protection_policy='none', partial_side_evidence=tmp_path/'observation.json')
    assert not (tmp_path/'out').exists()
