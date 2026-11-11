"""Current-job provenance controls; root runs them under the global guard."""
import ast
import json
from pathlib import Path

import pytest

from phone_texture import side_candidate_structure as candidate


def result():
    return dict(candidate.flags(), hard_failures=[], artifacts={},
                eligible_for_visual_review=False, structural_alignment_applied=False,
                selected_for_review_export=False)


def test_bound_file_changes_are_integrity_errors_not_review_fallback(tmp_path):
    source = tmp_path/'raw.dat'
    source.write_bytes(b'current capture')
    old = candidate.sha(source)
    source.write_bytes(b'another capture')
    with pytest.raises(candidate.SideCandidateIntegrityError):
        candidate.bind({}, source, old)
    assert not issubclass(candidate.SideCandidateIntegrityError, ValueError)


def test_saved_report_and_in_memory_report_must_agree(tmp_path):
    path = tmp_path/'result.json'
    path.write_text(json.dumps(dict(source_sha256='a'*64, eligible_for_visual_review=False)))
    changed = dict(source_sha256='b'*64, eligible_for_visual_review=False, result_path=str(path))
    with pytest.raises(candidate.SideCandidateIntegrityError, match='differs'):
        candidate.read_record(changed, 'result_path', {})


@pytest.mark.parametrize('payload', ['invalid json', '[]', 'null'])
def test_malformed_saved_reports_are_not_photo_rejections(tmp_path, payload):
    path = tmp_path/'result.json'; path.write_text(payload)
    with pytest.raises(candidate.SideCandidateIntegrityError):
        candidate.read_record(path, 'result_path', {})


def test_source_review_rejection_has_intact_bindings_and_saved_reason(tmp_path):
    source = tmp_path/'input.dat'; source.write_bytes(b'current input')
    inputs = {str(source): candidate.sha(source)}
    outcome = result()
    error = candidate.ReviewRejected('Independent hardware overlaps a band')
    outcome['hard_failures'].append(dict(stage='current_hardware', error=str(error)))
    with pytest.raises(candidate.SideCandidateStructureFailure) as rejected:
        candidate._finish(tmp_path, 'result.json', outcome, inputs, {}, error)
    assert rejected.value.result['inputs_unchanged'] is True
    assert rejected.value.result['implementation_unchanged'] is True
    assert rejected.value.result['eligible_for_visual_review'] is False
    assert rejected.value.result['structural_alignment_applied'] is False
    assert rejected.value.result_path == str(tmp_path/'result.json')
    saved = json.loads((tmp_path/'result.json').read_text())
    assert saved['hard_failures'][0]['error'] == str(error)


def test_post_render_code_or_source_change_cannot_select_candidate(tmp_path):
    source = tmp_path/'input.dat'; source.write_bytes(b'before')
    inputs = {str(source): candidate.sha(source)}
    source.write_bytes(b'changed')
    outcome = result()
    outcome.update(eligible_for_visual_review=True, structural_alignment_applied=True,
                   selected_for_review_export=True)
    with pytest.raises(candidate.SideCandidateIntegrityError) as rejected:
        candidate._finish(tmp_path, 'result.json', outcome, inputs, {})
    assert rejected.value.result['inputs_unchanged'] is False
    assert rejected.value.result['implementation_unchanged'] is False
    assert rejected.value.result['eligible_for_visual_review'] is False
    assert rejected.value.result['selected_for_review_export'] is False


@pytest.mark.parametrize('error', [OSError('file unavailable'), KeyError('wrong schema'), RuntimeError('internal failure')])
def test_non_photo_failure_does_not_become_expected_review(tmp_path, error):
    outcome = result(); outcome['hard_failures'].append(dict(stage='preflight', error=str(error)))
    with pytest.raises(candidate.SideCandidateIntegrityError):
        candidate._finish(tmp_path, 'result.json', outcome, {}, {}, error)


def test_package_modules_have_no_historical_loader_or_sample_allowlist():
    folder = Path(candidate.__file__).parent
    for path in folder.glob('side_candidate_structure*.py'):
        source = path.read_text(); tree = ast.parse(source)
        assert 'importlib' not in source and 'spec_from_file_location' not in source
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert '/Users/' not in node.value
                assert 'continuation-r' not in node.value
                assert 'ip17pro/sku' not in node.value
        for node in tree.body:
            if isinstance(node, ast.Assign):
                assert not any(isinstance(name, ast.Name) and name.id in ('ROOT', 'KEYS', 'SELECTION', 'PINS', 'FIXED')
                               for name in node.targets)


def test_review_flags_do_not_claim_strict_material_visual_or_uv_approval():
    flags = candidate.flags()
    for key in ('strict_structure_verified', 'strict_endpoint_evidence_complete',
                'material_classification_complete', 'quality_checks_complete', 'selected_for_delivery',
                'visual_approved', 'model_uv_verified', 'seam_executed', 'reference_certified'):
        assert flags[key] is False
    assert flags['reference_review_required'] is True
