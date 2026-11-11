"""Actual saved JSON semantics, with asset I/O delegated to a recording audit.

No image/NPZ read, render, inference, server or subprocess. These controls do not
certify asset bytes: the production loader's Audit must hash them. The mutation
double intentionally preserves original document digests to isolate semantic
rejections after a JSON descriptor has been admitted by the caller.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest
from phone_texture import saved_review_bindings as bindings


ROOT = Path(__file__).resolve().parents[2]
# Regression against local saved-review records beside the repository;
# skipped where those private records are absent (e.g. a fresh clone).
if not (ROOT/'continuation-r7/review-assembly-proposal/plans').exists():
    pytest.skip('local saved-review records not present', allow_module_level=True)


def read_record(path):
    data = Path(path).read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


class MetadataAudit:
    def __init__(self):
        self.documents = {}
        self.file_calls = []
        self.reject_path = None

    def file(self, path, expected=None):
        normalized = str(Path(path).resolve())
        self.file_calls.append((normalized, expected))
        if self.reject_path == normalized:
            raise ValueError('Controlled asset SHA mismatch')
        if expected is not None and not bindings._sha(expected):
            raise ValueError('Invalid expected SHA')
        return expected

    def json(self, path, expected=None):
        path = str(Path(path).resolve())
        if Path(path).suffix != '.json':
            raise AssertionError('Only JSON documents may be read in these tests')
        if path not in self.documents:
            self.documents[path] = read_record(path)
        value, digest = self.documents[path]
        if expected is not None and expected != digest:
            raise ValueError('JSON evidence SHA mismatch')
        return copy.deepcopy(value)


def fixture(sku='sku14', face='left'):
    plan, _ = read_record(ROOT/'continuation-r7/review-assembly-proposal/plans'/('ip17pro-'+sku+'.json'))
    entry = copy.deepcopy(plan['sides'][face]); audit = MetadataAudit()
    for field in ('lighting_result', 'structure_result', 'preparation'):
        path = entry[field]['path']; record, digest = read_record(path)
        assert digest == entry[field]['sha256']
        audit.documents[str(Path(path).resolve())] = (record, digest)
    baseline, digest = read_record(plan['baseline_result']['path'])
    assert digest == plan['baseline_result']['sha256']
    light = audit.documents[entry['lighting_result']['path']][0]
    baseline['_selected_source_hashes'] = copy.deepcopy(light['source_identity_before']['hashes'])
    baseline['_geometry_sha256'] = {f: baseline['per_face_stages'][f]['source_sha256'] for f in bindings.FACES}
    structure = audit.documents[entry['structure_result']['path']][0]
    preparation = audit.documents[entry['preparation']['path']][0]
    return plan['key'], face, entry, baseline, audit, light, structure, preparation


@pytest.mark.parametrize('sku,face', [('sku14','left'), ('sku14','right'), ('sku10','left'), ('sku10','right')])
def test_four_actual_saved_sides_remain_review_not_strict_pass(sku, face):
    key, face, entry, baseline, audit, light, structure, prep = fixture(sku, face)
    original = copy.deepcopy((entry, baseline, light, structure, prep))
    result = bindings.validate_side(key, face, entry, baseline, audit)
    assert result['structure_applied'] is True
    assert result['selected_sha256'] == entry['expected_selected_sha256']
    assert result['lighting_sha256'] == entry['expected_lighting_sha256']
    assert result['review_evidence']['visual_approved'] is False
    assert result['review_evidence']['quality_checks_complete'] is False
    assert result['review_evidence']['selected_for_delivery'] is False
    assert result['reference_review_required'] is True
    assert (entry, baseline, light, structure, prep) == original
    if sku == 'sku10':
        reference = result['source_reference_status']
        assert reference['original_status'] == 'source_tail_applicability_unknown'
        assert reference['original_applicable'] is None
        assert reference['original_unknown_preserved'] is True
        assert reference['reference_certified'] is False
    if structure['review_policy'] == bindings.V3:
        proof = result['review_evidence']['structure_checks']['contour_sampling']
        assert proof['whole_core_rgb_rigid'] is False
        assert proof['original_opacity_contract_passed'] is False
    else:
        proof = result['review_evidence']['structure_checks']['mapping_and_alpha_review']
        assert proof['source_rgb_rigid_inside_integer_core'] is True
        assert proof['original_opacity_contract_passed'] is False


@pytest.mark.parametrize('field,value', [('alpha_zero_actual_donors_used',1),
    ('internal_opacity_conflicts_permitted',1), ('fixed_rgba_changed',1),
    ('transparent_rgba_changed',1), ('alpha_changed',1), ('whole_core_rgb_rigid',True)])
def test_v3_integrity_or_rigid_claim_cannot_be_softened(field, value):
    key, face, entry, baseline, audit, light, structure, prep = fixture()
    structure['checks']['contour_sampling'][field] = value
    with pytest.raises(ValueError, match=field):
        bindings.validate_side(key, face, entry, baseline, audit)


@pytest.mark.parametrize('field,value', [('alpha_zero_donors_used',1), ('internal_conflicts_permitted',1)])
def test_v2_zero_or_internal_donor_cannot_be_softened(field, value):
    key, face, entry, baseline, audit, light, structure, prep = fixture('sku14', 'right')
    structure['checks']['mapping_and_alpha_review'][field] = value
    with pytest.raises(ValueError, match=field):
        bindings.validate_side(key, face, entry, baseline, audit)


def test_actual_selected_source_hash_mismatch_rejects_even_with_same_key():
    key, face, entry, baseline, audit, *_ = fixture()
    baseline['_selected_source_hashes']['left.jpg'] = '0'*64
    with pytest.raises(ValueError, match='raw sources differ'):
        bindings.validate_side(key, face, entry, baseline, audit)


@pytest.mark.parametrize('change', ['wrong_key', 'wrong_face', 'missing_strict_false', 'hard_failure', 'not_eligible', 'wrong_source'])
def test_failed_or_misbound_structure_never_falls_back_to_lighting(change):
    key, face, entry, baseline, audit, light, structure, prep = fixture()
    if change == 'wrong_key': structure['key'] = 'ip17pro/other'
    elif change == 'wrong_face': structure['face'] = 'right'
    elif change == 'missing_strict_false': del structure['strict_structure_verified']
    elif change == 'hard_failure': structure['hard_failures'] = ['actual failure']
    elif change == 'not_eligible': structure['eligible_for_visual_review'] = False
    else: prep['source_sha256'] = '0'*64
    with pytest.raises(ValueError):
        bindings.validate_side(key, face, entry, baseline, audit)


def test_unknown_reference_cannot_be_relabelled_as_no_tail():
    key, face, entry, baseline, audit, light, structure, prep = fixture('sku10', 'right')
    for record in (structure, prep):
        record['reference_review']['original_status'] = 'no_tail'
        record['reference_review']['original_unknown_preserved'] = False
    with pytest.raises(ValueError, match='explicit unknown'):
        bindings.validate_side(key, face, entry, baseline, audit)


def test_measured_center_failure_cannot_be_accepted_by_saved_status():
    key, face, entry, baseline, audit, light, structure, prep = fixture()
    structure['checks']['final_central_measurements'][0]['center_error_px'] = 99
    with pytest.raises(ValueError, match='center/width'):
        bindings.validate_side(key, face, entry, baseline, audit)


def test_historical_rejection_must_remain_bound_and_unchanged():
    key, face, entry, baseline, audit, light, structure, prep = fixture()
    structure['historical_same_x_rejection'] = []
    with pytest.raises(ValueError, match='historical same-x rejection'):
        bindings.validate_side(key, face, entry, baseline, audit)


def test_lighting_effect_failure_is_not_a_review_warning():
    key, face, entry, baseline, audit, light, structure, prep = fixture()
    effect = next(iter(light['checks']['original_and_tangent_effects'].values()))
    effect['effect_check_passed'] = False
    with pytest.raises(ValueError, match='effect_check_passed'):
        bindings.validate_side(key, face, entry, baseline, audit)


def test_caller_asset_hash_error_propagates_without_selection():
    key, face, entry, baseline, audit, light, structure, prep = fixture()
    audit.reject_path = str(Path(structure['artifacts']['candidate']).resolve())
    with pytest.raises(ValueError, match='Controlled asset SHA mismatch'):
        bindings.validate_side(key, face, entry, baseline, audit)


def test_missing_structure_requires_explicit_lighting_only():
    key, face, entry, baseline, audit, *_ = fixture()
    del entry['structure_result']
    with pytest.raises(ValueError, match='explicit lighting_only'):
        bindings.validate_side(key, face, entry, baseline, audit)
    entry['structure_policy'] = 'lighting_only'
    result = bindings.validate_side(key, face, entry, baseline, audit)
    assert result['structure_applied'] is False
    assert result['selected_sha256'] == result['lighting_sha256']
    assert result['review_evidence']['structure_state'] == 'not_selected_explicit_lighting_only'
