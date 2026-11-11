"""Actual R6 baseline metadata closure; no image/NPZ decoding or processing.

The audit double records asset SHA obligations and allows controlled byte failure.
Production Audit, not these metadata controls, certifies the actual asset bytes.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest
from phone_texture.saved_review_baseline import validate_baseline

ROOT = Path(__file__).resolve().parents[2]
# Regression against local saved-review records beside the repository;
# skipped where those private records are absent (e.g. a fresh clone).
if not (ROOT/'continuation-r6/e2e/r6-frozen/ip17pro').exists():
    pytest.skip('local saved-review records not present', allow_module_level=True)
FACES = ('front', 'back', 'left', 'right', 'top', 'bottom')


class MetadataAudit:
    def __init__(self):
        self.documents = {}
        self.file_calls = []
        self.reject_path = None

    def add(self, path):
        path = str(Path(path).resolve()); raw = Path(path).read_bytes()
        value = json.loads(raw); sha = hashlib.sha256(raw).hexdigest()
        self.documents[path] = (value, sha)
        return value, dict(path=path, sha256=sha)

    def json(self, path, expected):
        value, digest = self.documents[str(Path(path).resolve())]
        if digest != expected:
            raise ValueError('JSON SHA mismatch')
        return copy.deepcopy(value)

    def file(self, path, expected):
        path = str(Path(path).resolve())
        assert isinstance(expected, str) and len(expected) == 64
        self.file_calls.append((path, expected))
        if path == self.reject_path:
            raise ValueError('Controlled artifact SHA mismatch')
        return expected


def fixture(sku='sku10'):
    folder = ROOT/'continuation-r6/e2e/r6-frozen/ip17pro'/sku
    audit = MetadataAudit()
    baseline, result_descriptor = audit.add(folder/'actual-result.json')
    case = Path(baseline['process'])
    intake, intake_descriptor = audit.add(case/'work/intake.json')
    geometry, geometry_descriptor = audit.add(case/'work/geometry-records/report.json')
    entry = dict(key='ip17pro/'+sku, sku=sku, model='iphone17pro',
        source_sha256={f:intake['skus'][sku][f]['sha256'] for f in FACES},
        options=dict(model='iphone17pro', protection_policy='none', review_fallback='preserve-source',
            lighting_profile='accepted-shadow-lift-v2', bottom_alignment='appearance', source_body_reference_review=True),
        baseline_result=result_descriptor, baseline_intake=intake_descriptor, geometry_report=geometry_descriptor)
    return audit, entry, baseline, geometry, intake


@pytest.mark.parametrize('sku', ['sku10', 'sku14'])
def test_actual_baseline_closes_originals_geometry_bottom_lighting_without_quality_upgrade(sku):
    audit, entry, baseline, report, intake = fixture(sku)
    original = copy.deepcopy((entry, baseline, report, intake))
    geometry = validate_baseline(audit, entry, baseline, report)
    assert geometry == {row['face']:row['output_sha256'] for row in report['images']}
    assert (entry, baseline, report, intake) == original
    assert baseline['per_face_stages']['back']['lighting'] == 'effect_warning'
    assert baseline['quality_checks_complete'] is False
    assert baseline['six_faces_processed'] is False
    calls = dict(audit.file_calls)
    for face in FACES:
        raw = intake['skus'][sku][face]
        assert calls[raw['source']] == calls[raw['snapshot']] == entry['source_sha256'][face]
        selected = str(Path(baseline['bottom_alignment']['selected_dir'])/(face+'.png'))
        assert calls[selected] == (baseline['bottom_alignment']['selected_sha256'] if face == 'bottom' else geometry[face])
    bottom = baseline['bottom_alignment']
    assert (bottom['selected_sha256'] != geometry['bottom']) is (sku == 'sku10')
    assert baseline['lighting_checks']['bottom']['source_sha256'] == bottom['selected_sha256']


def test_mixed_new_capture_geometry_and_old_front_baseline_is_rejected():
    audit, entry, baseline, report, intake = fixture()
    entry['source_sha256']['front'] = '0'*64
    intake['skus'][entry['sku']]['front']['sha256'] = '0'*64
    front = next(row for row in report['images'] if row['face'] == 'front')
    front.update(source_sha256='0'*64, output_sha256='1'*64)
    with pytest.raises(ValueError, match='geometry-to-stage input mismatch: front'):
        validate_baseline(audit, entry, baseline, report)


@pytest.mark.parametrize('field,value', [('protection_policy','sticker_only'),
    ('lighting_profile','baseline'), ('source_body_reference_review_enabled',False)])
def test_other_four_faces_cannot_claim_a_different_processing_profile(field, value):
    audit, entry, baseline, report, _ = fixture()
    baseline[field] = value
    with pytest.raises(ValueError, match='processing configuration'):
        validate_baseline(audit, entry, baseline, report)


@pytest.mark.parametrize('field', ['baseline_intake', 'geometry_report'])
def test_other_case_report_path_is_rejected_even_with_same_digest(field):
    audit, entry, baseline, report, _ = fixture()
    entry[field]['path'] = str(ROOT/'some-other-case'/Path(entry[field]['path']).name)
    with pytest.raises(ValueError, match='outside the baseline case'):
        validate_baseline(audit, entry, baseline, report)


def test_bottom_original_geometry_hash_must_match_before_port_correction():
    audit, entry, baseline, report, _ = fixture()
    baseline['bottom_alignment']['source_sha256'] = '0'*64
    with pytest.raises(ValueError, match='bottom original source mismatch'):
        validate_baseline(audit, entry, baseline, report)


def test_corrected_bottom_must_be_the_actual_lighting_input():
    audit, entry, baseline, report, _ = fixture()
    baseline['lighting_checks']['bottom']['source_sha256'] = next(row['output_sha256'] for row in report['images'] if row['face'] == 'bottom')
    with pytest.raises(ValueError, match='lighting source/output'):
        validate_baseline(audit, entry, baseline, report)


def test_kept_bottom_cannot_change_pixels_under_a_notice():
    audit, entry, baseline, report, _ = fixture('sku14')
    baseline['bottom_alignment']['selected_sha256'] = '0'*64
    with pytest.raises(ValueError, match='kept bottom differs'):
        validate_baseline(audit, entry, baseline, report)


def test_port_companion_hash_failure_propagates():
    audit, entry, baseline, report, _ = fixture()
    audit.reject_path = str(Path(baseline['bottom_alignment']['selected_dir'])/'front.png')
    with pytest.raises(ValueError, match='Controlled artifact SHA mismatch'):
        validate_baseline(audit, entry, baseline, report)


@pytest.mark.parametrize('field', ['alpha_changed', 'paper_changed', 'outside_material_changed'])
def test_original_lighting_hard_pixel_check_must_be_zero(field):
    audit, entry, baseline, report, _ = fixture()
    baseline['lighting_checks']['front'][field] = 1
    with pytest.raises(ValueError, match=field):
        validate_baseline(audit, entry, baseline, report)


def test_final_file_must_match_the_selected_stage():
    audit, entry, baseline, report, _ = fixture()
    baseline['final_sha256']['top'] = '0'*64
    with pytest.raises(ValueError, match='selected/final mismatch: top'):
        validate_baseline(audit, entry, baseline, report)


def test_original_intake_source_cannot_be_relabelled_as_another_face():
    audit, entry, baseline, report, intake = fixture()
    intake['skus'][entry['sku']]['front']['source'] = intake['skus'][entry['sku']]['back']['source']
    with pytest.raises(ValueError, match='raw or snapshot path'):
        validate_baseline(audit, entry, baseline, report)


def test_no_quality_or_structure_complete_schema_is_silently_accepted():
    audit, entry, baseline, report, _ = fixture()
    baseline['stage_status']['structure'] = 'complete'
    with pytest.raises(ValueError, match='unsupported baseline stage schema'):
        validate_baseline(audit, entry, baseline, report)
