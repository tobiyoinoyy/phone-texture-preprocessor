"""Portable bundle controls using synthetic byte/metadata fixtures only.

No product image, model, historical case or image library is used. Real-candidate
portability must additionally be checked by the root's serialized integration run.
"""
import base64
import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest
from phone_texture import review_bundle as bundle


PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg==')
OPTIONS = dict(model='iphone17pro', protection_policy='none', review_fallback='preserve-source',
    lighting_profile='accepted-shadow-lift-v2', bottom_alignment='appearance',
    source_body_reference_review=True, side_candidate_review=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return str(path), sha(data)


def record(path, value):
    return write(path, json.dumps(value, sort_keys=True).encode())


def fixture(tmp_path, structure=False):
    sku = 'new-device'
    case, source = tmp_path/'current-case', tmp_path/'current-originals'
    source.mkdir(); case.mkdir()
    raw = {}; assets = {}; intake = {}; details = {}; faces = {}; finals = {}
    for face in bundle.FACES:
        data = ('current raw ' + face).encode()
        path = source/(face+'.jpg'); path.write_bytes(data); raw[face] = path
        snapshot, raw_sha = write(case/'work/input'/sku/(face+'.jpg'), data)
        assets[snapshot] = raw_sha
        intake[face] = dict(source=str(path), snapshot=snapshot, sha256=raw_sha)
        geometry, geometry_sha = write(case/'work/geometry'/sku/(face+'.png'), ('geometry '+face).encode())
        assets[geometry] = geometry_sha
        candidate, final_sha = write(case/'candidate'/sku/(face+'.png'), PNG)
        finals[face] = final_sha
        details[face] = dict(selected_stage='lighting', source_sha256=geometry_sha,
            selected_sha256=final_sha, selected_stage_file=candidate, lighting='complete', structure='blocked')
        faces[face] = dict(source_sha256=geometry_sha, output_sha256=final_sha,
            checks=dict(source_identity_verified=True, pixel_integrity_verified=True,
                output_selection_verified=True, hard_failures=[], measured=dict(alpha_changed=0, outside_changed=0)))
    options = copy.deepcopy(OPTIONS)
    if structure:
        options['side_candidate_review'] = True
        face = 'left'; geometry = str(case/'work/geometry'/sku/(face+'.png'))
        lighting_path, lighting_sha = write(case/'work/fresh-light/left.png', PNG)
        assets[lighting_path] = lighting_sha
        final_path = str(case/'candidate'/sku/'left.png'); assets[final_path] = finals['left']
        for name, source_path, source_sha, output, output_sha in (
                ('lighting', geometry, faces[face]['source_sha256'], lighting_path, lighting_sha),
                ('structure', lighting_path, lighting_sha, final_path, finals[face])):
            preparation, preparation_sha = record(case/'work'/name/'preparation.json', {'current': name})
            assets[preparation] = preparation_sha
            report_path = str(case/'work'/name/'result.json')
            report = dict(source=source_path, source_sha256=source_sha, candidate=output,
                candidate_sha256=output_sha, eligible_for_visual_review=True, hard_failures=[],
                visual_approved=False, model_uv_verified=False, strict_structure_verified=False,
                material_classification_complete=False, selected_for_delivery=False,
                artifacts=dict(result=report_path, preparation=preparation, candidate=output),
                artifact_sha256={preparation:preparation_sha, output:output_sha}, input_sha256={source_path:source_sha})
            _, report_sha = record(Path(report_path), report); assets[report_path] = report_sha
            faces[face]['checks'][name] = report
        details[face].update(selected_stage='structure_review', structure='review_required', lighting='review_applied')
    record(case/'work/intake.json', dict(source_root=str(source), skus={sku:intake}))
    row = dict(sku=sku, process=str(case), source=str(source), candidate=str(case/'candidate'),
        status='source_preserved_needs_review', six_faces_generated=True, six_faces_processed=False,
        quality_checks_complete=False, visual_review_passed=False, model_uv_verified=False,
        visual_approved=False, strict_structure_verified=False, material_classification_complete=False,
        selected_for_delivery=False, final_sha256=finals, per_face_stages=details,
        stage_status=dict(geometry='complete', lighting='review_required', structure='review_required', seam='blocked'),
        reference_review_required=True, source_reference_status_by_face=dict(left='source_tail_applicability_unknown'),
        review_warnings=[dict(kind='illumination_assumption_review', remaining_checks=['visual', 'UV'])],
        origin_note='/old/unavailable/capture/report.json',
        portable_review_evidence=dict(schema=bundle.EVIDENCE_SCHEMA, options=options, assets=assets, faces=faces))
    return row, case, raw, options


def export(tmp_path, structure=False):
    row, case, raw, options = fixture(tmp_path, structure)
    root = tmp_path/'portable'
    descriptor = bundle.package(row, case, root)
    return row, case, raw, options, root, descriptor


def repin(root, transform):
    path = root/'manifest.json'
    value = json.loads(path.read_bytes()); transform(value)
    data = json.dumps(value, sort_keys=True).encode(); path.write_bytes(data)
    return sha(data)


@pytest.mark.parametrize('structure', [False, True])
def test_move_bundle_and_originals_remove_case_then_verify_and_replay(tmp_path, structure):
    row, case, raw, options, root, descriptor = export(tmp_path, structure)
    original = copy.deepcopy(row)
    moved = tmp_path/'moved-bundle'; root.rename(moved)
    new_raw_dir = tmp_path/'moved-originals'; new_raw_dir.mkdir()
    current = {}
    for face, path in raw.items():
        current[face] = new_raw_dir/path.name; shutil.copyfile(path, current[face])
    shutil.rmtree(case); shutil.rmtree(next(iter(raw.values())).parent)
    verified = bundle.verify(moved, current, sku=row['sku'], options=options,
                             expected_manifest_sha256=descriptor['manifest_sha256'])
    assert verified['historical_paths_read'] is False
    target = tmp_path/'replayed'
    result = bundle.replay(moved, current, sku=row['sku'], options=options, case=target,
                           expected_manifest_sha256=descriptor['manifest_sha256'])
    for field in ('status', 'stage_status', 'review_warnings', 'quality_checks_complete',
                  'six_faces_processed', 'source_reference_status_by_face', 'final_sha256'):
        assert result[field] == original[field]
    assert row == original
    assert result['review_bundle_provenance']['historical_paths_read'] is False
    assert result['review_bundle_provenance']['pipeline_stages_executed'] == []
    for face in bundle.FACES:
        assert (target/'candidate'/row['sku']/(face+'.png')).read_bytes() == PNG
    assert not case.exists()


def test_six_pngs_without_current_evidence_do_not_make_a_package(tmp_path):
    row, case, _, _ = fixture(tmp_path)
    del row['portable_review_evidence']
    with pytest.raises(bundle.BundleError, match='Current producer evidence'):
        bundle.package(row, case, tmp_path/'not-created')
    assert not (tmp_path/'not-created').exists()


@pytest.mark.parametrize('field', ['asset', 'candidate'])
def test_missing_recorded_digest_never_becomes_an_unpinned_export(tmp_path, field):
    row, case, _, _ = fixture(tmp_path)
    if field == 'asset':
        key = next(iter(row['portable_review_evidence']['assets']))
        row['portable_review_evidence']['assets'][key] = None
    else:
        row['final_sha256']['front'] = None
    with pytest.raises(bundle.BundleError, match='Invalid SHA-256'):
        bundle.package(row, case, tmp_path/'not-created')
    assert not (tmp_path/'not-created').exists()


@pytest.mark.parametrize('field', bundle.FACE_VERIFIED)
def test_unverified_current_face_never_exports(tmp_path, field):
    row, case, _, _ = fixture(tmp_path)
    row['portable_review_evidence']['faces']['front']['checks'][field] = False
    with pytest.raises(bundle.BundleError, match='hard checks incomplete'):
        bundle.package(row, case, tmp_path/'not-created')
    assert not (tmp_path/'not-created').exists()


@pytest.mark.parametrize('kind', ['hard_failure', 'broken_chain', 'missing_artifact'])
def test_structure_review_requires_both_fresh_stage_reports(tmp_path, kind):
    row, case, _, _ = fixture(tmp_path, structure=True)
    evidence = row['portable_review_evidence']; report = evidence['faces']['left']['checks']['structure']
    if kind == 'hard_failure':
        report['hard_failures'] = ['real donor conflict']
    elif kind == 'broken_chain':
        report['source_sha256'] = '0'*64
    else:
        del evidence['assets'][report['artifacts']['preparation']]
    with pytest.raises(bundle.BundleError):
        bundle.package(row, case, tmp_path/'not-created')
    assert not (tmp_path/'not-created').exists()


def test_identical_asset_aliases_are_deduplicated_without_losing_bindings(tmp_path):
    row, case, raw, options = fixture(tmp_path)
    first = next(iter(row['portable_review_evidence']['assets']))
    alias, digest = write(case/'work/duplicate-snapshot', Path(first).read_bytes())
    row['portable_review_evidence']['assets'][alias] = digest
    descriptor = bundle.package(row, case, tmp_path/'portable')
    bundle.verify(tmp_path/'portable', raw, sku=row['sku'], options=options,
                  expected_manifest_sha256=descriptor['manifest_sha256'])
    manifest = json.loads((tmp_path/'portable/manifest.json').read_bytes())
    assert len(manifest['origin_provenance']['asset_origins']['assets/'+digest+'.blob']) == 2


@pytest.mark.parametrize('kind', ['source', 'model', 'config', 'pin', 'sku'])
def test_mismatches_reject_before_creating_replay_case(tmp_path, kind):
    row, _, raw, options, root, descriptor = export(tmp_path)
    expected = descriptor['manifest_sha256']; sku = row['sku']
    if kind == 'source': raw['front'].write_bytes(b'changed capture')
    elif kind == 'model': options['model'] = 'iphone17promax'
    elif kind == 'config': options['side_candidate_review'] = True
    elif kind == 'pin': expected = '0'*64
    else: sku = 'different-device'
    target = tmp_path/'not-created'
    with pytest.raises(bundle.BundleError):
        bundle.replay(root, raw, sku=sku, options=options, case=target, expected_manifest_sha256=expected)
    assert not target.exists()


@pytest.mark.parametrize('kind', ['stale', 'missing', 'extra', 'symlink'])
def test_bundle_files_fail_closed(tmp_path, kind):
    row, _, raw, options, root, descriptor = export(tmp_path)
    asset = next((root/'assets').iterdir())
    if kind == 'stale': asset.write_bytes(b'changed evidence')
    elif kind == 'missing': asset.unlink()
    elif kind == 'extra': (root/'extra.txt').write_text('unlisted')
    else:
        outside = tmp_path/'outside'; outside.write_bytes(asset.read_bytes())
        asset.unlink(); asset.symlink_to(outside)
    with pytest.raises(bundle.BundleError):
        bundle.verify(root, raw, sku=row['sku'], options=options, expected_manifest_sha256=descriptor['manifest_sha256'])


@pytest.mark.parametrize('kind', ['unknown_schema', 'traversal', 'absolute_path', 'approval', 'missing_semantics'])
def test_newly_pinned_but_invalid_manifest_is_still_rejected(tmp_path, kind):
    row, _, raw, options, root, _ = export(tmp_path)
    def change(manifest):
        if kind == 'unknown_schema': manifest['schema'] = 'unknown'
        elif kind == 'traversal': manifest['faces']['front']['path'] = '../outside.png'
        elif kind == 'absolute_path': manifest['faces']['front']['path'] = '/outside.png'
        elif kind == 'approval': manifest['semantic']['row']['visual_approved'] = True
        else: del manifest['semantic']['evidence']['faces']['front']['checks']
    expected = repin(root, change)
    with pytest.raises(bundle.BundleError):
        bundle.verify(root, raw, sku=row['sku'], options=options, expected_manifest_sha256=expected)


def test_external_unlisted_asset_is_not_read_as_current_evidence(tmp_path):
    row, case, _, _ = fixture(tmp_path)
    external, digest = write(tmp_path/'old-history/report.json', b'old report')
    row['portable_review_evidence']['assets'][external] = digest
    with pytest.raises(bundle.BundleError, match='outside current case/runtime'):
        bundle.package(row, case, tmp_path/'not-created')


def test_repinning_semantics_cannot_override_the_immutable_current_report(tmp_path):
    row, _, raw, options, root, _ = export(tmp_path, structure=True)
    def change(manifest):
        report = manifest['semantic']['evidence']['faces']['left']['checks']['structure']
        report['claimed_extra_check'] = 'not present in the actual producer report'
    expected = repin(root, change)
    with pytest.raises(bundle.BundleError, match='semantic checks differ'):
        bundle.verify(root, raw, sku=row['sku'], options=options, expected_manifest_sha256=expected)


def test_existing_replay_case_is_never_overwritten(tmp_path):
    row, _, raw, options, root, descriptor = export(tmp_path)
    target = tmp_path/'already-there'; target.mkdir(); (target/'keep').write_text('original')
    with pytest.raises(bundle.BundleError, match='fresh case'):
        bundle.replay(root, raw, sku=row['sku'], options=options, case=target,
                      expected_manifest_sha256=descriptor['manifest_sha256'])
    assert (target/'keep').read_text() == 'original'
