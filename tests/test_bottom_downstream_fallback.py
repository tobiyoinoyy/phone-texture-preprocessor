"""Portable R5b routing controls; copy beside test_independent_stages.py.

Prepared statically, not executed by the author. All images are tiny synthetic
fixtures created only when the root runs pytest. Port/lighting/seam algorithms
are stubbed: these tests prove selection, provenance and failure handling only.
The .png mask sentinels are deliberately text, never decoded as masks.
"""
import copy
import shutil
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from phone_texture import bottom_alignment, source_preservation
from phone_texture import stage_processing as stages
from test_independent_stages import checkpoint, fake_stages


def setup_attempt(tmp_path, monkeypatch):
    case, original_files, original_geometry, manifest = checkpoint(tmp_path)
    process = case / 'work/bottom-alignment'
    selected = process / 'selected'
    selected.mkdir(parents=True)
    files = {}
    for face, source in original_files.items():
        files[face] = selected / (face + '.png')
        shutil.copy2(source, files[face])
    # A changed, source-bound candidate, not a simulated successful port fit.
    Image.new('RGBA', (40, 80), (91, 100, 110, 255)).save(files['bottom'])
    geometry = copy.deepcopy(original_geometry)
    item = next(x for x in geometry['images'] if x['face'] == 'bottom')
    item['pre_bottom_alignment_sha256'] = item['output_sha256']
    item['output_sha256'] = stages.w.sha(files['bottom'])
    report = dict(applied=True, status='applied_with_notice',
        source=str(original_files['bottom']),
        source_sha256=stages.w.sha(original_files['bottom']),
        selected_sha256=stages.w.sha(files['bottom']), selected_dir=str(selected),
        notices=[], model_uv_verified=False, physical_accuracy_verified=False)
    stages.write(process / 'report.json', report)
    stages.write(process / 'effective-geometry.json', geometry)
    observed = fake_stages(monkeypatch)
    monkeypatch.setattr(stages.pipeline, 'overview', lambda *a, **k: None)
    monkeypatch.setattr(source_preservation, 'source_seam_observations',
        lambda *a, **k: {'test_stub': True})
    context = dict(case=case, files=files, geometry=geometry, manifest=manifest,
        original_files=original_files, original_geometry=original_geometry,
        original_geometry_snapshot=copy.deepcopy(original_geometry),
        original_sha={f: stages.w.sha(p) for f, p in original_files.items()},
        candidate_sha=report['selected_sha256'], report=report, process=process,
        observed=observed, callbacks=[], materials=[], seam_calls=[])

    def fallback(error):
        context['callbacks'].append(error)
        # The real callback mutates these same mappings, rather than replacing
        # locals that process_stages cannot see.
        return bottom_alignment.retain_source_after_downstream_failure(
            files, geometry, original_files, original_geometry, report, process, error)

    context['fallback'] = fallback
    original_material = stages.auto_material.propose_job

    def material(source, face, folder, **kwargs):
        job, analysis, allowed, paper = original_material(source, face, folder, **kwargs)
        record = dict(face=face, attempt=folder.name, source_sha256=stages.w.sha(source))
        context['materials'].append(record)
        sentinel = folder.name + ':' + record['source_sha256']
        for name in ('main_surface.png', 'paper.png'):
            (folder / name).write_text(sentinel, encoding='utf-8')
        nested = folder / 'source-evidence'
        nested.mkdir()
        (nested / 'binding.txt').write_text(sentinel, encoding='utf-8')
        return job, analysis, allowed, paper

    monkeypatch.setattr(stages.auto_material, 'propose_job', material)
    original_seams = stages.frame_seams.run

    def seams(work, sku, model, output, records):
        # This is the fixed canonical path consumed by the real frame_seams.
        canonical = work / 'material-inputs/bottom'
        selection = stages.w.read(canonical / 'material-selection.json')
        selected_assets = Path(selection['selected_assets'])
        expected = selected_assets.name + ':' + stages.w.sha(files['bottom'])
        assert selection['source_sha256'] == stages.w.sha(files['bottom'])
        assert Path(selection['selected_job']) == selected_assets / 'job.json'
        for name in ('main_surface.png', 'paper.png', 'source-evidence/binding.txt'):
            assert (canonical / name).read_text(encoding='utf-8') == expected
            assert selection['copied_assets_sha256'][name] == stages.w.sha(canonical / name)
        for relative, digest in selection['copied_assets_sha256'].items():
            assert stages.w.sha(selected_assets / relative) == digest
            assert stages.w.sha(canonical / relative) == digest
        context['seam_calls'].append(selected_assets.name)
        return original_seams(work, sku, model, output, records)

    monkeypatch.setattr(stages.frame_seams, 'run', seams)
    return context


def run(context, *, callback=True, allow_partial=True):
    return stages.process_stages(context['case'], 'sku', context['geometry'],
        context['files'], context['manifest'], 'iphone17pro', 'none',
        'accepted-shadow-lift-v1', lambda *a, **k: None, lambda *a: None,
        allow_partial=allow_partial,
        bottom_fallback=context['fallback'] if callback else None)


def reject_attempts(monkeypatch, names, *, phase='material', exception=ValueError):
    names = set(names)
    if phase == 'material':
        original = stages.auto_material.propose_job

        def material(source, face, folder, **kwargs):
            result = original(source, face, folder, **kwargs)
            if folder.name in names:
                message = 'Bottom paper material boundary unresolved: ' + folder.name
                stages.write(folder / 'failure-evidence.json', {'error': message})
                raise exception(message)
            return result

        monkeypatch.setattr(stages.auto_material, 'propose_job', material)
    else:
        original = stages.lighting.run

        def light(engine, path):
            result = original(engine, path)  # Keep the rejected output as diagnostic.
            if path.parent.name in names:
                raise exception('Lighting correction discontinuity: ' + path.parent.name)
            return result

        monkeypatch.setattr(stages.lighting, 'run', light)


def assert_once_except_bottom(context, bottom_attempts):
    counts = Counter(x['face'] for x in context['materials'])
    assert counts == Counter({face: bottom_attempts if face == 'bottom' else 1
        for face in stages.pipeline.FACES})


@pytest.mark.parametrize('phase', ['material', 'render'])
def test_bottom_retries_original_once_preserving_diagnostics_and_canonical_masks(tmp_path, monkeypatch, phase):
    c = setup_attempt(tmp_path, monkeypatch)
    reject_attempts(monkeypatch, ['bottom-port-candidate'], phase=phase)
    result = run(c)
    assert_once_except_bottom(c, 2)
    assert len(c['callbacks']) == 1
    assert c['seam_calls'] == ['bottom-source-fallback']
    assert all(x == 'complete' for x in result['stage_status'].values())
    assert not result['blocked_checks']
    assert not result['visual_review_passed'] and not result['quality_checks_complete']
    row = result['per_face_stages']['bottom']
    assert row['source_sha256'] == c['original_sha']['bottom']
    assert row['optional_bottom_candidate_rejection']['candidate_source_sha256'] == c['candidate_sha']
    assert row['optional_bottom_candidate_rejection']['error'] == str(c['callbacks'][0])
    assert 'ValueError' in row['optional_bottom_candidate_rejection']['traceback']
    work = c['case'] / 'work'
    failed_assets = work / 'material-inputs/bottom-port-candidate'
    assert (failed_assets / 'main_surface.png').read_text().endswith(c['candidate_sha'])
    assert (work / 'material-inputs/bottom-source-fallback/job.json').is_file()
    assert (work / 'lighting-attempts/bottom-source-fallback/bottom.png').is_file()
    if phase == 'material':
        assert (failed_assets / 'failure-evidence.json').is_file()
    else:
        assert (work / 'lighting-attempts/bottom-port-candidate/bottom.png').is_file()
    assert c['report']['status'] == 'kept_source_with_notice'
    assert c['report']['downstream_source_retry'] and not c['report']['applied']
    assert c['report']['selected_sha256'] == c['original_sha']['bottom']
    attempt = stages.w.read(c['process'] / 'attempt-report.json')
    assert attempt['applied'] and attempt['selected_sha256'] == c['candidate_sha']
    assert c['report']['attempt_report_sha256'] == stages.w.sha(c['process'] / 'attempt-report.json')
    stages.geometry_identity(c['geometry'], c['files'], 'sku')
    assert c['original_geometry'] == c['original_geometry_snapshot']
    for face in stages.pipeline.FACES:
        assert stages.w.sha(c['original_files'][face]) == c['original_sha'][face]
        assert stages.w.sha(c['files'][face]) == c['original_sha'][face]
    with pytest.raises(RuntimeError, match='Only one applied bottom candidate'):
        c['fallback'](ValueError('must not retry twice'))


def test_successful_optional_candidate_does_not_retry_and_supplies_canonical_masks(tmp_path, monkeypatch):
    c = setup_attempt(tmp_path, monkeypatch)
    result = run(c)
    assert_once_except_bottom(c, 1)
    assert c['callbacks'] == []
    assert c['seam_calls'] == ['bottom-port-candidate']
    assert 'optional_bottom_candidate_rejection' not in result['per_face_stages']['bottom']
    assert not (c['process'] / 'attempt-report.json').exists()


def test_second_material_failure_stays_blocked_without_canonical_acceptance(tmp_path, monkeypatch):
    c = setup_attempt(tmp_path, monkeypatch)
    reject_attempts(monkeypatch, ['bottom-port-candidate', 'bottom-source-fallback'])
    result = run(c)
    assert_once_except_bottom(c, 2)
    assert len(c['callbacks']) == 1 and c['seam_calls'] == []
    assert result['stage_status'] == dict(geometry='complete', lighting='partial', structure='complete', seam='blocked')
    assert not result['lighting_complete'] and not result['six_faces_processed']
    assert not result['seam_accepted']
    assert 'bottom' not in result['lighting_checks']
    row = result['per_face_stages']['bottom']
    assert row['lighting'] == 'failed' and row['material_status'] == 'unresolved'
    assert row['selected_stage'] == 'geometry'
    assert 'bottom-source-fallback' in row['lighting_error']
    assert any(b['face'] == 'bottom' and b['stage'] == 'material' for b in result['blocked_checks'])
    assert (c['case'] / 'candidate/sku/bottom.png').read_bytes() == c['original_files']['bottom'].read_bytes()
    assert not (c['case'] / 'work/material-inputs/bottom').exists()
    for name in ('bottom-port-candidate', 'bottom-source-fallback'):
        assert (c['case'] / 'work/material-inputs' / name / 'failure-evidence.json').is_file()


def test_unrelated_face_failure_never_invokes_bottom_callback(tmp_path, monkeypatch):
    c = setup_attempt(tmp_path, monkeypatch)
    reject_attempts(monkeypatch, ['back'], phase='render')
    result = run(c)
    assert_once_except_bottom(c, 1)
    assert c['callbacks'] == []
    assert result['per_face_stages']['back']['lighting'] == 'failed'
    assert result['per_face_stages']['bottom']['lighting'] == 'complete'


def test_absent_callback_keeps_single_original_attempt_contract(tmp_path, monkeypatch):
    c = setup_attempt(tmp_path, monkeypatch)
    reject_attempts(monkeypatch, ['bottom'])
    result = run(c, callback=False)
    assert_once_except_bottom(c, 1)
    assert c['callbacks'] == []
    assert result['per_face_stages']['bottom']['lighting'] == 'failed'
    assert not (c['case'] / 'work/material-inputs/bottom-source-fallback').exists()


@pytest.mark.parametrize('damage', ['geometry', 'raw'])
def test_changed_source_is_verified_before_callback_and_fails_closed(tmp_path, monkeypatch, damage):
    c = setup_attempt(tmp_path, monkeypatch)
    original = stages.auto_material.propose_job

    def material(source, face, folder, **kwargs):
        result = original(source, face, folder, **kwargs)
        if face == 'bottom':
            target = source if damage == 'geometry' else Path(c['manifest']['skus']['sku']['bottom']['snapshot'])
            target.write_bytes(b'changed source; never valid for fallback')
            raise ValueError('Candidate material failed after source changed')
        return result

    monkeypatch.setattr(stages.auto_material, 'propose_job', material)
    with pytest.raises(ValueError):
        run(c)
    assert c['callbacks'] == []
    assert not (c['process'] / 'source-fallback-selected').exists()
    assert not (c['case'] / 'work/material-inputs/bottom-source-fallback').exists()


@pytest.mark.parametrize('exception', [OSError, RuntimeError])
def test_io_and_runtime_integrity_errors_do_not_trigger_fallback(tmp_path, monkeypatch, exception):
    c = setup_attempt(tmp_path, monkeypatch)
    reject_attempts(monkeypatch, ['bottom-port-candidate'], exception=exception)
    with pytest.raises(exception, match='bottom-port-candidate'):
        run(c)
    assert c['callbacks'] == []
    assert not (c['case'] / 'work/material-inputs/bottom-source-fallback').exists()


@pytest.mark.parametrize('incomplete', ['neither_mapping', 'files_only'])
def test_callback_cannot_return_an_unbound_source(tmp_path, monkeypatch, incomplete):
    c = setup_attempt(tmp_path, monkeypatch)
    reject_attempts(monkeypatch, ['bottom-port-candidate'])

    def incomplete_callback(error):
        c['callbacks'].append(error)
        restored = c['original_files']['bottom']
        if incomplete == 'files_only':
            c['files']['bottom'] = restored
        return restored

    c['fallback'] = incomplete_callback
    expected = RuntimeError if incomplete == 'neither_mapping' else ValueError
    with pytest.raises(expected):
        run(c)
    assert len(c['callbacks']) == 1
    assert not (c['case'] / 'work/material-inputs/bottom-source-fallback').exists()


def test_canonical_copy_corruption_is_a_hard_integrity_failure(tmp_path, monkeypatch):
    c = setup_attempt(tmp_path, monkeypatch)
    original = stages.shutil.copytree

    def corrupt_copy(source, target, *args, **kwargs):
        copied = original(source, target, *args, **kwargs)
        if Path(target) == c['case'] / 'work/material-inputs/bottom':
            (Path(target) / 'paper.png').write_text('wrong selected mask', encoding='utf-8')
        return copied

    monkeypatch.setattr(stages.shutil, 'copytree', corrupt_copy)
    with pytest.raises(RuntimeError, match='Final material alias changed during copy'):
        run(c)
    assert c['callbacks'] == [] and c['seam_calls'] == []
    assert not (c['case'] / 'work/material-inputs/bottom/material-selection.json').exists()
