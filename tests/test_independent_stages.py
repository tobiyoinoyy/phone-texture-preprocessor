import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from phone_texture import stage_processing as stages, automatic, source_antenna_evidence



def exported(path, source, face, model='iphone17pro'):
    """2026-11-09: the back is resampled once to the model's standard canvas."""
    if face != 'back':
        return path.read_bytes() == source.read_bytes()
    from phone_texture.back_canvas import standardize, STANDARD_BACK_SIZE
    import tempfile
    from PIL import Image as _I
    expected = Path(tempfile.mkdtemp())/'back.png'; standardize(source, expected, model)
    return path.read_bytes() == expected.read_bytes() and _I.open(path).size == STANDARD_BACK_SIZE[model]

def checkpoint(tmp_path, name='old'):
    source = tmp_path/'raw'
    source.mkdir(exist_ok=True)
    for face in stages.pipeline.FACES:
        path = source/(face+'.png')
        if not path.exists():
            Image.new('RGBA', (40, 80), (90, 100, 110, 255)).save(path)
    case = tmp_path/name
    manifest = stages.w.intake(source, case/'work', single_name='sku')
    folder = case/'work/geometry-records/staged/sku'
    folder.mkdir(parents=True)
    files, items = {}, []
    for face in stages.pipeline.FACES:
        p = folder/(face+'.png')
        shutil.copy2(source/(face+'.png'), p)
        files[face] = p
        items.append(dict(sku='sku', face=face, source=manifest['skus']['sku'][face]['snapshot'],
            source_sha256=stages.w.sha(source/(face+'.png')), output_sha256=stages.w.sha(p),
            geometry=dict(output_size=[40, 80]), review_reasons=[]))
    geometry = dict(candidates_complete=True, geometry_status='needs_review', images=items)
    stages.write(case/'work/geometry-records/report.json', geometry)
    return case, files, geometry, manifest


def fake_stages(monkeypatch, failed_faces=(), failed_side=None, seam_failure=False):
    observed = dict(lighting=[], structure_inputs=None)
    monkeypatch.setattr(source_antenna_evidence, 'create', lambda *a, **k: {})
    def propose(files, **kwargs):
        side = kwargs['sides'][0]
        if side == failed_side:
            raise ValueError('Opaque antenna endpoint lacks paired source evidence')
        return {side: {'source_sha256': stages.w.sha(files[side])}}
    monkeypatch.setattr(stages.auto_structure, 'propose', propose)
    def material(source, face, folder, **kwargs):
        folder.mkdir(parents=True)
        observed['lighting'].append(face)
        shape = (80, 40)
        job = dict(face=face, source=str(source))
        return job, {}, np.ones(shape, bool), np.zeros(shape, bool)
    monkeypatch.setattr(stages.auto_material, 'propose_job', material)
    monkeypatch.setattr(stages, 'apply_profile', lambda job, *args: (job, {}))
    monkeypatch.setattr(stages.auto_material, 'lighting_scope', lambda *args: {})
    def light(engine, path):
        job = stages.w.read(path)
        target = Path(job['output_dir'])/(job['face']+'.png')
        target.parent.mkdir(parents=True)
        image = np.asarray(Image.open(job['source'])).copy()
        image[..., :3] += 10
        Image.fromarray(image).save(target)
        if job['face'] in failed_faces:
            raise ValueError('Lighting correction discontinuity at test')
        return {'views': {job['face']: {}}}
    monkeypatch.setattr(stages.lighting, 'run', light)
    monkeypatch.setattr(stages, 'measure_effect', lambda *args: {'passed': True})
    monkeypatch.setattr(stages.lighting_review, 'evidence_images', lambda *args: {})
    monkeypatch.setattr(stages.auto_structure, 'bind_lighting_sources', lambda proposal, original, selected: proposal)
    def structure(folder, spec, process):
        files = stages.pipeline.batch_files(folder)['sku']
        observed['structure_inputs'] = {f: stages.w.sha(p) for f, p in files.items()}
        dest = process/'candidate/sku'
        dest.mkdir(parents=True)
        for face, path in files.items():
            data = np.asarray(Image.open(path)).copy()
            if face in ('left', 'right'):
                data[..., :3] += 1
            Image.fromarray(data).save(dest/(face+'.png'))
        return dict(checks={'test': True}, outputs={'sku': {f: stages.w.sha(dest/(f+'.png')) for f in files}})
    monkeypatch.setattr(stages.sr, 'prepare', structure)
    def seams(work, sku, model, output, records):
        if seam_failure:
            raise ValueError('UV seam color difference exceeds the delivery limit')
        output.mkdir(parents=True)
        for face in stages.pipeline.FACES:
            shutil.copy2(work/'structure-records/candidate'/sku/(face+'.png'), output/(face+'.png'))
        return dict(output_sha256={f: stages.w.sha(output/(f+'.png')) for f in stages.pipeline.FACES})
    monkeypatch.setattr(stages.frame_seams, 'run', seams)
    return observed


def run_stages(case, files, geometry, manifest):
    return stages.process_stages(case, 'sku', geometry, files, manifest, 'iphone17pro',
        'none', 'accepted-shadow-lift-v1', lambda *a, **k: None, lambda *a: None)


def test_failed_back_lighting_does_not_stop_other_faces_or_supported_structure(tmp_path, monkeypatch):
    case, files, geometry, manifest = checkpoint(tmp_path)
    observed = fake_stages(monkeypatch, failed_faces=('back',))
    result = run_stages(case, files, geometry, manifest)
    assert observed['lighting'] == list(stages.pipeline.FACES)
    assert observed['structure_inputs']['back'] == stages.w.sha(files['back'])
    assert observed['structure_inputs']['left'] != stages.w.sha(files['left'])
    assert result['stage_status'] == dict(geometry='complete', lighting='partial', structure='complete', seam='complete')
    assert result['structural_alignment_applied'] and not result['six_faces_processed']
    assert result['per_face_stages']['back']['selected_stage'] == 'geometry'
    assert (case/'work/lighting-attempts/back/back.png').exists()
    assert exported(case/'candidate/sku/back.png', files['back'], 'back')


def test_one_side_proposal_failure_does_not_block_independent_lighting(tmp_path, monkeypatch):
    case, files, geometry, manifest = checkpoint(tmp_path)
    observed = fake_stages(monkeypatch, failed_side='left')
    result = run_stages(case, files, geometry, manifest)
    assert observed['lighting'] == list(stages.pipeline.FACES)
    assert result['stage_status']['lighting'] == 'complete'
    assert result['stage_status']['structure'] == 'blocked'
    assert not result['structural_alignment_applied'] and not result['seam_accepted']
    assert result['blocked_checks'][0]['face'] == 'left'
    assert (case/'candidate/sku/top.png').read_bytes() != files['top'].read_bytes()


def test_seam_failure_retains_verified_structure_and_lighting(tmp_path, monkeypatch):
    case, files, geometry, manifest = checkpoint(tmp_path)
    fake_stages(monkeypatch, seam_failure=True)
    result = run_stages(case, files, geometry, manifest)
    assert result['structural_alignment_applied'] and not result['seam_accepted']
    assert result['stage_status']['seam'] == 'failed'
    assert result['per_face_stages']['left']['selected_stage'] == 'structure'
    for face in stages.pipeline.FACES:
        assert exported(case/'candidate/sku'/(face+'.png'), case/'work/structure-records/candidate/sku'/(face+'.png'), face)
    assert not result['visual_review_passed'] and not result['quality_checks_complete']


def test_unresolved_paper_hint_is_not_rendered_or_counted_as_lighting_complete(tmp_path, monkeypatch):
    case, files, geometry, manifest = checkpoint(tmp_path)
    observed = fake_stages(monkeypatch)
    original = stages.auto_material.propose_job
    def unresolved(source, face, folder, **kwargs):
        job, analysis, allowed, paper = original(source, face, folder, **kwargs)
        if face == 'bottom':
            job['views'] = [dict(source_label_location_hint_mask='diagnostic-location-only',
                source_label_material_evidence=dict(material_boundary_resolution='unresolved_no_source_cross_edge'))]
        return job, analysis, allowed, paper
    monkeypatch.setattr(stages.auto_material, 'propose_job', unresolved)
    result = run_stages(case, files, geometry, manifest)
    assert observed['lighting'] == list(stages.pipeline.FACES)
    assert result['stage_status']['lighting'] == 'partial'
    assert result['structural_alignment_applied']
    assert result['stage_status']['seam'] == 'blocked' and not result['seam_accepted']
    assert not result['lighting_complete'] and not result['six_faces_processed']
    assert result['per_face_stages']['bottom']['material_status'] == 'unresolved'
    assert result['per_face_stages']['bottom']['selected_stage'] == 'geometry'
    assert 'bottom' not in result['lighting_checks']
    assert not (case/'work/lighting-attempts/bottom/bottom.png').exists()
    assert (case/'candidate/sku/bottom.png').read_bytes() == files['bottom'].read_bytes()
    assert any(b['stage'] == 'material' and b['kind'] == 'material_classification' for b in result['blocked_checks'])


def test_resume_reuses_exact_geometry_without_forging_visual_approval(tmp_path):
    old, files, geometry, old_manifest = checkpoint(tmp_path)
    new = tmp_path/'new'
    manifest = stages.w.intake(tmp_path/'raw', new/'work', single_name='sku')
    resumed, copied = stages.resume_geometry(old, new, 'sku', manifest)
    assert resumed['geometry_status'] == 'needs_review'
    assert not resumed['resume_provenance']['visual_approval_added']
    for face in files:
        assert copied[face].read_bytes() == files[face].read_bytes()


@pytest.mark.parametrize('damage', ['geometry', 'raw', 'sku'])
def test_resume_rejects_changed_or_cross_sku_evidence(tmp_path, damage):
    old, files, geometry, old_manifest = checkpoint(tmp_path)
    new = tmp_path/'new'
    manifest = stages.w.intake(tmp_path/'raw', new/'work', single_name='sku')
    if damage == 'geometry':
        Image.new('RGBA', (40, 80), 'red').save(files['left'])
    elif damage == 'raw':
        Image.new('RGBA', (40, 80), 'red').save(tmp_path/'raw/left.png')
    else:
        geometry['images'][0]['sku'] = 'other'
        stages.write(old/'work/geometry-records/report.json', geometry)
    with pytest.raises(ValueError):
        stages.resume_geometry(old, new, 'sku', manifest)
    assert not (new/'work/geometry-records/staged').exists()


def test_full_effect_warning_candidate_exports_and_hash_tampering_fails(tmp_path):
    old, files, geometry, manifest = checkpoint(tmp_path)
    candidate = tmp_path/'candidate/sku'
    candidate.mkdir(parents=True)
    for face, path in files.items():
        shutil.copy2(path, candidate/(face+'.png'))
    row = dict(sku='sku', status='full_stages_effect_warning_needs_review', six_faces_generated=True,
        six_faces_processed=False, candidate=str(candidate.parent), final_sha256={f: stages.w.sha(p) for f, p in files.items()})
    work = tmp_path/'exportwork'
    work.mkdir()
    assert automatic.export_complete_candidates([row], work, tmp_path/'output') == ['sku']
    Image.new('RGBA', (40, 80), 'red').save(candidate/'left.png')
    work2 = tmp_path/'badexportwork'
    work2.mkdir()
    with pytest.raises(ValueError, match='changed before export'):
        automatic.export_complete_candidates([row], work2, tmp_path/'badoutput')
    assert not (tmp_path/'badoutput').exists()
