# -*- coding: utf-8 -*-
"""Explicit, source-bound continuation of saved review candidates.

This does not generate new material/endpoint evidence or run image corrections.
Current six sources, options, historical evidence and selected bytes must match.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil

from . import workspace as w
from .pipeline import FACES

SCHEMA = 'phone_texture_saved_review_manifest/v1'
MODE = 'saved_bound_face_outputs_review_v1'
MODELS = {'iphone17pro': 'ip17pro', 'iphone17promax': 'ip17promax'}
OPTION_KEYS = {'model','protection_policy','review_fallback','lighting_profile',
               'bottom_alignment','source_body_reference_review'}


class SavedReviewError(ValueError):
    pass


def require(value, message):
    if not value:
        raise SavedReviewError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def valid_sha(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


class Audit:
    def __init__(self, materialize=False):
        self.hashes = {}
        self.materialize = materialize

    def file(self, path, expected=None):
        path = Path(path).expanduser().resolve(strict=True)
        require(path.is_file(), 'Expected evidence file: '+str(path))
        digest = sha(path)
        if expected is not None:
            require(valid_sha(expected) and digest==expected, 'Stale source/evidence SHA: '+str(path))
        require(self.hashes.setdefault(str(path),digest)==digest, 'Evidence changed: '+str(path))
        return digest

    def json(self, path, expected=None):
        self.file(path,expected)
        value=json.loads(Path(path).read_text(encoding='utf-8'))
        require(isinstance(value,dict), 'Expected JSON evidence object')
        return value

    def finish(self):
        for path,digest in self.hashes.items():
            require(sha(path)==digest, 'Source/evidence changed during continuation: '+path)


def binding(audit, entry):
    require(isinstance(entry,dict) and set(entry)=={'path','sha256'}, 'Explicit evidence path/SHA required')
    require(Path(entry['path']).is_absolute(), 'Saved evidence paths must be explicit absolute paths')
    return audit.json(entry['path'],entry['sha256'])


def copy_bound(audit, source, target, expected):
    audit.file(source,expected)
    if not audit.materialize:
        return
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    require(not target.exists(), 'Refusing saved artifact overwrite')
    shutil.copy2(source,target)
    require(sha(target)==expected, 'Selected artifact changed during copy')


def reference_status(side):
    value=side['source_reference_status']
    return value.get('original_status') if isinstance(value,dict) else value


def _manifest(path, expected=None):
    audit=Audit();path=Path(path).expanduser().resolve(strict=True)
    data=audit.json(path,expected)
    require(data.get('schema')==SCHEMA and isinstance(data.get('entries'),list), 'Unknown saved review schema')
    identities=set()
    for entry in data['entries']:
        require(isinstance(entry,dict), 'Malformed saved review entry')
        model,sku=entry.get('model'),entry.get('sku')
        require(model in MODELS and isinstance(sku,str) and sku not in ('','.','..')
            and Path(sku).name==sku, 'Invalid saved model/SKU identity')
        require((model,sku) not in identities, 'Duplicate saved model/SKU entry')
        identities.add((model,sku))
        require(entry.get('key')==MODELS[model]+'/'+sku, 'Saved model/key mismatch')
        hashes=entry.get('source_sha256')
        require(isinstance(hashes,dict) and set(hashes)==set(FACES)
            and all(valid_sha(v) for v in hashes.values()), 'Six explicit original-source hashes required')
        options=entry.get('options')
        require(isinstance(options,dict) and set(options)==OPTION_KEYS and options['model']==model,
            'Saved processing options are incomplete')
        require(options==dict(model=model,protection_policy='none',review_fallback='preserve-source',
            lighting_profile='accepted-shadow-lift-v2',bottom_alignment='appearance',
            source_body_reference_review=True) and type(options['source_body_reference_review']) is bool,
            'Unsupported saved-review processing configuration')
        require(set(entry.get('sides',{}))=={'left','right'}, 'Both side selections must be explicit')
    audit.finish()
    return path,audit.hashes[str(path)],data


def _current_sources(source):
    files=w.face_files(Path(source))
    return files,{face:sha(path) for face,path in files.items()}


def _record(entry,source):
    return dict(unique_source_key=entry['key'],sku=entry['sku'],
        existing_raw_copy=str(Path(source).resolve()),source_hashes=entry['source_sha256'])


def load_selected(manifest_path, source_by_sku, options, *, expected_manifest_sha256=None):
    """Preflight before work creation; an invalid matched entry never becomes a miss."""
    path,digest,data=_manifest(manifest_path,expected_manifest_sha256)
    require(isinstance(options,dict) and set(options)==OPTION_KEYS and options['model'] in MODELS,
        'Explicit current model and processing options required')
    require(type(options['source_body_reference_review']) is bool, 'Current reference option must be bool')
    selected={}
    for sku,source in source_by_sku.items():
        candidates=[(i,e) for i,e in enumerate(data['entries']) if e['sku']==sku]
        matches=[(i,e) for i,e in candidates if e['model']==options['model']]
        if not candidates:
            continue
        files,current=_current_sources(source)
        if not matches:
            require(not any(e['source_sha256']==current for _,e in candidates),
                'Saved original capture belongs to another requested model')
            continue
        index,entry=matches[0]
        require(entry['source_sha256']==current, 'Current six sources differ from saved entry: '+sku)
        require(entry['options']==options, 'Saved review processing options differ: '+sku)
        audit=Audit();audit.file(path,digest)
        for face,file in files.items():audit.file(file,current[face])
        # Dry validation does not create case files or approve any image.
        _assemble(audit,entry,_record(entry,source),Path(source).parent/('.unwritten-review-'+sku))
        audit.finish()
        selected[sku]=dict(manifest_path=str(path),manifest_sha256=digest,entry_index=index,
            entry_sha256=object_sha(entry),key=entry['key'],source=str(Path(source).resolve()),
            options=copy.deepcopy(options))
    require(sha(path)==digest, 'Manifest changed during preflight')
    return selected


def process_saved(descriptor, *, sku, source, case, options):
    """Revalidate inside the real worker, then copy selected bytes into its new case."""
    require(isinstance(descriptor,dict) and set(descriptor)=={
        'manifest_path','manifest_sha256','entry_index','entry_sha256','key','source','options'},
        'Malformed saved-review task descriptor')
    source,case=Path(source).resolve(),Path(case).resolve()
    require(str(source)==descriptor['source'] and options==descriptor['options'],
        'Saved task source/options changed after preflight')
    current=load_selected(descriptor['manifest_path'],{sku:source},options,
        expected_manifest_sha256=descriptor['manifest_sha256'])
    require(current.get(sku)==descriptor, 'Saved descriptor no longer selects the same evidence')
    _,_,data=_manifest(descriptor['manifest_path'],descriptor['manifest_sha256'])
    entry=data['entries'][descriptor['entry_index']]
    require(object_sha(entry)==descriptor['entry_sha256'], 'Saved entry changed')
    case.mkdir(parents=True,exist_ok=False)
    intake=w.intake(source,case/'work',single_name=sku)
    require({f:record['sha256'] for f,record in intake['skus'][sku].items()}==entry['source_sha256'],
        'Current intake differs from saved capture')
    audit=Audit(materialize=True);audit.file(descriptor['manifest_path'],descriptor['manifest_sha256'])
    for record in intake['skus'][sku].values():
        audit.file(record['source'],record['sha256']);audit.file(record['snapshot'],record['sha256'])
    row=_assemble(audit,entry,_record(entry,source),case)
    from PIL import Image
    for face,digest in row['final_sha256'].items():
        candidate=Path(row['candidate'])/sku/(face+'.png')
        require(sha(candidate)==digest, 'Final selected bytes changed')
        with Image.open(candidate) as image:
            require(image.format=='PNG' and image.mode=='RGBA','Selected review face must be RGBA PNG')
            image.verify()
    w.check_sources(intake);audit.finish()
    row.update(saved_review_reused=True,saved_processing_results_reused=True,
        historical_processing_evidence_reused=True,human_approval_reused=False,
        source_body_reference_review_enabled=options['source_body_reference_review'],
        pipeline_rerun=False,events=[dict(stage='verified_saved_review_continuation',seconds=0)],
        saved_review_provenance=dict(manifest_path=descriptor['manifest_path'],
            manifest_sha256=descriptor['manifest_sha256'],entry_sha256=descriptor['entry_sha256'],
            worker_pid=os.getpid(),source_revalidated_in_worker=True,byte_copy_only=True,
            pipeline_stages_executed=[],input_dependencies_unchanged=True,
            source_hashes=entry['source_sha256'],input_sha256=audit.hashes))
    w.save(case/'work/saved-review/provenance.json',row['saved_review_provenance'])
    w.save(case/'work/stage-report.json',row)
    w.save(case/'result.json',row)
    return row


def _assemble(audit, plan, record, case):
    from .saved_review_bindings import validate_side
    key, sku = record['unique_source_key'], record['sku']
    baseline = binding(audit, plan['baseline_result'])
    require(baseline.get('sku') == sku and baseline.get('six_faces_generated') is True,
            'R6 baseline must have six actual files for this SKU')
    baseline_path = Path(plan['baseline_result']['path']).resolve()
    baseline_group = baseline_path.parent
    require(baseline_path == (baseline_group/'actual-result.json').resolve(), 'Use the actual same-key R6 baseline')
    require(set(baseline.get('final_sha256', {})) == set(FACES), 'R6 six-face hash record incomplete')
    copy_bound(audit, baseline_path, case/'work/saved-review/evidence/R6-actual-result.json', plan['baseline_result']['sha256'])
    geometry_report = binding(audit, plan['geometry_report'])
    geometry_path = Path(plan['geometry_report']['path'])
    from .saved_review_baseline import validate_baseline
    geometry = validate_baseline(audit, plan, baseline, geometry_report)
    require(set(geometry) == set(FACES), 'R6 original geometry identities incomplete')
    for face in FACES:
        audit.file(geometry_path.parent/'staged'/sku/(face+'.png'), geometry[face])
        audit.file(baseline_group/'output'/sku/(face+'.png'), baseline['final_sha256'][face])
    baseline['_geometry_sha256'] = geometry
    baseline['_selected_source_hashes'] = record['source_hashes']
    row = copy.deepcopy(baseline)
    row.pop('_geometry_sha256', None); row.pop('_selected_source_hashes', None)
    row['historical_blocked_checks'] = copy.deepcopy(baseline.get('blocked_checks', []))
    row['historical_review_warnings'] = copy.deepcopy(baseline.get('review_warnings', []))
    row['source_baseline_evidence'] = dict(plan['baseline_result'],
        blocked_checks_sha256=object_sha(row['historical_blocked_checks']),
        review_warnings_sha256=object_sha(row['historical_review_warnings']),
        role='historical_strict_checks_not_reclassified_as_pass')
    row['review_warnings'] = [copy.deepcopy(w) for w in baseline.get('review_warnings', [])
        if not (w.get('face') in ('left', 'right') and
                w.get('code') in ('structure_proposal_blocked', 'lighting_blocked'))]
    require(set(plan.get('sides', {})) == {'left', 'right'}, 'Each side requires an explicit saved-lighting selection')
    selected = {}; sides = {}; applied = []
    for face in FACES:
        if face in ('left', 'right'):
            side = validate_side(key, face, plan['sides'][face], baseline, audit)
            require(side['lighting_sha256'] == plan['sides'][face]['expected_lighting_sha256'] and
                    side['selected_sha256'] == plan['sides'][face]['expected_selected_sha256'],
                    'Selected candidate differs from the explicit assembly plan: '+face)
            sides[face] = side
            selected[face] = (Path(side['selected_path']), side['selected_sha256'])
            if side['structure_applied']:
                applied.append(face)
        else:
            selected[face] = (baseline_group/'output'/sku/(face+'.png'), baseline['final_sha256'][face])
    candidate = case/'candidate'
    for face, (source, digest) in selected.items():
        copy_bound(audit, source, candidate/sku/(face+'.png'), digest)
    for face, side in sides.items():
        entry = dict(lighting='review_applied', structure='review_required', seam='blocked',
            selected_stage='structure_review' if side['structure_applied'] else 'lighting_review',
            source_sha256=geometry[face], lighting_sha256=side['lighting_sha256'], lighting_file=side['lighting_path'],
            selected_sha256=side['selected_sha256'], selected_stage_file=str(candidate/sku/(face+'.png')),
            structure_operation_applied=side['structure_applied'], strict_structure_verified=False,
            strict_endpoint_evidence_complete=False, material_classification_complete=False,
            illumination_assumption_review=True, structure_review_required=True,
            source_reference_status=reference_status(side), reference_review_required=side['reference_review_required'],
            review_evidence=side['review_evidence'], selected_for_review_export=True,
            strict_checks_passed=False, visual_approved=False, model_uv_verified=False)
        row['per_face_stages'][face] = entry
        row['face_processing'][face] = entry['selected_stage']
        row.setdefault('lighting_review_checks', {})[face] = side['lighting_checks']
        row.setdefault('item_effect_warnings', []).extend(side['warnings'])
        # Persist full reports alongside the assembly, keeping original files and
        # their selected_for_delivery=False flags byte-identical.
        for name in ('lighting_result', 'structure_result'):
            bound = plan['sides'][face].get(name)
            if bound:
                copy_bound(audit, bound['path'], case/'work/saved-review/evidence'/face/(name+'.json'), bound['sha256'])
    row.update(status='source_preserved_needs_review', sku=sku, candidate=str(candidate), process=str(case),
        source=str(Path(record['existing_raw_copy']).resolve()), final_sha256={f: d for f, (_, d) in selected.items()},
        stage_status=dict(geometry='complete', lighting='review_required', structure='review_required', seam='blocked'),
        structural_review_applied_faces=applied, structural_alignment_applied=bool(applied),
        strict_structure_verified=False, structural_alignment_certified=False, strict_endpoint_evidence_complete=False,
        material_classification_complete=False, six_faces_generated=True, six_faces_processed=False,
        mandatory_program_checks_complete=False, quality_checks_complete=False, lighting_complete=False,
        visual_review_passed=False, visual_approved=False, model_uv_verified=False, seam_accepted=False,
        seam_executed=False, selected_for_delivery=False, selected_for_review_export=True,
        saved_candidate_review_assembly=True, assembly_mode=MODE,
        review_continuation_saved=True, automatic_retry_eligible=False,
        reference_review_required=bool(baseline.get('reference_review_required')) or any(s['reference_review_required'] for s in sides.values()),
        reference_certified=False, independent_bottom_rails_certified=False,
        source_reference_status_by_face={f: reference_status(s) for f, s in sides.items()},
        assembly_review=dict(baseline=plan['baseline_result'], sides={f: s['review_evidence'] for f, s in sides.items()},
            historical_stage_status=baseline.get('stage_status'), original_strict_blocked_checks=baseline.get('blocked_checks', []),
            historical_side_stages={f: baseline['per_face_stages'][f] for f in sides},
            historical_side_failures={f: s['historical_failures'] for f, s in sides.items()},
            baseline_other_four_faces_byte_identical=True, pipeline_rerun=False,
            illumination_or_structure_recomputed=False, original_prototype_records_modified=False))
    row['blocked_checks'] = [
        dict(stage='structure', kind='strict_structure_certification_pending',
             applied_review_faces=applied,
             error=('已应用部分或全部侧面复核位置调整；' if applied else '尚未应用侧面结构复核；')+
                   '原严格端点与材料证据尚未满足，不能作为严格结构验收。',
             historical_evidence=copy.deepcopy(row['source_baseline_evidence'])),
        dict(stage='seam', kind='review_candidate_seam_unverified',
             error='此次仅组装已保存候选，未运行接缝处理或实际模型 UV 验证。')]
    row['blocked_check'] = copy.deepcopy(row['blocked_checks'][0])
    # Original strict failures are retained as history, never rewritten to pass
    # or presented as a current side-lighting operation that did not execute.
    require(row.get('lighting_effect_warnings') == baseline.get('lighting_effect_warnings') and
            row.get('bottom_alignment') == baseline.get('bottom_alignment') and
            row.get('historical_blocked_checks') == baseline.get('blocked_checks') and
            row.get('historical_review_warnings') == baseline.get('review_warnings'), 'Historical evidence was lost')
    for face in ('front', 'back', 'top', 'bottom'):
        require(row['per_face_stages'][face] == baseline['per_face_stages'][face]
                and row['final_sha256'][face] == baseline['final_sha256'][face], 'Unselected face changed')
    return row
