"""Resumable six-face workflow. CLI, desktop and Skills use this same runner.

Reviewed masks/landmarks are inputs. Missing evidence pauses the pipeline and
never turns a failed processing stage into a copied face or an approved result.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil

from . import __version__, pipeline, workspace as w
from .geometry_review import verify_report
from . import lighting, structure_review

STAGES = ('geometry', 'geometry_review', 'lighting_scope', 'lighting',
          'structure', 'structure_review', 'model_reference', 'publish')
LABELS = dict(zip(STAGES, ('六面几何处理', '几何复核', '光照范围', '材质光照处理',
                         '本机背面结构对齐', '结构复核', '固定模型坐标检查', '六面发布')))


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save(path, value):
    """Replace only our state atomically, including when a worker is interrupted."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    w.save(temporary, value)
    temporary.replace(path)


@contextmanager
def locked(work):
    # macOS/Linux advisory lock is released by the OS even if the process dies.
    import fcntl
    with (Path(work) / '.workflow.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('该批次正在运行，请等待当前处理结束')
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def create(source, work, output, *, input_kind='raw', material='aluminum', skus=None,
           single_name='SKU_01', alignment_basis='back_image', model_reference=None,
           delivery_goal='model_textures', goal_reason=None):
    source, work, output = pipeline.separate_paths(source, work, output)
    pipeline.fresh(output)
    if input_kind not in ('raw', 'cutout', 'existing'):
        raise ValueError('输入类型必须为 raw/cutout/existing')
    if material not in ('aluminum', 'stainless', 'mixed'):
        raise ValueError('材质必须为 aluminum/stainless/mixed')
    if alignment_basis not in ('back_image', 'front_back_images'):
        raise ValueError('结构基准必须为 back_image/front_back_images')
    reference = None
    if model_reference:
        from .model_reference import load
        p = Path(model_reference).expanduser().resolve()
        load(p)  # Check the real model and reference images before intake.
        reference = dict(path=str(p), sha256=w.sha(p))
    manifest = w.intake(source, work, skus, single_name, delivery_goal, goal_reason)
    contract = dict(output=str(output), input_kind=input_kind, material=material,
                    alignment_basis=alignment_basis, model_reference=reference,
                    delivery_goal=delivery_goal)
    manifest['workflow_contract'] = contract
    w.save(work / 'intake.json', manifest)
    config = dict(schema='phone-six-view-workflow-v1', version=__version__, **contract,
                  geometry=dict(guides=None, review=None, device='auto'),
                  lighting=dict(scope=None, jobs=None), structure=dict(spec=None, review=None),
                  model_review=None)
    save(work / 'workflow.json', config)
    return record(work, 'ready', 'geometry', '已导入六面图，可开始处理')


def record(work, status, stage, message, **extra):
    value = dict(status=status, stage=stage, stage_label=LABELS[stage], message=message, **extra)
    save(Path(work) / 'workflow-state.json', value)
    return value


def status(work):
    work = Path(work).expanduser().resolve()
    value = w.read(work / 'workflow-state.json')
    value['work'] = str(work)
    return value


def resolve(work, path):
    if not path:
        return None
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (work / p).resolve()


def waiting(work, stage, message, path=None):
    return record(work, 'needs_review', stage, message,
                  required_file=str(path) if path else None)


def check_basis(spec, basis):
    for sides in spec.values():
        for cfg in sides.values():
            if cfg.get('basis', 'back_image') != basis:
                raise ValueError('结构标定与本批冻结的对齐基准不一致')


def light_candidates(work, config, baseline, scope):
    """Build a complete candidate batch and plan using the bundled engines."""
    jobs_path = resolve(work, config['lighting'].get('jobs'))
    if not jobs_path or not jobs_path.is_file():
        return None, '请登记逐面光照任务与蒙版', work / 'lighting-jobs.json'
    jobs = w.read(jobs_path)
    if set(jobs) != set(baseline) or any(set(fs) != set(pipeline.FACES) for fs in jobs.values()):
        raise ValueError('光照任务必须覆盖本批全部 SKU 和六面')
    plan = {}
    for sku, files in baseline.items():
        plan[sku] = {}
        for face, item in files.items():
            src = Path(item['path']); intended = scope[sku][face]; entry = jobs[sku][face]
            action = intended['action']
            if action == 'aluminum' and intended.get('protection_policy') != 'sticker_only':
                raise ValueError('本项目铝材只保护真实贴纸，光照范围须使用 sticker_only')
            if config['material'] != 'mixed' and action not in ('copy', config['material']):
                raise ValueError('光照任务材质与本批材质不一致')
            if entry.get('action') != action:
                raise ValueError('光照任务未履行冻结的逐面范围')
            folder = work / 'lighting' / sku
            folder.mkdir(parents=True, exist_ok=True)
            dest = folder / (face + '.png')
            reason = intended['reason']
            if action == 'copy':
                if dest.exists() and w.sha(dest) != item['sha256']:
                    raise ValueError('原样保留面的候选已改变')
                if not dest.exists():
                    shutil.copy2(src, dest)
                plan[sku][face] = dict(action='copy', reason=reason)
                continue
            job_path = resolve(work, entry.get('job'))
            if not job_path or not job_path.is_file():
                return None, f'请补充 {sku}/{face} 的材质任务', job_path
            job = w.read(job_path)
            records = work / 'lighting-records' / sku / face
            if action == 'aluminum':
                views = job.get('views', [])
                if len(views) != 1 or views[0].get('name') != face or Path(views[0]['source']).resolve() != src:
                    raise ValueError('光照任务必须使用该面已通过复核的几何图，每个任务登记一面')
                job.update(output_dir=str(folder), records_dir=str(records))
                allowed, protected = records / (face + '-editable.png'), records / (face + '-protected.png')
            else:
                if Path(job['source']).resolve() != src:
                    raise ValueError('不锈钢任务必须绑定该面的几何基线')
                job.update(output=str(dest), records_dir=str(records))
                allowed, protected = records / 'editable.png', records / 'protected.png'
            fingerprint = signature(dict(job=job, dependencies=lighting.dependencies(job)))
            proof = records / 'workflow-proof.json'
            if proof.exists():
                saved = w.read(proof)
                if saved['signature'] != fingerprint or any(w.sha(p) != h for p, h in saved['artifacts'].items()):
                    raise ValueError(f'{sku}/{face} 光照参数、参考或输出已改变，请使用新批次重算')
            else:
                if dest.exists() or records.exists():
                    raise ValueError(f'{sku}/{face} 有未完成的光照输出，请检查后使用新批次重算')
                task = work / 'lighting-tasks' / sku / (face + '.json')
                task.parent.mkdir(parents=True, exist_ok=True)
                w.save(task, job)
                lighting.run(action, task)
                w.save(proof, dict(signature=fingerprint, artifacts={str(p): w.sha(p) for p in
                                     (dest, allowed, protected, task, records / 'validation.json')}))
            result = dict(action=action, reason=reason, output=str(dest), allowed_mask=str(allowed),
                          protected_mask=str(protected), job=str(work / 'lighting-tasks' / sku / (face + '.json')))
            for key in ('contains_sticker', 'boundary_checks', 'perimeter_review', 'local_review'):
                if key in entry:
                    result[key] = entry[key]
            plan[sku][face] = result
    save(work / 'delivery-plan.json', plan)
    return plan, None, None


def run(work):
    work = Path(work).expanduser().resolve()
    with locked(work):
        stage = 'geometry'
        try:
            config = w.read(work / 'workflow.json')
            manifest = w.read(work / 'intake.json')
            contract = manifest['workflow_contract']
            if any(config.get(k) != v for k, v in contract.items()):
                raise ValueError('批次输入类型、材质、对齐基准、模型或输出目标已改变，请新建批次')
            w.check_sources(manifest)
            if config.get('model_reference'):
                ref = config['model_reference']
                if w.sha(ref['path']) != ref['sha256']:
                    raise ValueError('固定模型参考已改变，请重新建批次')
            output = Path(config['output'])
            if output.exists():
                # Never let a stale "completed" flag hide a modified delivery.
                validation = w.read(work / 'validation.json')
                actual = pipeline.batch_files(output)
                expected = validation['skus']
                if set(actual) != set(expected) or any(w.sha(p) != expected[s][f]['output_sha256']
                        for s, fs in actual.items() for f, p in fs.items()):
                    raise ValueError('已发布文件发生变化')
                _, _, light, _ = w.validate_lighting(work, work / 'delivery-plan.json')
                if contract['delivery_goal'] == 'model_textures':
                    spec_path = resolve(work, config['structure']['spec'])
                    check_basis(w.read(spec_path), contract['alignment_basis'])
                    if w.read(spec_path) != w.read(work / 'structure-records/spec.json'):
                        raise ValueError('已发布批次的结构标定发生变化')
                    structure_review.verify_delivery(light, work / 'structure-records',
                                                     resolve(work, config['structure']['review']))
                if contract['model_reference']:
                    from .model_reference import verify_proof
                    verify_proof(actual, contract['model_reference'], validation.get('model_reference'))
                return record(work, 'completed', 'publish', '六面成品已发布', output=str(output))
            geometry, process = work / 'geometry', work / 'geometry-records'
            report_path = process / 'report.json'
            guides = resolve(work, config['geometry'].get('guides'))
            geometry_task = dict(input_kind=contract['input_kind'], device=config['geometry'].get('device', 'auto'),
                                 guides=w.read(guides) if guides else None)
            task_signature = signature(dict(task=geometry_task, dependencies=lighting.dependencies(geometry_task)))
            task_record = work / 'geometry-task.json'
            if task_record.exists() and w.read(task_record)['signature'] != task_signature:
                raise ValueError('几何参数或标定发生变化，请使用新批次重算')
            if not report_path.exists():
                save(task_record, dict(signature=task_signature))
                record(work, 'running', stage, '正在处理六面几何')
                mode = {'raw': 'standardize', 'cutout': 'deskew', 'existing': 'existing'}[config['input_kind']]
                pipeline.process_batch(work / 'input', geometry, process, mode=mode,
                    device=config['geometry'].get('device', 'auto'),
                    frame_guides=guides)
            report = w.read(report_path)
            if not report.get('candidates_complete'):
                raise ValueError('几何处理未完成，请检查几何报告并在新批次重试')
            stage = 'geometry_review'
            if not geometry.exists():
                review = resolve(work, config['geometry'].get('review'))
                if not review or not review.is_file():
                    return waiting(work, stage, '请复核六面轮廓、四角和支架区域后继续', process / 'review-template.json')
                from .geometry_review import approve
                approve(process, review, geometry)
            if not (work / 'geometry.json').exists():
                w.checkpoint(work, geometry, report_path)
            baseline = w.read(work / 'geometry.json')
            records = pipeline.batch_files(geometry)
            verify_report(report_path, records)
            if w.sha(report_path) != w.read(work / 'geometry-proof.json')['sha256']:
                raise ValueError('几何复核报告在冻结后发生变化')
            if set(records) != set(baseline) or any(w.sha(p) != baseline[s][f]['sha256']
                    for s, fs in records.items() for f, p in fs.items()):
                raise ValueError('几何基线在冻结后发生变化')
            stage = 'lighting_scope'
            if not (work / 'lighting-scope.json').exists():
                scope_path = resolve(work, config['lighting'].get('scope'))
                if not scope_path or not scope_path.is_file():
                    return waiting(work, stage, '请登记六面材料范围、真实贴纸和局部效果检查', work / 'lighting-scope-input.json')
                w.freeze_scope(work, scope_path)
            scope = w.load_scope(work)
            stage = 'lighting'
            record(work, 'running', stage, '正在执行已登记的材质光照任务')
            plan, message, required = light_candidates(work, config, baseline, scope)
            if plan is None:
                return waiting(work, stage, message, required)
            for sku, faces in plan.items():
                for face, entry in faces.items():
                    intended = scope[sku][face]
                    key = ('perimeter_review' if intended['mode'] == 'perimeter' else
                           'local_review' if intended.get('local_checks') else None)
                    if key and (not entry.get(key) or not Path(entry[key]).is_file()):
                        return waiting(work, stage, f'请复核 {sku}/{face} 的光照效果并补充记录',
                                       work / 'lighting-records' / sku / face)
            # Run the existing full lighting gate before structural remapping.
            # validate_lighting never creates a delivery folder.
            w.validate_lighting(work, work / 'delivery-plan.json')
            structure_process, structure_review_path = None, None
            if contract['delivery_goal'] == 'model_textures':
                stage = 'structure'
                spec_path = resolve(work, config['structure'].get('spec'))
                if not spec_path or not spec_path.is_file():
                    return waiting(work, stage, '请标定本机背面天线四边界及独立按钮范围', work / 'structure-spec.json')
                spec = w.read(spec_path)
                check_basis(spec, contract['alignment_basis'])
                structure_process = work / 'structure-records'
                if not structure_process.exists():
                    record(work, 'running', stage, '正在按本批基准对齐左右侧天线')
                    structure_review.prepare(work / 'lighting', spec_path, structure_process)
                elif w.read(structure_process / 'spec.json') != spec:
                    raise ValueError('结构标定在处理后发生变化，请使用新批次重算')
                stage = 'structure_review'
                structure_review_path = resolve(work, config['structure'].get('review'))
                if not structure_review_path or not structure_review_path.is_file():
                    return waiting(work, stage, '请复核左右上下天线、两端与完整按钮', structure_process / 'review-template.json')
                structure_review.verify_delivery(pipeline.batch_files(work / 'lighting'),
                                                 structure_process, structure_review_path)
            stage = 'model_reference'
            model_proof = None
            if contract['model_reference']:
                from .model_reference import verify
                review = resolve(work, config.get('model_review'))
                if not review or not review.is_file():
                    from .model_reference import template
                    final = structure_process / 'candidate' if structure_process else work / 'lighting'
                    template_path = work / 'model-review-template.json'
                    if not template_path.exists():
                        template(pipeline.batch_files(final), contract['model_reference']['path'], template_path)
                    return waiting(work, stage, '请核对六面图与固定模型的轮廓、镜头、按钮及接口坐标', template_path)
                final = structure_process / 'candidate' if structure_process else work / 'lighting'
                model_proof = verify(pipeline.batch_files(final), contract['model_reference']['path'], review)
            stage = 'publish'
            w.publish(work, work / 'delivery-plan.json', output, structure_process, structure_review_path,
                      model_proof=model_proof)
            return record(work, 'completed', stage, '六面成品已通过本批发布检查', output=str(output))
        except Exception as exc:
            record(work, 'failed', stage, str(exc))
            raise
