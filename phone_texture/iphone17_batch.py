"""Batch adapter for the standard glass-back profile and workshop review exports."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil
import multiprocessing
import time

from . import iphone17, pipeline, workspace as w


def process_case(args):
    source, work, output, name = args
    case = work/'cases'/name
    case.mkdir(parents=True)
    started = time.monotonic()
    try:
        report = iphone17.run(source, case, output/name, single_name=name, cpu_threads=1)
        complete = report.get('processed') == 1
        if complete:
            face_stages = {f: 'antenna_and_buttons' if f in ('left','right') else
                           'bottom_registration' if f == 'bottom' else 'lighting' for f in pipeline.FACES}
        else:
            strict_error = report.get('error')
            from .iphone17_review import continue_review
            report = continue_review(case, name)
            face_stages = report['face_processing']
            if strict_error and not any(b['error']==strict_error for b in report['blocked_checks']):
                report['blocked_checks'].append(dict(stage='standard_pipeline',error=strict_error))
        candidate = Path(report['candidate'])
        if set(report['final_sha256']) != set(pipeline.FACES):
            raise ValueError('标准版六面校验记录不完整')
        pending = output/('.'+name+'.pending')
        pending.mkdir()
        for face in pipeline.FACES:
            target = pending/(face+'.png')
            shutil.copy2(candidate/target.name, target)
            if w.sha(target) != report['final_sha256'][face]:
                raise ValueError('标准版候选在导出时发生变化')
        pending.rename(output/name)
        lit = {f for f, stage in face_stages.items() if stage in
               ('lighting','antenna','antenna_and_buttons','bottom_registration')}
        geom = all(stage not in ('source_matte','frame_without_paper_check') for stage in face_stages.values())
        row = dict(sku=name, model='iphone17', seconds=time.monotonic()-started,
                   status='processed_needs_visual_review' if complete else 'source_preserved_needs_review',
                   six_faces_processed=complete, six_faces_generated=True,
                   final_sha256=report['final_sha256'], structural_alignment_applied=complete,
                   seam_accepted=complete, stage_status=dict(geometry='complete' if geom else 'partial',
                       lighting='complete' if len(lit)==6 else 'partial' if lit else 'blocked',
                       structure='complete' if complete else 'blocked', seam='complete' if complete else 'blocked'),
                   blocked_checks=report.get('blocked_checks', []), face_processing=face_stages,
                   per_face_stages={f: dict(selected_stage=stage, lighting='complete' if f in lit else 'blocked')
                                    for f, stage in face_stages.items()})
    except Exception as exc:
        row = dict(sku=name, model='iphone17', status='failed', error=str(exc),
                   six_faces_processed=False, six_faces_generated=False)
    w.save(case/'result.json', row)
    return row


def run(source, work, output, names, jobs, single=False):
    source, work, output = pipeline.separate_paths(source, work, output)
    if not 1 <= jobs <= 8 or not names or len(set(names)) != len(names):
        raise ValueError('无效的标准版批次参数')
    if any(not n or n.startswith('.') or Path(n).name != n for n in names):
        raise ValueError('无效的 SKU 目录名')
    if single and names != [source.name]:
        raise ValueError('单台原图目录与批次名称不符')
    pipeline.fresh(work)
    pipeline.fresh(output)
    tasks = [(source if single else source/n, work, output, n) for n in names]
    for folder, *_ in tasks:
        w.face_files(folder)
    work.mkdir(parents=True)
    output.mkdir(parents=True)
    from .automatic import initialize_worker
    context = multiprocessing.get_context('spawn')
    slots = context.Semaphore(min(2,jobs,len(tasks)))
    with ProcessPoolExecutor(max_workers=min(jobs,len(tasks)), mp_context=context,
                             initializer=initialize_worker, initargs=(slots,)) as pool:
        rows = list(pool.map(process_case,tasks))
    exported = [r['sku'] for r in rows if r['six_faces_generated']]
    processed = sum(r['six_faces_processed'] for r in rows)
    if processed == len(rows):
        status = 'processed_needs_visual_review'
    elif len(exported) == len(rows):
        status = 'review_candidates_needs_visual_review'
    else:
        status = 'partial_candidates_needs_visual_review' if exported else 'failed'
    report = dict(model='iphone17', status=status, total=len(rows), processed=processed,
                  generated=len(exported), attempts=rows, exported_skus=exported)
    if status.startswith(('partial','review')):
        w.save(work/'partial-candidate-manifest.json', dict(generated=len(exported), items=rows))
    w.save(work/'report.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--work', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--skus', dest='names', required=True, nargs='+')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--single', action='store_true')
    raise SystemExit(0 if run(**vars(parser.parse_args()))['generated'] else 2)
