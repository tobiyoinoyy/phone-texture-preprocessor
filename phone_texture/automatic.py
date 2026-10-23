"""Original-only aluminum studio-photo processing, with auditable proposals.

All six computational stages finish before a candidate directory is exposed.
Automatic measurements are never written as human/agent visual approvals.
The established final visual/model acceptance gates remain separate.
"""
from pathlib import Path
import contextlib
import concurrent.futures
import datetime
import json
import multiprocessing
import os
import resource
import shutil
import time
import traceback

import numpy as np
from PIL import Image,ImageOps
from . import pipeline,workspace as w,geometry_review as gr,structure_review as sr,lighting
from .auto_geometry import frame_geometry,plate_geometry,trim_visible_canvas
from .auto_material import paper_mask,propose_job,save_mask,lighting_scope
from .auto_structure import propose

PROFILE='iphone17-aluminum-studio-v9-independent-landmarks-lower-plane'
_INFERENCE_SLOTS=None


def initialize_worker(inference_slots):
    global _INFERENCE_SLOTS
    _INFERENCE_SLOTS=inference_slots


def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    w.save(temporary,value);temporary.replace(path)


def measure_masks(source,output,allowed,paper):
    a=np.array(Image.open(source).convert('RGBA'));b=np.array(Image.open(output).convert('RGBA'))
    if a.shape!=b.shape or not np.array_equal(a[...,3],b[...,3]):raise ValueError('Automatic lighting changed geometry')
    changed=np.any(a!=b,axis=2)
    if np.any(changed&paper) or np.any(changed&~allowed):raise ValueError('Automatic lighting crossed its source material/paper boundary')
    return dict(alpha_changed=0,paper_changed=0,outside_material_changed=0,changed_pixels=int(changed.sum()),
        source_sha256=w.sha(source),output_sha256=w.sha(output))


def process_one(task):
    """One fresh model per SKU; no historical masks/landmarks/reviews as input."""
    sku,source,case,device,cpu_threads,model=task;source=Path(source);case=Path(case)
    case.mkdir(parents=True,exist_ok=False);started=time.perf_counter();events=[]
    result=dict(sku=sku,status='running',profile=PROFILE,source=str(source),process=str(case),
        historical_annotations_reused=False,runtime_agent_required=False,manual_annotations_required=False,visual_review_passed=False,model_uv_verified=False)
    def progress(step, face=None, completed=0):
        write(case/'live-progress.json',dict(stage=step,face=face,completed_units=completed,
            total_units=15,updated_at=time.time(),seconds=time.perf_counter()-started))
    progress('loading')
    def stage(name,begin):events.append(dict(stage=name,seconds=time.perf_counter()-begin))
    try:
        import torch
        import cv2
        torch.set_num_threads(cpu_threads)
        cv2.setNumThreads(1)
        from phone_normalizer.api import Normalizer,NormalizerConfig
        manifest=w.intake(source,case/'work',single_name=sku)
        inputs=manifest['skus'][sku];records=case/'work/geometry-records';records.mkdir()
        folder=records/'staged'/sku;folder.mkdir(parents=True)
        geometry=dict(mode='original_only_automatic',success=False,geometry_status='needs_review',
            candidates_complete=False,images=[],automatic_profile=PROFILE)
        device_observations=[];forward_observations=[];begin=time.perf_counter()
        with Normalizer(NormalizerConfig(remove_support=False,deskew=False,tight_crop=False,device=device)) as norm:
            original_forward=norm.backend._run_model
            def observe_forward(model,inputs,torch_module):
                value=original_forward(model,inputs,torch_module)
                if norm.backend.device=='mps':
                    torch_module.mps.synchronize()
                    forward_observations.append(dict(face=face,measurement='post_forward_before_cleanup_not_true_peak',
                        mps_driver_bytes=int(torch_module.mps.driver_allocated_memory()),
                        mps_tensor_bytes=int(torch_module.mps.current_allocated_memory())))
                return value
            norm.backend._run_model=observe_forward
            for face in pipeline.FACES:
                progress('geometry',face,len(geometry['images']))
                item=inputs[face];raw=ImageOps.exif_transpose(Image.open(item['snapshot'])).convert('RGB')
                # Narrow front rims need more model pixels than the ordinary
                # 1024 cap. Every SKU uses the same 1536 front setting; it is
                # not a stored per-case fix. A shared batch semaphore bounds
                # simultaneous model calls, including allocator cleanup.
                norm.backend.max_inference_side=1536 if face=='front' else 1024
                with (_INFERENCE_SLOTS if _INFERENCE_SLOTS is not None else contextlib.nullcontext()):
                    matte,info=norm.normalize(raw)
                stages={}
                def capture(name,image):stages[name]=image.copy()
                try:
                    if face in pipeline.FRAMES:
                        image,evidence=frame_geometry(matte,face,capture,source_image=raw)
                    else:
                        paper,paper_info=paper_mask(matte,face)
                        image,evidence=plate_geometry(matte,face,paper,capture,source_image=raw)
                        evidence['source_paper_proposal']=paper_info
                except Exception:
                    failed=records/'failed-source-analysis';failed.mkdir(exist_ok=True)
                    matte.save(failed/(face+'-matte.png'))
                    write(failed/(face+'.json'),dict(source=item['source'],source_sha256=item['sha256']))
                    raise
                tight_paper,_=paper_mask(image,face)
                image,tight_proof=trim_visible_canvas(image,tight_paper)
                evidence['visible_canvas_trim']=tight_proof
                evidence['output_size']=list(image.size)
                p=folder/(face+'.png');image.save(p)
                artifacts=gr.make_evidence(item['snapshot'],image,stages,records/'geometry-review'/sku/face)
                geometry['images'].append(dict(sku=sku,face=face,source=item['snapshot'],
                    source_sha256=item['sha256'],output_sha256=w.sha(p),geometry=evidence,
                    review_reasons=evidence['review_reasons'],geometry_evidence=artifacts))
                backend=norm.backend;obs=dict(face=face,device=backend.device,fallback_count=backend.fallback_count,
                    model_input_size=list(info.model_input_size) if info.model_input_size else None)
                if backend.device=='mps':
                    obs.update(mps_driver_bytes=int(torch.mps.driver_allocated_memory()),
                        mps_tensor_bytes=int(torch.mps.current_allocated_memory()))
                device_observations.append(obs)
        geometry['candidates_complete']=True;write(records/'report.json',geometry)
        write(records/'review-template.json',dict(report_sha256=w.sha(records/'report.json'),reviewer='',
            items={sku+'/'+face:{**dict.fromkeys(gr.CHECKS,False),'notes':'','boundary_assertions':{}} for face in pipeline.FACES}))
        stage('segmentation_geometry_and_evidence',begin)
        result['device_observations']=device_observations
        result['post_forward_memory_samples']=forward_observations
        geometry_files={f:folder/(f+'.png') for f in pipeline.FACES}
        # Freeze the actual landmarks before illumination can alter gradients.
        structure_proposal=propose(geometry_files,coordinate_mode='canvas_pixel_centers')
        write(case/'work/geometry-structure-proposal.json',structure_proposal)
        light=case/'work/lighting'/sku;checks={};jobs={};analyses={};begin=time.perf_counter()
        for face in pipeline.FACES:
            progress('lighting',face,6+len(checks))
            p=folder/(face+'.png');assets=case/'work/material-inputs'/face
            job,analysis,allowed,paper=propose_job(p,face,assets,companions=geometry_files,landmarks=structure_proposal)
            effect_spec=lighting_scope(p,face,job,assets/'effect-source-witnesses')
            write(assets/'lighting-scope.json',effect_spec)
            job.update(output_dir=str(light),records_dir=str(case/'work/lighting-records'/face))
            jp=assets/'job.json';write(jp,job);write(assets/'analysis.json',analysis)
            lighting.run('aluminum',jp)
            checks[face]=measure_masks(p,light/(face+'.png'),allowed,paper)
            from . import lighting_review as effect_gate
            effect=effect_gate.measure_perimeter(effect_spec,p,light/(face+'.png'),allowed,paper) if face in ('front','back') else effect_gate.measure_local(effect_spec,p,light/(face+'.png'),allowed,paper)
            checks[face]['source_relative_effect_checks']=effect
            effect_gate.evidence_images(effect_spec,p,light/(face+'.png'),assets/'effect-visual-review')
            jobs[face]=str(jp);analyses[face]=analysis
        stage('fresh_material_analysis_lighting_and_checks',begin)
        write(case/'work/lighting-jobs.json',{sku:jobs});write(case/'work/lighting-checks.json',checks)
        begin=time.perf_counter();files={f:light/(f+'.png') for f in pipeline.FACES}
        progress('alignment',completed=12)
        from .auto_structure import bind_lighting_sources
        spec={sku:bind_lighting_sources(structure_proposal,geometry_files,files)}
        spec_path=case/'work/structure-spec.json';write(spec_path,spec)
        structural=sr.prepare(case/'work/lighting',spec_path,case/'work/structure-records')
        stage('fresh_landmarks_structure_and_checks',begin)
        begin=time.perf_counter()
        progress('checking',completed=13)
        from . import frame_seams
        seam_output=case/'work/seam-candidates'/sku
        seam_report=frame_seams.run(case/'work',sku,model,seam_output,case/'work/seam-records')
        candidate=case/'candidate';pipeline.publish(case/'work/seam-candidates',candidate)
        # Computational completeness is explicit, distinct from visual approval.
        final=pipeline.batch_files(candidate);assert set(final)=={sku}
        canvas_checks={}
        for face in pipeline.FACES:
            with Image.open(final[sku][face]) as im:
                if im.mode!='RGBA' or im.format!='PNG':raise ValueError('Final face is not RGBA PNG')
                paper=np.array(Image.open(case/'work/material-inputs'/face/'paper.png'))>0
                _,trim=trim_visible_canvas(im,paper)
                if any(trim['removed_margins_ltrb']):raise ValueError('Final canvas contains excess soft-matte padding: '+face)
                canvas_checks[face]=trim
        w.check_sources(manifest)
        pipeline.overview({sku:{f:folder/(f+'.png') for f in pipeline.FACES}},final,case/'overview.png',faces=pipeline.FACES)
        stage('candidate_output_and_final_integrity',begin)
        result.update(status='processed_needs_visual_review',six_faces_processed=True,
            candidate=str(candidate),lighting_checks=checks,structure_checks=structural['checks'],
            frame_uv_seam_checks=seam_report,final_canvas_checks=canvas_checks,
            final_sha256={f:w.sha(p) for f,p in final[sku].items()},source_hashes_unchanged=True)
    except Exception as exc:
        result.update(status='failed',six_faces_processed=False,error=str(exc),traceback=traceback.format_exc())
    result.update(seconds=time.perf_counter()-started,events=events,
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(case/'result.json',result)
    progress('complete' if result.get('six_faces_processed') else 'failed',completed=15 if result.get('six_faces_processed') else 0)
    return result


def worker(task):
    case=Path(task[2]);case.parent.mkdir(parents=True,exist_ok=True)
    with (case.parent/(case.name+'.log')).open('w') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        return process_one(task)


def concurrency_limits(jobs, inference_slots=None, experimental=False):
    limit = 10 if experimental else 4
    if type(jobs) is not int or not 1 <= jobs <= limit:
        raise ValueError('自动入口支持 1–4 个执行进程；显式实验模式最多 10 个，不启动 30 份模型')
    slots = min(jobs, 2) if inference_slots is None else inference_slots
    if type(slots) is not int or not 1 <= slots <= min(jobs, 4):
        raise ValueError('推理并发必须在 1 至 min(jobs, 4) 之间')
    if slots > 2 and not experimental:
        raise ValueError('超过两份同时推理必须显式开启实验模式')
    return slots


def run(source,work=None,output=None,*,jobs=2,device='auto',skus=None,single_name='SKU_01',cpu_threads=2,
        inference_slots=None,experimental_concurrency=False,model=None):
    from .frame_seams import profile
    concurrency_limits(jobs,inference_slots,experimental_concurrency)
    model_data,_=profile(model)
    if work is None or output is None:
        import uuid
        stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6]
        root=Path.cwd()/'outputs'/('phone-auto-'+stamp)
        work=Path(work) if work is not None else root/'process'
        output=Path(output) if output is not None else root/'six-face-candidates'
    source,work,output=pipeline.separate_paths(source,work,output)
    pipeline.fresh(work);pipeline.fresh(output)
    slots=concurrency_limits(jobs,inference_slots,experimental_concurrency)
    if not isinstance(cpu_threads,int) or not 1<=cpu_threads<=8:raise ValueError('cpu_threads must be 1..8')
    is_single=any(p.is_file() and p.stem.lower() in pipeline.FACES for p in source.iterdir())
    names=[single_name] if is_single else (skus or sorted(p.name for p in source.iterdir() if p.is_dir() and not p.name.startswith('.')))
    if not names or len(names)!=len(set(names)):raise ValueError('Empty or duplicate SKU selection')
    # Validate the entire queue before any expensive inference.
    for name in names:
        if Path(name).name!=name or name in ('.','..'):raise ValueError('Unsafe SKU name')
        w.face_files(source if is_single else source/name)
    work.mkdir(parents=True);begin=time.perf_counter()
    protocol=dict(profile=PROFILE,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        input=str(source),output=str(output),jobs=jobs,cpu_threads_per_worker=cpu_threads,
        skus=names,historical_masks_guides_landmarks_or_reviews_reused=False,runtime_agent_required=False,manual_annotations_required=False,
        segmentation_model='ZhengPeng7/BiRefNet_dynamic',alignment_basis='back_image',
        segmentation_max_side_by_face={f:1536 if f=='front' else 1024 for f in pipeline.FACES},
        maximum_simultaneous_segmentation_calls=slots,experimental_concurrency=experimental_concurrency,
        final_visual_review_included_in_processing_time=False,model_uv_verified=False,
        frame_seam_model=model,frame_seam_model_sha256=model_data['model_sha256'])
    repo=Path(__file__).resolve().parents[1]
    protocol['implementation_sha256']={str(p.relative_to(repo)):w.sha(p) for p in
        sorted([*repo.glob('phone_texture/auto*.py'),repo/'phone_texture/top_frame.py',repo/'skills/phone-aluminum-frame-restore/scripts/harmonize_aluminum.py',repo/'skills/phone-aluminum-frame-restore/scripts/match_uv_seams.py',repo/'phone_texture/frame_seams.py',*repo.glob('phone_texture/seam_profiles/*.json')])}
    write(work/'protocol.json',protocol)
    tasks=[(n,str(source if is_single else source/n),str(work/'cases'/n),device,cpu_threads,model) for n in names]
    result=[]
    context=multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs,mp_context=context,
            initializer=initialize_worker,initargs=(context.Semaphore(slots),)) as pool:
        futures={pool.submit(worker,t):t[0] for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            try:row=future.result()
            except Exception as exc:row=dict(sku=futures[future],status='failed',six_faces_processed=False,error=str(exc))
            result.append(row);write(work/'progress.json',dict(finished=len(result),total=len(tasks),attempts=result))
            print(json.dumps({k:row.get(k) for k in ('sku','status','seconds','error')},ensure_ascii=False),flush=True)
    result.sort(key=lambda r:r['sku']);completed=sum(r.get('six_faces_processed',False) for r in result)
    report=dict(protocol=protocol,attempts=result,processed=completed,total=len(tasks),
        visual_approved=0,seconds=time.perf_counter()-begin,status='failed' if completed!=len(tasks) else 'processed_needs_visual_review')
    # No partial top-level six-face delivery is presented as a complete batch.
    if completed==len(tasks):
        staged=work/'complete-candidates';staged.mkdir()
        for row in result:shutil.copytree(Path(row['candidate'])/row['sku'],staged/row['sku'])
        pipeline.publish(staged,output)
        report['output']=str(output)
    write(work/'report.json',report)
    return report
