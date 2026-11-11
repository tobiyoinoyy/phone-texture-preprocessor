# -*- coding: utf-8 -*-
"""Source-bound studio processing with independent, auditable stage results.

Only complete six-face files are exposed; processing completeness is separate.
Automatic measurements are never written as human/agent visual approvals.
Explicit geometry resume verifies the same SKU's sources and output hashes.
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
import sys
import time
import traceback

import numpy as np
from PIL import Image,ImageOps
from . import pipeline,workspace as w,geometry_review as gr,structure_review as sr,lighting
from .auto_geometry import frame_geometry,plate_geometry,trim_visible_canvas
from .auto_material import paper_mask,propose_job,save_mask,lighting_scope
from .auto_structure import propose

PROFILE='iphone17-aluminum-studio-v10-source-material-boundaries'
_INFERENCE_SLOTS=None


def initialize_worker(inference_slots):
    global _INFERENCE_SLOTS
    _INFERENCE_SLOTS=inference_slots


def peak_process_memory():
    """Normalize the worker's lifetime high-water RSS, not a per-SKU/tree peak."""
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # getrusage(2): Linux reports KiB; Darwin reports bytes. Do not label
    # an unverified platform's native value as bytes.
    unit, scale = {'linux': ('KiB', 1024), 'darwin': ('bytes', 1)}.get(sys.platform, (None, None))
    return dict(peak_process_rss_bytes=raw * scale if scale is not None else None,
        peak_process_rss_measurement=dict(source='getrusage(RUSAGE_SELF).ru_maxrss',
            raw_value=raw, raw_unit=unit, platform=sys.platform, scope='worker_process_lifetime'))


def write(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    w.save(temporary,value);temporary.replace(path)


def implementation_manifest(repo):
    """Fingerprint the shipped runtime, including delegated material helpers.

    Installed packages need no Git checkout. Newly added Python modules and
    frozen recognition/seam resources cannot silently fall outside provenance.
    """
    repo=Path(repo);paths=set()
    for package in ('phone_texture','phone_normalizer','texture_alignment'):
        paths.update(p for p in (repo/package).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.npz'))
    paths.update((repo/'scripts').glob('*.py'))
    for skill in ('phone-aluminum-frame-restore','phone-six-view-restore'):
        paths.update((repo/'skills'/skill/'scripts').glob('*.py'))
    return {str(p.relative_to(repo)):w.sha(p) for p in sorted(paths)}


def measure_masks(source,output,allowed,paper):
    a=np.array(Image.open(source).convert('RGBA'));b=np.array(Image.open(output).convert('RGBA'))
    if a.shape!=b.shape or not np.array_equal(a[...,3],b[...,3]):raise ValueError('Automatic lighting changed geometry')
    changed=np.any(a!=b,axis=2)
    if np.any(changed&paper) or np.any(changed&~allowed):raise ValueError('Automatic lighting crossed its source material/paper boundary')
    return dict(alpha_changed=0,paper_changed=0,outside_material_changed=0,changed_pixels=int(changed.sum()),
        source_sha256=w.sha(source),output_sha256=w.sha(output))


def no_zone_effects(source,output,analysis):
    """Report actual deterministic changes; protection removal is not identity."""
    a=np.array(Image.open(source).convert('RGBA'));b=np.array(Image.open(output).convert('RGBA'))
    label_info=analysis.get('source_label_fit_exclusion',{})
    mask=(np.array(Image.open(label_info['path']).convert('L'))>0) if label_info.get('path') else np.zeros(a.shape[:2],bool)
    delta=abs(a[...,:3].astype(int)-b[...,:3].astype(int));changed=np.any(delta>0,axis=2)
    report=dict(protection_policy='none',protected_zone_pixels=0,source_label_fit_pixels=int(mask.sum()),
        label_pixels_changed=int((changed&mask).sum()),label_maximum_RGB_change=int(delta[mask].max()) if mask.any() else 0,
        total_RGB_pixels_changed=int(changed.sum()),alpha_pixels_changed=int(np.sum(a[...,3]!=b[...,3])))
    if mask.any():
        import cv2,hashlib
        y,x=np.nonzero(mask);box=[max(0,x.min()-5),max(0,y.min()-5),min(a.shape[1],x.max()+6),min(a.shape[0],y.max()+6)];x0,y0,x1,y1=box
        def payloads(image):
            found=set();observations=[]
            for pad in [4,5,6,8]:
                x0,y0,x1,y1=max(0,int(x.min())-pad),max(0,int(y.min())-pad),min(a.shape[1],int(x.max())+pad+1),min(a.shape[0],int(y.max())+pad+1)
                rgb=image[y0:y1,x0:x1,:3];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
                for name,value in [('RGB',rgb),('gray',gray),('CLAHE',cv2.createCLAHE(2.,(8,8)).apply(gray)),('Otsu',cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])]:
                    text,_,_=cv2.QRCodeDetector().detectAndDecode(value)
                    if text:found.add(text)
                    observations.append(dict(padding=pad,contrast=name,decoded=bool(text)))
            return found,observations
        before,before_obs=payloads(a);after,after_obs=payloads(b);same=bool(before&after) if before else None
        report.update(source_QR_decoded=bool(before),output_QR_decoded=bool(after),QR_payload_unchanged=same,
            source_QR_payload_sha256=sorted(hashlib.sha256(value.encode()).hexdigest() for value in before),
            QR_decode_observations=dict(source=before_obs,output=after_obs),review_warnings=[])
        if before and after and not same:raise ValueError('Programmatic illumination decoded a different source QR payload')
        if before and not after:report['review_warnings'].append('QR readability could not be confirmed; visual review required')
    return report


def process_one(task):
    """Fresh processing, or explicitly bound saved-review continuation."""
    protection_policy=task[6] if len(task)>=7 else 'sticker_only'
    review_fallback=task[7] if len(task)>=8 else 'fail'
    lighting_profile=task[8] if len(task)>=9 else 'baseline'
    resume_case=task[9] if len(task)>=10 else None
    bottom_alignment=task[10] if len(task)>=11 else 'appearance'
    source_body_reference_review=task[11] if len(task)>=12 else False
    saved_review_descriptor=task[12] if len(task)>=13 else None
    require_geometry_resume=task[13] if len(task)>=14 else False
    side_candidate_review=task[14] if len(task)>=15 else False
    if type(side_candidate_review) is not bool:raise ValueError('side_candidate_review must be an explicit bool')
    if type(require_geometry_resume) is not bool:raise ValueError('require_geometry_resume must be an explicit bool')
    if type(source_body_reference_review) is not bool:raise ValueError('source_body_reference_review must be an explicit bool')
    if bottom_alignment not in ('appearance','off'):raise ValueError('Unknown bottom alignment mode')
    from .shadow_lift_profile import PROFILES
    if lighting_profile not in PROFILES:raise ValueError('Unknown lighting profile')
    sku,source,case,device,cpu_threads,model=task[:6];source=Path(source);case=Path(case)
    if protection_policy not in ('sticker_only','none'):raise ValueError('Unknown automatic protection policy')
    if side_candidate_review and (saved_review_descriptor is not None or not source_body_reference_review
            or protection_policy!='none' or lighting_profile!='accepted-shadow-lift-v2'):
        raise ValueError('Fresh side review requires current-source normal processing with none/accepted-shadow-lift-v2')
    if saved_review_descriptor is not None:
        # This branch precedes normal intake/segmentation/material processing.
        # A matched but invalid record fails; it never silently becomes a new run.
        from .saved_review import process_saved
        if case.exists():raise ValueError('Saved review case already exists')
        started=time.perf_counter()
        options=dict(model=model,protection_policy=protection_policy,review_fallback=review_fallback,
            lighting_profile=lighting_profile,bottom_alignment=bottom_alignment,
            source_body_reference_review=source_body_reference_review)
        try:
            import cv2
            cv2.setNumThreads(0)
            if cv2.getNumThreads()!=1:
                raise RuntimeError('Cannot verify sequential OpenCV execution')
            result=process_saved(saved_review_descriptor,sku=sku,source=source,case=case,options=options)
            result['opencv_runtime']=dict(requested_native_threads=0,effective_threads=1,native_pool='disabled')
            route='saved_review'
        except Exception as exc:
            case.mkdir(parents=True,exist_ok=True)
            result=dict(sku=sku,status='failed',source=str(source),process=str(case),
                six_faces_generated=False,six_faces_processed=False,quality_checks_complete=False,
                visual_review_passed=False,model_uv_verified=False,saved_review_reused=False,
                saved_review_requested=True,error=str(exc),traceback=traceback.format_exc())
            route='saved_review_rejected'
        result.update(seconds=time.perf_counter()-started,worker_pid=os.getpid(),
            worker_parent_pid=os.getppid(),worker_process_group=os.getpgrp(),
            worker_nice=os.getpriority(os.PRIO_PROCESS,0),
            worker_thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')},
            execution_route=route,**peak_process_memory())
        write(case/'result.json',result)
        write(case/'live-progress.json',dict(stage='complete' if result.get('six_faces_generated') else 'failed',
            completed_units=15 if result.get('six_faces_generated') else 0,total_units=15,
            updated_at=time.time(),seconds=result['seconds']))
        return result
    case.mkdir(parents=True,exist_ok=False);started=time.perf_counter();events=[]
    result=dict(sku=sku,status='running',profile=PROFILE,source=str(source),process=str(case),
        historical_annotations_reused=False,runtime_agent_required=False,manual_annotations_required=False,visual_review_passed=False,model_uv_verified=False,protection_policy=protection_policy,lighting_profile=lighting_profile,source_body_reference_review_enabled=source_body_reference_review,
        worker_pid=os.getpid(),worker_parent_pid=os.getppid(),worker_process_group=os.getpgrp(),
        worker_nice=os.getpriority(os.PRIO_PROCESS,0),
        worker_thread_environment={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')},
        execution_route='normal_pipeline')
    result['side_candidate_review_enabled']=side_candidate_review
    def progress(step, face=None, completed=0):
        write(case/'live-progress.json',dict(stage=step,face=face,completed_units=completed,
            total_units=15,updated_at=time.time(),seconds=time.perf_counter()-started))
    progress('loading')
    def stage(name,begin):events.append(dict(stage=name,seconds=time.perf_counter()-begin))
    geometry=None;geometry_files=None;checks={}
    try:
        import cv2
        # GCD (the installed Mac backend) ignores positive thread counts.
        # Disabling the native pool gives one sequential execution thread.
        cv2.setNumThreads(0)
        if cv2.getNumThreads()!=1:
            raise RuntimeError('Cannot verify sequential OpenCV execution')
        result['opencv_runtime']=dict(requested_native_threads=0,effective_threads=cv2.getNumThreads(),native_pool='disabled')
        manifest=w.intake(source,case/'work',single_name=sku)
        inputs=manifest['skus'][sku]
        from .stage_processing import resume_geometry,process_stages
        resumed=resume_geometry(resume_case,case,sku,manifest) if resume_case else None
        if require_geometry_resume and resumed is None:
            raise ValueError('Required verified geometry checkpoint unavailable; fresh segmentation was not started')
        if resumed is not None:
            geometry,geometry_files=resumed
            result['geometry_resume']=geometry['resume_provenance']
            events.append(dict(stage='verified_geometry_resume',seconds=0))
        else:
            import torch
            torch.set_num_threads(cpu_threads)
            from phone_normalizer.api import Normalizer,NormalizerConfig
            records=case/'work/geometry-records';records.mkdir()
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
                    pre_trim_image=None;analysis_stage='geometry'
                    try:
                        if face in pipeline.FRAMES:
                            image,evidence=frame_geometry(matte,face,capture,source_image=raw)
                        else:
                            if protection_policy=='none':
                                paper,paper_info=paper_mask(matte,face,resolve_end=False)
                                image,evidence=plate_geometry(matte,face,paper,capture,source_image=raw,reconstruct_paper_tail=False)
                            else:
                                paper,paper_info=paper_mask(matte,face)
                                image,evidence=plate_geometry(matte,face,paper,capture,source_image=raw)
                            evidence['source_paper_proposal']=paper_info
                        pre_trim_image=image
                        analysis_stage='paper_before_visible_canvas_trim'
                        tight_paper,_=paper_mask(image,face,resolve_end=False) if protection_policy=='none' else paper_mask(image,face)
                        analysis_stage='visible_canvas_trim'
                        image,tight_proof=trim_visible_canvas(image,None if protection_policy=='none' else tight_paper)
                    except Exception:
                        failed=records/'failed-source-analysis';failed.mkdir(exist_ok=True)
                        matte.save(failed/(face+'-matte.png'))
                        failure=dict(source=item['source'],source_sha256=item['sha256'],analysis_stage=analysis_stage)
                        if pre_trim_image is not None:
                            failed_image=failed/(face+'-geometry-before-final-trim.png')
                            pre_trim_image.save(failed_image)
                            failure.update(geometry_source_image=str(failed_image),
                                geometry_source_sha256=w.sha(failed_image),
                                geometry_source_size=list(pre_trim_image.size),
                                coordinate_space='post_geometry_before_visible_canvas_trim')
                        write(failed/(face+'.json'),failure)
                        raise
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
        # Original/resumed geometry stays immutable. Every downstream mask and
        # source binding is measured afresh against the selected bottom image.
        from .bottom_alignment import prepare as prepare_bottom
        from .face_perspective import prepare as prepare_perspective
        from .stage_processing import geometry_identity
        geometry_identity(geometry,geometry_files,sku)
        # Default planar perspective for front/back (user decision 2026-11-05).
        # Each face is measured on its own; a rejected fit keeps the exact PNG.
        begin=time.perf_counter()
        planar_files,planar_geometry,perspective_report=prepare_perspective(
            geometry_files,geometry,case/'work/face-perspective')
        geometry_identity(planar_geometry,planar_files,sku)
        stage('planar_face_perspective',begin)
        result['face_perspective']=perspective_report
        begin=time.perf_counter()
        effective_files,effective_geometry,bottom_report=prepare_bottom(
            planar_files,planar_geometry,model,case/'work/bottom-alignment',enabled=bottom_alignment=='appearance')
        stage('optional_bottom_alignment',begin)
        result['bottom_alignment']=bottom_report
        bottom_fallback=None
        bottom_quality=None
        if bottom_report['applied']:
            from .bottom_alignment import retain_source_after_downstream_failure
            def bottom_fallback(error):
                return retain_source_after_downstream_failure(effective_files,effective_geometry,
                    planar_files,planar_geometry,bottom_report,case/'work/bottom-alignment',error)
            from .bottom_alignment import original_source_for_quality_comparison,retain_source_after_quality_comparison
            def quality_source():
                return original_source_for_quality_comparison(effective_files,effective_geometry,
                    planar_files,planar_geometry,bottom_report)
            def quality_commit(comparison_path):
                return retain_source_after_quality_comparison(effective_files,effective_geometry,
                    planar_files,planar_geometry,bottom_report,case/'work/bottom-alignment',comparison_path)
            bottom_quality=dict(source=quality_source,commit=quality_commit)
        reference_metadata = (dict(raw_path=inputs['back']['snapshot'],back_path=geometry_files['back'],
            geometry_report_path=case/'work/geometry-records/report.json',
            geometry_item=next(entry for entry in geometry['images'] if entry['face']=='back'),
            perspective_output_sha256=perspective_report['faces']['back']['output_sha256']
                if perspective_report['faces']['back']['applied'] else None)
            if source_body_reference_review else None)
        result.update(process_stages(case,sku,effective_geometry,effective_files,manifest,model,
            protection_policy,lighting_profile,progress,stage,allow_partial=review_fallback=='preserve-source',bottom_fallback=bottom_fallback,
            source_body_reference_review=source_body_reference_review,source_body_reference_metadata=reference_metadata,
            bottom_quality=bottom_quality,side_candidate_review=side_candidate_review))
        if result.get('side_candidate_review_attempted'):
            result['execution_route']='normal_pipeline_with_side_review'
        result.setdefault('review_warnings',[]).extend(
            dict(face='bottom',**notice) for notice in bottom_report['notices'])
        result['review_warnings'].extend(dict(notice) for notice in perspective_report['notices']
            if notice['severity']=='review')
        geometry_identity(geometry,geometry_files,sku)
        write(case/'work/stage-report.json',result)
    except Exception as exc:
        original_trace=traceback.format_exc()
        result.update(status='failed',six_faces_generated=False,six_faces_processed=False,error=str(exc),traceback=original_trace)
    for item in (geometry or {}).get('images',[]):
        if 'source_contact_appearance_unverified' in item.get('review_reasons',[]):
            result.setdefault('review_warnings',[]).append(dict(face=item['face'],
                code='source_contact_appearance_unverified',
                message='支架接触处保留源图混合像素，轮廓与颜色仍需视觉核对；未重建被遮挡内容'))
    result.update(seconds=time.perf_counter()-started,events=events,**peak_process_memory())
    if result.get('side_candidate_review_complete') is True and result.get('six_faces_generated') is True:
        begin=time.perf_counter()
        try:
            from .side_candidate_export import create as create_review_package
            options=dict(model=model,protection_policy=protection_policy,review_fallback=review_fallback,
                lighting_profile=lighting_profile,bottom_alignment=bottom_alignment,
                source_body_reference_review=source_body_reference_review,side_candidate_review=side_candidate_review)
            result['portable_review_package']=create_review_package(result,case,manifest,effective_files,options)
            stage('portable_current_review_packaging',begin)
        except Exception as exc:
            result.update(status='failed',six_faces_generated=False,six_faces_processed=False,
                portable_review_packaging_failed=True,error=str(exc),traceback=traceback.format_exc())
        result.update(seconds=time.perf_counter()-started,events=events,**peak_process_memory())
    write(case/'result.json',result)
    progress('complete' if result.get('six_faces_generated',result.get('six_faces_processed')) else 'failed',completed=15 if result.get('six_faces_generated',result.get('six_faces_processed')) else 0)
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


def export_complete_candidates(attempts,work,output):
    """Export only complete six-face candidates, never an incomplete SKU."""
    selected=[row for row in attempts if row.get('six_faces_generated',row.get('six_faces_processed')) and row.get('status')!='failed']
    if not selected:return []
    staged=Path(work)/'complete-candidates';pipeline.fresh(staged);staged.mkdir()
    for row in selected:
        source=Path(row['candidate'])/row['sku']
        for face in pipeline.FACES:
            path=source/(face+'.png')
            if w.sha(path)!=row['final_sha256'][face]:raise ValueError('Complete candidate changed before export: '+row['sku']+'/'+face)
        shutil.copytree(source,staged/row['sku'])
    pipeline.publish(staged,output)
    for row in selected:
        package=row.get('portable_review_package')
        if package is not None:
            source=Path(package['zip_path'])
            if w.sha(source)!=package['zip_sha256']:
                raise ValueError('Portable review package changed before export: '+row['sku'])
            target=Path(output)/row['sku']/'候选证据包.zip'
            shutil.copy2(source,target)
            if w.sha(target)!=package['zip_sha256']:
                raise RuntimeError('Portable review package changed during export: '+row['sku'])
    return [row['sku'] for row in selected]


def run(source,work=None,output=None,*,jobs=1,device='auto',skus=None,single_name='SKU_01',cpu_threads=1,
        inference_slots=None,experimental_concurrency=False,model=None,protection_policy='sticker_only',review_fallback='fail',lighting_profile='accepted-shadow-lift-v2',resume_geometry_from=None,bottom_alignment='appearance',source_body_reference_review=False,
        saved_review_manifest=None,saved_review_manifest_sha256=None,require_geometry_resume=False,
        side_candidate_review=False):
    from .frame_seams import profile
    if type(source_body_reference_review) is not bool:raise ValueError('source_body_reference_review must be an explicit bool')
    if type(require_geometry_resume) is not bool:raise ValueError('require_geometry_resume must be an explicit bool')
    if type(side_candidate_review) is not bool:raise ValueError('side_candidate_review must be an explicit bool')
    if side_candidate_review and (saved_review_manifest is not None or not source_body_reference_review
            or protection_policy!='none' or lighting_profile!='accepted-shadow-lift-v2'):
        raise ValueError('Fresh side review requires no saved manifest, current source reference, none protection and accepted-shadow-lift-v2')
    if bottom_alignment not in ('appearance','off'):raise ValueError('Unknown bottom alignment mode')
    if protection_policy not in ('sticker_only','none'):raise ValueError('Unknown automatic protection policy')
    if review_fallback not in ('fail','preserve-source'):raise ValueError('Unknown optional processing failure policy')
    if saved_review_manifest is None and saved_review_manifest_sha256 is not None:
        raise ValueError('Saved review manifest SHA requires a manifest path')
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
    saved_review_descriptors={}
    saved_review_digest=None
    if saved_review_manifest is not None:
        from .saved_review import load_selected
        saved_review_manifest=Path(saved_review_manifest).expanduser().resolve(strict=True)
        saved_review_digest=w.sha(saved_review_manifest)
        if saved_review_manifest_sha256 is not None and saved_review_manifest_sha256!=saved_review_digest:
            raise ValueError('Saved review manifest changed since the request was created')
        saved_review_descriptors=load_selected(saved_review_manifest,
            {name:source if is_single else source/name for name in names},
            dict(model=model,protection_policy=protection_policy,review_fallback=review_fallback,
                lighting_profile=lighting_profile,bottom_alignment=bottom_alignment,
                source_body_reference_review=source_body_reference_review),
            expected_manifest_sha256=saved_review_digest)
    resume_root=Path(resume_geometry_from).expanduser().resolve() if resume_geometry_from is not None else None
    if require_geometry_resume and resume_root is None and set(saved_review_descriptors)!=set(names):
        raise ValueError('Required geometry resume needs an explicit checkpoint for ordinary processing')
    if resume_root is not None and not (resume_root/'cases').is_dir():raise ValueError('Geometry resume requires an existing batch work directory')
    if resume_root is not None:pipeline.separate_paths(resume_root,work,output)
    work.mkdir(parents=True);begin=time.perf_counter()
    protocol=dict(profile=PROFILE,protection_policy=protection_policy,hard_pixel_protection_zones=protection_policy!='none',started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        input=str(source),output=str(output),jobs=jobs,cpu_threads_per_worker=cpu_threads,
        skus=names,historical_masks_guides_landmarks_or_reviews_reused=bool(saved_review_descriptors),runtime_agent_required=False,manual_annotations_required=False,
        segmentation_model='ZhengPeng7/BiRefNet_dynamic',alignment_basis='back_image',
        segmentation_max_side_by_face={f:1536 if f=='front' else 1024 for f in pipeline.FACES},
        maximum_simultaneous_segmentation_calls=slots,experimental_concurrency=experimental_concurrency,
        final_visual_review_included_in_processing_time=False,model_uv_verified=False,
        frame_seam_model=model,frame_seam_model_sha256=model_data['model_sha256'])
    repo=Path(__file__).resolve().parents[1]
    protocol['implementation_sha256']=implementation_manifest(repo)
    protocol.update(protection_policy=protection_policy,review_fallback=review_fallback,lighting_profile=lighting_profile,quality_and_generation_counts_separate=True,
        resume_geometry_from=str(resume_root) if resume_root else None,geometry_reuse_requires_current_source_hashes=True,
        bottom_alignment=bottom_alignment,bottom_alignment_blocks_generation=False,
        source_body_reference_review_enabled=source_body_reference_review,
        side_candidate_review_enabled=side_candidate_review,
        side_candidate_generation_requires_current_sources=True,
        saved_review_manifest=str(saved_review_manifest) if saved_review_manifest is not None else None,
        saved_review_manifest_sha256=saved_review_digest,
        saved_review_selected_skus=sorted(saved_review_descriptors),
        saved_processing_results_reused=bool(saved_review_descriptors),
        normal_processing_skus=[n for n in names if n not in saved_review_descriptors],
        saved_review_requires_current_sources_and_evidence=True,require_geometry_resume=require_geometry_resume)
    write(work/'protocol.json',protocol)
    tasks=[(n,str(source if is_single else source/n),str(work/'cases'/n),device,cpu_threads,model,protection_policy,review_fallback,lighting_profile,str(resume_root/'cases'/n) if resume_root else None,bottom_alignment,source_body_reference_review,saved_review_descriptors.get(n),require_geometry_resume,side_candidate_review) for n in names]
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
    generated=sum(r.get('six_faces_generated',r.get('six_faces_processed',False)) for r in result)
    report=dict(protocol=protocol,attempts=result,processed=completed,generated=generated,source_preserved_candidates=sum(r.get('status')=='source_preserved_needs_review' for r in result),effect_warning_candidates=sum(r.get('status')=='full_stages_effect_warning_needs_review' for r in result),total=len(tasks),
        visual_approved=0,seconds=time.perf_counter()-begin,status='failed' if completed!=len(tasks) else 'processed_needs_visual_review')
    # A partial batch is explicitly partial; every exported SKU still has
    # six verified faces. One failed SKU must not hide other complete results.
    if completed==len(tasks) or (protection_policy=='none' and generated>0):
        names_exported=export_complete_candidates(result,work,output)
        report.update(output=str(output),exported_skus=names_exported)
        if completed!=len(tasks):
            report.update(status='review_candidates_needs_visual_review' if generated==len(tasks) else 'partial_candidates_needs_visual_review',partial_batch=generated!=len(tasks))
            from .candidate_manifest import candidate_manifest_item
            write(work/'partial-candidate-manifest.json',dict(generated=len(names_exported),total=len(tasks),
                visual_approved=0,notice='Only complete six-face candidates are exported; incomplete SKUs are listed, not presented as completed.',
                items=[candidate_manifest_item(row) for row in result]))
    write(work/'report.json',report)
    return report
