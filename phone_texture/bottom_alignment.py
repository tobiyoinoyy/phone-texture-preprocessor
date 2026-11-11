"""Optional bottom registration before all lighting masks and source bindings.

A rejected transform keeps the exact input PNG; it never fails six-face
creation on its own. Inputs and original geometry records remain immutable.
"""
from pathlib import Path
import copy,json,shutil
import numpy as np
from PIL import Image
from .geometry_review import digest,write
from .bottom_rectify import rectify,Rejected

MODEL_KEYS={'iphone17':'iphone17','iphone17pro':'ip17pro','iphone17promax':'ip17promax',
            'ip17pro':'ip17pro','ip17promax':'ip17promax'}

def validate_transform(source,out,mapx,proof):
    h,w=source.shape[:2];unit=w/1030
    if out.shape!=source.shape or out.dtype!=np.uint8:raise Rejected('candidate_format_or_canvas_changed')
    if mapx.shape!=(h,w) or not np.isfinite(mapx).all():raise Rejected('nonfinite_or_invalid_mapping')
    if mapx.min()<0 or mapx.max()>w-1:raise Rejected('sampling_outside_visible_source')
    jac=np.gradient(mapx,axis=1)
    if jac.min()<.65 or jac.max()>1.4:raise Rejected('fold_or_excessive_local_stretch')
    if not np.allclose(mapx,mapx[0:1],rtol=0,atol=0):raise Rejected('unsupported_vertical_variation')
    cap=max(12,int(.6*h));identity=np.arange(w,dtype='float32')
    if not (np.array_equal(mapx[:,:cap],np.broadcast_to(identity[:cap],(h,cap))) and
            np.array_equal(mapx[:,-cap:],np.broadcast_to(identity[-cap:],(h,cap)))):
        raise Rejected('endcap_mapping_changed')
    if not np.array_equal(source[:,:cap],out[:,:cap]) or not np.array_equal(source[:,-cap:],out[:,-cap:]):
        raise Rejected('endcap_pixels_changed')
    bounds=[];supports=[]
    for image in [source,out]:
        mask=image[...,3]>=128;supports.append(mask.any(axis=0))
        bounds.append(np.stack([mask.argmax(axis=0),h-1-mask[::-1].argmax(axis=0)]))
    if not np.array_equal(*supports):raise Rejected('body_column_support_changed')
    movement=float(abs(bounds[1]-bounds[0])[:,supports[0]].max())
    if movement>int(np.ceil(2*unit)):raise Rejected('body_outline_changed_excessively')
    if abs(proof.get('port_width_change_fraction',0))>.2:raise Rejected('port_shape_change_too_large')
    return dict(canvas_unchanged=True,endcaps_exact=True,source_sampling_in_bounds=True,
        finite_monotone_map=True,minimum_sampling_derivative=float(jac.min()),
        maximum_sampling_derivative=float(jac.max()),max_boundary_move_px=movement,
        alpha_changed_pixels=int(np.count_nonzero(source[...,3]!=out[...,3])))

def prepare(files,geometry,model,process,*,enabled=True):
    process=Path(process);process.mkdir(parents=True,exist_ok=False)
    selected=process/'selected';selected.mkdir()
    before={f:digest(p) for f,p in files.items()};source=Path(files['bottom'])
    with Image.open(source) as image:
        if image.mode!='RGBA' or image.format!='PNG':raise ValueError('Bottom geometry must be RGBA PNG')
        a=np.asarray(image).copy()
    report=dict(version='bottom-appearance-nonblocking-v1',goal='visually_centered_with_natural_screws',
        model=model,source=str(source),source_sha256=before['bottom'],status='kept_source_with_notice',
        applied=False,blocks_generation=False,model_uv_verified=False,physical_accuracy_verified=False,
        manual_annotation_required=False,notices=[],metrics={},source_geometry_unmodified=True)
    out=None;key=MODEL_KEYS.get(model)
    if not enabled:
        report.update(status='disabled',notices=[])
    else:
        try:
            if not key:raise Rejected('unsupported_model_no_inferred_profile')
            out,proof,mx,active=rectify(a,key)
            report['metrics']=proof;report['integrity']=validate_transform(a,out,mx,proof)
            if proof['action'].startswith('unchanged'):
                report.update(status='unchanged',applied=False)
            else:
                report.update(status='applied_with_notice',applied=True)
                report['notices'].append(dict(code='image_space_port_alignment',severity='info',
                    message='底部已按可见结构校正；口沿精确层级未经标定，不阻断外观用途生成。'))
                np.save(process/'source_x.npy',mx)
                report['source_x_mapping']=dict(path=str(process/'source_x.npy'),sha256=digest(process/'source_x.npy'))
        except (Rejected,ValueError,RuntimeError,FloatingPointError) as error:
            out=None
            reason=str(error);evidence=getattr(error,'evidence',{})
            report.update(status='kept_source_with_notice',applied=False,reason=reason,rejected_transform_evidence=evidence)
            report['notices'].append(dict(code='bottom_alignment_not_applied',severity='review',
                message='底面未应用充电口校正；后续步骤继续尝试，请核对最终底面。',reason=reason))
    for face,path in files.items():
        target=selected/(face+'.png')
        if face=='bottom' and report['applied']:Image.fromarray(out).save(target)
        else:shutil.copy2(path,target)
    after={f:digest(selected/(f+'.png')) for f in files}
    if any(digest(files[f])!=h for f,h in before.items()):raise ValueError('Geometry input changed during optional bottom alignment')
    if any(after[f]!=before[f] for f in files if f!='bottom'):raise ValueError('Bottom alignment changed a companion face')
    if not report['applied'] and after['bottom']!=before['bottom']:raise ValueError('Bottom fallback is not byte identical')
    report.update(selected_dir=str(selected),selected_sha256=after['bottom'],companion_faces_byte_identical=True,
        source_hashes_unchanged=True,fallback_byte_identical=(after['bottom']==before['bottom']) if not report['applied'] else None)
    effective=copy.deepcopy(geometry)
    item=next(v for v in effective['images'] if v['face']=='bottom')
    if item['output_sha256']!=before['bottom']:raise ValueError('Geometry provenance mismatch before bottom alignment')
    item['pre_bottom_alignment_sha256']=before['bottom'];item['output_sha256']=after['bottom']
    item.setdefault('geometry',{})['optional_bottom_alignment']=dict(status=report['status'],
        source_sha256=before['bottom'],output_sha256=after['bottom'],record=str(process/'report.json'))
    write(process/'report.json',report);write(process/'effective-geometry.json',effective)
    return {f:selected/(f+'.png') for f in files},effective,report


def retain_source_after_downstream_failure(files,geometry,original_files,original_geometry,report,process,error):
    """One bottom-only retry after a rejected optional candidate, never an approval."""
    from .stage_processing import geometry_identity
    process=Path(process)
    if not report.get('applied') or report.get('downstream_source_retry'):
        raise RuntimeError('Only one applied bottom candidate may retry its source')
    sku=original_geometry['images'][0].get('sku')
    geometry_identity(original_geometry,original_files,sku)
    geometry_identity(geometry,files,sku)
    attempt=copy.deepcopy(report)
    write(process/'attempt-report.json',attempt)
    write(process/'attempt-effective-geometry.json',geometry)
    retained=process/'source-fallback-selected';retained.mkdir()
    for face,path in files.items():
        source=original_files['bottom'] if face=='bottom' else path
        target=retained/(face+'.png');shutil.copy2(source,target)
        if digest(target)!=digest(source):raise RuntimeError('Bottom fallback copy changed source')
        files[face]=target
    original=copy.deepcopy(next(x for x in original_geometry['images'] if x['face']=='bottom'))
    original['pre_bottom_alignment_sha256']=report['source_sha256']
    original.setdefault('geometry',{})['optional_bottom_alignment']=dict(
        status='kept_source_with_notice',record=str(process/'report.json'),
        attempted_source_sha256=attempt['selected_sha256'],output_sha256=digest(files['bottom']))
    geometry['images']=[original if x['face']=='bottom' else x for x in geometry['images']]
    geometry_identity(geometry,files,sku)
    # Final-selection metrics must not describe a rejected transformed image.
    report['attempted_metrics']=report.pop('metrics',{})
    report['metrics']={}
    report['attempted_integrity']=report.pop('integrity',{})
    if 'source_x_mapping' in report:
        report['attempted_source_x_mapping']=report.pop('source_x_mapping')
    report['integrity']=dict(fallback_byte_identical=True,alpha_changed_pixels=0,
        canvas_unchanged=True,selected_source='verified_original_geometry')
    report.update(status='kept_source_with_notice',applied=False,reason='downstream_candidate_rejected: '+str(error),
        attempted_applied=True,attempt_report=str(process/'attempt-report.json'),
        attempt_report_sha256=digest(process/'attempt-report.json'),
        attempt_selected_dir=attempt['selected_dir'],attempt_selected_sha256=attempt['selected_sha256'],
        selected_dir=str(retained),selected_sha256=digest(files['bottom']),fallback_byte_identical=True,
        downstream_source_retry=True,downstream_source_retry_reason=str(error),
        notices=[dict(code='bottom_alignment_not_applied',severity='review',
            message='充电口校正候选未通过后续检查，已退回原底面重试调光；最终调光状态见逐面报告。',reason=str(error))])
    write(process/'report.json',report);write(process/'effective-geometry.json',geometry)
    return files['bottom']


def original_source_for_quality_comparison(files,geometry,original_files,original_geometry,report):
    """Validate a read-only trial source without changing the selected mapping."""
    from .stage_processing import geometry_identity
    if not report.get('applied') or report.get('downstream_source_retry') or report.get('downstream_quality_selection'):
        raise RuntimeError('Only a currently applied bottom candidate may compare its original')
    sku=original_geometry['images'][0].get('sku')
    geometry_identity(original_geometry,original_files,sku)
    geometry_identity(geometry,files,sku)
    if (digest(original_files['bottom'])!=report['source_sha256'] or
            digest(files['bottom'])!=report['selected_sha256']):
        raise RuntimeError('Bottom comparison sources differ from alignment report')
    return original_files['bottom']


def retain_source_after_quality_comparison(files,geometry,original_files,original_geometry,report,process,comparison_path):
    """Commit a fully checked original; never convert a warning into an error."""
    from .stage_processing import geometry_identity
    from .bottom_quality_selection import verify_attempt
    original_source_for_quality_comparison(files,geometry,original_files,original_geometry,report)
    comparison_path=Path(comparison_path)
    comparison=json.loads(comparison_path.read_text())
    if (comparison.get('version')!='bottom_quality_selection_v1' or comparison.get('selected')!='original'
            or comparison.get('warning_reclassified_as_error') is not False
            or comparison.get('original',{}).get('original_effect_check_passed') is not True
            or comparison.get('port',{}).get('original_effect_check_passed') is not False):
        raise RuntimeError('Original bottom selection requires the complete passing comparison')
    for name in ('port','original'):
        verify_attempt(comparison[name])
    if (comparison['port']['source_sha256']!=report['selected_sha256'] or
            comparison['original']['source_sha256']!=report['source_sha256']):
        raise RuntimeError('Bottom quality comparison belongs to another source')
    process=Path(process);attempt=copy.deepcopy(report)
    write(process/'quality-attempt-report.json',attempt)
    write(process/'quality-attempt-effective-geometry.json',geometry)
    retained=process/'source-quality-selected';retained.mkdir()
    before={face:digest(path) for face,path in files.items()}
    original_before={face:digest(path) for face,path in original_files.items()}
    destinations={}
    for face,path in files.items():
        source=original_files['bottom'] if face=='bottom' else path
        target=retained/(face+'.png');shutil.copy2(source,target)
        if digest(target)!=digest(source):raise RuntimeError('Bottom quality selection copy changed source')
        destinations[face]=target
    if ({face:digest(path) for face,path in files.items()}!=before or
            {face:digest(path) for face,path in original_files.items()}!=original_before):
        raise RuntimeError('Bottom source changed during quality selection')
    original=copy.deepcopy(next(x for x in original_geometry['images'] if x['face']=='bottom'))
    original['pre_bottom_alignment_sha256']=report['source_sha256']
    original.setdefault('geometry',{})['optional_bottom_alignment']=dict(
        status='original_selected_after_quality_comparison',record=str(process/'report.json'),
        attempted_source_sha256=attempt['selected_sha256'],output_sha256=digest(destinations['bottom']))
    files.update(destinations)
    geometry['images']=[original if x['face']=='bottom' else x for x in geometry['images']]
    geometry_identity(geometry,files,original_geometry['images'][0].get('sku'))
    report['attempted_metrics']=report.pop('metrics',{});report['metrics']={}
    report['attempted_integrity']=report.pop('integrity',{})
    if 'source_x_mapping' in report:report['attempted_source_x_mapping']=report.pop('source_x_mapping')
    report['integrity']=dict(fallback_byte_identical=True,alpha_changed_pixels=0,
        canvas_unchanged=True,selected_source='verified_original_geometry')
    report.update(status='original_selected_after_quality_comparison',applied=False,
        reason='original_bottom_all_effect_checks_passed_port_effect_warning_retained_in_evidence',
        attempted_applied=True,attempt_report=str(process/'quality-attempt-report.json'),
        attempt_report_sha256=digest(process/'quality-attempt-report.json'),
        attempt_selected_dir=attempt['selected_dir'],attempt_selected_sha256=attempt['selected_sha256'],
        selected_dir=str(retained),selected_sha256=digest(files['bottom']),fallback_byte_identical=True,
        downstream_quality_selection=True,quality_comparison=str(comparison_path),
        quality_comparison_sha256=digest(comparison_path),
        notices=[dict(code='bottom_original_selected_after_quality_comparison',severity='review',
            message='充电口候选保留效果预警；原底面调光通过本轮程序检查，已选原底面，最终效果待核对。')])
    write(process/'report.json',report);write(process/'effective-geometry.json',geometry)
    return files['bottom']
