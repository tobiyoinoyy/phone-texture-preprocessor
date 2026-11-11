"""Export an explicitly unaligned review candidate without inventing structure."""
import json,shutil,traceback,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from . import pipeline,lighting,lighting_review,auto_material,frame_seams
from .geometry_review import digest,write


def failure_kind(error):
    text=str(error)
    antenna=('Opaque antenna','Original independent source edges','No missing opaque source endpoint',
      'Raw central pair','Source endpoint','No paired endpoint','Too few source-supported paired antenna',
      'Endpoint antenna shape','Frozen antenna','Rear companion frozen boundaries','Rear antenna landmarks',
      'Original source antenna','Independent hardware contours overlap an antenna','No supported antenna strip',
      'Insufficient central paired antenna','Unresolved opaque antenna','Antenna central polarity','Antenna requires opposing central material edges')
    if text.startswith(antenna):return 'antenna_evidence'
    if text.startswith(('Unsupported editable illumination','Unsupported editable contour illumination',
      'Perimeter effect failed','Lighting correction discontinuity','Lighting pixel correction discontinuity','UV seam correction field discontinuity',
      'Regional effect failed','Local effect failed','UV seam lacks required jointly editable material coverage',
      'UV seam color difference exceeds the delivery limit','Seam contains no jointly editable samples','Overlapping seam bands need joint solve')):return 'effect_or_seam'
    if text.startswith(('Ambiguous crossing side hardware contours','Rear panel corner lacks source seam support',
      'Unresolved left hardware','Unresolved right hardware','Rear plastic strip has insufficient',
      'Side hardware','No independent complete hardware contours','Front rim lacks nearby opaque material support',
      'Bottom paper material boundary unresolved')):return 'material_classification'
    return None


def require_material_boundary(job, scope=None):
    """A location hint cannot certify the paper/metal domain used for lighting."""
    records = [view.get('source_label_material_evidence', {}) for view in job.get('views', [])
        if view.get('source_label_location_hint_mask') or view.get('source_label_material_evidence')]
    if scope is not None and 'source_paper_material_boundary' in scope:
        records.append(scope['source_paper_material_boundary'])
    for evidence in records:
        resolution = evidence.get('material_boundary_resolution')
        if resolution == 'source_contour_frame_domain':
            from .source_material_bridge import validate_source_contour_material
            view = next((v for v in job.get('views', [])
                         if v.get('source_label_material_evidence') == evidence), None)
            if view is None:
                raise ValueError('Source-closed scope has no matching material view')
            validate_source_contour_material(view,evidence,scope=scope)
            continue
        if resolution not in ('original_source_closure', 'source_aligned_metal_window'):
            error = ValueError('Bottom paper material boundary unresolved: '+str(resolution or 'missing source boundary evidence'))
            error.material_boundary_evidence = evidence
            raise error


def source_seam_observations(files,geometry_files,model):
    """Measure actual pixels at fixed UV profiles; never classify this as a pass."""
    profile,path=frame_seams.profile(model);chains=frame_seams.resized_chains(profile,files)
    engine=lighting.engine('aluminum');observations=[]
    for chain in chains:
        arc=np.asarray(chain['arc']);ss=np.linspace(0,arc[-1],192);values=[];original=[];valid=np.ones(192,bool)
        for side in chain['sides']:
            name=side['name'];p=np.column_stack([np.interp(ss,arc,np.asarray(side['points'])[:,k]) for k in range(2)])
            for collection,source in [(values,files[name]),(original,geometry_files[name])]:
                a=np.array(Image.open(source).convert('RGBA'));sample=np.column_stack([map_coordinates(a[...,k].astype(float),[p[:,1],p[:,0]],order=1,mode='nearest') for k in range(4)])
                valid &= sample[:,3]>=240;collection.append(engine.rgb_to_lab(sample[:,:3]))
        delta=np.linalg.norm(values[0]-values[1],axis=1);before=np.linalg.norm(original[0]-original[1],axis=1)
        observations.append(dict(names=chain['names'],opaque_samples=int(valid.sum()),samples=192,
            candidate_deltaE76_p50=float(np.median(delta[valid])) if valid.any() else None,
            candidate_deltaE76_p95=float(np.percentile(delta[valid],95)) if valid.any() else None,
            geometry_source_deltaE76_p95=float(np.percentile(before[valid],95)) if valid.any() else None))
    return dict(status='measured_only_not_uv_accepted',method='actual_opaque_pixels_at_fixed_model_UV_profiles',
        material_classification_verified=False,alignment_applied=False,correction_applied=False,
        profile_sha256=digest(path),observations=observations)


def preserve(case,sku,geometry,geometry_files,checks,model,error,original_traceback,lighting_profile='baseline'):
    from .shadow_lift_profile import apply as apply_lighting_profile,measure_effect
    """Every chosen PNG is a verified stage file; no structural interpolation."""
    from .automatic import measure_masks,no_zone_effects
    kind=failure_kind(error)
    if not kind:raise ValueError('This failure is not an optional transformation failure')
    items={r['face']:r for r in geometry['images']}
    if set(items)!=set(pipeline.FACES) or not geometry.get('candidates_complete'):
        raise ValueError('Preservation requires complete validated geometry')
    for face,p in geometry_files.items():
        if digest(p)!=items[face]['output_sha256']:raise ValueError('Geometry source changed before preservation')
    staged=case/'work/source-preserved-review'/sku;pipeline.fresh(staged);staged.mkdir(parents=True)
    face_records={};additional_errors=[];light_checks={};back_canvas_proof=None
    for face in pipeline.FACES:
        source=geometry_files[face];selected=source;stage='geometry_source';lighting_file=case/'work/lighting'/sku/(face+'.png')
        # Unknown antenna evidence never enters an inferred material mask or warp.
        must_keep_source=kind=='antenna_evidence' and face in ('back','left','right')
        checked=checks.get(face,{})
        if not must_keep_source and checked.get('source_relative_effect_checks'):
            if digest(source)!=checked['source_sha256'] or digest(lighting_file)!=checked['output_sha256']:
                raise ValueError('Checked lighting source changed before preservation')
            selected=lighting_file;stage='pixel_verified_lighting_effect_warning' if checked['source_relative_effect_checks'].get('original_effect_check_passed') is False else 'verified_lighting';light_checks[face]=checked
        elif face in ('front','top','bottom'):
            assets=case/'work/source-preserved-analysis'/face
            try:
                job,analysis,allowed,paper=auto_material.propose_job(source,face,assets,protection_policy='none')
                write(assets/'material-proposal.json',job);write(assets/'analysis.json',analysis)
                require_material_boundary(job)
                job,profile_proof=apply_lighting_profile(job,face,assets,lighting_profile)
                analysis['lighting_profile']=profile_proof
                scope=auto_material.lighting_scope(source,face,job,assets/'effect-witnesses')
                require_material_boundary(job,scope)
                job.update(output_dir=str(assets/'output'),records_dir=str(assets/'records'))
                jp=assets/'job.json';write(jp,job);lighting.run('aluminum',jp);output=assets/'output'/(face+'.png')
                pixel=measure_masks(source,output,allowed,paper)
                effect=measure_effect(lighting_review.measure_perimeter if face=='front' else lighting_review.measure_local,scope,source,output,allowed,paper,lighting_profile)
                pixel['source_relative_effect_checks']=effect;pixel['actual_no_zone_effects']=no_zone_effects(source,output,analysis)
                selected=output;stage='pixel_verified_lighting_effect_warning' if effect.get('original_effect_check_passed') is False else 'independently_verified_lighting';light_checks[face]=pixel
            except ValueError as exc:
                # This optional effect was never selected. Retain the trusted
                # stage image; mandatory source hashes/alpha checks below and
                # the outer intake checks still apply, and the error is kept.
                additional_errors.append(dict(face=face,error=str(exc),traceback=traceback.format_exc(),action='retained_geometry_source_without_effect_claim'))
        if digest(source)!=items[face]['output_sha256']:raise ValueError('Geometry source changed during optional processing')
        a=np.array(Image.open(source).convert('RGBA'));b=np.array(Image.open(selected).convert('RGBA'))
        if a.shape!=b.shape or not np.array_equal(a[...,3],b[...,3]):raise ValueError('Preservation changed source position or alpha')
        target=staged/(face+'.png')
        if face=='back':
            from .back_canvas import standardize
            back_canvas_proof=standardize(selected,target,model)
        else:
            shutil.copy2(selected,target)
            if digest(target)!=digest(selected):raise ValueError('Preserved stage PNG differs after copying')
        face_records[face]=dict(stage=stage,source=str(source),selected_stage_file=str(selected),source_sha256=digest(source),
            selected_sha256=digest(selected),source_RGBA_byte_identical=digest(target)==digest(source),
            RGB_changed_pixels=int(np.any(a[...,:3]!=b[...,:3],axis=2).sum()),alpha_changed_pixels=0,
            structural_warp_applied=False,spatial_resampling_displacement_px=0,antenna_position_reconstructed=False,
            geometry_transform_from_raw=items[face]['geometry'])
    seams=source_seam_observations({f:staged/(f+'.png') for f in pipeline.FACES},geometry_files,model)
    proposed_seams=seams;reverted=[]
    # Fixed UV measurements on unaligned source positions are review evidence,
    # not permission to discard independently verified lighting stages.
    seams['source_baseline_nonregression']=all(row['candidate_deltaE76_p95'] is not None and row['geometry_source_deltaE76_p95'] is not None and row['candidate_deltaE76_p95']<=row['geometry_source_deltaE76_p95']+1e-8 for row in seams['observations'])
    seams['lighting_retained_for_independent_review']=True
    for face,source in geometry_files.items():
        if digest(source)!=items[face]['output_sha256']:raise ValueError('Geometry source changed during optional processing')
    candidate=case/'candidate';pipeline.publish(staged.parent,candidate)
    final=pipeline.batch_files(candidate)
    warnings=[dict(face='all',code='source_positions_preserved_alignment_unverified',
        message='天线保留原位置；未做结构对齐，接缝待核对'),
        dict(face='all',code='optional_processing_check_failed',message='部分处理未应用，已保留源图，详见逐面记录')]
    record=dict(status='source_preserved_needs_review',six_faces_generated=True,six_faces_processed=False,
        processing_level='source_positions_preserved_review_candidate',mandatory_program_checks_complete=False,quality_checks_complete=False,
        structural_alignment_applied=False,seam_accepted=False,candidate=str(candidate),review_warnings=warnings,
        blocked_check=dict(kind=kind,error=str(error),traceback=original_traceback),preserved_faces=face_records,
        additional_optional_failures=additional_errors,lighting_checks=light_checks,frame_uv_seam_checks=seams,
        rejected_partial_effects=reverted,proposed_partial_effect_seam_observations=proposed_seams,
        back_standard_canvas=back_canvas_proof,
        final_sha256={f:digest(p) for f,p in final[sku].items()},source_hashes_unchanged=True)
    write(case/'work/source-preservation-report.json',record)
    pipeline.overview({sku:geometry_files},final,case/'overview.png',faces=pipeline.FACES)
    return record
