"""Current-job shared-L illumination assumption for review candidates.

A single metal-fitted low-frequency L field crosses unresolved source antenna
endpoints. This is not a material, endpoint or visual certificate.
"""
import copy
import hashlib
import json
from pathlib import Path

POLICY='metal_fit_shared_low_frequency_L_review_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value,message):
    if not value:raise ValueError('Shared illumination assumption: '+message)


def read(path):
    return json.loads(Path(path).read_text())


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def prepare(job_path, effect_scope_path, folder, *, input_bindings):
    """Transform a frozen partial job; do NOT render or invoke its finalizer.

    Reuses the existing post-profile main field, hardware domains, and actual
    source-known checks. The entire source-selected antenna envelope is absent
    from fitting, yet receives the same additive field as adjoining frame RGB.
    """
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    from . import lighting,shadow_lift_profile as profile
    from .lighting_review import validate_spec

    job_path, effect_scope_path = (Path(p).resolve() for p in (job_path, effect_scope_path))
    job = copy.deepcopy(read(job_path)); scope = copy.deepcopy(read(effect_scope_path))
    bindings = dict(input_bindings)
    require(str(job_path) in bindings and str(effect_scope_path) in bindings,
            'current job/scope were not bound before preparing the field')
    for path, digest in bindings.items():
        require(sha(path) == digest, 'bound current input changed: '+path)
    require(job.get('color_mode')=='source_lightness' and len(job.get('views',[]))==1,
            'one source-lightness side required')
    view=job['views'][0];face=view['name'];source=Path(view['source'])
    require(face in ('left','right') and scope.get('source_sha256') == view.get('source_sha256'),'wrong side identity')
    require(sha(source)==view['source_sha256']==scope['source_sha256'],'source identity differs')
    require(view.get('protection_policy')=='none','this prototype requires the authorized no-zone policy')
    with Image.open(source) as image:
        require(image.mode=='RGBA' and image.format=='PNG','source must be immutable RGBA PNG')
        a=np.asarray(image).copy()
    h,w=a.shape[:2];visible=a[...,3]>0
    def mask(path):
        require(str(Path(path)) in bindings and sha(path)==bindings[str(Path(path))],
                'mask lacks current input binding: '+str(path))
        with Image.open(path) as image:value=np.asarray(image.convert('L'))>0
        require(value.shape==(h,w),'mask dimensions differ')
        return value
    require(not view.get('protect') and not mask(view['protected_mask']).any(),'no hidden pixel protection')
    require(not view.get('editable_mask') and not view.get('exclude_from_fit'),
            'unhandled global edit/fit restriction')
    partial=view.get('partial_side_lighting_scope')
    require(partial and partial==scope.get('partial_side_lighting_scope'),'partial source scopes differ')
    require(partial['source_file_sha256']==sha(source) and partial['face']==face,'partial source binding differs')
    require(partial.get('classification_complete') is False and partial.get('strict_endpoint_evidence_complete') is False,
            'this is not a strict material/remap path')
    hold=mask(partial['assets']['retained-unknown']['path'])
    envelope=np.zeros((h,w),bool)
    require([v['band'] for v in partial['bands']]==[0,1],'both source-selected band envelopes required')
    for record in partial['bands']:
        lo,hi=record['local_window_yxy']
        require(type(lo) is int and type(hi) is int and 0<=lo<hi<=h,'invalid source envelope')
        require(record.get('window_is_processing_scope_not_material_boundary') is True,
                'envelope must not be called a material boundary')
        envelope[lo:hi]=True
    require(not np.any(hold&~envelope),'old unknown scope escapes source envelope')
    domains={d['name']:d for d in view['domains']}
    require(len(domains)==len(view['domains']) and {'main_surface','antenna_0','antenna_1'}<=set(domains),
            'expected distinct existing main/antenna domains')
    masks={name:mask(d['mask']) for name,d in domains.items()}
    for name in ('antenna_0','antenna_1'):
        require(not np.any(masks[name]&~envelope),'known antenna pixels escape fit exclusion envelope')
    others=[d for d in view['domains'] if d['name'] not in ('main_surface','antenna_0','antenna_1')]
    other_mask=np.zeros((h,w),bool)
    for domain in others:
        require(not np.any(other_mask&masks[domain['name']]),'other domains overlap')
        other_mask|=masks[domain['name']]
    carrier=(masks['main_surface']|masks['antenna_0']|masks['antenna_1']|hold)&visible
    require(not np.any(carrier&other_mask),'shared carrier would enter independent hardware/paper')
    if view.get('source_label_fit_exclusion_mask'):
        require(not np.any(carrier&mask(view['source_label_fit_exclusion_mask'])),
                'known paper needs its independent field; no shared-carrier paper support here')
    main=copy.deepcopy(domains['main_surface'])
    require(main.get('field','gaussian')=='gaussian','reuse only the existing Gaussian side field')
    fit=mask(main.get('fit_mask',main['mask']))&masks['main_surface']&~envelope&(a[...,3]>=240)
    require(int(fit.sum())>=20,'fewer than the original 20 metal fit samples')
    require(not np.any(fit&envelope) and not np.any(fit&other_mask),'fit leaked into antenna/hardware envelope')
    require(main.get('feather')==0 and main['strength'][1:]==[0,0] and main['max_lab_shift'][1:]==[0,0],
            'shared field must be unfeathered source L only')
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=False)
    assets=folder/'assets';assets.mkdir()
    def persist(name,value):
        path=assets/(name+'.png');Image.fromarray(value.astype(np.uint8)*255).save(path)
        return str(path)
    carrier_path=persist('carrier-application',carrier)
    fit_path=persist('confirmed-metal-fit-outside-entire-envelope',fit)
    envelope_path=persist('source-selected-envelope-not-material-boundary',envelope&visible)
    known_main_path=persist('original-known-metal',masks['main_surface']&(a[...,3]>=240))
    main.update(mask=carrier_path,fit_mask=fit_path,
        application_semantics='illumination_assumption_not_one_material',fit_material='source_selected_metal_outside_envelope')
    render=lighting.engine('aluminum')
    engine_path = str(Path(render.__file__).resolve())
    profile_path = str(Path(profile.__file__).resolve())
    require(engine_path in bindings and sha(engine_path) == bindings[engine_path],
            'current rendering engine lacks an intact binding')
    require(profile_path in bindings and sha(profile_path) == bindings[profile_path],
            'current profile lacks an intact binding')
    # The initial metal reference can lie INSIDE the newly excluded envelope.
    # Reuse the unchanged V2 selection rule, now on admissible source metal.
    new_box,reference_proof=profile.reference(a,main,render)
    old_boxes=copy.deepcopy(main['reference_boxes']);main['reference_boxes']=[new_box]
    require('acceptance_reference_boxes' not in main,'unexpected separate acceptance target')
    revised=[]
    for domain in view['domains']:
        if domain['name']=='main_surface':revised.append(main)
        elif domain['name'] not in ('antenna_0','antenna_1'):revised.append(domain)
    view['domains']=revised;view['reference_boxes']=[new_box]
    blends=[];removed=[]
    for blend in view.get('correction_blends',[]):
        if any(name in ('antenna_0','antenna_1') for name in blend['domains']):
            require(set(blend['domains']) in ({'antenna_0','main_surface'},{'antenna_1','main_surface'}) and
                    blend.get('kind')=='source_material_transition','unhandled multi-domain blend')
            removed.append(blend)
        else:blends.append(blend)
    view['correction_blends']=blends;view['antenna_boundary_bands']=[]
    for key in ('partial_side_lighting_scope','required_processing_finalize'):view.pop(key,None)
    allowed=carrier|other_mask
    view['required_editable_mask']=persist('required-entire-carrier-and-independent-domains',allowed)
    view['illumination_assumption_review']=dict(policy=POLICY,scope_mask=envelope_path,
        material_classification_complete=False,strict_endpoint_evidence_complete=False,
        structural_alignment_applied=False,source_geometry_unchanged=True,
        output_hold_mask_used=False,post_render_taper_used=False,quality_checks_complete=False,
        warning='未确认天线端点仅采用同源连续照明假设；保留原位置，效果待核对')
    # Existing witnesses remain exactly source-bound, including their actual
    # known antenna portions and every independent hardware component check.
    original_checks=copy.deepcopy(scope['local_checks'])
    for key in ('partial_side_lighting_scope','requires_partial_scope_finalization',
                'partial_lab_fixed_taper_px','unknown_endpoint_pixels_are_not_effect_witnesses'):
        scope.pop(key,None)
    scope.update(illumination_assumption_review=copy.deepcopy(view['illumination_assumption_review']),
        material_classification_complete=False,structural_alignment_certified=False,
        unknown_endpoint_pixels_are_not_material_witnesses=True)
    boundary_witnesses={}
    for record in partial['bands']:
        lo,hi=record['local_window_yxy'];witness=np.zeros((h,w),bool)
        witness[max(0,lo-1):min(h,lo+1)]=True
        witness[max(0,hi-1):min(h,hi+1)]=True
        witness&=carrier&(a[...,3]>=240)
        key='carrier_fit_envelope_'+str(record['band'])+'_entry_exit'
        path=persist(key,witness);boundary_witnesses[key]=path
        scope['local_checks'][key]=dict(mask=path,min_L=0,max_L=100,percentile=10,
            max_added_step_L=1.25,max_pixel_added_step_L=1.25,step_axis=0,
            bounds_basis='Correction-field continuity across fit-exclusion entry/exit; not a material/endpoint truth')
        whole=np.zeros((h,w),bool);whole[max(0,lo-1):min(h,hi+1)]=True
        whole&=carrier&(a[...,3]>=240)
        key='carrier_envelope_'+str(record['band'])+'_tangent_field'
        path=persist(key,whole);boundary_witnesses[key]=path
        scope['local_checks'][key]=dict(mask=path,min_L=0,max_L=100,percentile=10,
            max_added_step_L=1.25,max_pixel_added_step_L=1.25,step_axis=0,
            bounds_basis='Added-L field tangent continuity on all opaque carrier pixels, including unknown material; not endpoint truth')
    require(all(scope['local_checks'][k]==value for k,value in original_checks.items()),'original source checks changed')
    hardware_boundaries={}
    for domain in others:
        if not domain['name'].startswith('hardware_'):continue
        hardware=masks[domain['name']]
        require(domain==domains[domain['name']],'hardware field parameters changed')
        seam=(hardware&ndimage.binary_dilation(carrier))|(carrier&ndimage.binary_dilation(hardware))
        hardware_boundaries[domain['name']]=persist(domain['name']+'-source-boundary-diagnostic',seam&(a[...,3]>=240))
    validate_spec(scope,source)
    job.update(output_dir=str(folder/'render'),records_dir=str(folder/'records'))
    save(folder/'job.json',job);save(folder/'scope.json',scope)
    info=dict(policy=POLICY,status='illumination_assumption_review',source=str(source),source_sha256=sha(source),
        source_size=[w,h],input_sha256=bindings,implementation_sha256=sha(__file__),
        original_job=str(Path(job_path).resolve()),original_scope=str(Path(effect_scope_path).resolve()),
        engine_sha256=sha(render.__file__),profile_sha256=sha(profile.__file__),
        carrier_mask=carrier_path,fit_mask=fit_path,envelope_mask=envelope_path,known_metal_mask=known_main_path,
        allowed_mask=view['required_editable_mask'],protected_mask=view['protected_mask'],
        hardware_boundary_diagnostic_masks=hardware_boundaries,field_entry_exit_witnesses=boundary_witnesses,
        original_known_local_checks_preserved=list(original_checks),
        original_reference_boxes=old_boxes,reference_proof=reference_proof,
        unchanged_field_parameters={k:main.get(k) for k in ('sigma','strength','max_lab_shift','correction_smooth','feather',
            'gamut_continuity_step_L','gamut_continuity_follow_field','gamut_continuity_follow_source_edges')},
        removed_antenna_only_blends=removed,fit_pixels=int(fit.sum()),carrier_pixels=int(carrier.sum()),
        restored_former_hold_pixels=int((carrier&hold).sum()),fit_envelope_pixels=int((envelope&visible).sum()),
        support_floor_unchanged=.008,minimum_fit_samples_unchanged=20,continuity_limit_unchanged=1.25,
        support_must_still_pass_engine=True,source_geometry_changed=False,output_hold_mask_used=False,
        material_classification_complete=False,strict_endpoint_evidence_complete=False,
        structural_alignment_applied=False,quality_checks_complete=False,visual_approved=False,
        selected_for_delivery=False,render_executed=False)
    info['prepared_assets_sha256']={str(path):sha(path) for path in folder.rglob('*') if path.is_file()}
    require(all(sha(path)==digest for path,digest in bindings.items()),'current source/implementation changed')
    save(folder/'preparation.json',info)
    return job,scope,info


def verify_source_contract(preparation_path,candidate_path):
    """Hard source/alpha/scope checks only; does not accept effect/material."""
    import numpy as np
    from PIL import Image
    proof=read(preparation_path)
    for path,digest in {**proof['input_sha256'],**proof['prepared_assets_sha256']}.items():
        require(sha(path)==digest,'bound source/job/scope changed: '+path)
    require(proof['implementation_sha256']==sha(__file__),'prototype implementation changed')
    with Image.open(proof['source']) as image:a=np.asarray(image.convert('RGBA'))
    with Image.open(candidate_path) as image:
        require(image.mode=='RGBA' and image.format=='PNG','candidate format differs')
        b=np.asarray(image)
    with Image.open(proof['allowed_mask']) as image:allowed=np.asarray(image.convert('L'))>0
    require(a.shape==b.shape and np.array_equal(a[...,3],b[...,3]),'candidate geometry/alpha differs')
    require(np.array_equal(a[~allowed],b[~allowed]),'pixels outside allowed scope changed')
    return dict(status='illumination_assumption_review',source_and_alpha_contract_passed=True,
        alpha_changed=0,outside_allowed_changed=0,changed_pixels=int(np.any(a!=b,axis=2).sum()),
        actual_effect_checks_executed=False,material_classification_complete=False,
        strict_endpoint_evidence_complete=False,structural_alignment_applied=False,
        quality_checks_complete=False,visual_approved=False,selected_for_delivery=False)


def field_and_hardware_diagnostics(preparation_path,candidate_path):
    """Measure final added L, never invent endpoint/material witnesses.

    Existing local/hardware scope checks remain the acceptance constraints.
    These two-axis statistics expose changes at source hardware boundaries and
    across the full assumption envelope; they do not certify naturalness.
    """
    import numpy as np
    from PIL import Image
    from .lighting_review import lightness
    verify_source_contract(preparation_path,candidate_path)
    info=read(preparation_path)
    with Image.open(info['source']) as image:a=np.asarray(image.convert('RGBA'))
    with Image.open(candidate_path) as image:b=np.asarray(image.convert('RGBA'))
    before,after=lightness(a[...,:3]),lightness(b[...,:3]);delta=after-before
    paths=dict(info['hardware_boundary_diagnostic_masks'])
    paths['entire_source_envelope']=info['envelope_mask']
    result={}
    for name,path in paths.items():
        with Image.open(path) as image:region=(np.asarray(image.convert('L'))>0)&(a[...,3]>=240)
        axes={}
        for axis in (0,1):
            pairs=(region[1:]&region[:-1]) if axis==0 else (region[:,1:]&region[:,:-1])
            values=np.abs(np.diff(delta,axis=axis))[pairs]
            old=np.diff(before,axis=axis)[pairs];new=np.diff(after,axis=axis)[pairs]
            axes[str(axis)]=dict(pair_count=int(pairs.sum()),
                added_L_step_max=float(values.max()) if len(values) else None,
                added_L_step_p75=float(np.percentile(values,75)) if len(values) else None,
                added_L_step_p95=float(np.percentile(values,95)) if len(values) else None,
                pairs_above_1_25_L=int(np.sum(values>1.25)),
                source_contrast_sign_reversals_above_1_25_L=int(np.sum((np.abs(old)>1.25)&(old*new<0))))
        result[name]=dict(axes=axes,semantic_material_certification=False,automatic_acceptance=False)
    return dict(status='illumination_assumption_review',measurements=result,
        source_texture_subtracted=True,strict_material_or_structure_evidence_added=False,
        visual_approved=False,selected_for_delivery=False)
