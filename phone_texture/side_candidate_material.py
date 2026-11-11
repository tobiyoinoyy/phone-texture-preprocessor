"""Local current-source adapter; the ordinary material path is untouched.

The proposal preserves the existing partial-material masks, fit policy and
hardware checks. Only left/right with explicit current observations can enter.
"""
from pathlib import Path
import numpy as np
from scipy import ndimage
from PIL import Image
from .geometry_review import digest
from .auto_material import (paper_mask, save_mask, front_metal, rear_camera_glass,
    rear_body_contour, _independent_rear_material_reference, rear_panel,
    reference_box, REAR_GAMUT_CONTINUITY_STEP_L)


def propose_job(source,face,folder,companions=None,landmarks=None,protection_policy='sticker_only',partial_side_evidence=None):
    """Create fresh material masks/reference boxes for one geometry candidate."""
    if face not in ('left', 'right') or protection_policy != 'none' or partial_side_evidence is None:
        raise ValueError('Side review material requires current observations, side face and explicit no-zone policy')
    from .auto_structure import antenna_rows,per_column_boundaries,hardware_ranges
    image=Image.open(source).convert('RGBA');a=np.array(image);h,w=a.shape[:2]
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    if protection_policy not in ('sticker_only','none'):raise ValueError('Unknown automatic protection policy')
    visible=a[...,3]>0;label_fit=np.zeros((h,w),bool);label_fit_proof=None;label_hint=None
    if protection_policy=='none':
        label_fit,label_fit_proof=paper_mask(image,face,resolve_end=False)
        if face=='bottom' and label_fit.any():
            from .bottom_paper_closure import resolve_material_strip_end
            label_hint=label_fit.copy()
            try:
                label_fit,label_fit_proof=resolve_material_strip_end(a,label_fit,label_fit_proof)
                if label_fit_proof.get('material_boundary_resolution') == 'unresolved_no_source_cross_edge':
                    # A returned unresolved hint must reach the independent
                    # material proposal path; it is never an accepted domain.
                    raise ValueError('Bottom paper material boundary unresolved: '+
                        label_fit_proof.get('material_boundary_warning', 'unresolved source cross edge'))
            except ValueError as original_error:
                if not str(original_error).startswith('Bottom paper material boundary unresolved:'):
                    raise
                # A tight source crop may lack the old three-row metal window.
                # Independently measure a closed paper boundary and frame on
                # this source. Canny peer avoidance is not a material verdict.
                from .source_material_bridge import resolve_current_source_material
                label_fit,label_fit_proof=resolve_current_source_material(
                    source,label_hint,label_fit_proof,folder/'source-contour-material',
                    legacy_error=str(original_error))
        paper=np.zeros((h,w),bool);paper_proof=dict(method='no_pixel_protection_zones',protected_pixels=0)
    else:paper,paper_proof=paper_mask(image,face)
    if protection_policy!='none' and face in ('back','bottom') and not paper.any():
        raise ValueError('This QC studio profile requires a resolved printed paper label on back/bottom')
    paper &= visible;paper_path=save_mask(folder/'paper.png',paper)
    remaining=visible&~paper;materials=[];proof={'paper':paper_proof};fit_masks={}
    partial_scope=None
    if face=='front':
        remaining,rim=front_metal(image);remaining &= ~paper;proof['front_rim']=rim
    if face in ('left','right'):
        if partial_side_evidence is not None:
            # PROPOSAL: explicit opt-in only. This is scoped lighting, never a
            # substitute for the unchanged strict structure/full-span route.
            from .side_candidate_scope import from_bound_report
            partial_scope=from_bound_report(source,face,partial_side_evidence)
            for index in (0,1):
                name='antenna_'+str(index);region=partial_scope['antenna_masks'][name]&remaining
                seeds=partial_scope['metadata']['bands'][index]['source_seeds']
                materials.append((name,region,[max(6,(seeds[1]-seeds[0])*.30),max(8,w*.20)]))
                remaining &= ~region
            remaining &= ~partial_scope['hold']
            antenna_proof=dict(method='observed_source_pairs_partial_scope',
                strict_endpoint_evidence_complete=False,classification_complete=False,
                source_report=str(partial_side_evidence))
        else:
            if landmarks is not None:
                cfg=landmarks[face]
                if cfg['source_sha256']!=digest(source):raise ValueError('Material landmarks belong to another source')
                lines=np.asarray(cfg['source_lines'])[cfg['antenna_indices']]
                rows=np.median(lines[:,int(w*.4):int(w*.6)],axis=1)
                # Revalidate with independently detected source seeds. Feeding the
                # already-smoothed curves back as new seeds can select a neighboring
                # peak and falsely reject this detector's own frozen proposal.
                detection=image
                if cfg.get('antenna_source_evidence'):
                    from .source_antenna_evidence import load
                    detection=load(cfg['antenna_source_evidence'],source,face)
                from .auto_structure import frozen_source_rows
                source_rows=frozen_source_rows(detection,[int(w*.4),int(w*.6)],cfg)
                supported,boundary_proof=per_column_boundaries(detection,source_rows,return_proof=True)
                if np.max(np.abs(lines-supported))>2.5:
                    raise ValueError('Frozen antenna boundaries disagree with paired source material edges; regenerate geometry landmarks')
                antenna_proof=dict(cfg['source_detection'],shared_geometry_landmarks=True,boundary_detection=boundary_proof)
            else:
                rows,signs,antenna_proof=antenna_rows(image,[int(w*.4),int(w*.6)],require_full_span=True)
                lines,boundary_proof=per_column_boundaries(image,rows,return_proof=True)
                antenna_proof=dict(antenna_proof,boundary_detection=boundary_proof)
            yy=np.arange(h)[:,None]
            for i in range(2):
                region=(yy>=lines[2*i])&(yy<lines[2*i+1])&remaining
                materials.append(('antenna_'+str(i),region,[max(6,(rows[2*i+1]-rows[2*i])*.30),max(8,w*.20)]))
                remaining &= ~region
        buttons,boxes=hardware_ranges(image);proof.update(antenna=antenna_proof,hardware_boxes=boxes)
        # Structural locking boxes intentionally include weak edge fragments;
        # they must never be used as material labels or independent color islands.
        from .side_material import hardware_materials
        components=hardware_materials(image,face);hw=np.zeros((h,w),bool)
        proof['hardware_materials']=[]
        for index,(region,component_proof) in enumerate(components):
            if np.any(region&~remaining):raise ValueError('Hardware material crosses antenna/paper')
            name=f'hardware_{index}';hw |= region
            materials.append((name,region,[max(6,h*.012),max(2,w*.08)]))
            # Fit each component from its own interior; a groove or a separate
            # dark control cannot determine another button's illumination.
            depth=ndimage.distance_transform_edt(region)
            fit_masks[name]=region&(depth>max(3.,w*.045))
            proof['hardware_materials'].append(dict(name=name,**component_proof))
        remaining &= ~hw
        fit_masks['main_surface']=remaining&~ndimage.binary_dilation(hw,iterations=max(2,int(w*.025)))
    if face=='back':
        # Camera glass/flash components are distinct from aluminum and do not
        # enter metal fitting. Shape + position are required, not darkness alone.
        glass,glass_proof=rear_camera_glass(a,remaining)
        remaining &= ~glass;proof['camera_glass_pixels']=int(glass.sum())
        proof['camera_glass_source_separation']=glass_proof
        body_contour,body_proof=rear_body_contour(image)
        body_path=save_mask(folder/'body-contour.png',body_contour);proof['rear_body_contour']=body_proof
        distance=ndimage.distance_transform_edt(np.pad(a[...,3]>=128,1))[1:-1,1:-1];rim=remaining&(distance<=max(8,w*.035))
        # Rear plastic antenna bars must never enter the aluminum illumination
        # field. Reuse complete frozen source evidence when available; otherwise
        # classify from this rear image's existing three-window consensus.
        # This material-only path does not certify a side antenna or alignment.
        complete_landmarks=landmarks is not None and all(side in landmarks for side in ('left','right'))
        if complete_landmarks and companions is None:
            raise ValueError('Current same-SKU geometry companions required for rear material classification')
        middle=a[int(h*.25):int(h*.75),:,3]>=128
        left=int(np.median(np.argmax(middle,axis=1)));right=int(np.median(w-1-np.argmax(middle[:,::-1],axis=1)))
        rear_proof={};yy=np.arange(h)[:,None];xx=np.arange(w)[None,:]
        for side in ('left','right'):
            if complete_landmarks:
                side_image=Image.open(companions[side]).convert('RGBA');sw=side_image.width
                side_cfg=landmarks[side]
                if side_cfg['source_sha256']!=digest(companions[side]):raise ValueError('Rear companion landmarks belong to another side source')
                detection=side_image
                if side_cfg.get('antenna_source_evidence'):
                    from .source_antenna_evidence import load
                    detection=load(side_cfg['antenna_source_evidence'],companions[side],side)
                from .auto_structure import frozen_source_rows
                source_rows=frozen_source_rows(detection,[int(sw*.4),int(sw*.6)],side_cfg)
                supported,_=per_column_boundaries(detection,source_rows,return_proof=True)
                frozen=np.asarray(side_cfg['source_lines'])[side_cfg['antenna_indices']]
                if np.max(np.abs(frozen-supported))>2.5:
                    raise ValueError('Rear companion frozen boundaries disagree with source evidence')
                ref=side_cfg['back']
                if ref['sha256']!=digest(source):raise ValueError('Rear material landmarks belong to another source')
                br,xr,brproof=ref['antenna_rows'],ref['x_range'],dict(ref['source_detection'],shared_geometry_landmarks=True)
            else:
                br,xr,brproof=_independent_rear_material_reference(
                    image,side,right if side=='left' else left,source)
            side_columns=(xx>w/2) if side=='left' else (xx<w/2)
            for i in range(2):
                region=rim&side_columns&(yy>=br[2*i])&(yy<br[2*i+1])
                if np.count_nonzero(region&(a[...,3]>=240))<20:raise ValueError('Rear plastic strip has insufficient independent material samples')
                materials.append((f'rear_antenna_{side}_{i}',region,[max(6,(br[2*i+1]-br[2*i])*.5),max(8,w*.006)]))
                remaining &= ~region;rim &= ~region
            rear_proof[side]=dict(antenna_rows=br,signal_columns=xr,measurement=brproof,
                rear_source_sha256=digest(source),shared_geometry_landmarks=complete_landmarks)
            if complete_landmarks:
                rear_proof[side]['side_geometry_sha256']=digest(companions[side])
        proof['rear_plastic_materials']=rear_proof
        if rim.sum()>=30:materials.append(('rear_perimeter',rim,[max(6,h*.012),max(2,w*.004)]));remaining &= ~rim
        panel,panel_proof=rear_panel(image);proof['rear_panel']=panel_proof
        camera=remaining&~panel
        if camera.sum()>=30:materials.append(('camera_surround',camera,[max(6,h*.015),max(6,w*.02)]));remaining &= ~camera
    if remaining.sum()>=30:
        materials.append(('main_surface',remaining,[max(3,h*.018),max(3,w*.12)] if face in ('left','right','top','bottom') else [max(8,h*.015),max(8,w*.025)]))
    if partial_scope is not None:
        for name,region,_ in materials:
            fit_masks[name]=fit_masks.get(name,region)&~partial_scope['fit_exclusion']
    label_components=[]
    if protection_policy=='none' and label_fit.any():
        from .lighting import engine
        render=engine('aluminum');revised=[]
        for name,region,sigma in materials:
            source_fit=fit_masks.get(name,region)&~label_fit&(a[...,3]>=240)
            label_part=region&label_fit
            # Previously protected paper can be a thin continuation of a
            # Gaussian material field with no local metal fit. Give only such
            # source-label parts their own source-sampled illumination domain;
            # do not lower support or insert a median/nearest-pixel field.
            unsupported=region&(render.gaussian(source_fit.astype(float),sigma)<.008)
            if (name!='rear_perimeter' and np.any(unsupported&label_part)
                    and np.count_nonzero(label_part&(a[...,3]>=240))>=20):
                key='label_component_'+str(len(label_components));revised.append((name,region&~label_part,sigma))
                label_components.append((key,label_part,sigma));fit_masks[key]=label_part
                if name in fit_masks:fit_masks[name]&=~label_part
            else:revised.append((name,region,sigma))
        materials=[item for item in revised if item[1].any()]+label_components
    domains=[];allowed=np.zeros((h,w),bool)
    for name,region,sigma in materials:
        if np.count_nonzero(region&(a[...,3]>=240))<20:raise ValueError('Material region has no reliable fit samples: '+name)
        reference_region=fit_masks.get(name,region)
        if protection_policy=='none' and not name.startswith('label_component_'):
            reference_region=reference_region&~label_fit
            fit_masks[name]=reference_region
        ref=reference_box(a,reference_region);mp=save_mask(folder/(name+'.png'),region);allowed|=region
        domains.append(dict(name=name,mask=mp,reference_boxes=[ref],sigma=sigma,
            strength=[.72,0,0],feather=0,correction_smooth=1.5))
        if name in fit_masks:
            domains[-1]['fit_mask']=save_mask(folder/(name+'-fit.png'),fit_masks[name])
        if face in ('left','right') and name=='main_surface':
            # The downward-facing long rail has a much narrower illumination
            # scale than the full strip. A broad normal fit leaves a dark rail
            # beside a bright halo. Fit along the rail over a longer span and
            # across it over >=6 pixels; only the additive L field is changed.
            # The normal floor retains independently tested fine scratches,
            # while the larger lift budget can recover deep source shadows.
            domains[-1].update(sigma=[max(36,h*.03),max(6,w*.05)],
                strength=[.9,0,0],max_lab_shift=[60,0,0],
                gamut_continuity_follow_field=True,
                gamut_continuity_follow_source_edges=True,
                gamut_continuity_source_edge_guard=max(12,w*.1))
        if face in ('left','right') and name.startswith('antenna_'):
            # The same narrow cast shadow crosses the plastic strip. Fit its
            # own material/reference at the rail's normal scale too; leaving
            # a broad fit here can reverse a valid source endpoint contrast
            # once adjacent aluminum is correctly illuminated.
            domains[-1].update(sigma=[sigma[0],max(6,w*.05)],
                               max_lab_shift=[60,0,0])
        if face in ('left','right','top','bottom') and (name=='main_surface' or name.startswith('hardware_')):
            # Final gamut clipping can flatten bright texture on every frame
            # face. Propagate actual gamut caps through the correction field,
            # never smooth source RGB or let paper seed the metal correction.
            domains[-1].update(gamut_continuity_step_L=(.2 if face in ('top','bottom') else .6),
                              gamut_continuity_opaque_seeds=True)
        if face=='back' and name in ('camera_surround','main_surface'):
            # Saturated source highlights can make the final per-pixel gamut
            # limiter discontinuous well away from a material-domain seam.
            domains[-1]['gamut_continuity_step_L']=REAR_GAMUT_CONTINUITY_STEP_L
        if name=='rear_perimeter' or (face=='front' and name=='main_surface'):
            domains[-1].update(field='contour',sigma=[max(18,h*.012),max(5,w*.006)])
        if name=='rear_perimeter':domains[-1].update(reference_profile='contour_depth_lift',strength=[.92,0,0],max_lab_shift=[48,0,0],contour_alpha_mask=body_path,source_body_contour=body_proof,
            correction_smoothing='source_fit_coverage')
        if face=='front':
            reliable=region&(a[...,3]>=240)
            shoulder=float(ndimage.distance_transform_edt(~reliable)[region&(a[...,3]>=128)].max())
            if shoulder>max(8,w*.045):raise ValueError('Front rim lacks nearby opaque material support')
            domains[-1].update(reference_profile='contour_depth',contour_source_inset_limit=max(8.,w*.045),
                sigma=[max(18,h*.012),2.])
            proof['front_opaque_shoulder_px']=shoulder
    if not domains:raise ValueError('No reliable material domains')
    required_pixels=allowed.copy()
    if face=='back':
        # Independent source silhouette coverage must not inherit holes from
        # the material classifier being checked. Both metal and plastic at the
        # outer rim are editable; only actual paper is protected.
        required_pixels |= visible&~paper&(distance<=max(8,w*.035))
    required=save_mask(folder/'required-editable.png',required_pixels)
    view=dict(name=face,source=str(Path(source).resolve()),source_sha256=digest(source),
        protection_policy=protection_policy,sticker_mask=paper_path,protected_mask=paper_path,
        required_editable_mask=required,reference_boxes=domains[0]['reference_boxes'],domains=domains)
    if partial_scope is not None:
        from .side_candidate_scope import write_scope
        scope_record=write_scope(partial_scope,folder/'partial-side-scope')
        view['partial_side_lighting_scope']=scope_record
        view['required_processing_finalize']='partial_side_lighting.finalize_candidate_then_independent_checks'
        proof['partial_side_lighting_scope']=scope_record
        proof['full_material_classification_verified']=False
    if protection_policy=='none':
        label_path=save_mask(folder/'source-label-fit-exclusion.png',label_fit)
        view['source_label_fit_exclusion_mask']=label_path
        proof['protection_policy']='none';proof['protected_zone_pixels']=0
        proof['source_label_fit_exclusion']=dict(path=label_path,pixels=int(label_fit.sum()),source_analysis=label_fit_proof,
            editable_label_pixels=int((label_fit&allowed).sum()),hard_pixel_protection=False)
        if label_hint is not None:
            view['source_label_location_hint_mask']=save_mask(folder/'source-label-location-hint.png',label_hint)
            view['source_label_material_evidence']=label_fit_proof
            proof['released_label_hint_metal_pixels']=int((label_hint&~label_fit).sum())
    if face=='back':
        view['correction_blends']=[dict(domains=['rear_perimeter','camera_surround','main_surface'],
            boundary_domain='rear_perimeter',sigma=max(3.,w*.008))]
    if face in ('left','right'):
        # Source photo edges contain mixed-material sensor pixels. Blend only
        # their additive fields, keeping both materials' independent targets
        # and every original RGB detail; do not blur the resulting photograph.
        view['correction_blends']=[dict(domains=[name,'main_surface'],
            boundary_domain=name,sigma=1.25,kind='source_material_transition')
            for name in ('antenna_0','antenna_1')]
        view['antenna_boundary_bands']=[0,1]
    if label_components:
        masks={d['name']:np.array(Image.open(d['mask']).convert('L'))>0 for d in domains}
        for key,region,_ in label_components:
            for other,other_mask in masks.items():
                if other==key or other.startswith('label_component_'):continue
                if np.any(ndimage.binary_dilation(region)&other_mask):
                    view.setdefault('correction_blends',[]).append(dict(domains=[key,other],boundary_domain=key,sigma=1.25,kind='source_material_transition'))
        proof['independent_label_illumination_components']=[dict(name=name,pixels=int(region.sum()),source_fit_support_threshold=.008,hard_pixel_protection=False) for name,region,_ in label_components]
    if face=='back':
        perimeter=next(d for d in domains if d['name']=='rear_perimeter')
        contour=perimeter.get('source_body_contour',{})
        region=np.zeros((h,w),bool);metal=np.array(Image.open(perimeter['mask']).convert('L'))>0
        for side,entry in contour.get('sides',{}).items():
            for lo,hi in entry.get('protrusion_rows',[]):
                witness=metal&(a[...,3]>=240);padding=max(20,int(h*.015))
                witness[:max(0,lo-padding)]=False;witness[min(h,hi+padding):]=False
                if side=='left':witness[:,w//2:]=False
                else:witness[:,:w//2]=False
                region|=witness
        if region.any():
            view['quantization_continuity']=[dict(mask=save_mask(folder/'source-button-quantization.png',region),max_added_step_L=1.25,
                basis='Current source protrusions and opaque rear metal; adjacent RGB code values only; fitted field unchanged')]
    job=dict(color_mode='source_lightness',defaults=dict(max_lab_shift=[35,0,0],feather=0),views=[view])
    proof.update(materials=[dict(name=d['name'],reference_boxes=d['reference_boxes'],mask=d['mask']) for d in domains],source_sha256=digest(source),source=str(source))
    return job,proof,allowed,paper



def lighting_scope(source, face, job, folder):
    from .auto_material import lighting_scope as ordinary_scope
    scope = ordinary_scope(source, face, job, folder)
    partial = job['views'][0].get('partial_side_lighting_scope')
    if partial is None:
        raise ValueError('Side review scope requires its bound partial preparation')
    scope.update(partial_side_lighting_scope=partial, requires_partial_scope_finalization=True,
        material_classification_complete=False, structural_alignment_certified=False,
        unknown_endpoint_pixels_are_not_effect_witnesses=True)
    return scope
