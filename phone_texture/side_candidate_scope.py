"""Current-source partial material scopes for the explicit review route.

These masks only prepare the shared-field proposal. The old hold/taper is never
applied to the output; strict material/endpoint completeness remains false.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt


POLICY = 'source_bound_partial_side_lighting_review_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError('Partial side scope: '+message)


def build_scope(source, bands, *, face, source_file_sha256, taper_px=12.):
    """Create masks from observed pairs; never interpolate missing columns.

    taper_px is a proposal parameter, not a relaxed acceptance threshold.
    The final candidate still needs the original source-relative continuity
    checks. A smooth weight alone does not certify a visually smooth result.
    """
    require(face in ('left', 'right'), 'only independent side faces are eligible')
    require(source.dtype == np.uint8 and source.ndim == 3 and source.shape[2] == 4,
            'source must be unchanged 8-bit RGBA')
    require(np.isfinite(taper_px) and taper_px > 0, 'invalid transition radius')
    h,w = source.shape[:2]; visible = source[...,3] > 0
    require(set(bands) == {'0','1'}, 'both observed antenna bands are required')
    yy = np.arange(h)[:,None]
    hold = np.zeros((h,w),bool); antennas = {}; records = []
    for index in (0,1):
        evidence = bands[str(index)]['original_RGB_geometry_seeds']
        require(evidence.get('observer_reached') is True, 'no observed central/pair evidence')
        require(evidence.get('width') == w and evidence.get('height') == h,
                'evidence canvas differs')
        columns = evidence['columns']
        require(len(columns) == w and [v['column'] for v in columns] == list(range(w)),
                'missing or reordered column evidence')
        seeds = evidence['selected_band_seeds']; lo,hi = map(float,seeds)
        require(np.isfinite([lo,hi]).all() and 3 < hi-lo < h*.1, 'invalid source central band')
        # Same broad source-local analysis envelope as terminal normalization.
        # No claim is made about the true endpoint boundary within this window.
        margin = 3*(hi-lo)
        y0,y1 = max(0,int(np.floor(lo-margin))),min(h,int(np.ceil(hi+margin))+1)
        unknown = np.array([not v.get('paired_source_supported',False) for v in columns])
        for end in evidence.get('endpoint_shape_diagnostic',[]):
            if end.get('would_reject'):
                unknown[np.asarray(end['columns'],dtype=int)] = True
        region = np.zeros((h,w),bool)
        for col,value in enumerate(columns):
            if unknown[col]:
                continue
            pair = np.asarray(value['measured_pair'],float)
            require(pair.shape == (2,) and np.isfinite(pair).all() and pair[1] > pair[0]+2,
                    'supported column has no measured pair')
            require(value.get('prediction_only') is False, 'prediction cannot label a material')
            require(y0 <= pair[0] < pair[1] < y1, 'measured pair escapes declared local scope')
            region[:,col] = (np.arange(h)>=pair[0])&(np.arange(h)<pair[1])
        local_hold = (yy>=y0)&(yy<y1)&unknown[None,:]&visible
        hold |= local_hold
        antennas['antenna_'+str(index)] = region&visible&~local_hold
        records.append(dict(band=index,source_seeds=seeds,local_window_yxy=[y0,y1],
            retained_columns=np.flatnonzero(unknown).tolist(),retained_pixels=int(local_hold.sum()),
            window_is_processing_scope_not_material_boundary=True,
            missing_pairs_interpolated=False,actual_strict_gate_passed=evidence['actual_gate_passed']))
    require(hold.any(), 'this route is for explicitly unresolved source pixels')
    require(not np.any(antennas['antenna_0']&antennas['antenna_1']), 'antenna domains overlap')
    distance = distance_transform_edt(~hold)
    t = np.clip(distance/float(taper_px),0,1)
    weight = t*t*(3-2*t)
    weight[hold] = 0.
    # Fit exclusion is separate from output retention: only hold is exact-zero,
    # its surrounding halo has a smooth correction taper and independent fits.
    fit_exclusion = (distance < taper_px)&visible
    metadata = dict(policy=POLICY,face=face,source_file_sha256=source_file_sha256,
        source_RGBA_sha256=array_sha(source),source_size=[w,h],bands=records,
        unknown_pixels=int(hold.sum()),fit_excluded_pixels=int(fit_exclusion.sum()),
        taper_px=float(taper_px),taper='2D distance smoothstep applied to RGB correction only',
        hold_mask_sha256=array_sha(hold),fit_exclusion_sha256=array_sha(fit_exclusion),
        correction_weight_sha256=array_sha(weight),
        classification_complete=False,material_complete=False,
        strict_endpoint_evidence_complete=False,structural_alignment_applied=False,
        lighting_status='scoped_candidate_needs_review',quality_checks_complete=False,
        visual_approved=False,model_uv_verified=False,
        full_height_column_retention=False,requires_post_render_finalization=True)
    return dict(hold=hold,fit_exclusion=fit_exclusion,weight=weight,
                antenna_masks=antennas,metadata=metadata)


def from_bound_report(source_path, face, report_path, *, taper_px=12.):
    """Consume only this invocation's source-bound observation schema."""
    from PIL import Image
    source_path, report_path = Path(source_path).resolve(), Path(report_path).resolve()
    report = json.loads(report_path.read_text())
    require(report.get('schema') == 'phone_texture_current_side_observation/v1', 'current observation schema required')
    require(report.get('face') == face, 'observation face differs')
    require(report.get('processing_state') == 'completed_diagnostic_only' and
            report.get('observations_computed_from_current_source') is True and
            report.get('historical_observations_reused') is False and report.get('hard_failures') == [],
            'current observation is incomplete or reuses historical evidence')
    require(report.get('inputs_unchanged') is True and report.get('code_unchanged') is True,
            'observation did not finish with intact provenance')
    context = report['source_context']
    require(context['face'] == face and sha(source_path) == context['geometry_sha256'] == report['source_sha256'],
            'source/face differs from observed geometry')
    bindings = dict(report['input_sha256'])
    for name, digest in report['code_sha256'].items():
        require(name not in bindings or bindings[name] == digest, 'conflicting observation binding')
        bindings[name] = digest
    for name, digest in [(context['raw_source'], context['raw_source_sha256']),
                         (context['path'], context['sha256'])]:
        require(str(Path(name).resolve()) in bindings and bindings[str(Path(name).resolve())] == digest,
                'raw observation source lacks matching binding')
    require(str(source_path) in bindings and bindings[str(source_path)] == sha(source_path),
            'geometry lacks matching observation binding')
    require(bool(report['code_sha256']), 'observation implementation binding missing')
    for name, digest in bindings.items():
        require(isinstance(digest, str) and len(digest) == 64 and sha(name) == digest,
                'observation input or code changed: '+name)
    with Image.open(source_path) as image:
        require(image.mode == 'RGBA' and image.format == 'PNG', 'geometry must remain RGBA PNG')
        source = np.asarray(image).copy()
    result = build_scope(source, report['bands'], face=face, source_file_sha256=sha(source_path), taper_px=taper_px)
    result['metadata'].update(evidence_report=str(report_path), evidence_report_sha256=sha(report_path),
        raw_source_sha256=context['raw_source_sha256'], current_observation_schema=report['schema'],
        historical_observation_reused=False)
    return result


def write_scope(scope,folder):
    """Persist provenance/masks in a NEW job-assets directory only."""
    from PIL import Image
    folder = Path(folder); folder.mkdir(parents=True,exist_ok=False)
    files = {}
    for name,mask in [('retained-unknown',scope['hold']),('fit-exclusion',scope['fit_exclusion'])]:
        path=folder/(name+'.png'); Image.fromarray(mask.astype('uint8')*255).save(path)
        files[name]=dict(path=str(path.resolve()),sha256=sha(path))
    path=folder/'correction-weight.npy'; np.save(path,scope['weight'],allow_pickle=False)
    files['correction-weight']=dict(path=str(path.resolve()),sha256=sha(path))
    value=dict(scope['metadata'],assets=files)
    (folder/'scope.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    return value

