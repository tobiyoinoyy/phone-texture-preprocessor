"""Current source observations for explicit review candidates.

Partial observations preserve the original strict error. Missing pairs and
shape diagnostics are never upgraded to confirmed material/endpoint evidence.
"""
import math
import numpy as np
from scipy.ndimage import gaussian_filter1d
from . import antenna_material as material
from .side_candidate_structure_math import require

def reference_contract(reference, geometry):
    """Keep existing canvas coordinates; never manufacture a source-rail result.

    The own-corner branch is an explicit image-coordinate assumption. It does
    not map the corner plane, certify the duplicated rails, or change unknown.
    Quality numbers below are the existing recorded own-corner acceptance
    values (auto_geometry.py), not new relaxed material/reference thresholds.
    """
    require(reference.get('selected_coordinate_mode') == 'canvas_pixel_centers', 'Unsupported reference coordinate mode')
    applicability = reference.get('applicability', {})
    status = reference.get('status')
    require(applicability.get('status', status) == status, 'Reference/applicability status differs')
    base = dict(coordinate_mode='canvas_pixel_centers', original_status=status,
        original_applicable=applicability.get('applicable'), reference_review_required=True,
        reference_certified=False, independent_bottom_rails_certified=False,
        new_remote_tail_measurement_performed=False, geometry_changed=False,
        targets_basis='same immutable back antenna rows normalized by actual canvas height')
    if status == 'no_in_shaft_remote_extension_detected':
        require(applicability.get('applicable') is False, 'Reference status/applicability disagree')
        return dict(base, mode='existing_canvas_reference_review', original_unknown_preserved=False)
    require(status == 'source_tail_applicability_unknown' and applicability.get('applicable') is None,
            'Other reference state cannot enter this review branch')
    require(applicability.get('reason') == 'Exposed two-flank rail evidence required; unsupported geometry branch',
            'Unknown reference has an unsupported reason')
    require(geometry.get('method') == 'source_exposed_plate_rgb_alpha_consensus_v1' and geometry.get('clear_bands') is None,
            'Unknown reference is not the recorded own-corner geometry branch')
    corner = geometry.get('own_corner_evidence') or {}
    require(corner.get('method') == 'own_top_bottom_corner_curve_consensus_v1' and
            corner.get('symmetric_body_assumption') is True,
            'Explicit own-corner symmetry evidence required')
    supports = corner.get('own_corner_support_fractions', [])
    require(len(supports) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) and .6 <= v <= 1 for v in supports),
            'Recorded two-corner support is insufficient')
    residual = corner.get('corner_residual_p95_px'); raw_support = corner.get('raw_rgb_supported_fraction')
    require(isinstance(residual, (int, float)) and math.isfinite(residual) and 0 <= residual <= 3 and
            isinstance(raw_support, (int, float)) and math.isfinite(raw_support) and .6 <= raw_support <= 1,
            'Recorded corner residual/raw RGB support differs from original requirements')
    require(type(corner.get('samples')) is int and corner['samples'] > 0 and
            geometry.get('axis_evidence', {}).get('method') == 'independent_long_side_axis_v1',
            'Bound own-corner/axis record incomplete')
    rail = geometry.get('bottom_rail')
    require(isinstance(rail, list) and len(rail) == 2 and all(math.isfinite(v) for v in rail) and
            rail == geometry.get('alpha_lower_rail'), 'Expected the explicitly duplicated own-corner plane record')
    for name in ('source_size', 'output_size'):
        values = geometry.get(name)
        require(isinstance(values, list) and len(values) == 2 and all(type(v) is int and v > 0 for v in values),
                'Missing geometry canvas dimensions')
    angle = geometry.get('rotation_ccw_degrees')
    require(isinstance(angle, (int, float)) and not isinstance(angle, bool) and math.isfinite(angle),
            'Missing finite geometry rotation angle')
    crop = geometry.get('canvas_trim_xyxy'); visible = geometry.get('visible_canvas_trim', {})
    vcrop = visible.get('crop_xyxy')
    require(isinstance(crop, list) and len(crop) == 4 and all(type(v) is int for v in crop) and
            0 <= crop[0] < crop[2] and 0 <= crop[1] < crop[3] and
            isinstance(vcrop, list) and len(vcrop) == 4 and all(type(v) is int for v in vcrop) and
            0 <= vcrop[0] < vcrop[2] and 0 <= vcrop[1] < vcrop[3] and
            visible.get('output_size') == geometry['output_size'], 'Recorded crop chain incomplete')
    intermediate_size = [crop[2] - crop[0], crop[3] - crop[1]]
    require(visible.get('source_size') == intermediate_size and vcrop[2] <= intermediate_size[0] and
            vcrop[3] <= intermediate_size[1] and [vcrop[2] - vcrop[0], vcrop[3] - vcrop[1]] == geometry['output_size'],
            'Recorded first/final crop sizes disagree')
    return dict(base, mode='existing_canvas_own_corner_assumption_review', original_unknown_preserved=True,
        symmetric_body_assumption=True, duplicated_rail_is_not_independent_evidence=True,
        corner_plane_used_to_generate_targets=False,
        limitation='New source-only physical reference would require fresh branch-aware raw witnesses; this candidate only reuses the existing canvas convention')

def groups(mask):
    result = []
    for x in np.flatnonzero(mask).tolist():
        if not result or x != result[-1][-1]+1: result.append([])
        result[-1].append(x)
    return result

def trace_band(image, rows, band):
    observed = {}
    def capture(index, chosen, basis):
        observed.update(chosen=chosen.copy(), basis=basis.copy())
    result = dict(band=band, seeds=list(rows), selected_band_seeds=list(rows[2*band:2*band+2]),
        actual_gate_passed=False, observer_reached=False,
        prediction_is_source_evidence=False, shape_statistics_are_diagnostic=True)
    try:
        _, proof = material.trace_boundaries(image, rows, bands=(band,), _observer=capture)
        result.update(actual_gate_passed=True, actual_gate_proof=proof)
    except ValueError as error:
        result.update(error=str(error), error_type=type(error).__name__)
    if not observed:
        result['quantification_unavailable_reason'] = 'Existing gate failed before observer/central pair fit'
        return result, None
    result['observer_reached'] = True
    a = np.asarray(image.convert('RGBA')); h,w = a.shape[:2]; xx = np.arange(w)
    chosen,basis = observed['chosen'],observed['basis']
    reliable = np.isfinite(chosen).all(axis=0)
    edge_alpha = a[np.clip(np.rint(basis).astype(int),0,h-1),xx,3]
    opaque_both = np.min(edge_alpha,axis=0)>=240
    opaque_any = np.any(edge_alpha>=240,axis=0)
    terminal = ((xx<xx[reliable].min())|(xx>xx[reliable].max())) if reliable.any() else np.ones(w,bool)
    missing_both = ~reliable&opaque_both
    missing_terminal = ~reliable&terminal&opaque_any
    rgb = gaussian_filter1d(a[...,:3].astype(float),1.,axis=0)
    gray = gaussian_filter1d(a[...,:3].astype(float).mean(axis=2),1.,axis=0)
    gradient = np.gradient(gray,axis=0)
    z = np.linspace(-.5,1.5,41); template = ((z>0)&(z<1)).astype(float); template-=template.mean()
    columns = []
    for col in range(w):
        value = dict(column=col, predicted_pair=basis[:,col], measured_pair=chosen[:,col],
            paired_source_supported=bool(reliable[col]), basis_alpha=edge_alpha[:,col],
            opaque_both_edges=bool(opaque_both[col]), opaque_any_edge=bool(opaque_any[col]),
            missing_opaque_terminal=bool(missing_terminal[col]),
            source_support_claim=bool(reliable[col]), prediction_only=not bool(reliable[col]))
        if reliable[col]:
            lo,hi = chosen[:,col]; ys = lo+z*(hi-lo)
            sample = np.array([np.interp(ys,np.arange(h),rgb[:,col,k]) for k in range(3)]).T
            outer = (z<-.15)|(z>1.15)
            sample-=np.column_stack([np.polyval(np.polyfit(z[outer],sample[outer,k],1),z) for k in range(3)])
            sample-=sample.mean(axis=0)
            value.update(chosen_alpha=a[np.rint(chosen[:,col]).astype(int),col,3],
                correlation=float(np.linalg.norm(sample.T@template)/(np.linalg.norm(sample)*np.linalg.norm(template)+1e-12)),
                contrast_rgb_rms=float(np.min(np.linalg.norm(material._material_vectors(rgb,chosen[:,col],col),axis=-1))/np.sqrt(3)),
                gradient_strength=float(np.min(np.abs(gradient[np.rint(chosen[:,col]).astype(int),col]))))
        else:
            value['shape_correlation_contribution'] = 0. if opaque_both[col] else None
        columns.append(value)
    shapes = []
    for end, indexes in [('left',range(8)),('right',range(w-8,w))]:
        values = [columns[x] for x in indexes]
        corr = [v['correlation'] if v['paired_source_supported'] else 0. for v in values
                if v['paired_source_supported'] or v['opaque_both_edges']]
        contrast = [v['contrast_rgb_rms'] for v in values if v['paired_source_supported']]
        strength = [v['gradient_strength'] for v in values if v['paired_source_supported']]
        median = float(np.median(corr)) if corr else 0.
        c = float(np.median(contrast)) if contrast else 0.
        g = float(np.median(strength)) if strength else 0.
        shapes.append(dict(end=end,columns=list(indexes),correlation_values=corr,
            median=median,contrast_rgb_rms=c,gradient_strength=g,
            would_reject=(c<6 or g<2) and median<.65,
            not_actual_gate_proof_when_earlier_gate_failed=True))
    center = (xx>=int(w*.2))&(xx<int(w*.8))
    measured_center = center&reliable
    residual = np.abs(chosen[:,measured_center]-basis[:,measured_center])
    result.update(width=w,height=h,columns=columns,endpoint_shape_diagnostic=shapes,
        support_columns=np.flatnonzero(reliable),supported_count=int(reliable.sum()),
        unsupported_columns=np.flatnonzero(~reliable),unsupported_width_ratio=float((~reliable).sum()/w),
        opaque_missing_columns=np.flatnonzero(missing_both),opaque_missing_count=int(missing_both.sum()),
        opaque_missing_width_ratio=float(missing_both.sum()/w),
        opaque_missing_fraction_of_opaque_columns=float(missing_both.sum()/max(1,opaque_both.sum())),
        terminal_missing_columns=np.flatnonzero(missing_terminal),terminal_missing_count=int(missing_terminal.sum()),
        terminal_missing_width_ratio=float(missing_terminal.sum()/w),
        internal_missing_components=groups(missing_both&~terminal),
        all_opaque_missing_components=groups(missing_both),
        central_fit=dict(search_center_columns=[int(w*.2),int(w*.8)],
            observed_center_pairs=int(measured_center.sum()),
            measured_pair_to_basis_residual_p95_px=float(np.percentile(residual,95)) if residual.size else None,
            polynomial_degree=2,statistics_scope='Post-pair diagnostic residual; not internal fitter inlier proof'),
        small_missing_span_review_opportunity=0<int(missing_both.sum())<=3,
        observer_precedes_lower_band_phase_refinement=True)
    return result, observed
