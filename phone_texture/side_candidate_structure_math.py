"""Generic review-only band transport, installed with the current package.

Numerical construction retained from the isolated v1/v3 candidates. No sample
IDs, historical paths, dynamic imports, image work or source lookup at import.
"""
import math

MAPPING_POLICY = 'integer_band_center_translation_fixed_alpha_review_v1'
POLICY = 'row_alpha_mass_center_contour_sampling_review_v3'

class ReviewRejected(ValueError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence or {}

def require(value, message, evidence=None):
    if not value:
        raise ReviewRejected(message, evidence)

def build_mapping(height, width, bands, middle):
    """A forward source-y -> destination-y map with analytic C1/monotonic proof.

    Each core is measured-edge extent plus one observed central band width.
    The margin is processing scope, not inferred endpoint/material evidence.
    No window-size or displacement retry is performed.
    """
    require(type(height) is int and type(width) is int and height > 2 and width > 0,
            'Invalid canvas dimensions')
    require(len(bands) == 2 and len(middle) == 2, 'Two bands and one middle reservation required')
    require(all(type(v) is int for v in middle) and 0 < middle[0] < middle[1] < height - 1,
            'Middle reservation must be ordered source rows')
    cap = max(8, int(math.ceil(width * .75)))
    require(cap < middle[0] < middle[1] < height - 1 - cap, 'No space outside fixed caps/middle')
    pieces = []
    for index, item in enumerate(bands):
        values = item['source_pair'] + item['observed_bounds'] + item['target_pair']
        require(len(values) == 6 and all(math.isfinite(float(v)) for v in values), 'Nonfinite band measurement')
        u, v = map(float, item['source_pair'])
        p, q = map(float, item['target_pair'])
        observed_lo, observed_hi = map(float, item['observed_bounds'])
        require(observed_lo <= u < v <= observed_hi and p < q, 'Band measurement order differs')
        bandwidth = v - u
        lo = math.floor(observed_lo - bandwidth)
        hi = math.ceil(observed_hi + bandwidth)
        delta = int(round((p + q - u - v) / 2))
        a, d = (cap, middle[0]) if index == 0 else (middle[1], height - 1 - cap)
        require(a < lo < hi < d, 'Measured core overlaps a fixed cap or control range',
                dict(band=index, fixed_bounds=[a, d], core=[lo, hi]))
        # max d/dt smoothstep = 1.5; endpoint derivatives equal 1 exactly.
        left_extreme = 1 + 1.5 * delta / (lo - a)
        right_extreme = 1 - 1.5 * delta / (d - hi)
        minimum = min(1., left_extreme, right_extreme)
        maximum = max(1., left_extreme, right_extreme)
        require(minimum > 0, 'C1 mapping folds or has zero derivative; no parameter retry',
                dict(band=index, delta=delta, core=[lo, hi], fixed_bounds=[a, d],
                     analytic_min_jacobian=minimum))
        require(a < lo + delta < hi + delta < d, 'Moved rigid core enters a fixed range')
        pieces.append(dict(band=index, source_pair=[u, v], target_pair=[p, q],
            observed_bounds=[observed_lo, observed_hi], core=[lo, hi], fixed_bounds=[a, d],
            delta=delta, unrounded_delta=(p + q - u - v) / 2,
            target_center_quantization_error=delta - (p + q - u - v) / 2,
            core_margin_bandwidth=bandwidth, analytic_min_jacobian=minimum,
            analytic_max_jacobian=maximum, processing_core_is_material_evidence=False))
    return dict(policy=MAPPING_POLICY, height=height, width=width, bands=pieces, middle=list(middle),
                cap=cap, fixed_ranges=[[0, cap + 1], [middle[0], middle[1] + 1], [height - 1 - cap, height]],
                same_map_for_all_columns=True, alpha_policy='immutable_source_alpha',
                strict_endpoint_evidence_complete=False)

def forward_rows(y, plan):
    import numpy as np
    y = np.asarray(y, dtype=float)
    require(np.isfinite(y).all() and np.all((y >= 0) & (y <= plan['height'] - 1)), 'Forward rows outside canvas')
    out = y.copy()
    for item in plan['bands']:
        a, d = item['fixed_bounds']; lo, hi = item['core']; delta = item['delta']
        left = (y > a) & (y < lo)
        t = (y[left] - a) / (lo - a)
        out[left] += delta * t * t * (3 - 2 * t)
        core = (y >= lo) & (y <= hi)
        out[core] += delta
        right = (y > hi) & (y < d)
        t = (y[right] - hi) / (d - hi)
        out[right] += delta * (1 - t * t * (3 - 2 * t))
    return out

def inverse_rows(target_y, plan):
    """Unique inverse of the analytically monotonic forward map, not y-d(y)."""
    import numpy as np
    target = np.asarray(target_y, dtype=float)
    require(np.isfinite(target).all() and np.all((target >= 0) & (target <= plan['height'] - 1)),
            'Inverse targets outside canvas')
    lo = np.zeros_like(target); hi = np.full_like(target, plan['height'] - 1)
    for _ in range(60):
        center = (lo + hi) / 2
        below = forward_rows(center, plan) < target
        lo = np.where(below, center, lo); hi = np.where(below, hi, center)
    rows = (lo + hi) / 2
    exact = np.abs(rows - np.rint(rows)) < 1e-9
    rows = np.where(exact, np.rint(rows), rows)
    require(np.max(np.abs(forward_rows(rows, plan) - target), initial=0) < 1e-7,
            'Inverse solver did not reproduce the declared forward map')
    return rows

def shoulder_diagnostics(before, after, plan):
    """Report final saved-pixel statistics, never certify naturalness by proxy.

    Both original and displaced cores are excluded: a translated true material
    edge must not be called a lighting discontinuity. Shoulder texture/gradient
    changes remain diagnostics, with no relaxed or newly invented pass limit.
    """
    import numpy as np
    from .lighting_review import lightness
    original = lightness(before[..., :3]); final = lightness(after[..., :3])
    correction = final - original
    reports = []
    for band in plan['bands']:
        a, d = band['fixed_bounds']; lo, hi = band['core']; delta = band['delta']
        region = np.zeros(before.shape[:2], bool)
        region[a + 1:d] = True
        region[lo:hi + 1] = False
        region[lo + delta:hi + delta + 1] = False
        region &= before[..., 3] >= 240
        axes = {}
        for axis in (0, 1):
            pairs = region[1:] & region[:-1] if axis == 0 else region[:, 1:] & region[:, :-1]
            old = np.diff(original, axis=axis)[pairs]
            new = np.diff(final, axis=axis)[pairs]
            added = np.abs(np.diff(correction, axis=axis)[pairs])
            axes[str(axis)] = dict(pair_count=int(pairs.sum()),
                original_gradient_abs_p95=float(np.percentile(np.abs(old), 95)) if len(old) else None,
                final_gradient_abs_p95=float(np.percentile(np.abs(new), 95)) if len(new) else None,
                correction_gradient_max=float(added.max()) if len(added) else None,
                correction_gradient_p95=float(np.percentile(added, 95)) if len(added) else None,
                pairs_over_1_25_L_descriptive_only=int(np.sum(added > 1.25)),
                source_contrast_sign_reversals_above_1_25_L=int(np.sum((np.abs(old) > 1.25) & (old * new < 0))))
        reports.append(dict(band=band['band'], region_pixels=int(region.sum()), axes=axes))
    return dict(reports=reports, both_axes_measured=True, acceptance_threshold_applied=False,
                material_boundaries_excluded_by_processing_scope_not_certification=True,
                visual_approved=False, automatic_effect_pass=False)

def edge_uncertainty(alpha):
    """Four-connected alpha<255 components touching a canvas boundary.

    Interior alpha0 is never seeded. This is mask-confidence topology only,
    not a new assertion that pixels are phone foreground or a particular material.
    """
    import numpy as np
    from scipy.ndimage import binary_propagation, generate_binary_structure
    require(alpha.ndim == 2 and alpha.dtype == np.uint8 and min(alpha.shape) > 0,
            'Expected nonempty source uint8 alpha')
    uncertain = alpha < 255
    seeds = np.zeros_like(uncertain)
    seeds[0] = uncertain[0]; seeds[-1] = uncertain[-1]
    seeds[:, 0] = uncertain[:, 0]; seeds[:, -1] = uncertain[:, -1]
    return binary_propagation(seeds, structure=generate_binary_structure(2, 1), mask=uncertain)

def row_mass_coordinates(alpha_row):
    """Exclusive-prefix alpha-mass coordinates at positive pixel centers."""
    import numpy as np
    require(alpha_row.ndim == 1 and alpha_row.dtype == np.uint8, 'Expected a uint8 alpha row')
    x = np.flatnonzero(alpha_row > 0)
    require(len(x) > 0, 'Empty visible donor/target row', dict(visible_pixels=0))
    require(len(x) == int(x[-1] - x[0] + 1), 'Internal transparent hole or multiple visible segments',
            dict(first_positive=int(x[0]), last_positive=int(x[-1]), positive_columns=x.tolist()))
    mass = alpha_row[x].astype(float)
    u = (np.cumsum(mass) - .5 * mass) / float(mass.sum())
    require(np.isfinite(u).all() and u[0] > 0 and u[-1] < 1 and np.all(np.diff(u) > 0),
            'Invalid positive-row CDF centers')
    return x, u

def inverse_mass_coordinates(alpha_row, u):
    """Invert center nodes; explicit constant endpoint-pixel extension in [0,1].

    This never bridges an internal hole or repairs an empty row. Extension only
    reuses the observed first/last positive pixel within its outer half mass bin.
    """
    import numpy as np
    x, centers = row_mass_coordinates(alpha_row)
    values = np.asarray(u, dtype=float)
    require(np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)),
            'Normalized mass coordinate outside [0,1]', dict(values=values.tolist()))
    sampled = np.asarray(np.interp(values, centers, x.astype(float)), dtype=float)
    # Exact integer nodes should not accidentally activate an alpha0 neighbour.
    sampled = np.where(np.abs(sampled - np.rint(sampled)) < 1e-12, np.rint(sampled), sampled)
    lower = values < centers[0]; upper = values > centers[-1]
    evidence = dict(first_positive=int(x[0]), last_positive=int(x[-1]),
        lower_center_u=float(centers[0]), upper_center_u=float(centers[-1]),
        first_outer_half_bin_mass=float(centers[0]), last_outer_half_bin_mass=float(1 - centers[-1]),
        lower_extension_count=int(np.count_nonzero(lower)), upper_extension_count=int(np.count_nonzero(upper)),
        lower_u_range=[float(values[lower].min()), float(values[lower].max())] if np.any(lower) else None,
        upper_u_range=[float(values[upper].min()), float(values[upper].max())] if np.any(upper) else None,
        endpoint_extension='constant observed positive endpoint RGB; not hole filling or material evidence')
    return sampled, evidence

def remap_array(rgba, plan, *, return_transport=False):
    """Sample all positive xy weights inside each row's own contiguous contour.

    Longitudinal mapping is unchanged. Horizontal sampling changes with row
    coverage; this is not a 2-D rigid, injective or area-conserving transform.
    """
    import numpy as np
    require(rgba.dtype == np.uint8 and rgba.shape == (plan['height'], plan['width'], 4), 'RGBA dimensions/type differ')
    measurements = [{k: b[k] for k in ('source_pair', 'target_pair', 'observed_bounds')} for b in plan['bands']]
    require(build_mapping(plan['height'], plan['width'], measurements, plan['middle']) == plan, 'Longitudinal map derivation differs')
    h, w = rgba.shape[:2]; alpha = rgba[..., 3]
    y = np.arange(h, dtype=float); source_y = inverse_rows(y, plan)
    ylo = np.floor(source_y).astype(int); yhi = np.minimum(ylo + 1, h - 1); fy = source_y - ylo
    active_rows = np.abs(source_y - y) > 1e-9
    uncertainty = edge_uncertainty(alpha)
    output = rgba.copy()
    # Float64 coordinates preserve the actual sampling provenance in an NPZ.
    xlo_map = np.full((h, w), np.nan); xhi_map = np.full((h, w), np.nan)
    u_map = np.full((h, w), np.nan)
    events = []; row_reports = []; hard = []; sampled_rows = set(); changed_x = []
    used_contributions = 0; extension_count = 0; extension_y_weight = 0.
    mass_cache = {}

    def mass(row):
        if row not in mass_cache:
            try:
                mass_cache[row] = row_mass_coordinates(alpha[row])
            except ReviewRejected as error:
                error.evidence.update(row=int(row))
                raise
        return mass_cache[row]

    for target_y in np.flatnonzero(active_rows).tolist():
        if not np.any(alpha[target_y] > 0):
            continue  # Transparent target RGBA remains untouched.
        tx, u = mass(target_y); u_map[target_y, tx] = u
        target_alpha = alpha[target_y, tx]
        required = np.where(target_alpha >= 240, 240, target_alpha)
        rgb = np.zeros((len(tx), 3), float)
        for kind, sy, wy in (('low', int(ylo[target_y]), float(1 - fy[target_y])),
                             ('high', int(yhi[target_y]), float(fy[target_y]))):
            if wy == 0:
                continue  # No validation/sampling of an unused high-y row.
            mass(sy)  # Records row-specific failure and rejects all internal holes.
            sx, extension = inverse_mass_coordinates(alpha[sy], u)
            require(np.all(np.diff(sx) >= 0), 'Contour CDF inverse is not nondecreasing')
            sampled_rows.add(sy)
            (xlo_map if kind == 'low' else xhi_map)[target_y, tx] = sx
            dx = sx - tx; changed_x.extend(dx.tolist())
            scale = np.diff(sx)  # target positive columns are contiguous, dx_target=1.
            count = extension['lower_extension_count'] + extension['upper_extension_count']
            extension_count += count; extension_y_weight += count * wy
            row_reports.append(dict(target_y=target_y, source_y=sy, y_donor=kind, y_weight=wy,
                target_first=int(tx[0]), target_last=int(tx[-1]),
                dx_min=float(dx.min()), dx_max=float(dx.max()), dx_abs_max=float(np.abs(dx).max()),
                horizontal_step_min=float(scale.min()) if len(scale) else None,
                horizontal_step_max=float(scale.max()) if len(scale) else None,
                compression_steps=int(np.count_nonzero(scale < 1)), expansion_steps=int(np.count_nonzero(scale > 1)),
                zero_step_endpoint_extension=int(np.count_nonzero(scale == 0)), extension=extension,
                endpoint_extension_total_xy_weight=float(count * wy)))
            xl = np.floor(sx).astype(int); xh = np.minimum(xl + 1, w - 1); fx = sx - xl
            for x_kind, columns, wx in (('low', xl, 1 - fx), ('high', xh, fx)):
                weights = wy * wx; used = weights > 0
                used_contributions += int(np.count_nonzero(used))
                donor_alpha = alpha[sy, columns]
                zero = used & (donor_alpha == 0)
                # Preserve frozen v2 review qualification at the ACTUAL xy donor.
                shortfall = used & (donor_alpha < required)
                pair_ok = ((target_alpha == 255) | uncertainty[target_y, tx]) & (
                    (donor_alpha == 255) | uncertainty[sy, columns])
                internal = shortfall & ~zero & ~pair_ok
                for label, invalid in [('alpha_zero_actual_xy_donor', zero), ('internal_actual_xy_opacity_conflict', internal)]:
                    if np.any(invalid):
                        indices = np.flatnonzero(invalid)
                        hard.extend(dict(kind=label, target_y=target_y, target_x=int(tx[i]),
                            source_y=sy, source_x=int(columns[i]), weight=float(weights[i]),
                            target_alpha=int(target_alpha[i]), donor_alpha=int(donor_alpha[i])) for i in indices)
                review = shortfall & ~zero & pair_ok
                for i in np.flatnonzero(review).tolist():
                    events.append(dict(target_y=target_y, target_x=int(tx[i]), source_y=sy,
                        source_x=int(columns[i]), y_donor=kind, x_donor=x_kind, weight=float(weights[i]),
                        target_alpha=int(target_alpha[i]), donor_alpha=int(donor_alpha[i]),
                        original_required_alpha=int(required[i]),
                        target_edge_uncertain=bool(uncertainty[target_y, tx[i]]),
                        donor_edge_uncertain=bool(uncertainty[sy, columns[i]]),
                        original_opacity_contract_passed=False, state='boundary_alpha_confidence_review'))
                rgb += weights[:, None] * rgba[sy, columns, :3].astype(float)
        output[target_y, tx, :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    require(not hard, 'Invalid actual xy donor or internal opacity conflict; no v3 candidate',
            dict(hard_failures=hard, opacity_review_events=events))
    require(np.array_equal(output[..., 3], alpha), 'Alpha changed')
    require(np.array_equal(output[alpha == 0], rgba[alpha == 0]), 'Transparent RGBA changed')
    for a, b in plan['fixed_ranges']:
        require(np.array_equal(output[a:b], rgba[a:b]), 'Fixed controls/endcaps changed')
    # Old same-x losses remain a diagnostic, not a claim about this new map.
    old_core = []
    for band in plan['bands']:
        a, b = band['core']; delta = band['delta']
        sa = alpha[a:b + 1]; ta = alpha[a + delta:b + delta + 1]
        old_core.append(dict(band=band['band'], same_x_visible_to_zero=int(np.sum((sa > 0) & (ta == 0))),
            same_x_opaque_to_nonzero_below240=int(np.sum((sa >= 240) & (ta > 0) & (ta < 240))),
            evaluated_as_v3_acceptance=False))
    dx_values = np.asarray(changed_x)
    proof = dict(policy=POLICY, review_required=True, whole_core_rgb_rigid=False,
        source_rgb_rigid_inside_integer_core=False, all_source_pixels_retained_claimed=False,
        same_y_map_for_all_columns=True, full_map_same_for_all_columns=False,
        analytic_C1_positive_jacobian_proof_scope='longitudinal y map only',
        horizontal_map_nondecreasing=True, strict_2d_invertibility_claimed=False, area_conservation_claimed=False,
        horizontal_resampling_applied=bool(np.any(np.abs(dx_values) > 1e-12)),
        dx_min=float(dx_values.min()) if len(dx_values) else 0.,
        dx_max=float(dx_values.max()) if len(dx_values) else 0.,
        dx_abs_p95=float(np.percentile(np.abs(dx_values), 95)) if len(dx_values) else 0.,
        horizontal_step_min=min((r['horizontal_step_min'] for r in row_reports if r['horizontal_step_min'] is not None), default=None),
        horizontal_step_max=max((r['horizontal_step_max'] for r in row_reports if r['horizontal_step_max'] is not None), default=None),
        horizontal_compression_steps=sum(r['compression_steps'] for r in row_reports),
        horizontal_expansion_steps=sum(r['expansion_steps'] for r in row_reports),
        row_transport=row_reports, endpoint_extension_count=extension_count,
        endpoint_extension_total_xy_weight=extension_y_weight,
        opacity_review_events=events, original_opacity_contract_passed=not events,
        original_opacity_contract_scope='actual_positive_xy_donors_under_v3_only; not historical same-x or reverse-core coverage',
        actual_xy_donor_original_thresholds_passed=not events,
        legacy_same_x_contract_reclassified_as_passed=False,
        all_actual_positive_xy_donors_checked=True, actual_xy_donor_contributions=used_contributions,
        alpha_zero_actual_donors_used=0, internal_opacity_conflicts_permitted=0,
        validated_nonempty_contiguous_source_rows=sorted(sampled_rows),
        fixed_identity_rows_not_reinterpreted_as_sampling=True,
        alpha_changed=0, transparent_rgba_changed=0, fixed_rgba_changed=0,
        legacy_same_x_core_diagnostic=old_core,
        legacy_same_x_guard_not_applicable_reason='Horizontal contour coordinates changed; valid new donors do not prove retention of every old source pixel',
        changed_rgb_pixels=int(np.any(output != rgba, axis=2).sum()),
        strict_structure_verified=False, visual_approved=False, selected_for_delivery=False)
    transport = dict(source_y=source_y, x_for_low_y=xlo_map, x_for_high_y=xhi_map, target_alpha_mass_u=u_map)
    return (output, proof, transport) if return_transport else (output, proof)
