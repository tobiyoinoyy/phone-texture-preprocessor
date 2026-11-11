"""Separate source-bound normal measurements from processing completion."""
from pathlib import Path
import numpy as np

POLICY = 'review_insufficient_normal_on_source_closed_material'
CONTEXT_POLICY = 'review_bottom_normal_context_with_source_material_binding'
KIND = 'normal_coverage_unavailable'
MESSAGE = '局部法向对照不足，保留已调光候选待核对'


class EffectCoverageUnavailable(ValueError):
    def __init__(self, coverage, reports, failures):
        self.coverage, self.reports, self.failures = coverage, reports, failures
        super().__init__(MESSAGE)


def review_enabled(spec, source):
    policy = spec.get('normal_coverage_policy')
    if policy is None:
        return False
    if policy == CONTEXT_POLICY:
        view=spec.get('source_material_view',{})
        if (spec.get('mode')!='regional' or Path(view.get('source','')).resolve()!=Path(source).resolve()):
            raise ValueError('Normal context review requires the current bottom source')
        from .resolved_bottom_material import validate
        validate(view,spec.get('source_paper_material_boundary',{}),scope=spec)
        return True
    if policy != POLICY:
        raise ValueError('Unknown normal coverage policy')
    view = spec.get('source_contour_material_view', {})
    evidence = spec.get('source_paper_material_boundary', {})
    if (view.get('name') != 'bottom' or spec.get('mode') != 'regional'
            or Path(view.get('source', '')).resolve() != Path(source).resolve()
            or evidence.get('material_boundary_resolution') != 'source_contour_frame_domain'):
        raise ValueError('Normal coverage review requires the new source-closed bottom path')
    # Supplied by the independent material bridge; never replace with a flag.
    from .source_material_bridge import validate_source_contour_material
    validate_source_contour_material(view, evidence, scope=spec)
    return True


def scoped_normal_review(spec, source, name, zone):
    return (name == 'bottom_paper_metal_normal' and 'normal_peer_mask' in zone
            and review_enabled(spec, source))


def measure_normal(delta, mask, peer, limit, *, contextual_review=False):
    """Use unchanged source-selected pixels; keep every available row/count.

    No new mask, interpolation, count padding, or presumed physical support.
    Default behavior keeps over-limit residuals as hard errors. Explicit bottom
    contextual review records over-limit rows without certifying their quality;
    the caller retains them as warnings and continues the remaining checks.
    """
    normal = np.diff(delta, axis=0)
    target_pairs = mask[1:] & mask[:-1]
    peer_pairs = peer[1:] & peer[:-1]
    rows, missing, exceeded = [], [], []
    qualified_pairs = 0
    for y, (target, control) in enumerate(zip(target_pairs, peer_pairs)):
        target_count, peer_count = int(target.sum()), int(control.sum())
        if not target_count:
            continue
        row = dict(row=y, source_rows=[y, y+1], witness_pairs=target_count,
                   peer_pairs=peer_count, witness_columns=np.flatnonzero(target).tolist(),
                   peer_columns=np.flatnonzero(control).tolist(),
                   target_added_gradient_L=normal[y, target].tolist(),
                   peer_added_gradient_L=normal[y, control].tolist())
        if peer_count:
            reference = float(np.median(normal[y, control]))
            residual = float(np.percentile(abs(normal[y, target]-reference), 75))
            row.update(peer_added_gradient_median_L=reference,
                       added_gradient_residual_L75=residual)
            if residual > limit:
                row['threshold_exceeded']=True
                exceeded.append(y)
            if residual > limit and not contextual_review:
                error = ValueError(f'Lighting normal peer discontinuity: row {y}: {residual:.2f} L')
                error.normal_coverage_evidence = dict(rows=rows+[row], limit_L=limit,
                    low_sample_count=target_count < 10 or peer_count < 10)
                raise error
        qualified = target_count >= 10 and peer_count >= 10
        row['legacy_sample_condition_met'] = qualified
        rows.append(row)
        if qualified:
            qualified_pairs += target_count
        else:
            missing.append(dict(row=y, witness_pairs=target_count, peer_pairs=peer_count,
                reasons=(["target_pairs_below_10"] if target_count < 10 else [])
                        + (["peer_pairs_below_10"] if peer_count < 10 else [])))
    reasons = []
    if not target_pairs.any():
        reasons.append('no_pairs_in_original_normal_witness')
    if qualified_pairs < 20:
        reasons.append('qualified_target_pairs_below_20')
    if missing:
        reasons.append('some_source_selected_rows_have_insufficient_pairs')
    complete = not reasons
    measured_rows = [r for r in rows if 'added_gradient_residual_L75' in r]
    return dict(threshold_exceeded_rows=exceeded,contextual_review=contextual_review,
        original_threshold_passed=complete and not exceeded,
        status='complete' if complete else 'limited' if measured_rows else 'insufficient',
        complete=complete, assessed_direction='source_image_axis_0',
        source_mask_pixels=int(mask.sum()), normal_pairs=int(target_pairs.sum()),
        qualified_target_pairs=qualified_pairs, minimum_pairs_per_row=10,
        minimum_total_pairs=20, limit_L=limit, rows=rows, missing_rows=missing,
        reason_codes=reasons, sampling_algorithm_unchanged=True,
        not_applicable_certified=False, crop_is_unique_cause_claimed=False,
        physical_transition_support_certified=False,
        max_observed_residual_L75=max((r['added_gradient_residual_L75'] for r in measured_rows), default=None))


def effect_review_result(error):
    warnings = [dict(kind=KIND, message=MESSAGE, error=str(error))]
    if error.failures:
        warnings.append(dict(kind='lighting_effect_target_warning',
            message='局部提亮未达预定目标，保留已处理候选待核对',
            error='; '.join(item['error'] for item in error.failures)))
    return dict(passed=False, original_effect_check_passed=False,
        kind=KIND, status='effect_review_required', error=str(error), warnings=warnings,
        effect_failures=error.failures, effect_measurements=error.reports,
        normal_coverage=error.coverage, normal_scope_checks_complete=False,
        non_aesthetic_scope_checks_complete=False, quality_checks_complete=False,
        action='continue_remaining_stages_with_warning_no_raw_fallback')
