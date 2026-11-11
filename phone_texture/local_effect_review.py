"""Explicit appearance-review states; source/geometry and real hard errors remain."""
import numpy as np
POLICY='quantization_bounded_pixel_review_v1'

class EffectLocalReviewRequired(ValueError):
    def __init__(self,issues,reports,failures):
        self.issues,self.reports,self.failures=issues,reports,failures
        super().__init__('; '.join(x['message'] for x in issues))

def _joint_chain_intervals(low,high,witness,axis,limit):
    """Certify one simultaneous delta-L assignment on every witness chain.

    The reachable interval at each pixel is its candidate-only quantization
    interval intersected with the preceding reachable interval plus/minus the
    SAME limit. This is exact for a path graph. A false witness splits paths;
    a measured-under-limit edge does not. Comparisons use no epsilon.
    """
    if axis not in (0,1):raise ValueError('Invalid quantization review axis')
    low,high,witness=np.asarray(low),np.asarray(high),np.asarray(witness,dtype=bool)
    if low.ndim!=2 or high.shape!=low.shape or witness.shape!=low.shape:
        raise ValueError('Source-sized candidate intervals and witness required')
    if not np.isfinite(limit) or limit<=0:raise ValueError('Invalid quantization review limit')
    if np.any(witness&(~np.isfinite(low)|~np.isfinite(high)|(low>high))):
        raise ValueError('Invalid candidate quantization interval')
    lows,highs,active=(np.moveaxis(a,axis,0) for a in (low,high,witness))
    if not len(lows):
        return dict(feasible=True,axis=axis,checked_pair_count=0,connected_chain_count=0,
            method='joint_delta_L_path_interval_propagation',all_witness_edges_checked=True,
            comparison='strict_no_epsilon')
    reachable_low=lows[0].copy();reachable_high=highs[0].copy()
    previous_edge=np.zeros(active.shape[1],bool);pair_count=chain_count=0
    for index in range(1,len(lows)):
        edge=active[index-1]&active[index]
        pair_count+=int(edge.sum());chain_count+=int((edge&~previous_edge).sum())
        lower=np.maximum(lows[index],reachable_low-limit)
        upper=np.minimum(highs[index],reachable_high+limit)
        if np.any(edge&(lower>upper)):return None
        # Reset only at a real witness gap, never because a measured edge was
        # already under limit or because an intermediate pixel was unchanged.
        reachable_low=np.where(edge,lower,lows[index])
        reachable_high=np.where(edge,upper,highs[index])
        previous_edge=edge
    return dict(feasible=True,axis=axis,checked_pair_count=pair_count,
        connected_chain_count=chain_count,method='joint_delta_L_path_interval_propagation',
        all_witness_edges_checked=True,comparison='strict_no_epsilon')


def quantization_pixel_review(source,candidate,witness,axis,limit):
    """Candidate ±half an 8-bit code value, source fixed; not proof of causation."""
    from .lighting_review import lightness
    if source.dtype!=np.uint8 or candidate.dtype!=np.uint8 or source.shape!=candidate.shape:return None
    if axis not in (0,1):raise ValueError('Invalid quantization review axis')
    a=lightness(source[...,:3]);b=lightness(candidate[...,:3])
    delta=b-a;pairs=(witness[1:]&witness[:-1]) if axis==0 else (witness[:,1:]&witness[:,:-1])
    steps=np.abs(np.diff(delta,axis=axis));exceeded=pairs&(steps>limit)
    if not exceeded.any():return None
    low=lightness(np.clip(candidate[...,:3].astype(float)-.5,0,255))-a
    high=lightness(np.clip(candidate[...,:3].astype(float)+.5,0,255))-a
    lo=[slice(None),slice(None)];hi=lo.copy();lo[axis]=slice(None,-1);hi[axis]=slice(1,None);lo,hi=tuple(lo),tuple(hi)
    signed_low=low[hi]-high[lo];signed_high=high[hi]-low[lo]
    min_step=np.where((signed_low<=0)&(signed_high>=0),0,np.minimum(abs(signed_low),abs(signed_high)))
    if np.any(min_step[exceeded]>limit):return None
    joint=_joint_chain_intervals(low,high,witness,axis,limit)
    if joint is None:return None
    yy,xx=np.nonzero(exceeded)
    return dict(kind='quantization_compatible_pixel_warning',
        message='局部单像素明暗差略超程序阈值，落在8位量化可能范围内；保留候选待核对',
        limit_L=limit,original_threshold_passed=False,measured_max_step_L=float(steps[pairs].max()),
        pair_count=int(pairs.sum()),over_limit_pairs=int(exceeded.sum()),
        coordinates_yx=list(zip(yy.tolist(),xx.tolist())),measured_steps_L=steps[exceeded].tolist(),
        minimum_possible_steps_with_candidate_quantization_L=min_step[exceeded].tolist(),
        joint_chain_feasibility=joint,
        source_pixels_held_fixed=True,candidate_RGB_uncertainty_per_channel=.5,
        rounding_cause_proven=False,quality_checks_complete=False)

def effect_result(error):
    warnings=list(error.issues)
    if error.failures:warnings.append(dict(kind='lighting_effect_target_warning',message='局部提亮未达预定目标，保留候选待核对',error='; '.join(x['error'] for x in error.failures)))
    return dict(passed=False,original_effect_check_passed=False,kind='local_effect_review_required',
        status='effect_review_required',error=str(error),warnings=warnings,
        effect_failures=error.failures,effect_measurements=error.reports,
        non_aesthetic_scope_checks_complete=False,quality_checks_complete=False,
        action='continue_remaining_stages_with_explicit_review_no_raw_fallback')
