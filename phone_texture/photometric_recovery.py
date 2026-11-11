"""Recover source pairs after independently fitting the surrounding lighting.

Only missing opaque terminal columns can use this evidence. All standard pair
gates remain active, followed by independent raw-source two-edge checks.
The dimensionless contrast scores are empirical checks, not probabilities.
"""
import numpy as np
from . import antenna_material as am
from .photometric_field import normalize,AnalyticalRGBA
from .photometric_evidence import evidence


def recover(image,rows):
    try:
        am.trace_boundaries(image,rows)
        return image,dict(method='ordinary_source_pair',normalized_columns=[])
    except ValueError as exc:
        original_error=exc
    raw=np.asarray(image.convert('RGBA')).astype(float)
    # Establish applicability before fitting any illumination. A failed shape
    # or central-pair check is not a missing-terminal-column problem. Keep its
    # actual rejection instead of replacing it with a recovery precondition.
    plan=[];failures=[];inapplicable=[]
    for band in (0,1):
        observed={}
        def capture(index,chosen,basis):observed[index]=(chosen,basis)
        try:
            am.trace_boundaries(image,rows,bands=(band,),_observer=capture)
            continue
        except ValueError as exc:
            error=exc
        row=dict(band=band,error=str(error),missing_opaque_terminal_columns=[])
        failures.append(row)
        if band not in observed:
            inapplicable.append(error);continue
        chosen,basis=observed[band];reliable=np.isfinite(chosen).all(axis=0)
        if not reliable.any():
            inapplicable.append(error);continue
        xx=np.arange(raw.shape[1]);terminal=(xx<xx[reliable].min())|(xx>xx[reliable].max())
        edge_alpha=raw[np.clip(np.rint(basis).astype(int),0,len(raw)-1),xx,3]
        missing=(~reliable)&terminal&np.any(edge_alpha>=240,axis=0)
        row['missing_opaque_terminal_columns']=np.flatnonzero(missing).tolist()
        terminal_failure=str(error).startswith((
            'Opaque antenna endpoint lacks paired source evidence; no extrapolation;',
            'No paired endpoint evidence:'))
        if not terminal_failure or not missing.any():
            inapplicable.append(error);continue
        plan.append((band,missing))
    if inapplicable:
        error=inapplicable[0]
        error.photometric_recovery_failures=failures
        raise error
    if not plan:
        original_error.photometric_recovery_failures=failures
        raise original_error
    try:
        return _recover_terminals(image,rows,raw,plan,failures)
    except ValueError as error:
        error.photometric_recovery_failures=failures
        raise


def _recover_terminals(image,rows,raw,plan,failures):
    normalized,field=normalize(image,rows);hybrid=raw.copy();required=[]
    for band,missing in plan:
        lo,hi=rows[2*band:2*band+2];width=hi-lo
        lower=max(0,int(lo-width*3));upper=min(len(raw),int(hi+width*3)+1)
        hybrid[lower:upper,missing,:3]=normalized.data[lower:upper,missing,:3]
        required.extend((band,int(col)) for col in np.flatnonzero(missing))
    observed={}
    def capture(index,chosen,basis):observed[index]=(chosen,basis)
    reference=AnalyticalRGBA(hybrid)
    am.trace_boundaries(reference,rows,_observer=capture)
    proof=[]
    for band,col in required:
        pair=observed[band][0][:,col]
        if not np.isfinite(pair).all():raise ValueError('Source endpoint still unresolved after normalization')
        model=field[band]['models'][col]
        if model is None:raise ValueError('Missing independently supported source background')
        stat=evidence(raw,pair,col,model['degree'])
        if min(stat['edge_t'])<1.5 or stat['joint_rectangular_t']<3 or stat['opposing_agreement']<.8:
            raise ValueError('Original independent source edges lack evidence: '+str((band,col,stat)))
        proof.append(dict(band=band,column=col,rows=pair.tolist(),source_evidence=stat,background=model))
    return reference,dict(method='source_background_cv_terminal_pairs_v1',normalized_columns=proof,
        original_band_failures=failures,
        source_dtype='original_RGB',edge_score_minimum=1.5,joint_score_minimum=3,
        opposing_agreement_minimum=.8,all_original_pair_gates_retained=True)
