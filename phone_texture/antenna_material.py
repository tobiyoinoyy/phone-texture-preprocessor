"""Paired source boundaries across a side antenna's contrast crossover.

A dark metal rim can be darker than the plastic antenna. Both edges then
reverse polarity together; isolated per-column same-sign maxima are unsafe.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, label


def _material_vectors(colors, pair, col):
    y0,y1=np.rint(pair).astype(int)
    inset=min(2,max(1,(y1-y0-1)//2))
    inside=np.median(colors[y0+inset:y1-inset+1,col],axis=0)
    above=np.median(colors[max(0,y0-2):y0,col],axis=0)
    below=np.median(colors[y1+1:min(len(colors),y1+3),col],axis=0)
    return np.array([inside-above,inside-below])


def _color_continuation(rgb, alpha, basis, chosen):
    """Continue actual paired edges, using adjacent material colour evidence.

    RGB direction comes only from already observed neighbouring pairs.  A
    candidate cannot pick whichever channel happens to contain a noise peak.
    The same +/-3 search follows local measured contours, never a wider
    central extrapolation, and every added column must prove both edges.
    """
    h,w=alpha.shape;gradient=np.gradient(rgb,axis=0)
    additions=[]
    for step,columns in [(1,range(w//2,w)),(-1,range(w//2-1,-1,-1))]:
        for col in columns:
            if np.isfinite(chosen[:,col]).all():continue
            neighbours=[]
            for offset in range(1,4):
                other=col-step*offset
                if not 0<=other<w or not np.isfinite(chosen[:,other]).all():break
                neighbours.append(other)
            # Never bridge an unproved terminal column or trust a lone peak.
            if len(neighbours)<2:continue
            vectors=np.array([_material_vectors(rgb,chosen[:,other],other) for other in neighbours])
            reference=np.median(vectors.reshape(-1,3),axis=0)
            norm=np.linalg.norm(reference)
            if norm<1:continue
            agreement=np.sum(vectors*reference,axis=-1)/(np.linalg.norm(vectors,axis=-1)*norm+1e-12)
            if np.min(agreement)<.8:continue
            # L1 normalization keeps the neutral-colour scale equal to mean
            # RGB, preserving the existing .5 gradient / 1 contrast gates.
            direction=reference/np.sum(abs(reference))
            refined=chosen[:,neighbours].copy()
            for k in range(2):
                for n,other in enumerate(neighbours):
                    row=int(round(chosen[k,other]));ys=np.arange(max(1,row-1),min(h-1,row+2))
                    weights=np.maximum(0,gradient[ys,other]@direction*(1 if k==0 else -1))
                    if weights.sum()>0:refined[k,n]=np.average(ys,weights=weights)
            # The measured peak centroid prevents a rounded pixel maximum
            # from pinning a curved endpoint's search window to a flat row.
            local=np.round([np.polyval(np.polyfit(neighbours,refined[k],1),col) for k in range(2)],8)
            windows=[np.arange(max(1,int(np.floor(row))-3),min(h-1,int(np.ceil(row))+4)) for row in local]
            candidates=[]
            for y0 in windows[0]:
                for y1 in windows[1]:
                    if y1<=y0+2 or abs((y1-y0)-(basis[1,col]-basis[0,col]))>3:continue
                    if min(alpha[[y0,y1],col])<128:continue
                    g=gradient[[y0,y1],col]@direction
                    if g[0]<.5 or g[1]>-.5:continue
                    material=_material_vectors(rgb,[y0,y1],col)
                    if np.min(material@direction)<1:continue
                    agreement=(material@reference)/(np.linalg.norm(material,axis=1)*norm+1e-12)
                    if np.min(agreement)<.8:continue
                    score=min(g[0],-g[1])*np.exp(-sum((np.array([y0,y1])-local)**2)/8.)
                    candidates.append((score,y0,y1))
            if candidates:
                _,y0,y1=max(candidates);chosen[:,col]=y0,y1
                g=gradient[[y0,y1],col]@direction
                material=_material_vectors(rgb,[y0,y1],col)
                additions.append(dict(column=int(col),rows=[int(y0),int(y1)],reference_columns=neighbours,
                    reference_centroids=refined.tolist(),rgb_direction=direction.tolist(),local_prediction=local.tolist(),
                    projected_gradients=g.tolist(),material_contrasts=(material@direction).tolist(),
                    source_alpha=alpha[[y0,y1],col].tolist()))
    return additions


def _refine_pair_phase(gradient, chosen, reliable, polarity):
    """Locate an already supported pair within its native peak pixels.

    A bounded parabola removes integer-argmax phase quantization. It cannot
    create support, bridge missing columns, change polarity or move a peak
    beyond half a pixel. RGB-only continuation keeps its original evidence.
    """
    result=chosen.copy();refined=[];maximum=0.
    for col in np.flatnonzero(reliable&(polarity!=0)):
        pair=[]
        for k in range(2):
            y=int(chosen[k,col]);sign=np.sign(gradient[y,col])
            gm,g0,gp=gradient[y-1:y+2,col]*sign
            denominator=gm-2*g0+gp
            epsilon=32*np.finfo(float).eps*max(1.,abs(gm),abs(g0),abs(gp))
            if g0<.5 or g0+epsilon<max(gm,gp) or denominator>=-1e-9:break
            offset=.5*(gm-gp)/denominator
            if not np.isfinite(offset) or abs(offset)>.5+32*np.finfo(float).eps:break
            # Exact half-pixel ties differ at machine precision after the
            # derivative; clamp only that rounding residue, never a new peak.
            pair.append(y+float(np.clip(offset,-.5,.5)))
        if len(pair)==2:
            result[:,col]=pair;refined.append(int(col))
            maximum=max(maximum,float(np.max(np.abs(result[:,col]-chosen[:,col]))))
    return result,dict(method='bounded_supported_pair_peak_phase_v1',
                       columns=refined,maximum_shift_px=maximum)


def trace_boundaries(image, rows, bands=(0,1)):
    a=np.asarray(image.convert('RGBA'));h,w=a.shape[:2];x=np.arange(w)
    rows=np.asarray(rows,float)
    if rows.shape!=(4,) or not np.isfinite(rows).all() or np.any(np.diff(rows)<=2):
        raise ValueError('Four distinct source antenna boundaries required')
    rgb_colors=gaussian_filter1d(a[...,:3].astype(float),1.,axis=0)
    colors=gaussian_filter1d(a[...,:3].astype(float).mean(axis=2),1.,axis=0)
    gradient=np.gradient(colors,axis=0);center=(x>=int(w*.2))&(x<int(w*.8))
    lines=[];proof=[]
    if not bands or len(set(bands))!=len(bands) or any(b not in (0,1) for b in bands):
        raise ValueError('Explicit antenna band selection required')
    for band in bands:
        pair=rows[band*2:band*2+2];width=pair[1]-pair[0]
        radius=max(7,int(np.ceil(width*.6)));basis=[];central_sign=[]
        for row in pair:
            lo=max(1,int(round(row))-radius);hi=min(h-1,int(round(row))+radius+1)
            sign=np.sign(np.median(gradient[int(round(row)),int(w*.4):int(w*.6)]))
            if sign==0:raise ValueError('Antenna central polarity is ambiguous')
            central_sign.append(sign);peaks=lo+np.argmax(gradient[lo:hi]*sign,axis=0)
            fit=center&(a[peaks,x,3]>=240)&(gradient[peaks,x]*sign>.5)
            for _ in range(4):
                if fit.sum()<w*.45:raise ValueError('Insufficient central paired antenna support')
                c=np.polyfit(x[fit]/w,peaks[fit],2);pred=np.polyval(c,x/w)
                fit &= abs(peaks-pred)<=2
            if np.percentile(abs(peaks[fit]-pred[fit]),95)>1.5:
                raise ValueError('Unstable central antenna contour')
            basis.append(pred)
        if central_sign[0]*central_sign[1]>=0:
            raise ValueError('Antenna requires opposing central material edges')
        basis=np.asarray(basis);chosen=np.full((2,w),np.nan);polarity=np.zeros(w)
        for col in x:
            candidates=[]
            windows=[np.arange(max(1,int(np.floor(row))-3),min(h-1,int(np.ceil(row))+4)) for row in basis[:,col]]
            for y0 in windows[0]:
                for y1 in windows[1]:
                    if y1<=y0+2 or abs((y1-y0)-(basis[1,col]-basis[0,col]))>3:continue
                    g0,g1=gradient[[y0,y1],col]
                    if g0*g1>=0 or min(abs(g0),abs(g1))<.5 or min(a[[y0,y1],col,3])<128:continue
                    sign=np.sign(g0)
                    if center[col] and sign!=central_sign[0]:continue
                    inset=min(2,max(1,(y1-y0-1)//2))
                    inside=np.median(colors[y0+inset:y1-inset+1,col])
                    # Use the immediately adjacent material, not a wider
                    # window that can contain another reflection or scratch.
                    above=np.median(colors[max(0,y0-2):y0,col]);below=np.median(colors[y1+1:min(h,y1+3),col])
                    if min(sign*(inside-above),-sign*(below-inside))<1:continue
                    score=min(abs(g0),abs(g1))*np.exp(-sum((np.array([y0,y1])-basis[:,col])**2)/8.)
                    candidates.append((score,y0,y1,sign))
            if candidates:
                _,y0,y1,sign=max(candidates);chosen[:,col]=y0,y1;polarity[col]=sign
        color_evidence=_color_continuation(rgb_colors,a[...,3],basis,chosen)
        reliable=np.isfinite(chosen).all(axis=0)
        if reliable.sum()<w*.6:raise ValueError('Too few source-supported paired antenna columns')
        for side,cols in [('left',x[:max(8,int(w*.2))]),('right',x[-max(8,int(w*.2)):])]:
            if reliable[cols].sum()<3:raise ValueError('No paired endpoint evidence: '+side)
        edge_alpha=a[np.clip(np.rint(basis).astype(int),0,h-1),x,3]
        opaque=np.min(edge_alpha,axis=0)>=240
        terminal=(x<x[reliable].min())|(x>x[reliable].max())
        if np.any(terminal&np.any(edge_alpha>=240,axis=0)):
            missing_endpoint=x[terminal&np.any(edge_alpha>=240,axis=0)]
            raise ValueError(f'Opaque antenna endpoint lacks paired source evidence; no extrapolation; band {band}, columns {missing_endpoint.tolist()}')
        missing,count=label(~reliable&opaque)
        if any(np.sum(missing==i)>max(5,w*.12) for i in range(1,count+1)):
            raise ValueError('Unresolved opaque antenna span; no endpoint extrapolation')
        phase=None
        if band==1:
            chosen,phase=_refine_pair_phase(gradient,chosen,reliable,polarity)
        for k in range(2):
            curve=np.interp(x,x[reliable],chosen[k,reliable]);lines.append(gaussian_filter1d(curve,1.))
        proof.append(dict(band=band,supported_columns=int(reliable.sum()),
            opposite_polarity_columns=x[polarity==-central_sign[0]].tolist(),
            support_columns=x[reliable].tolist(),central_polarities=central_sign,
            maximum_opaque_gap_px=max([int(np.sum(missing==i)) for i in range(1,count+1)],default=0),
            rgb_continuation=color_evidence,subpixel_phase=phase))
    result=np.asarray(lines)
    if np.any(np.diff(result,axis=0)<=0):raise ValueError('Paired antenna boundaries cross')
    return result,dict(method='paired_source_material_edges_v3',bands=proof)
