"""Paired source boundaries across a side antenna's contrast crossover.

A dark metal rim can be darker than the plastic antenna. Both edges then
reverse polarity together; isolated per-column same-sign maxima are unsafe.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, label


def trace_boundaries(image, rows, bands=(0,1)):
    a=np.asarray(image.convert('RGBA'));h,w=a.shape[:2];x=np.arange(w)
    rows=np.asarray(rows,float)
    if rows.shape!=(4,) or not np.isfinite(rows).all() or np.any(np.diff(rows)<=2):
        raise ValueError('Four distinct source antenna boundaries required')
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
        reliable=np.isfinite(chosen).all(axis=0)
        if reliable.sum()<w*.6:raise ValueError('Too few source-supported paired antenna columns')
        for side,cols in [('left',x[:max(8,int(w*.2))]),('right',x[-max(8,int(w*.2)):])]:
            if reliable[cols].sum()<3:raise ValueError('No paired endpoint evidence: '+side)
        edge_alpha=a[np.clip(np.rint(basis).astype(int),0,h-1),x,3]
        opaque=np.min(edge_alpha,axis=0)>=240
        terminal=(x<x[reliable].min())|(x>x[reliable].max())
        if np.any(terminal&np.any(edge_alpha>=240,axis=0)):
            raise ValueError('Opaque antenna endpoint lacks paired source evidence; no extrapolation')
        missing,count=label(~reliable&opaque)
        if any(np.sum(missing==i)>max(5,w*.12) for i in range(1,count+1)):
            raise ValueError('Unresolved opaque antenna span; no endpoint extrapolation')
        for k in range(2):
            curve=np.interp(x,x[reliable],chosen[k,reliable]);lines.append(gaussian_filter1d(curve,1.))
        proof.append(dict(band=band,supported_columns=int(reliable.sum()),
            opposite_polarity_columns=x[polarity==-central_sign[0]].tolist(),
            support_columns=x[reliable].tolist(),central_polarities=central_sign,
            maximum_opaque_gap_px=max([int(np.sum(missing==i)) for i in range(1,count+1)],default=0)))
    result=np.asarray(lines)
    if np.any(np.diff(result,axis=0)<=0):raise ValueError('Paired antenna boundaries cross')
    return result,dict(method='paired_source_material_edges_v2',bands=proof)
