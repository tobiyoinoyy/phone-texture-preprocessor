"""Paired source boundaries across a side antenna's contrast crossover.

A dark metal rim can be darker than the plastic antenna. Both edges then
reverse polarity together; isolated per-column same-sign maxima are unsafe.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d, label, median_filter

# Unmeasured opaque end columns allowed per antenna end (shaded end cap).
ENDPOINT_EXTENSION_MAX_PX = 8


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


def _fit_central_peaks(peaks, gradient, alpha, x, center, sign):
    w=len(x)
    fit=center&(alpha[peaks,x]>=240)&(gradient[peaks,x]*sign>.5)
    for _ in range(4):
        if fit.sum()<w*.45:raise ValueError('Insufficient central paired antenna support')
        c=np.polyfit(x[fit]/w,peaks[fit],2);pred=np.polyval(c,x/w)
        fit &= abs(peaks-pred)<=2
    if np.percentile(abs(peaks[fit]-pred[fit]),95)>1.5:
        raise ValueError('Unstable central antenna contour')
    return pred


def _material_crossing_curve(colors, alpha, x, center, pair, index):
    """Fit native half-contrast crossings between this strip and its neighbouring metal."""
    width=pair[1]-pair[0]
    row=int(round(pair[index]))
    pad=max(2,round(width*.2))
    reach=max(pad+2,round(width*.55))
    inner=colors[int(np.ceil(pair[0]))+pad:int(np.floor(pair[1]))-pad+1]
    outer=colors[row-reach:row-pad] if index==0 else colors[row+pad+1:row+reach+1]
    if not len(inner) or not len(outer):
        raise ValueError('Material plateaus are too short')
    plastic=np.median(inner,axis=0)
    metal=np.median(outer,axis=0)
    delta=plastic-metal
    magnitude=np.linalg.norm(delta,axis=1)
    direction=delta/np.maximum(magnitude[:,None],1e-8)
    radius=max(4,round(width*.35))
    start=row-radius
    level=((colors[start:row+radius+1]-(plastic+metal)/2)*direction).sum(axis=2)
    if index==1:
        level=-level
    crossing=(level[:-1]<=0)&(level[1:]>0)
    slope=np.diff(level,axis=0)
    # Require the photographed material transition to exceed the variability
    # inside both neighbouring plateaus. Multiple crossings are ambiguous.
    scatter=lambda patch,base:np.median(abs(((patch-base)*direction).sum(axis=2)),axis=0)
    noise=np.maximum(scatter(inner,plastic),scatter(outer,metal))
    k=crossing.argmax(axis=0)
    fit=center&(crossing.sum(axis=0)==1)&(magnitude>np.maximum(3,4*noise))&(slope[k,x]>.75)
    fit &= (alpha[start+k,x]>=240)&(alpha[start+k+1,x]>=240)
    rows=start+k-level[k,x]/np.maximum(.001,slope[k,x])
    mid=fit&(x>=len(x)*.4)&(x<len(x)*.6)
    local_check=np.median(level[:,(x>=len(x)*.4)&(x<len(x)*.6)],axis=1)
    central_cross=np.flatnonzero((local_check[:-1]<=0)&(local_check[1:]>0))
    if len(central_cross)!=1 or abs(start+central_cross[0]-local_check[central_cross[0]]/
            (local_check[central_cross[0]+1]-local_check[central_cross[0]])-row)>2.5 or not mid.any():
        raise ValueError('Native material transition disagrees with central source reading')
    native=fit.copy()
    for _ in range(4):
        if fit.sum()<len(x)*.45:
            raise ValueError('Insufficient native paired material coverage')
        pred=np.polyval(np.polyfit(x[fit]/len(x),rows[fit],2),x/len(x))
        fit &= abs(rows-pred)<=2
    if fit.sum()<len(x)*.45 or np.percentile(abs(rows[fit]-pred[fit]),95)>1.5:
        # Local dents need not follow a quadratic. Alternating columns must
        # independently resolve the same contour before it can guide a trace.
        estimates=[]
        for parity in (0,1):
            cols=np.flatnonzero(native & (x%2==parity))
            if len(cols)<len(x)*.225 or np.diff(cols).max(initial=0)>max(6,.08*len(x)):
                raise ValueError('Material transition is not stable across central columns')
            estimates.append(np.interp(x,cols,rows[cols]))
        if np.percentile(np.abs(estimates[0][native]-estimates[1][native]),95)>1.5:
            raise ValueError('Independent native material traces disagree')
        local=(estimates[0]+estimates[1])/2
        pred[center]=local[center]
    return pred


def _windowed_edge_guide(colors, alpha, x, center, pair, idx):
    # Use disjoint metal windows only to aim the native paired-column search.
    # Each window locates a signed transition independently of its neighbours.
    row=int(round(pair[idx]))
    r=max(4,round((pair[1]-pair[0])*.45))
    positions=[];xs=[]
    for cols in np.array_split(x[center],9):
        if len(cols)<3 or np.min(alpha[row-r:row+r+1,cols])<240:
            continue
        profile=np.median(colors[row-r:row+r+1,cols],axis=1)
        outer=np.median(profile[:3],axis=0)
        inner=np.median(profile[-3:],axis=0)
        delta=inner-outer
        norm=np.linalg.norm(delta)
        if norm<5:continue
        line=(profile-(inner+outer)/2)@(delta/norm)
        hits=np.flatnonzero((line[:-1]<=0)&(line[1:]>0))
        if len(hits)!=1:continue
        k=hits[0];rise=line[k+1]-line[k]
        if rise<.75:continue
        y=row-r+k-line[k]/rise
        if abs(y-row)<=r-2:
            xs.append(cols.mean());positions.append(y)
    if len(xs)<7 or abs(np.median(positions)-row)>2.5:
        raise ValueError('Independent metal windows do not confirm the source boundary')
    xs=np.asarray(xs);positions=np.asarray(positions)
    good=np.ones(len(xs),dtype=bool)
    while good.sum()>=7:
        alternatives=[]
        for degree in (1,2):
            errors=[]
            for j in np.flatnonzero(good):
                keep=good.copy()
                keep[j]=False
                model=np.polyfit(xs[keep],positions[keep],degree)
                errors.append(abs(np.polyval(model,xs[j])-positions[j]))
            alternatives.append(errors)
        errors=min(alternatives,key=max)
        if max(errors)<=1.5 and np.ptp(xs[good])>=.4*len(x):
            return np.interp(x,xs[good],positions[good])
        good[np.flatnonzero(good)[int(np.argmax(errors))]]=False
    raise ValueError('Metal-window contour fails withheld source observations')


def _neighbour_peak_curve(gradient, alpha, x, center, sign, row, lo, hi):
    # A scratch in one column may be stronger than the actual strip. First
    # measure a coherent neighbourhood, then refit native signed RGB maxima.
    window=gradient[lo:hi]*sign
    guide=lo+np.argmax(median_filter(window,size=(1,7)),axis=0)
    middle=(x>=len(x)*.4)&(x<len(x)*.6)
    if abs(np.median(guide[middle])-row)>2:
        raise ValueError('Neighbour curve does not match the observed central boundary')
    local=(window>=np.roll(window,1,axis=0))&(window>=np.roll(window,-1,axis=0))
    local[[0,-1]]=False
    near=abs(np.arange(lo,hi)[:,None]-guide)<=2
    scores=np.where(local&near,window,-np.inf)
    peaks=lo+scores.argmax(axis=0)
    measured=center&np.isfinite(scores.max(axis=0))
    return _fit_central_peaks(peaks,gradient,alpha,x,measured,sign)


def _rgb_half_pairs(colors, alpha, basis, chosen, center, central_sign):
    """Read missing boundaries on each native RGB column with two-sided material support."""
    h,w=alpha.shape
    observed=np.isfinite(chosen).all(axis=0)
    additions=[]
    for col in np.flatnonzero(~observed):
        expected=basis[:,col]
        a,b=np.rint(expected).astype(int)
        width=b-a
        inset=max(2,round(.22*width))
        flank=max(4,round(.5*width))
        if width<6 or a-flank<1 or b+flank>=h-1 or a+inset>=b-inset:
            continue
        inside=colors[a+inset:b-inset+1,col]
        metal=[colors[a-flank:a-2,col],colors[b+3:b+flank+1,col]]
        if min(map(len,metal))<2:
            continue
        plastic=np.median(inside,axis=0)
        pair=[];axes=[]
        for k,(segment,target) in enumerate(zip(metal,expected)):
            outer=np.median(segment,axis=0)
            color=plastic-outer
            norm=np.linalg.norm(color)
            axis=color/max(.001,norm)
            noise=max(np.median(abs((inside-plastic)@axis)),np.median(abs((segment-outer)@axis)),1.)
            if norm<max(5,4*noise) or (center[col] and np.sign(color.sum())!=central_sign[0]):
                break
            start=int(np.floor(target))-3
            profile=(colors[start:int(np.ceil(target))+5,col]-(outer+plastic)/2)@axis
            if k==1:profile=-profile
            cross=np.flatnonzero((profile[:-1]<=0)&(profile[1:]>0))
            if len(cross)!=1:
                break
            idx=int(cross[0])
            slope=profile[idx+1]-profile[idx]
            if slope<.75 or alpha[start+idx:start+idx+2,col].min()<128:
                break
            pair.append(start+idx-profile[idx]/slope)
            axes.append(axis)
        if len(pair)==2 and axes[0]@axes[1]>.85 and abs(pair[1]-pair[0]-width)<=3:
            additions.append((int(col),pair))
    # Only extend the evidence with a run of actual, mutually consistent
    # paired observations. An isolated noise crossing does not vote.
    verified=[]
    for col,pair in additions:
        near=[p for c,p in additions if 0<abs(c-col)<=2 and np.max(abs(np.subtract(p,pair)))<=2]
        native=np.flatnonzero(observed & (abs(np.arange(w)-col)<=2))
        if not near and not any(np.max(abs(chosen[:,c]-pair))<=2 for c in native):
            continue
        chosen[:,col]=pair
        verified.append(dict(column=col,rows=pair))
    return verified


def trace_boundaries(image, rows, bands=(0,1), *, _observer=None, recover_center=True):
    def run(hold,observer):
        try:
            return _trace(image,rows,bands,observer,hold,False)
        except ValueError:
            if not recover_center:
                raise
        try:
            return _trace(image,rows,bands,observer,hold,True)
        except ValueError as original:
            if 'opaque antenna span' not in str(original).lower() and 'endpoint' not in str(original).lower():
                raise
        return _trace(image,rows,bands,observer,hold,True,bounded_guide=True)
    result,proof=run((),_observer)
    sides=sorted({item['side'] for item in proof['endpoint_shape_weak']})
    if sides and len(bands)>1:
        result,proof=run(tuple(sides),None)
    proof['endpoint_held_sides']=sides
    return result,proof


def _trace(image, rows, bands, _observer, hold, recover_center, bounded_guide=False):
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
            try:
                pred=_fit_central_peaks(peaks,gradient,a[...,3],x,center,sign)
            except ValueError:
                if not recover_center:
                    raise
                try:
                    pred=_neighbour_peak_curve(gradient,a[...,3],x,center,sign,row,lo,hi)
                except ValueError:
                    try:
                        pred=_material_crossing_curve(rgb_colors,a[...,3],x,center,pair,len(basis))
                    except ValueError:
                        # Suppress isolated dents in the search guide. All
                        # selected column pairs below still use native RGB.
                        y0=max(0,int(pair[0])-2*int(width))
                        y1=min(h,int(pair[1])+2*int(width)+1)
                        guided=median_filter(rgb_colors[y0:y1],size=(1,5,1))
                        try:
                            pred=y0+_material_crossing_curve(guided,a[y0:y1,:,3],x,center,pair-y0,len(basis))
                        except ValueError:
                            pred=_windowed_edge_guide(rgb_colors,a[...,3],x,center,pair,len(basis))
            if bounded_guide:
                # A quadratic fitted to the middle can miss a straight rail
                # at either end. Use the last central sample as the *search
                # guide*, then require native paired observations as usual.
                core=x[center]
                pred=np.interp(x,core,pred[core])
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
                    candidates.append((min(abs(g0),abs(g1))>=.5,score,y0,y1,sign))
            if candidates:
                _,_,y0,y1,sign=max(candidates);chosen[:,col]=y0,y1;polarity[col]=sign
        color_evidence=_color_continuation(rgb_colors,a[...,3],basis,chosen)
        half_pairs=_rgb_half_pairs(rgb_colors,a[...,3],basis,chosen,center,central_sign) if recover_center else []
        if _observer is not None:
            # Diagnostic copies cannot alter the candidates or disable gates.
            observed=chosen.copy();observed.setflags(write=False)
            predicted=basis.copy();predicted.setflags(write=False)
            _observer(band,observed,predicted)
        reliable=np.isfinite(chosen).all(axis=0)
        if reliable.sum()<w*.6:raise ValueError('Too few source-supported paired antenna columns')
        for side,cols in [('left',x[:max(8,int(w*.2))]),('right',x[-max(8,int(w*.2)):])]:
            if reliable[cols].sum()<3:raise ValueError('No paired endpoint evidence: '+side)
        edge_alpha=a[np.clip(np.rint(basis).astype(int),0,h-1),x,3]
        opaque=np.min(edge_alpha,axis=0)>=240
        terminal=(x<x[reliable].min())|(x>x[reliable].max())
        missing_endpoint=x[terminal&np.any(edge_alpha>=240,axis=0)]
        # User decision 2026-11-05: the rail's end cap curves into shadow, so
        # metal and plastic both go dark there. A few unmeasured end columns
        # follow the supported paired curve (np.interp holds the last
        # measured rows; no polynomial extrapolation). Wider gaps still fail.
        extended=[]
        for side_cols in (missing_endpoint[missing_endpoint<x[reliable].min()],
                          missing_endpoint[missing_endpoint>x[reliable].max()]):
            if len(side_cols)>ENDPOINT_EXTENSION_MAX_PX:
                raise ValueError(f'Opaque antenna endpoint lacks paired source evidence; no extrapolation; band {band}, columns {missing_endpoint.tolist()}')
            extended.extend(int(c) for c in side_cols)
        missing,count=label(~reliable&opaque)
        if any(np.sum(missing==i)>max(5,w*.12) for i in range(1,count+1)):
            raise ValueError('Unresolved opaque antenna span; no endpoint extrapolation')
        shape_evidence=[];shape_warnings=[]
        z=np.linspace(-.5,1.5,41)
        template=((z>0)&(z<1)).astype(float);template-=template.mean()
        # Endpoint shape is judged on the 8 columns next to the shaded end cap
        # (from the first/last measured column inward), not on the extended
        # cap itself. Unmeasured opaque columns inside the window still count
        # against the strip, so endpoint noise cannot pass by being skipped.
        first,last=int(x[reliable].min()),int(x[reliable].max())
        for side,cols in [('left',x[first:first+8]),('right',x[max(0,last-7):last+1])]:
            correlations=[];contrasts=[];strengths=[]
            for col in cols:
                if not reliable[col]:
                    # Missing opaque columns are evidence against a coherent
                    # strip. Scoring only selected noise peaks biases the
                    # median upward and hides an interpolated opaque gap.
                    if opaque[col]:correlations.append(0.)
                    continue
                lo,hi=chosen[:,col];ys=lo+z*(hi-lo)
                sample=np.array([np.interp(ys,np.arange(h),rgb_colors[:,col,k]) for k in range(3)]).T
                outer=(z<-.15)|(z>1.15)
                sample-=np.column_stack([np.polyval(np.polyfit(z[outer],sample[outer,k],1),z) for k in range(3)])
                sample-=sample.mean(axis=0)
                correlation=np.linalg.norm(sample.T@template)/(np.linalg.norm(sample)*np.linalg.norm(template)+1e-12)
                correlations.append(float(correlation))
                contrasts.append(float(np.min(np.linalg.norm(_material_vectors(rgb_colors,chosen[:,col],col),axis=-1))/np.sqrt(3)))
                strengths.append(float(np.min(np.abs(gradient[np.rint(chosen[:,col]).astype(int),col]))))
            median=float(np.median(correlations)) if correlations else 0.
            contrast=float(np.median(contrasts)) if contrasts else 0.
            strength=float(np.median(strengths)) if strengths else 0.
            shape_evidence.append(dict(side=side,columns=cols.tolist(),correlations=correlations,median=median,contrast_rgb_rms=contrast,gradient_strength=strength))
            # User decision 2026-11-06: a weak end strip is reported, not
            # blocking. Its columns are not trusted as measurements: they hold
            # the nearest coherent pair, like the shaded end cap. Wide
            # unmeasured ends and opaque gaps still fail above.
            weak=(contrast<6 or strength<2) and median<.65
            if weak:shape_warnings.append(side)
            if weak or side in hold:
                reliable[cols]=False
                extended=sorted(set(extended)|{int(c) for c in cols if opaque[c] or edge_alpha[:,c].max()>=240})
        if reliable.sum()<w*.6:raise ValueError('Too few source-supported paired antenna columns')
        phase=None
        if band==1:
            chosen,phase=_refine_pair_phase(gradient,chosen,reliable,polarity)
        for k in range(2):
            curve=np.interp(x,x[reliable],chosen[k,reliable]);lines.append(gaussian_filter1d(curve,1.))
        proof.append(dict(band=band,supported_columns=int(reliable.sum()),
            opposite_polarity_columns=x[polarity==-central_sign[0]].tolist(),
            support_columns=x[reliable].tolist(),central_polarities=central_sign,
            maximum_opaque_gap_px=max([int(np.sum(missing==i)) for i in range(1,count+1)],default=0),
            endpoint_extended_columns=extended,endpoint_extension_max_px=ENDPOINT_EXTENSION_MAX_PX,
            rgb_continuation=color_evidence,native_rgb_half_pairs=half_pairs,subpixel_phase=phase,endpoint_shape=shape_evidence,
            endpoint_shape_weak_sides=shape_warnings))
    result=np.asarray(lines)
    if np.any(np.diff(result,axis=0)<=0):raise ValueError('Paired antenna boundaries cross')
    weak=[dict(band=item['band'],side=side) for item in proof for side in item['endpoint_shape_weak_sides']]
    return result,dict(method='paired_source_material_edges_v3',bands=proof,
        endpoint_shape_weak=weak,endpoint_shape_review_required=bool(weak))
