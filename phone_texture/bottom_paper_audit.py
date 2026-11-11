"""Source-only audit of demonstrably oversized bottom-paper masks.

This rejection guard does not generate masks or certify complete paper edges.
"""
import cv2
import numpy as np
from scipy import ndimage


def _source_cross_edge(a, paper, proof):
    from .auto_material import red_markers
    h, w = paper.shape
    rows = np.flatnonzero(paper.any(axis=1))
    if len(rows) < h * .4:
        raise ValueError('Paper strip has insufficient source rows')
    left = np.array([np.flatnonzero(paper[y])[0] for y in rows])
    right = np.array([np.flatnonzero(paper[y])[-1] for y in rows])
    lf = np.polyfit(rows, left, 1)
    rf = np.polyfit(rows, right, 1)
    radius = max(6, int(np.ceil(h * .07)))
    components = proof.get('components', [])
    markers = [v for v in red_markers(a)
               if paper[min(h-1, int(v[0])), min(w-1, int(v[1]))]]
    # The old extrapolated full-height strip cannot authenticate a marker.
    # A white component, when present, supplies an independent vertical bound.
    if components:
        lower = max(c['box'][3] for c in components)
        markers = [v for v in markers if v[0] < lower + 1]
        predicted = float(lower - 1)
        censored = lower >= h - max(2, int(h * .10)) - 1
        lo, hi = max(int(h*.45), lower-radius), min(h-2, lower+radius)
        if censored:
            hi = h-2
        anchor = 'source_white_component_censored' if censored else 'source_white_component'
    elif markers:
        cy, cx, bw, bh = max(markers)
        predicted = cy + h * .14
        lo, hi = max(int(cy+bh), int(predicted-radius)), min(h-2, int(predicted+radius))
        anchor = 'source_print_marker_search_only'
    else:
        predicted = None
        lo, hi = int(h*.5), h-2
        anchor = 'independent_strip_closure_search'
    if hi-lo < 3:
        raise ValueError('Physical paper end has no independent search interval')

    gray = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY).astype(float)
    grad = -np.gradient(ndimage.gaussian_filter1d(gray, .8, axis=0), axis=0)
    foreground, background, contrasts = [], [], []
    outside_fields = []
    for y in range(h):
        l, r = int(round(np.polyval(lf,y))), int(round(np.polyval(rf,y)))
        span = r-l+1
        gap = max(3, int(span*.04))
        count = max(4, int(span*.09))
        inner = np.r_[np.arange(l+gap,l+gap+count), np.arange(r-gap-count+1,r-gap+1)]
        outer_l = np.arange(l-gap-count,l-gap)
        outer_r = np.arange(r+gap+1,r+gap+count+1)
        if outer_l.min()<0 or outer_r.max()>=w:
            raise ValueError('Paper closure lacks independent lateral metal')
        if np.mean(a[y, np.r_[inner,outer_l,outer_r],3]>=128) < .8:
            foreground.append(np.nan); background.append(np.nan); contrasts.append(np.nan)
        else:
            foreground.append(float(np.median(grad[y,inner])))
            background.append(float(np.median(grad[y,np.r_[outer_l,outer_r]])))
            contrasts.append(float(np.median(gray[y,inner])-np.median(gray[y,np.r_[outer_l,outer_r]])))
        outside_fields.append((float(np.median(grad[y,outer_l])),float(np.median(grad[y,outer_r]))))
    foreground, background = np.array(foreground), np.array(background)
    signal = foreground-background
    candidates = np.arange(lo, hi+1)
    usable = np.isfinite(signal[candidates])
    if not usable.any():
        raise ValueError('Paper closure lacks opaque source/background samples')
    candidates = candidates[usable]
    peak = int(candidates[np.argmax(signal[candidates])])
    # Preserve the original raw-gradient gate and require paper-specific
    # evidence, so a shared metal highlight cannot become a paper boundary.
    if peak in (lo,hi) or foreground[peak]<.7 or signal[peak]<.7:
        raise ValueError('Physical paper end lacks independent strip-specific contrast')

    l, r = int(round(np.polyval(lf,peak))), int(round(np.polyval(rf,peak)))
    cols = np.arange(l+3,r-2)
    ys = np.arange(max(lo,peak-3), min(hi,peak+3)+1)
    bg = np.array(outside_fields)[ys]
    blend = (cols-l)/max(1,r-l)
    corrected = grad[ys[:,None],cols]-((1-blend)*bg[:,0,None]+blend*bg[:,1,None])
    index = np.argmax(corrected,axis=0)
    measured = ys[index]
    strong = (corrected[index,np.arange(len(cols))]>.7)&(grad[measured,cols]>.7)
    if strong.mean()<.75:
        raise ValueError('Paper end is not supported across the strip')
    line = np.polyfit(cols[strong],measured[strong],1)
    residual = np.abs(np.polyval(line,cols)-measured)
    good = strong&(residual<=2.5)
    if good.mean()<.75 or abs(line[0])>.15:
        raise ValueError('Paper end does not support a bounded source cross-edge')
    line = np.polyfit(cols[good],measured[good],1)
    return line, lf, rf, dict(source_end_line=line.tolist(), source_end_search=[lo, hi],
        source_end_peak=peak, source_end_gradient=float(foreground[peak]),
        source_end_specific_gradient=float(signal[peak]), source_end_support=float(good.mean()),
        source_end_maximum_residual=float(residual[good].max()))


def audit_bottom_paper_end(a, paper, proof):
    """Audit proven overprotection only; NEVER certify complete paper coverage.

    The original mask is read-only. Lack of evidence is unresolved, not acceptance.
    This audit is independent of output RGB and does not produce a replacement mask.
    """
    result = {'status': 'unresolved', 'certifies_complete_paper': False}
    try:
        line, lf, rf, edge = _source_cross_edge(a, paper, proof)
    except ValueError as exc:
        return dict(result, reason=str(exc))
    result.update(edge)
    h, w = paper.shape
    # 4 px is outside sigma=.8 / truncate=4 gradient support (ceil(3.2)).
    edge_y = float(np.polyval(line, (np.polyval(lf, edge['source_end_peak']) +
                                   np.polyval(rf, edge['source_end_peak'])) / 2))
    y0 = int(np.ceil(edge_y)) + 4
    witnesses, stats = [], []
    for y in range(y0, min(h, y0 + 3)):
        l, r = int(round(np.polyval(lf, y))), int(round(np.polyval(rf, y)))
        span = r-l+1
        width = max(8, int(np.ceil(span * .2)))
        gap = max(4, int(np.ceil(span * .08)))
        xl = np.arange(l-gap-width, l-gap)
        xr = np.arange(r+gap+1, r+gap+width+1)
        if xl.min()<0 or xr.max()>=w:
            continue
        if not (np.all(a[y,xl,3]>=240) and np.all(a[y,xr,3]>=240)):
            continue
        left = np.median(a[y,xl,:3], axis=0)
        right = np.median(a[y,xr,:3], axis=0)
        xs = np.r_[xl, xr]
        def bg(x):
            t = (np.asarray(x)-np.median(xl))/(np.median(xr)-np.median(xl))
            return left + t[...,None]*(right-left)
        exterior = np.asarray(a[y,xs,:3], float)-bg(xs)
        # Independent metal's own 99th percentile residual; one RGB quantization
        # unit is a numerical floor, not a photographic per-case tolerance.
        tolerance = max(1., float(np.quantile(np.linalg.norm(exterior,axis=1), .99)))
        inner = np.arange(l+gap, r-gap+1)
        inside_error = np.linalg.norm(np.asarray(a[y,inner,:3],float)-bg(inner),axis=1)
        metal = (inside_error <= tolerance) & (a[y,inner,3]>=240)
        # Demand a contiguous quarter-width run to reject isolated glare/noise.
        labels,n = ndimage.label(metal)
        run = max((np.flatnonzero(labels==i) for i in range(1,n+1)),
                  key=len, default=np.array([],dtype=int))
        minimum = max(8, int(np.ceil(len(inner)*.25)))
        stats.append({'row':y,'metal_noise_rgb':tolerance,'compatible_count':int(metal.sum()),
                      'longest_run':int(len(run)), 'required_run':minimum})
        if len(run)<minimum:
            continue
        xrun=inner[run]
        # Independently verify source immediately above cross-edge was paper,
        # using the same x run and its own lateral-metal noise calibration.
        ya=int(np.floor(edge_y))-4
        if ya<0 or not np.all(a[ya,np.r_[xl,xr,xrun],3]>=240):
            continue
        al=np.median(a[ya,xl,:3],axis=0); ar=np.median(a[ya,xr,:3],axis=0)
        def abg(x):
            t=(np.asarray(x)-np.median(xl))/(np.median(xr)-np.median(xl))
            return al+t[...,None]*(ar-al)
        anoise=max(1.,float(np.quantile(np.linalg.norm(a[ya,xs,:3].astype(float)-abg(xs),axis=1),.99)))
        paper_error=np.linalg.norm(a[ya,xrun,:3].astype(float)-abg(xrun),axis=1)
        if np.mean(paper_error>anoise)<.75:
            continue
        stats[-1].update(paper_row=ya,paper_noise_rgb=anoise,
                        paper_residual_median=float(np.median(paper_error)))
        witnesses.extend((int(x),y) for x in xrun if paper[y,x])
    result['metal_rows']=stats
    if not witnesses:
        return dict(result, reason='No independently supported protected metal run beyond source cross-edge')
    xy=np.asarray(witnesses)
    bbox=[int(xy[:,0].min()), int(xy[:,1].min()), int(xy[:,0].max()+1), int(xy[:,1].max()+1)]
    return dict(result,status='rejected',reason='Original paper mask protects source-confirmed metal beyond transverse paper edge',
                protected_metal_source_bbox=bbox,protected_metal_pixels=len(witnesses))


def guard_bottom_paper_end(a, paper, proof):
    audit=audit_bottom_paper_end(a,paper,proof)
    if audit['status']=='rejected':
        import json
        raise ValueError('Bottom paper overprotection: '+json.dumps(audit,ensure_ascii=False,sort_keys=True))
    return paper,dict(proof,bottom_source_audit=audit)
