"""Source-derived geometry proposals for named studio phone photographs.

No stored per-SKU rectangles, masks or accepted output coordinates are read.
These proposals remain subject to the normal independent/visual review gates.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from phone_normalizer.frame_strip import _line
from scripts.geometry_postprocess import rotate_alpha_safe


def trim_visible_canvas(image, paper=None):
    """Discard remote soft-matte fringe, retaining body and one AA pixel.

    This crops source pixels only. It never thresholds the retained alpha,
    erodes corners, resizes a face, or trims a real paper tail.
    """
    a=np.array(image.convert('RGBA'));h,w=a.shape[:2]
    core=a[...,3]>=128
    if not core.any():raise ValueError('No visible body for tight canvas')
    retained=core.copy()
    if paper is not None:
        paper=np.asarray(paper,dtype=bool)
        if paper.shape!=(h,w):raise ValueError('Paper geometry differs from canvas')
        retained |= paper&(a[...,3]>0)
    yy,xx=np.nonzero(retained)
    box=[max(0,int(xx.min())-1),max(0,int(yy.min())-1),
         min(w,int(xx.max())+2),min(h,int(yy.max())+2)]
    outside=np.ones((h,w),bool);outside[box[1]:box[3],box[0]:box[2]]=False
    if np.any(outside&retained):raise ValueError('Tight canvas would clip visible body or paper')
    result=image.crop(box)
    return result,dict(method='visible_alpha_body_bbox_with_one_pixel_antialias',
        alpha_body_threshold=128,antialias_allowance_px=1,source_size=[w,h],
        output_size=list(result.size),crop_xyxy=box,
        removed_margins_ltrb=[box[0],box[1],w-box[2],h-box[3]],
        removed_max_alpha=int(a[...,3][outside].max()) if outside.any() else 0,
        body_pixels_lost=0,paper_pixels_lost=0,retained_rgba_unchanged=True,resampled=False)


def consensus_line(x,y,tolerance,max_slope,min_fraction=.6):
    """Bounded-residual source-point consensus, with explicit outlier support."""
    x=np.asarray(x,float);y=np.asarray(y,float)
    if len(x)<12 or np.ptp(x)<4:raise ValueError('Too few independent edge samples')
    indices=np.unique(np.linspace(0,len(x)-1,min(24,len(x))).astype(int));best=None
    for i in indices:
        for j in indices:
            if x[j]-x[i]<np.ptp(x)*.3:continue
            slope=(y[j]-y[i])/(x[j]-x[i])
            if abs(slope)>max_slope:continue
            line=np.array([slope,y[i]-slope*x[i]])
            keep=np.abs(np.polyval(line,x)-y)<=tolerance
            for _ in range(2):
                if keep.sum()<12:break
                line=np.polyfit(x[keep],y[keep],1)
                keep=np.abs(np.polyval(line,x)-y)<=tolerance
            if keep.mean()<min_fraction or keep.sum()<12:continue
            if not np.any(keep&(x<x.min()+np.ptp(x)*.2)) or not np.any(keep&(x>x.max()-np.ptp(x)*.2)):continue
            error=float(np.percentile(np.abs(np.polyval(line,x[keep])-y[keep]),95))
            score=(int(keep.sum()),-error)
            if best is None or score>best[0]:best=(score,line,keep,error)
    if best is None:raise ValueError('No bounded-residual source edge consensus')
    return best[1],best[2],best[3]


def principal(alpha, threshold=128):
    labels, n = ndimage.label(alpha >= threshold)
    if not n:
        raise ValueError('No opaque phone foreground')
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    return labels == sizes.argmax()


def _plate_axis_strict(rgba,raw_rgb,rows):
    """Deskew from independently visible long sides, never a stand contact."""
    mask=principal(rgba[...,3]);h,w=mask.shape;ys=rows[int(len(rows)*.2):int(len(rows)*.8)];fits=[];proof=[]
    gradients=np.linalg.norm(np.gradient(ndimage.gaussian_filter1d(raw_rgb.astype(float),.8,axis=1),axis=1),axis=2)
    for side,values in [('left',np.argmax(mask[ys],axis=1)),('right',w-1-np.argmax(mask[ys,::-1],axis=1))]:
        line,keep,error=consensus_line(ys+.5,values.astype(float),2.5,np.tan(np.deg2rad(4)))
        xx=np.rint(np.polyval(line,ys+.5)[:,None]+np.arange(-4,5)).astype(int);xx=np.clip(xx,0,w-1)
        g=gradients[ys[:,None],xx];observed=xx[np.arange(len(ys)),np.argmax(g,axis=1)]
        rgb,rkeep,rerror=consensus_line(ys+.5,observed.astype(float),2.5,np.tan(np.deg2rad(4)))
        discrepancy=float(np.max(np.abs(np.polyval(rgb,ys[[0,-1]])-np.polyval(line,ys[[0,-1]]))))
        orientation_error=float(abs(rgb[0]-line[0])*np.ptp(ys))
        source_supported=g.max(axis=1)>3
        ends=max(1,int(len(ys)*.2))
        if min(source_supported.mean(),source_supported[:ends].mean(),source_supported[-ends:].mean())<.6:
            raise ValueError('Long body side lacks two-ended raw RGB edge support')
        # The silhouette supplies the axis. Raw RGB independently witnesses
        # the edge in its bounded neighbourhood; its brightest bevel can move
        # between adjacent facets and is not itself a second body silhouette.
        fits.append(line);proof.append(dict(side=side,alpha_line=line.tolist(),raw_rgb_line=rgb.tolist(),
            alpha_residual_p95_px=error,raw_rgb_residual_p95_px=rerror,raw_rgb_alpha_discrepancy_px=discrepancy,
            orientation_discrepancy_over_shaft_px=orientation_error,raw_rgb_supported_fraction=float(source_supported.mean())))
    slope=float(np.mean([v[0] for v in fits]));return -slope,dict(method='independent_long_side_axis_v1',sides=proof,
        rotation_ccw_degrees=float(np.degrees(np.arctan(-slope))))



def plate_axis(rgba,raw_rgb,rows,*,exclude_front_controls=False):
    """Keep ordinary source proof; recover one ragged matte side from raw RGB.

    An independently fitted opposite silhouette and both actual photograph
    edges are mandatory. This changes the source of the axis proposal, not
    the 2.5-pixel raw edge residual or the four-degree rotation search.
    """
    try:return _plate_axis_strict(rgba,raw_rgb,rows)
    except ValueError as strict_error:
        original_error=str(strict_error)
    mask=principal(rgba[...,3]);h,w=mask.shape
    ys=rows[int(len(rows)*.2):int(len(rows)*.8)];span=np.ptp(ys)
    values=[np.argmax(mask[ys],axis=1),w-1-np.argmax(mask[ys,::-1],axis=1)]
    alpha=[]
    for value in values:
        try:
            line,keep,error=consensus_line(ys+.5,value.astype(float),2.5,np.tan(np.deg2rad(4)))
            alpha.append((line,keep,error))
        except ValueError:alpha.append(None)
    if sum(v is not None for v in alpha)!=1:raise ValueError(original_error)
    anchor=next(v[0] for v in alpha if v is not None)
    gradients=np.linalg.norm(np.gradient(ndimage.gaussian_filter1d(raw_rgb.astype(float),.8,axis=1),axis=1),axis=2)
    proof=[];axis=[]
    for side,value,fit in zip(('left','right'),values,alpha):
        if fit is None:
            proposal=np.polyfit(ys+.5,value.astype(float),1)
            radius=8
            centers=np.polyval(proposal,ys+.5)
        else:
            proposal,_,_=fit;radius=4;centers=value
        xx=np.rint(centers[:,None]+np.arange(-radius,radius+1)).astype(int);xx=np.clip(xx,0,w-1)
        strength=gradients[ys[:,None],xx]
        observed=xx[np.arange(len(ys)),np.argmax(strength,axis=1)]
        rgb,keep,error=consensus_line(ys+.5,observed.astype(float),2.5,np.tan(np.deg2rad(4)))
        supported=strength.max(axis=1)>3;ends=max(1,int(len(ys)*.2))
        if min(supported.mean(),supported[:ends].mean(),supported[-ends:].mean())<.6:
            raise ValueError('Fallback body axis lacks two-ended raw RGB edge support')
        agreement=float(abs(rgb[0]-anchor[0])*span)
        if agreement>3:raise ValueError('Raw body axis disagrees with independent opposite silhouette')
        alpha_distance=float(np.percentile(abs(np.polyval(rgb,ys+.5)-value),95))
        straight_sections = None
        if fit is None and alpha_distance>radius+1:
            # The silhouette follows protruding keys. Fit only extended bare
            # metal sections, witnessed by the photograph and the opposite rail.
            outward=(np.polyval(rgb,ys+.5)-value)*(1 if side=='left' else -1)
            if not exclude_front_controls or np.percentile(outward,5)<-2.5:
                raise ValueError('Raw body axis exceeds bounded matte neighborhood')
            labels,count=ndimage.label(np.abs(outward)<=2.5)
            spans=[run[0] for run in ndimage.find_objects(labels) if run is not None
                   and run[0].stop-run[0].start>=max(12,int(.015*len(ys)))]
            bare=np.zeros(len(ys),bool)
            for run in spans:
                bare[run]=True
            if len(spans)<2 or bare.mean()<.35 or np.ptp(ys[bare])<.6*span:
                raise ValueError('Too little straight front rail outside the controls')
            rgb,keep,error=consensus_line(ys[bare]+.5,observed[bare].astype(float),2.5,np.tan(np.deg2rad(4)))
            alpha_distance=float(np.percentile(abs(np.polyval(rgb,ys[bare]+.5)-value[bare]),95))
            agreement=float(abs(rgb[0]-anchor[0])*span)
            if alpha_distance>2.5 or agreement>3 or any(supported[run].mean()<.6 for run in spans):
                raise ValueError('Bare front rail disagrees with the photograph or opposite silhouette')
            straight_sections=[[int(ys[run.start]),int(ys[run.stop-1])] for run in spans]
        axis.append(rgb if fit is None else fit[0])
        proof.append(dict(side=side,alpha_consensus_available=fit is not None,alpha_proposal=proposal.tolist(),
            raw_rgb_line=rgb.tolist(),raw_rgb_residual_p95_px=error,raw_rgb_supported_fraction=float(supported.mean()),
            source_consensus_fraction=float(keep.mean()),source_search_radius_px=radius,
            opposite_alpha_axis_disagreement_over_shaft_px=agreement,alpha_distance_p95_px=alpha_distance))
        if straight_sections:
            proof[-1]['straight_sections_excluding_controls']=straight_sections
    slope=float(np.mean([v[0] for v in axis]))
    return -slope,dict(method='one_opposite_silhouette_and_two_raw_sides_v1',sides=proof,
        original_failure=original_error,rotation_ccw_degrees=float(np.degrees(np.arctan(-slope))),
        raw_source_residual_limit_px=2.5,independent_opposite_axis_limit_over_shaft_px=3,
        source_RGB_edited=False,alpha_edited_for_axis=False)


def fixed_slope_consensus(x,y,slope,tolerance=3):
    """Estimate a physical cross-edge offset after locking the body axis."""
    x=np.asarray(x,float);y=np.asarray(y,float);intercepts=y-slope*x
    if len(x)<20:raise ValueError('Too few independent cross-edge samples')
    base=int(np.floor(intercepts.min()));bins=np.rint(intercepts-base).astype(int)
    center=base+ndimage.gaussian_filter1d(np.bincount(bins).astype(float),1).argmax()
    keep=np.abs(intercepts-center)<=tolerance
    if keep.mean()<.6 or not np.any(keep&(x<x.min()+np.ptp(x)*.2)) or not np.any(keep&(x>x.max()-np.ptp(x)*.2)):
        raise ValueError('Cross-edge lacks two-ended body-axis consensus')
    offset=float(np.median(intercepts[keep]));residual=np.abs(intercepts-offset)
    keep=residual<=tolerance
    if keep.mean()<.6:raise ValueError('Cross-edge source support changed after fitting')
    return np.array([slope,offset]),keep,float(np.percentile(residual[keep],95))


def _top_corner_sag_model(mask, wide_rows, max_taper_px=24.0):
    """Fit left/right plate sides (allowing mild trapezoid perspective taper)
    and build normalized top-corner sag lookup tables."""
    h, w = mask.shape
    ys = wide_rows[int(len(wide_rows) * .2):int(len(wide_rows) * .8)]
    left = np.argmax(mask[ys], axis=1).astype(float)
    right = (w - 1 - np.argmax(mask[ys, ::-1], axis=1)).astype(float)
    l, _, _ = consensus_line(ys + .5, left, 2.5, np.tan(np.deg2rad(4)))
    r, _, _ = consensus_line(ys + .5, right, 2.5, np.tan(np.deg2rad(4)))
    taper = float(abs(l[0] - r[0]) * len(ys))
    if taper > max_taper_px:
        raise ValueError('Plate sides taper exceeds studio capture bound')
    slope = float((l[0] + r[0]) / 2)
    cols = np.arange(max(0, int(min(left.min(), np.polyval(l, wide_rows[0])) - 15)),
                     min(w, int(max(right.max(), np.polyval(r, wide_rows[-1])) + 16)))
    cols = cols[mask[:, cols].any(axis=0)]
    top_y = np.argmax(mask[:, cols], axis=0).astype(float)
    xl0 = float(np.polyval(l, wide_rows[0]))
    xr0 = float(np.polyval(r, wide_rows[0]))
    top_w = xr0 - xl0
    core = (cols > xl0 + .25 * top_w) & (cols < xr0 - .25 * top_w)
    top_line, _, top_res = fixed_slope_consensus(cols[core] + .5, top_y[core], -slope)
    if top_res > 3:
        raise ValueError('Top plate rail lacks straight consensus')
    l_top = np.polyval(l, top_y + .5)
    r_top = np.polyval(r, top_y + .5)
    w_top = np.maximum(1.0, r_top - l_top)
    d_top_l = (cols + .5 - l_top) / w_top
    d_top_r = (r_top - (cols + .5)) / w_top
    sag_top = top_y - np.polyval(top_line, cols + .5)
    grid = np.linspace(0.0, 0.26, 261)

    def side_sag(d_vals):
        valid = (d_vals >= 0.0) & (d_vals <= 0.26) & (sag_top >= -2) & (sag_top < .28 * top_w)
        if valid.sum() < 20:
            raise ValueError('Top corner lacks sufficient silhouette samples')
        order = np.argsort(d_vals[valid])
        return np.interp(grid, d_vals[valid][order], np.maximum(0.0, sag_top[valid][order]))

    return dict(
        l=l, r=r, slope=slope, taper=taper, top_line=top_line, top_w=top_w,
        cols=cols, grid=grid, sag_l=side_sag(d_top_l), sag_r=side_sag(d_top_r),
    )


def _eval_taper_corner_curve(width, base, l, r, grid, sag_l, sag_r):
    xx = np.arange(width, dtype=float) + .5
    curve = base.copy()
    for _ in range(3):
        lx = np.polyval(l, curve)
        rx = np.polyval(r, curve)
        wb = np.maximum(1.0, rx - lx)
        dl = (xx - lx) / wb
        dr = (rx - xx) / wb
        sag = np.where(dl < .25, np.interp(np.clip(dl, 0.0, .26), grid, sag_l),
              np.where(dr < .25, np.interp(np.clip(dr, 0.0, .26), grid, sag_r), 0.0))
        curve = base - sag
    return curve


def plate_plane_from_own_corners(rgba,raw_rgb,wide_rows):
    """Infer the hidden bottom plane from this phone's visible corner curves.

    Industrial front/rear captures have a symmetric rounded body. Its own top
    curve is measured, never copied as image pixels or from another SKU. The
    mirrored curve is fitted to both exposed bottom corners, not the stand.
    """
    mask=principal(rgba[...,3]);h,w=mask.shape
    ys=wide_rows[int(len(wide_rows)*.2):int(len(wide_rows)*.8)]
    left=np.argmax(mask[ys],axis=1);right=w-1-np.argmax(mask[ys,::-1],axis=1)
    l,_,_=consensus_line(ys+.5,left.astype(float),2.5,np.tan(np.deg2rad(4)))
    r,_,_=consensus_line(ys+.5,right.astype(float),2.5,np.tan(np.deg2rad(4)))
    if abs(l[0]-r[0])*len(ys)<=4:
        slope=float((l[0]+r[0])/2);normal=np.sqrt(1+slope*slope)
        uleft=float(np.percentile((left-slope*ys)/normal,75));uright=float(np.percentile((right-slope*ys)/normal,25));width=uright-uleft
        cols=np.arange(max(0,int(left.min())-12),min(w,int(right.max())+13))
        cols=cols[mask[:,cols].any(axis=0)]
        top=np.argmax(mask[:,cols],axis=0);tu=(cols-slope*top)/normal;tv=(slope*cols+top)/normal
        core=(tu>uleft+.25*width)&(tu<uright-.25*width)
        vtop=float(np.median(tv[core]));valid=(tu>=uleft)&(tu<=uright)&(tv<vtop+.28*width)
        order=np.argsort(tu[valid]);curve_u=tu[valid][order];curve_v=tv[valid][order]-vtop
        bottom=h-1-np.argmax(mask[::-1,cols],axis=0);bu=(cols-slope*bottom)/normal;bv=(slope*cols+bottom)/normal
        lc=(bu>uleft+.025*width)&(bu<uleft+.22*width);rc=(bu>uright-.22*width)&(bu<uright-.025*width)
        usable=(lc|rc)&(bottom<h-8)
        estimate=bv+np.interp(bu,curve_u,curve_v)
        usable &= (estimate-vtop>1.7*width)&(estimate-vtop<2.4*width)
        if min(np.count_nonzero(usable&lc),np.count_nonzero(usable&rc))<20:raise ValueError('Both lower corners must be exposed')
        histogram=ndimage.gaussian_filter1d(np.bincount(np.maximum(0,np.rint(estimate[usable]).astype(int))),1.)
        center=float(histogram.argmax());keep=usable&(np.abs(estimate-center)<=3)
        fractions=[float(np.count_nonzero(keep&side)/np.count_nonzero(usable&side)) for side in (lc,rc)]
        if min(fractions)<.6:raise ValueError('Upper/lower corner curves disagree')
        plane=float(np.median(estimate[keep]));residual=float(np.percentile(np.abs(estimate[keep]-plane),95))
        if residual>3:raise ValueError('Corner-derived plate plane exceeds residual limit')
        gradient=np.linalg.norm(np.gradient(ndimage.gaussian_filter1d(raw_rgb.astype(np.float32),.8,axis=0),axis=0),axis=2)
        xx=cols[keep];yy=bottom[keep]
        edge=np.max(gradient[np.clip(yy[:,None]+np.arange(-2,3),0,h-1),xx[:,None]],axis=1)
        if np.mean(edge>3)<.6:raise ValueError('Corner contour lacks independent raw RGB support')
        line=np.array([-slope,normal*plane])
        return line,dict(method='own_top_bottom_corner_curve_consensus_v1',
            own_corner_support_fractions=fractions,corner_residual_p95_px=residual,
            raw_rgb_supported_fraction=float(np.mean(edge>3)),body_u=[uleft,uright],top_v=vtop,bottom_v=plane,
            symmetric_body_assumption=True,samples=int(keep.sum()),axis_slope=slope,
            upper_curve_u=curve_u.tolist(),upper_curve_v=curve_v.tolist())

    # Mild trapezoid perspective taper (4..24 px across the shaft): mirror the
    # phone's own top-left/top-right curves along its actual fitted side rails.
    model = _top_corner_sag_model(mask, wide_rows, max_taper_px=24.0)
    cols = model['cols']; slope = model['slope']
    bottom = h - 1 - np.argmax(mask[::-1, cols], axis=0)
    l_bot = np.polyval(model['l'], bottom + .5)
    r_bot = np.polyval(model['r'], bottom + .5)
    w_bot = np.maximum(1.0, r_bot - l_bot)
    d_bot_l = (cols + .5 - l_bot) / w_bot
    d_bot_r = (r_bot - (cols + .5)) / w_bot
    lc = (d_bot_l > .025) & (d_bot_l < .20)
    rc = (d_bot_r > .025) & (d_bot_r < .20)
    usable = (lc | rc) & (bottom < h - 8)
    sag = np.where(lc, np.interp(np.clip(d_bot_l, 0.0, .26), model['grid'], model['sag_l']),
                   np.interp(np.clip(d_bot_r, 0.0, .26), model['grid'], model['sag_r']))
    estimate = bottom.astype(float) + sag + slope * (cols + .5)
    top_mid = float(np.polyval(model['top_line'], w / 2))
    mid_height = estimate - slope * (w / 2) - top_mid
    usable &= (mid_height > 1.7 * model['top_w']) & (mid_height < 2.4 * model['top_w'])
    if min(np.count_nonzero(usable & lc), np.count_nonzero(usable & rc)) < 20:
        raise ValueError('Both lower corners must be exposed')
    histogram = ndimage.gaussian_filter1d(np.bincount(np.maximum(0, np.rint(estimate[usable]).astype(int))), 1.)
    center = float(histogram.argmax())
    keep = usable & (np.abs(estimate - center) <= 3.5)
    fractions = [float(np.count_nonzero(keep & side) / np.count_nonzero(usable & side)) for side in (lc, rc)]
    if min(fractions) < .6:
        raise ValueError('Upper/lower corner curves disagree')
    plane = float(np.median(estimate[keep]))
    residual = float(np.percentile(np.abs(estimate[keep] - plane), 95))
    if residual > 3.2:
        raise ValueError('Corner-derived plate plane exceeds residual limit')
    gradient = np.linalg.norm(np.gradient(ndimage.gaussian_filter1d(raw_rgb.astype(np.float32), .8, axis=0), axis=0), axis=2)
    xx = cols[keep]; yy = bottom[keep]
    edge = np.max(gradient[np.clip(yy[:, None] + np.arange(-2, 3), 0, h - 1), xx[:, None]], axis=1)
    if np.mean(edge > 3) < .6:
        raise ValueError('Corner contour lacks independent raw RGB support')
    line = np.array([-slope, plane])
    # If the phone's own exposed flat bottom columns consistently sit within a
    # few pixels of the mirrored top-corner plane (mild camera pitch), anchor
    # the intercept to their directly observed raw RGB outer boundary so we
    # neither step inward above real body nor leave soft stand alpha below it.
    xl_b = float(np.polyval(model['l'], wide_rows[-1]))
    xr_b = float(np.polyval(model['r'], wide_rows[-1]))
    wb_b = xr_b - xl_b
    fx = np.arange(max(0, int(xl_b + .18 * wb_b)), min(w, int(xr_b - .18 * wb_b)))
    if len(fx) >= 20:
        fb = h - 1 - np.argmax(mask[::-1, fx], axis=0)
        exp_flat = fb < h - 8
        if exp_flat.sum() >= 20:
            fx_e = fx[exp_flat]; fb_e = fb[exp_flat]
            deltas = fb_e.astype(float) - np.polyval(line, fx_e + .5)
            near_flat = np.abs(deltas) <= 5.0
            if near_flat.sum() >= 20 and np.percentile(np.abs(deltas[near_flat] - np.median(deltas[near_flat])), 90) <= 1.5:
                fx_n = fx_e[near_flat]; fb_n = fb_e[near_flat]
                sy = np.clip(fb_n[:, None] + np.arange(-4, 5), 0, h - 1)
                g_win = gradient[sy, fx_n[:, None]]
                rgb_y = sy[np.arange(len(fx_n)), np.argmax(g_win, axis=1)] + .5
                valid_g = (g_win.max(axis=1) > 3.0) & (np.abs(rgb_y - (fb_n + .5)) <= 3.0)
                if valid_g.sum() >= 20:
                    rgb_deltas = rgb_y[valid_g] - np.polyval(line, fx_n[valid_g] + .5)
                    if np.percentile(np.abs(rgb_deltas - np.median(rgb_deltas)), 90) <= 2.0:
                        shift = float(np.median(rgb_deltas))
                        if -3.0 <= shift <= 4.0:
                            line[1] += shift
    return line, dict(method='own_top_bottom_corner_curve_consensus_taper_v1',
        own_corner_support_fractions=fractions, corner_residual_p95_px=residual,
        raw_rgb_supported_fraction=float(np.mean(edge > 3)),
        taper_px=model['taper'], left_line=model['l'].tolist(), right_line=model['r'].tolist(),
        sag_grid=model['grid'].tolist(), sag_left=model['sag_l'].tolist(), sag_right=model['sag_r'].tolist(),
        symmetric_body_assumption=True, samples=int(keep.sum()), axis_slope=slope)


def plate_corner_contact_limit(rgba, raw_rgb, line, corner, support_columns, paper=None):
    """Remove a supported contact outside the curved body, not a rectangle.

    The existing two-corner fit supplies a geometric search neighbourhood.
    Independent raw RGB edges validate each affected corner before its local
    silhouette can change. Exposed corners, body reflections and paper stay
    unchanged. A poor correspondence is a failure, never a smaller body.
    """
    a=np.asarray(rgba);h,w=a.shape[:2];x=np.arange(w,dtype=float)
    base=np.polyval(line,x+.5)
    if 'left_line' in corner:
        l_line = np.asarray(corner['left_line'], float)
        r_line = np.asarray(corner['right_line'], float)
        grid = np.asarray(corner['sag_grid'], float)
        sag_l = np.asarray(corner['sag_left'], float)
        sag_r = np.asarray(corner['sag_right'], float)
        curve = _eval_taper_corner_curve(w, base, l_line, r_line, grid, sag_l, sag_r)
        lx = np.polyval(l_line, base); rx = np.polyval(r_line, base); wb = np.maximum(1.0, rx - lx)
        xx = x + .5
        corner_columns = ((xx > lx + .025 * wb) & (xx < lx + .24 * wb)) | ((xx > rx - .24 * wb) & (xx < rx - .025 * wb))
        left_side = xx < (lx + rx) / 2
    else:
        curve=base.copy()
        slope=float(corner['axis_slope']);normal=np.sqrt(1+slope*slope)
        u0,u1=corner['body_u'];width=u1-u0
        for _ in range(4):
            u=(x-slope*curve)/normal
            curve=normal*(corner['bottom_v']-np.interp(u,corner['upper_curve_u'],corner['upper_curve_v']))-slope*x
        corner_columns=((u>u0+.025*width)&(u<u0+.23*width))|((u>u1-.23*width)&(u<u1-.025*width))
        left_side = u < (u0 + u1) / 2
    present=a[...,3]>0
    bottom=h-1-np.argmax(present[::-1],axis=0)
    # A real exposed corner follows the fitted curve. Only a clearly attached
    # projection extends farther, so do not recut every already good corner.
    support_columns=np.asarray(support_columns,dtype=bool).reshape(-1)
    if support_columns.shape!=(w,):raise ValueError('Stand projection columns differ from source width')
    affected=corner_columns&(bottom>curve+4)&(base>curve+4)&support_columns&present.any(axis=0)
    if paper is not None:
        affected &= ~np.any(np.asarray(paper,dtype=bool)&(np.arange(h)[:,None]>=curve[None,:]-3),axis=0)
    limit=base.copy();details=[]
    if not affected.any():return limit,dict(method='raw_supported_local_corner_contact',changed_columns=[],corners=[])
    gradient=np.linalg.norm(np.gradient(ndimage.gaussian_filter1d(raw_rgb.astype(np.float32),.8,axis=0),axis=0),axis=2)
    for name,side in [('left',left_side),('right',~left_side)]:
        cols=np.flatnonzero(affected&side)
        if not len(cols):continue
        # Validate a full local neighbourhood, including exposed witnesses on
        # both sides of the contact. Do not fit the stand's distant white rail.
        check=np.arange(max(0,cols.min()-8),min(w,cols.max()+9))
        ys=np.rint(curve[check,None]+np.arange(-3,4)).astype(int)
        values=gradient[np.clip(ys,0,h-1),check[:,None]]
        peak=np.argmax(values,axis=1);strength=values[np.arange(len(check)),peak]
        measured=ys[np.arange(len(check)),peak].astype(float)
        supported=(strength>3)&(peak>0)&(peak<6)
        if supported.mean()<.6 or supported.sum()<12:
            raise ValueError('Corner contact lacks independent raw RGB boundary support: '+name)
        displacement=measured[supported]-curve[check[supported]]
        offset=float(np.median(displacement));residual=float(np.percentile(np.abs(displacement-offset),95))
        if abs(offset)>2.5 or residual>2.5:
            raise ValueError('Corner contact raw RGB disagrees with own body curve: '+name)
        # Retain a full source pixel beyond the independently measured edge.
        # This is a conservative alpha-only trim, never synthesized RGB.
        local=curve[cols]+max(0.,offset)+1.
        limit[cols]=np.minimum(base[cols],local)
        details.append(dict(side=name,columns=[int(cols.min()),int(cols.max())],
            rgb_support_fraction=float(supported.mean()),rgb_offset_px=offset,
            rgb_residual_p95_px=residual,source_samples=int(supported.sum())))
    return limit,dict(method='raw_supported_local_corner_contact',changed_columns=np.flatnonzero(limit<base).tolist(),corners=details)


def plate_corner_wedge_limit(rgba, raw_rgb, wide_rows, line, support_columns, paper=None):
    """Trim stand wedges tucked under rounded bottom corners using the phone's
    own top-corner curve and isotropic raw RGB edge verification.

    Applies strictly to stand-adjacent bottom-corner columns and retains a
    conservative outer envelope (never cuts inward above the verified bezel).
    """
    a = np.asarray(rgba); h, w = a.shape[:2]
    mask = principal(a[..., 3])
    xx = np.arange(w, dtype=float) + .5
    base = np.polyval(line, xx)
    try:
        model = _top_corner_sag_model(mask, wide_rows, max_taper_px=24.0)
    except ValueError:
        return base, None
    curve = _eval_taper_corner_curve(w, base, model['l'], model['r'],
                                     model['grid'], model['sag_l'], model['sag_r'])
    support_columns = np.asarray(support_columns, dtype=bool).reshape(-1)
    if support_columns.shape != (w,):
        return base, None
    present = a[..., 3] > 0
    bottom = h - 1 - np.argmax(present[::-1], axis=0)
    lx = np.polyval(model['l'], base); rx = np.polyval(model['r'], base); wb = np.maximum(1.0, rx - lx)
    corner_columns = ((xx > lx + .025 * wb) & (xx < lx + .24 * wb)) | ((xx > rx - .24 * wb) & (xx < rx - .025 * wb))
    affected = corner_columns & (bottom > curve + 2.5) & (base > curve + 2.0) & support_columns & present.any(axis=0)
    if paper is not None and paper.any():
        affected &= ~np.any(np.asarray(paper, dtype=bool) & (np.arange(h)[:, None] >= curve[None, :] - 3), axis=0)
    if not affected.any():
        return base, dict(method='isotropic_corner_wedge_trim_v1', changed_columns=[], corners=[])
    sm = ndimage.gaussian_filter(raw_rgb.astype(np.float32), (.8, .8, 0))
    grad = np.hypot(np.linalg.norm(np.gradient(sm, axis=1), axis=2),
                    np.linalg.norm(np.gradient(sm, axis=0), axis=2))
    limit = base.copy(); details = []
    left_side = xx < (lx + rx) / 2
    offs = np.arange(-5, 9)
    for name, side in (('left', left_side), ('right', ~left_side)):
        cols = np.flatnonzero(affected & side)
        if not len(cols):
            continue
        check = np.arange(max(0, cols.min() - 8), min(w, cols.max() + 9))
        ys = np.rint(curve[check, None] + offs).astype(int)
        vals = grad[np.clip(ys, 0, h - 1), check[:, None]]
        lmax = (vals >= np.roll(vals, 1, axis=1)) & (vals >= np.roll(vals, -1, axis=1))
        lmax[:, [0, -1]] = False
        lmax &= (vals > 4.0) & (vals >= .30 * vals.max(axis=1, keepdims=True))
        dist = np.abs(ys - curve[check, None])
        pk = np.argmin(np.where(lmax, dist, 999.0), axis=1)
        supported = lmax.any(axis=1)
        if supported.mean() < .70 or supported.sum() < 10:
            continue
        disp = ys[np.arange(len(check)), pk][supported] - curve[check[supported]]
        offset = float(np.median(disp))
        if abs(offset) > 3.2:
            continue
        local = curve + max(0.0, offset) + .5
        # Continue smoothly from the outer affected wedge to where local meets base
        # so the curved corner blends continuously into the flat rail without a step.
        if name == 'left':
            arc = np.flatnonzero(side & (xx >= cols.min()) & (local < base) & present.any(axis=0))
        else:
            arc = np.flatnonzero(side & (xx <= cols.max()) & (local < base) & present.any(axis=0))
        if paper is not None and paper.any():
            paper_cols = np.any(np.asarray(paper, dtype=bool) & (np.arange(h)[:, None] >= curve[None, :] - 3), axis=0)
            arc = arc[~paper_cols[arc]]
        target_cols = np.union1d(cols, arc)
        limit[target_cols] = np.minimum(base[target_cols], local[target_cols])
        details.append(dict(side=name, columns=[int(target_cols.min()), int(target_cols.max())],
                            rgb_support_fraction=float(supported.mean()),
                            rgb_offset_px=offset, source_samples=int(supported.sum())))
    return limit, dict(method='isotropic_corner_wedge_trim_v1',
                       changed_columns=np.flatnonzero(limit < base).tolist(), corners=details)


def line_samples(alpha, lo, hi):
    mask = principal(alpha)
    ys, xs = np.nonzero(mask)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    width = x1-x0
    xx = np.concatenate([np.arange(int(x0+lo*width), int(x0+hi*width)),
                         np.arange(int(x1-hi*width), int(x1-lo*width))])
    xx = xx[mask[:, xx].any(axis=0)]
    top = np.argmax(mask[:, xx], axis=0).astype(float)
    bottom = mask.shape[0]-np.argmax(mask[::-1, xx], axis=0).astype(float)
    return xx, top, bottom, (x0, x1)


def frame_rails(rgba,source_rgb=None):
    """Fit independently exposed rails; reject inconsistent endcap estimates."""
    alpha = rgba[..., 3]
    candidates = []
    for lo, hi in ((.04,.08),(.06,.10),(.08,.12),(.06, .12), (.10, .17), (.12,.18),(.15, .23), (.21, .29)):
        x, top, bottom, extent = line_samples(alpha, lo, hi)
        b, keep, residual = _line(x+.5, bottom)
        support = min(keep[:len(x)//2].mean(), keep[len(x)//2:].mean())
        if residual <= 3 and support >= .6:
            candidates.append((b, residual, [lo, hi]))
    if not candidates:
        raise ValueError('No consistent exposed lower-rail estimate')
    endpoints = np.array(extent, float)
    levels = np.array([np.polyval(c[0], endpoints) for c in candidates])
    # Clear, straight long-edge estimates must agree. Curved endcaps are not
    # accepted merely because they produce a plausible thin rectangle.
    median = np.median(levels, axis=0)
    valid = np.max(np.abs(levels-median), axis=1) <= 3
    disagreement=not valid.any()
    if disagreement:
        # Competing curved-endcap proposals are ranked by their observed
        # residual, then checked independently against the raw RGB edge below.
        # Record disagreement; it remains visible in the mandatory review.
        best_index=min(range(len(candidates)),key=lambda i:candidates[i][1])
        valid[best_index]=True
    bottom = np.median(np.array([c[0] for c, k in zip(candidates, valid) if k]), axis=0)
    # A separate measurement uses RGB gradients in the original photograph,
    # rather than relaxing the alpha residual/support limits on dark frames.
    best=next(c for c,k in zip(candidates,valid) if k)
    xx,_,_,_=line_samples(alpha,*best[2])
    if source_rgb is None:raise ValueError('Independent raw photograph required for automatic rails')
    if source_rgb.shape!=rgba[...,:3].shape:raise ValueError('Raw photograph dimensions differ from matte')
    colors=ndimage.gaussian_filter1d(source_rgb.astype(np.float32),.8,axis=0)
    gradients=np.linalg.norm(np.gradient(colors,axis=0),axis=2)
    yy=np.polyval(bottom,xx+.5)-.5
    offsets=np.arange(-6,7);rows=np.rint(yy[:,None]+offsets).astype(int)
    strength=gradients[np.clip(rows,0,alpha.shape[0]-1),xx[:,None]]
    peak=rows[np.arange(len(xx)),np.argmax(strength,axis=1)]
    measured,k,r=_line(xx+.5,peak+.5)
    rgb_error=float(np.max(np.abs(np.polyval(measured,endpoints)-np.polyval(bottom,endpoints))))
    if r>3 or min(k[:len(k)//2].mean(),k[len(k)//2:].mean())<.6:
        raise ValueError('Source RGB and alpha lower-rail measurements disagree')
    # A soft segmentation transition is often inside the photographed edge.
    # Its independent RGB outer edge may extend the proposal, never erode it.
    delta=np.polyval(measured,endpoints)-np.polyval(bottom,endpoints)
    if np.any(delta < -3) or np.any(delta > 7):
        raise ValueError('RGB outer edge is outside the bounded proposal neighbourhood')
    if np.all(delta>=0):bottom=measured
    mask = principal(alpha); x0, x1 = extent; width=x1-x0
    x = np.arange(int(x0+.04*width), int(x1-.04*width))
    tops = np.argmax(mask[:, x], axis=0)
    intercept = tops-bottom[0]*(x+.5)
    # The inward supported plateau is the front-facing frame surface; rear
    # camera projections create outward plateaus and are recorded separately.
    hist = ndimage.gaussian_filter1d(np.bincount(np.maximum(0,np.rint(intercept).astype(int))), 1.)
    peaks = np.where((hist >= np.roll(hist,1)) & (hist >= np.roll(hist,-1)) & (hist > max(8,.025*width)))[0]
    plausible = [int(p) for p in peaks if .035*width < bottom[1]-p < .25*width]
    if not plausible:
        raise ValueError('No supported thin-frame upper rail')
    level = max(plausible)
    keep = np.abs(intercept-level) <= 3
    if keep.sum() < .08*width:
        raise ValueError('Insufficient source support for the upper rail')
    top = np.array([bottom[0], np.median(intercept[keep])])
    if max(abs(top[0]),abs(bottom[0])) > np.tan(np.deg2rad(4)):
        raise ValueError('Frame rotation exceeds the studio profile')
    return top, bottom, dict(method='independent_exposed_rail_consensus_v1',
        x_extent=list(extent), estimates=[dict(line=c[0].tolist(), residual_p95=c[1], bands=c[2], accepted=bool(k)) for c,k in zip(candidates,valid)],
        upper_support_pixels=int(keep.sum()), upper_plateaus=plausible,
        rgb_boundary_error_px=rgb_error,rgb_boundary_residual_p95_px=r,
        competing_alpha_rail_proposals=disagreement,
        top_rail=top.tolist(), bottom_rail=bottom.tolist())


def _bottom_upper_body_rail(rgba, raw, top, bottom, extent):
    """Recover the near rail when a broad raised object fills the central silhouette."""
    x0,x1 = extent
    width = x1-x0
    alpha = rgba[...,3]
    mask = principal(alpha)
    fraction = (np.arange(x0,x1)-x0)/width
    x = np.arange(x0,x1)
    transition = np.argmax(mask[:,x], axis=0)
    ends = ((fraction>.025)&(fraction<.045), (fraction>.955)&(fraction<.975))
    if any(use.sum() < 3 for use in ends):
        return top, None
    gaps = [np.median(transition[use]-np.polyval(top,x[use]+.5)) for use in ends]
    if min(gaps) < .025*width or abs(gaps[0]-gaps[1]) > .012*width:
        return top, None
    grad = np.linalg.norm(np.gradient(ndimage.gaussian_filter1d(raw.astype(np.float32),.8,axis=0),
                                      axis=0),axis=2)
    seed = top.copy()
    seed[1] += np.median(gaps)
    tracks = []
    # Disjoint end and middle windows on both sides of the port. The port
    # and the central printed label never enter the fit.
    for lo,hi in ((.025,.055),(.10,.18),(.24,.32),(.66,.74),(.82,.90),(.945,.975)):
        cols = x[(fraction>=lo)&(fraction<hi)]
        predicted = np.polyval(seed,cols+.5)
        rows = np.rint(predicted[:,None]+np.arange(-9,10)).astype(int)
        if rows.min()<0 or rows.max()>=len(alpha):
            return top,None
        support = grad[rows,cols[:,None]]
        peak = np.argmax(support,axis=1)
        good = (peak>0)&(peak<18)&(support.max(axis=1)>3)
        if good.mean()<.75:
            return top,None
        tracks.append((cols[good]+.5,rows[good,peak[good]]+.5))
    line = np.polyfit([np.median(c) for c,y in tracks],[np.median(y) for c,y in tracks],1)
    residuals = []
    for c, y in tracks:
        cols = (c-.5).astype(int)
        rows = np.rint(np.polyval(line,c)[:,None]-.5+np.arange(-4,5)).astype(int)
        if rows.min() < 0 or rows.max() >= len(alpha):
            return top, None
        signal = grad[rows,cols[:,None]]
        local = (signal[:,1:-1]>=signal[:,:-2])&(signal[:,1:-1]>signal[:,2:])&(signal[:,1:-1]>3)
        local &= signal[:,1:-1]>=.5*signal.max(axis=1,keepdims=True)
        distance = abs(rows[:,1:-1]+.5-np.polyval(line,c)[:,None])
        peaks = 1+np.argmin(np.where(local,distance,np.inf),axis=1)
        seen = local.any(axis=1)
        if seen.mean()<.8:
            return top,None
        residuals.append(float(np.percentile(abs(rows[seen,peaks[seen]]+.5-np.polyval(line,c[seen])),95)))
    if max(residuals)>3 or abs(line[0]-bottom[0])*width>5:
        return top,None
    if any(abs(np.median(transition[use]-np.polyval(line,x[use]+.5)))>5 for use in ends):
        return top,None
    if np.min(np.polyval(bottom-line,[x0,x1]))<.035*width:
        return top,None
    return line, dict(method='six_disjoint_source_rgb_upper_body_windows',
                      original_rail=top.tolist(),window_residuals_px=residuals)


def frame_geometry(matte, face, stage_sink=None,source_image=None,separate_support_edge=True, *, model=None):
    a=np.array(matte.convert('RGBA')); original=a.copy()
    raw=np.array(source_image.convert('RGB')) if source_image else None
    side_button_trim=None
    if face == 'top':
        from .top_frame import TopCalibrationError, calibrate_top, trim_top_side_button, verify_top_mask
        a, side_button_trim = trim_top_side_button(a, raw)
        original = a.copy()
        matte_top = Image.fromarray(a) if side_button_trim is not None else matte
        try:
            top,bottom,proof=calibrate_top(a,raw)
        except TopCalibrationError as failure:
            if not str(failure).startswith(('TOP_CORNER_AMBIGUOUS', 'TOP_BODY_RAILS_DISAGREE')):
                raise
            from .frame_contact import recover_frame
            try:
                recovered, proof = recover_frame(matte_top, source_image, face, stage_sink)
                proof['original_failure'] = str(failure)
                if side_button_trim is not None:
                    proof['side_button_trim'] = side_button_trim
                return recovered, proof
            except ValueError as unsupported:
                if model is not None:
                    from .top_body_template import normalize_top
                    try:
                        recovered, proof = normalize_top(matte_top,source_image,model,stage_sink,
                                             previous_error=f'{failure} (source contact check: {unsupported})')
                        if side_button_trim is not None:
                            proof['side_button_trim'] = side_button_trim
                        return recovered, proof
                    except ValueError as template_error:
                        raise TopCalibrationError(f'{failure} (source contact check: {unsupported}; '
                                                  f'thickness template: {template_error})') from template_error
                raise TopCalibrationError(f'{failure} (source contact check: {unsupported})') from unsupported
        if side_button_trim is not None:
            proof['side_button_trim'] = side_button_trim
    else:
        try:
            top,bottom,proof=frame_rails(a,raw)
        except ValueError as error:
            if raw is None or str(error) not in ('No consistent exposed lower-rail estimate',
                    'Source RGB and alpha lower-rail measurements disagree'):
                raise
            from .frame_contact import recover_frame
            try:
                recovered, proof = recover_frame(matte, source_image, face, stage_sink)
            except ValueError as contact_error:
                raise ValueError(f'{error} (source contact check: {contact_error})') from contact_error
            proof['original_failure'] = str(error)
            return recovered, proof
        if face == 'bottom':
            top, body_rail = _bottom_upper_body_rail(a,raw,top,bottom,proof['x_extent'])
            if body_rail:
                proof.update(upper_body_evidence=body_rail, top_rail=top.tolist())
    x=np.arange(a.shape[1])[None,:]+.5; y=np.arange(a.shape[0])[:,None]
    # Preserve native curved endcaps with their segmentation alpha. Rail
    # masking removes only projections/support, with an outward allowance.
    upper=np.ceil(np.polyval(top,x)) if face == 'top' else np.polyval(top,x)-1.
    # The old extra outward pixel can retain the stand's white contact seam.
    # Use the independently agreed physical lower edge on side strips only;
    # retain fractional pixel coverage rather than eroding the photographed
    # body. Ambiguous source/alpha agreement keeps the conservative proposal
    # and remains explicitly unresolved for geometry review.
    lower_allowance = 1.
    if face in ('left', 'right') and separate_support_edge:
        agreed = (proof['rgb_boundary_error_px'] <= 1. and
                  proof['rgb_boundary_residual_p95_px'] <= 1.)
        if agreed:
            lower_allowance = 0.
        proof['support_edge_separation'] = dict(
            method='independently_agreed_side_lower_edge_fractional_coverage',
            status='candidate_requires_geometry_review' if agreed else 'unresolved_source_alpha_disagreement',
            outward_allowance_px=lower_allowance,
            source_rgb_painted=False, phone_core_inward_offset_px=0.)
    lower=np.polyval(bottom,x)+lower_allowance
    coverage=np.clip(np.minimum(y+1,lower)-np.maximum(y,upper),0,1)
    a[...,3]=np.minimum(a[...,3],np.rint(255*coverage).astype(np.uint8))
    if face == 'top':
        proof['automatic_checks']=verify_top_mask(original,a,top,bottom,proof)
    if stage_sink:
        stage_sink('source_matte',matte);stage_sink('after_support',Image.fromarray(a))
    angle=float(np.degrees(np.mean(np.arctan([top[0],bottom[0]])))) if face == 'top' else float(np.degrees(np.arctan(bottom[0])))
    rotated=Image.fromarray(rotate_alpha_safe(a,angle))
    if stage_sink:stage_sink('after_rotation',rotated);stage_sink('after_crop',rotated)
    box=rotated.getchannel('A').getbbox()
    if box is None:raise ValueError('Empty frame after source-derived geometry')
    result=rotated.crop(box)
    if face == 'top':
        from .top_frame import verify_top_transform
        proof['transform_checks']=verify_top_transform(original,rotated,result,top,bottom,angle,box)
    if face in ('left','right'):
        result=result.transpose(Image.Transpose.ROTATE_90 if face=='left' else Image.Transpose.ROTATE_270)
    return result,dict(**proof,rotation_ccw_degrees=angle,canvas_trim_xyxy=list(box),
        source_size=list(matte.size),output_size=list(result.size),rgb_painted=False,
        review_reasons=['extrapolated_frame_edges','complete_outline_requires_review'])


def exposed_lower_plane(mask, raw_gradient, x0, x1, last_body_row, paper=None, body_slope=None):
    """Fit the photographed cross-edge independently of the long-side axis.

    Mild perspective makes those axes non-perpendicular. Require both exposed
    halves and independent RGB support; never fit the central stand or paper.
    """
    h,w=mask.shape;bw=x1-x0;options=[]
    max_slope_delta = max(4.0, .0045 * bw) / max(1.0, bw)
    for lo,hi in ((.12,.28),(.14,.30),(.16,.30)):
        x=np.r_[np.arange(int(x0+lo*bw),int(x0+hi*bw)),
                np.arange(int(x1-hi*bw),int(x1-lo*bw))]
        bottom=h-np.argmax(mask[::-1,x],axis=0)
        valid=(bottom<h-8)&(bottom>last_body_row-.12*bw)
        if paper is not None:valid &= ~paper[np.clip(bottom-1,0,h-1),x]
        x,bottom=x[valid],bottom[valid]
        halves=[x<(x0+x1)/2,x>=(x0+x1)/2]
        if min(int(s.sum()) for s in halves)<20:continue
        try:
            alpha,ak,ae=consensus_line(x+.5,bottom.astype(float),3,np.tan(np.deg2rad(4)))
            yy=np.rint(np.polyval(alpha,x+.5)[:,None]-.5+np.arange(-4,5)).astype(int)
            strength=raw_gradient[np.clip(yy,0,h-1),x[:,None]]
            rgb_y=yy[np.arange(len(x)),np.argmax(strength,axis=1)]+.5
            rgb,rk,re=consensus_line(x+.5,rgb_y,3,np.tan(np.deg2rad(4)))
        except ValueError:continue
        if body_slope is not None and (abs(rgb[0] - body_slope) > max_slope_delta or
                                       abs(alpha[0] - body_slope) > max_slope_delta):
            continue
        fractions=[float((ak&rk&(strength.max(axis=1)>3))[s].mean()) for s in halves]
        if min(fractions)<.6:continue
        disagreement=float(np.max(np.abs(np.polyval(rgb,[x0,x1])-np.polyval(alpha,[x0,x1]))))
        if disagreement>3:continue
        options.append((ae+re,alpha,rgb,len(x),[lo,hi],dict(
            method='independent_exposed_lower_plane_rgb_alpha',
            half_support_fractions=fractions,alpha_residual_p95_px=ae,
            raw_rgb_residual_p95_px=re,rgb_alpha_disagreement_px=disagreement)))
    if not options:raise ValueError('Lower plane lacks two-sided raw/alpha support')
    return min(options,key=lambda item:item[0])


def front_support_coverage(alpha,limit,support_columns):
    """Keep measured source coverage, without a second outer safety margin."""
    yy=np.arange(alpha.shape[0])[:,None]
    coverage=np.clip(np.asarray(limit)-yy,0,1)
    selected=np.broadcast_to(support_columns,alpha.shape)
    result=np.array(alpha,copy=True)
    result[selected]=np.minimum(result[selected],np.rint(255*coverage[selected]).astype(np.uint8))
    return result


def front_lower_closure(mask,raw_rgb,x0,x1,last_body_row,source_alpha=None):
    """Conserve the observed alpha rail; independently verify glass closure.

    Chrome can have several stronger internal RGB transitions. This fallback
    does not select an inward reflection as the silhouette or paint a rim.
    """
    h,w=mask.shape;bw=x1-x0
    x=np.r_[np.arange(int(x0+.30*bw),int(x0+.45*bw)),
            np.arange(int(x0+.55*bw),int(x0+.70*bw))]
    edge=h-np.argmax(mask[::-1,x],axis=0)
    valid=(edge<h-16)&(abs(edge-last_body_row)<=.04*bw)
    x,edge=x[valid],edge[valid]
    halves=[x<(x0+x1)/2,x>=(x0+x1)/2]
    if min(int(s.sum()) for s in halves)<20:raise ValueError('Front lower closure lacks two exposed alpha gaps')
    line,kept,residual=consensus_line(x+.5,edge.astype(float),3,np.tan(np.deg2rad(4)))
    if min(float(kept[s].mean()) for s in halves)<.8:raise ValueError('Front lower alpha gaps disagree')
    # Retain every near-body opaque source row rather than cutting at a
    # stronger inner glass/chrome reflection. Far support projections cannot
    # authorize the rail position.
    xx=np.arange(int(x0+.1*bw),int(x1-.1*bw));bottom=h-np.argmax(mask[::-1,xx],axis=0)
    witnesses=abs(bottom-np.polyval(line,xx+.5))<=6
    if witnesses.sum()<.4*bw:raise ValueError('Front lower conservation witnesses are insufficient')
    # Conservation and the exposed-edge witnesses must refer to the same
    # phone component. Disconnected matte specks cannot extend its lower end.
    core=mask if source_alpha is None else (np.asarray(source_alpha)>=240)&mask
    core_bottom=h-np.argmax(core[::-1,xx],axis=0)
    outward=max(0.,float(np.max(core_bottom[witnesses]-np.polyval(line,xx[witnesses]+.5))))
    if outward>3:raise ValueError('Front lower contour needs unsupported outward extension')
    line[1]+=outward
    body_row=int(round(np.polyval(line,(x0+x1)/2)-10))
    body_width_fraction=float(mask[body_row,x0:x1].mean())
    if body_width_fraction<.6:raise ValueError('Front lower closure is a narrow support rather than the phone')
    sample=np.arange(int(x0+.2*bw),int(x1-.2*bw));yy=np.rint(np.polyval(line,sample+.5)).astype(int)
    if yy.min()<26 or yy.max()+16>=h:raise ValueError('Front lower RGB closure is clipped')
    gray=np.mean(raw_rgb.astype(float),axis=2)
    inside=np.median(gray[yy[:,None]+np.arange(-25,-14),sample[:,None]],axis=1)
    outside=np.median(gray[yy[:,None]+np.arange(5,16),sample[:,None]],axis=1)
    groups=np.array_split(outside-inside,2);fractions=[float(np.mean(v>40)) for v in groups]
    if min(fractions)<.90:raise ValueError('Front lower lacks independent dark glass to background closure')
    proof=dict(method='front_alpha_plateau_with_independent_dark_glass_closure',
        alpha_residual_p95_px=residual,alpha_samples=len(x),source_conservation_outward_px=outward,source_core_alpha_threshold=240,
        glass_background_contrast_medians=[float(np.median(v)) for v in groups],
        dark_glass_closure_fractions=fractions,observed_body_width_fraction=body_width_fraction,
        source_conservation_component='principal_phone',
        RGB_does_not_set_inward_cut=True)
    return 2*residual,line,line,len(x),[.30,.45],proof


def paper_without_stand(paper, limit, *, allow_tail=True):
    """A sticker tail ends just past the body; white paper-like pixels beyond
    the lower body edge that run on to the photograph's bottom border are the
    stand (paper detection can flood into it) and get no paper protection.
    When allow_tail=False (profiles whose sticker wraps flush onto the bottom
    frame without hanging below the phone), pixels past limit receive no paper
    protection so gray stand lips below the sticker are cut cleanly."""
    if paper is None or not paper.any():return paper
    h=paper.shape[0];yy=np.arange(h)[:,None]
    if not allow_tail:
        return paper & (yy <= np.broadcast_to(limit, paper.shape) + .5)
    beyond=paper&(yy>np.broadcast_to(limit,paper.shape)+1)
    labels,n=ndimage.label(beyond)
    stand=np.isin(labels,np.unique(labels[-1][labels[-1]>0]))
    return paper&~stand


def _plate_geometry_strict(matte, face, paper=None, stage_sink=None,source_image=None,reconstruct_paper_tail=True,_front_closure_evidence=False):
    """Remove the attached stand locally, preserving the segmented round corners."""
    a=np.array(matte.convert('RGBA'));mask=principal(a[...,3]);h,w=mask.shape
    widths=mask.sum(axis=1);wide=widths>widths.max()*.8
    labels,n=ndimage.label(wide);sizes=np.bincount(labels);sizes[0]=0
    rows=np.flatnonzero(labels==sizes.argmax())
    if len(rows)<.4*h:raise ValueError('No tall, wide phone plate in the photograph')
    center=mask[rows[len(rows)//2]];xs=np.flatnonzero(center);x0,x1=int(xs[0]),int(xs[-1])+1;bw=x1-x0
    if source_image is None:raise ValueError('Independent raw photograph required for automatic plate geometry')
    raw_rgb=np.array(source_image.convert('RGB'))
    body_slope,axis_proof=plate_axis(a,raw_rgb,rows,exclude_front_controls=face=='front')
    # Clear strips between the round corners and the stand are an independent
    # body measurement. A central RGB maximum can be the stand's white joint.
    clear_options=[];raw_gradient=np.linalg.norm(np.gradient(ndimage.gaussian_filter1d(raw_rgb.astype(np.float32),.8,axis=0),axis=0),axis=2)
    corner_proof=None
    try:
        corner_line,corner_proof=plate_plane_from_own_corners(a,raw_rgb,rows)
        if abs(corner_line[0]-body_slope)*bw<=3:
            clear_options.append((-1.,corner_line,corner_line,corner_proof['samples'],None))
    except ValueError:pass
    for lo,hi in ((.16,.23),(.18,.25),(.20,.27),(.22,.29),(.24,.31)):
        groups=[np.arange(int(x0+lo*bw),int(x0+hi*bw)),np.arange(int(x1-hi*bw),int(x1-lo*bw))]
        cx=np.concatenate(groups);cb=h-np.argmax(mask[::-1,cx],axis=0)
        clear=cb<h-8
        split=len(groups[0])
        if min(clear[:split].sum(),clear[split:].sum())<12:continue
        try:cline,ck,ce=fixed_slope_consensus(cx[clear]+.5,cb[clear].astype(float),body_slope)
        except ValueError:continue
        positions=np.polyval(cline,cx[clear]+.5)-.5
        search_y=np.rint(positions[:,None]+np.arange(-4,5)).astype(int)
        colors=raw_gradient[np.clip(search_y,0,h-1),cx[clear,None]]
        rgb_y=search_y[np.arange(len(positions)),np.argmax(colors,axis=1)]+.5
        try:rline,rk,re=fixed_slope_consensus(cx[clear]+.5,rgb_y,body_slope)
        except ValueError:continue
        error=np.max(np.abs(np.polyval(rline,[x0,x1])-np.polyval(cline,[x0,x1])))
        if error>3:continue
        clear_options.append((ce+re,cline,rline,int(clear.sum()),[lo,hi]))
    lower_plane_proof=None
    if _front_closure_evidence and face=='front':
        try:
            closure = front_lower_closure(mask,raw_rgb,x0,x1,rows[-1],a[...,3])
        except ValueError:
            closure = None
        if closure is not None:
            old = min(clear_options,key=lambda item:item[0]) if clear_options else None
            center = (x0+x1)/2
            if old is None or np.polyval(old[2],center)-np.polyval(closure[2],center)>3:
                score,al,rl,n,bands,lower_plane_proof=closure
                clear_options=[(score,al,rl,n,bands)]
    if not clear_options:
        try:
            if _front_closure_evidence:
                score,al,rl,n,bands,lower_plane_proof=front_lower_closure(mask,raw_rgb,x0,x1,rows[-1],a[...,3])
            else:score,al,rl,n,bands,lower_plane_proof=exposed_lower_plane(mask,raw_gradient,x0,x1,rows[-1],paper,body_slope=body_slope)
            clear_options.append((score,al,rl,n,bands))
        except ValueError:pass
    if clear_options:
        score,alpha_line,rgb_line,n_clear,bands=min(clear_options,key=lambda v:v[0])
        # The RGB transition gives a subpixel contour centre, with original
        # alpha retained at both exposed, already segmented round corners.
        line=rgb_line;residual=corner_proof['corner_residual_p95_px'] if bands is None else score/2;inlier_fraction=1.
        yy=np.arange(h)[:,None];xx=np.arange(w)[None,:]+.5;limit=np.polyval(line,xx)
        base_limit=limit.copy()
        paper_tail_proof=None
        if reconstruct_paper_tail and face=='back' and paper is not None and paper.any():
            from .auto_material import source_paper_tail
            paper,tail_info=source_paper_tail(np.dstack([raw_rgb,a[...,3]]),paper,{},physical_end=limit[0])
            paper_tail_proof=tail_info.get('source_paper_tail')
        paper_cut = paper_without_stand(paper, limit, allow_tail=reconstruct_paper_tail)
        remote=(yy>limit+12)&(a[...,3]>0)
        far_start=min(h-1,max(0,int(np.min(limit))+12))
        support_columns=ndimage.binary_dilation((a[far_start:,:,3]>0).any(axis=0),iterations=8)[None,:]
        contact_curve_proof=None
        if bands is None:
            # With a tilted body, a single global far-start row can still
            # intersect the phone at the opposite end. This new corner-only
            # projection must be genuinely beyond each column's body plane.
            corner_support=ndimage.binary_dilation(remote.any(axis=0),iterations=8)
            try:
                local_limit,contact_curve_proof=plate_corner_contact_limit(a,raw_rgb,line,corner_proof,corner_support,paper_cut)
                limit=local_limit[None,:]
            except ValueError:
                if face == 'back':
                    pass
                elif corner_proof and corner_proof.get('method') == 'own_top_bottom_corner_curve_consensus_taper_v1':
                    # On mildly tapered front captures, fall back cleanly to an
                    # exposed clear-band/closure line rather than failing a
                    # capture whose flat lower rail is already directly observed.
                    others = [v for v in clear_options if v[4] is not None]
                    if others:
                        score,alpha_line,rgb_line,n_clear,bands=min(others,key=lambda v:v[0])
                        line=rgb_line;residual=score/2
                        limit=np.polyval(line,xx);base_limit=limit.copy()
                        remote=(yy>limit+12)&(a[...,3]>0)
                        far_start=min(h-1,max(0,int(np.min(limit))+12))
                        support_columns=ndimage.binary_dilation((a[far_start:,:,3]>0).any(axis=0),iterations=8)[None,:]
                else:
                    raise
        if face == 'back':
            wedge_support = ndimage.binary_dilation(remote.any(axis=0), iterations=24)
            wedge_limit, wedge_proof = plate_corner_wedge_limit(a, raw_rgb, rows, line, wedge_support, paper_cut)
            if wedge_proof and wedge_proof.get('changed_columns'):
                limit = np.minimum(limit, wedge_limit[None, :])
                support_columns = support_columns | (limit < base_limit)
                if contact_curve_proof is None or not contact_curve_proof.get('changed_columns'):
                    contact_curve_proof = wedge_proof
                else:
                    contact_curve_proof['wedge_refinement'] = wedge_proof
        if lower_plane_proof and lower_plane_proof.get('method')=='front_alpha_plateau_with_independent_dark_glass_closure':
            a[...,3]=front_support_coverage(a[...,3],limit,support_columns)
            lower_plane_proof['source_cut']='fractional_coverage_at_exclusive_alpha_rail_no_extra_stand_pixel'
        else:
            paper=paper_without_stand(paper,limit,allow_tail=reconstruct_paper_tail)
            outside=(yy>limit+1)&support_columns
            if paper is not None:outside &= ~paper
            a[...,3][outside]=0
            contact=(yy>=limit-2)&(yy<=limit+1)&mask&support_columns
            if paper is not None:contact &= ~paper
            coverage=np.clip((limit+1-yy)/3.,0,1)
            a[...,3][contact]=np.minimum(a[...,3][contact],np.rint(255*coverage[contact]).astype(np.uint8))
        if stage_sink:stage_sink('source_matte',matte);stage_sink('after_support',Image.fromarray(a))
        angle=axis_proof['rotation_ccw_degrees'];rotated=Image.fromarray(rotate_alpha_safe(a,angle))
        if stage_sink:stage_sink('after_rotation',rotated);stage_sink('after_crop',rotated)
        box=rotated.getchannel('A').getbbox();result=rotated.crop(box)
        return result,dict(method='source_exposed_plate_rgb_alpha_consensus_v1',source_size=list(matte.size),
            output_size=list(result.size),body_x=[x0,x1],bottom_rail=line.tolist(),alpha_lower_rail=alpha_line.tolist(),
            clear_bands=bands,exposed_samples=n_clear,lower_edge_residual_p95_px=residual,
            own_corner_evidence=corner_proof if bands is None else None,axis_evidence=axis_proof,
            corner_contact_evidence=contact_curve_proof,
            source_paper_tail=paper_tail_proof,
            lower_plane_evidence=lower_plane_proof,
            rotation_ccw_degrees=angle,canvas_trim_xyxy=list(box),rgb_painted=False,
            review_reasons=['heuristic_support_cut','complete_outline_requires_review']+(['source_contact_appearance_unverified'] if lower_plane_proof and lower_plane_proof.get('method')=='front_alpha_plateau_with_independent_dark_glass_closure' else []))
    x=np.arange(int(x0+.1*bw),int(x1-.1*bw));bottom=h-np.argmax(mask[::-1,x],axis=0)
    exposed=(bottom<h-8)&(bottom>rows[-1]-bw*.12)
    if exposed.sum()<.08*bw:raise ValueError('Insufficient exposed plate edge')
    hist=ndimage.gaussian_filter1d(np.bincount(bottom[exposed]),2.)
    seed=int(hist.argmax())
    # Locate the photographed lower edge independently across both halves,
    # including the contact area, rather than extending a rectangle crop.
    gx=np.arange(int(x0+.20*bw),int(x1-.20*bw))
    color=ndimage.gaussian_filter1d(raw_rgb.astype(np.float32),.8,axis=0)
    gradient=np.linalg.norm(np.gradient(color,axis=0),axis=2)
    search=np.arange(max(0,seed-8),min(h,seed+max(12,int(bw*.02))+1))
    strength=gradient[search[:,None],gx[None,:]]
    local_peak=(strength>=np.roll(strength,1,axis=0))&(strength>=np.roll(strength,-1,axis=0))
    local_peak[0]=False;local_peak[-1]=False
    local_peak &= strength>=np.max(strength,axis=0)[None,:]*.20
    # The inner display/rim transition can be stronger than the outer body
    # boundary. Select the last supported source peak, not the brightest one.
    gy=np.max(np.where(local_peak,search[:,None],-1),axis=0)
    absent=gy<0;gy[absent]=search[np.argmax(strength[:,absent],axis=0)]
    valid=np.max(strength,axis=0)>3
    if paper is not None:valid &= ~paper[np.clip(gy,0,h-1),gx]
    if valid.sum()<.2*bw:raise ValueError('Weak source RGB lower boundary')
    options=[]
    for proposal in (gy,search[np.argmax(strength,axis=0)]):
        try:fitted,kept,error=fixed_slope_consensus(gx[valid]+.5,proposal[valid]+.5,body_slope)
        except ValueError:continue
        if error>3 and kept.mean()>=.6:
            fitx=gx[valid][kept]+.5;fity=proposal[valid][kept]+.5
            curve=np.polyfit(fitx,fity,2)
            curve_error=float(np.percentile(np.abs(np.polyval(curve,fitx)-fity),95))
            displacement=np.polyval(curve,np.array([x0,(x0+x1)/2,x1]))-np.polyval(fitted,np.array([x0,(x0+x1)/2,x1]))
            if curve_error<=3 and np.max(np.abs(displacement))<=6:
                fitted,error=curve,curve_error
        slope=float(np.polyval(np.polyder(fitted),(x0+x1)/2))
        if error<=3 and kept.mean()>=.6 and abs(slope)<=np.tan(np.deg2rad(4)):
            # Keep the source-derived plate edge, not a stronger inner glass
            # transition. Alpha at exposed phone columns supplies a separate
            # outward-conservation constraint.
            delta=bottom[exposed]-np.polyval(fitted,x[exposed]+.5)
            near=delta[np.abs(delta)<12]
            if len(near)>=max(20,exposed.sum()*.6) and abs(float(np.median(near)))<=5 and np.percentile(near,75)<=4:
                options.append((float(np.polyval(fitted,(x0+x1)/2)),fitted,error,float(kept.mean())))
    if not options:
        if face == 'back' or _front_closure_evidence:
            from .joined_stand import cut_joined_stand
            try:
                return cut_joined_stand(matte, mask, raw_rgb, paper, x0, x1, rows[-1], axis_proof, stage_sink)
            except ValueError:
                pass
        raise ValueError('Plate lower boundary is uncertain')
    _,line,residual,inlier_fraction=max(options,key=lambda v:v[0])
    yy=np.arange(h)[:,None];xx=np.arange(w)[None,:]+.5;limit=np.polyval(line,xx)+1.
    paper_tail_proof=None
    if reconstruct_paper_tail and face=='back' and paper is not None and paper.any():
        from .auto_material import source_paper_tail
        paper,tail_info=source_paper_tail(np.dstack([raw_rgb,a[...,3]]),paper,{},physical_end=np.polyval(line,xx)[0])
        paper_tail_proof=tail_info.get('source_paper_tail')
    far=mask[int(np.polyval(line,(x0+x1)/2)+12):]
    support_columns=ndimage.binary_dilation(far.any(axis=0),iterations=8)
    paper=paper_without_stand(paper,limit,allow_tail=reconstruct_paper_tail)
    corner_wedge_proof=None
    if face == 'back':
        remote=(yy>np.polyval(line,xx)+12)&(a[...,3]>0)
        wedge_support=ndimage.binary_dilation(remote.any(axis=0),iterations=24)
        wedge_limit,corner_wedge_proof=plate_corner_wedge_limit(a,raw_rgb,rows,line,wedge_support,paper)
        if corner_wedge_proof and corner_wedge_proof.get('changed_columns'):
            limit=np.minimum(limit,wedge_limit[None,:]+1.)
            support_columns=support_columns|(wedge_limit<np.polyval(line,xx[0]))
    support=ndimage.binary_fill_holes(np.broadcast_to(support_columns,(h,w))) & (yy>=limit)
    if paper is not None:support &= ~paper
    a[...,3][support]=0
    if corner_wedge_proof and corner_wedge_proof.get('changed_columns'):
        wedge_cols=np.zeros(w,bool);wedge_cols[corner_wedge_proof['changed_columns']]=True
        contact=(yy>=limit-3)&(yy<limit)&mask&wedge_cols[None,:]
        if paper is not None:contact &= ~paper
        coverage=np.clip((limit-yy)/3.,0,1)
        a[...,3][contact]=np.minimum(a[...,3][contact],np.rint(255*coverage[contact]).astype(np.uint8))
    if stage_sink:stage_sink('source_matte',matte);stage_sink('after_support',Image.fromarray(a))
    slope=float(np.polyval(np.polyder(line),(x0+x1)/2))
    angle=axis_proof['rotation_ccw_degrees'];rotated=Image.fromarray(rotate_alpha_safe(a,angle))
    if stage_sink:stage_sink('after_rotation',rotated);stage_sink('after_crop',rotated)
    box=rotated.getchannel('A').getbbox()
    if not box:raise ValueError('Empty plate')
    result=rotated.crop(box)
    return result,dict(method='local_stand_removal_from_exposed_plate_edges_v1',
        source_size=list(matte.size),output_size=list(result.size),body_x=[x0,x1],
        bottom_rail=line.tolist(),lower_edge_residual_p95_px=residual,exposed_samples=int(valid.sum()),
        source_paper_tail=paper_tail_proof,corner_contact_evidence=corner_wedge_proof,
        rotation_ccw_degrees=angle,canvas_trim_xyxy=list(box),removed_support_pixels=int(support.sum()),
        rgb_painted=False,axis_evidence=axis_proof,review_reasons=['heuristic_support_cut','complete_outline_requires_review'])


def check_front_support_residue(image):
    """Reject a bright attached tail; this check never invents a crop line."""
    a=np.asarray(image.convert('RGBA'));mask=principal(a[...,3])
    widths=mask.sum(axis=1);wide=np.flatnonzero(widths>widths.max()*.8)
    last=int(np.flatnonzero(mask.any(axis=1))[-1]);end=int(wide[-1])
    depth=last-end
    if depth<=max(12,.04*widths.max()):
        return dict(narrow_tail_rows=depth,bright_support_residue=False)
    gray=a[...,:3].mean(axis=2)
    tail=mask.copy();tail[:end+1]=False
    inside=mask[max(0,end-30):max(1,end-15)]
    body=gray[max(0,end-30):max(1,end-15)][inside]
    contrast=float(np.median(gray[tail])-np.median(body))
    if contrast>40:
        raise ValueError('Front retains a bright narrow stand projection below body')
    return dict(narrow_tail_rows=depth,tail_contrast=contrast,bright_support_residue=False)


def plate_geometry(matte,face,paper=None,stage_sink=None,source_image=None,reconstruct_paper_tail=True,*,prefer_front_closure=False):
    """Keep accepted geometry unchanged; recover a different exposed source gap."""
    try:
        image,proof = _plate_geometry_strict(matte,face,paper,stage_sink,source_image,reconstruct_paper_tail,
                                            _front_closure_evidence=prefer_front_closure)
        if face=='front':
            proof['front_support_residue_check']=check_front_support_residue(image)
        return image,proof
    except ValueError as original:
        if str(original)!='Plate lower boundary is uncertain' or face!='front':raise
        try:
            image,proof=_plate_geometry_strict(matte,face,paper,stage_sink,source_image,reconstruct_paper_tail,
                _front_closure_evidence=True)
        except ValueError:
            raise original
        if not proof.get('lower_plane_evidence'):
            raise original
        proof['lower_plane_evidence']['recovery']='independent_exposed_gap_between_attached_support_contacts'
        proof['lower_plane_evidence']['original_failure']=str(original)
        proof['front_support_residue_check']=check_front_support_residue(image)
        return image,proof
