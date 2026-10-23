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


def plate_axis(rgba,raw_rgb,rows):
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
    if abs(l[0]-r[0])*len(ys)>4:raise ValueError('Plate sides are not a parallel studio capture')
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
        symmetric_body_assumption=True,samples=int(keep.sum()))


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


def frame_geometry(matte, face, stage_sink=None,source_image=None):
    a=np.array(matte.convert('RGBA')); original=a.copy()
    raw=np.array(source_image.convert('RGB')) if source_image else None
    if face == 'top':
        from .top_frame import calibrate_top, verify_top_mask
        top,bottom,proof=calibrate_top(a,raw)
    else:
        top,bottom,proof=frame_rails(a,raw)
    x=np.arange(a.shape[1])[None,:]+.5; y=np.arange(a.shape[0])[:,None]
    # Preserve native curved endcaps with their segmentation alpha. Rail
    # masking removes only projections/support, with an outward allowance.
    upper=np.ceil(np.polyval(top,x)) if face == 'top' else np.polyval(top,x)-1.
    lower=np.polyval(bottom,x)+1.
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


def exposed_lower_plane(mask, raw_gradient, x0, x1, last_body_row, paper=None):
    """Fit the photographed cross-edge independently of the long-side axis.

    Mild perspective makes those axes non-perpendicular. Require both exposed
    halves and independent RGB support; never fit the central stand or paper.
    """
    h,w=mask.shape;bw=x1-x0;options=[]
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


def plate_geometry(matte, face, paper=None, stage_sink=None,source_image=None):
    """Remove the attached stand locally, preserving the segmented round corners."""
    a=np.array(matte.convert('RGBA'));mask=principal(a[...,3]);h,w=mask.shape
    widths=mask.sum(axis=1);wide=widths>widths.max()*.8
    labels,n=ndimage.label(wide);sizes=np.bincount(labels);sizes[0]=0
    rows=np.flatnonzero(labels==sizes.argmax())
    if len(rows)<.4*h:raise ValueError('No tall, wide phone plate in the photograph')
    center=mask[rows[len(rows)//2]];xs=np.flatnonzero(center);x0,x1=int(xs[0]),int(xs[-1])+1;bw=x1-x0
    if source_image is None:raise ValueError('Independent raw photograph required for automatic plate geometry')
    raw_rgb=np.array(source_image.convert('RGB'))
    body_slope,axis_proof=plate_axis(a,raw_rgb,rows)
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
    if not clear_options:
        try:
            score,al,rl,n,bands,lower_plane_proof=exposed_lower_plane(mask,raw_gradient,x0,x1,rows[-1],paper)
            clear_options.append((score,al,rl,n,bands))
        except ValueError:pass
    if clear_options:
        score,alpha_line,rgb_line,n_clear,bands=min(clear_options,key=lambda v:v[0])
        # The RGB transition gives a subpixel contour centre, with original
        # alpha retained at both exposed, already segmented round corners.
        line=rgb_line;residual=corner_proof['corner_residual_p95_px'] if bands is None else score/2;inlier_fraction=1.
        yy=np.arange(h)[:,None];xx=np.arange(w)[None,:]+.5;limit=np.polyval(line,xx)
        far_start=min(h-1,max(0,int(np.min(limit))+12))
        support_columns=ndimage.binary_dilation((a[far_start:,:,3]>0).any(axis=0),iterations=8)[None,:]
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
            lower_plane_evidence=lower_plane_proof,
            rotation_ccw_degrees=angle,canvas_trim_xyxy=list(box),rgb_painted=False,
            review_reasons=['heuristic_support_cut','complete_outline_requires_review'])
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
        raise ValueError('Plate lower boundary is uncertain')
    _,line,residual,inlier_fraction=max(options,key=lambda v:v[0])
    yy=np.arange(h)[:,None];xx=np.arange(w)[None,:]+.5;limit=np.polyval(line,xx)+1.
    far=mask[int(np.polyval(line,(x0+x1)/2)+12):]
    support_columns=ndimage.binary_dilation(far.any(axis=0),iterations=8)
    support=ndimage.binary_fill_holes(np.broadcast_to(support_columns,(h,w))) & (yy>=limit)
    if paper is not None:support &= ~paper
    a[...,3][support]=0
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
        rotation_ccw_degrees=angle,canvas_trim_xyxy=list(box),removed_support_pixels=int(support.sum()),
        rgb_painted=False,axis_evidence=axis_proof,review_reasons=['heuristic_support_cut','complete_outline_requires_review'])
