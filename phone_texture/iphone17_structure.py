"""Align side bands to independently measured antenna breaks on the rear rim."""
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d, median_filter
from .auto_structure import antenna_rows,gray_signal
from .geometry_review import digest
from texture_alignment.align_textures import warp_rows


def per_column_boundaries(image,rows):
    """Signed per-column antenna edges, as reviewed on the standard sample.

    The Pro route moved to antenna_material.trace_boundaries (end-cap hold
    and band revisions); the glass-back route keeps this tracer so its
    reviewed source mapping stays unchanged.
    """
    a=np.array(image.convert('RGBA'));h,w=a.shape[:2]
    colors=gaussian_filter1d(a[...,:3].astype(float).mean(axis=2),1.,axis=0)
    gradient=np.gradient(colors,axis=0)
    center_gradient=np.gradient(gaussian_filter1d(gray_signal(a,[int(w*.4),int(w*.6)]),1.))
    lines=[]
    band_widths=[rows[1]-rows[0]]*2+[rows[3]-rows[2]]*2
    for i,(row,width) in enumerate(zip(rows,band_widths)):
        radius=max(7,int(np.ceil(width*.6)))
        lo=max(1,row-radius);hi=min(h-1,row+radius+1)
        if i:lo=max(lo,int(np.ceil((rows[i-1]+row)/2)))
        if i<len(rows)-1:hi=min(hi,int(np.floor((row+rows[i+1])/2))+1)
        # Follow the same signed material transition across the strip.
        sign=np.sign(center_gradient[row])
        if sign==0:raise ValueError('Antenna material polarity is ambiguous')
        response=gradient*sign
        peaks=lo+np.argmax(response[lo:hi],axis=0)
        reliable=(a[peaks,np.arange(w),3]>=128)&(response[peaks,np.arange(w)]>.5)
        if reliable.sum()<w*.6:raise ValueError('Too few supported antenna columns')
        x=np.arange(w);line=np.interp(x,x[reliable],peaks[reliable].astype(float))
        line=gaussian_filter1d(median_filter(line,size=5),1.)
        lines.append(line)
    result=np.array(lines)
    if np.any(np.diff(result,axis=0)<=0):
        from .antenna_material import trace_boundaries
        result, _ = trace_boundaries(image, rows)
    return result


def gradient_peak(gradient, target, sign, radius=4):
    """Subpixel extremum; the target only bounds the independent RGB search."""
    start=max(1,int(target)-radius);end=min(len(gradient)-1,int(target)+radius+1)
    response=gradient*sign;k=start+int(np.argmax(response[start:end]))
    if k in (start,end-1) or response[k]<.5:
        raise ValueError('Antenna boundary is ambiguous in the independent RGB search')
    a,b,c=response[k-1:k+2];den=a-2*b+c
    return float(k+(.5*(a-c)/den if abs(den)>1e-9 else 0.))


def _rear_antennas(image, *, reflective_rim=False, refine_bands=('upper','lower'), sides=('left','right'),
                   bands=('upper','lower')):
    """Measure bright plastic breaks in the narrow dark rear metal rim.

    The glass has no antenna; the outer rim does. Follow the current alpha
    contour and require agreement across three independent inward tracks.
    Rear-view image left is the phone's right (power/camera-control) side.
    """
    a=np.asarray(image.convert('RGBA'));h,w=a.shape[:2]
    gray=a[...,:3].mean(2);scale=w/988
    result={}
    for edge_name,side in [('left','right'),('right','left')]:
        if side not in sides:
            continue
        edge=np.argmax(a[...,3]>200,axis=1) if edge_name=='left' else w-1-np.argmax(a[:,::-1,3]>200,axis=1)
        edge=ndimage.gaussian_filter1d(ndimage.median_filter(edge.astype(float),max(3,int(h*.01)//2*2+1)),max(1,h*.0025))
        if reflective_rim:
            tracks = []
            orientation = 1 if edge_name=='left' else -1
            offsets = np.arange(2,round(16*scale)+1)
            x = edge[:,None]+orientation*offsets
            yy=np.broadcast_to(np.arange(h)[:,None],x.shape)
            stack = ndimage.map_coordinates(gray,[yy,x],order=1,mode='nearest')
            smooth = ndimage.gaussian_filter1d(stack,.7,axis=1)
            peak = np.argmax(smooth,axis=1)
            line = ndimage.median_filter(peak.astype(float),size=(round(h*.022)//2*2+1))
            line = ndimage.gaussian_filter1d(line,2.)
            for offset in (-1.,0.,1.):
                xs = edge+orientation*(line+offsets[0]+offset)
                sig=ndimage.map_coordinates(gray,[np.arange(h),xs],order=1,mode='nearest')
                tracks.append(-ndimage.gaussian_filter1d(sig,max(.7,h/2058)))
        else:
            tracks=[]
            for offset in (1.,2.,3.):
                xs=edge+(offset*scale if edge_name=='left' else -offset*scale)
                sig=ndimage.map_coordinates(gray,[np.arange(h),xs],order=1,mode='nearest')
                tracks.append(ndimage.gaussian_filter1d(sig,max(.7,h/2058)))
        consensus=np.median(tracks,axis=0)
        rows=[];proof=[]
        for band,(lo,hi) in zip(('upper','lower'),[(int(h*.065),int(h*.16)),(int(h*.84),int(h*.94))]):
            if band not in bands:
                continue
            def pairs(sig,start,end):
                grad=np.gradient(sig);found=[]
                pad=max(2,round(h*.001));span=max(pad+2,round(h*.004))
                for y in range(start,end):
                    for width in range(max(5,round(h*.0044)),round(h*.0115)+1):
                        z=y+width
                        if z+span>=h:continue
                        contrast=float(np.mean(sig[y+pad:z-pad+1])-(np.mean(sig[y-span:y-pad])+np.mean(sig[z+pad:z+span]))/2)
                        strength=min(grad[y],-grad[z])
                        if strength>1.5 and contrast>6:
                            found.append((float(strength*2+contrast*.3),y,z,contrast))
                if not found:raise ValueError('Rear rim antenna lacks two source-supported boundaries')
                return max(found)
            best=pairs(consensus,lo,hi);_,y,z,contrast=best
            # A competing, spatially separate feature must not be comparable.
            alternatives=[]
            for start,end in [(lo,max(lo,y-30)),(min(hi,z+30),hi)]:
                if end>start:
                    try:alternatives.append(pairs(consensus,start,end)[0])
                    except ValueError:pass
            if alternatives and max(alternatives)>.8*best[0]:
                raise ValueError('Rear rim antenna has an ambiguous competing feature')
            measured=[];subpixel=[]
            for sig in tracks:
                grad=np.gradient(sig);local=[]
                for target,sign in [(y,1),(z,-1)]:
                    radius=max(3,round(h*.002));start=target-radius;end=target+radius+1
                    row=start+int(np.argmax(grad[start:end]*sign))
                    if row in (start,end-1) or grad[row]*sign<1.5:
                        raise ValueError('Rear antenna boundary not supported across independent rim tracks')
                    local.append(row)
                measured.append(local)
                subpixel.append([gradient_peak(grad,y,1),gradient_peak(grad,z,-1)])
            if np.max(np.ptp(measured,axis=0))>max(3,h*.0015):
                raise ValueError('Rear antenna rim tracks disagree')
            # The pooled integer peak can fall a row below every individual
            # track. Use their robust subpixel consensus, not the pooled peak.
            refined=np.median(subpixel,axis=0).tolist() if band in refine_bands else [y,z]
            rows.extend(refined);proof.append(dict(rows=refined,coarse_rows=[y,z],contrast=contrast,
                track_rows=measured,subpixel_track_rows=subpixel,band=band))
        result[side]=dict(rows=rows,rear_image_edge=edge_name,checks=proof)
    return result



def _rim_band_witnesses(image, side, band):
    """Independently locate a finite stripe on adjacent tracks inside the same rear rim."""
    a = np.asarray(image.convert('RGBA'))
    if side == 'left':
        a = a[:, ::-1]
    h, w = a.shape[:2]
    edge = np.argmax(a[..., 3] > 200, axis=1)
    offsets = np.arange(1, 1+max(5, round(6*w/988)))
    cols = np.clip(edge[:, None]+offsets, 0, w-1)
    yy = np.arange(h)[:, None]
    samples = gaussian_filter1d(a[yy, cols, :3].mean(axis=2), max(.7, h/2058), axis=0)
    opacity = a[yy, cols, 3]
    lo, hi = (.065,.16) if band == 'upper' else (.84,.94)
    start = np.arange(round(h*lo), round(h*hi))
    margin = max(3, round(h*.002))
    candidates = []
    for col, signal in enumerate(samples.T):
        gradient = np.gradient(signal)
        integral = np.r_[0., np.cumsum(signal)]
        found = []
        for width in range(max(5,round(h*.0044)), round(h*.0115)+1):
            end = start+width
            mid = (integral[end-2]-integral[start+2])/(width-4)
            before = (integral[start-1]-integral[start-margin-1])/margin
            after = (integral[end+margin+2]-integral[end+2])/margin
            for sign in (-1,1):
                edge_strength = np.minimum(sign*gradient[start], -sign*gradient[end])
                contrast = np.minimum(sign*(mid-before), sign*(mid-after))
                score = np.where((edge_strength>1.5)&(contrast>6), 2*edge_strength+.3*contrast, 0)
                for k in np.flatnonzero((score>0)&(score==ndimage.maximum_filter1d(score,5))):
                    y,z = start[k],end[k]
                    if np.min(opacity[y-margin:z+margin+1,col]) < 200:
                        continue
                    found.append((score[k], y, z, sign))
        if not found:
            continue
        best = max(found)
        if any(v[0] >= .8*best[0] and abs(v[1]-best[1]) > margin for v in found):
            continue
        _,y,z,sign = best
        try:
            pair = [gradient_peak(gradient,y,sign),gradient_peak(gradient,z,-sign)]
        except ValueError:
            continue
        candidates.append((col,pair,sign))
    if len(candidates)<4:
        raise ValueError('Rear rim lacks four independent opaque stripe tracks')
    positions = np.array([v[1] for v in candidates])
    if np.max(np.ptp(positions,axis=0))>2 or len({v[2] for v in candidates})!=1:
        raise ValueError('Independent rear rim stripes disagree')
    target = np.median(positions, axis=0).tolist()
    return target, dict(band=band, rows=target, subpixel_track_rows=positions.tolist(),
                        track_offsets_px=[int(offsets[v[0]]) for v in candidates],
                        method='independent_signed_plateau_on_opaque_rear_rim')


def _rim_stripe_candidates(image, side, band, *, follow_ridge=False):
    """Find a finite stripe crossing adjacent metal tracks, with glass as a negative control."""
    rgba = np.asarray(image.convert('RGBA'))
    if side == 'left':
        rgba = rgba[:, ::-1]
    height, width = rgba.shape[:2]
    scale = width / 988
    # Sample at fixed subpixel distances from a smoothed contour. Integer
    # jumps in a tight alpha crop otherwise look like horizontal RGB edges.
    edge = np.argmax(rgba[..., 3] > 200, axis=1).astype(float)
    edge = gaussian_filter1d(median_filter(edge, 7), 2.)
    depths = np.arange(1, max(8, round(11 * scale)) + 1)
    glass_depth = max(depths[-1] + 4, round(26 * scale))
    gray = rgba[..., :3].mean(axis=2)
    if follow_ridge:
        # Track the local highlight between two darker shoulders. The glass
        # can be brighter than this ridge and must not attract the track.
        offsets = np.arange(1, max(12, round(17*scale))+1)
        xx = edge[:, None] + offsets
        yy = np.broadcast_to(np.arange(height)[:, None], xx.shape)
        profile = ndimage.map_coordinates(gray, [yy, xx], order=1, mode='nearest')
        prominence = profile[:, 2:-2] - np.maximum(profile[:, :-4], profile[:, 4:])
        peak = np.argmax(prominence, axis=1)+2
        rows = np.arange(height)
        left, center, right = (profile[rows, peak+k] for k in (-1, 0, 1))
        delta = np.clip((right-left)/np.maximum(1., 2*(2*center-left-right)), -.5, .5)
        ridge = gaussian_filter1d(median_filter(offsets[peak]+delta, 7), 1.)
        depths = np.arange(-3, 4)
        track_x = edge[:, None] + ridge[:, None] + depths
    else:
        track_x = edge[:, None] + depths
    x = np.column_stack((track_x, edge+glass_depth))
    y = np.broadcast_to(np.arange(height)[:, None], x.shape)
    samples = ndimage.map_coordinates(gray, [y, x], order=1, mode='nearest')
    opacity = ndimage.map_coordinates(rgba[..., 3], [y, x], order=1, mode='nearest')
    smooth = gaussian_filter1d(samples, max(.85, height / 2058), axis=0)
    gradient = np.gradient(smooth, axis=0)
    integral = np.vstack([np.zeros(samples.shape[1]), smooth.cumsum(axis=0)])
    lo, hi = (.065, .16) if band == 'upper' else (.84, .94)
    starts = np.arange(round(lo * height), round(hi * height))
    flank = max(4, round(.003 * height))
    candidates = []
    for span in range(max(6, round(.0044 * height)), round(.0115 * height) + 1):
        ends = starts + span
        inside = (integral[ends-2] - integral[starts+2]) / (span-4)
        before = (integral[starts-2] - integral[starts-2-flank]) / flank
        after = (integral[ends+3+flank] - integral[ends+3]) / flank
        sign = np.sign(inside - (before+after)/2)
        contrast = np.minimum((inside-before)*sign, (inside-after)*sign)
        strength = np.minimum(gradient[starts]*sign, -gradient[ends]*sign)
        supported = (contrast > 6) & (strength > 1.5) & (abs(after-before) < contrast)
        # A common abrupt transition reaching the inner glass can be a
        # scanline or a reflection extending across the photographed back.
        for idx in np.flatnonzero((supported[:, :-1].sum(axis=1) >= 3) & ~supported[:, -1]):
            row, end = starts[idx], ends[idx]
            active = supported[idx, :-1] & (opacity[row-flank:end+flank+1, :-1].min(axis=0)>200)
            for polarity in (-1, 1):
                track_ids = np.flatnonzero(active & (sign[idx, :-1] == polarity))
                for group in np.split(track_ids, np.flatnonzero(np.diff(track_ids)>1)+1):
                    if len(group) < 3:
                        continue
                    outer_stripe = not follow_ridge and len(group) >= 4 and depths[group[0]] <= round(6*scale)
                    # A narrow polished ridge may span only three tracks.
                    # Locate its peak independently above and below the dark
                    # interruption, with a darker metal shoulder on each side.
                    on_ridge = False
                    if polarity < 0:
                        for col in group[1:-1]:
                            if col < 2 or col+2 >= len(depths):
                                continue
                            if all(profile[col] - max(profile[col-2], profile[col+2]) > 15
                                   for profile in (before[idx], after[idx])):
                                on_ridge = True
                                break
                    if not (outer_stripe or on_ridge):
                        continue
                    try:
                        pair = np.array([[gradient_peak(gradient[:, col], row, polarity, radius=3),
                                          gradient_peak(gradient[:, col], end, -polarity, radius=3)] for col in group])
                    except ValueError:
                        continue
                    if np.ptp(pair,axis=0).max() > 2:
                        continue
                    strength_score = np.median(contrast[idx, group]+2*strength[idx, group])
                    candidates.append((float(strength_score * np.sqrt(len(group))), pair, group))
    if not candidates:
        raise ValueError('No repeatable stripe crosses adjacent metal tracks')
    candidates.sort(key=lambda item: item[0], reverse=True)
    # Many nearby integer search windows converge on the same subpixel peak.
    options = []
    for strength, positions, group in candidates:
        target = np.median(positions, axis=0)
        if any(np.max(abs(target-other)) < 1e-6 for _,other,_ in options):
            continue
        proof = dict(band=band, rows=target.tolist(), subpixel_track_rows=positions.tolist(),
                     track_offsets_px=depths[group].tolist(), glass_control_depth_px=glass_depth,
                     method='stripe_crossing_tracked_ridge' if follow_ridge else 'continuous_stripe_on_adjacent_metal_tracks')
        options.append((strength, target, proof))
    return options


def _coherent_rim_stripe(image, side, band, *, follow_ridge=False):
    options = _rim_stripe_candidates(image, side, band, follow_ridge=follow_ridge)
    score, target, proof = options[0]
    if any(value >= .8*score and np.max(abs(pair-target)) > 3 for value,pair,_ in options[1:]):
        raise ValueError('Competing rim stripes have comparable support')
    return target.tolist(), proof


def _joint_rim_stripes(image, side):
    """Disambiguate measured stripes by the paired layout on this one rail."""
    bands = []
    for band in ('upper','lower'):
        proposals = []
        for follow in (False,True):
            try:
                proposals.extend(_rim_stripe_candidates(image, side, band, follow_ridge=follow))
            except ValueError:
                continue
        bands.append(proposals)
    matches = []
    for us, upper, up in bands[0]:
        for ls, lower, lp in bands[1]:
            widths = [upper[1]-upper[0], lower[1]-lower[0]]
            if max(widths) > 1.33*min(widths) or abs(upper.mean()+lower.mean()+1-image.height) > min(widths):
                continue
            matches.append((np.sqrt(us*ls), np.r_[upper,lower], (up,lp)))
    if not matches:
        raise ValueError('No jointly supported upper and lower rear breaks')
    score, target, proofs = max(matches, key=lambda item:item[0])
    if any(strength >= .8*score and np.max(abs(rows-target)) > 2 for strength,rows,_ in matches):
        raise ValueError('Competing paired rear breaks remain unresolved')
    return [(proof['rows'], dict(proof, joint_layout_checked=True)) for proof in proofs]


def _projected_section(signal, gradient, alpha, row, end, flank):
    # Interleave depth tracks into three disjoint observations. Each projects
    # onto its own material difference, estimated away from either boundary.
    # Thus a one-column scratch cannot establish a stripe on its own.
    before=signal[row-flank-2:row-2].mean(axis=0)
    after=signal[end+3:end+flank+3].mean(axis=0)
    inside=signal[row+2:end-1].mean(axis=0)
    vector=inside-(before+after)/2
    sg=np.sign(vector[-1])
    if min((inside[-1]-before[-1])*sg,(inside[-1]-after[-1])*sg)>6 and min(
            sg*gradient[row,-1],-sg*gradient[end,-1])>1.5:
        return None
    strip=np.arange(signal.shape[1]-1)
    positions=[];weights=[]
    # Only use opaque metal samples throughout this particular observation.
    opaque=alpha[row-flank:end+flank+1].min(axis=0)>200
    for offset in range(3):
        cols=strip[offset::3]
        cols=cols[opaque[cols]]
        if len(cols)<2:
            return None
        material=vector[cols]
        norm=np.linalg.norm(material)
        if norm<8:
            return None
        axis=material/norm
        contrast=min((inside[cols]-before[cols])@axis,(inside[cols]-after[cols])@axis)
        slopes=min(gradient[row,cols]@axis,-gradient[end,cols]@axis)
        if contrast<7 or slopes<1.5 or abs((after[cols]-before[cols])@axis)>contrast:
            return None
        lo=row-4;hi=end+5
        profile=gradient[lo:hi,cols]@axis
        try:
            pair=[lo+gradient_peak(profile,row-lo,1,radius=3),lo+gradient_peak(profile,end-lo,-1,radius=3)]
        except ValueError:
            return None
        positions.append(pair);weights.append(contrast+2*slopes)
    positions=np.asarray(positions)
    if np.ptp(positions,axis=0).max()>2:
        return None
    return float(np.linalg.norm(weights)),np.median(positions,axis=0),vector[:-1],strip,positions


def _native_ridge(rgba, outline):
    h=rgba.shape[0]
    xx=outline[:,None]+np.arange(1,18)
    yy=np.broadcast_to(np.arange(h)[:,None],xx.shape)
    radial=ndimage.map_coordinates(rgba[...,:3].mean(axis=2),[yy,xx],order=1,mode='nearest')
    prominence=radial[:,2:-2]-np.maximum(radial[:,:-4],radial[:,4:])
    seed=np.argmax(median_filter(prominence,size=(max(5,round(h*.025))//2*2+1,1)),axis=1)+2
    near=np.clip(seed[:,None]+np.arange(-1,2),1,radial.shape[1]-2)
    yy=np.arange(h)
    peak=near[yy,radial[yy[:,None],near].argmax(axis=1)]
    a,b,c=(radial[yy,peak+k] for k in (-1,0,1))
    centre=peak+1+np.clip((c-a)/np.maximum(1.,2*(2*b-a-c)),-.5,.5)
    return centre


def _section_observations(image, side, *, projected=False, ridge_centered=False):
    """Compare the two photographed metal cross-sections on a single rear edge."""
    rgba = np.asarray(image.convert('RGBA'))
    if side == 'left':
        rgba = rgba[:, ::-1]
    h, w = rgba.shape[:2]
    outline = gaussian_filter1d(median_filter(np.argmax(rgba[...,3]>200, axis=1).astype(float),7),2.)
    offsets = np.arange(1, max(7, round(12*w/988))+1)
    if ridge_centered:
        # The metal reflection wanders sideways as the edge curves. Track its
        # photographed position on both sides of the break, holding the local
        # ridge through the interruption instead of following another highlight.
        centre=_native_ridge(rgba,outline)
        offsets=np.arange(-4,5)
        x=np.column_stack((outline[:,None]+centre[:,None]+offsets,outline+round(26*w/988)))
    else:
        x = outline[:,None] + np.r_[offsets, round(26*w/988)]
    y = np.broadcast_to(np.arange(h)[:,None], x.shape)
    signal = ndimage.map_coordinates(rgba[...,:3].mean(axis=2), [y,x], order=1, mode='nearest')
    alpha = ndimage.map_coordinates(rgba[...,3], [y,x], order=1, mode='nearest')
    signal = gaussian_filter1d(signal, max(.85,h/2058), axis=0)
    grad = np.gradient(signal,axis=0)
    margin = max(4, round(h*.003))
    options = []
    for lo,hi in ((.065,.16),(.84,.94)):
        events = []
        for row in range(round(h*lo),round(h*hi)):
            for width in range(max(6,round(h*.0044)),round(h*.0115)+1):
                end = row+width
                if row-margin-2<0 or end+margin+3>=h:
                    continue
                if projected:
                    observation=_projected_section(signal,grad,alpha,row,end,margin)
                    if observation is not None:
                        events.append(observation)
                    continue
                inside = signal[row+2:end-1].mean(axis=0)
                before = signal[row-margin-2:row-2].mean(axis=0)
                after = signal[end+3:end+margin+3].mean(axis=0)
                direction = np.sign(inside-(before+after)/2)
                contrast = np.minimum((inside-before)*direction,(inside-after)*direction)
                slope = np.minimum(grad[row]*direction, -grad[end]*direction)
                support = (contrast>6)&(slope>1.5)&(alpha[row-margin:end+margin+1].min(axis=0)>200)
                if support[-1] or support[:-1].sum()<3:
                    continue
                group = np.flatnonzero(support[:-1])
                if np.max(np.diff(group),initial=0)>3 or contrast[group].max()<12:
                    continue
                # Plastic may brighten diffuse metal while interrupting the
                # specular ridge. Each track must independently see both edges.
                try:
                    positions=np.array([[gradient_peak(grad[:,c],row,direction[c],radius=3),
                                         gradient_peak(grad[:,c],end,-direction[c],radius=3)] for c in group])
                except ValueError:
                    continue
                if np.ptp(positions,axis=0).max()>2:
                    continue
                pair=np.median(positions,axis=0)
                vector=(inside-(before+after)/2)[:-1]
                weight=float(np.linalg.norm(np.minimum(contrast[group],60)+2*slope[group]))
                events.append((weight,pair,vector,group,positions))
        # Distinct hypotheses only; adjacent integer windows frequently
        # identify the exact same two subpixel extrema.
        unique=[]
        for event in sorted(events,key=lambda z:-z[0]):
            if all(np.max(abs(event[1]-old[1]))>1 for old in unique):
                unique.append(event)
        options.append(unique)
    name=('tracked_ridge_material_sections' if ridge_centered else
          'interleaved_native_metal_cross_sections' if projected else 'paired_native_metal_cross_sections')
    return [[(*event,offsets[event[3]].tolist(),name) for event in band] for band in options]


def _section_breaks(image, side):
    collected=[[],[]]
    for projected,ridge in ((False,False),(True,False),(True,True)):
        data=_section_observations(image,side,projected=projected,ridge_centered=ridge)
        for group,new in zip(collected,data):
            group.extend(new)
        try:
            return _match_section_breaks(image.height,data)
        except ValueError:
            continue
    return _match_section_breaks(image.height,collected)


def _match_section_breaks(h, options):
    matches=[]
    for upper in options[0]:
        for lower in options[1]:
            widths=[np.diff(e[1])[0] for e in (upper,lower)]
            if max(widths)>1.33*min(widths) or abs(upper[1].mean()+lower[1].mean()+1-h)>min(widths):
                continue
            # Independently require a similar material signature at both
            # ends, allowing a one-pixel depth shift from perspective.
            u,l=upper[2],lower[2]
            if len(u)!=len(l):
                continue
            agreement=max(np.dot(u[1:-1],l[1+k:len(l)-1+k]) /
                          max(1.,np.linalg.norm(u[1:-1])*np.linalg.norm(l[1+k:len(l)-1+k])) for k in (-1,0,1))
            if agreement < .75:
                continue
            score=np.sqrt(upper[0]*lower[0])*agreement
            matches.append((score,np.r_[upper[1],lower[1]],upper,lower,agreement))
    if not matches:
        raise ValueError('No paired material signature on this rear rail')
    score,rows,upper,lower,agreement=max(matches,key=lambda z:z[0])
    if any(v>=.8*score and np.max(abs(pair-rows))>2 for v,pair,*_ in matches):
        raise ValueError('Multiple paired cross-section signatures on this rear rail')
    return [(e[1].tolist(),dict(rows=e[1].tolist(), band=band, subpixel_track_rows=e[4].tolist(),
             track_offsets_px=e[5], paired_material_agreement=float(agreement),
             method=e[6])) for band,e in zip(('upper','lower'),(upper,lower))]


def _matched_partner(image, side, *, ridge=False):
    """Use one measured break as a material reference for the other on the same rail."""
    rgba=np.asarray(image.convert('RGBA'))
    if side=='left':
        rgba=rgba[:,::-1]
    h,w=rgba.shape[:2]
    edge=gaussian_filter1d(median_filter(np.argmax(rgba[...,3]>200,axis=1).astype(float),7),2.)
    offsets=np.arange(1,max(9,round(12*w/988))+1)
    if ridge:
        offsets=np.arange(-4,5)
        xx=edge[:,None]+_native_ridge(rgba,edge)[:,None]+offsets
    else:
        xx=edge[:,None]+offsets
    yy=np.broadcast_to(np.arange(h)[:,None],xx.shape)
    colors=gaussian_filter1d(ndimage.map_coordinates(rgba[...,:3].mean(axis=2),[yy,xx],order=1),1.,axis=0)
    alpha=ndimage.map_coordinates(rgba[...,3],[yy,xx],order=1)
    glass=gaussian_filter1d(ndimage.map_coordinates(rgba[...,:3].mean(axis=2),
                 [np.arange(h),edge+round(26*w/988)],order=1),1.)
    candidates=[]
    for band in ('upper','lower'):
        try:
            pair,proof=_confirmed_rim_pair(image,side,band)
        except ValueError:
            continue
        a,b=np.rint(pair).astype(int)
        width=b-a
        pad=max(4,width//2)
        # Learn the spectral contrast from the anchor itself, well away from
        # its two edge pixels. No coordinate from the anchor is assigned.
        signature=colors[a+2:b-1].mean(axis=0)-.5*(colors[a-pad:a-2].mean(axis=0)+colors[b+3:b+pad+1].mean(axis=0))
        lines=[]
        for parity in (0,1):
            cols=np.arange(parity,len(offsets),2)
            direction=signature[cols]
            norm=np.linalg.norm(direction)
            if norm<8:break
            signal=colors[:,cols]@(direction/norm)
            gradient=np.gradient(signal)
            middle=h-1-np.mean(pair)
            observed=[]
            for first in range(round(middle-1.7*width),round(middle+.7*width)):
                lo,hi=(.84,.94) if band=='upper' else (.065,.16)
                if not h*lo<=first<=h*hi:
                    continue
                for span in range(max(6,int(np.ceil(width/1.33))),int(np.floor(1.33*width))+1):
                    end=first+span
                    if abs((first+end)/2-middle)>min(width,span):
                        continue
                    inside=signal[first+2:end-1].mean()
                    flank1=signal[first-pad:first-2]
                    flank2=signal[end+3:end+pad+1]
                    contrast=min(inside-flank1.mean(),inside-flank2.mean())
                    noise=max(1,np.std(flank1),np.std(flank2))
                    slope=min(gradient[first],-gradient[end])
                    if contrast<max(6,3*noise) or slope<1.5 or np.min(alpha[first-pad:end+pad+1,cols])<200:
                        continue
                    # The adjacent glass must not carry the same finite stripe.
                    glass_step=glass[first+2:end-1].mean()-.5*(glass[first-pad:first-2].mean()+glass[end+3:end+pad+1].mean())
                    if abs(glass_step)>max(6,.5*contrast):
                        continue
                    try:
                        loc=np.array([gradient_peak(gradient,first,1,radius=3),gradient_peak(gradient,end,-1,radius=3)])
                    except ValueError:
                        continue
                    observed.append((slope*2+contrast*.3,loc))
            if not observed:break
            best,loc=max(observed,key=lambda z:z[0])
            if any(s>=.8*best and np.max(abs(p-loc))>2 for s,p in observed):break
            lines.append(loc)
        if len(lines)!=2 or np.max(np.ptp(lines,axis=0))>2:
            continue
        target=np.median(lines,axis=0).tolist()
        measured=(target,dict(band='lower' if band=='upper' else 'upper',rows=target,
                subpixel_track_rows=np.asarray(lines).tolist(),anchor_rows=pair,
                method='same_rail_ridge_template' if ridge else 'same_rail_material_template_on_disjoint_tracks'))
        pair_proof=(pair,proof)
        candidates.append([pair_proof,measured] if band=='upper' else [measured,pair_proof])
    if not candidates:
        raise ValueError('No independent optical match to the partner break')
    if len(candidates)==2 and np.max(abs(np.array([v[0] for v in candidates[0]])-np.array([v[0] for v in candidates[1]])))>2:
        raise ValueError('Reciprocal partner measurements disagree')
    return candidates[0]


def _matched_ridge_partner(image, side):
    return _matched_partner(image,side,ridge=True)


def _ridge_height_and_area(image, side):
    """Measure each ridge interruption with channel peaks and integrated metal brightness."""
    a=np.asarray(image.convert('RGBA'));a=a[:,::-1] if side=='left' else a
    h,w=a.shape[:2];yy=np.arange(h);edge=ndimage.gaussian_filter1d(ndimage.median_filter(np.argmax(a[...,3]>200,axis=1).astype(float),7),2.)
    ridge=_native_ridge(a,edge);xx=edge[:,None]+ridge[:,None]+np.arange(-3,4);yc=np.broadcast_to(yy[:,None],xx.shape)
    rgb=np.array([ndimage.map_coordinates(a[...,c].astype(float),[yc,xx],order=1) for c in range(3)]).transpose(1,2,0)
    peaks=rgb[:,2:5].max(axis=1)
    area=rgb[:,1:6].mean(axis=(1,2))
    sig=ndimage.gaussian_filter1d(np.column_stack((peaks,area)),1.,axis=0);grad=np.gradient(sig,axis=0);opts=[]
    glass=a[yy,np.clip(np.rint(edge+26*w/988).astype(int),0,w-1),:3].mean(axis=1)
    def av(samples,indices):
        return samples[indices].mean(axis=1)
    for lo,hi in ((.065,.16),(.84,.94)):
        found=[];margin=max(5,round(.003*h));starts=np.arange(round(lo*h),round(hi*h))
        for span in range(round(.0044*h),round(.0115*h)+1):
            ends=starts+span
            mids=starts[:,None]+np.arange(2,span-1);pre=starts[:,None]-np.arange(2,margin+2);post=ends[:,None]+np.arange(2,margin+2)
            inner=av(sig,mids);outer1=av(sig,pre);outer2=av(sig,post)
            diff=np.minimum(outer1-inner,outer2-inner)
            strength=np.minimum(-grad[starts],grad[ends])
            supported=(diff>8)&(strength>1.5)&(diff>2*np.maximum(sig[pre].std(axis=1),sig[post].std(axis=1)))
            # Check that this is a finite interruption of a polished metal stripe.
            shoulders=rgb[:,[0,-1]].mean(axis=2).max(axis=1)
            prom=peaks.mean(axis=1)-shoulders
            valid=(supported[:,:3].sum(axis=1)>=2)&supported[:,3]&(np.minimum(av(prom,pre),av(prom,post))>12)
            valid &= (abs(av(glass,mids)-.5*(av(glass,pre)+av(glass,post)))<np.maximum(6,.45*diff[:,-1]))
            for idx in np.flatnonzero(valid):
                cols=np.flatnonzero(supported[idx]);y,z=starts[idx],ends[idx]
                try:pos=np.array([[gradient_peak(grad[:,c],y,-1,3),gradient_peak(grad[:,c],z,1,3)] for c in cols])
                except ValueError:continue
                if np.ptp(pos,axis=0).max()>2:continue
                st=float(np.median(diff[idx,cols]+2*strength[idx,cols]));found.append((st,np.median(pos,axis=0),pos))
        keep=[]
        for o in sorted(found,key=lambda x:-x[0]):
            if all(np.max(abs(o[1]-p[1]))>1 for p in keep):keep.append(o)
        opts.append(keep)
    matches=[]
    for upper in opts[0]:
        for lower in opts[1]:
            widths=[np.diff(z[1])[0] for z in (upper,lower)]
            if max(widths)>1.33*min(widths) or abs(upper[1].mean()+lower[1].mean()+1-h)>min(widths):continue
            matches.append((np.sqrt(upper[0]*lower[0]),np.r_[upper[1],lower[1]],upper,lower))
    if not matches:raise ValueError('No paired interruption of the native metal ridge')
    best=max(matches,key=lambda x:x[0])
    if any(o[0]>=.8*best[0] and np.max(abs(o[1]-best[1]))>2 for o in matches):raise ValueError('Competing ridge interruptions on this rear edge')
    return [(v[1].tolist(),dict(band=band,rows=v[1].tolist(),subpixel_track_rows=v[2].tolist(),method='native_ridge_height_and_area')) for band,v in zip(('upper','lower'),best[2:])]


def _confirmed_rim_pair(image, side, band):
    observations = []
    for follow_ridge in (False, True):
        try:
            observations.append(_coherent_rim_stripe(image, side, band, follow_ridge=follow_ridge))
        except ValueError:
            continue
    if not observations:
        raise ValueError('Rear stripe lacks independent metal support')
    if len(observations) == 2:
        fixed, ridge = observations
        if np.max(np.abs(np.subtract(fixed[0], ridge[0]))) > 2:
            raise ValueError('Fixed and ridge-following metal tracks disagree')
        fixed[1]['ridge_confirmation_rows'] = ridge[0]
    return observations[0]


def _paired_rim_stripes(image, side):
    measured = [_confirmed_rim_pair(image, side, band) for band in ('upper', 'lower')]
    upper, lower = (np.asarray(pair) for pair, _ in measured)
    widths = [upper[1]-upper[0], lower[1]-lower[0]]
    # The two physical breaks of this standard glass-back rail are near
    # opposite ends. An unrelated scratch can have sharp local boundaries.
    # Use the paired layout only to reject a mismatched pair, never to
    # replace a coordinate, average edges or infer a missing break.
    end_difference = abs(upper.mean() - (image.height-1-lower.mean()))
    if end_difference > min(widths) or max(widths) > 1.33*min(widths):
        raise ValueError('Rear stripe pair is inconsistent with opposite ends of the same rail')
    return measured


def rear_antennas(image, *, refine_bands=('upper','lower'), sides=('left','right')):
    result = {}
    for side in sides:
        try:
            found = _rear_antennas(image,refine_bands=refine_bands,sides=(side,))
        except ValueError as original:
            w,h = image.size
            glass = image.crop((int(.4*w),int(.43*h),int(.6*w),int(.56*h))).convert('RGB')
            dark = np.median(np.asarray(glass))<95
            if dark:
                try:
                    found = _rear_antennas(image,reflective_rim=True,refine_bands=refine_bands,sides=(side,))
                    found[side]['reference_track'] = 'dark_break_across_three_specular_ridge_tracks'
                    result.update(found)
                    continue
                except ValueError:
                    pass
            entry = dict(rows=[], checks=[], rear_image_edge='right' if side=='left' else 'left')
            for band in ('upper','lower'):
                for reflective in ((False,True) if dark else (False,)):
                    try:
                        measured = _rear_antennas(image, reflective_rim=reflective,refine_bands=refine_bands,
                                                   sides=(side,),bands=(band,))[side]
                        pair, check = measured['rows'], measured['checks'][0]
                        break
                    except ValueError:
                        continue
                else:
                    try:
                        pair,check = _rim_band_witnesses(image,side,band)
                    except ValueError:
                        # If the legacy reference is incomplete, independently
                        # remeasure both bands on this same metal edge. Mixing
                        # the new stripe with an unrelated old reflection can
                        # yield a numerically precise but false rear target.
                        detectors=(_paired_rim_stripes,_joint_rim_stripes,_section_breaks,_matched_partner,_matched_ridge_partner,_ridge_height_and_area)
                        for detector in detectors:
                            try:
                                measured_bands=detector(image,side)
                                break
                            except ValueError:
                                continue
                        else:
                            raise original from None
                        entry['rows'] = [row for pair, _ in measured_bands for row in pair]
                        entry['checks'] = [proof for _, proof in measured_bands]
                        break
                entry['rows'].extend(pair)
                entry['checks'].append(check)
            found = {side:entry}
        result.update(found)
    return result

def hardware(image,side):
    a=np.asarray(image.convert('RGBA'));h,w=a.shape[:2]
    gray=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY)
    boxes=[]
    for threshold in (130,160):
        mask=((gray<threshold)&(a[...,3]>=240)).astype('uint8')
        mask[:int(h*.15)]=0;mask[int(h*.85):]=0
        mask[:,:int(w*.2)]=0;mask[:,int(w*.8):]=0
        contours,_=cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x,y,bw,bh=cv2.boundingRect(contour)
            if .2*w<bw<.7*w and .035*h<bh<.19*h and bh/bw>2:
                boxes.append([x,y,x+bw,y+bh])
    selected=[]
    for box in sorted(boxes,key=lambda b:(b[2]-b[0])*(b[3]-b[1]),reverse=True):
        if any(min(box[3],q[3])-max(box[1],q[1])>.7*min(box[3]-box[1],q[3]-q[1]) for q in selected):continue
        selected.append(box)
    selected.sort(key=lambda b:b[1])
    expected=4 if side=='left' else 2
    if len(selected)!=expected:
        # Dark rails can merge with the absolute-threshold mask. Use the
        # same complete capsule, cap and rail witnesses as the workshop intake.
        from .side_material import hardware_materials
        try:
            parts = hardware_materials(image,side)
        except ValueError:
            raise ValueError(f'{side}: expected {expected} independent complete capsule contours, found {len(selected)}') from None
        selected = [part['box'] for _,part in parts]
    return selected


def capsule_mask(image,box,side=None):
    """Use the actual closed source outline, never the detection box as paint."""
    a=np.asarray(image.convert('RGBA'));gray=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY)
    h,w=gray.shape;x0,y0,x1,y1=box
    roi=np.zeros((h,w),np.uint8)
    left,top,right,bottom=max(0,x0-2),max(0,y0-2),min(w,x1+2),min(h,y1+2)
    roi[top:bottom,left:right]=(gray[top:bottom,left:right]<160).astype('uint8')
    contours,_=cv2.findContours(roi,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for c in contours:
        x,y,bw,bh=cv2.boundingRect(c)
        if (x,y,x+bw,y+bh)==(left,top,right,bottom) and cv2.contourArea(c)>=(bw-1)*(bh-1)*.98:
            continue  # The artificial ROI perimeter on an entirely dark surface.
        if max(abs(np.asarray([x,y,x+bw,y+bh])-box))<=4 and cv2.contourArea(c)>.5*(x1-x0)*(y1-y0):
            candidates.append(c)
    if not candidates:
        if side is not None:
            from .side_material import hardware_materials
            matches = [mask for mask,proof in hardware_materials(image,side)
                       if np.max(np.abs(np.asarray(proof['box'])-box))<=2]
            if len(matches)==1:
                return matches[0]
        raise ValueError('Detected capsule lacks a complete source contour')
    contour=max(candidates,key=cv2.contourArea);mask=np.zeros((h,w),np.uint8)
    cv2.drawContours(mask,[contour],-1,1,cv2.FILLED)
    return mask>0


def propose(files, *, refine_bands=('upper','lower'), sides=('left','right')):
    back=Image.open(files['back']);result={}
    rear=rear_antennas(back,refine_bands=refine_bands, sides=sides)
    for side in sides:
        image=Image.open(files[side]).convert('RGBA');h,w=image.height,image.width
        rows,signs,proof=antenna_rows(image,[int(w*.4),int(w*.6)])
        lines=per_column_boundaries(image,rows)
        boxes=hardware(image,side)
        back_rows=np.asarray(rear[side]['rows'],dtype=float)
        target=(back_rows+.5)/back.height*h-.5
        cap=max(8,int(w*.75));middle=[max(0,min(b[1] for b in boxes)-8),min(h,max(b[3] for b in boxes)+8)]
        if not cap+4<min(lines[0].min(),target[0]) or not max(lines[-1].max(),target[-1])<h-cap-4:
            raise ValueError('Standard antenna would intersect a rounded end')
        if middle[0]<=max(lines[1].max(),target[1])+4 or middle[1]>=min(lines[2].min(),target[2])-4:
            raise ValueError('Standard hardware intersects antenna remap')
        source=np.vstack([np.full(w,cap),lines[:2],np.full(w,middle[0]),np.full(w,middle[1]),lines[2:],np.full(w,h-1-cap)])
        dest=source.copy();dest[[1,2,5,6]]=target[:,None]
        result[side]=dict(source_sha256=digest(files[side]),back_sha256=digest(files['back']),
            basis='same_sku_back_rim_measured_pixel_center_ratio',back_antenna_present=True,
            rear_detection=rear[side],back_size=list(back.size),
            reference_policy='corresponding_rear_edge',side=side,
            antenna_lines=lines.tolist(),antenna_rows=rows,signs=signs,source_detection=proof,
            rear_canvas_rows=back_rows.tolist(),target_rows=target.tolist(),hardware_boxes=boxes,
            locked_middle=middle,endcap_ranges=[[0,cap+1],[h-1-cap,h]],source=source.tolist(),target=dest.tolist())
        try:
            profile_lines=_profile_edges(image,rows,lines)
        except ValueError:
            continue
        if not np.array_equal(lines,profile_lines):
            result[side]['source_profile_lines']=profile_lines.tolist()
    return result


def measure_side(image,target,signs):
    b=np.asarray(image.convert('RGBA'));h,w=b.shape[:2];gray=b[...,:3].mean(2);checks={}
    for name,lo,hi in [('left',.08,.23),('center',.4,.6),('right',.77,.92)]:
        sig=np.median(gray[:,int(w*lo):int(w*hi)],axis=1)
        g=np.gradient(ndimage.gaussian_filter1d(sig,1.))
        measured=[gradient_peak(g,y,sign) for y,sign in zip(target,signs)]
        checks[name]=dict(measured=measured,max_error_px=float(np.max(abs(np.asarray(measured)-target))))
    return checks



def _warp_along_rail(image, source, dest, *, interpolate_profile=False):
    """Follow the observed rail when the source and destination rows taper differently."""
    from .iphone17_buttons import _sample_xy
    from texture_alignment.align_textures import anchor_matrix
    w,h = image.size
    src = anchor_matrix(source,h,w,'source')
    dst = anchor_matrix(dest,h,w,'target')
    yy,xx = np.indices((h,w),dtype=float)
    sy = np.stack([np.interp(yy[:,x],dst[:,x],src[:,x]) for x in range(w)],axis=1)
    alpha = np.asarray(image.getchannel('A'),dtype=float)
    sx = xx.copy()
    for field, x, sign in ((alpha,xx,1),(alpha[:,::-1],w-1-xx,-1)):
        hit = field>=128
        first = hit.argmax(axis=1)
        before = field[np.arange(h),np.maximum(0,first-1)]
        after = field[np.arange(h),first]
        rail = np.maximum(0,first-1+(128-before)/np.maximum(1,after-before))
        rail[first==0] = 0
        moved = (sy!=yy) & (x < rail[:,None]+10)
        if not moved.any():
            continue
        y = sy[moved]
        if not (hit.any(axis=1)[np.floor(y).astype(int)].all() and
                hit.any(axis=1)[np.ceil(y).astype(int)].all() and
                hit.any(axis=1)[yy[moved].astype(int)].all()):
            raise ValueError('Antenna correction has no observed rail')
        shift = np.interp(y,np.arange(h),rail)-rail[yy[moved].astype(int)]
        if interpolate_profile:
            # Interpolating crossing *positions* is only exact when the two
            # photographed edge slopes agree. Locate half coverage on the
            # bilinear sampler's actual blended alpha profile instead.
            low = np.floor(y).astype(int)
            high = np.ceil(y).astype(int)
            frac = (y-low)[:, None]
            blended = field[low]*(1-frac) + field[high]*frac
            inside = blended >= 128
            if not inside.any(axis=1).all():
                raise ValueError('Antenna correction has no observed interpolated rail')
            first = inside.argmax(axis=1)
            indices = np.arange(len(y))
            outer = blended[indices, np.maximum(first-1, 0)]
            inner = blended[indices, first]
            crossing = np.where(first == 0, 0., first-1+(128-outer)/np.maximum(1, inner-outer))
            shift = crossing-rail[yy[moved].astype(int)]
        if abs(shift).max()>3:
            raise ValueError('Antenna rail moves beyond the observed local contour')
        weight = np.clip(1-(x[moved]-rail[yy[moved].astype(int)])/10,0,1)
        sx[moved] += sign*shift*weight
    # The tight crop can bisect an antialiased outer sample. Its subpixel
    # footprint terminates at the last photographed column. Keep the actual
    # boundary sample only for that fractional, translucent footprint.
    outside = (sx<0)|(sx>w-1)
    beyond = (sx < -1)|(sx > w)
    if np.any(outside & (alpha>=128)) or np.any(beyond & (alpha>0)):
        raise ValueError('Antenna correction would extrapolate the body rail')
    sx[beyond] = xx[beyond]
    sy[beyond] = yy[beyond]
    sx = np.clip(sx,0,w-1)
    return _sample_xy(image,sx,sy)

def _profile_edges(image, rows, baseline):
    """Trace each source transition using its photographed central intensity profile."""
    rgba=np.asarray(image.convert('RGBA'))
    h,w=rgba.shape[:2]
    gray=gaussian_filter1d(rgba[...,:3].mean(axis=2),1.,axis=0)
    x=np.arange(w)
    lines=[]
    for idx,row in enumerate(rows):
        width=rows[idx//2*2+1]-rows[idx//2*2]
        radius=max(4,round(width*.48))
        ys=np.arange(int(round(row))-radius,int(round(row))+radius+1)
        ref=np.median(gray[ys,int(.4*w):int(.6*w)],axis=1)
        ref=ref-ref.mean()
        if np.linalg.norm(ref)<5:
            raise ValueError('Source profile has insufficient optical contrast')
        scores=[];shifts=np.arange(-max(4,width//2),max(4,width//2)+1)
        for shift in shifts:
            patch=gray[ys+shift]
            observed=patch-patch.mean(axis=0)
            corr=(observed.T@ref)/np.maximum(.001,np.linalg.norm(observed,axis=0)*np.linalg.norm(ref))
            corr[np.min(rgba[ys+shift,:,3],axis=0)<128]=-1
            scores.append(corr)
        scores=np.array(scores);peak=scores.argmax(axis=0);best=scores[peak,x]
        ok=(best>=.88)&(peak>0)&(peak<len(shifts)-1)
        for j in np.flatnonzero(ok):
            if np.max(scores[abs(shifts-shifts[peak[j]])>=3,j],initial=-1)>best[j]-.025:
                ok[j]=False
        if ok.sum()<.6*w:
            lines.append(baseline[idx].copy())
            continue
        u=np.flatnonzero(ok)
        k=peak[u]
        a,b,c=scores[k-1,u],scores[k,u],scores[k+1,u]
        delta=np.clip((c-a)/np.maximum(.001,2*(2*b-a-c)),-.5,.5)
        line=baseline[idx].copy()
        line[u]=row+shifts[k]+delta
        lines.append(line)
    lines=np.asarray(lines)
    if np.any(np.diff(lines,axis=0)<=2):
        raise ValueError('Source profile edges intersect')
    return lines


def _warp_inside_native_rail(image, source, dest):
    """Map between the photographed spans on each bracketing source row."""
    from texture_alignment.align_textures import anchor_matrix
    rgba=np.asarray(image.convert('RGBA'))
    h,w=rgba.shape[:2]
    src=anchor_matrix(source,h,w,'source')
    dst=anchor_matrix(dest,h,w,'target')
    yy,xx=np.indices((h,w))
    sy=np.stack([np.interp(yy[:,col],dst[:,col],src[:,col]) for col in range(w)],axis=1)
    moved=(np.abs(sy-yy)>1e-8)&(rgba[...,3]>0)
    oy,ox=np.nonzero(moved)
    alpha=rgba[...,3]
    spans=[]
    for threshold in (1,128,240):
        hit=alpha>=threshold
        spans.append((hit.argmax(axis=1),w-1-hit[:,::-1].argmax(axis=1),hit.any(axis=1)))
    channels=np.zeros((len(ox),3))
    total=np.zeros(len(ox))
    low=np.floor(sy[moved]).astype(int)
    frac=sy[moved]-low
    for ry,yweight in ((low,1-frac),(np.minimum(h-1,low+1),frac)):
        # Left/right outer, half and opaque crossings form ordered knots.
        # At a cropped edge several knots can share one column; the inner
        # (more opaque) knot owns that sample. This avoids folding the edge.
        if any(not (valid[ry]&valid[oy]).all() for _,_,valid in spans):
            raise ValueError('Native rail has an unobserved source row')
        left,right,_=spans[1]
        dw=right[oy]-left[oy]
        sw=right[ry]-left[ry]
        if np.any(dw<=0) or np.any(abs(sw-dw)>np.maximum(3,.06*dw)):
            raise ValueError('Native rail spans have incompatible widths')
        knots=[s[0] for s in spans]+[s[1] for s in reversed(spans)]
        sample_x=np.zeros(len(ox))
        for idx in (0,4,1,3,2):
            start,end=knots[idx],knots[idx+1]
            length=end[oy]-start[oy]
            take=(ox>=start[oy])&(ox<=end[oy])&(length>0)
            blend=(ox[take]-start[oy[take]])/length[take]
            sample_x[take]=start[ry[take]]+blend*(end[ry[take]]-start[ry[take]])
        if np.any(abs(sample_x-ox)>max(4,.06*w)):
            raise ValueError('Native rail trajectories are too far apart')
        ix=sample_x.astype(int)
        fx=sample_x-ix
        for cx,xweight in ((ix,1-fx),(np.minimum(w-1,ix+1),fx)):
            pix=rgba[ry,cx]
            weight=xweight*yweight*pix[:,3]
            channels+=weight[:,None]*pix[:,:3]
            total+=weight
    if np.any(total<=0) or np.any((alpha[moved]>=240)&(total<128)):
        raise ValueError('Native rail sampling lacks matching photographed coverage')
    # The move changes the material coordinates along the rail. Its original
    # antialiased silhouette is already in the desired destination coordinates.
    result=rgba.copy()
    result[oy,ox,:3]=np.rint(channels/total[:,None]).clip(0,255).astype('uint8')
    return Image.fromarray(result)


def apply(image,cfg, *, refine_indices=(0,1,2,3)):
    try:
        return _apply(image,cfg,refine_indices=refine_indices)
    except ValueError as exc:
        if str(exc) not in ('Antenna rail moves beyond the observed local contour',
                            'Antenna correction would extrapolate the body rail',
                            'Sampling correction exceeds source-bound interpolation limit'):
            raise
        original=exc
    try:
        result,proof=_apply(image,cfg,refine_indices=refine_indices,native_span=True)
    except ValueError:
        if str(original)!='Sampling correction exceeds source-bound interpolation limit' or 'source_profile_lines' not in cfg:
            raise original from None
        anchors=np.array(cfg['source'],copy=True)
        anchors[[1,2,5,6]]=cfg['source_profile_lines']
        try:
            result,proof=_apply(image,dict(cfg,source=anchors),refine_indices=refine_indices,native_span=True)
        except ValueError:
            raise original from None
        proof['source_profile_measurement']=True
    proof['native_rail_spans']=True
    return result,proof


def _apply(image,cfg, *, refine_indices, native_span=False):
    if tuple(refine_indices) not in ((0,1),(2,3),(0,1,2,3)):
        raise ValueError('Refine both boundaries of each selected antenna together')
    a=np.asarray(image.convert('RGBA'));h,w=a.shape[:2]
    if cfg.get('basis')!='same_sku_back_rim_measured_pixel_center_ratio' or cfg.get('back_antenna_present') is not True:
        raise ValueError('Measured rear rim antenna reference is required; native-side projection is invalid')
    edge={'left':'right','right':'left'}.get(cfg.get('side'))
    if cfg.get('reference_policy')!='corresponding_rear_edge' or cfg['rear_detection']['rear_image_edge']!=edge:
        raise ValueError('Each side must use its own corresponding rear edge, never a common opposite-side target')
    rear=np.asarray(cfg['rear_detection']['rows'],dtype=float)
    expected=(rear+.5)/cfg['back_size'][1]*h-.5
    if not np.allclose(expected,cfg['target_rows'],atol=1e-6):
        raise ValueError('Side targets differ from measured rear pixel-centre proportions')
    source=np.array(cfg['source'],copy=True);original_source=source.copy();dest=np.asarray(cfg['target'])
    # Re-render from the unchanged lighting baseline each iteration. Small
    # interpolation-phase corrections do not redefine the frozen rear target.
    anchors=[1,2,5,6];positions=np.array([.155,.5,.845])*w
    factor=cfg['back_size'][1]/h
    follow_rail = False
    interpolate_profile = False
    for attempt in range(5):
        out=warp_rows(image,source,dest)
        if np.any((a[...,3]>=240)&(np.asarray(out)[...,3]<128)):
            follow_rail = True
        if native_span:
            out=_warp_inside_native_rail(image,source,dest)
        elif follow_rail:
            out=_warp_along_rail(image,source,dest,interpolate_profile=interpolate_profile)
            if not interpolate_profile and np.any((a[...,3]>=240)&(np.asarray(out)[...,3]<128)):
                interpolate_profile = True
                out=_warp_along_rail(image,source,dest,interpolate_profile=interpolate_profile)
        checks=measure_side(out,expected,cfg['signs'])
        errors=np.asarray([checks[k]['measured'] for k in ('left','center','right')])-expected
        widths=(errors[:,[1,3]]-errors[:,[0,2]])*factor
        if max(abs(errors[:,refine_indices]).ravel())*factor<=.6 and max(abs(widths[:,[i//2 for i in refine_indices if i%2]]).ravel(),default=0)<=.8:
            break
        if attempt==4:raise ValueError('Final rendered antenna fails subpixel rear-reference alignment')
        for i in refine_indices:
            source[anchors[i]]+=np.interp(np.arange(w),positions,errors[:,i])
        if np.max(abs(source-original_source))>2.5:
            raise ValueError('Sampling correction exceeds source-bound interpolation limit')
    b=np.asarray(out)
    for lo,hi in [cfg['locked_middle'],*cfg['endcap_ranges']]:
        if not np.array_equal(a[lo:hi],b[lo:hi]):raise ValueError('Standard rectification changed a locked button/endcap')
    if np.any((a[...,3]>=240)&(b[...,3]<128)):
        raise ValueError('Standard band rectification lost an opaque body pixel')
    if max(c['max_error_px'] for c in checks.values())>2.5:
        raise ValueError('Standard antenna endpoint failed alignment')
    return out,dict(endpoint_checks=checks,alpha_changed=int(np.count_nonzero(a[...,3]!=b[...,3])),
        opaque_body_lost=0,buttons_changed=0,endcaps_changed=0,followed_native_rail=follow_rail,
        rail_interpolate_profile=interpolate_profile if follow_rail else None,
        model_uv_verified=False,basis=cfg['basis'],back_sha256=cfg['back_sha256'],
        rear_reference_rows=rear.tolist(),rear_image_edge=cfg['rear_detection']['rear_image_edge'],
        target_rows=expected.tolist(),reference_policy=cfg['reference_policy'],side=cfg['side'],
        refined_indices=list(refine_indices),render_iterations=attempt+1,
        rendered_error_rear_px=(errors*factor).tolist(),rendered_width_error_rear_px=widths.tolist(),
        sampling_correction=(source-original_source).tolist())


def validate_render_checks(check):
    """Release checks on the rendered raster, in rear-image pixel units."""
    expected_edge={'left':'right','right':'left'}.get(check.get('side'))
    if check.get('reference_policy')!='corresponding_rear_edge' or check.get('rear_image_edge')!=expected_edge:
        raise ValueError('Incorrect corresponding rear edge in rendered antenna review')
    indices=check.get('refined_indices')
    if indices not in ([0,1],[2,3],[0,1,2,3]):
        raise ValueError('Rendered boundary and width checks are missing')
    errors=np.asarray(check.get('rendered_error_rear_px',[]))
    widths=np.asarray(check.get('rendered_width_error_rear_px',[]))
    if errors.shape!=(3,4) or widths.shape!=(3,2) or not np.isfinite(errors).all() or not np.isfinite(widths).all():
        raise ValueError('Three independent rendered endpoint/centre checks are required')
    if np.max(abs(errors[:,indices]))>.6 or np.max(abs(widths[:,sorted(set(i//2 for i in indices))]))>.8:
        raise ValueError('Rendered rear-reference boundary/width residual exceeds tolerance')


def refine_lower(files,lighting_path,previous_path,side):
    """Recompute only the lower mapping from the original lighting baseline."""
    cfg=propose(files,refine_bands=('lower',))[side]
    out,check=apply(Image.open(lighting_path),cfg,refine_indices=(2,3))
    previous=np.asarray(Image.open(previous_path).convert('RGBA'));actual=np.asarray(out)
    if previous.shape!=actual.shape:raise ValueError('Local correction changed the canvas')
    lo=cfg['locked_middle'][1];hi=cfg['endcap_ranges'][1][0]
    if not np.array_equal(previous[:lo+1],actual[:lo+1]) or not np.array_equal(previous[hi:],actual[hi:]):
        raise ValueError('Lower correction changed an accepted upper antenna, button, or endcap')
    validate_render_checks(check)
    check['previous_sha256']=digest(previous_path)
    check['unchanged_row_ranges']=[[0,lo+1],[hi,len(actual)]]
    return out,cfg,check
