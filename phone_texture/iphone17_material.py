"""Current-image material proposals for the glass-back standard iPhone 17."""
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage
from .auto_material import (paper_mask, save_mask, reference_box,
                            front_metal, rear_body_contour)
from .geometry_review import digest


def paper(image, face):
    if face == 'bottom':
        return np.zeros(image.size[::-1], dtype=bool), dict(method='bottom_lit_without_label_boundary', protected_pixels=0)
    # The Pro printed-tail reconstruction assumes the Pro label layout.
    return paper_mask(image, face, source_tail=False)


def _standard_paper_end(a, proposal, proof):
    h,w = a.shape[:2]
    gray = cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY).astype(float)
    # The standard label has a different print layout. Measure its physical
    # lower boundary, rather than predicting it from the Pro's red dot.
    cols = np.flatnonzero(proposal[h//2])
    if len(cols)<12:
        raise ValueError('Bottom paper sides are unresolved')
    interior = cols[3:-3]
    gy = np.abs(np.gradient(ndimage.gaussian_filter1d(gray,.7,axis=0),axis=0))
    gx = np.abs(np.gradient(ndimage.gaussian_filter1d(gray,.7,axis=1),axis=1))
    edges=np.asarray(proof['source_strip_edges'])
    side_responses=[]
    for line in edges:
        centers=np.polyval(line,np.arange(h)+.5)
        xx=np.rint(centers[:,None]+np.arange(-2,4)).astype(int)
        side_responses.append(gx[np.arange(h)[:,None],np.clip(xx,0,w-1)].max(axis=1))
    side_response=np.minimum(*side_responses)
    baseline=float(np.median(side_response[int(h*.58):int(h*.72)]))
    absent=ndimage.uniform_filter1d((side_response<baseline*.4).astype(float),3)>=.99
    candidates=np.flatnonzero(absent&(np.arange(h)>h*.6)&(np.arange(h)<h*.9))
    if baseline<3 or not len(candidates):
        return cross_strip_end(proposal, proof, gray, cols, side_responses)
    side_end=int(candidates[0])
    rows = np.arange(max(int(h*.56),side_end-7),min(h-2,side_end+1))
    response = np.median(gy[rows[:,None],interior[None,:]],axis=1)
    peak = int(np.argmax(response))
    if peak==0 or response[peak]<1:
        raise ValueError('Standard bottom paper end lacks independent RGB support')
    center = int(rows[peak]); search = np.arange(max(rows[0],center-3),min(side_end+1,center+4))
    local = gy[search[:,None],interior[None,:]]
    ends = search[np.argmax(local,axis=0)]
    supported = local.max(axis=0)>1
    if supported.mean()<.8 or np.percentile(np.abs(ends-np.median(ends)),90)>4:
        raise ValueError('Bottom paper end lacks cross-strip consensus')
    # Keep native per-column irregularities and their AA; no old rectangle.
    boundary=np.interp(np.arange(w),interior,ndimage.median_filter(ends,size=5))+2
    result=proposal&(np.arange(h)[:,None]<=boundary[None,:])
    proof.update(end_method='source_gradient_cross_strip_consensus',
        source_end_y_by_column=boundary[cols].tolist(),source_end_columns=cols.tolist(),
        end_support_fraction=float(supported.mean()),end_response=float(response[peak]))
    return result,proof


def cross_strip_end(proposal, proof, gray, cols, side_responses):
    """Find a paper end bounded by its two measured sides, excluding metal rails."""
    h,w = gray.shape
    if not .44*h <= len(cols) <= .73*h:
        raise ValueError('Paper side endings are unresolved')
    interior = cols[4:-4]
    signed = -np.gradient(ndimage.gaussian_filter1d(gray,.7,axis=0),axis=0)
    rows = np.arange(int(.58*h),int(.96*h))
    responses = np.median(signed[:,interior],axis=1)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(responses[rows], height=5, distance=4)
    accepted = []
    for y in rows[peaks]:
        width = np.arange(y-2,y+3)
        field = signed[width[:,None],interior[None,:]]
        ends = width[np.argmax(field,axis=0)]
        if (field.max(axis=0)>4).mean()<.9 or np.percentile(abs(ends-y),90)>1:
            continue
        lo,hi = (round(np.polyval(line,y+.5)) for line in proof['source_strip_edges'])
        # A body bevel continues through both exterior holdouts. A label edge
        # must stop at both measured paper sides and have support at each end.
        probes = [np.arange(lo+2,lo+7),np.arange(hi-6,hi-1),
                  np.arange(lo-8,lo-3),np.arange(hi+4,hi+9)]
        if min(lo, w-hi)<10:
            continue
        left,right,out_l,out_r = [np.median(signed[width[:,None],x].max(axis=0)) for x in probes]
        if min(left,right)<5 or max(out_l,out_r)>min(left,right)*.4:
            continue
        # Compare each side to itself in neighbouring rows, avoiding a global
        # brightness threshold for pale metal.
        if any(np.median(side[y-8:y-2])<2 or
               np.median(side[y+3:y+7])>np.median(side[y-8:y-2])*.65 for side in side_responses):
            continue
        accepted.append((y,ends))
    if len(accepted)!=1:
        raise ValueError('Paper side endings are unresolved')
    y,ends = accepted[0]
    boundary = np.interp(np.arange(w),interior,ndimage.median_filter(ends,size=5))+2
    mask = proposal & (np.arange(h)[:,None]<=boundary[None,:])
    return mask, dict(proof, end_method='independent_paper_corners_and_exterior_metal_holdouts',
                      source_end_columns=cols.tolist(),source_end_y_by_column=boundary[cols].tolist(),
                      end_response=float(responses[y]))


def _glass_plateau_edge(gray, indices, sections, vertical):
    """Independently locate a metal-to-glass transition across the same edge."""
    from scipy.signal import find_peaks
    readings=[]
    for section in np.array_split(sections,5):
        profile=np.median(gray[section],axis=0) if vertical else np.median(gray[:,section],axis=1)
        values=profile[indices]
        gradient=np.gradient(ndimage.gaussian_filter1d(values,.7))
        amplitude=abs(gradient)
        peaks=find_peaks(amplitude,height=3.5,prominence=1)[0]
        plateau=np.median(values[-6:])
        observed=[]
        for k in peaks:
            if k<2 or k+8>=len(values):
                continue
            change=plateau-np.median(values[k-2:k])
            after=values[k+4:k+9]
            if (abs(change)>=6 and np.sign(change)==np.sign(gradient[k])
                    and abs(np.median(after)-plateau)<=5 and np.ptp(after)<=5
                    and amplitude[k+4:].max()<=3):
                observed.append((int(k),int(np.sign(change))))
        if len(observed)>1:
            return None
        readings.append(observed[0] if observed else None)
    found=[r for r in readings if r is not None]
    if (len(found)<4 or not readings[0] or not readings[-1]
            or len({r[1] for r in found})!=1 or np.ptp([r[0] for r in found])>2):
        return None
    return int(np.median([r[0] for r in found]))


def glass_back_rim(image,sticker=None, *, allow_partial=False):
    """Measure the bright glass transition inwards from the body silhouette."""
    a=np.array(image.convert('RGBA'));h,w=a.shape[:2]
    body,body_proof=rear_body_contour(image)
    if sticker is not None and sticker.any():
        # Reconstruct only the illumination coordinate across the occluding
        # paper tail. The photo/alpha and paper pixels are not modified.
        columns=np.arange(w);bottom=h-1-np.argmax(body[::-1],axis=0)
        affected=sticker[int(h*.85):].any(axis=0)
        good=(~affected)&body.any(axis=0)
        if good.sum()<w*.7:raise ValueError('Rear lower body contour is occluded by paper')
        repaired=np.interp(columns,columns[good],bottom[good])
        body[:,affected]&=np.arange(h)[:,None]<=repaired[None,affected]
    distance=ndimage.distance_transform_edt(np.pad(body,1))[1:-1,1:-1]
    # The body contour is a lighting coordinate, not the delivered alpha.
    gray=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY).astype(float)
    ys=np.arange(int(h*.32),int(h*.68));xs=np.arange(int(w*.32),int(w*.68))
    opaque=a[...,3]>=128
    edges=[float(np.median(np.argmax(opaque[ys],axis=1))),
           float(np.median(w-1-np.argmax(opaque[ys,::-1],axis=1))),
           float(np.median(np.argmax(opaque[:,xs],axis=0))),
           float(np.median(h-1-np.argmax(opaque[::-1,xs],axis=0)))]
    widths=[];measurements=[]
    for i,(edge,sign) in enumerate(zip(edges,(1,-1,1,-1))):
        offsets=np.arange(3,max(10,int(w*.035)))
        indices=np.rint(edge+sign*offsets).astype(int)
        if i<2:
            signal=np.median(gray[ys],axis=0)
        else:
            signal=np.median(gray[:,xs],axis=1)
        gradient=sign*np.gradient(ndimage.gaussian_filter1d(signal,.7))
        values=gradient[indices]
        from scipy.signal import find_peaks
        peaks,_=find_peaks(values,height=max(.8,values.max()*.35))
        if not len(peaks) and np.median(gray[h//2,w//3:w*2//3])<95:
            # A specular outer lip can be the largest response at the first
            # sample. Identify the inner transition adjoining flat dark glass
            # in three disjoint spans along this same edge.
            readings=[]
            for section in np.array_split(ys if i<2 else xs,3):
                local=np.median(gray[section],axis=0) if i<2 else np.median(gray[:,section],axis=1)
                samples=local[indices]
                response=np.gradient(ndimage.gaussian_filter1d(samples,.7))
                features=find_peaks(response,height=1.5)[0]
                stable=[k for k in features if k>=2 and k+7<len(samples)
                        and abs(np.median(samples[k+4:k+8])-np.median(samples[-5:]))<=3
                        and np.ptp(samples[k+4:k+8])<=3]
                if len(stable)!=1:break
                readings.append(stable[0])
            if len(readings)==3 and np.ptp(readings)<=2:
                peaks=np.array([int(np.median(readings))])
        if not len(peaks):
            crossing=_glass_plateau_edge(gray,indices,ys if i<2 else xs,i<2)
            if crossing is None:
                if allow_partial:
                    widths.append(0.)
                    continue
                raise ValueError('Glass-back metal boundary lacks source gradient')
            peaks=np.array([crossing])
        k=int(peaks[np.argmax(values[peaks])])
        width=float(offsets[k]+1)
        if not 3<=width<=w*.035:
            if allow_partial:
                widths.append(0.)
                continue
            raise ValueError('Implausible standard rear rim width')
        widths.append(width);measurements.append(dict(edge=i,width_px=width,gradient=float(values[k])))
    dy,dx=np.gradient(ndimage.gaussian_filter(distance,3));den=np.maximum(abs(dx)+abs(dy),1e-8)
    local=(abs(dx)*np.where(dx>=0,widths[0],widths[1])+abs(dy)*np.where(dy>=0,widths[2],widths[3]))/den
    observed = (a[...,3]>0)&(distance<=local)
    if allow_partial:
        if len(measurements) < 2:
            raise ValueError('Fewer than two rear metal edges have usable source evidence')
        # Avoid turning an unmeasured fourth edge into an inferred corner
        # material. Only treat the near-normal part of each supported edge.
        x_supported = np.where(dx>=0,widths[0]>0,widths[1]>0)
        y_supported = np.where(dy>=0,widths[2]>0,widths[3]>0)
        observed &= ((x_supported & (abs(dx)/den>=.92)) |
                     (y_supported & (abs(dy)/den>=.92)))
    return observed,dict(method='standard_glass_back_source_transition',
        edge_measurements=measurements,body_contour=body_proof),body


def _bottom_apertures(a, gray, remaining):
    """Locate recessed bottom ports, speaker holes and screws relative to local metal.

    A fixed global threshold cuts across dark anodized aluminum grain and splits
    smooth metal into dark patches when lighting lifts the surrounding frame.
    Estimate column-wise rail metal outside the central aperture band, then
    require compact solid geometry and local contrast against adjacent metal.
    """
    h, w = gray.shape
    vis = remaining & (a[..., 3] >= 128)
    rail = vis.copy()
    rail[:int(h * .14), :] = False
    rail[int(h * .86):, :] = False
    rail[int(h * .26):int(h * .74), :] = False
    mid_rail = rail.copy()
    mid_rail[:, :int(w * .15)] = False
    mid_rail[:, int(w * .85):] = False
    if mid_rail.sum() < 40:
        return np.zeros((h, w), bool)
    metal_med = float(np.median(gray[mid_rail]))
    min_col = max(2, int(h * .05))
    col_metal = np.full(w, metal_med, dtype=float)
    for x in range(w):
        vals = gray[rail[:, x], x]
        if len(vals) >= min_col:
            col_metal[x] = float(np.median(vals))
    smooth_metal = ndimage.median_filter(
        col_metal, size=max(15, int(w * .21)) | 1, mode='nearest')
    threshold = np.minimum(145., smooth_metal * .72)
    dark = (gray.astype(float) < threshold[None, :]) & remaining
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark.astype('uint8'), 8)
    holes = np.zeros((h, w), bool)
    pad = 8
    for i in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        cx, cy = x + bw / 2, y + bh / 2
        if not (.12 * w < cx < .88 * w and .25 * h < cy < .75 * h
                and bh >= h * .14 and bw >= h * .14
                and area > 25 and bw < w * .18 and .3 < bw / max(1, bh) < 5):
            continue
        y0, y1 = max(0, y - pad), min(h, y + bh + pad)
        x0, x1 = max(0, x - pad), min(w, x + bw + pad)
        raw = labels[y0:y1, x0:x1] == i
        comp = ndimage.binary_fill_holes(raw)
        if int(comp.sum()) / max(1, bw * bh) < .55:
            continue
        ring = (ndimage.binary_dilation(comp, iterations=7)
                & ~ndimage.binary_dilation(comp, iterations=2)
                & vis[y0:y1, x0:x1])
        ring_med = float(np.median(gray[y0:y1, x0:x1][ring])) if ring.sum() >= 10 else metal_med
        if ring_med > max(metal_med + 35., 210.) and bw < w * .08:
            continue
        if ring_med - float(np.median(gray[y0:y1, x0:x1][raw])) < max(20., ring_med * .22):
            continue
        holes[y0:y1, x0:x1] |= comp
    return holes


def job(source,face,folder,structure):
    from PIL import Image
    image=Image.open(source).convert('RGBA');a=np.array(image);h,w=a.shape[:2]
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    sticker,sticker_proof=paper(image,face);visible=a[...,3]>0
    domains=[];allowed=np.zeros((h,w),bool);regions=[]
    proof=dict(source_sha256=digest(source),paper=sticker_proof)
    camera_plan = None
    if face=='front':
        rim,rim_proof=front_metal(image);proof['rim']=rim_proof
        regions=[('main_surface',rim,[max(18,h*.012),2.],True)]
    elif face=='back':
        rim,rim_proof,body=glass_back_rim(image,sticker);proof['rim']=rim_proof
        body_path=save_mask(folder/'body-contour.png',body)
        regions=[('rear_perimeter',rim,[max(18,h*.012),2.],True)]
        from .camera_metal import dual_scope, measure
        try:
            camera_plan = measure(a, 'dual')
        except ValueError as error:
            proof['camera_metal_notice'] = str(error)
        else:
            if all(sum(lens['supported']) >= 210 for lens in camera_plan['lenses']):
                capsule = dual_scope((h,w), camera_plan) & (a[...,3]==255) & ~rim & ~sticker
                if capsule.sum() >= 100:
                    regions.append(('camera_capsule',capsule,[3.,3.],False))
                else:
                    proof['camera_metal_notice'] = '双摄凸台可用范围不足'
            else:
                proof['camera_metal_notice'] = '双摄金属框的原图边界证据不足'

    else:
        remaining=visible&~sticker
        if face in ('left','right'):
            lines=np.asarray(structure[face]['antenna_lines']);yy=np.arange(h)[:,None]
            proof['material_notes']=list(structure[face].get('material_notes',[]))
            for i in range(len(lines)//2):
                mask=remaining&(yy>=lines[2*i])&(yy<lines[2*i+1])
                regions.append((f'antenna_{i}',mask,[max(4,np.median(lines[2*i+1]-lines[2*i])*.3),max(4,w*.2)],False));remaining &= ~mask
            from .iphone17_structure import capsule_mask
            gray=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY)
            for index,box in enumerate(structure[face]['hardware_boxes']):
                try:
                    hw=capsule_mask(image,box,face)&remaining
                except ValueError:
                    proof['material_notes'].append('按键轮廓不清，按实拍边框整体调光')
                    continue
                cut,_=cv2.threshold(gray[hw].reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                dark=hw&(gray<=cut);face_mask=hw&~dark
                for suffix,mask in [('inset',dark),('face',face_mask)]:
                    regions.append((f'hardware_{index}_{suffix}',mask,[max(6,h*.008),max(2,w*.05)],False))
                remaining &= ~hw
        else:
            # Apertures exist only on the bottom rail and need local-contrast
            # darkness plus compact geometric shape support.
            gray=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY)
            if face=='bottom':
                holes=_bottom_apertures(a,gray,remaining)
                if holes.any():
                    # Isolated dark fringes lack a material reference. Keep them
                    # in the broad source-lightness field instead of deriving a
                    # separate illumination from scratches, print or partial rims.
                    try:
                        reference_box(a, holes & remaining)
                    except ValueError:
                        holes[:]=False
                if holes.any():
                    regions.append(('apertures',holes,[max(3,h*.05),max(3,w*.008)],False));remaining &= ~holes
            # A full-thickness antenna is a supported vertical stripe, not a
            # global lightness threshold applied to arbitrary materials.
            y0,y1=int(h*.2),int(h*.8)
            signal=np.median(gray[y0:y1].astype(float),axis=0)
            from scipy.signal import find_peaks
            g=np.gradient(ndimage.gaussian_filter1d(signal,.8));peaks,_=find_peaks(abs(g),height=2,distance=3)
            bands=[]
            witnesses = None
            if face == 'bottom':
                # The terminal break must appear across the rail height. Ink
                # and the paper edge near the port are not antenna evidence.
                witnesses = [np.gradient(ndimage.gaussian_filter1d(np.median(section, axis=0), .8))
                             for section in np.array_split(gray[y0:y1].astype(float), 3)]
            for lo in peaks:
                for hi in peaks:
                    if not (.004*w<hi-lo<.025*w and g[lo]<-2 and g[hi]>2 and .12*w<lo<.9*w):
                        continue
                    if witnesses is not None:
                        if not (hi < .30*w or lo > .70*w):
                            continue
                        if not all(v[lo-2:lo+3].min() < -1.5 and v[hi-2:hi+3].max() > 1.5 for v in witnesses):
                            continue
                    bands.append((min(-g[lo],g[hi]),int(lo),int(hi)))
            if bands:
                _,lo,hi=max(bands);stripe=remaining&(np.arange(w)[None,:]>=lo)&(np.arange(w)[None,:]<hi)
                regions.append(('antenna',stripe,[max(3,h*.1),max(2,(hi-lo)*.3)],False));remaining &= ~stripe
                proof['antenna_columns']=[lo,hi]
            else:
                proof['material_notes']=['天线边缘不清，按实拍边框整体调光']
        regions.append(('main_surface',remaining,[max(3,h*.018),max(3,w*.12)],False))
    for name,mask,sigma,contour in regions:
        mask &= visible&~sticker
        if (mask&(a[...,3]>=240)).sum()<20:raise ValueError('Insufficient material evidence: '+name)
        path=save_mask(folder/(name+'.png'),mask);ref=reference_box(a,mask);allowed|=mask
        d=dict(name=name,mask=path,reference_boxes=[ref],sigma=sigma,strength=[.72,0,0],feather=0,correction_smooth=1.5)
        if name=='camera_capsule':
            d['strength']=[0.,0.,0.]
        if contour:
            reliable=mask&(a[...,3]>=240)
            shoulder=float(ndimage.distance_transform_edt(~reliable)[mask&(a[...,3]>=128)].max())
            if shoulder>max(8,w*.045):raise ValueError('Standard rim lacks nearby opaque metal support')
            d.update(field='contour',reference_profile='contour_depth',contour_inset_depth=max(3.,shoulder+1))
            d['fallback_contour_source_inset_limit'] = max(8., w*.045)
            if face=='back':d['contour_alpha_mask']=body_path
        domains.append(d)
    sticker_path=save_mask(folder/'paper.png',sticker)
    required=save_mask(folder/'required-editable.png',allowed)
    view=dict(name=face,source=str(Path(source).resolve()),source_sha256=digest(source),
        protection_policy='none' if face=='bottom' else 'sticker_only',sticker_mask=sticker_path,protected_mask=sticker_path,
        required_editable_mask=required,reference_boxes=domains[0]['reference_boxes'],domains=domains)
    if face=='back':
        if 'camera_metal_notice' in proof:
            view['camera_metal_notice'] = proof['camera_metal_notice']
        else:
            view['camera_metal_layout'] = 'dual'
    return dict(color_mode='source_lightness',defaults=dict(max_lab_shift=[35,0,0],feather=0),views=[view]),proof,allowed,sticker
