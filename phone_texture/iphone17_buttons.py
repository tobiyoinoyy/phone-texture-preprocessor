"""Source-derived rear-rim button alignment, independent of antenna targets."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.interpolate import PchipInterpolator
from .iphone17_structure import hardware,gradient_peak
from .geometry_review import digest
from texture_alignment.align_textures import warp_rows


def _edge_track(a,edge,level):
    alpha=a[...,3].astype(float)/255
    if edge=='right':alpha=alpha[:,::-1]
    h=len(alpha);idx=np.argmax(alpha>=level,axis=1)
    before=alpha[np.arange(h),np.maximum(0,idx-1)];after=alpha[np.arange(h),idx]
    return -(idx-1+(level-before)/np.maximum(.001,after-before))


def _rear_buttons(image,side,rail_window=0, optical_tracks=None):
    """Locate independent protrusions; no side-photo positions used as targets.

    The contour shoulder is measured at 25% of the protrusion's fitted depth,
    above the stable body rail. Three alpha isocontours must agree. The RGB
    signal in the exposed metal must also differ from the neighbouring rail.
    Flush camera control and SIM are deliberately outside this detector.
    """
    a=np.asarray(image.convert('RGBA'));h,w=a.shape[:2]
    edge={'left':'right','right':'left'}[side];expected=3 if side=='left' else 1
    lo,hi=round(h*.146),round(h*.559);scale=w/988
    all_tracks=[];profiles=[]
    for index,level in enumerate((.35,.5,.65)):
        raw = _edge_track(a,edge,level) if optical_tracks is None else optical_tracks[index]
        track=ndimage.gaussian_filter1d(raw,1.)
        ys=np.arange(lo,hi);coef=np.polyfit(ys,track[ys],2)
        for _ in range(15):
            delta=track[ys]-np.polyval(coef,ys)
            coef=np.polyfit(ys,track[ys],2,w=np.where(delta>0,.1,1.))
        base=np.polyval(coef,np.arange(h))
        if rail_window:
            # Remove long-wave silhouette curvature before finding protrusions.
            # The final shoulders still use the bare rail on each side.
            base = ndimage.grey_opening(track, size=round(h*rail_window))
        res=track-base
        mask=res>scale;mask[:lo]=False;mask[hi:]=False
        labs,n=ndimage.label(mask);found=[]
        for label in range(1,n+1):
            iy=np.flatnonzero(labs==label)
            if len(iy)<h*.025 or len(iy)>h*.15 or iy[0]<=lo or iy[-1]>=hi-1:continue
            start,end=int(iy[0]),int(iy[-1])
            # Refit the immediate bare rail on both sides. A global rail fit
            # drifts near the third button and mistakes its soft tail for metal.
            bare=np.r_[np.arange(start-25,start-12),np.arange(end+12,end+25)]
            local_base=np.polyval(np.polyfit(bare,track[bare],1),np.arange(h))
            local_res=track-local_base
            peak=float(np.percentile(local_res[start:end+1],85))
            if peak<2*scale:raise ValueError('Rear button protrusion is too weak')
            threshold=max(.75*scale,.25*peak)
            region=np.arange(max(lo,start-12),min(hi,end+13));inds=region[local_res[region]>threshold]
            # Contiguous component containing the independently found maximum.
            local_labels,_=ndimage.label(local_res[region]>threshold)
            label_at_peak=local_labels[np.argmax(local_res[region])]
            inds=region[local_labels==label_at_peak]
            s,e=int(inds[0]),int(inds[-1])
            if s<=region[0] or e>=region[-1]:raise ValueError('Rear button endpoint lacks a complete transition')
            s0=s-1+(threshold-local_res[s-1])/(local_res[s]-local_res[s-1])
            e0=e+(local_res[e]-threshold)/(local_res[e]-local_res[e+1])
            found.append([float(s0),float(e0)])
        if len(found)!=expected:raise ValueError(f'{side}: expected {expected} rear button protrusions, got {len(found)}')
        all_tracks.append(found);profiles.append((base,res))
    readings=np.asarray(all_tracks)
    if np.max(np.ptp(readings,axis=0))>2.5:
        raise ValueError(f'Rear button boundaries disagree across independent alpha contours: {readings.tolist()}')
    rows=np.median(readings,axis=0);base,res=profiles[1];proof=[]
    gray=a[...,:3].mean(2)
    # RGB differences across the metal shoulder (not alpha-only acceptance).
    body=-base if edge=='left' else w-1+base
    for bounds in rows:
        start,end=map(int,bounds);mid=np.arange(start+10,end-10)
        inward=body+(1.5*scale if edge=='left' else -1.5*scale)
        outward=body+(-1.5*scale if edge=='left' else 1.5*scale)
        inside=ndimage.map_coordinates(gray,[np.arange(h),inward],order=1)
        outside=ndimage.map_coordinates(gray,[np.arange(h),outward],order=1)
        contrast=float(np.percentile(abs(inside[mid]-outside[mid]),75))
        if contrast<8:raise ValueError('Rear button lacks independent metal RGB support')
        proof.append(dict(rows=bounds.tolist(),rgb_shoulder_contrast=contrast))
    return dict(rear_image_edge=edge,rows=rows.tolist(),alpha_levels=[.35,.5,.65],
                independent_contour_rows=readings.tolist(),checks=proof,
                definition='25_percent_protrusion_depth_shoulder_above_current_body_rail')


def rear_buttons(image,side):
    try:
        return _rear_buttons(image,side)
    except ValueError as original:
        # Require agreement between two independent rail lengths, three alpha
        # contours and each button's own RGB shoulder before accepting a fit.
        try:
            a = _rear_buttons(image,side,.105)
            b = _rear_buttons(image,side,.13)
        except ValueError:
            raise original
        if np.max(abs(np.array(a['rows'])-b['rows']))>1.:
            raise original
        return dict(a, rail_method='multi_scale_opening_with_local_bare_rail_refit',
                    alternate_rail_rows=b['rows'])


def button_edges(image,boxes):
    a=np.asarray(image.convert('RGBA'));g=a[...,:3].mean(2);rows=[]
    for x0,y0,x1,y1 in boxes:
        cx=(x0+x1)/2;half=max(2,(x1-x0)*.1)
        sig=np.median(g[:,int(cx-half):int(cx+half)+1],axis=1)
        grad=np.gradient(ndimage.gaussian_filter1d(sig,.7))
        rows.append([gradient_peak(grad,y0,-1,8),gradient_peak(grad,y1-1,1,8)])
    return np.asarray(rows)


def source_bound_buttons(back_path, back, side, geometry_report):
    # Tight transparency trimming can leave an antialiased button touching
    # the first column. Its outer interpolation sample belongs to the wider
    # pre-crop photograph, not to that first pixel a second time.
    column = 0 if side == 'right' else back.width-1
    y0, y1 = round(.146*back.height)-5, round(.559*back.height)+5
    border = np.asarray(back.getchannel('A'))[y0:y1,column]
    if not np.any(border >= .35*255):
        try:
            return rear_buttons(back,side)
        except ValueError:
            if geometry_report is None:
                raise
    if geometry_report is None or not Path(geometry_report).is_file():
        raise ValueError('Rear button reaches crop boundary; pre-crop evidence required')
    from .workspace import read
    back_digest = digest(back_path)
    matches = [r for r in read(geometry_report).get('images', []) if r['face']=='back'
               and r['output_sha256']==back_digest]
    if len(matches)!=1:
        raise ValueError('Missing source-bound rear geometry for button measurement')
    record = matches[0]
    captures = [a for a in record.get('geometry_evidence', [])
                if Path(a['path']).name=='after-rotation-canvas.png']
    if (len(captures)!=1 or not Path(captures[0]['path']).is_file() or
            digest(captures[0]['path'])!=captures[0]['sha256']):
        raise ValueError('Pre-crop rear evidence is missing or has changed')
    geometry = record.get('geometry', {})
    crop = geometry.get('canvas_trim_xyxy', [])
    trim = geometry.get('visible_canvas_trim', {}).get('crop_xyxy', [])
    if len(crop)!=4 or len(trim)!=4:
        raise ValueError('Pre-crop rear geometry lacks its crop coordinates')
    ox,oy = crop[:2]
    left,top,right,bottom = trim
    rect = (ox+left,oy+top,ox+right,oy+bottom)
    with Image.open(captures[0]['path']) as canvas:
        if (rect[0]<3 or rect[2]+3>canvas.width or rect[1]<0 or rect[3]>canvas.height or
                not np.array_equal(np.asarray(canvas.crop(rect)),np.asarray(back))):
            raise ValueError('Pre-crop rear pixels disagree with the selected button reference')
        extended = canvas.crop((rect[0]-3,rect[1],rect[2]+3,rect[3]))
    if np.any(np.asarray(extended.getchannel('A'))[y0:y1,0 if side=='right' else -1]>=.35*255):
        raise ValueError('Button outline still extends beyond the observed rear edge')
    try:
        measured = rear_buttons(extended,side)
    except ValueError:
        return photographic_buttons(record, rect, captures[0], side)
    # Retain previously stable subpixel anchors only after agreement with the
    # actual neighbouring pixels; otherwise use the three measured isocontours.
    try:
        previous = rear_buttons(back,side)
    except ValueError:
        previous = None
    if previous and np.max(abs(np.array(previous['rows'])-measured['rows']))<=.1:
        result = previous
    else:
        result = measured
    return dict(result, pre_crop_contour_rows=measured['rows'], pre_crop_sha256=captures[0]['sha256'])



def photographic_buttons(record, rect, capture, side):
    """Measure the outer shoulder directly against the photographed background."""
    from PIL import ImageOps
    source = Path(record['source'])
    if digest(source) != record['source_sha256']:
        raise ValueError('Rear photograph changed since geometry measurement')
    with Image.open(source) as original:
        photo = ImageOps.exif_transpose(original).convert('RGB')
    angle = record['geometry']['rotation_ccw_degrees']
    if abs(angle) >= 1e-5:
        photo = photo.rotate(angle, Image.Resampling.BICUBIC, expand=True)
    left, top, right, bottom = rect
    extended = (left-14, top, right+14, bottom)
    with Image.open(capture['path']) as canvas:
        if photo.size != canvas.size or left < 14 or right+14 > canvas.width:
            raise ValueError('Rear photograph lacks the required background margin')
        matte = canvas.crop(extended)
    edge = 'left' if side == 'right' else 'right'
    rgba = np.asarray(matte)
    rgb = np.asarray(photo.crop(extended))
    if edge == 'right':
        rgb = rgb[:,::-1]
    h,w = rgba.shape[:2]
    centers = np.clip(np.rint(-_edge_track(rgba,edge,.5)).astype(int),12,w-14)
    offsets = np.arange(-12,14)
    rows = np.arange(h)
    values = rgb[rows[:,None],centers[:,None]+offsets].astype(float)
    values = ndimage.gaussian_filter1d(values,1.5,axis=0)
    exterior = np.median(values[:,:5],axis=1)
    delta = abs(values-exterior[:,None,:])
    results = []
    for threshold in (8,12,18,24,30):
        tracks = []
        for channel in range(3):
            field = delta[:,:,channel]
            crossed = field > threshold
            crossed[:,:5] = False
            k = crossed.argmax(axis=1)
            before = field[rows,np.maximum(0,k-1)]
            after = field[rows,k]
            fraction = np.clip((threshold-before)/np.maximum(.1,after-before),0,1)
            track = -(centers+offsets[k]-1+fraction)
            tracks.append(track)
        try:
            observed = _rear_buttons(matte,side,optical_tracks=tracks)
        except ValueError:
            results.append(None)
            continue
        # Every shoulder must have observed background and an actual first
        # material crossing in all channels immediately on either side.
        valid = (delta[:,5:].max(axis=1)>threshold).all(axis=1)
        quiet = (delta[:,:5].max(axis=(1,2))<threshold)
        if any(not (valid & quiet)[round(y)-4:round(y)+5].all()
               for bounds in observed['rows'] for y in bounds):
            results.append(None)
            continue
        results.append(observed)
    stable = [(a,b) for a,b in zip(results,results[1:]) if a and b and
              np.max(abs(np.array(a['rows'])-b['rows'])) <= 1.]
    if not stable:
        raise ValueError('Rear button shoulders lack repeatable photographic RGB edges')
    members = [r for pair in stable for r in pair]
    if np.max(np.ptp([r['rows'] for r in members],axis=0)) > 1.5:
        raise ValueError('Competing photographic button shoulders')
    measured = dict(members[0])
    measured.pop('alpha_levels')
    measured['rgb_channel_rows'] = measured.pop('independent_contour_rows')
    measured.update(definition='25_percent_photographed_protrusion_above_bare_rail',
                    photographic_threshold_rows={str(t):r['rows'] for t,r in
                        zip((8,12,18,24,30),results) if r is not None},
                    raw_photo_sha256=record['source_sha256'], pre_crop_sha256=capture['sha256'])
    return measured



def _button_face_plane(image,boxes,sources,targets):
    """Recover the shared scale from the button faces inside their dark grooves."""
    h=image.height
    gray=np.asarray(image.convert('RGB'),dtype=float).mean(axis=2)
    gradient=np.gradient(ndimage.gaussian_filter1d(gray,.75,axis=0),axis=0)
    inner=[]
    for (x0,_,x1,_),bounds in zip(boxes,sources):
        center=(x0+x1)/2
        band=max(1,round((x1-x0)*.05))
        pairs=[]
        for offset in (-band,0,band):
            col=round(center)+offset
            signal=np.median(gradient[:,col-1:col+2],axis=1)
            rows=[]
            for outer,sign in zip(bounds,(1,-1)):
                search=np.rint(outer+sign*np.arange(1,max(7,round((x1-x0)*.27)))).astype(int)
                k=np.argmax(sign*signal[search])
                if k==0 or k==len(search)-1 or sign*signal[search[k]]<5:
                    raise ValueError('Button face lacks a second distinct cap edge')
                rows.append(gradient_peak(signal,search[k],sign,3))
            pairs.append(rows)
        if np.max(np.ptp(pairs,axis=0))>2.5:
            raise ValueError('Button face contours disagree across the cap')
        inner.append(np.median(pairs,axis=0))
    inner=np.asarray(inner)
    residuals=[]
    for held in range(3):
        train=np.arange(3)!=held
        coeff=np.polyfit(inner[train].ravel(),targets[train].ravel(),1)
        residuals.extend(np.polyval(coeff,inner[held])-targets[held])
    if np.max(abs(np.asarray(residuals)))>h*.003:
        raise ValueError('Held-out button face rejects common rail scale')
    q=np.r_[np.polyfit(inner.ravel()/h,targets.ravel()/h,1),0.]
    fitted=np.polyval(q[:2],inner.ravel()/h)*h
    if np.max(abs(fitted-targets.ravel()))>h*.003 or not .75<q[0]<1.25:
        raise ValueError('Button face scale disagrees with the rear shoulders')
    return q,fitted,dict(inner_rows=inner.tolist(),held_out_error_px=residuals)

def propose(back_path,side_path,side,geometry_report=None):
    back=Image.open(back_path);im=Image.open(side_path);h,w=im.height,im.width
    boxes=hardware(im,side);count=3 if side=='left' else 1
    refs=source_bound_buttons(back_path,back,side,geometry_report);targets=(np.asarray(refs['rows'])+.5)/back.height*h-.5
    sources=button_edges(im,boxes[:count])
    if np.max(abs(targets-sources))>h*.06:raise ValueError('Button correction exceeds independently supported correspondence')
    ratios=np.diff(targets,axis=1)/np.diff(sources,axis=1)
    if np.any((ratios<.75)|(ratios>1.25)):raise ValueError('Button length change is inconsistent with rear protrusion')
    union_lo=np.minimum(sources[:,0],targets[:,0]);union_hi=np.maximum(sources[:,1],targets[:,1])
    pad=max(24,int(h*.025));boundaries=[union_lo[0]-pad]
    for i in range(count-1):boundaries.append((union_hi[i]+union_lo[i+1])/2)
    boundaries.append(union_hi[-1]+pad)
    locked_boxes=boxes[count:]
    if locked_boxes and boundaries[-1]>=min(b[1] for b in locked_boxes)-8:
        raise ValueError('Button correction would reach the camera control or SIM tray')
    src=[];dst=[]
    for i in range(count):
        if i==0:src.append(boundaries[i]);dst.append(boundaries[i])
        src.extend(sources[i]);dst.extend(targets[i])
        src.append(boundaries[i+1]);dst.append(boundaries[i+1])
    if min(np.diff(src))<3 or min(np.diff(dst))<3:raise ValueError('Button correction leaves no safe metal gap')
    sn=sources.ravel()/h;tn=targets.ravel()/h
    if count>=2:
        q=np.linalg.lstsq(np.c_[sn,np.ones(len(sn)),-tn*sn],tn,rcond=None)[0]
        mode='shared_projective_with_small_smooth_landmark_residual'
    else:
        q=np.r_[np.polyfit(sn,tn,1),0.]
        mode='affine_projective_limit_two_reliable_endpoints'
    fitted=(q[0]*sn+q[1])/(q[2]*sn+1)*h
    cap_proof={}
    if np.max(abs(fitted-targets.ravel()))>h*.003 or abs(q[2])>.5:
        if count!=3:
            raise ValueError('Button references do not support a stable shared perspective model')
        q,fitted,evidence=_button_face_plane(im,boxes[:count],sources,targets)
        cap_proof['face_plane']=evidence
        mode='shared_affine_from_three_independently_measured_button_faces'
    return dict(**cap_proof,side=side,perspective_coefficients=q.tolist(),mapping_mode=mode,
                shared_projective_residual_px=(fitted-targets.ravel()).tolist(),back_sha256=digest(back_path),source_sha256=digest(side_path),back_size=list(back.size),
                rear_detection=refs,source_boxes=boxes,source_rows=sources.tolist(),target_rows=targets.tolist(),
                source_anchors=src,target_anchors=dst,allowed_rows=[int(np.floor(boundaries[0])),int(np.ceil(boundaries[-1]))+1],
                locked_hardware_boxes=locked_boxes,scope='three_left_buttons_or_right_power_only')


def apply(image,cfg):
    a=np.asarray(image.convert('RGBA'));h,w=a.shape[:2]
    expected_edge={'left':'right','right':'left'}[cfg['side']]
    if cfg['rear_detection']['rear_image_edge']!=expected_edge:raise ValueError('Wrong rear edge for button targets')
    targets=(np.asarray(cfg['rear_detection']['rows'])+.5)/cfg['back_size'][1]*h-.5
    if not np.allclose(targets,cfg['target_rows']):raise ValueError('Buttons are not referenced to current rear image')
    q=np.asarray(cfg['perspective_coefficients']);src=np.asarray(cfg['source_rows']).ravel()
    target_flat=targets.ravel();lo,hi=cfg['allowed_rows']
    moved=cfg['source_boxes'][:len(targets)]
    x0=min(box[0] for box in moved);x1=max(box[2] for box in moved)
    xweight=np.interp(np.arange(w),[0,max(1,x0-16),x0-5,x1+5,min(w-2,x1+16),w-1],[0,0,1,1,0,0])
    yy,xx=np.mgrid[:h,:w].astype(float);yn=np.arange(h)/h
    inv=(q[1]-yn)/(q[2]*yn-q[0])*h
    inv_at=(q[1]-target_flat/h)/(q[2]*target_flat/h-q[0])*h
    landmark_residual=src-inv_at;correction=np.zeros_like(src)
    # Smoothly attach the local perspective plane to the locked body features.
    weight_y=np.interp(np.arange(h),[lo,target_flat[0]-8,target_flat[-1]+8,hi-1],[0,1,1,0])
    weight_y=weight_y*weight_y*(3-2*weight_y)
    influence=weight_y[:,None]*xweight[None,:]
    for attempt in range(5):
        residual=PchipInterpolator(target_flat,landmark_residual+correction,extrapolate=True)(np.clip(np.arange(h),target_flat[0],target_flat[-1]))
        sy=yy+influence*((inv+residual)[:,None]-yy)
        ref=float(np.mean(src));cx=float(np.mean([x0,x1]))
        perspective_x=(q[2]*sy/h+1)/(q[2]*ref/h+1)
        sx=xx+influence*(xx-cx)*(perspective_x-1)
        derivative=np.diff(sy,axis=0)
        if derivative.min()<.15 or derivative.max()>3:raise ValueError(f'Perspective transition folds or excessively stretches the source: {derivative.min()}, {derivative.max()}')
        out=_sample_xy(image,sx,sy);b=np.asarray(out)
        actual=hardware(out,cfg['side'])[:len(targets)]
        measured=button_edges(out,actual)
        error=(measured-targets)*cfg['back_size'][1]/h
        lengths=error[:,1]-error[:,0]
        if np.max(abs(error))<=1.5 and np.max(abs(lengths))<=2:break
        if attempt==4:raise ValueError('Final perspective button contour fails rear endpoints/length')
        correction+=(measured-targets).ravel()
        if np.max(abs(correction))>3:raise ValueError('Button interpolation correction exceeds 3 source pixels')
    if not np.array_equal(a[...,3],b[...,3]):
        out = _retain_interior_coverage(a, b, xx, yy, sx, sy)
        b = np.asarray(out)
        # Do not carry a pre-correction hardware measurement to the output.
        # The final hardware and endpoint checks below read this exact RGBA.

    if not np.array_equal(a[:lo],b[:lo]) or not np.array_equal(a[hi:],b[hi:]):raise ValueError('Button warp changed a locked region')
    for box in cfg['locked_hardware_boxes']:
        x0,y0,x1,y1=box
        if not np.array_equal(a[y0:y1,x0:x1],b[y0:y1,x0:x1]):raise ValueError('Camera control or SIM changed')
    if np.any((a[...,3]>=240)&(b[...,3]<128)):raise ValueError('Button alignment lost opaque body pixels')
    actual=hardware(out,cfg['side'])[:len(targets)]
    measured=button_edges(out,actual);error=(measured-targets)*cfg['back_size'][1]/h
    centers=error.mean(1);lengths=error[:,1]-error[:,0]
    if np.max(abs(error))>1.5 or np.max(abs(lengths))>2:
        raise ValueError(f'Final complete button contour does not match rear endpoints/length: {measured.tolist()} vs {targets.tolist()} errors {error.tolist()}')
    return out,dict(side=cfg['side'],rear_image_edge=expected_edge,reference='current_back_protrusions',
                   target_rows=targets.tolist(),final_boxes=actual,endpoint_error_rear_px=error.tolist(),
                   center_error_rear_px=centers.tolist(),length_error_rear_px=lengths.tolist(),
                   unchanged_rows=[[0,lo],[hi,h]],locked_hardware_unchanged=True,opaque_body_lost=0,
                   model_uv_verified=False,render_iterations=attempt+1,sampling_correction=correction.tolist(),alpha_changed=0,
                   **({'face_plane':cfg['face_plane']} if 'face_plane' in cfg else {}),
                   mapping_mode=cfg['mapping_mode'],perspective_coefficients=q.tolist(),
                   shared_projective_residual_px=cfg['shared_projective_residual_px'],
                   source_jacobian_range=[float(derivative.min()),float(derivative.max())])


def _retain_interior_coverage(source, rendered, xx, yy, sx, sy):
    # Some segmentation mattes vary between 250 and 255 in the solid body.
    # Button alignment moves colour over the intact metal; coverage belongs
    # to the original canvas. Require an opaque margin around every source
    # interpolation footprint and every moved destination before fixing alpha.
    moved = (abs(sx-xx)>1e-8) | (abs(sy-yy)>1e-8)
    core = ndimage.binary_erosion(source[...,3]>=250,iterations=3)
    if not core[moved].all():
        raise ValueError('Button correction changed the body alpha contour')
    x,y = sx[moved],sy[moved]
    for ix in (np.floor(x).astype(int), np.ceil(x).astype(int)):
        for iy in (np.floor(y).astype(int), np.ceil(y).astype(int)):
            if not core[iy,ix].all():
                raise ValueError('Button correction would sample the body alpha contour')
    result = rendered.copy()
    result[...,3] = source[...,3]
    return Image.fromarray(result)


def _sample_xy(image,x,y):
    """Premultiplied bilinear source sampling, with identity pixels byte-locked."""
    a=np.asarray(image.convert('RGBA'));h,w=a.shape[:2]
    if x.min() < -1e-6 or x.max()>w-1+1e-6 or y.min() < -1e-6 or y.max()>h-1+1e-6:
        raise ValueError('Perspective alignment requests unobserved pixels')
    xf=np.floor(x).astype(int);yf=np.floor(y).astype(int)
    x1=np.minimum(xf+1,w-1);y1=np.minimum(yf+1,h-1)
    tx=(x-xf)[...,None];ty=(y-yf)[...,None]
    pre=a.astype(float);pre[...,:3]*=pre[...,3:]/255
    v=(pre[yf,xf]*(1-tx)+pre[yf,x1]*tx)*(1-ty)+(pre[y1,xf]*(1-tx)+pre[y1,x1]*tx)*ty
    v[...,:3]=np.divide(v[...,:3]*255,v[...,3:],out=np.zeros_like(v[...,:3]),where=v[...,3:]>0)
    result=np.clip(np.rint(v),0,255).astype('uint8')
    exact=(abs(x-np.rint(x))<1e-9)&(abs(y-np.rint(y))<1e-9)
    result[exact]=a[np.rint(y[exact]).astype(int),np.rint(x[exact]).astype(int)]
    return Image.fromarray(result)


def validate(check):
    count=3 if check['side']=='left' else 1
    errors=np.asarray(check.get('endpoint_error_rear_px',[]));lengths=np.asarray(check.get('length_error_rear_px',[]))
    expected_edge={'left':'right','right':'left'}[check['side']]
    if check.get('rear_image_edge')!=expected_edge or errors.shape!=(count,2) or lengths.shape!=(count,):
        raise ValueError('Every physical button needs its own rear reference and complete edge checks')
    if not np.isfinite(errors).all() or np.max(abs(errors))>1.5 or np.max(abs(lengths))>2:
        raise ValueError('Button endpoint/length check failed')
    if check.get('alpha_changed')!=0 or check.get('locked_hardware_unchanged') is not True:
        raise ValueError('Button correction changed locked geometry or hardware')
    if check.get('mapping_mode')=='shared_affine_from_three_independently_measured_button_faces':
        evidence=check.get('face_plane',{})
        if (count!=3 or np.shape(evidence.get('inner_rows'))!=(3,2)
                or np.shape(evidence.get('held_out_error_px'))!=(6,)):
            raise ValueError('Missing independent three-button face measurements')
    elif check.get('mapping_mode') not in ('shared_projective_with_small_smooth_landmark_residual','affine_projective_limit_two_reliable_endpoints'):
        raise ValueError('A source-supported perspective/scaling model is required')


def publish_revision(work,review_path,output):
    """Recompute the source-bound partial revision before releasing six PNGs."""
    import shutil
    from . import workspace as ws,pipeline
    work,output=Path(work),Path(output);r=ws.read(work/'revision.json');review=ws.read(review_path)
    if review.get('report_sha256')!=ws.sha(work/'revision.json') or not review.get('reviewer'):
        raise ValueError('Button review is missing or stale')
    if review.get('checked_buttons')!=['left_1','left_2','left_3','right_1']:
        raise ValueError('All four moved buttons require native/4x visual review')
    for ev in review.get('evidence',[]):
        if ws.sha(Path(ev['path']))!=ev['sha256']:raise ValueError('Button review evidence changed')
    if len(review.get('evidence',[]))<5:raise ValueError('Button review needs native and individual closeups')
    back=Path(r['reference_back']['path'])
    if ws.sha(back)!=r['reference_back']['sha256']:raise ValueError('Rear reference changed')
    candidate=Path(r['candidate'])
    for face,item in r['baseline'].items():
        source=Path(item['path']);dest=candidate/(face+'.png')
        if ws.sha(source)!=item['sha256'] or ws.sha(dest)!=r['final_sha256'][face]:
            raise ValueError('Button baseline or final image changed')
        if face in ('left','right'):
            cfg=propose(back,source,face);expected,check=apply(Image.open(source),cfg);validate(check)
            if not np.array_equal(np.asarray(expected),np.asarray(Image.open(dest).convert('RGBA'))):
                raise ValueError('Button image differs from the reviewed deterministic perspective correction')
        elif ws.sha(source)!=ws.sha(dest):raise ValueError('Button revision changed another face')
    pipeline.fresh(output);output.mkdir(parents=True)
    for face in pipeline.FACES:shutil.copy2(candidate/(face+'.png'),output/(face+'.png'))
    ws.save(work/'delivery.json',dict(output=str(output),final_sha256=r['final_sha256'],model_uv_verified=False,
        review_sha256=ws.sha(Path(review_path)),method='source_pixel_perspective_button_revision'))
    return output
