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
    if np.any(np.diff(result,axis=0)<=0):raise ValueError('Per-column antenna boundaries cross')
    return result


def gradient_peak(gradient, target, sign, radius=4):
    """Subpixel extremum; the target only bounds the independent RGB search."""
    start=max(1,int(target)-radius);end=min(len(gradient)-1,int(target)+radius+1)
    response=gradient*sign;k=start+int(np.argmax(response[start:end]))
    if k in (start,end-1) or response[k]<.5:
        raise ValueError('Antenna boundary is ambiguous in the independent RGB search')
    a,b,c=response[k-1:k+2];den=a-2*b+c
    return float(k+(.5*(a-c)/den if abs(den)>1e-9 else 0.))


def rear_antennas(image, *, refine_bands=('upper','lower')):
    """Measure bright plastic breaks in the narrow dark rear metal rim.

    The glass has no antenna; the outer rim does. Follow the current alpha
    contour and require agreement across three independent inward tracks.
    Rear-view image left is the phone's right (power/camera-control) side.
    """
    a=np.asarray(image.convert('RGBA'));h,w=a.shape[:2]
    gray=a[...,:3].mean(2);scale=w/988
    result={}
    for edge_name,side in [('left','right'),('right','left')]:
        edge=np.argmax(a[...,3]>200,axis=1) if edge_name=='left' else w-1-np.argmax(a[:,::-1,3]>200,axis=1)
        edge=ndimage.gaussian_filter1d(ndimage.median_filter(edge.astype(float),max(3,int(h*.01)//2*2+1)),max(1,h*.0025))
        tracks=[]
        for offset in (1.,2.,3.):
            xs=edge+(offset*scale if edge_name=='left' else -offset*scale)
            sig=ndimage.map_coordinates(gray,[np.arange(h),xs],order=1,mode='nearest')
            tracks.append(ndimage.gaussian_filter1d(sig,max(.7,h/2058)))
        consensus=np.median(tracks,axis=0)
        rows=[];proof=[]
        for band,(lo,hi) in zip(('upper','lower'),[(int(h*.065),int(h*.16)),(int(h*.84),int(h*.94))]):
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
    if len(selected)!=expected:raise ValueError(f'{side}: expected {expected} independent complete capsule contours, found {len(selected)}')
    return selected


def capsule_mask(image,box):
    """Use the actual closed source outline, never the detection box as paint."""
    a=np.asarray(image.convert('RGBA'));gray=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY)
    h,w=gray.shape;x0,y0,x1,y1=box
    roi=np.zeros((h,w),np.uint8)
    roi[max(0,y0-2):min(h,y1+2),max(0,x0-2):min(w,x1+2)]=(gray[max(0,y0-2):min(h,y1+2),max(0,x0-2):min(w,x1+2)]<160).astype('uint8')
    contours,_=cv2.findContours(roi,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    candidates=[]
    for c in contours:
        x,y,bw,bh=cv2.boundingRect(c)
        if max(abs(np.asarray([x,y,x+bw,y+bh])-box))<=4 and cv2.contourArea(c)>.5*(x1-x0)*(y1-y0):
            candidates.append(c)
    if not candidates:raise ValueError('Detected capsule lacks a complete source contour')
    contour=max(candidates,key=cv2.contourArea);mask=np.zeros((h,w),np.uint8)
    cv2.drawContours(mask,[contour],-1,1,cv2.FILLED)
    return mask>0


def propose(files, *, refine_bands=('upper','lower')):
    back=Image.open(files['back']);result={}
    rear=rear_antennas(back,refine_bands=refine_bands)
    for side in ('left','right'):
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
    return result


def measure_side(image,target,signs):
    b=np.asarray(image.convert('RGBA'));h,w=b.shape[:2];gray=b[...,:3].mean(2);checks={}
    for name,lo,hi in [('left',.08,.23),('center',.4,.6),('right',.77,.92)]:
        sig=np.median(gray[:,int(w*lo):int(w*hi)],axis=1)
        g=np.gradient(ndimage.gaussian_filter1d(sig,1.))
        measured=[gradient_peak(g,y,sign) for y,sign in zip(target,signs)]
        checks[name]=dict(measured=measured,max_error_px=float(np.max(abs(np.asarray(measured)-target))))
    return checks


def apply(image,cfg, *, refine_indices=(0,1,2,3)):
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
    source=np.asarray(cfg['source']);original_source=source.copy();dest=np.asarray(cfg['target'])
    # Re-render from the unchanged lighting baseline each iteration. Small
    # interpolation-phase corrections do not redefine the frozen rear target.
    anchors=[1,2,5,6];positions=np.array([.155,.5,.845])*w
    factor=cfg['back_size'][1]/h
    for attempt in range(5):
        out=warp_rows(image,source,dest)
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
        opaque_body_lost=0,buttons_changed=0,endcaps_changed=0,
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
