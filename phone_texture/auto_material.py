"""Conservative source-derived proposals for the aluminum studio-photo profile.

Material classification, fit exclusion and pixel protection are separate.
Only detected real paper enters pixel protection. Proposals are auditable and
never derive a mask from the corrected image or read a historical SKU mask.
"""
import cv2
import numpy as np
from scipy import ndimage
from PIL import Image
from pathlib import Path
from .geometry_review import digest


def red_markers(a):
    h,w=a.shape[:2];hsv=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2HSV)
    red=(((hsv[...,0]<10)|(hsv[...,0]>170))&(hsv[...,1]>80)&(hsv[...,2]>80)&(a[...,3]>128)).astype(np.uint8)
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(red,8);result=[]
    for i in range(1,n):
        x,y,bw,bh,area=map(int,stats[i]);cx,cy=centroids[i]
        if 2<=bw<=h*.09 and 2<=bh<=h*.09 and 4<=area<=h*h*.006 and cy>h*.35 and .4<bw/max(1,bh)<2.5:
            result.append((float(cy),float(cx),bw,bh))
    return result


def seeded_paper(a,face):
    """Recover printed paper on pale metal using print seeds + local background.

    The printed QR/vertical text locates the current label. GrabCut classifies
    its actual source pixels against a row-wise background colour field; no
    stored label rectangle is used as the final protection mask.
    """
    h,w=a.shape[:2];rgb=a[...,:3];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    if face=='back':
        start=int(h*.55);ok,points=cv2.QRCodeDetector().detect(rgb[start:])
        x_offset=0
        if not ok:
            # QR finder on a bounded central crop avoids a pale camera/body
            # texture confusing the full-frame detector. It still measures
            # the current printed quadrilateral, without stored label boxes.
            x_offset=int(w*.2);start=int(h*.65)
            ok,points=cv2.QRCodeDetector().detect(gray[start:,x_offset:int(w*.8)])
        if not ok:return None,None
        q=points.reshape(-1,2);q[:,1]+=start;q[:,0]+=x_offset
        qx0,qy0=q.min(axis=0);qx1,qy1=q.max(axis=0);scale=max(qx1-qx0,qy1-qy0)
        if not .02*w<scale<.25*w:return None,None
        x0=max(0,int(qx0-.8*scale));x1=min(w,int(qx1+.8*scale))
        y0=max(int(h*.55),int(qy0-1.7*scale));y1=h
        seed_box=[int(qx0),int(qy0),int(qx1)+1,int(qy1)+1]
        prior=(int(qx0-.20*scale),int(qx1+.20*scale),int(qy1+.7*scale))
        cue='detected_QR_quadrilateral'
    elif face=='bottom':
        smoothed=ndimage.gaussian_filter1d(gray.astype(float),.6,axis=0)
        nearby_dark=ndimage.minimum_filter1d(gray,size=3,axis=0)<110
        magnitude=np.abs(np.gradient(smoothed,axis=0))
        vertical=(magnitude>12)&(magnitude>np.roll(magnitude,1,axis=0))&(magnitude>=np.roll(magnitude,-1,axis=0))&nearby_dark
        vertical[:int(h*.12)]=False;vertical[int(h*.88):]=False
        count=ndimage.uniform_filter1d(vertical.sum(axis=0).astype(float),5)
        count[:int(w*.18)]=0;count[int(w*.82):]=0
        peak=int(count.argmax())
        if count[peak]<max(5,h*.035):return None,None
        active=count>max(3,count[peak]*.60)
        active[:max(0,int(peak-h*.75))]=False;active[min(w,int(peak+h*.75)+1):]=False
        printed=np.flatnonzero(active)
        if not len(printed):return None,None
        lo,hi=int(printed.min()),int(printed.max())+1
        # Include the complete printed glyphs, not just their strongest
        # strokes. Reject large aperture/rim components before edge search.
        ink=(gray<120)&(a[...,3]>=128)
        n,components,stats,centroids=cv2.connectedComponentsWithStats(ink.astype(np.uint8),8)
        markers=[v for v in red_markers(a) if abs(v[1]-peak)<h*.65]
        marker=max(markers) if markers else None
        boxes=[];glyphs=[]
        for i in range(1,n):
            ix,iy,iw,ih,area=map(int,stats[i]);cx,cy=centroids[i]
            if lo-h*.18<cx<hi+h*.18 and iw<w*.10 and ih<h*.90 and 3<area<h*h*.10 and .05*h<cy<.95*h and active[ix:ix+iw].any():
                boxes.append([ix,ix+iw])
            circle=.7<iw/max(1,ih)<1.3 and area/max(1,iw*ih)>.55 and ih>h*.08
            in_label=(marker is None or (marker[1]-h*.65<cx<marker[1]+max(5,marker[2]*1.5) and cy<marker[0]-2))
            if abs(cx-peak)<h*.75 and 4<=ih<h*.30 and iw<w*.045 and 3<area<h*h*.05 and not circle and in_label:
                glyphs.append((cx,cy,ix,iy,iw,ih))
        supported_glyphs=[]
        for glyph in glyphs:
            group=[v for v in glyphs if abs(v[0]-glyph[0])<max(4,h*.06)]
            if len(group)>=2 and np.ptp([v[1] for v in group])>h*.12:
                supported_glyphs.extend(group)
        if supported_glyphs:
            boxes.extend([[v[2],v[2]+v[4]] for v in supported_glyphs])
        if boxes:lo=min(lo,min(b[0] for b in boxes));hi=max(hi,max(b[1] for b in boxes))
        if marker:
            lo=max(lo,int(marker[1]-h*.65));hi=min(hi,int(np.ceil(marker[1]+max(5,marker[2]*1.5))))
        if hi-lo>w*.18:raise ValueError('Printed paper columns are not separated from hardware')
        scale=max(hi-lo,w*.035)
        padding=max(.8*scale,h*.6)
        x0=max(0,int(lo-padding));x1=min(w,int(hi+padding));y0=0;y1=h
        seed_box=[lo,int(h*.18),hi,int(h*.82)];prior=(int(lo-h*.25),int(hi+h*.25),h)
        cue='source_vertical_print_density'
    else:return None,None
    roi=rgb[y0:y1,x0:x1];rh,rw=roi.shape[:2]
    if min(rh,rw)<10:return None,None
    margin=max(3,int(rw*.12))
    l=np.median(roi[:,:margin].astype(float),axis=1);r=np.median(roi[:,-margin:].astype(float),axis=1)
    t=np.linspace(0,1,rw)[None,:,None];background=l[:,None,:]*(1-t)+r[:,None,:]*t
    features=np.clip(128+(roi.astype(float)-background)*2,0,255).astype(np.uint8)
    labels=np.full((rh,rw),cv2.GC_PR_BGD,np.uint8)
    pl=max(margin,prior[0]-x0);pr=min(rw-margin,prior[1]-x0)
    labels[:,pl:pr]=cv2.GC_PR_FGD
    labels[:,:margin]=cv2.GC_BGD;labels[:,-margin:]=cv2.GC_BGD
    if face=='back':
        tail_center=int((seed_box[0]+seed_box[2])/2)-x0
        labels[max(0,prior[2]-y0):]=cv2.GC_PR_BGD
        labels[max(0,prior[2]-y0):,max(margin,int(tail_center-.38*scale)):min(rw-margin,int(tail_center+.38*scale))]=cv2.GC_PR_FGD
        labels[:max(1,int(scale*.08))]=cv2.GC_BGD
    sx0,sy0,sx1,sy1=seed_box;sx0-=x0;sx1-=x0;sy0-=y0;sy1-=y0
    # Print seeds are certainly paper; white margins within the print block
    # also train the foreground model so the result is not ink-only.
    labels[max(0,sy0):min(rh,sy1),max(margin,sx0):min(rw-margin,sx1)]=cv2.GC_FGD
    labels[np.array(a[y0:y1,x0:x1,3])==0]=cv2.GC_BGD
    cv2.setRNGSeed(0)
    cv2.grabCut(features,labels,None,np.zeros((1,65)),np.zeros((1,65)),5,cv2.GC_INIT_WITH_MASK)
    fg=(labels==cv2.GC_FGD)|(labels==cv2.GC_PR_FGD)
    n,components,stats,_=cv2.connectedComponentsWithStats(fg.astype(np.uint8),8)
    seed_region=components[max(0,sy0):min(rh,sy1),max(0,sx0):min(rw,sx1)]
    counts=np.bincount(seed_region.ravel(),minlength=n);counts[0]=0
    if not counts.max():return None,None
    fg=ndimage.binary_fill_holes(components==counts.argmax())
    result=np.zeros((h,w),bool);result[y0:y1,x0:x1]=fg;result &= a[...,3]>0
    strip_edges=None
    if face=='bottom':
        from phone_normalizer.frame_strip import _line
        valid_rows=np.arange(int(h*.12),int(h*.88))
        valid_rows=valid_rows[result[valid_rows].any(axis=1)]
        if len(valid_rows)<h*.40:raise ValueError('Printed paper has insufficient continuous source rows')
        gradient=np.gradient(ndimage.gaussian_filter(gray.astype(float),.7),axis=1)
        fitted_by_side={};common_slope=None
        edge_order=[(seed_box[0],-1),(seed_box[2],1)]
        if marker and marker[1]>w/2:edge_order.reverse()
        for core,direction in edge_order:
            font=max(h*.10,float(np.percentile([v[5] for v in supported_glyphs],85))) if supported_glyphs else h*.12
            quiet=max(4,font*(.55 if direction<0 else .40));center_prior=core+direction*quiet
            if marker:
                # This named QC capture profile has a printed red registration
                # dot. Its layout prior supplies a bounded search neighbourhood,
                # not a final mask: both edges are fitted to this source photo.
                label_width=h*.56
                center_prior=marker[1]+label_width*(.20 if direction>0 else -.80)
            radius=6
            xx=np.arange(max(1,int(center_prior)-radius),min(w-1,int(center_prior)+radius+1))
            if not len(xx):raise ValueError('Printed paper edge search is empty')
            best=None;middle=float(np.median(valid_rows))
            for slope in ([common_slope] if common_slope is not None else np.linspace(-.12,.12,25)):
                columns=np.rint(xx[None,:]+slope*(valid_rows[:,None]-middle)).astype(int)
                columns=np.clip(columns,1,w-2)
                values=-direction*gradient[valid_rows[:,None],columns]
                prior_weight=np.exp(-.5*((xx-center_prior)/3.)**2)
                score=(np.median(values,axis=0)+.4*np.percentile(values,25,axis=0))*prior_weight
                k=int(np.argmax(score));candidate=(float(score[k]),slope,float(xx[k]))
                if best is None or candidate[0]>best[0]:best=candidate
            _,slope,center=best
            predicted=center+slope*(valid_rows-middle)
            columns=np.rint(predicted[:,None]+np.arange(-2,3)).astype(int);columns=np.clip(columns,1,w-2)
            values=-direction*gradient[valid_rows[:,None],columns]
            measured=columns[np.arange(len(valid_rows)),np.argmax(values,axis=1)];good=values.max(axis=1)>.7
            if good.sum()<h*.3:raise ValueError('Printed paper side has weak source contrast')
            line,keep,error=_line(valid_rows[good]+.5,measured[good].astype(float))
            if common_slope is not None:
                intercept=float(np.median(measured[good]-common_slope*(valid_rows[good]+.5)))
                line=np.array([common_slope,intercept])
                residual=np.abs(np.polyval(line,valid_rows[good]+.5)-measured[good])
                keep=residual<=2.5
                error=float(np.percentile(residual[keep],95)) if keep.any() else float('inf')
            if error>2.5:
                from .auto_geometry import consensus_line
                try:line,keep,error=consensus_line(valid_rows[good]+.5,measured[good].astype(float),2.5,.15)
                except ValueError:pass
            if error>2.5 and keep.mean()>=.6:
                # A real paper strip can bow around the frame. Fit its measured
                # source curve instead of forcing a straight protection box.
                yy_fit=valid_rows[good][keep]+.5;xx_fit=measured[good][keep].astype(float)
                curve=np.polyfit(yy_fit,xx_fit,2)
                curve_error=float(np.percentile(np.abs(np.polyval(curve,yy_fit)-xx_fit),95))
                displacement=np.polyval(curve,np.array([0,h/2,h]))-np.polyval(line,np.array([0,h/2,h]))
                if curve_error<=2.5 and np.max(np.abs(displacement))<=5:
                    line,error=curve,curve_error
            if error>2.5 or keep.mean()<.6:
                raise ValueError(f'Printed paper side is not independently resolved: center={center:.1f}, residual={error:.2f}, support={keep.mean():.2f}')
            fitted_by_side[direction]=line
            if common_slope is None:common_slope=float(np.polyval(np.polyder(line),h/2))
        fitted=[fitted_by_side[-1],fitted_by_side[1]]
        yy=np.arange(h)[:,None];xx=np.arange(w)[None,:]
        l=np.polyval(fitted[0],yy+.5)-.6;r=np.polyval(fitted[1],yy+.5)+.6
        if np.any(r-l<5) or np.any(r-l>w*.18):raise ValueError('Printed paper width is inconsistent')
        result=(xx>=l)&(xx<=r)&(a[...,3]>0)
        strip_edges=[line.tolist() for line in fitted]
    proof=dict(method='printed_seed_local_background_grabcut_v1',cue=cue,
        seed_box=seed_box,analysis_roi=[x0,y0,x1,y1],pixels=int(result.sum()),source_strip_edges=strip_edges)
    if face=='bottom':result,proof=trim_strip_to_source_end(a,result,proof,font)
    return result,proof


def trim_strip_to_source_end(a,paper,proof,font=None):
    """Resolve the physical paper end; do not freeze the metal below a label."""
    h,w=paper.shape;font=float(font or h*.13)
    markers=[v for v in red_markers(a) if paper[min(h-1,int(v[0])),min(w-1,int(v[1]))]]
    if not markers:return paper,dict(proof,source_paper_end='unresolved_no_marker')
    cy,cx,bw,bh=max(markers)
    predicted=cy+h*.56*.25;radius=9
    if predicted>=h-2:
        return paper,dict(proof,source_paper_end='continues_to_face_canvas',red_marker=[cx,cy,bw,bh])
    ys=np.arange(max(int(cy+bh),int(predicted-radius)),min(h-2,int(predicted+radius)+1))
    if len(ys)<3:raise ValueError('Paper end falls outside the measurable frame')
    gray=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY).astype(float)
    gy=-np.gradient(ndimage.gaussian_filter1d(gray,.8,axis=0),axis=0)
    profile=[]
    for y in ys:
        xx=np.flatnonzero(paper[y])
        if len(xx)<10:profile.append(0.);continue
        cols=np.concatenate([xx[2:min(7,len(xx)//3)],xx[max(0,len(xx)-7):len(xx)-2]])
        profile.append(float(np.median(gy[y,cols])))
    weighted=np.array(profile)*np.exp(-.5*((ys-predicted)/5.)**2)
    peak=int(np.argmax(weighted));end=int(ys[peak])+1
    if peak in (0,len(ys)-1) or profile[peak]<.7:
        raise ValueError(f'Physical paper bottom edge lacks independent source support: predicted={predicted:.1f}, search={ys[0]}..{ys[-1]}, peak={ys[peak]}, strength={profile[peak]:.2f}')
    result=paper.copy();result[end+1:]=False
    return result,dict(proof,source_paper_end=end,red_marker=[cx,cy,bw,bh],source_end_gradient=float(profile[peak]))


def paper_mask(image,face):
    a=np.array(image.convert('RGBA'));h,w=a.shape[:2];rgb=a[...,:3]
    mask=np.zeros((h,w),bool)
    if face not in ('back','bottom'):return mask,dict(method='profile_no_label_face',components=[])
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    white=(hsv[...,1]<65)&(hsv[...,2]>155)&(rgb.min(axis=2)>120)&(a[...,3]>128)
    if face=='back':white[:int(h*.55)]=False
    if face=='bottom':
        # A white paper strip can touch a specular outer rail. Discover its
        # printed interior independently, rather than merging the whole rail.
        band=max(2,int(h*.10));white[:band]=False;white[h-band:]=False
    opaque=a[...,3]>=128
    widest=int(np.argmax(opaque.sum(axis=1)));body_x=np.flatnonzero(opaque[widest])
    if not len(body_x):raise ValueError('Paper detection requires visible phone foreground')
    body_left,body_right=int(body_x[0]),int(body_x[-1]);body_width=body_right-body_left
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(white.astype(np.uint8),8)
    selected=[]
    for i in range(1,n):
        x,y,bw,bh,area=map(int,stats[i]);cx,cy=centroids[i]
        if area<max(20,.00012*h*w) or not .18*w<cx<.82*w:continue
        if face=='back' and not body_left+.15*body_width<cx<body_right-.15*body_width:continue
        if face=='back' and not (.008*w<bw<.18*w and bh>.02*h):continue
        if face=='bottom' and not (.008*w<bw<.18*w and bh>.20*h):continue
        component=(labels==i).astype(np.uint8)
        contours,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        polygon=np.zeros_like(component);cv2.drawContours(polygon,contours,-1,1,cv2.FILLED)
        # Printed text/QR supplies a second cue; bright metal or a flash alone
        # cannot become a protected sticker.
        ink=(rgb.mean(axis=2)<110)&(polygon>0)
        if ink.sum()<max(8,polygon.sum()*.02):continue
        if face=='bottom':
            sy,sx=np.nonzero(component)
            # Fit the two actual paper sides from the white margins on source
            # rows. The signed print holes are filled, not painted or edited.
            unique=np.unique(sy);lefts=np.array([sx[sy==r].min() for r in unique]);rights=np.array([sx[sy==r].max() for r in unique])
            widths=rights-lefts+1;good=widths>=np.percentile(widths,65)
            if good.sum()<max(8,h*.2):continue
            lfit=np.polyfit(unique[good],lefts[good],1);rfit=np.polyfit(unique[good],rights[good],1)
            yy=np.arange(h)[:,None];xx=np.arange(w)[None,:]
            polygon=((xx>=np.polyval(lfit,yy)-.5)&(xx<=np.polyval(rfit,yy)+.5)&(a[...,3]>0)).astype(np.uint8)
        mask|=polygon>0;selected.append(dict(box=[x,y,x+bw,y+bh],white_pixels=area,ink_pixels=int(ink.sum())))
    if not mask.any():
        recovered,proof=seeded_paper(a,face)
        if recovered is not None:return recovered,proof
    proof=dict(method='white_paper_component_with_enclosed_print_v1',components=selected)
    if face=='bottom' and mask.any():mask,proof=trim_strip_to_source_end(a,mask,proof)
    return mask,proof


def source_color(image,mask):
    a=np.array(image.convert('RGBA'));rgb=a[...,:3]
    samples=rgb[mask&(a[...,3]>=240)]
    if len(samples)<20:raise ValueError('Too few opaque material pixels')
    return np.median(samples,axis=0)


def save_mask(path,mask):
    Image.fromarray(mask.astype(np.uint8)*255,'L').save(path)
    return str(Path(path).resolve())


def reference_box(a,mask):
    """Choose a complete, opaque source patch within this material only."""
    from .lighting_review import lightness
    opaque=mask&(a[...,3]>=250);h,w=mask.shape
    L=lightness(a[...,:3]);target=float(np.percentile(L[opaque],60)) if opaque.any() else 0
    distance=ndimage.distance_transform_edt(opaque)
    for bh,bw in ((7,7),(12,3),(3,12),(24,1),(1,24)):
        if bh>h or bw>w:continue
        valid=ndimage.minimum_filter(opaque.astype(np.uint8),size=(bh,bw),mode='constant')>0
        yy,xx=np.nonzero(valid)
        if not len(xx):continue
        score=np.abs(L[yy,xx]-target)-np.minimum(distance[yy,xx],6)*.05
        i=int(np.argmin(score));y,x=int(yy[i]),int(xx[i])
        y0=max(0,y-bh//2);x0=max(0,x-bw//2);y1=min(h,y0+bh);x1=min(w,x0+bw)
        if opaque[y0:y1,x0:x1].all() and (y1-y0)*(x1-x0)>=20:return [x0,y0,x1,y1]
    raise ValueError('No complete opaque source reference patch for material domain')


def front_metal(image):
    """Measure the glass/rim transition on four straight source edges."""
    a=np.array(image.convert('RGBA'));h,w=a.shape[:2];opaque=a[...,3]>=128
    distance=ndimage.distance_transform_edt(np.pad(a[...,3]>=128,1))[1:-1,1:-1]
    y=np.arange(int(h*.30),int(h*.70));left=np.median(np.argmax(opaque[y],axis=1));right=np.median(w-1-np.argmax(opaque[y,::-1],axis=1))
    gray=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY).astype(float)
    g=np.gradient(ndimage.gaussian_filter1d(gray,.8,axis=1),axis=1)
    profile=np.median(g[y],axis=0);transitions=[];support_checks=[]
    x=np.arange(int(w*.30),int(w*.70));top=np.median(np.argmax(opaque[:,x],axis=0));bottom=np.median(h-1-np.argmax(opaque[::-1,x],axis=0))
    gy=np.gradient(ndimage.gaussian_filter1d(gray,.8,axis=0),axis=0)
    yprofile=np.median(gy[:,x],axis=1)
    for edge_index,(edge,sign,values_profile,extent) in enumerate(((left,1,profile,w),(right,-1,profile,w),(top,1,yprofile,h),(bottom,-1,yprofile,h))):
        offsets=np.arange(-2,max(8,int(w*.045)))
        xx=np.rint(edge+sign*offsets).astype(int);xx=np.clip(xx,0,extent-1)
        # The metal-to-glass transition darkens inwards. The opposite sign is
        # the transparent edge, while a deeper, weaker transition is the LCD
        # bezel and must not be included in the aluminum fitting mask.
        values=-sign*values_profile[xx]
        from scipy.signal import find_peaks
        # Alpha's soft outer shoulder also creates a darkening RGB peak.
        # A metal/glass boundary must be supported by opaque source pixels
        # across the sampled straight edge, not just a strong RGB gradient.
        alpha_samples=a[y[:,None],xx[None,:],3] if edge_index<2 else a[xx[:,None],x[None,:],3].T
        opaque_support=np.mean(alpha_samples>=240,axis=0)
        valid=(offsets>=(6 if edge_index<2 else 1))&(opaque_support>=.8)
        if not valid.any():raise ValueError('Front glass/metal transition lacks opaque support')
        peaks,_=find_peaks(values,height=max(.5,float(values[valid].max())*.30))
        peaks=peaks[valid[peaks]]
        if not len(peaks):raise ValueError('Front glass/metal transition is uncertain')
        selected=int(peaks[0] if edge_index<2 else peaks[np.argmax(values[peaks])])
        transitions.append(float(np.abs(xx[selected]-edge)))
        support_checks.append(float(opaque_support[selected]))
    thickness=np.array(transitions)+1
    if np.any(thickness<2) or np.any(thickness>w*.045):raise ValueError('Implausible front metal rim thickness')
    # Photographing the bevel exposes different widths on the four edges.
    # Use the local contour normal to interpolate only around curved corners;
    # a side width must not extend across bottom glass or leave a top rim out.
    dy,dx=np.gradient(ndimage.gaussian_filter(distance,3));total=np.maximum(np.abs(dx)+np.abs(dy),1e-8)
    vertical=np.where(dx>=0,thickness[0],thickness[1]);horizontal=np.where(dy>=0,thickness[2],thickness[3])
    local=(np.abs(dx)*vertical+np.abs(dy)*horizontal)/total
    return (a[...,3]>0)&(distance<=local),dict(method='four_source_glass_rim_gradients_v3_opaque_boundary',opaque_boundary_support=support_checks,edge_order=['left','right','top','bottom'],thickness_px=thickness.tolist(),edge_measurements_px=transitions)


def rear_panel(image):
    """Measure the actual inset rear panel seam in this studio capture.

    Independent straight-edge peaks and four source-supported rounded corners
    replace the old image-height split. Only classifies material; never edits
    the silhouette or alpha. Unsupported captures are rejected.
    """
    from scipy.signal import find_peaks
    a=np.array(image.convert('RGBA'));h,w=a.shape[:2]
    gray=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY).astype(float)
    ridge=ndimage.gaussian_filter(gray,3)-ndimage.gaussian_filter(gray,.6)
    vertical=np.percentile(ridge[int(h*.45):int(h*.70)],70,axis=0)
    horizontal=np.median(ridge[:,int(w*.3):int(w*.7)],axis=1)
    def peak(profile,start,stop,last=False):
        lo,hi=int(len(profile)*start),int(len(profile)*stop)
        values=profile[lo:hi]
        peaks,_=find_peaks(values,height=max(.8,float(values.max())*.25),distance=5)
        if not len(peaks):raise ValueError('Rear panel boundary is uncertain')
        index=int(peaks[-1] if last else peaks[np.argmax(values[peaks])])+lo
        return index,float(profile[index])
    left,ls=peak(vertical,.04,.13);right,rs=peak(vertical,.87,.96)
    top,ts=peak(horizontal,.27,.39,True);bottom,bs=peak(horizontal,.90,.99)
    if min(ls,rs,ts,bs)<2 or not .7*w<right-left<.95*w:
        raise ValueError('Rear panel has insufficient independent seam evidence')
    angles=np.linspace(.08,np.pi/2-.08,100)
    corners=[]
    for x,y,sx,sy in [(left,top,1,1),(right,top,-1,1),(left,bottom,1,-1),(right,bottom,-1,-1)]:
        radii=np.arange(int(w*.035),int(w*.2))
        scores=[]
        for r in radii:
            xx=x+sx*r*(1-np.cos(angles));yy=y+sy*r*(1-np.sin(angles))
            scores.append(float(ndimage.map_coordinates(ridge,[yy,xx],order=1).mean()))
        radius=int(radii[np.argmax(scores)])
        if max(scores)<2 or radius in (radii[0],radii[-1]):
            raise ValueError('Rear panel corner lacks source seam support')
        corners.append(dict(radius=radius,mean_seam_evidence=max(scores)))
    # Fit corners independently: perspective can expose different radii.
    yy,xx=np.indices((h,w));panel=(xx>=left)&(xx<=right)&(yy>=top)&(yy<=bottom)
    for (x,y,sx,sy),c in zip([(left,top,1,1),(right,top,-1,1),(left,bottom,1,-1),(right,bottom,-1,-1)],corners):
        r=c['radius'];cx=x+sx*r;cy=y+sy*r
        corner=((xx-cx)*sx<0)&((yy-cy)*sy<0)
        panel &= ~corner|((xx-cx)**2+(yy-cy)**2<=r*r)
    return panel,dict(method='source_inset_panel_seam_v1',bounds=[left,top,right,bottom],
        edge_evidence=[ls,ts,rs,bs],corners=corners,source_rgba_sha256=digest_image_array(a))


def digest_image_array(a):
    import hashlib
    return hashlib.sha256(a.tobytes()).hexdigest()



def rear_body_contour(image):
    """Illumination coordinates follow the body, never the protruding buttons.

    A quantile quadratic estimates each exposed long edge from the current
    alpha, rejecting outward hardware. This is only a coordinate mask: the
    saved image silhouette and editable button pixels are never cropped.
    """
    from scipy.optimize import linprog
    a=np.array(image.convert('RGBA'));h,w=a.shape[:2];inside=a[...,3]>=128
    rows=np.arange(int(h*.12),int(h*.88),max(1,h//500))
    if len(rows)<60 or not inside[rows].any(axis=1).all():
        raise ValueError('Rear body lacks continuous source edges')
    t=rows/h*2-1;X=np.column_stack([t*t,t,np.ones(len(t))]);n=len(rows)
    edges=[np.argmax(inside,axis=1),w-1-np.argmax(inside[:,::-1],axis=1)]
    virtual=inside.copy();proof=dict(method='source_body_quantile_contour_v1',sides={})
    yy=np.arange(h);xx=np.arange(w)[None,:]
    ramp=np.clip(np.minimum(yy-h*.12,h*.88-yy)/(h*.025),0,1);ramp=ramp*ramp*(3-2*ramp)
    for side,edge,q,sign in zip(('left','right'),edges,(.75,.25),(-1,1)):
        result=linprog(np.r_[np.zeros(3),np.full(n,q),np.full(n,1-q)],
            A_eq=np.column_stack([X,np.eye(n),-np.eye(n)]),b_eq=edge[rows],
            bounds=[(None,None)]*3+[(0,None)]*(2*n),method='highs')
        if not result.success:raise ValueError('Rear body contour fit failed')
        coefficients=result.x[:3];curve=np.polyval(coefficients,yy/h*2-1)
        residual=np.abs(edge[rows]-curve[rows]);support=float(np.mean(residual<=1.5))
        if support<.55 or np.max(np.abs(edge[rows]-curve[rows]))>w*.025:
            raise ValueError('Rear body contour lacks independent source support')
        blended=edge+(curve-edge)*ramp
        strip=xx<w*.1 if side=='left' else xx>w*.9
        replacement=xx>=blended[:,None] if side=='left' else xx<=blended[:,None]
        virtual=np.where(strip&((ramp>0)[:,None]),replacement,virtual)
        protrusion=(sign*(edge-curve)>2)&(ramp>.99)
        protrusion=ndimage.binary_closing(protrusion,iterations=3)
        components,count=ndimage.label(protrusion);ranges=[]
        for label in range(1,count+1):
            found=np.flatnonzero(components==label)
            if len(found)>=max(10,h*.012):ranges.append([int(found[0]),int(found[-1])+1])
        proof['sides'][side]=dict(coefficients=coefficients.tolist(),source_support=support,protrusion_rows=ranges)
    if np.any(virtual.sum(axis=1)[rows]<w*.75):raise ValueError('Rear body contour width is inconsistent')
    return virtual,proof

def propose_job(source,face,folder,companions=None,landmarks=None):
    """Create fresh material masks/reference boxes for one geometry candidate."""
    from .auto_structure import antenna_rows,per_column_boundaries,hardware_ranges
    image=Image.open(source).convert('RGBA');a=np.array(image);h,w=a.shape[:2]
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    visible=a[...,3]>0;paper,paper_proof=paper_mask(image,face)
    if face in ('back','bottom') and not paper.any():
        raise ValueError('This QC studio profile requires a resolved printed paper label on back/bottom')
    paper &= visible;paper_path=save_mask(folder/'paper.png',paper)
    remaining=visible&~paper;materials=[];proof={'paper':paper_proof}
    if face=='front':
        remaining,rim=front_metal(image);remaining &= ~paper;proof['front_rim']=rim
    if face in ('left','right'):
        if landmarks is not None:
            cfg=landmarks[face]
            if cfg['source_sha256']!=digest(source):raise ValueError('Material landmarks belong to another source')
            lines=np.asarray(cfg['source_lines'])[cfg['antenna_indices']]
            rows=np.median(lines[:,int(w*.4):int(w*.6)],axis=1)
            antenna_proof=dict(cfg['source_detection'],shared_geometry_landmarks=True)
        else:
            rows,signs,antenna_proof=antenna_rows(image,[int(w*.4),int(w*.6)])
            lines=per_column_boundaries(image,rows)
        yy=np.arange(h)[:,None]
        for i in range(2):
            region=(yy>=lines[2*i])&(yy<lines[2*i+1])&remaining
            materials.append(('antenna_'+str(i),region,[max(6,(rows[2*i+1]-rows[2*i])*.30),max(8,w*.20)]))
            remaining &= ~region
        buttons,boxes=hardware_ranges(image);proof.update(antenna=antenna_proof,hardware_boxes=boxes)
        # Complete detected hardware has its own material/reference, including
        # dark glass control surfaces. It is not a frozen pixel rectangle.
        hw=np.zeros((h,w),bool)
        for x0,y0,x1,y1 in boxes:
            hw[y0:y1,x0:x1]=True
        hw &= remaining
        if hw.sum()>=30:materials.append(('hardware',hw,[max(6,h*.012),max(2,w*.08)]));remaining &= ~hw
    if face=='back':
        # Camera glass/flash components are distinct from aluminum and do not
        # enter metal fitting. Shape + position are required, not darkness alone.
        gray=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY);upper=np.zeros((h,w),bool);upper[:int(h*.30)]=True
        dark=(gray<max(35,float(np.percentile(gray[remaining&upper],15))))&remaining&upper
        n,labels,stats,_=cv2.connectedComponentsWithStats(dark.astype(np.uint8),8);glass=np.zeros((h,w),bool)
        for i in range(1,n):
            x,y,bw,bh,area=map(int,stats[i])
            if .025*w<bw<.35*w and .025*w<bh<.35*w and .5<bw/max(1,bh)<2 and area>40:
                glass|=ndimage.binary_fill_holes(labels==i)
        remaining &= ~glass;proof['camera_glass_pixels']=int(glass.sum())
        body_contour,body_proof=rear_body_contour(image)
        body_path=save_mask(folder/'body-contour.png',body_contour);proof['rear_body_contour']=body_proof
        distance=ndimage.distance_transform_edt(np.pad(a[...,3]>=128,1))[1:-1,1:-1];rim=remaining&(distance<=max(8,w*.035))
        # Rear plastic antenna bars must never enter the aluminum illumination
        # field. Locate them before lighting from this same fresh SKU's side
        # widths and independently agreeing rear RGB strips.
        from .auto_structure import body_rows,rear_reference
        if companions is None:raise ValueError('Current same-SKU geometry companions required for rear material classification')
        middle=a[int(h*.25):int(h*.75),:,3]>=128
        left=int(np.median(np.argmax(middle,axis=1)));right=int(np.median(w-1-np.argmax(middle[:,::-1],axis=1)))
        rear_proof={};yy=np.arange(h)[:,None];xx=np.arange(w)[None,:];back_body=body_rows(image)
        for side in ('left','right'):
            side_image=Image.open(companions[side]).convert('RGBA');sw=side_image.width
            rows,_,_=antenna_rows(side_image,[int(sw*.4),int(sw*.6)]);body=body_rows(side_image)
            expected=float(np.mean([rows[1]-rows[0],rows[3]-rows[2]])*(back_body[1]-back_body[0])/(body[1]-body[0]))
            if landmarks is not None:
                ref=landmarks[side]['back']
                if ref['sha256']!=digest(source):raise ValueError('Rear material landmarks belong to another source')
                br,xr,brproof=ref['antenna_rows'],ref['x_range'],dict(ref['source_detection'],shared_geometry_landmarks=True)
            else:br,_,xr,brproof=rear_reference(image,side,right if side=='left' else left,expected)
            side_columns=(xx>w/2) if side=='left' else (xx<w/2)
            for i in range(2):
                region=rim&side_columns&(yy>=br[2*i])&(yy<br[2*i+1])
                if np.count_nonzero(region&(a[...,3]>=240))<20:raise ValueError('Rear plastic strip has insufficient independent material samples')
                materials.append((f'rear_antenna_{side}_{i}',region,[max(6,(br[2*i+1]-br[2*i])*.5),max(8,w*.006)]))
                remaining &= ~region;rim &= ~region
            rear_proof[side]=dict(antenna_rows=br,signal_columns=xr,measurement=brproof,
                side_geometry_sha256=digest(companions[side]))
        proof['rear_plastic_materials']=rear_proof
        if rim.sum()>=30:materials.append(('rear_perimeter',rim,[max(6,h*.012),max(2,w*.004)]));remaining &= ~rim
        panel,panel_proof=rear_panel(image);proof['rear_panel']=panel_proof
        camera=remaining&~panel
        if camera.sum()>=30:materials.append(('camera_surround',camera,[max(6,h*.015),max(6,w*.02)]));remaining &= ~camera
    if remaining.sum()>=30:
        materials.append(('main_surface',remaining,[max(3,h*.018),max(3,w*.12)] if face in ('left','right','top','bottom') else [max(8,h*.015),max(8,w*.025)]))
    domains=[];allowed=np.zeros((h,w),bool)
    for name,region,sigma in materials:
        if np.count_nonzero(region&(a[...,3]>=240))<20:raise ValueError('Material region has no reliable fit samples: '+name)
        ref=reference_box(a,region);mp=save_mask(folder/(name+'.png'),region);allowed|=region
        domains.append(dict(name=name,mask=mp,reference_boxes=[ref],sigma=sigma,
            strength=[.72,0,0],feather=0,correction_smooth=1.5))
        if name=='rear_perimeter' or (face=='front' and name=='main_surface'):
            domains[-1].update(field='contour',sigma=[max(18,h*.012),max(5,w*.006)])
        if name=='rear_perimeter':domains[-1].update(reference_profile='contour_depth_lift',strength=[.92,0,0],max_lab_shift=[48,0,0],contour_alpha_mask=body_path,source_body_contour=body_proof)
        if face=='front':
            reliable=region&(a[...,3]>=240)
            shoulder=float(ndimage.distance_transform_edt(~reliable)[region&(a[...,3]>=128)].max())
            if shoulder>max(8,w*.045):raise ValueError('Front rim lacks nearby opaque material support')
            domains[-1].update(reference_profile='contour_depth',contour_inset_depth=max(3.,shoulder+1),
                sigma=[max(18,h*.012),2.])
            proof['front_opaque_shoulder_px']=shoulder
    if not domains:raise ValueError('No reliable material domains')
    required=save_mask(folder/'required-editable.png',allowed)
    view=dict(name=face,source=str(Path(source).resolve()),source_sha256=digest(source),
        protection_policy='sticker_only',sticker_mask=paper_path,protected_mask=paper_path,
        required_editable_mask=required,reference_boxes=domains[0]['reference_boxes'],domains=domains)
    if face=='back':
        view['correction_blends']=[dict(domains=['rear_perimeter','camera_surround','main_surface'],
            boundary_domain='rear_perimeter',sigma=max(3.,w*.008))]
    job=dict(color_mode='source_lightness',defaults=dict(max_lab_shift=[35,0,0],feather=0),views=[view])
    proof.update(materials=[dict(name=d['name'],reference_boxes=d['reference_boxes'],mask=d['mask']) for d in domains],source_sha256=digest(source),source=str(source))
    return job,proof,allowed,paper


def lighting_scope(source,face,job,folder):
    """Freeze source-relative perimeter/local effect witnesses before correction."""
    from .lighting_review import ZONES,lightness,validate_spec
    image=Image.open(source).convert('RGBA');a=np.array(image);h,w=a.shape[:2]
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    view=job['views'][0];domains={d['name']:d for d in view['domains']}
    spec=dict(source_sha256=digest(source),action='aluminum',reason='Fresh source-derived aluminum QC studio profile',
        mode='perimeter' if face in ('front','back') else 'regional',protection_policy='sticker_only',sticker_mask=view['sticker_mask'],local_checks={})
    def domain_mask(name):return np.array(Image.open(domains[name]['mask']).convert('L'))>0
    def target(name):
        boxes=domains[name]['reference_boxes'];values=[]
        for x0,y0,x1,y1 in boxes:values.append(float(np.median(lightness(a[y0:y1,x0:x1,:3]))))
        return float(np.median(values))
    normal_target=None
    profile_name='main_surface' if face=='front' else 'rear_perimeter'
    if face in ('front','back') and domains[profile_name].get('reference_profile')=='contour_depth':
        from .lighting import engine
        module=engine('aluminum');metal=domain_mask(profile_name)
        normal_target,profile_info=module.contour_depth_reference(module.rgb_to_lab(a[...,:3]),metal&(a[...,3]>=240),a[...,3])
        spec['source_normal_reference_profile']=profile_info
    def point_witness(material,x,y,name,minimum):
        candidates=material&(a[...,3]>=240)
        # Choose a complete local population, not the single nearest opaque
        # speck at an antialiased corner. The source-only search preserves the
        # existing sample gate and cannot inspect corrected pixel values.
        populations=ndimage.uniform_filter(candidates.astype(float),size=11,mode='constant')*121
        yy,xx=np.nonzero(candidates&(populations>=20-1e-6))
        if not len(xx):raise ValueError('Effect witness has no opaque source material')
        idx=int(np.argmin((xx-x)**2+(yy-y)**2));cx,cy=int(xx[idx]),int(yy[idx])
        rr=np.zeros((h,w),bool);rr[max(0,cy-5):min(h,cy+6),max(0,cx-5):min(w,cx+6)]=True;rr &= candidates
        if rr.sum()<20:raise ValueError('Too few independent source effect pixels')
        if normal_target is not None:minimum=float(np.median(normal_target[rr]))
        source_values=lightness(a[...,:3])[rr]
        source_L=float(np.median(source_values))
        if face=='back' and domains['rear_perimeter'].get('reference_profile')=='contour_depth_lift':
            # Freeze an actual shadow-reduction requirement, not just a wide
            # corruption guard that accepts the original dark border.
            source_dark=float(np.percentile(source_values,10))
            path=save_mask(folder/(name+'.png'),rr)
            spec['local_checks']['rear_shadow_'+name]=dict(mask=path,percentile=10,
                min_L=source_dark+.50*max(0.,minimum-source_dark),max_L=min(100.,max(minimum,source_L)+20.),
                source_before_L=source_dark,source_reference_L=minimum,
                bounds_basis='Recover at least half the source shadow deficit to same-face metal reference')
            return dict(mask=path,min_L=max(.01,source_L+.55*max(0.,minimum-source_L)),
                max_L=min(100.,max(minimum,source_L)+20.),source_before_L=source_L,source_reference_L=minimum,
                bounds_basis='Recover at least 55% of source median shadow deficit; preserve source detail')
        # A source-relative damage guard cannot label real dark wear or a
        # polished facet defective merely for differing from a global target.
        # Both measured source and source-profile reference are frozen before
        # correction; desired appearance still requires visual acceptance.
        return dict(mask=save_mask(folder/(name+'.png'),rr),min_L=max(0.,min(minimum,source_L)*.55),
            max_L=min(100.,max(minimum,source_L)+25.),source_before_L=source_L,source_reference_L=minimum,
            bounds_basis='source-relative corruption guard; does not certify removal of black borders')
    if face in ('front','back'):
        name='rear_perimeter' if face=='back' else 'main_surface';metal=domain_mask(name)
        spec.update(required_metal_mask=save_mask(folder/'required-metal.png',metal),
            reference_basis='Current source material patches; L bounds relative to this face, no shared color/SKU threshold',zones={})
        visible=a[...,3]>0;yy,xx=np.nonzero(visible);x0,x1,y0,y1=int(xx.min()),int(xx.max()),int(yy.min()),int(yy.max())
        points={'left':(x0+3,(y0+y1)/2),'right':(x1-3,(y0+y1)/2),
            'top':((x0+x1)/2,y0+3),'bottom':((x0+x1)/2,y1-3),
            'top_left':(x0+6,y0+6),'top_right':(x1-6,y0+6),
            'bottom_left':(x0+6,y1-6),'bottom_right':(x1-6,y1-6)}
        for zone in ZONES:spec['zones'][zone]=point_witness(metal,*points[zone],zone,target(name))
    if face=='back' and domains['rear_perimeter'].get('source_body_contour'):
        metal=domain_mask('rear_perimeter');opaque=metal&(a[...,3]>=240)
        for side,entry in domains['rear_perimeter']['source_body_contour']['sides'].items():
            for index,(lo,hi) in enumerate(entry['protrusion_rows']):
                witness=opaque.copy();padding=max(20,int(h*.015))
                witness[:max(0,lo-padding)]=False;witness[min(h,hi+padding):]=False
                if side=='left':witness[:,w//2:]=False
                else:witness[:,:w//2]=False
                key=f'rear_button_{side}_{index}'
                if witness.sum()<20:raise ValueError('Rear button lacks independent source material samples')
                spec['local_checks'][key]=dict(mask=save_mask(folder/(key+'.png'),witness),
                    min_L=0,max_L=100,percentile=10,max_added_step_L=1.25,step_axis=0,
                    bounds_basis='Continuous added illumination around source-detected protrusion; no button protection')
    if face=='back':
        # Independent witnesses across the old artificial split, excluding
        # actual material boundaries, paper and lens openings. Erosion limits
        # the test samples only; it never freezes processing pixels.
        interior=np.zeros((h,w),bool)
        for name in ('camera_surround','main_surface'):
            interior |= ndimage.binary_erosion(domain_mask(name),iterations=4)
        interior &= a[...,3]>=240
        y=int(h*.30);half=max(12,int(h*.008))
        for name,lo,hi in [('rear_upper_join_left',.06,.22),('rear_upper_join_center',.30,.70),('rear_upper_join_right',.78,.94)]:
            witness=np.zeros((h,w),bool)
            witness[max(0,y-half):min(h,y+half+1),int(w*lo):int(w*hi)]=True
            witness &= interior
            if (witness[1:]&witness[:-1]).sum()<20:
                raise ValueError('Rear junction lacks source continuity samples')
            spec['local_checks'][name]=dict(mask=save_mask(folder/(name+'.png'),witness),
                min_L=0,max_L=100,percentile=10,max_added_step_L=1.25,step_axis=0,
                bounds_basis='Added lightness step across same-source-material pixels; real texture subtracted')
    if face=='back' and view.get('correction_blends'):
        metal=domain_mask('rear_perimeter')&(a[...,3]>=240)
        peers=(domain_mask('camera_surround')|domain_mask('main_surface'))&(a[...,3]>=240)
        edge=metal&ndimage.binary_dilation(peers)
        yy,xx=np.indices((h,w))
        nearest=np.argmin(np.stack([xx,w-1-xx,yy,h-1-yy]),axis=0)
        peer_path=save_mask(folder/'rear-transition-peers.png',peers)
        for side,index in [('left',0),('right',1),('top',2),('bottom',3)]:
            key='rear_inner_boundary_'+side
            spec['local_checks'][key]=dict(mask=save_mask(folder/(key+'.png'),edge&(nearest==index)),
                boundary_peer_mask=peer_path,min_L=0,max_L=100,percentile=10,max_boundary_step_L=2.,
                bounds_basis='99th percentile of added L step across source-defined metal-domain interfaces; both axes, peak <= 3 L')
    if face in ('left','right'):
        for name in ('antenna_0','antenna_1'):
            material=domain_mask(name);yy,xx=np.nonzero(material&(a[...,3]>=240));middle=float(np.median(yy))
            for end,x in [('left',w*.12),('right',w*.88)]:
                key=name+'_'+end;spec['local_checks'][key]=point_witness(material,x,middle,key,target(name))
    validate_spec(spec,source)
    return spec
