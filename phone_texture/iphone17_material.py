"""Current-image material proposals for the glass-back standard iPhone 17."""
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage
from .auto_material import (paper_mask, seeded_paper, save_mask, reference_box,
                            front_metal, rear_body_contour)
from .geometry_review import digest


def paper(image, face):
    if face != 'bottom':
        # The Pro printed-tail reconstruction assumes the Pro label layout.
        return paper_mask(image, face, source_tail=False)
    a = np.array(image.convert('RGBA')); h,w = a.shape[:2]
    proposal, proof = seeded_paper(a, face, resolve_end=False)
    if proposal is None:
        raise ValueError('Standard iPhone 17 bottom label has no supported print seed')
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
    if baseline<3 or not len(candidates):raise ValueError('Paper side endings are unresolved')
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


def glass_back_rim(image,sticker=None):
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
        if not len(peaks):raise ValueError('Glass-back metal boundary lacks source gradient')
        k=int(peaks[np.argmax(values[peaks])])
        width=float(offsets[k]+1)
        if not 3<=width<=w*.035:raise ValueError('Implausible standard rear rim width')
        widths.append(width);measurements.append(dict(edge=i,width_px=width,gradient=float(values[k])))
    dy,dx=np.gradient(ndimage.gaussian_filter(distance,3));den=np.maximum(abs(dx)+abs(dy),1e-8)
    local=(abs(dx)*np.where(dx>=0,widths[0],widths[1])+abs(dy)*np.where(dy>=0,widths[2],widths[3]))/den
    return (a[...,3]>0)&(distance<=local),dict(method='standard_glass_back_source_transition',
        edge_measurements=measurements,body_contour=body_proof),body


def job(source,face,folder,structure):
    from PIL import Image
    image=Image.open(source).convert('RGBA');a=np.array(image);h,w=a.shape[:2]
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    sticker,sticker_proof=paper(image,face);visible=a[...,3]>0
    domains=[];allowed=np.zeros((h,w),bool);regions=[]
    proof=dict(source_sha256=digest(source),paper=sticker_proof)
    if face=='front':
        rim,rim_proof=front_metal(image);proof['rim']=rim_proof
        regions=[('main_surface',rim,[max(18,h*.012),2.],True)]
    elif face=='back':
        rim,rim_proof,body=glass_back_rim(image,sticker);proof['rim']=rim_proof
        body_path=save_mask(folder/'body-contour.png',body)
        regions=[('rear_perimeter',rim,[max(18,h*.012),2.],True)]
    else:
        remaining=visible&~sticker
        if face in ('left','right'):
            lines=np.asarray(structure[face]['antenna_lines']);yy=np.arange(h)[:,None]
            for i in range(2):
                mask=remaining&(yy>=lines[2*i])&(yy<lines[2*i+1])
                regions.append((f'antenna_{i}',mask,[max(4,np.median(lines[2*i+1]-lines[2*i])*.3),max(4,w*.2)],False));remaining &= ~mask
            from .iphone17_structure import capsule_mask
            gray=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY)
            for index,box in enumerate(structure[face]['hardware_boxes']):
                hw=capsule_mask(image,box)&remaining
                cut,_=cv2.threshold(gray[hw].reshape(-1,1),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                dark=hw&(gray<=cut);face_mask=hw&~dark
                for suffix,mask in [('inset',dark),('face',face_mask)]:
                    regions.append((f'hardware_{index}_{suffix}',mask,[max(6,h*.008),max(2,w*.05)],False))
                remaining &= ~hw
        else:
            # Apertures have geometric shape support as well as darkness.
            gray=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY)
            n,labels,stats,_=cv2.connectedComponentsWithStats(((gray<145)&remaining).astype('uint8'),8)
            holes=np.zeros((h,w),bool)
            for i in range(1,n):
                x,y,bw,bh,area=stats[i];cx=x+bw/2;cy=y+bh/2
                if (.25*h<cy<.75*h and bh>h*.08 and bw>h*.08 and
                    area>25 and bw<w*.18 and .3< bw/max(1,bh)<5):
                    holes|=ndimage.binary_fill_holes(labels==i)
            if holes.any():
                regions.append(('apertures',holes,[max(3,h*.05),max(3,w*.008)],False));remaining &= ~holes
            # A full-thickness antenna is a supported vertical stripe, not a
            # global lightness threshold applied to arbitrary materials.
            y0,y1=int(h*.2),int(h*.8)
            signal=np.median(gray[y0:y1].astype(float),axis=0)
            from scipy.signal import find_peaks
            g=np.gradient(ndimage.gaussian_filter1d(signal,.8));peaks,_=find_peaks(abs(g),height=2,distance=3)
            bands=[]
            for lo in peaks:
                for hi in peaks:
                    if .004*w<hi-lo<.025*w and g[lo]<-2 and g[hi]>2 and .12*w<lo<.9*w:
                        bands.append((min(-g[lo],g[hi]),int(lo),int(hi)))
            if not bands:raise ValueError('Standard top/bottom antenna stripe is unresolved')
            _,lo,hi=max(bands);stripe=remaining&(np.arange(w)[None,:]>=lo)&(np.arange(w)[None,:]<hi)
            regions.append(('antenna',stripe,[max(3,h*.1),max(2,(hi-lo)*.3)],False));remaining &= ~stripe
            proof['antenna_columns']=[lo,hi]
        regions.append(('main_surface',remaining,[max(3,h*.018),max(3,w*.12)],False))
    for name,mask,sigma,contour in regions:
        mask &= visible&~sticker
        if (mask&(a[...,3]>=240)).sum()<20:raise ValueError('Insufficient material evidence: '+name)
        path=save_mask(folder/(name+'.png'),mask);ref=reference_box(a,mask);allowed|=mask
        d=dict(name=name,mask=path,reference_boxes=[ref],sigma=sigma,strength=[.72,0,0],feather=0,correction_smooth=1.5)
        if contour:
            reliable=mask&(a[...,3]>=240)
            shoulder=float(ndimage.distance_transform_edt(~reliable)[mask&(a[...,3]>=128)].max())
            if shoulder>max(8,w*.045):raise ValueError('Standard rim lacks nearby opaque metal support')
            d.update(field='contour',reference_profile='contour_depth',contour_inset_depth=max(3.,shoulder+1))
            if face=='back':d['contour_alpha_mask']=body_path
        domains.append(d)
    sticker_path=save_mask(folder/'paper.png',sticker)
    required=save_mask(folder/'required-editable.png',allowed)
    view=dict(name=face,source=str(Path(source).resolve()),source_sha256=digest(source),
        protection_policy='sticker_only',sticker_mask=sticker_path,protected_mask=sticker_path,
        required_editable_mask=required,reference_boxes=domains[0]['reference_boxes'],domains=domains)
    return dict(color_mode='source_lightness',defaults=dict(max_lab_shift=[35,0,0],feather=0),views=[view]),proof,allowed,sticker
