"""Measure a filled camera control against the neighbouring metal on the same rail."""
import cv2
import numpy as np
from scipy import ndimage


def _refine_boundary(mask, rgb, color_step):
    """Find the signed optical transition along the local outward normal."""
    contours=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    points=contours[0][:,0].astype(float)
    tangent=np.roll(points,-3,axis=0)-np.roll(points,3,axis=0)
    normal=np.column_stack((-tangent[:,1],tangent[:,0]))
    normal/=np.maximum(1,np.linalg.norm(normal,axis=1))[:,None]
    if np.median(((points-points.mean(axis=0))*normal).sum(axis=1))<0:
        normal*=-1
    signal=rgb @ (color_step/np.linalg.norm(color_step))
    offsets=np.arange(-5,5.5,.5)
    coords=points[:,None,:]+normal[:,None,:]*offsets[None,:,None]
    profile=ndimage.map_coordinates(signal,[coords[...,1],coords[...,0]],order=1)
    grad=np.gradient(profile,.5,axis=1)
    k=np.argmax(grad[:,2:-2],axis=1)+2
    strength=grad[np.arange(len(k)),k]
    valid=(k>2)&(k<len(offsets)-3)&(strength>1)&(profile[:,-1]-profile[:,0]>3)
    if valid.mean()<.95:
        return mask
    shift=np.where(valid,offsets[k],0)
    shift=ndimage.median_filter(shift,size=5,mode='wrap')
    refined=points+normal*shift[:,None]
    out=np.zeros_like(mask,dtype=np.uint8)
    cv2.fillPoly(out,[np.rint(refined).astype(np.int32)],1)
    return out.astype(bool)


def _fit_control_edges(mask, distances, rgb, step, projection):
    """Fit the capsule's two centres and radius to witnessed photographic edges."""
    from scipy.optimize import least_squares
    x,y,width,height=cv2.boundingRect(mask.astype('uint8'))
    radius=(width-1)*.5
    cx=x+radius
    initial=np.array([cx,y+radius,cx,y+height-1-radius,radius])
    theta=np.linspace(0,np.pi,36)
    t=np.linspace(0,1,64)
    def samples(p):
        x0,y0,x1,y1,r=p
        axis=np.array([x1-x0,y1-y0]);axis/=np.linalg.norm(axis)
        normal=np.array([axis[1],-axis[0]])
        ends=np.array([[x0,y0],[x1,y1]])
        middle=(1-t[:,None])*ends[0]+t[:,None]*ends[1]
        cap=np.cos(theta)[:,None]*normal+np.sin(theta)[:,None]*axis
        rays=np.vstack((-cap,np.tile(normal,(len(t),1)),cap,np.tile(-normal,(len(t),1))))
        return np.vstack((ends[0]-r*cap,middle+r*normal,ends[1]+r*cap,middle-r*normal)),rays
    projected=rgb@(step/max(1,np.linalg.norm(step)))
    def residual(p):
        points,rays=samples(p)
        # Each residual is measured on the photo, including both end caps.
        observed=[ndimage.map_coordinates(d,[points[:,1],points[:,0]],order=1) for d in distances]
        inside=points-2*rays
        outside=points+2*rays
        contrast=(ndimage.map_coordinates(projected,[outside[:,1],outside[:,0]],order=1)-
                  ndimage.map_coordinates(projected,[inside[:,1],inside[:,0]],order=1))
        fraction=ndimage.map_coordinates(projection,[points[:,1],points[:,0]],order=1)
        return np.r_[np.median(observed,axis=0),np.maximum(0.,4.-contrast)*.5,
                     3.*np.clip(fraction-.5,-1,1)]
    delta=np.array([6,8,6,8,5])
    bounds=(initial-delta,initial+delta)
    starts=[initial,initial+np.array([2.,0.,2.,0.,-2.]),initial+np.array([-2.,0.,-2.,0.,2.])]
    fits=[least_squares(residual,start,bounds=bounds,loss='soft_l1',diff_step=1e-4) for start in starts]
    points,_=samples(min(fits,key=lambda fit:fit.cost).x)
    result=np.zeros_like(mask,dtype=np.uint8)
    cv2.fillConvexPoly(result,np.rint(points).astype(np.int32),1)
    return result.astype(bool)


def _native_normal_edges(mask, rgb, ys, xs, tolerance):
    signed=ndimage.distance_transform_edt(~mask)-ndimage.distance_transform_edt(mask)
    gy,gx=np.gradient(ndimage.gaussian_filter(signed,1.))
    vx=gx[ys,xs];vy=gy[ys,xs]
    length=np.maximum(1e-6,np.hypot(vx,vy))
    offsets=np.arange(-5.,5.1,.5)
    yy=ys[:,None]+(vy/length)[:,None]*offsets
    xx=xs[:,None]+(vx/length)[:,None]*offsets
    witnesses=[]
    for channel in range(3):
        signal=ndimage.map_coordinates(rgb[...,channel],[yy,xx],order=1)
        rise=np.gradient(signal,.5,axis=1)
        strongest=rise.argmax(axis=1)
        # Require an outward rise confined to the measured boundary. A
        # brighter halo farther out cannot witness a missing control edge.
        step=np.median(signal[:,-4:],axis=1)-np.median(signal[:,:4],axis=1)
        local=(abs(offsets[strongest])<=tolerance)&(rise.max(axis=1)>1.5)&(step>5.)
        witnesses.append(local)
    return np.sum(witnesses,axis=0)>=2


def filled_control(image, minimum_y, *, channel_witnesses=False):
    a = np.asarray(image.convert('RGBA'))
    h,w = a.shape[:2]
    rgb = cv2.GaussianBlur(a[..., :3].astype(np.float32), (0,0), 1.)
    x = np.arange(w)
    left = np.median(rgb[:,int(.20*w):int(.27*w)],axis=1)
    right = np.median(rgb[:,int(.73*w):int(.80*w)],axis=1)
    mix = np.clip((x/w-.235)/.53,0,1)[None,:,None]
    metal = left[:,None,:]*(1-mix)+right[:,None,:]*mix
    diff = metal-rgb
    roi = (a[...,3]>=240)&(x>.24*w)&(x<.76*w)
    roi[:minimum_y]=False
    roi[int(.85*h):]=False
    darkness = diff @ [.299,.587,.114]
    def components(binary):
        contours = cv2.findContours(binary.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours:
            x0,y,bw,bh = cv2.boundingRect(contour)
            if (.2*w<bw<.65*w and .035*h<bh<.23*h and 2<bh/bw<9
                    and .4*w<x0+bw/2<.6*w and y>minimum_y and y+bh<.85*h
                    and cv2.contourArea(contour)>.80*bw*bh):
                yield contour,(x0,y,bw,bh)
    seeds = []
    for threshold in (6,12,20,32,48):
        for _,box in components(roi & (darkness>threshold)):
            if not any(np.max(np.abs(np.subtract(box,old)))<8 for old in seeds):
                seeds.append(box)
    edges = cv2.Canny(np.asarray(rgb,dtype=np.uint8),4,12)
    distance = ndimage.distance_transform_edt(edges==0)
    tolerance = max(1.5,.015*w)
    if channel_witnesses:
        # A single multichannel Canny map can switch to an adjacent metallic
        # reflection. Require independent native-channel edges at the mask.
        channel_distances=[ndimage.distance_transform_edt(cv2.Canny(plane,4,12)==0)
                           for plane in cv2.split(rgb.astype(np.uint8))]
    accepted=[]
    for x0,y,bw,bh in seeds:
        center=int(x0+bw/2)
        interior = diff[y+bh//4:y+3*bh//4,center-bw//6:center+bw//6]
        step = np.median(interior,axis=(0,1))
        if step @ [.299,.587,.114] < 6:
            continue
        projection = (diff@step)/float(step@step)
        local = np.zeros_like(roi)
        local[max(0,y-12):y+bh+12, max(0,x0-12):x0+bw+12]=True
        measurements=[]
        for level in (.45,.50,.55):
            binary = roi & local & (projection>level)
            found = list(components(binary))
            if len(found)!=1:
                break
            contour,box=found[0]
            x1,y1,width,height=box
            if np.max(abs(np.subtract(box,(x0,y,bw,bh))))>10:
                break
            mask=np.zeros((h,w),np.uint8)
            cv2.drawContours(mask,[contour],-1,1,-1)
            mask=mask.astype(bool)
            if np.any(mask & (a[...,3]<240)) or binary[mask].mean()<.98:
                break
            mask=(_fit_control_edges(mask,channel_distances,rgb,step,projection) if channel_witnesses else _refine_boundary(mask,rgb,step))
            x1,y1,width,height=cv2.boundingRect(mask.astype(np.uint8))
            if np.any(mask&(a[...,3]<240)):
                break
            shape=mask[y1:y1+height,x1:x1+width]
            yy,xx=np.indices(shape.shape);r=(width-1)/2
            cap=(xx-r)**2+(yy-np.clip(yy,r,height-1-r))**2<=r*r
            iou=(shape&cap).sum()/(shape|cap).sum()
            k=max(2,int(width*.15))
            corners=(shape[:k,:k],shape[-k:,:k],shape[:k,-k:],shape[-k:,-k:])
            if iou<.90 or max(part.mean() for part in corners)>.25:
                break
            yy,xx=np.nonzero(mask & ~ndimage.binary_erosion(mask))
            supported=distance[yy,xx]<=tolerance
            if channel_witnesses:
                nearby=np.stack([d[yy,xx]<=tolerance for d in channel_distances])
                supported |= (nearby.sum(axis=0)>=2) | _native_normal_edges(mask,rgb,yy,xx,tolerance)
            parts=(xx<x1+width*.25,xx>x1+width*.75,yy<y1+r,yy>y1+height-r)
            support=[float(np.mean(supported[part])) for part in parts]
            if min(support)<.75 or supported.mean()<.95:
                break
            measurements.append((mask,dict(box=[x1,y1,x1+width,y1+height],pixels=int(mask.sum()),
                capsule_iou=float(min(1,iou)),edge_support=support,boundary_support=float(supported.mean()),
                proposal_method='filled_control_independent_channel_edges' if channel_witnesses else 'filled_control_source_rgb_half_step')))
        if len(measurements)!=3:
            continue
        if np.ptp([proof['box'] for _,proof in measurements],axis=0).max()>2*tolerance:
            continue
        mask,proof=measurements[1]
        proof['source_threshold_boxes']=[q['box'] for _,q in measurements]
        proof['interior_to_metal_rgb']=step.tolist()
        accepted.append((mask,proof))
    return accepted
