"""Source-supported bottom-plane registration using standard-model hole sites.

The eight asymmetric holes fit the planar projectivity. Port and visible screw
are holdouts. Native rounded endcaps remain fixed: this is not a reconstruction
of an unobserved curved surface, nor a metric correction of the whole phone.
"""
import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares
from PIL import Image

APPLE_DRAWING='https://developer.apple.com/download/files/accessories/dimensional-drawings/iphone-17.pdf'
WIDTH_MM=71.45
# Sheet 1, bottom view: third-angle diagram is reversed relative to this
# rear-facing capture. The count pattern independently establishes orientation.
DRAWING_HOLES=np.array([19.71,21.965,24.22,47.23,49.485,51.74,53.995,56.25])


def measure(image):
    a=np.asarray(image.convert('RGBA'));h,w=a.shape[:2]
    gray=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY)
    readings=[]
    for threshold in (80,100,120):
        n,labels,stats,centers=cv2.connectedComponentsWithStats(((gray<threshold)&(a[...,3]>=128)).astype('uint8'),8)
        ports=[];circles=[]
        for box,centroid in zip(stats[1:],centers[1:]):
            x,y,bw,bh,area=box;cx=x+(bw-1)/2;cy=y+(bh-1)/2
            if .4*w<cx<.6*w and .08*w<bw<.18*w and .18*h<bh<.5*h and 2<bw/bh<5:
                ports.append([cx,cy,int(x),int(y),int(bw),int(bh)])
            if (.17*w<cx<.85*w and .3*h<cy<.7*h and .011*w<bw<.03*w and
                .08*h<bh<.3*h and .65<bw/bh<1.5 and area>40):
                circles.append([cx,cy])
        if len(ports)!=1:raise ValueError('Standard USB-C port has no threshold consensus')
        pc=ports[0];left=sorted([p for p in circles if p[0]<pc[0]-.12*w]);right=sorted([p for p in circles if p[0]>pc[0]+.12*w])
        if sorted([len(left),len(right)])!=[3,5]:raise ValueError('Standard bottom requires the complete asymmetric 3/5-hole groups')
        holes=np.asarray(left+right)
        if any(np.std(np.diff(np.asarray(group)[:,0]))>1.5 for group in (left,right)):
            raise ValueError('Standard hole pitch is inconsistent')
        screws=[p for p in circles if pc[0]+.06*w<p[0]<pc[0]+.12*w]
        readings.append(dict(port=pc,holes=holes,screw=screws))
    holes=np.median(np.asarray([r['holes'] for r in readings]),axis=0)
    port=np.median(np.asarray([r['port'] for r in readings]),axis=0)
    if np.ptp(np.asarray([r['holes'] for r in readings]),axis=0).max()>3:
        raise ValueError('Standard hole coordinates are unstable')
    first=readings[0]
    mirrored=int(np.sum(holes[:,0]<port[0]))==5
    nominal=np.sort(WIDTH_MM-DRAWING_HOLES) if mirrored else DRAWING_HOLES.copy()
    screw=np.median(np.asarray([r['screw'][0] for r in readings if len(r['screw'])==1]),axis=0) if any(len(r['screw'])==1 for r in readings) else None
    return dict(holes=holes,nominal=nominal,port=port,screw=screw,mirrored=mirrored)


def mapping(image):
    m=measure(image);w,h=image.size;x=m['nominal'];observed=m['holes'][:,0]
    scale=float(np.median(np.diff(observed[:3]))/2.255)
    def project(p,points):return (p[0]*points+p[1])/(1+p[2]*points)
    fit=least_squares(lambda p:project(p,x)-observed,[scale,observed.mean()-scale*x.mean(),0],
        bounds=([scale*.7,-w,-.005],[scale*1.3,w,.005]))
    errors=project(fit.x,x)-observed
    if not fit.success or abs(errors).max()>1.5:raise ValueError('Standard bottom plane does not fit the independent holes')
    # Leave two holes out to test the fit, rather than accepting 3 fitted points.
    train=np.ones(8,bool);train[[1,6]]=False
    held=least_squares(lambda p:(project(p,x)-observed)[train],fit.x)
    hold_error=float(abs(project(held.x,x)-observed).max())
    if hold_error>1.5:raise ValueError('Held-out standard holes reject bottom projectivity')
    nominal_center=WIDTH_MM/2
    predicted=float(project(fit.x,nominal_center))
    port_residual=float(m['port'][0]-predicted)
    if abs(port_residual)>2.5:raise ValueError('Independent USB-C holdout rejects bottom-plane correction')
    cap=max(12,int(h*.6))
    targets=x/WIDTH_MM*(w-1)
    source=project(fit.x,x)
    dst=np.r_[0,cap,targets,w-1-cap,w-1]
    src=np.r_[0,cap,source,w-1-cap,w-1]
    if np.any(np.diff(dst)<=0) or np.any(np.diff(src)<=0):raise ValueError('Bottom cap-preserving map folds')
    mx=PchipInterpolator(dst,src)(np.arange(w)).astype('float32')
    jac=np.gradient(mx)
    if jac.min()<.65 or jac.max()>1.4:raise ValueError('Bottom correction would oversample/stretch source detail')
    out_port=float(np.interp(m['port'][0],mx,np.arange(w)))
    if abs(out_port-(w-1)/2)>2.5:raise ValueError('USB-C did not reach nominal centre within the source measurement tolerance')
    proof=dict(reference=APPLE_DRAWING,reference_sheet=1,profile='standard_iPhone17_asymmetric_3_5_holes',
        hole_source=observed.tolist(),hole_nominal_mm=x.tolist(),hole_fit_errors_px=errors.tolist(),
        holdout_holes_max_error_px=hold_error,projectivity=fit.x.tolist(),port_used_for_fitting=False,
        port_before_x=float(m['port'][0]),port_after_x=out_port,canvas_center_x=(w-1)/2,
        port_nominal_holdout_error_px=port_residual,source_sampling_scale_range=[float(jac.min()),float(jac.max())],
        native_endcaps_fixed_px=cap,unobserved_surface_generated=False,model_uv_verified=False,
        scope='bottom-plane image registration; photographed endcaps retained, not full curved-surface metric rectification')
    if m['screw'] is not None:
        nominal_screw=42.65 if m['mirrored'] else 42.65
        screw_error=float(m['screw'][0]-project(fit.x,nominal_screw))
        if abs(screw_error)>2.5:raise ValueError('Independent screw holdout rejects standard bottom fit')
        proof['screw_holdout_error_px']=screw_error
    return mx,proof


def apply(image,mx):
    a=np.asarray(image.convert('RGBA'));h,w=a.shape[:2]
    if mx.shape!=(w,) or mx.min()<0 or mx.max()>w-1:raise ValueError('Invalid standard bottom sampling map')
    mapx=np.broadcast_to(mx,(h,w)).copy();mapy=np.broadcast_to(np.arange(h,dtype='float32')[:,None],(h,w)).copy()
    f=a.astype('float32')/255;f[...,:3]*=f[...,3,None]
    q=cv2.remap(f,mapx,mapy,cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
    rgb=np.zeros_like(q[...,:3]);np.divide(q[...,:3],q[...,3,None],out=rgb,where=q[...,3,None]>1e-6)
    out=np.clip(np.rint(np.dstack([rgb,q[...,3]])*255),0,255).astype('uint8')
    unchanged=mx==np.arange(w)
    out[:,unchanged]=a[:,unchanged]
    return Image.fromarray(out)
