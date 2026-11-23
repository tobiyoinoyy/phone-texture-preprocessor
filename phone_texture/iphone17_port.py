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


def _measure(image, thresholds, filter_print=False):
    a=np.asarray(image.convert('RGBA'));h,w=a.shape[:2]
    gray=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY)
    readings=[]
    for threshold in thresholds:
        n,labels,stats,centers=cv2.connectedComponentsWithStats(((gray<threshold)&(a[...,3]>=128)).astype('uint8'),8)
        ports=[];circles=[]
        for box,centroid in zip(stats[1:],centers[1:]):
            x,y,bw,bh,area=box;cx=x+(bw-1)/2;cy=y+(bh-1)/2
            if .4*w<cx<.6*w and .08*w<bw<.18*w and .18*h<bh<.5*h and 2<bw/bh<5:
                ports.append([cx,cy,int(x),int(y),int(bw),int(bh)])
            if (.17*w<cx<.85*w and .3*h<cy<.7*h and .011*w<bw<.03*w and
                .08*h<bh<.3*h and .65<bw/bh<1.5 and area>40):
                circles.append([cx,cy,bw,bh])
        if len(ports)!=1:raise ValueError('Standard USB-C port has no threshold consensus')
        if filter_print and circles:
            # Eight collinear bores share a diameter. Ink fragments on the nearby
            # label can cross the same threshold, but lack this joint support.
            diameter = float(np.median([min(c[2:]) for c in circles]))
            axis = float(np.median([c[1] for c in circles]))
            circles = [c for c in circles if min(c[2:]) >= diameter*.82
                       and abs(c[1]-axis) <= diameter*.20]
        circles = [c[:2] for c in circles]
        pc=ports[0];left=sorted([p for p in circles if p[0]<pc[0]-.12*w]);right=sorted([p for p in circles if p[0]>pc[0]+.12*w])
        if sorted([len(left),len(right)])!=[3,5]:raise ValueError('Standard bottom requires the complete asymmetric 3/5-hole groups')
        holes=np.asarray(left+right)
        if any(np.std(np.diff(np.asarray(group)[:,0]))>1.5 for group in (left,right)):
            raise ValueError('Standard hole pitch is inconsistent')
        screws=sorted([p for p in circles if .06*w<abs(p[0]-pc[0])<.12*w])
        if not screws or len({p[0]<pc[0] for p in screws}) != len(screws):
            raise ValueError('Standard screw missing or ambiguous')
        readings.append(dict(port=pc,holes=holes,screw=screws))
    holes=np.median(np.asarray([r['holes'] for r in readings]),axis=0)
    port=np.median(np.asarray([r['port'] for r in readings]),axis=0)
    if np.ptp(np.asarray([r['holes'] for r in readings]),axis=0).max()>3:
        raise ValueError('Standard hole coordinates are unstable')
    mirrored=int(np.sum(holes[:,0]<port[0]))==5
    nominal=np.sort(WIDTH_MM-DRAWING_HOLES) if mirrored else DRAWING_HOLES.copy()
    screw_rows=[r['screw'] for r in readings]
    if len({len(row) for row in screw_rows}) != 1 or np.ptp(screw_rows,axis=0).max()>2:
        raise ValueError('Standard screw coordinates are unstable')
    return dict(holes=holes,nominal=nominal,port=port,screws=np.median(screw_rows,axis=0),mirrored=mirrored)


def _dark_rail(gray):
    h, w = gray.shape
    metal = gray[int(.35*h):int(.65*h), int(.12*w):int(.18*w)]
    return metal.size > 0 and np.median(metal) < 140


def measure(image):
    from .bottom_rectify import port_rounded_rect
    gray = np.asarray(image.convert('L'))
    runs = []
    run = []
    for level in range(5, 255, 5):
        try:
            single = _measure(image, (level,), filter_print=True)
        except ValueError:
            run = []
            continue
        run.append((level, single))
        if len(run) < 4:
            continue
        window = run[-4:]
        samples = [m for _, m in window]
        if (len({len(m['screws']) for m in samples}) != 1
                or any(np.ptp([m[k] for m in samples],axis=0).max()>limit
                       for k,limit in [('port',2),('holes',2),('screws',2)])):
            continue
        runs.append(window)
    # Prefer the most stable complete layout. Distinct viable hole rows
    # cannot be resolved by choosing a more convenient threshold.
    if not runs:
        raise ValueError('No stable grayscale interval for asymmetric holes, screw and USB-C rim')
    reference = runs[0][0][1]
    if any(m['mirrored'] != reference['mirrored'] or
           abs(m['holes'] - reference['holes']).max() > 3 for window in runs for _, m in window):
        raise ValueError('Competing stable asymmetric hole layouts')
    runs.sort(key=lambda window: np.ptp([m['port'] for _,m in window],axis=0).max())
    tried=set()
    for window in runs:
        levels, samples = zip(*window)
        proposals = [m['port'] for m in samples]
        seed=tuple(np.median(proposals,axis=0))
        if seed in tried:
            continue
        tried.add(seed)
        try:
            port, spread, _, evidence = port_rounded_rect(gray, proposals, image.width/1000)
        except ValueError:
            continue
        if spread <= 2 and abs(port[0]-seed[0]) <= 2:
            result = _measure(image, levels, filter_print=True)
            result.update(port=port, selected_thresholds=list(levels), port_contour_evidence=evidence)
            return result
    raise ValueError('No independently supported complete USB-C rim')


def mapping(image):
    try:
        return _component_mapping(image)
    except ValueError as first_error:
        from . import bottom_rectify
        try:
            mx, _, _, check, measurements = bottom_rectify.plan(
                np.asarray(image.convert('RGBA')), 'iphone17', allow_local_contrast=_dark_rail(np.asarray(image.convert('L'))))
        except bottom_rectify.Rejected as error:
            raise ValueError(str(first_error) + ' (' + str(error) + ')') from error
        holdout = max(abs(v) for v in check['holdout_only_errors_px'])
        if (holdout > 1.5 or check['hole_fit_max_px'] > 1.5
                or abs(check['port_holdout_error_px']) > 2.5
                or max(map(abs, check['screw_holdout_errors_px'])) > 2.5):
            raise first_error
        center = float(np.interp(measurements['port'][0], mx[0], np.arange(image.width)))
        if abs(center - (image.width-1)/2) > 2.5:
            raise first_error
        check.update(source_visible_screw_sides=['left' if s[0]<measurements['port'][0] else 'right'
                                                 for s in measurements['screws']],
                     reference=APPLE_DRAWING, reference_sheet=1, port_after_x=center,
                     holdout_holes_max_error_px=holdout, measurement_method='independent_whole_port',
                     source_sampling_scale_range=check['sampling_derivative_range'])
        return mx[0], check


def final_port(image, proof):
    from .bottom_rectify import PROFILES, measure as measure_groups
    a = np.asarray(image.convert('RGBA'))
    causes = {}
    nominal = np.asarray(proof['hole_nominal_mm'])
    px_per_mm = (image.width - 1) / WIDTH_MM
    for rendered_contrast in ('adaptive_gray', 'raw', 'local'):
        try:
            if rendered_contrast == 'adaptive_gray':
                measured = measure(image)
            else:
                measured = measure_groups(a, 'iphone17', rendered_contrast, rendered=True)
        except ValueError as error:
            causes[rendered_contrast] = str(error)
            continue
        if not np.array_equal(measured['nominal'], nominal):
            causes[rendered_contrast] = 'hole_orientation_changed'
            continue
        offsets = measured['holes'][:, 0] - nominal * px_per_mm
        if len(offsets) != 8 or np.max(abs(offsets)) > 1.5:
            causes[rendered_contrast] = 'rendered_holes_outside_tolerance'
            continue
        screws = np.asarray(PROFILES['iphone17']['screws'])
        if measured['mirrored']:
            screws = np.sort(WIDTH_MM - screws)
        checks = []
        for side, coordinate in zip(('left', 'right'), screws):
            found = [s[0] for s in measured['screws']
                     if (s[0] < measured['port'][0]) == (side == 'left')]
            target = float(coordinate * px_per_mm)
            error = float(found[0] - target) if len(found) == 1 else None
            status = ('not_observed' if not found else 'ambiguous' if len(found) > 1 else
                      'confirmed' if abs(error) <= 2.5 else 'outside_tolerance')
            checks.append(dict(side=side, target_x=target, measured_x=float(found[0]) if len(found) == 1 else None,
                               error_px=error, status=status))
        statuses = {c['status'] for c in checks}
        confirmed = {c['side'] for c in checks if c['status']=='confirmed'}
        if (not confirmed or not set(proof.get('source_visible_screw_sides', ())) <= confirmed
                or not statuses <= {'confirmed', 'not_observed'}):
            causes[rendered_contrast] = 'rendered_screws_outside_tolerance_or_ambiguous'
            continue
        proof.update(rendered_port_method='whole_rounded_port_with_independent_holes_and_screws',
                     rendered_hole_errors_px=offsets.tolist(), rendered_screw_checks=checks,
                     rendered_hole_contrast=rendered_contrast,
                     rendered_hole_thresholds=measured['selected_thresholds'],
                     rendered_port_contour=measured['port_contour_evidence'])
        return measured['port']
    raise ValueError(f'Final USB-C witnesses: {causes}')


def verify_export(path, proof):
    from .workspace import sha
    with Image.open(path) as image:
        port = final_port(image, proof)
        error = float(port[0] - (image.width-1)/2)
    if abs(error)>2.5:
        raise ValueError('Final rendered USB-C centre failed independent measurement')
    proof.update(final_measured_port_x=float(port[0]), final_rendered_center_error_px=error,
                 final_verified_sha256=sha(path))


def _component_mapping(image):
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
    from .bottom_rectify import PROFILES
    screws=np.asarray(PROFILES['iphone17']['screws'])
    if m['mirrored']:
        screws=np.sort(WIDTH_MM-screws)
    errors=[float(s[0]-project(fit.x,screws[int(s[0]>m['port'][0])])) for s in m['screws']]
    if max(map(abs,errors))>2.5:
        raise ValueError('Independent screw holdout rejects standard bottom fit')
    proof.update(source_visible_screw_sides=['left' if s[0]<m['port'][0] else 'right' for s in m['screws']],
                 selected_thresholds=m['selected_thresholds'], port_contour_evidence=m['port_contour_evidence'],
                 visible_screw_count=len(errors), screw_holdout_errors_px=errors)
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
