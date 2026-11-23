"""Model-calibrated standard 17 frame ends and dual-lens texture registration."""
import contextlib
import io
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from . import workspace as w
from .frame_seams import resized_chains, seam_engine
from . import lighting

PROFILE = Path(__file__).with_name('seam_profiles')/'iphone17.json'
FRAMES = ('top','bottom','left','right')


def ease(value):
    t = np.clip(value,0,1)
    return t*t*(3-2*t)


def aligned_metal(work, face, size, *, review_alignment=None):
    material = work/'material-inputs'/face
    mask = np.asarray(Image.open(material/'main_surface.png').convert('L'))
    if mask.shape != size[::-1]:
        raise ValueError('Metal mask dimensions differ from the aligned frame')
    if review_alignment is not None and face not in review_alignment:
        return mask >= 254
    if face == 'bottom':
        mx = (np.load(work/'bottom-source-x.npy', allow_pickle=False) if review_alignment is None
              else review_alignment[face])
        if mx.shape != (size[0],):
            raise ValueError('Missing aligned bottom mask mapping')
        mask = ndimage.map_coordinates(mask.astype(float),
            np.broadcast_arrays(np.arange(size[1])[:,None], mx[None,:]),order=1)
    elif face in ('left','right'):
        if review_alignment is None:
            spec = w.read(work/'structure-proposal.json')[face]
            checks = w.read(work/'structure-checks.json')[face]
        else:
            spec, checks = review_alignment[face]
        source, target = np.array(spec['source']), np.array(spec['target'])
        source[[1,2,5,6]] += np.array(checks['sampling_correction'])[[1,2,5,6]]
        rows = np.arange(size[1])
        mask = np.stack([np.interp(np.interp(rows,target[:,x],source[:,x]),rows,mask[:,x])
                         for x in range(size[0])],axis=1)
    return mask >= 254



def _current_seam_readings(files, chains, masks, sigma):
    engine=seam_engine()
    low={}
    for face in FRAMES:
        rgba=np.asarray(Image.open(files[face]).convert('RGBA'))
        fit=masks[face] & (rgba[...,3]>=240)
        den=engine.gaussian(fit.astype(float),sigma)
        lab=engine.rgb_to_lab(rgba[...,:3])
        low[face]=engine.gaussian(lab*fit[...,None],sigma)/np.maximum(den[...,None],1e-8)
    readings=[]
    for chain in chains:
        samples=[]
        valid=np.ones(192,dtype=bool)
        arc=chain['arc']
        for side in chain['sides']:
            name=side['name']
            positions=np.asarray(side['points'])
            points=np.column_stack([np.interp(np.linspace(0,arc[-1],192),arc,axis)
                                    for axis in positions.T])
            samples.append(engine.sample(low[name],points))
            valid &= engine.sample(masks[name][...,None],points)[:,0]>.98
        if valid.mean()<.98:
            raise ValueError('Current model seam lacks jointly editable photographed metal')
        delta=np.linalg.norm(samples[0]-samples[1],axis=1)[valid]
        readings.append(dict(pair=chain['names'],edge_ids=chain['edges'],
            joint_coverage=float(valid.mean()),metric_samples=int(valid.sum()),total_samples=len(valid),
            after_p95_deltaE76=float(np.quantile(delta,.95)),after_mean_deltaE76=float(delta.mean())))
    return readings

def finish_frames(files, work, output, records, *, review_alignment=None):
    # Use the same source-bound domains and registration as the approved images.
    # Only terminal metal participates. Buttons, paper, holes and antennas are
    # separated from the blend even where they approach an end.
    profile = w.read(PROFILE)
    records.mkdir(parents=True,exist_ok=False)
    helper = lighting.engine('aluminum')
    match_uv_seams = seam_engine()
    views, references = {}, []
    arrays, masks, coords = {}, {}, {}
    for face in FRAMES:
        a = np.asarray(Image.open(files[face]).convert('RGBA'))
        h,width = a.shape[:2]
        mask = aligned_metal(work,face,(width,h),review_alignment=review_alignment) & (a[...,3]>0)
        y,x = np.ogrid[:h,:width]
        along,across,length,thickness = (x,y,width,h) if width>h else (y,x,h,width)
        distance = np.minimum(along,length-1-along)/thickness
        # Sample nearby uniform metal after the corner darkening on both ends.
        samples = (distance>1.05)&(distance<1.65)&(across>.25*thickness)&(across<.75*thickness)&mask&(a[...,3]>=240)
        lab = helper.rgb_to_lab(a[...,:3])
        for end in (along<length/2,along>=length/2):
            values=lab[...,0][samples&end]
            if len(values)<60:
                raise ValueError('Insufficient plain metal beside frame end: '+face)
            references.append(float(np.median(values)))
        arrays[face],masks[face],coords[face] = (a,lab),mask,distance
    source_hashes = {f: w.sha(files[f]) for f in FRAMES}
    current_chains=resized_chains(profile,files)
    input_readings=_current_seam_readings(files,current_chains,masks,2) if review_alignment is not None else None
    target=float(np.median(references))
    if review_alignment is not None and all(r['after_p95_deltaE76']<=2 for r in input_readings):
        output.mkdir(parents=True)
        for face in FRAMES:
            shutil.copy2(files[face],output/(face+'.png'))
            Image.fromarray(masks[face].astype('uint8')*255).save(records/(face+'-metal.png'))
        saved_hashes = {f: w.sha(output/(f+'.png')) for f in FRAMES}
        if source_hashes != saved_hashes:
            raise ValueError('Frame copy differs from the measured source')
        return dict(input_sha256=source_hashes, output_sha256=saved_hashes, seams=input_readings,input_seams=input_readings,already_matched=True,
                    model_sha256=profile['model_sha256'],profile_sha256=w.sha(PROFILE))
    for face in FRAMES:
        a,lab = arrays[face]
        mask,distance = masks[face],coords[face]
        fit=mask&(a[...,3]>=240)
        den=ndimage.gaussian_filter(fit.astype(float),3.)
        low=ndimage.gaussian_filter(lab[...,0]*fit,3.)/np.maximum(den,1e-8)
        blend=ease((1.45-distance)/.72)
        active=mask&(blend>0)
        if np.any(den[active]<.06):
            broad_den=ndimage.gaussian_filter(fit.astype(float),9.)
            if np.any((den[active]<.06)&(broad_den[active]<.008)):
                raise ValueError('Frame end lacks metal lighting support: '+face)
            broad=ndimage.gaussian_filter(lab[...,0]*fit,9.)/np.maximum(broad_den,1e-8)
            low=broad+(low-broad)*ease((den-.008)/.052)
        corrected=lab.copy()
        corrected[...,0]+=np.clip((target-low)*.94,-30,65)*blend
        out=a.copy()
        out[active,:3]=helper.lab_to_rgb(corrected)[active]
        src=records/(face+'-ends.png')
        Image.fromarray(out).save(src)
        mp=records/(face+'-metal.png')
        Image.fromarray((mask*255).astype('uint8')).save(mp)
        pp=records/(face+'-excluded.png')
        Image.fromarray((~mask*255).astype('uint8')).save(pp)
        views[face]=dict(source=str(src),editable_mask=str(mp),protected_mask=str(pp))
    baseline=w.read(PROFILE.with_name('iphone17-frame-baseline.json'))
    w.save(records/'chains.json',dict(chains=resized_chains(baseline,files)))
    w.save(records/'scene.json',{})
    job=dict(scene=str(records/'scene.json'),seam_chains=str(records/'chains.json'),views=views,
             output_dir=str(records/'baseline-frames'),records_dir=str(records/'engine'),width_units=.18,
             guard_pixels=18,sigma=2,protection_feather=0,max_lab_shift=[28,16,16],
             minimum_joint_coverage=.98,maximum_p95_deltaE76=2.,sampling_refinements=2,
             maximum_correction_step_deltaE76=3.,continuity_retry_sigmas=[2,4],
             continuity_warning_limit_deltaE76=8.)
    w.save(records/'job.json',job)
    effect_warnings = []
    # Preserve the reviewed frame lighting wherever it already matches the
    # current mesh. The previous seam layout supplies a starting image only.
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            baseline_checks = match_uv_seams.run(records/'job.json')
        resolution = baseline_checks.get('continuity_resolution')
        if resolution and 'warning_limit' in resolution:
            effect_warnings.append(dict(resolution, stage='baseline'))
        starting={face:records/'baseline-frames'/(face+'.png') for face in FRAMES}
    except ValueError as exc:
        w.save(records/'baseline-warning.json',dict(reason=str(exc)))
        starting={face:Path(views[face]['source']) for face in FRAMES}
    chains=current_chains
    before=_current_seam_readings(starting,chains,masks,job['sigma'])
    needed=[chain for chain,check in zip(chains,before) if check['after_p95_deltaE76']>job['maximum_p95_deltaE76']]
    if needed:
        current_views={face:dict(view,source=str(starting[face])) for face,view in views.items()}
        w.save(records/'current-chains.json',dict(chains=needed))
        current=dict(job,views=current_views,seam_chains=str(records/'current-chains.json'),
                     output_dir=str(output),records_dir=str(records/'current-engine'))
        w.save(records/'current-job.json',current)
        with contextlib.redirect_stdout(io.StringIO()):
            current_checks = match_uv_seams.run(records/'current-job.json')
        resolution = current_checks.get('continuity_resolution')
        if resolution and 'warning_limit' in resolution:
            effect_warnings.append(dict(resolution, stage='current'))
        after=_current_seam_readings({f:output/(f+'.png') for f in FRAMES},chains,masks,job['sigma'])
    else:
        output.mkdir(parents=True,exist_ok=True)
        for face,path in starting.items():
            shutil.copy2(path,output/(face+'.png'))
        after=[dict(entry) for entry in before]
    for first,last in zip(before,after):
        last.update(before_p95_deltaE76=first['after_p95_deltaE76'],before_mean_deltaE76=first['after_mean_deltaE76'])
        if last['after_p95_deltaE76']>job['maximum_p95_deltaE76']:
            raise ValueError('Final frame fails the current model seam colour check')
    for face in FRAMES:
        a,_=arrays[face]
        out=np.asarray(Image.open(output/(face+'.png')))
        if not np.array_equal(out[...,3],a[...,3]) or not np.array_equal(out[~masks[face]],a[~masks[face]]):
            raise ValueError('Frame finish changed geometry or nonmetal: '+face)
    result = dict(seams=after, reference_L=target, model_sha256=profile['model_sha256'],
                  profile_sha256=w.sha(PROFILE), input_sha256=source_hashes,
                  output_sha256={f: w.sha(output/(f+'.png')) for f in FRAMES})
    if effect_warnings:
        result['continuity_warnings'] = effect_warnings
    if input_readings is not None:
        result['input_seams'] = input_readings
    return result


def verified_frame_finish(report, hashes):
    if not report or not set(FRAMES).issubset(hashes):
        return False
    profile = w.read(PROFILE)
    if (report.get('model_sha256') != profile['model_sha256']
            or report.get('profile_sha256') != w.sha(PROFILE)
            or report.get('output_sha256') != {f: hashes.get(f) for f in FRAMES}):
        return False
    checks = report.get('seams', [])
    expected = {frozenset(chain['names']) for chain in profile['chains']}
    return (len(checks) == len(expected)
            and {frozenset(c.get('pair', ())) for c in checks} == expected
            and all(0 <= c.get('after_p95_deltaE76', float('inf')) <= 2.
                    and .98 <= c.get('joint_coverage', 0) <= 1
                    and c.get('total_samples') == 192
                    and 189 <= c.get('metric_samples', 0) <= 192 for c in checks))



def _rim_ellipse(contour,w,h):
    if len(contour)<10:
        return None
    (x,y),(dx,dy),_=cv2.fitEllipse(contour)
    radius=(dx+dy)/4
    if (.072*w<radius<.12*w and .89<dx/dy<1.12 and .10*w<x<.27*w and
            .035*h<y<.24*h and .97<cv2.contourArea(contour)/(np.pi*radius**2)<1.03):
        return np.array([x,y,radius])
    return None


def _photographed_lens_rings(roi,alpha,w,h):
    # The dark glass merges with a dark camera platform. A complete bright
    # annulus still has two independently closed, approximately concentric rims.
    readings=[]
    for threshold in range(35,116,5):
        bright=(roi>threshold)&(alpha>=240)
        contours,tree=cv2.findContours(bright.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        pair=[]
        for idx,contour in enumerate(contours):
            parent=tree[0,idx,3]
            if parent<0:continue
            inner=_rim_ellipse(contour,w,h)
            if inner is None:continue
            outer=_rim_ellipse(contours[parent],w,h)
            if outer is None:continue
            if (.055*inner[2]<outer[2]-inner[2]<.21*inner[2] and
                    np.linalg.norm(inner[:2]-outer[:2])<.035*inner[2]):
                pair.append(inner)
        if len(pair)==2:
            readings.append(sorted(pair,key=lambda v:v[1]))
    if len(readings)<4:
        raise ValueError('Both camera rings require closed inner and outer contours')
    tracks=np.asarray(readings)
    cluster=max((tracks[np.max(abs(tracks-r),axis=(1,2))<1.4*w/988] for r in tracks),key=len)
    if len(cluster)<4:
        raise ValueError('Camera ring boundaries disagree across brightness levels')
    # Re-read the original subpixel transition in independent angular sectors.
    angles=np.arange(240)*2*np.pi/240
    step=.25
    result=[]
    for x,y,r in np.median(cluster,axis=0):
        radii=np.arange(r-5,r+5+step,step)
        xs=x+np.cos(angles)[:,None]*radii
        ys=y+np.sin(angles)[:,None]*radii
        signal=ndimage.map_coordinates(roi.astype(float),[ys,xs],order=1)
        slope=np.gradient(ndimage.gaussian_filter1d(signal,2,axis=1),axis=1)
        k=slope.argmax(axis=1)
        good=(k>1)&(k<len(radii)-2)&(slope[np.arange(len(k)),k]>1.)
        if min(np.mean(part) for part in np.array_split(good,8))<.65:
            raise ValueError('Camera glass lacks an independent edge in every sector')
        points=np.column_stack((xs[np.arange(len(k)),k],ys[np.arange(len(k)),k])).astype('float32')
        fits=[]
        for group in (np.arange(len(k))%2==0,np.arange(len(k))%2==1):
            (cx,cy),(dx,dy),_=cv2.fitEllipse(points[good&group])
            fits.append([cx,cy,(dx+dy)/4])
        if np.max(np.ptp(fits,axis=0))>.65*w/988:
            raise ValueError('Independent camera glass contours disagree')
        result.append(np.mean(fits,axis=0))
    return np.asarray(result)

def lens_pair(a):
    h,width=a.shape[:2]
    gray=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY)
    roi=gray[:round(.27*h),:round(.38*width)]
    readings=[]
    for t in (30,36,43,50,57,64,72,80,88):
        dark=(roi<t)&(a[:roi.shape[0],:roi.shape[1],3]>=240)
        contours,_=cv2.findContours(dark.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        pair=[]
        for c in contours:
            if len(c)<5:
                continue
            (x,y),(dx,dy),_=cv2.fitEllipse(c)
            rad=(dx+dy)/4
            if (.073*width<rad<.112*width and .86<dx/dy<1.16 and .10*width<x<.27*width
                    and .035*h<y<.24*h and .94<cv2.contourArea(c)/(np.pi*rad**2)<1.05):
                pair.append([x,y,rad])
        if len(pair)==2:
            readings.append(sorted(pair,key=lambda v:v[1]))
    # Select a consistent cluster, rejecting ring/step contours when a higher
    # threshold merges the lens glass with its surround.
    samples=np.array(readings)
    groups=[samples[(np.max(abs(samples-s),axis=(1,2))<=1.4*width/988)] for s in samples]
    group=max(groups,key=len,default=[])
    if len(group)>=3:
        result=np.median(group,axis=0)
    else:
        try:
            result=_photographed_lens_rings(roi,a[:roi.shape[0],:roi.shape[1],3],width,h)
        except ValueError as exc:
            raise ValueError('Two separate standard lenses lack source support: '+str(exc)) from exc
    if abs(result[1,0]-result[0,0])>.15*result[0,2] or not 2.2<(result[1,1]-result[0,1])/result[:,2].mean()<2.8:
        raise ValueError('Standard vertical lens layout is unresolved')
    return result


def register_camera(image, *, center_tolerance=.65, measure=lens_pair):
    profile=w.read(PROFILE)['camera']
    a=np.asarray(image.convert('RGBA'));h,width=a.shape[:2]
    scale=np.array([width,h])/profile['reference_size']
    target=(np.array(profile['centers_px'])+.5)*scale-.5
    measured=measure(a)
    radius=np.asarray(profile['radii_px'])*scale[0]
    shift=measured[:,:2]-target
    if np.max(np.linalg.norm(shift,axis=1)/radius)>.23 or np.max(abs(measured[:,2]/radius-1))>.06:
        raise ValueError('Lens correction exceeds supported local registration')
    if (np.max(np.linalg.norm(shift,axis=1))/np.mean(scale)<=center_tolerance
            and np.max(abs(measured[:,2]-radius))/scale[0]<=.9):
        return image.copy(),dict(before=measured.tolist(),after=measured.tolist(),
            target=target.tolist(),model_reference_error_px=(np.linalg.norm(shift,axis=1)/np.mean(scale)).tolist(),
            jacobian=[1.,1.])
    # Both disks move with their photographed rings. Smoothly join the shifts
    # between the two lenses, keeping the outer body and flash fixed.
    y,x=np.mgrid[:int(.32*h),:int(.405*width)]
    blend=ease((y-(target[0,1]+radius[0]))/(target[1,1]-radius[1]-target[0,1]-radius[0]))
    dx=np.zeros_like(x,dtype=float);dy=dx.copy()
    for i,k in enumerate((1-blend,blend)):
        ratio=measured[i,2]/radius[i]-1
        dx+=k*(shift[i,0]+ratio*(x-target[i,0]))
        dy+=k*(shift[i,1]+ratio*(y-target[i,1]))
    cy=np.clip(y,target[0,1],target[1,1])
    cx=np.interp(cy,target[:,1],target[:,0])
    dist=np.hypot(x-cx,y-cy)
    body=ndimage.distance_transform_edt(np.pad(a[...,3]>=250,1))[1:-1,1:-1]
    contour_weight=ease((body[:y.shape[0],:y.shape[1]]-18*scale[0])/(52*scale[0]))
    # Keep the same outer footprint. If the original transition compresses
    # glass too much, begin its taper closer to the ring on the same photo.
    radial=dist/radius.mean()
    widths=[.47]
    inner=max(1.17,1.15*max(radius)/radius.mean())
    if inner < 1.27:
        widths.append(1.74-inner)
    if center_tolerance < .65 and 1.13 * radius.mean() >= 1.10 * max(radius):
        # Keep a ten-percent margin beyond both glass disks and taper within
        # the same footprint. The original Jacobian and contour limits apply.
        widths.append(.61)
    for width_ratio in widths:
        weight=ease((1.74-radial)/width_ratio)*contour_weight
        mx=(x+dx*weight).astype('float32');my=(y+dy*weight).astype('float32')
        jac=np.gradient(mx,axis=1)*np.gradient(my,axis=0)-np.gradient(mx,axis=0)*np.gradient(my,axis=1)
        if jac.min()>=.50 and jac.max()<=1.55:
            break
    else:
        raise ValueError('Camera warp would distort the glass transition')
    changed=weight>0
    sample=cv2.remap(a,mx,my,cv2.INTER_CUBIC)
    if not (sample[changed,3]==255).all() or not (a[:y.shape[0],:y.shape[1]][changed,3]==255).all():
        raise ValueError('Camera correction would touch transparent body contour')
    out=a.copy()
    out[:y.shape[0],:y.shape[1]][changed]=sample[changed]
    actual=measure(out)
    residual=np.linalg.norm(actual[:,:2]-target,axis=1)/np.mean(scale)
    if residual.max()>center_tolerance or np.max(abs(actual[:,2]-radius))/scale[0]>.9:
        raise ValueError('Rendered standard lenses fail model UV reference check')
    return Image.fromarray(out),dict(before=measured.tolist(),target=target.tolist(),
                  after=actual.tolist(),model_reference_error_px=residual.tolist(),
                  jacobian=[float(jac.min()),float(jac.max())])
