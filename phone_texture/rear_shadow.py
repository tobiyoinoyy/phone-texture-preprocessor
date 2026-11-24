"""Source-measured camera-ring shadows for the three-lens 17-series profile.

Geometry and material support are frozen before lighting. Correction operates
on the existing floating-point illumination result, never on a saved PNG.
"""
import cv2
import numpy as np
from scipy import ndimage

ANGLES = 360


class InsufficientRearShadowEvidence(ValueError):
    """The photo cannot support the optional camera-shadow adjustment."""


def _smoothstep(t):
    t=np.clip(t,0,1)
    return t*t*(3-2*t)


def _polar(ellipse):
    cx,cy,aa,bb,angle=ellipse;radius=(aa+bb)/4
    rho=np.arange(.98,1.90,.5/radius)
    theta=np.arange(ANGLES)*2*np.pi/ANGLES
    u=np.cos(theta)[:,None]*rho*aa/2;v=np.sin(theta)[:,None]*rho*bb/2
    phi=np.deg2rad(angle)
    return rho,cx+np.cos(phi)*u-np.sin(phi)*v,cy+np.sin(phi)*u+np.cos(phi)*v


def _plateau_core(L):
    """Conservative flat interior supported by three source edge windows.

    The ordinary camera material includes the step's bevel and its own shadow.
    Those differently oriented surfaces must not seed/receive lens shadows.
    This source-fitted rounded plateau limits an EXTRA operation; it never crops pixels
    or removes any area from the ordinary material's editable coverage.
    """
    h,w=L.shape;gy,gx=np.gradient(ndimage.gaussian_filter(L,1))
    definitions=[('left',gx,.035,.085,.06,.24,True),
                 ('right',gx,.90,.97,.10,.20,True),
                 ('top',gy,.015,.05,.40,.65,False),
                 ('bottom',gy,.24,.29,.57,.80,False)]
    edges={};proof={}
    for name,g,lo,hi,a,b,vertical in definitions:
        size=w if vertical else h;along=h if vertical else w
        data=g[int(a*along):int(b*along),:] if vertical else g[:,int(a*along):int(b*along)].T
        signal=np.median(data,axis=0);indices=np.arange(int(lo*size),int(hi*size))
        anchor=indices[np.argmax(abs(signal[indices]))]
        positions=[];strength=[]
        for window in np.array_split(data,3):
            s=np.median(window,axis=0);near=indices[abs(indices-anchor)<=4]
            position=int(near[np.argmax(abs(s[near]))]);positions.append(position);strength.append(float(abs(s[position])))
        if min(strength)<1.2 or np.ptp(positions)>4:
            raise InsufficientRearShadowEvidence('Rear shadow: plateau '+name+' lacks three-window source evidence')
        edges[name]=max(positions)+6 if name in ('left','top') else min(positions)-6
        proof[name]=dict(positions=positions,strengths=strength)
    bounds=[edges['left']-6,edges['top']-6,edges['right']+6,edges['bottom']+6]
    left,top,right,bottom=bounds;angles=np.linspace(.15,np.pi/2-.15,100)
    corners=[]
    for x,y,sx,sy in [(left,top,1,1),(right,top,-1,1),(left,bottom,1,-1),(right,bottom,-1,-1)]:
        radii=np.arange(int(w*.025),int(w*.20));scores=[]
        for radius in radii:
            X=x+sx*radius*(1-np.cos(angles));Y=y+sy*radius*(1-np.sin(angles))
            normal=(ndimage.map_coordinates(gx,[Y,X],order=1)*np.cos(angles)
                    +sx*sy*ndimage.map_coordinates(gy,[Y,X],order=1)*np.sin(angles))
            scores.append(float(np.mean(abs(normal))))
        index=int(np.argmax(scores))
        if scores[index]<2 or index in (0,len(radii)-1):
            raise InsufficientRearShadowEvidence('Rear shadow: plateau corner lacks source edge evidence')
        corners.append(dict(radius=int(radii[index]),source_edge_strength=scores[index]))
    proof['corners']=corners
    return bounds,proof


def prepare(a, material, source_sha256, material_sha256):
    """Measure three separate glass disks and their actual outer ring joint."""
    from .auto_material import rear_camera_glass
    from .back_lens import lens_centres
    from .lighting import engine
    h,w=a.shape[:2]
    glass,_=rear_camera_glass(a,a[...,3]>0)
    n,labels,stats,centres=cv2.connectedComponentsWithStats(glass.astype('uint8'),8)
    disks=[i for i in range(1,n) if .12*w<stats[i,2]<.3*w and .7<stats[i,2]/stats[i,3]<1.4]
    proof=dict(method='source_camera_ring_shadow_v2',source_sha256=source_sha256,
               material_sha256=material_sha256,source_size=[w,h],lenses=[])
    if len(disks)!=3:
        return dict(proof,status='not_applicable',reason='not_three_separate_large_camera_disks')
    consensus,readings=lens_centres(a)
    if consensus is None:raise InsufficientRearShadowEvidence('Rear shadow: lens geometry has no threshold consensus')
    L=engine('aluminum').rgb_to_lab(a[...,:3])[...,0]
    bounds,edge_proof=_plateau_core(L)
    proof.update(flat_core_bounds=bounds,plateau_source_edges=edge_proof)
    for i in disks:
        contour=cv2.findContours((labels==i).astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0][0]
        if len(contour)<30:raise InsufficientRearShadowEvidence('Rear shadow: insufficient lens contour')
        (cx,cy),(aa,bb),angle=cv2.fitEllipse(contour);radius=(aa+bb)/4
        if np.min(np.linalg.norm(consensus-[cx,cy],axis=1))>2.5:
            raise InsufficientRearShadowEvidence('Rear shadow: glass ellipse disagrees with source lens centre')
        ellipse=[float(v) for v in (cx,cy,aa,bb,angle)];rho,X,Y=_polar(ellipse)
        polar=ndimage.map_coordinates(L,[Y,X],order=1,mode='nearest')
        polar=ndimage.gaussian_filter(polar,(2,1.5),mode=('wrap','nearest'))
        gradient=np.gradient(polar,.5,axis=1)
        search=(rho>1.06)&(rho<1.28)
        anchor=np.where(search)[0][np.argmin(np.median(gradient,axis=0)[search])]
        nearby=np.where(abs(rho-rho[anchor])<.045)[0]
        edge=nearby[np.argmin(gradient[:,nearby],axis=1)]
        strength=-gradient[np.arange(ANGLES),edge]
        if np.percentile(strength,5)<2 or np.ptp(rho[edge])*radius>8:
            raise InsufficientRearShadowEvidence('Rear shadow: outer ring edge is not independently supported')
        trough=[];rising_strength=[]
        for k in range(ANGLES):
            interval=np.where((rho>rho[edge[k]]+.010)&(rho<rho[edge[k]]+.080))[0]
            exit_index=interval[np.argmax(gradient[k,interval])]
            trough.append(rho[exit_index]);rising_strength.append(gradient[k,exit_index])
        if np.percentile(rising_strength,5)<1:
            raise InsufficientRearShadowEvidence('Rear shadow: outer ring exit lacks source edge evidence')
        trough=np.asarray(trough)
        trend=ndimage.median_filter(trough,size=9,mode='wrap')
        if np.percentile(abs(trough-trend)*radius,95)>2:
            raise InsufficientRearShadowEvidence('Rear shadow: ring joint has ambiguous angular breaks')
        # Use the conservative outer envelope of measured exits plus 0.75px
        # sensor transition. This is not a stored fraction of a particular SKU.
        joint=ndimage.gaussian_filter1d(ndimage.maximum_filter1d(trend,size=5,mode='wrap'),1,mode='wrap')
        width=(joint-rho[edge])*radius
        stable_width=float(np.percentile(width,50))
        # A shadow can delay the rising edge; do not treat that delay as hardware.
        stable_joint=rho[edge]+np.minimum(width,stable_width)/radius
        start=stable_joint+.75/radius
        if np.any(start<rho[edge]+.005):raise InsufficientRearShadowEvidence('Rear shadow: invalid ring separation')
        proof['lenses'].append(dict(ellipse=ellipse,start_radii=start.tolist(),
            joint_width_median_px=stable_width,
            shadow_delayed_exit_angles=int(np.count_nonzero(width>stable_width+1)),
            hardware_radii=(ndimage.gaussian_filter1d(rho[edge],1,mode='wrap')+.5/radius).tolist(),
            source_edge_strength_p05=float(np.percentile(strength,5)),
            source_exit_strength_p05=float(np.percentile(rising_strength,5)),
            source_edge_radius_range=[float(rho[edge].min()),float(rho[edge].max())],
            source_joint_radius_range=[float(joint.min()),float(joint.max())]))
    proof.update(status='ready',threshold_readings=readings)
    return proof


def _reference_plane(values, reference, X, Y, ellipse, shape):
    """Fit surrounding flat-metal illumination; never extrapolate its hull."""
    cx,cy,aa,bb,_=ellipse;radius=(aa+bb)/4
    if reference.sum()<300:raise InsufficientRearShadowEvidence('Rear shadow: insufficient independent surrounding metal reference')
    A=np.column_stack([np.ones(reference.sum()),(X[reference]-cx)/radius,(Y[reference]-cy)/radius])
    B=values[reference];weight=np.ones(len(B))
    for _ in range(5):
        coefficients,_,rank,_=np.linalg.lstsq(A*np.sqrt(weight[:,None]),B*np.sqrt(weight),rcond=None)
        if rank!=3:raise InsufficientRearShadowEvidence('Rear shadow: surrounding reference is not spatially independent')
        residual=B-A@coefficients
        weight=np.minimum(1,2/np.maximum(abs(residual),1e-9))
    inliers=abs(residual)<4
    if inliers.mean()<.60 or np.median(abs(residual))>2.5:
        raise InsufficientRearShadowEvidence('Rear shadow: surrounding metal cannot support a continuous lighting reference')
    points=np.column_stack([X[reference][inliers],Y[reference][inliers]])
    quadrants=(points[:,0]>cx).astype(int)+2*(points[:,1]>cy)
    if np.count_nonzero(np.bincount(quadrants,minlength=4)>50)<3:
        raise InsufficientRearShadowEvidence('Rear shadow: surrounding reference lacks three-sided evidence')
    hull=cv2.convexHull(np.rint(points).astype('int32'));mask=np.zeros(shape,'uint8')
    cv2.fillConvexPoly(mask,hull,1);mask=ndimage.binary_erosion(mask>0,iterations=2)
    supported=ndimage.map_coordinates(mask.astype(float),[Y,X],order=0,mode='constant')>0
    target=coefficients[0]+coefficients[1]*(X-cx)/radius+coefficients[2]*(Y-cy)/radius
    return target,supported,dict(samples=int(reference.sum()),inlier_fraction=float(inliers.mean()),
        median_residual_L=float(np.median(abs(residual))),coefficients=coefficients.tolist(),
        extrapolation_outside_reference_hull=False)


def _ring_material(material, plan, yy, xx):
    h, w = material.shape
    x0,y0,x1,y1=plan['flat_core_bounds']
    core=(xx>=x0)&(xx<x1)&(yy>=y0)&(yy<y1)
    corners=plan['plateau_source_edges']['corners']
    for (x,y,sx,sy),entry in zip([(x0,y0,1,1),(x1,y0,-1,1),(x0,y1,1,-1),(x1,y1,-1,-1)],corners):
        radius=entry['radius'];cx=x+sx*radius;cy=y+sy*radius
        quadrant=((xx-cx)*sx<0)&((yy-cy)*sy<0)
        core&=~quadrant|((xx-cx)**2+(yy-cy)**2<=radius*radius)
    material=material&ndimage.binary_erosion(core,iterations=6)
    # Every measured ring is excluded, including a neighbouring lens reached
    # by another lens's correction footprint.
    for lens in plan['lenses']:
        cx,cy,aa,bb,angle=lens['ellipse'];phi=np.deg2rad(angle)
        starts=np.asarray(lens.get('hardware_radii',lens['start_radii']))
        if starts.shape!=(ANGLES,) or not np.isfinite(starts).all() or np.any((starts<1)|(starts>1.4)):
            raise ValueError('Rear shadow: invalid measured ring boundary')
        U=(np.cos(phi)*(xx-cx)+np.sin(phi)*(yy-cy))/(aa/2)
        V=(-np.sin(phi)*(xx-cx)+np.cos(phi)*(yy-cy))/(bb/2)
        angle_map=np.mod(np.arctan2(V,U),2*np.pi)
        edge=np.interp(angle_map.ravel(),np.linspace(0,2*np.pi,ANGLES+1),np.r_[starts,starts[0]]).reshape((h,w))
        material&=np.sqrt(U*U+V*V)>edge
    return material


def correction(base_L, source_L, material, plan):
    """Return a bounded additive field and auditable support/appearance checks."""
    if plan.get('status')!='ready':
        return np.zeros_like(base_L),np.zeros_like(material),dict(status=plan.get('status'),reason=plan.get('reason'))
    h,w=material.shape
    if plan['source_size']!=[w,h] or len(plan['lenses'])!=3:
        raise ValueError('Rear shadow: geometry plan dimensions/lens count differ')
    field=np.zeros_like(base_L);details=[]
    yy,xx=np.ogrid[:h,:w]
    material = _ring_material(material, plan, yy, xx)
    # Exclude all measured rings and projected neighbourhoods from references.
    reference_material=ndimage.binary_erosion(material,iterations=3)
    for lens in plan['lenses']:
        cx,cy,aa,bb,angle=lens['ellipse'];phi=np.deg2rad(angle)
        U=(np.cos(phi)*(xx-cx)+np.sin(phi)*(yy-cy))/(aa/2)
        V=(-np.sin(phi)*(xx-cx)+np.cos(phi)*(yy-cy))/(bb/2)
        reference_material&=(U*U+V*V)>(max(lens['start_radii'])+.24)**2
    skipped = []
    excluded = np.zeros_like(material)
    for index, lens in enumerate(plan['lenses']):
        try:
            mapped, proof = _lens_field(base_L, source_L, material, reference_material, plan['method'], lens, yy, xx)
        except InsufficientRearShadowEvidence as error:
            skipped.append(dict(lens=index, reason=str(error), ellipse=lens['ellipse']))
            # Adjacent correction footprints may overlap. Keep the entire
            # possible footprint of an unsupported lens at its base lighting.
            cx, cy, aa, bb, angle = lens['ellipse']
            phi = np.deg2rad(angle)
            u = (np.cos(phi)*(xx-cx)+np.sin(phi)*(yy-cy))/(aa/2)
            v = (-np.sin(phi)*(xx-cx)+np.cos(phi)*(yy-cy))/(bb/2)
            excluded |= u*u+v*v <= (max(lens['start_radii'])+.42)**2
            continue
        field = np.maximum(field, mapped)
        details.append(dict(proof, lens=index))
    if not details:
        raise InsufficientRearShadowEvidence(skipped[0]['reason'])
    field[excluded] = 0
    scope = field > 0
    return field,scope,dict(status='partially_applied' if skipped else 'applied',method=plan['method'],
                           lenses=details, skipped_lenses=skipped, changed_scope_pixels=int(scope.sum()),
                           max_added_L=float(field.max()))


def guard_secondary_arcs(base_lab, source_lab, rendered_lab, material, plan, report):
    """Check the actual post-gamut result of each independently adjusted ring."""
    lenses = report.get('lenses')
    if not lenses:
        return rendered_lab, np.zeros_like(material)
    h, w = material.shape
    yy, xx = np.ogrid[:h, :w]
    material = _ring_material(material, plan, yy, xx)
    result = rendered_lab.copy()
    for proof in lenses:
        lens = plan['lenses'][proof['lens']]
        proof['secondary_arc_guard'] = _restore_secondary_arc(base_lab, source_lab, result, material, lens)
    changed = np.any(result != rendered_lab, axis=-1)
    return result, changed


def _restore_secondary_arc(base_lab, source_lab, result, material, lens):
    """Keep the lighting from introducing a second edge outside the real rim."""
    if 'hardware_radii' not in lens:
        return dict(status='no_measured_rim')
    cx, cy, aa, bb, angle = lens['ellipse']
    radius = (aa + bb) / 4
    phi = np.deg2rad(angle)
    theta = np.arange(ANGLES) * (2 * np.pi / ANGLES)
    offsets = np.arange(0., 35., .5)
    hardware = np.asarray(lens['hardware_radii'])
    rho = hardware[:, None] + offsets / radius
    u = np.cos(theta[:, None]) * rho * aa / 2
    v = np.sin(theta[:, None]) * rho * bb / 2
    coords = [cy + np.sin(phi)*u + np.cos(phi)*v,
              cx + np.cos(phi)*u - np.sin(phi)*v]
    valid = ndimage.map_coordinates(material, coords, order=0, mode='constant')
    raw, base, rendered = [ndimage.map_coordinates(a[..., 0], coords, order=1)
                           for a in (source_lab, base_lab, result)]
    support = ndimage.gaussian_filter(valid.astype(float), (3, 1.5), mode=('wrap', 'nearest'))

    def low(values):
        return (ndimage.gaussian_filter(values * valid, (3, 1.5), mode=('wrap', 'nearest')) /
                np.maximum(support, 1e-6))

    raw_low, base_low, after = [low(a) for a in (raw, base, rendered)]
    rows = np.arange(ANGLES)
    crest = (offsets >= 3.5) & (offsets <= 7.5) & (support >= .85)
    peak = np.argmax(np.where(crest, after, -np.inf), axis=1)
    eligible = ((offsets >= 6.5) & (offsets <= 13.5) &
                (offsets >= offsets[peak, None]+1.5) & (support >= .85))
    valley = np.argmin(np.where(eligible, after, np.inf), axis=1)

    def drop(a):
        return a[rows, peak] - a[rows, valley]

    new_drop = drop(after)
    proposed = (eligible.any(axis=1) & crest.any(axis=1) &
                (new_drop >= 4) & (new_drop-drop(raw_low) >= 4) &
                (new_drop-drop(base_low) >= 2))
    # Close short angular holes in the same fault before feathering. Otherwise
    # pixel-wise detection leaves alternating on/off sections of one visible line.
    closed = ndimage.minimum_filter1d(
        ndimage.maximum_filter1d(proposed, size=19, mode='wrap'), size=19, mode='wrap')
    runs = ndimage.minimum_filter1d(closed, size=7, mode='wrap')
    if not runs.any():
        return dict(status='clear')
    coherent = ndimage.maximum_filter1d(runs, size=13, mode='wrap')
    confidence = _smoothstep((ndimage.gaussian_filter1d(coherent.astype(float), 2.5, mode='wrap')-.40)/.44)
    reference_band = (offsets >= 20) & (offsets <= 30)
    reference = valid[:, reference_band] & (support[:, reference_band] >= .85)
    supported = reference.sum(axis=1) >= 15
    gain = np.zeros(ANGLES)
    for row in np.flatnonzero(supported):
        gain[row] = np.median((base[row]-raw[row])[reference_band][reference[row]])
    count = ndimage.gaussian_filter1d(supported.astype(float), 5, mode='wrap')
    gain = ndimage.gaussian_filter1d(gain, 5, mode='wrap') / np.maximum(count, 1e-6)
    # Keep the source texture, but remove a broad dip in the photo's lighting
    # if it would become a visible second line. The actual rim is outside this window.
    start = np.flatnonzero(offsets >= 3.5)[0]
    profile = raw_low[:, start:]
    rise = np.maximum.accumulate(profile, axis=1)
    smooth_rise = ndimage.gaussian_filter1d(np.clip(rise-profile, 0, 8), 1.4, axis=1)
    removal = np.zeros_like(raw_low)
    removal[:, start:] = smooth_rise
    # If the outer reference crosses a material boundary, drop the added
    # shadow adjustment there and retain the verified base lighting instead.
    rebuild = _smoothstep((count-.66)/.25)[:, None]
    feather = _smoothstep((offsets-2.)/1.5)*_smoothstep((29.-offsets)/9.)

    image_y, image_x = np.ogrid[:material.shape[0], :material.shape[1]]
    u = (np.cos(phi)*(image_x-cx)+np.sin(phi)*(image_y-cy))/(aa/2)
    v = (-np.sin(phi)*(image_x-cx)+np.cos(phi)*(image_y-cy))/(bb/2)
    phase = np.mod(np.arctan2(v, u), 2*np.pi)*ANGLES/(2*np.pi)
    distance = (np.hypot(u, v)-np.interp(phase, rows, hardware, period=ANGLES))*radius
    weight = np.interp(phase, rows, confidence, period=ANGLES)
    active = material & (distance > 2) & (distance < 29) & (weight > 0)
    radial = distance[active] / .5
    polar = [phase[active], radial]
    texture_lift = ndimage.map_coordinates(np.r_[removal, removal[:1]], polar, order=1)
    gain_xy = np.interp(phase[active], rows, gain, period=ANGLES)
    ref_weight = np.interp(phase[active], rows, rebuild[:, 0], period=ANGLES)
    blend = weight[active]*np.interp(distance[active], offsets, feather)
    source = source_lab[active]
    base_pixels = base_lab[active]
    pixels = result[active]
    target = np.clip(source[:, 0] + gain_xy + texture_lift, 0, 100)
    guarded = base_pixels[:, 0] + ref_weight*np.clip(target-base_pixels[:, 0], -35, 35)
    pixels[:, 0] += blend*(guarded-pixels[:, 0])
    color_count = _match_shadow_color(base_lab, source_lab, coords, reference_band, reference,
                                     supported, valid, polar, source, pixels, weight[active], distance[active])
    result[active] = pixels
    return dict(status='guarded', detected_angles=int(proposed.sum()),
                reconstructed_angles=int((proposed & (count >= .91)).sum()),
                same_material_color_pixels=color_count,
                changed_scope_pixels=int(active.sum()))


def _match_shadow_color(base_lab, source_lab, coords, reference_band, reference,
                        supported, valid, polar, source, pixels, confidence, distance):
    """Match the exterior shadow to stable Lab samples of the same flat metal."""
    rows = np.arange(ANGLES)
    far = np.stack([ndimage.map_coordinates(base_lab[..., i], coords, order=1)[:, reference_band]
                    for i in range(3)], axis=-1)
    samples = np.zeros((ANGLES, 3))
    stable = np.zeros(ANGLES, dtype=bool)
    for row in np.flatnonzero(supported):
        observed = far[row, reference[row]]
        median = np.median(observed, axis=0)
        spread = np.percentile(np.linalg.norm(observed-median, axis=1), 80)
        if spread <= 5.5:
            stable[row] = True
            samples[row] = median
    support = ndimage.gaussian_filter1d(stable.astype(float), 5, mode='wrap')
    strength = np.interp(polar[0], rows, _smoothstep((support-.66)/.25), period=ANGLES)
    strength *= confidence*_smoothstep((distance-2.7)/3.)*_smoothstep((23-distance)/7.)
    apply = strength > .01
    if not apply.any():
        return 0
    color = ndimage.gaussian_filter1d(samples, 5, axis=0, mode='wrap') / np.maximum(support[:, None], 1e-6)
    filtered = np.stack([ndimage.map_coordinates(source_lab[..., i], coords, order=1)
                         for i in range(3)], axis=-1)
    texture_support = ndimage.gaussian_filter(valid.astype(float), (3, 4.4), mode=('wrap', 'nearest'))
    smooth = ndimage.gaussian_filter(filtered*valid[..., None], (3, 4.4, 0),
                                     mode=('wrap', 'nearest', 'nearest'))
    smooth /= np.maximum(texture_support[..., None], 1e-6)
    fine = np.column_stack([ndimage.map_coordinates(np.r_[smooth[..., i], smooth[:1, :, i]], polar, order=1)
                            for i in range(3)])
    texture_limit = np.array([4., 1.7, 2.1])
    detail = .8*texture_limit*np.tanh((source-fine)/texture_limit)
    normal = np.column_stack([np.interp(polar[0], rows, color[:, i], period=ANGLES) for i in range(3)])
    # Retain minute color variation and cap unusually large differences.
    shift = np.clip(normal+detail-pixels, [-26, -9, -15], [26, 9, 15])
    pixels[apply] += strength[apply, None]*shift[apply]
    return int(apply.sum())


def _lens_field(base_L, source_L, material, reference_material, method, lens, yy, xx):
    h,w = material.shape
    cx,cy,aa,bb,angle=lens['ellipse'];radius=(aa+bb)/4;phi=np.deg2rad(angle)
    rho,X,Y=_polar(lens['ellipse']);start=np.asarray(lens['start_radii'])[:,None]
    values=ndimage.map_coordinates(base_L,[Y,X],order=1,mode='nearest')
    original=ndimage.map_coordinates(source_L,[Y,X],order=1,mode='nearest')
    valid=ndimage.map_coordinates(material.astype(float),[Y,X],order=0,mode='constant')>0
    reference=ndimage.map_coordinates(reference_material.astype(float),[Y,X],order=0,mode='constant')>0
    offset=rho[None,:]-start
    fit=valid&(offset>=1/radius)
    support=ndimage.gaussian_filter(fit.astype(float),(4,2),mode=('wrap','nearest'))
    low=ndimage.gaussian_filter(values*fit,(4,2),mode=('wrap','nearest'))/np.maximum(support,1e-9)
    band=(offset>=.30)&(offset<.52)
    reference&=band
    target,reference_supported,reference_proof=_reference_plane(values,reference,X,Y,lens['ellipse'],(h,w))
    # Use only interpolation inside independently measured surroundings.
    # Feather the additional field inside that support, not source RGB.
    periodic=np.pad(reference_supported,((8,8),(0,0)),mode='wrap')
    confidence=_smoothstep(ndimage.distance_transform_edt(periodic)[8:-8]/6)
    deficit=np.maximum(0,target-low)
    feather_px=1.0 if method=='source_camera_ring_shadow_v2' else 2.5
    weight=_smoothstep(offset/(feather_px/radius))*_smoothstep((.42-offset)/.18)*confidence
    active=valid&(weight>0)&(offset>=0)&(offset<.42)
    if not active.any():
        raise InsufficientRearShadowEvidence('Rear shadow: no jointly supported active shadow samples')
    if np.any(active&(support<.008)):
        raise InsufficientRearShadowEvidence('Rear shadow: unsupported editable illumination')
    if np.any(active&(original<.5)&(deficit>10)):
        raise InsufficientRearShadowEvidence('Rear shadow: source shadow has unrecoverable black pixels')
    if np.percentile(deficit[active],99.5)>35:
        raise InsufficientRearShadowEvidence('Rear shadow: required lightness recovery exceeds limit')
    delta=np.minimum(deficit,35)*weight*valid
    joint_proof=None
    if 'hardware_radii' in lens:
        hardware=np.asarray(lens['hardware_radii'])[:,None]
        offsets=np.arange(0,12,.5)
        aligned_rho=hardware+offsets/radius
        coords=[np.broadcast_to(np.arange(ANGLES)[:,None],aligned_rho.shape),
                (aligned_rho-rho[0])/(rho[1]-rho[0])]
        aligned=ndimage.map_coordinates(values-target,coords,order=1,mode='nearest')
        aligned_valid=ndimage.map_coordinates(valid.astype(float),coords,order=0)>0
        reference_counts=aligned_valid.sum(axis=0)
        # Require a supported inner reference band; each remaining radial
        # column is separately gated below, including boundary samples.
        reference_available=reference_counts>=ANGLES*.65
        if not reference_available[(offsets>=1)&(offsets<=4)].all():
            raise InsufficientRearShadowEvidence('Rear shadow: narrow joint lacks circumferential source support')
        joint_support=ndimage.gaussian_filter(aligned_valid.astype(float),(4,1),mode=('wrap','nearest'))
        smoothed=ndimage.gaussian_filter(aligned*aligned_valid,(4,1),mode=('wrap','nearest'))/np.maximum(joint_support,1e-9)
        observed=np.where(aligned_valid,smoothed,np.nan)
        light_profile=np.zeros(len(offsets));variation=np.zeros(len(offsets))
        supported_profile=observed[:,reference_available]
        light_profile[reference_available]=np.nanpercentile(supported_profile,75,axis=0)
        variation[reference_available]=(np.nanpercentile(supported_profile,75,axis=0)
            -np.nanpercentile(supported_profile,25,axis=0))
        excess=np.maximum(0,np.minimum(light_profile,0)[None,:]-smoothed-1)
        gate=_smoothstep((variation-3)/4)
        # A farther sample can leave the flat plateau. Taper within the
        # independently supported radial columns; never relax support.
        radial_confidence=_smoothstep(ndimage.distance_transform_edt(reference_available)/2)
        joint_delta=excess*gate[None,:]*radial_confidence[None,:]
        position=(rho[None,:]-hardware)*radius
        mapped_joint=ndimage.map_coordinates(joint_delta,[np.broadcast_to(np.arange(ANGLES)[:,None],position.shape),position/.5],order=1,mode='constant')
        band=_smoothstep(position/1.0)*_smoothstep(((start-hardware)*radius+4-position)/2)
        joint_field=mapped_joint*band*confidence*valid
        joint_active=joint_field>0
        mapped_support=ndimage.map_coordinates(joint_support,
            [np.broadcast_to(np.arange(ANGLES)[:,None],position.shape),position/.5],order=1,mode='constant')
        if np.any(joint_active&(mapped_support<.008)):
            raise InsufficientRearShadowEvidence('Rear shadow: unsupported narrow joint illumination')
        # This is the separately classified dark joint, not flat metal.
        # Near-black joint samples can change illumination, but cannot
        # certify recovery of texture absent from the source photograph.
        near_black_joint=int(np.count_nonzero(joint_active&(original<.5)))
        if joint_field.max()>35:
            raise InsufficientRearShadowEvidence('Rear shadow: narrow joint lightness recovery exceeds limit')
        delta=np.maximum(delta,joint_field)
        joint_proof=dict(method='source_aligned_joint_upper_quartile',
            minimum_reference_angles=int(reference_counts[reference_available].min()),
            maximum_added_L=float(joint_field.max()),
            active_polar_samples=int(joint_active.sum()),
            source_near_black_polar_samples=near_black_joint,
            recovered_missing_texture=False,
            min_active_support=float(mapped_support[joint_active].min()) if joint_active.any() else None,
            hardware_inner_boundary_unchanged=True)

    U=(np.cos(phi)*(xx-cx)+np.sin(phi)*(yy-cy))/(aa/2)
    V=(-np.sin(phi)*(xx-cx)+np.cos(phi)*(yy-cy))/(bb/2)
    R=np.sqrt(U*U+V*V);T=np.mod(np.arctan2(V,U),2*np.pi)
    # Angular wrap only. Radial out-of-range samples never wrap to the ring.
    rows=np.vstack([delta,delta[:1]])
    mapped=ndimage.map_coordinates(rows,[T/(2*np.pi)*ANGLES,(R-rho[0])/(rho[1]-rho[0])],order=1,mode='constant',cval=0)
    local_start=np.interp(T.ravel(),np.linspace(0,2*np.pi,ANGLES+1),np.r_[start[:,0],start[0,0]]).reshape((h,w))
    hardware=np.asarray(lens.get('hardware_radii',lens['start_radii']))
    local_inner=np.interp(T.ravel(),np.linspace(0,2*np.pi,ANGLES+1),np.r_[hardware,hardware[0]]).reshape((h,w))
    local=material&(R>local_inner)&(R<local_start+.42)
    mapped=np.where(local,mapped,0)
    eligible=valid&(offset>=0)&(offset<.42)
    return mapped, dict(reference_plane=reference_proof,
        reference_supported_pixel_fraction=float(np.mean(reference_supported[eligible])),
        unsupported_reference_angles=int(np.count_nonzero(np.any(eligible&~reference_supported,axis=1))),
        min_active_fit_support=float(support[active].min()),
        max_added_L=float(delta.max()),hardware_interior_edited=False,
        narrow_joint_correction=joint_proof,
        reference_gate='robust same-plane interpolation within three-sided source reference hull')
