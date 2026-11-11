"""Conservative bottom-image registration, never a 3-D reconstruction.

Fits only the hole row's 1-D projectivity. Port and screws are holdouts.
Native end strips are copied exactly; central alpha is explicitly audited.
No production imports.
"""
import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares,minimize


PROFILES = {
    "iphone17": dict(width=71.45, center=35.725,
        holes=[19.71,21.965,24.22,47.23,49.485,51.74,53.995,56.25],
        counts=(3,5), screws=[28.80,42.65]),
    "ip17pro": dict(width=71.85, center=35.92,
        holes=[15.40,17.65,19.91,22.16,24.42,47.43,49.69,51.94,54.20,56.45],
        counts=(5,5), screws=[29.00,42.85]),
    "ip17promax": dict(width=77.98, center=38.99,
        holes=[15.41,17.65,19.91,22.16,24.42,26.67,51.30,53.56,55.81,58.07,60.32,62.57],
        counts=(6,6), screws=[32.06,45.91]),
}


class Rejected(ValueError):
    def __init__(self, reason, evidence=None):
        super().__init__(reason)
        self.evidence = evidence or {}


def rounded_rect_outline(width, height, ratio, step=1.0):
    """Points and outward normals of a centred rounded rectangle, radius ratio*height/2."""
    r=ratio*height/2;hw,hh=max(width/2-r,0),max(height/2-r,0)
    points=[];normals=[]
    for x0,y0,x1,y1,nx,ny in [(-hw,-height/2,hw,-height/2,0,-1),(width/2,-hh,width/2,hh,1,0),
                              (hw,height/2,-hw,height/2,0,1),(-width/2,hh,-width/2,-hh,-1,0)]:
        n=max(2,int(np.hypot(x1-x0,y1-y0)/step));u=np.linspace(0,1,n,endpoint=False)
        points.append(np.c_[x0+u*(x1-x0),y0+u*(y1-y0)]);normals.append(np.tile([nx,ny],(n,1)))
    n=max(4,int(2*np.pi*r/step));angle=np.linspace(0,2*np.pi,n,endpoint=False)
    c=np.c_[np.cos(angle),np.sin(angle)]
    points.append(np.c_[np.where(c[:,0]>=0,hw,-hw),np.where(c[:,1]>=0,hh,-hh)]+r*c);normals.append(c)
    return np.vstack(points),np.vstack(normals).astype('float32')


def port_rounded_rect(gray, proposals, unit):
    """Hit the whole port outline with a rounded rectangle; holes never supply port position.

    A sticker beside the port is one straight edge; it cannot match the top,
    bottom and both rounded ends together. Gradients are clipped so a strong
    paper edge earns no more than a normal rim edge, and are signed so only
    dark-inside/bright-outside contours count. Paper covering the port itself
    shortens the visible outline; the hole-fit holdout then rejects it.
    """
    seed=np.median(proposals,axis=0);cx0,cy0,_,_,bw,bh=seed;h,w=gray.shape
    x0=int(max(0,cx0-bw));x1=int(min(w,cx0+bw));y0=int(max(0,cy0-1.2*bh));y1=int(min(h,cy0+1.2*bh))
    g=cv2.GaussianBlur(gray.astype('float32'),(0,0),.7*unit)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3)/8;gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)/8
    magnitude=np.hypot(gx,gy);cap=max(4.,float(np.percentile(magnitude[y0:y1,x0:x1],90)))
    clip=np.minimum(1,cap/np.maximum(magnitude,1e-6))/cap;gx*=clip;gy*=clip
    def score(cx,cy,width,height,ratio,parts=False):
        points,normals=rounded_rect_outline(width,height,ratio);points=points+[cx,cy]
        mx=points[:,0].astype('float32')[None];my=points[:,1].astype('float32')[None]
        e=(cv2.remap(gx,mx,my,cv2.INTER_LINEAR).ravel()*normals[:,0]+
           cv2.remap(gy,mx,my,cv2.INTER_LINEAR).ravel()*normals[:,1])
        if not parts:return float(e.mean())
        end=width/2-ratio*height/2
        return dict(left_end=float(e[points[:,0]-cx<-end].mean()),right_end=float(e[points[:,0]-cx>end].mean()),
                    top=float(e[(normals[:,1]<-.9)].mean()),bottom=float(e[(normals[:,1]>.9)].mean()))
    def fit(ratio,width_range):
        coarse=None
        for width in np.arange(width_range[0]*bw,width_range[1]*bw,2*unit):
            for height in np.arange(.8*bh,1.8*bh,2*unit):
                points,normals=rounded_rect_outline(width,height,ratio,.5)
                tw=int(np.ceil(width))+5;th=int(np.ceil(height))+5
                if tw>=x1-x0 or th>=y1-y0:continue
                tx=np.zeros((th,tw),'float32');ty=np.zeros((th,tw),'float32')
                ix=np.round(points[:,0]+tw/2-.5).astype(int);iy=np.round(points[:,1]+th/2-.5).astype(int)
                np.add.at(tx,(iy,ix),normals[:,0]/len(points));np.add.at(ty,(iy,ix),normals[:,1]/len(points))
                response=(cv2.matchTemplate(gx[y0:y1,x0:x1],tx,cv2.TM_CCORR)+
                          cv2.matchTemplate(gy[y0:y1,x0:x1],ty,cv2.TM_CCORR))
                _,value,_,loc=cv2.minMaxLoc(response)
                if coarse is None or value>coarse[0]:
                    coarse=(value,[x0+loc[0]+tw/2-.5,y0+loc[1]+th/2-.5,width,height])
        if coarse is None:return None
        # Corner ratio is refined too (Pro Max ports sit between the two seeds).
        lo,hi=width_range[0]*bw,width_range[1]*bw
        objective=lambda p:-score(*p[:4],float(np.clip(p[4],.6,1.)))+(0 if lo<=p[2]<=hi else 10)
        refined=minimize(objective,coarse[1]+[ratio],method='Nelder-Mead',
                         options=dict(xatol=.05,fatol=1e-4,maxiter=800))
        x=refined.x;r=float(np.clip(x[4],.6,1.))
        return dict(seed_ratio=ratio,width_range=list(width_range),ratio=r,center_x=float(x[0]),center_y=float(x[1]),
                    width=float(x[2]),height=float(x[3]),score=float(score(*x[:4],r)),
                    support=score(*x[:4],r,parts=True))
    fits=[f for f in (fit(ratio,(.9,1.5)) for ratio in (.8,1.)) if f]
    if not fits:raise Rejected('port_rounded_rect_search_window_missing',dict(seed=seed.tolist()))
    # Pro Max ports show an inner opening and an outer chamfer rim. A sticker
    # against the rim can pull a fit into a mixed inner/outer outline with one
    # weak end. Only then: also fit the opening itself. Fits whose two ends both
    # show an edge are preferred.
    if np.ptp([f['center_x'] for f in fits])>2*unit:
        fits+=[f for f in (fit(ratio,(.85,1.15)) for ratio in (.8,1.)) if f]
    pool=[f for f in fits if min(f['support']['left_end'],f['support']['right_end'])>=.5] or fits
    best=max(pool,key=lambda f:f['score'])
    proof=dict(method='rounded rectangle template hit on clipped signed RGB gradient, Nelder-Mead refined incl. corner ratio',
               seed=seed.tolist(),gradient_cap=cap,fits=fits,selected=best,independent_of_hole_fit=True)
    # Mean signed edge is 1.0 for a full-strength rim all round; top, bottom
    # and both ends must each show a real edge.
    if best['score']<.5 or min(best['support'].values())<.25:
        raise Rejected('port_rounded_rect_unsupported',proof)
    spread=float(np.ptp([f['center_x'] for f in pool]))
    left=best['center_x']-best['width']/2;right=best['center_x']+best['width']/2
    refined=seed.copy();refined[0]=best['center_x'];refined[1]=best['center_y']
    refined[2]=left;refined[3]=best['center_y']-best['height']/2;refined[4]=best['width'];refined[5]=best['height']
    return refined,spread,[[left,right]],proof


def hole_contrast_image(a, scale):
    """Darkness relative to nearby metal, in local noise units, for low-contrast colours."""
    gray=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY)
    # Paper and highlights well above the local metal are swapped for the local
    # background, so a sticker does not make the metal beside it read as dark.
    size=int(round(51*scale))|1;background=cv2.medianBlur(gray,size)
    for _ in range(2):
        bright=gray.astype('int16')>background.astype('int16')+20
        background=cv2.medianBlur(np.where(bright,background,gray).astype('uint8'),size)
    background=background.astype('float32')
    residual=cv2.GaussianBlur(gray.astype('float32'),(0,0),scale)-background
    values=residual[a[...,3]>=240];spread=1.4826*np.median(abs(values-np.median(values)))
    return np.clip(160+25*residual/max(spread,1.),0,255).astype('uint8')


def measure(a, model, contrast='raw'):
    """contrast='local' is the dark-colour fallback: holes are thresholded on
    metal-relative darkness with filled rings at 5-level steps, and the screw
    is confirmed separately (largest agreeing group) rather than per hole level."""
    profile=PROFILES[model];h,w=a.shape[:2];scale=w/1030
    gray=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY)
    local=contrast=='local';step=5 if local else 10
    feature=hole_contrast_image(a,scale) if local else None
    readings=[];failures=[];port_proposals=[]
    for threshold in range(30,151,step):
        binary=((gray<threshold)&(a[...,3]>=240)).astype('uint8')
        _,_,stats,_=cv2.connectedComponentsWithStats(binary,8)
        ports=[];circles=[]
        for box in stats[1:]:
            x,y,bw,bh,area=[int(v) for v in box];cx,cy=x+(bw-1)/2,y+(bh-1)/2
            if (.35*w<cx<.65*w and .08*w<bw<.18*w and .16*h<bh<.55*h
                    and 2<bw/bh<5 and .25*h<cy<.75*h):
                ports.append([cx,cy,x,y,bw,bh])
        if local:
            holes_binary=ndimage.binary_fill_holes((feature<threshold)&(a[...,3]>=240)).astype('uint8')
            _,_,stats,_=cv2.connectedComponentsWithStats(holes_binary,8)
        for box in stats[1:]:
            x,y,bw,bh,area=[int(v) for v in box];cx,cy=x+(bw-1)/2,y+(bh-1)/2
            if (.12*w<cx<.88*w and .25*h<cy<.75*h and
                    .010*w<bw<.033*w and .07*h<bh<.34*h and
                    .6<bw/bh<1.6 and area>30*scale**2):
                circles.append([cx,cy,bw,bh])
        for port in ports:port_proposals.append(dict(threshold=threshold,port=port))
        # No port estimate or predicted physical port centre enters hole runs.
        left_all=sorted([p for p in circles if p[0]<.42*w])
        right_all=sorted([p for p in circles if p[0]>.58*w])
        def runs(points,n):
            found=[]
            for i in range(len(points)-n+1):
                q=np.asarray(points[i:i+n]);pitch=np.diff(q[:,0])
                if (np.std(pitch)<=1.5*scale and .018*w<pitch.mean()<.04*w
                        and np.ptp(q[:,1])<=3*scale):found.append(points[i:i+n])
            return found
        layouts=[]
        for counts in sorted(set([tuple(profile['counts']),tuple(reversed(profile['counts']))])):
            for left in runs(left_all,counts[0]):
                for right in runs(right_all,counts[1]):
                    if abs(np.median(np.array(left)[:,1])-np.median(np.array(right)[:,1]))<3*scale:
                        layouts.append((left,right))
        if len(layouts)!=1:
            failures.append(dict(threshold=threshold,reason='hole_groups_ambiguous_or_missing',
                                 observed=[len(left_all),len(right_all)],layout_count=len(layouts)))
            continue
        left,right=layouts[0];row_y=float(np.median(np.asarray(left+right)[:,1]))
        screws=sorted([p for p in circles if left[-1][0]<p[0]<right[0][0]
                       and .05*w<abs(p[0]-(left[-1][0]+right[0][0])/2)<.12*w
                       and abs(p[1]-row_y)<max(3*scale,.045*h)
                       and p[2]>.016*w and p[3]>.12*h])
        if not screws or len(screws)>2:
            failures.append(dict(threshold=threshold,reason='independent_screw_missing_or_ambiguous'))
            if not local:continue
            screws=[]
        readings.append(dict(threshold=threshold,holes=left+right,counts=[len(left),len(right)],screws=screws))
    stable=[]
    for i in range(len(readings)):
        run=[readings[i]]
        for r in readings[i+1:]:
            if r['threshold']-run[-1]['threshold']!=step or r['counts']!=run[0]['counts']:break
            if not local and len(r['screws'])!=len(run[0]['screws']):break
            if any(np.ptp([q[k] for q in run+[r]],axis=0)[...,:2].max()>2*scale for k in (['holes'] if local else ['holes','screws'])):break
            run.append(r)
        if len(run)>=3:stable.append(run)
    audit=dict(readings=readings,failures=failures,port_seed_proposals=port_proposals)
    if not stable:raise Rejected('independent_hole_consensus_failed',audit)
    stable.sort(key=len,reverse=True);chosen=stable[0]
    for run in stable[1:]:
        if run[0]['counts']!=chosen[0]['counts'] or np.max(abs(np.asarray(run[0]['holes'])[:,:2]-np.asarray(chosen[0]['holes'])[:,:2]))>2*scale:
            raise Rejected('competing_stable_hole_layouts',audit)
    holes=np.median([r['holes'] for r in chosen],axis=0)
    screw_readings=[r['screws'] for r in chosen if r['screws']]
    if local:
        # A half-visible screw ring shifts its box; keep the largest group of agreeing readings.
        # One reading suffices: the screw is a holdout and must still land within 2.5 px of the hole fit.
        groups=[[s for s in screw_readings if len(s)==len(ref) and
                 np.abs(np.asarray(s)[:,:2]-np.asarray(ref)[:,:2]).max()<=scale] for ref in screw_readings]
        screw_readings=max(groups,key=len,default=[])
        if not screw_readings:raise Rejected('independent_screw_missing_or_ambiguous',audit)
    screws=np.median(screw_readings,axis=0)
    # A component is only a seed. Require repeat observation and reject multiple
    # separated seed clusters. Three consecutive thresholds are still required
    # for hole coordinates, but are no longer falsely required for ALL features.
    clusters=[]
    for proposal in port_proposals:
        p=np.array(proposal['port'])
        def same_object(q):
            q=np.asarray(q);ix=max(0,min(p[2]+p[4],q[2]+q[4])-max(p[2],q[2]))
            iy=max(0,min(p[3]+p[5],q[3]+q[5])-max(p[3],q[3]))
            return ix>=.75*min(p[4],q[4]) and ix*iy>=.65*min(p[4]*p[5],q[4]*q[5])
        near=[c for c in clusters if same_object(np.median([q['port'] for q in c],axis=0))]
        if len(near)>1:raise Rejected('port_seed_clusters_ambiguous',audit)
        if near:near[0].append(proposal)
        else:clusters.append([proposal])
    clusters=[c for c in clusters if len({r['threshold'] for r in c})>=2]
    if len(clusters)!=1:raise Rejected('port_seed_not_unique',audit)
    seeds=clusters[0]
    try:port,port_spread,port_tracks,edgeproof=port_rounded_rect(gray,[r['port'] for r in seeds],scale)
    except Rejected as e:
        e.evidence.update(hole_thresholds=[r['threshold'] for r in chosen],
                          port_seed_thresholds=sorted({r['threshold'] for r in seeds}),
                          detection_audit=audit)
        raise
    spread=max(float(np.ptp([r['holes'] for r in chosen],axis=0)[...,:2].max()),
               float(np.ptp(screw_readings,axis=0)[...,:2].max()))
    # Port fits only gate the result (holes alone drive the correction), so their
    # spread is recorded; the 2.5 px port holdout below is the independent check.
    if spread>2*scale:raise Rejected('feature_coordinates_unstable',audit)
    line=np.polyfit(holes[:,0],holes[:,1],1)
    line_error=float(abs(np.polyval(line,holes[:,0])-holes[:,1]).max())
    if line_error>1.5*scale or abs(np.degrees(np.arctan(line[0])))>.5:
        raise Rejected('hole_row_not_supported_by_horizontal_1d_model',dict(row_error=line_error,slope=float(line[0])))
    nominal=np.array(profile['holes']);mirrored=chosen[0]['counts']==[5,3] if model=='iphone17' else False
    if mirrored:nominal=np.sort(profile['width']-nominal)
    return dict(holes=holes,port=port,screws=screws,nominal=nominal,mirrored=mirrored,
                feature_spread_px=spread,port_spread_px=port_spread,port_edge_tracks=port_tracks,
                port_contour_evidence=edgeproof,port_seed_thresholds=sorted({r['threshold'] for r in seeds}),
                row_max_error_px=line_error,selected_thresholds=[r['threshold'] for r in chosen],
                hole_contrast=contrast,detection_audit=audit)


def project(params, t):
    a,b,c=params
    return b+a*np.asarray(t)/(1+c*np.asarray(t))


def fit_points(nominal, observed, width, image_width, train=None):
    t=np.asarray(nominal)/width-.5
    y=np.asarray(observed)
    keep=np.arange(len(t)) if train is None else np.asarray(train)
    slope,intercept=np.polyfit(t[keep],y[keep],1)
    fit=least_squares(lambda p:project(p,t[keep])-y[keep],
        [slope,intercept,0],bounds=([.55*image_width,.25*image_width,-.65],
                                 [1.6*image_width,.75*image_width,.65]),max_nfev=150)
    if not fit.success:raise Rejected('projectivity_fit_failed')
    return fit.x


def smoothstep(x):
    t=np.clip(x,0,1)
    # C1 cubic has maximum blend derivative 1.5 vs quintic 1.875.
    # Reduce transition compression without altering targets or acceptance gates.
    return t*t*(3-2*t)


def plan(a,model):
    h,w=a.shape[:2];unit=w/1030;profile=PROFILES[model]
    try:m=measure(a,model)
    except Rejected as error:
        # Dark colours: holes barely darker than the metal, or a dark port that only
        # separates from dark metal at one level. Retry on local contrast only then.
        if str(error) not in ('independent_hole_consensus_failed','port_seed_not_unique'):raise
        m=measure(a,model,'local')
    x=m['nominal'];observed=m['holes'][:,0]
    hold=[1,len(x)-2];train=[i for i in range(len(x)) if i not in hold]
    cvfit=fit_points(x,observed,profile['width'],w,train)
    cv_errors=project(cvfit,x/profile['width']-.5)-observed
    params=fit_points(x,observed,profile['width'],w)
    errors=project(params,x/profile['width']-.5)-observed
    proof=dict(model=model,scope='visible bottom interior 1-D registration; not 2-D homography or 3-D recovery',
        port_used_for_fitting=False,screws_used_for_fitting=False,params=params.tolist(),
        hole_source=m['holes'].tolist(),hole_nominal_mm=x.tolist(),mirrored=m['mirrored'],
        hole_fit_max_px=float(abs(errors).max()),holdout_indices=hold,training_indices=train,
        holdout_only_max_px=float(abs(cv_errors[hold]).max()),training_only_max_px=float(abs(cv_errors[train]).max()),
        hole_fit_errors_px=errors.tolist(),holdout_only_errors_px=cv_errors[hold].tolist(),
        port_before_x=float(m['port'][0]),port_before_width_px=float(m['port'][4]),
        canvas_center_x=(w-1)/2,nominal_width_mm=profile['width'],feature_spread_px=m['feature_spread_px'],
        port_spread_px=m['port_spread_px'],hole_row_max_error_px=m['row_max_error_px'],
        selected_thresholds=m['selected_thresholds'],
        port_measurement='rounded rectangle hit on whole port outline; independent component seeds',
        hole_contrast=m['hole_contrast'],
        port_contour_evidence=m['port_contour_evidence'],port_seed_thresholds=m['port_seed_thresholds'],
        detection_audit=m['detection_audit'],
        port_edge_tracks=m['port_edge_tracks'],
        physical_accuracy_verified=False,UV_verified=False,unobserved_surface_generated=False)
    if max(proof['hole_fit_max_px'],proof['holdout_only_max_px'])>1.5*unit:
        raise Rejected('independent_holes_reject_1d_projectivity',proof)
    # Symmetric Pro groups cannot determine handedness. Require both physical
    # alternatives to be equivalent within the measurement tolerance.
    center=profile['width']-profile['center'] if m['mirrored'] else profile['center']
    if model!='iphone17':
        alternative=np.sort(profile['width']-np.array(profile['holes']))
        p2=fit_points(alternative,observed,profile['width'],w)
        center_span=abs(project(params,center/profile['width']-.5)-
                        project(p2,(profile['width']-center)/profile['width']-.5))
        proof.update(orientation='symmetric_layout_equivalence_only',orientation_center_span_px=float(center_span))
        if center_span>.5*unit:raise Rejected('orientation_ambiguity_exceeds_tolerance',proof)
    predicted=float(project(params,center/profile['width']-.5))
    proof['port_holdout_error_px']=float(m['port'][0]-predicted)
    nominal_screws=np.array(profile['screws'])
    if m['mirrored']:nominal_screws=np.sort(profile['width']-nominal_screws)
    screw_errors=[]
    for s in m['screws']:
        nominal_screw=nominal_screws[0 if s[0]<m['port'][0] else 1]
        screw_errors.append(float(s[0]-project(params,nominal_screw/profile['width']-.5)))
    proof.update(screw_holdout_errors_px=screw_errors,visible_screw_count=len(screw_errors))
    if abs(proof['port_holdout_error_px'])>2.5*unit:raise Rejected('port_independent_holdout_failed',proof)
    if max(abs(e) for e in screw_errors)>2.5*unit:raise Rejected('screw_independent_holdout_failed',proof)
    col=np.arange(w,dtype=float)
    target=x/profile['width']*(w-1)
    cap=max(12,int(.6*h))
    lo,hi=float(target.min()-8*unit),float(target.max()+8*unit)
    if not cap+8<lo<hi<w-1-cap-8:raise Rejected('insufficient_supported_transition_space',proof)
    weight=smoothstep((col-cap)/(lo-cap))*smoothstep((w-1-cap-col)/(w-1-cap-hi))
    raw_map=project(params,col/(w-1)-.5)
    mx=col+weight*(raw_map-col)
    mx[:cap]=col[:cap];mx[-cap:]=col[-cap:]
    mapx=np.broadcast_to(mx.astype('float32'),(h,w)).copy()
    mapy=np.broadcast_to(np.arange(h,dtype='float32')[:,None],(h,w)).copy()
    mapx[:,:cap]=col[:cap];mapx[:,-cap:]=col[-cap:]
    jac=np.gradient(mapx,axis=1)
    proof.update(native_endcaps_fixed_px=cap,sampling_derivative_range=[float(jac.min()),float(jac.max())],
        silhouette_policy='central silhouette follows same 1-D sampling, <=2 reference px boundary change',
        transition='C1 cubic to identity, explicit whole-strip lock',
        nominal_port_target_x=center/profile['width']*(w-1))
    if jac.min()<.65 or jac.max()>1.4:raise Rejected('local_sampling_stretch_exceeds_limit',proof)
    active=abs(mapx-col[None,:])>1e-5
    if mapx.min()<0 or mapx.max()>w-1:
        raise Rejected('sampling_not_supported_by_visible_source',proof)
    return mapx,mapy,active,proof,m


def render(a,mapx,mapy,active):
    f=a.astype('float32')/255
    f[...,:3]*=f[...,3,None]
    sampled=cv2.remap(f,mapx,mapy,cv2.INTER_CUBIC,borderMode=cv2.BORDER_CONSTANT)
    rgb=np.zeros_like(sampled[...,:3])
    np.divide(sampled[...,:3],sampled[...,3,None],out=rgb,where=sampled[...,3,None]>1e-6)
    mapped=np.clip(np.rint(np.dstack([rgb,sampled[...,3]])*255),0,255).astype('uint8')
    out=a.copy();out[active]=mapped[active]
    return out


def rectify(a,model):
    mx,my,active,proof,before=plan(a,model)
    if abs(before['port'][0]-proof['nominal_port_target_x'])<=2.5*a.shape[1]/1030:
        # Passing measurement gates does not justify editing an already centred
        # image merely to make a small reported residual look smaller.
        ident=np.broadcast_to(np.arange(a.shape[1],dtype='float32'),a.shape[:2]).copy()
        proof.update(action='unchanged_already_within_center_tolerance',
            final_port_x=float(before['port'][0]),
            final_port_center_error_px=float(before['port'][0]-proof['nominal_port_target_x']),
            final_canvas_center_error_px=float(before['port'][0]-proof['canvas_center_x']),
            final_port_width_px=float(before['port'][4]),port_width_change_fraction=0.,
            final_hole_errors_px=None,final_hole_target_check='not applicable: no correction performed',
            cap_pixel_changes=0,alpha_changed=0,max_central_boundary_change_px=0.,
            locked_pixel_changes=0,changed_RGB_pixels=0,sampling_derivative_range=[1.,1.])
        return a.copy(),proof,ident,np.zeros(a.shape[:2],dtype=bool)
    out=render(a,mx,my,active)
    after=measure(out,model,before['hole_contrast'])
    cap=proof['native_endcaps_fixed_px']
    e=float(after['port'][0]-proof['nominal_port_target_x'])
    boundaries=[];column_support=[]
    for b in [a,out]:
        opaque=b[...,3]>=128
        column_support.append(opaque.any(axis=0))
        boundaries.append(np.stack([opaque.argmax(axis=0),len(b)-1-opaque[::-1].argmax(axis=0)]))
    if not np.array_equal(*column_support):raise Rejected('body_column_support_changed',proof)
    boundary_change=float(abs(boundaries[1]-boundaries[0])[:,column_support[0]].max())
    proof.update(action='corrected_needs_visual_review',final_port_x=float(after['port'][0]),final_port_center_error_px=e,
        final_canvas_center_error_px=float(after['port'][0]-proof['canvas_center_x']),
        final_port_width_px=float(after['port'][4]),
        port_width_change_fraction=float(after['port'][4]/before['port'][4]-1),
        final_hole_errors_px=(after['holes'][:,0]-before['nominal']/PROFILES[model]['width']*(a.shape[1]-1)).tolist(),
        cap_pixel_changes=int(np.count_nonzero(np.any(a[:,:cap]!=out[:,:cap],axis=2)))+
                         int(np.count_nonzero(np.any(a[:,-cap:]!=out[:,-cap:],axis=2))),
        alpha_changed=int(np.count_nonzero(a[...,3]!=out[...,3])),
        max_central_boundary_change_px=boundary_change,
        locked_pixel_changes=int(np.count_nonzero(np.any(a[~active]!=out[~active],axis=1))),
        changed_RGB_pixels=int(np.count_nonzero(np.any(a[...,:3]!=out[...,:3],axis=2))))
    if abs(e)>2.5*a.shape[1]/1030:raise Rejected('final_port_measurement_failed',proof)
    if max(abs(x) for x in proof['final_hole_errors_px'])>1.5*a.shape[1]/1030:
        raise Rejected('final_hole_measurement_failed',proof)
    if abs(proof['port_width_change_fraction'])>.2:raise Rejected('port_shape_change_too_large',proof)
    # The measured contour is an integer pixel index; round its scaled limit
    # upward rather than rejecting 2 pixels as >1.998 on a 1029-wide canvas.
    proof['central_boundary_limit_px']=int(np.ceil(2*a.shape[1]/1030))
    if boundary_change>proof['central_boundary_limit_px']:raise Rejected('central_silhouette_change_exceeds_limit',proof)
    if proof['cap_pixel_changes'] or proof['locked_pixel_changes']:
        raise Rejected('exact_pixel_invariant_failed',proof)
    return out,proof,mx,active
