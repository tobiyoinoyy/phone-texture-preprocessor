"""Measure current-image antenna bands and hardware for studio iPhone photos.

Coordinates come from pixels in each new SKU. The unmirrored rear view maps
physical left to image right, and physical right to image left. No SKU template
coordinates or prior accepted landmarks are used.
"""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from .geometry_review import digest
from . import structure_review as gate
from . import source_body_reference as sbr


def gray_signal(a,x_range):
    return np.median(a[:,x_range[0]:x_range[1],:3].astype(float).mean(axis=2),axis=1)


def antenna_rows(image,x_range,expected_width=None,require_full_span=False):
    a=np.array(image.convert('RGBA'));h,w=a.shape[:2]
    signal=gray_signal(a,x_range);g=np.gradient(gaussian_filter1d(signal,1.))
    peaks,_=find_peaks(np.abs(g),height=.5,distance=2)
    result=[];signs=[];scores=[];ranked=[]
    for lower,upper in ((.025,.18),(.82,.975)):
        group=[int(p) for p in peaks if lower*h<p<upper*h]
        pairs=[]
        for lo in group:
            for hi in group:
                width=hi-lo
                if not 3<=width<=max(6,.025*h) or g[lo]*g[hi]>=0:continue
                if expected_width is not None and not .65*expected_width<=width<=1.45*expected_width:continue
                if np.median(a[lo:hi+1,x_range[0]:x_range[1],3])<240:continue
                if min(np.median(a[y,x_range[0]:x_range[1],3]) for y in (lo,hi))<128:continue
                # A material strip has opposing, supported boundaries and a
                # sustained interior, rather than a single scratch or ramp.
                inside=float(np.median(signal[lo+1:hi]))
                left=float(np.median(signal[max(0,lo-width):lo-1]))
                right=float(np.median(signal[hi+2:min(h,hi+width+1)]))
                contrast=min(abs(inside-left),abs(inside-right))
                score=min(abs(g[lo]),abs(g[hi]))*np.sqrt(width)*min(1.,contrast/4.)
                if contrast>=1 and score>0:pairs.append((score,lo,hi))
        if not pairs:raise ValueError('No supported antenna strip in both body ends')
        ranked.append(sorted(pairs,reverse=True))
        score,lo,hi=max(pairs)
        if np.median(a[[lo,hi],x_range[0]:x_range[1],3])<128:
            raise ValueError('Antenna boundary falls outside opaque body')
        result.extend([lo,hi]);signs.extend([int(np.sign(g[lo])),int(np.sign(g[hi]))]);scores.append(float(score))
    selection=[]
    if require_full_span:
        # A button cap or a scratch can outrank the real antenna in the
        # central 1-D signal. Every chosen side candidate must independently
        # satisfy the existing full-width material-pair/endpoint gates.
        from .antenna_material import trace_boundaries
        for band,pairs in enumerate(ranked):
            rejected=[];first_error=None;weak=None
            for score,lo,hi in pairs:
                candidate=result.copy();candidate[2*band:2*band+2]=[lo,hi]
                try:_,gate_proof=trace_boundaries(image,candidate,bands=(band,))
                except ValueError as exc:
                    if first_error is None:first_error=exc
                    rejected.append(dict(rows=[lo,hi],reason=str(exc)))
                    continue
                if gate_proof['endpoint_shape_review_required']:
                    # Keep the pre-2026-11-06 choice: a candidate with a
                    # coherent end strip still outranks a weak-end one.
                    if weak is None:weak=(score,lo,hi,candidate)
                    rejected.append(dict(rows=[lo,hi],reason='endpoint_shape_weak_deferred'))
                    continue
                break
            else:
                if weak is None:raise first_error
                score,lo,hi,candidate=weak
            result=candidate;signs[2*band:2*band+2]=[int(np.sign(g[lo])),int(np.sign(g[hi]))]
            scores[band]=float(score)
            selection.append(dict(band=band,rows=[lo,hi],rejected_candidates=rejected,
                endpoint_shape_review_required=weak is not None and weak[3] is candidate))
    proof=dict(signal_columns=list(x_range),scores=scores,
        supported_central_candidates=[[[int(lo),int(hi)] for score,lo,hi in pairs] for pairs in ranked])
    if require_full_span:proof['full_span_candidate_selection']=selection
    return result,signs,proof


def rear_reference(image,side,edge,expected_width):
    """Prefer side-width prior, but never let a bad prior veto rear evidence."""
    try:
        return _rear_reference(image,side,edge,expected_width,2,per_band=expected_width is None)
    except ValueError:
        if expected_width is None:raise
        rows,signs,xr,proof=_rear_reference(image,side,edge,None,3,per_band=True)
        proof=dict(proof,width_prior_rejected=float(expected_width),
                   method='independent_rear_column_consensus')
        return rows,signs,xr,proof


def _rear_reference(image,side,edge,expected_width,minimum_support,per_band=False):
    candidates=[]
    for offset in range(6,max(13,int(image.width*.045)),4):
        xr=[edge-offset-7,edge-offset] if side=='left' else [edge+offset,edge+offset+7]
        if xr[0]<0 or xr[1]>image.width:continue
        try:
            rows,signs,proof=antenna_rows(image,xr,expected_width)
            candidates.append((rows,signs,xr,proof))
        except ValueError:continue
    stable=[]
    for candidate in candidates:
        neighbours=[c for c in candidates if np.max(np.abs(np.array(c[0])-candidate[0]))<=3]
        if len(neighbours)>=minimum_support:
            stable.append((len(neighbours),min(candidate[3]['scores']),candidate))
    if not stable:
        if per_band:return _rear_reference_per_band(image,candidates,minimum_support)
        raise ValueError('Rear antenna landmarks lack neighbouring-column consensus')
    support,_,chosen=max(stable,key=lambda v:(v[0],v[1]))
    rows,signs,xr,proof=chosen
    return rows,signs,xr,dict(proof,agreeing_columns=support)


def _rear_reference_per_band(image,candidates,minimum_support):
    """Fallback when no column agrees on all four rows at once.

    Each band first keeps the original top-candidate consensus. A band without
    one (2026-11-09, Pro Max: a rounded-corner shadow outranks a soft antenna
    strip) uses the physical fact that the antenna is horizontal: the same
    rows among every column's supported candidates, in at least 70% of
    columns. A curved corner shifts with the column, so it cannot gather that.
    """
    if len(candidates)<3:raise ValueError('Rear antenna landmarks lack neighbouring-column consensus')
    def near(a,b):return abs(a[0]-b[0])<=3 and abs(a[1]-b[1])<=3
    rows=[];method=[]
    for band in (0,1):
        tops=[tuple(c[0][2*band:2*band+2]) for c in candidates]
        best=None
        for pair in tops:
            agree=sum(near(pair,other) for other in tops)
            if agree>=minimum_support and (best is None or agree>best[0]):best=(agree,pair)
        if best is not None:
            rows.extend(best[1]);method.append(dict(band=band,method='top_candidate_consensus',agreeing_columns=best[0]));continue
        lists=[[tuple(p) for p in c[3]['supported_central_candidates'][band]] for c in candidates]
        options=[]
        for ranked in lists:
            for pair in ranked:
                ranks=[next((i for i,other in enumerate(r) if near(pair,other)),None) for r in lists]
                hits=[i for i in ranks if i is not None]
                options.append((len(hits),-sum(hits),pair))
        agree,_,pair=max(options)
        if agree<max(3,int(np.ceil(.7*len(candidates)))):
            raise ValueError('Rear antenna landmarks lack neighbouring-column consensus')
        rows.extend(pair);method.append(dict(band=band,method='horizontal_cross_column_candidates',agreeing_columns=agree))
    # Report rows and signs from one column whose own candidates hold both
    # chosen bands, using that column's exact pair, never a neighbour's rows.
    for found_rows,_,xr,proof in candidates:
        lists=proof['supported_central_candidates']
        own=[next((tuple(p) for p in lists[b] if near(tuple(rows[2*b:2*b+2]),tuple(p))),None) for b in (0,1)]
        if None not in own:break
    else:raise ValueError('Rear antenna landmarks lack neighbouring-column consensus')
    rows=[int(v) for pair in own for v in pair]
    g=np.gradient(gaussian_filter1d(gray_signal(np.array(image.convert('RGBA')),xr),1.))
    signs=[int(np.sign(g[r])) for r in rows]
    return rows,signs,xr,dict(proof,per_band_selection=method,method='per_band_rear_column_consensus')


def windows(rows,h,radius=4):
    result=[]
    for i,row in enumerate(rows):
        lo=max(1,int(np.floor(row))-radius);hi=min(h-2,int(np.ceil(row))+radius+1)
        if i:lo=max(lo,int(np.ceil((rows[i-1]+row)/2)))
        if i<len(rows)-1:hi=min(hi,int(np.floor((row+rows[i+1])/2))+1)
        if hi-lo<3:raise ValueError('Antenna boundaries too close for independent windows')
        result.append([lo,hi])
    return result


def body_rows(image,paper=None):
    a=np.array(image.convert('RGBA'));h,w=a.shape[:2];opaque=a[...,3]>=128
    x=np.arange(int(w*.25),int(w*.75))
    if paper is not None:
        # Remove only columns that contain the paper at either end; ordinary
        # body pixels remain measurements, not a protected band.
        x=x[~paper[int(h*.8):,x].any(axis=0)]
    if len(x)<10:raise ValueError('Insufficient body columns independent of paper')
    top=np.argmax(opaque[:,x],axis=0);bottom=h-1-np.argmax(opaque[::-1,x],axis=0)
    return [float(np.median(top)),float(np.median(bottom))]


def hardware_ranges(image, controls_only=False):
    """controls_only keeps centred contours at least a fifth of the side wide:
    the buttons, not narrow scratches near the rim."""
    a=np.array(image.convert('RGBA'));h,w=a.shape[:2]
    gray=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY)
    edges=cv2.Canny(gray,8,24);edges[:int(.15*h)]=0;edges[int(.85*h):]=0
    edges[:,:int(.15*w)]=0;edges[:,int(.85*w):]=0
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    ranges=[];boxes=[]
    for c in contours:
        x,y,bw,bh=cv2.boundingRect(c)
        if bh<max(12,h*.015) or bh>h*.32 or bw<max(3,w*.05) or bh/max(1,bw)<1.5:continue
        if np.median(a[y:y+bh,x:x+bw,3])<240:continue
        if controls_only and (bw<w*.2 or not w*.35<=x+bw/2<=w*.65):continue
        ranges.append([max(0,y-5),min(h,y+bh+5)]);boxes.append([x,y,x+bw,y+bh])
    if not ranges:raise ValueError('No independent complete hardware contours detected')
    merged=[]
    for lo,hi in sorted(ranges):
        if merged and lo<=merged[-1][1]+3:merged[-1][1]=max(hi,merged[-1][1])
        else:merged.append([lo,hi])
    return merged,boxes


def _retained_upper_boundaries(image,rows):
    rows=np.rint(rows).astype(int)
    a=np.array(image.convert('RGBA'));h,w=a.shape[:2]
    colors=gaussian_filter1d(a[...,:3].astype(float).mean(axis=2),1.,axis=0)
    gradient=np.gradient(colors,axis=0)
    center_gradient=np.gradient(gaussian_filter1d(gray_signal(a,[int(w*.4),int(w*.6)]),1.))
    lines=[]
    band_widths=[rows[1]-rows[0]]*2+[rows[3]-rows[2]]*2
    for i,(row,width) in enumerate(zip(rows[:2],band_widths[:2])):
        radius=max(7,int(np.ceil(width*.6)))
        lo=max(1,row-radius);hi=min(h-1,row+radius+1)
        if i:lo=max(lo,int(np.ceil((rows[i-1]+row)/2)))
        if i<len(rows)-1:hi=min(hi,int(np.floor((row+rows[i+1])/2))+1)
        # Follow the same signed material transition across the strip.
        # Unsigned RGB magnitude can jump onto a nearby opposite-polarity glint.
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


def per_column_boundaries(image,rows,return_proof=False,*,revision_bands=(0,1)):
    # New full processing verifies both bands. A recorded lower-only repair
    # can explicitly preserve its previously accepted upper mapping.
    from .antenna_material import trace_boundaries
    if tuple(revision_bands)==(1,):
        lower,proof=trace_boundaries(image,rows,bands=(1,))
        result=np.vstack([_retained_upper_boundaries(image,rows),lower])
    elif tuple(revision_bands)==(0,1):
        result,proof=trace_boundaries(image,rows)
    else:raise ValueError('Unsupported antenna revision scope')
    if np.any(np.diff(result,axis=0)<=0):raise ValueError('Per-column antenna boundaries cross')
    proof=dict(proof,revision_bands=list(revision_bands),upper_algorithm_retained=tuple(revision_bands)==(1,))
    return (result,proof) if return_proof else result


def bind_lighting_sources(proposal, geometry, lighting):
    """Retain frozen pre-lighting landmarks; bind their same-coordinate RGB output."""
    import copy
    result=copy.deepcopy(proposal)
    for side,cfg in result.items():
        for face in (side,'back'):
            a=np.array(Image.open(geometry[face]).convert('RGBA'))
            b=np.array(Image.open(lighting[face]).convert('RGBA'))
            if a.shape!=b.shape or not np.array_equal(a[...,3],b[...,3]):
                raise ValueError('Lighting changed landmark coordinate system')
        cfg['measurement_source']=dict(path=str(geometry[side]),sha256=digest(geometry[side]))
        cfg['source_sha256']=digest(lighting[side])
        cfg['back']['measurement_source']=dict(path=str(geometry['back']),sha256=digest(geometry['back']))
        cfg['back']['sha256']=digest(lighting['back'])
        gate.mapping(lighting,side,cfg)
    return result


def source_antenna_rows(image,xr,geometry_path,context=None,face=None,expected_width=None):
    """Use independently bound original pixels only after geometry fails."""
    try:
        rows,signs,proof=antenna_rows(image,xr,expected_width,require_full_span=True)
        return image,rows,signs,proof,None
    except ValueError as exc:
        if context is None:raise
        geometry_error=str(exc)
    from .source_antenna_evidence import load
    detection=load(context,geometry_path,face)
    geometry_rows,_,geometry_proof=antenna_rows(image,xr,expected_width)
    try:rows,signs,proof=antenna_rows(detection,xr,expected_width,require_full_span=True)
    except ValueError as exc:
        original_error=str(exc)
        from .source_antenna_evidence import create_photometric
        try:
            detection,context=create_photometric(context,geometry_path,geometry_rows,face)
            rows,signs,proof=antenna_rows(detection,xr,expected_width,require_full_span=True)
        except ValueError as recovery_error:
            recovery_error.antenna_evidence_failures=dict(face=face,
                geometry=geometry_error,original_rgb=original_error,
                photometric=str(recovery_error),
                bands=getattr(recovery_error,'photometric_recovery_failures',[]))
            raise
    agreements=[]
    for band in (0,1):
        candidates=geometry_proof['supported_central_candidates'][band]
        distances=[float(np.max(np.abs(np.asarray(rows[2*band:2*band+2])-candidate))) for candidate in candidates]
        index=int(np.argmin(distances))
        if distances[index]>2.5:
            raise ValueError('Original source antenna band disagrees with geometry central material edges')
        agreements.append(dict(band=band,geometry_candidate=candidates[index],maximum_difference_px=distances[index]))
    proof['geometry_central_agreement']=agreements
    return detection,rows,signs,proof,context


def frozen_source_rows(image,x_range,cfg):
    """Revalidate measured source seeds without reranking a different band."""
    seeds=cfg.get('source_seed_rows')
    if seeds is None:
        selection=cfg.get('source_detection',{}).get('full_span_candidate_selection',[])
        if len(selection)==2 and [v.get('band') for v in selection]==[0,1]:
            seeds=[row for v in selection for row in v['rows']]
    if seeds is None:return antenna_rows(image,x_range,require_full_span=True)[0]
    seeds=np.asarray(seeds,float)
    if seeds.shape!=(4,) or not np.isfinite(seeds).all() or np.any(np.diff(seeds)<=2):
        raise ValueError('Invalid frozen source antenna seeds')
    width=float(np.mean([seeds[1]-seeds[0],seeds[3]-seeds[2]]))
    _,_,central=antenna_rows(image,x_range,width)
    for band in (0,1):
        candidates=central['supported_central_candidates'][band]
        if not any(np.max(np.abs(seeds[2*band:2*band+2]-rows))<=2.5 for rows in candidates):
            raise ValueError('Frozen antenna seeds lack independent current source central evidence')
    return seeds.tolist()


def propose(files, coordinate_mode='body_endpoints', antenna_evidence=None, sides=('left','right'), body_reference=None):
    if coordinate_mode not in ('body_endpoints','canvas_pixel_centers',sbr.MODE):
        raise ValueError('Unknown image alignment coordinates')
    sides=tuple(sides)
    if not sides or len(set(sides))!=len(sides) or any(side not in ('left','right') for side in sides):
        raise ValueError('Explicit unique left/right structure sides required')
    if coordinate_mode==sbr.MODE:
        if body_reference is None:raise ValueError('Explicit source body review evidence required')
        source_back_body=sbr.validate(body_reference,files['back'])
    elif body_reference is not None:
        raise ValueError('Source body reference requires explicit review coordinate mode')
    result={};back=Image.open(files['back']).convert('RGBA');b=np.array(back);h,w=b.shape[:2]
    y0,y1=int(h*.25),int(h*.75);opaque=b[y0:y1,:,3]>=128
    left=int(np.median(np.argmax(opaque,axis=1)));right=int(np.median(w-1-np.argmax(opaque[:,::-1],axis=1)))
    for side in sides:
        image=Image.open(files[side]).convert('RGBA');sh,sw=image.height,image.width
        xr=[max(1,int(sw*.40)),min(sw-1,int(sw*.60))]
        independent_back=None;source_width=None
        review_body=body_rows(image) if coordinate_mode==sbr.MODE else None
        reference_scale=((review_body[1]-review_body[0])/(source_back_body[1]-source_back_body[0])
                         if coordinate_mode==sbr.MODE else sh/h)
        try:
            independent_back=_rear_reference(back,side,right if side=='left' else left,None,3)
            br0=independent_back[0]
            source_width=float(np.mean([br0[1]-br0[0],br0[3]-br0[2]])*reference_scale)
        except ValueError:pass
        detection,rows,signs,proof,source_context=source_antenna_rows(image,xr,files[side],
            (antenna_evidence or {}).get(side),side,source_width)
        if independent_back is not None:
            proof['independent_back_width_seed']=dict(width_pixels=source_width,source_detection=independent_back[3])
        buttons,boxes=hardware_ranges(image)
        # The physical side identities are established by the named, upright
        # source capture convention, never by copying another SKU's targets.
        body=review_body if review_body is not None else body_rows(image)
        back_body=source_back_body if coordinate_mode==sbr.MODE else body_rows(back)
        expected_width=np.mean([rows[1]-rows[0],rows[3]-rows[2]])*(back_body[1]-back_body[0])/(body[1]-body[0])
        br,bs,back_x,bproof=rear_reference(back,side,right if side=='left' else left,expected_width)
        if 'width_prior_rejected' in bproof:
            expected_side_width=np.mean([br[1]-br[0],br[3]-br[2]])*reference_scale
            detection,rows,signs,proof,source_context=source_antenna_rows(image,xr,files[side],
                (antenna_evidence or {}).get(side),side,expected_side_width)
            proof['width_reference']='independent_current_back'
        desired=np.array([(v-back_body[0])/(back_body[1]-back_body[0])*(body[1]-body[0])+body[0] for v in br])
        if coordinate_mode==sbr.MODE:
            desired=np.asarray(sbr.targets(body_reference,br,body,sh),float)
        elif coordinate_mode=='canvas_pixel_centers':
            desired=(np.asarray(br,dtype=float)+.5)/h*sh-.5
        middle_lo=min(r[0] for r in buttons)-3;middle_hi=max(r[1] for r in buttons)+3
        if middle_lo<=max(rows[1],desired[1])+4 or middle_hi>=min(rows[2],desired[2])-4:
            # A narrow rim scratch can pass as a contour; retry with the buttons only.
            try:buttons,_=hardware_ranges(image,controls_only=True)
            except ValueError:buttons=[]
            if buttons:
                middle_lo=min(r[0] for r in buttons)-3;middle_hi=max(r[1] for r in buttons)+3
                proof['hardware_ranges']='controls_only_after_rim_contour_overlap'
            if (not buttons or middle_lo<=max(rows[1],desired[1])+4
                    or middle_hi>=min(rows[2],desired[2])-4):
                raise ValueError('Independent hardware contours overlap an antenna remapping zone')
        measured_lines,boundary_proof=per_column_boundaries(detection,rows,return_proof=True)
        anchors=np.vstack([measured_lines[0:2],np.full(sw,middle_lo),np.full(sw,middle_hi),measured_lines[2:4]])
        antenna_indices=[0,1,4,5];outline_ranges=[]
        if coordinate_mode in ('canvas_pixel_centers',sbr.MODE):
            cap=max(8,int(np.ceil(sw*.75)))
            if min(measured_lines[0].min(),desired[0])<=cap+4 or max(measured_lines[-1].max(),desired[-1])>=sh-1-cap-4:
                raise ValueError('Antenna remapping overlaps the rounded endcaps')
            anchors=np.vstack([np.full(sw,cap),anchors,np.full(sw,sh-1-cap)])
            antenna_indices=[1,2,5,6];outline_ranges=[[0,cap+1],[sh-1-cap,sh]]
        measurements={}
        for name,xx in [('left_end',[max(2,int(sw*.07)),max(6,int(sw*.20))]),
                        ('center',xr),('right_end',[int(sw*.80),min(sw-2,int(sw*.93))])]:
            local_signs=signs
            if 0 in local_signs:raise ValueError('Endpoint antenna polarity is ambiguous')
            measurements[name]=dict(x_range=xx,search_windows=windows(desired,sh,6),signs=local_signs)
        cfg=dict(source_sha256=digest(files[side]),source_seed_rows=list(rows),basis='back_image',body_y=body,coordinate_mode=coordinate_mode,
            source_lines=anchors.tolist(),antenna_indices=antenna_indices,button_ranges=buttons,outline_ranges=outline_ranges,
            button_basis='Independent source Canny hardware contours; complete bounding ranges expanded five pixels',
            hardware_boxes=boxes,source_detection=proof,source_boundary_detection=boundary_proof,
            back=dict(face='back',sha256=digest(files['back']),body_y=back_body,
                antenna_rows=br,x_range=back_x,search_windows=windows(br,h),signs=bs,
                physical_side=side,edge_in_back_image='right' if side=='left' else 'left',
                view_orientation='unmirrored_rear',source_detection=bproof),measurements=measurements)
        if coordinate_mode==sbr.MODE:
            cfg['back']['source_body_reference']=body_reference
            cfg['reference_review_required']=True
        if source_context is not None:cfg['antenna_source_evidence']=source_context
        gate.mapping(files,side,cfg)
        result[side]=cfg
    return result
