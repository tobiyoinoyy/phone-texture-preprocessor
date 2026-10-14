"""Opt-in extraction of a horizontal phone frame from a BiRefNet matte.

Uses two exposed end bands to extrapolate the frame rails across camera/stand
protrusions. It is not semantic segmentation, perspective rectification or an
occlusion repair. RGB is copied, never painted or resampled.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage

from scripts.geometry_postprocess import validate_geometry_size, rotate_alpha_safe


def _line(x, y):
    keep = np.ones(len(x), dtype=bool)
    for _ in range(4):
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        residual = y - (slope*x + intercept)
        median = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep]-median))
        updated = np.abs(residual-median) <= max(1.5, 3*1.4826*mad)
        if updated.sum() < 8 or np.array_equal(updated, keep):
            break
        keep = updated
    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    residual = np.abs(y-(slope*x+intercept))
    return np.array([slope,intercept]), keep, float(np.percentile(residual[keep],95))


def _rails(alpha, threshold, face="bottom"):
    labels, count = ndimage.label(alpha >= threshold, structure=np.ones((3,3)))
    if not count:
        raise ValueError('No reliable foreground for frame extraction')
    sizes=np.bincount(labels.ravel());sizes[0]=0
    mask=labels == sizes.argmax()
    _, xs=np.nonzero(mask)
    x0,x1=int(xs.min()),int(xs.max())+1
    width=x1-x0
    # Endcaps must be visible and clear of both camera and support.
    groups=[np.arange(int(x0+lo*width),int(x0+hi*width)) for lo,hi in ((.05,.12),(.88,.95))]
    x=np.concatenate(groups)
    if min(map(len,groups)) < 8 or not mask[:,x].any(axis=0).all():
        raise ValueError('Insufficient exposed frame endcaps')
    # Side views can have a camera island over one end. Use the central
    # exposed upper rail; lower endcaps remain clear of the central support.
    top_x = (np.arange(int(x0+.35*width), int(x0+.65*width))
             if face in ('left','right') else x)
    if len(top_x) < 16 or not mask[:,top_x].any(axis=0).all():
        raise ValueError('Insufficient exposed upper frame rail')
    tops=np.argmax(mask[:,top_x],axis=0).astype(float)
    bottoms=(alpha.shape[0]-np.argmax(mask[::-1,x],axis=0)).astype(float)
    top,kt,rt=_line(top_x+.5,tops)
    bottom,kb,rb=_line(x+.5,bottoms)
    if face in ('left','right'):
        # The unobstructed lower rail supplies the side-frame roll. A narrow
        # upper sample can have several pixels of matte drift, making a free
        # slope extrapolate into the camera-side frame. Use a parallel upper
        # envelope through the clear central region instead; no RGB is warped.
        intercepts=tops-bottom[0]*(top_x+.5)
        level=float(np.percentile(intercepts,10))
        residual=np.abs(intercepts-np.median(intercepts))
        kt=residual <= max(2.,3*1.4826*np.median(residual))
        rt=float(np.percentile(residual[kt],95))
        top=np.array([bottom[0],level])
    split_top = len(top_x)//2 if face in ('left','right') else len(groups[0])
    support=[float(kt[:split_top].mean()),float(kt[split_top:].mean()),
             float(kb[:len(groups[0])].mean()),float(kb[len(groups[0]):].mean())]
    return top,bottom,{'x_extent':[x0,x1], 'endcap_bands_xy':[[int(g[0]),int(g[-1])+1] for g in groups],
                       'side_upper_rail_constraint': ('parallel_to_lower_clear_rail' if face in ('left','right') else None),
                       'top_sample_bands_xy': ([[int(top_x[0]),int(top_x[-1])+1]] if face in ('left','right') else [[int(g[0]),int(g[-1])+1] for g in groups]),
                       'inlier_support_fractions':support,'rail_residual_p95_px':[rt,rb]}


def extract_frame_strip(image: Image.Image, *, face='bottom', margin_px=0.5):
    """Return a tightly cropped frame PNG and source-coordinate audit metadata.

    Inputs must show a horizontal frame. Top/bottom use exposed endcaps; sides
    use the central upper rail and lower endcaps. Unreliable boundaries raise
    ValueError; do not substitute a guessed rectangle. margin_px is a small
    outward allowance, not an inward erosion of the photographed frame.
    """
    if face not in ('top','bottom','left','right'):
        raise ValueError('Frame strip mode supports top, bottom, left and right')
    if not np.isfinite(margin_px) or not 0 <= margin_px <= 2:
        raise ValueError('margin_px must be between 0 and 2')
    validate_geometry_size(image.size)
    rgba=np.asarray(image.convert('RGBA')).copy()
    alpha=rgba[...,3]
    top,bottom,evidence=_rails(alpha,128,face)
    x0,x1=evidence['x_extent'];width=x1-x0
    endpoints=np.array([x0+.5,x1-.5])
    thickness=np.polyval(bottom,endpoints)-np.polyval(top,endpoints)
    if min(thickness) < .035*width or max(thickness) > .25*width:
        raise ValueError('Endcaps do not describe a thin horizontal phone frame')
    if max(abs(top[0]),abs(bottom[0])) > .08:
        raise ValueError('Frame is too tilted for endcap-strip mode')
    residual_limit = max(3., .002*width) if face in ('left','right') else 3.
    evidence['rail_residual_limit_px'] = residual_limit
    if min(evidence['inlier_support_fractions']) < .6 or max(evidence['rail_residual_p95_px']) > residual_limit:
        raise ValueError('Exposed frame boundaries are inconsistent; manual review required')
    alternate=[]
    for threshold in (64,192):
        other_top,other_bottom,_=_rails(alpha,threshold,face)
        alternate.append(np.concatenate((np.polyval(other_top,endpoints),np.polyval(other_bottom,endpoints))))
    reference=np.concatenate((np.polyval(top,endpoints),np.polyval(bottom,endpoints)))
    spread=float(np.max(np.abs(np.asarray(alternate)-reference)))
    if spread > 3:
        raise ValueError('Frame boundary is unstable across alpha thresholds')
    xx=np.arange(rgba.shape[1],dtype=float)[None,:]+.5
    yy=np.arange(rgba.shape[0],dtype=float)[:,None]
    upper=top[0]*xx+top[1]-margin_px
    lower=bottom[0]*xx+bottom[1]+margin_px
    coverage=np.clip(np.minimum(yy+1,lower)-np.maximum(yy,upper),0,1)
    rgba[...,3]=np.rint(alpha*coverage).astype(np.uint8)
    ys,xs=np.nonzero(rgba[...,3]>=128)
    if not len(xs):
        raise ValueError('Frame mask is empty')
    crop=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
    cx0,cy0,cx1,cy1=crop
    result=Image.fromarray(rgba[cy0:cy1,cx0:cx1])
    return result, {
        'method':('exposed_side_frame_strip_v2' if face in ('left','right') else 'exposed_endcap_frame_strip_v1'),'face':face,
        'source_size':list(image.size),'output_size':list(result.size),
        'top_rail_y_from_x':top.tolist(),'bottom_rail_y_from_x':bottom.tolist(),
        'outward_margin_px':float(margin_px),'crop_xyxy':crop,
        'alpha_threshold_rail_difference_px':spread,**evidence,
        'rgb_resampled':False,'rgb_painted':False,'perspective_corrected':False,
        'outside_frame_policy':'mask_outside_extrapolated_rails',
        'requires_visible_clear_endcaps':True,'needs_visual_review':True,
    }


def normalize_frame_strip(image: Image.Image, *, face='bottom', deskew=None, stage_sink=None):
    """Extract, deskew, tightly crop, then apply the named view orientation.

    All four frame views default to deskew. Pass deskew=False to reproduce
    legacy extraction-only behavior. Quarter turns
    are exact pixel permutations; only the small deskew rotation resamples.
    """
    result, info = extract_frame_strip(image, face=face)
    if stage_sink:
        stage_sink('source_matte', image.copy())
        source_space = Image.new('RGBA', image.size)
        source_space.paste(result, tuple(info['crop_xyxy'][:2]))
        stage_sink('after_support', source_space)
    if deskew is None:
        deskew = True
    slopes=[info['top_rail_y_from_x'][0],info['bottom_rail_y_from_x'][0]]
    angle=float(np.degrees(np.mean(np.arctan(slopes)))) if deskew else 0.0
    # Avoid a redundant interpolation for mathematically horizontal rails.
    if abs(angle) < 1e-5:
        angle=0.0
    if angle:
        rotated=rotate_alpha_safe(np.asarray(result), angle)
        ys,xs=np.nonzero(rotated[...,3]>=128)
        crop=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
        result=Image.fromarray(rotated[crop[1]:crop[3],crop[0]:crop[2]])
    else:
        crop=[0,0,result.width,result.height]
    horizontal_size=list(result.size)
    if stage_sink:
        stage_sink('after_rotation', result.copy())
    turns={'right':-1,'left':1}.get(face,0)
    if turns:
        result=result.transpose(Image.Transpose.ROTATE_270 if turns==-1 else Image.Transpose.ROTATE_90)
    info.update({
        'extracted_size':info['output_size'],
        'deskew_rotation_ccw_degrees':angle,
        'post_deskew_crop_xyxy':crop,
        'post_deskew_crop_space':'expanded rotation of extracted frame',
        'horizontal_size':horizontal_size,
        'orientation_quarter_turns_ccw':turns,
        'output_size':list(result.size),
        'rgb_resampled':bool(angle),
        'resampling_passes':int(bool(angle)),
        'quarter_turn_resampled':False,
        'operation_order':['extract_frame','deskew','tight_crop','face_quarter_turn'],
    })
    return result,info


def deskew_cropped_frame(image: Image.Image, *, face):
    """Deskew an already extracted top/bottom PNG without masking its contents.

    Fit clear end bands, avoiding central labels/ports. Rotate the complete
    original RGBA once with premultiplied alpha, then crop only fully transparent
    outer rows/columns. No segmentation, perspective warp, or size normalization.
    """
    if face not in ('top', 'bottom'):
        raise ValueError('Expected an already horizontal top or bottom frame')
    validate_geometry_size(image.size)
    rgba = np.asarray(image.convert('RGBA'))
    top, bottom, evidence = _rails(rgba[..., 3], 128, face)
    angles = np.degrees(np.arctan([top[0], bottom[0]]))
    angle = float(np.mean(angles))
    if abs(angle) > 4 or abs(angles[0] - angles[1]) > .3:
        raise ValueError('Ambiguous frame angle or perspective; review required')
    if min(evidence['inlier_support_fractions']) < .6 or max(evidence['rail_residual_p95_px']) > 3:
        raise ValueError('Unreliable frame boundary; review required')
    alternate = []
    for threshold in (64, 192):
        t, b, _ = _rails(rgba[..., 3], threshold, face)
        alternate.append(float(np.degrees(np.mean(np.arctan([t[0], b[0]])))))
    if max(abs(value - angle) for value in alternate) > .1:
        raise ValueError('Frame angle unstable across alpha thresholds')
    if abs(angle) < 1e-5:
        angle = 0.
    rotated = rotate_alpha_safe(rgba, angle) if angle else rgba.copy()
    ys, xs = np.nonzero(rotated[..., 3] > 0)
    box = [int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)]
    result = Image.fromarray(rotated[box[1]:box[3], box[0]:box[2]])
    t, b, _ = _rails(np.asarray(result)[..., 3], 128, face)
    residual = np.degrees(np.arctan([t[0], b[0]]))
    if abs(float(np.mean(residual))) > .06 or max(abs(residual)) > .15:
        raise ValueError('Residual tilt exceeds limit; review required')
    return result, {'method': 'cropped_frame_rigid_deskew_v1', 'face': face,
                    'source_size': list(image.size), 'output_size': list(result.size),
                    'before_rail_angles_degrees': angles.tolist(),
                    'rotation_ccw_degrees': angle, 'after_rail_angles_degrees': residual.tolist(),
                    'crop_xyxy': box, 'resampling_passes': int(bool(angle)),
                    'rgb_painted': False, 'perspective_corrected': False,
                    'alpha_content_masked': False, **evidence}
