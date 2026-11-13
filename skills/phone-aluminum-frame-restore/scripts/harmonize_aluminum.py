#!/usr/bin/env python3
"""Source-coordinate aluminum color harmonization; Pillow + NumPy only.

Inputs are a reviewed per-view job, material/protection masks, and optional AI
color reference. Never substitutes generated geometry or texture for the source.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# Isolated editable pixels allowed to lack local source support per domain.
# They are left out of `reliable`, so smooth_material_correction gives them
# the neighbouring supported correction; RGB is never copied or filled.
CONTOUR_UNSUPPORTED_TOLERANCE_PX = 16       # opaque (alpha >= 128)
SOFT_FRINGE_UNSUPPORTED_TOLERANCE_PX = 256  # semi-transparent edge (alpha < 128)
SOFT_FRINGE_MAX_DISTANCE_PX = 3             # ... and touching supported pixels


def tolerated_unsupported(missing, alpha, supported):
    """True when unsupported pixels are only a few opaque ones plus a thin
    antialias fringe hugging supported material. A soft component away from
    any supported pixel is unrepresented material and is still rejected."""
    if not missing.any():
        return True
    if alpha is None or not supported.any():
        return int(missing.sum()) <= CONTOUR_UNSUPPORTED_TOLERANCE_PX
    from scipy.ndimage import distance_transform_edt
    opaque_missing = missing & (alpha >= 128)
    # 2026-11-09: a soft row under a tolerated opaque row (paper end at the
    # canvas edge) hugs that row, which itself borrows the neighbour field.
    anchor = supported | (opaque_missing if int(opaque_missing.sum()) <= CONTOUR_UNSUPPORTED_TOLERANCE_PX else False)
    near = distance_transform_edt(~anchor) <= SOFT_FRINGE_MAX_DISTANCE_PX
    fringe = missing & (alpha < 128) & near
    rest = missing & ~fringe
    if int(rest.sum()) > CONTOUR_UNSUPPORTED_TOLERANCE_PX:
        # 2026-11-09: a stand contact can thin a short stretch of the outer rim.
        # A gap at most 4 rows tall and 4% of the width long, with supported
        # material on both ends of every row, borrows the correction from both
        # sides; anything taller, longer or open-ended is still unrepresented.
        from scipy.ndimage import label
        labels, count = label(rest)
        span = max(16, int(.04 * missing.shape[1]))
        for index in range(1, count + 1):
            yy, xx = np.nonzero(labels == index)
            if np.ptp(yy) >= 4 or np.ptp(xx) > span:
                continue
            # Supported material must flank the gap at both ends (within 3 px).
            y0, y1 = yy.min(), yy.max()
            left = supported[max(0, y0 - 3):y1 + 4, max(0, xx.min() - 3):xx.min()].any()
            right = supported[max(0, y0 - 3):y1 + 4, xx.max() + 1:xx.max() + 4].any()
            if left and right:
                rest[labels == index] = False
    return (int(rest.sum()) <= CONTOUR_UNSUPPORTED_TOLERANCE_PX
            and int(fringe.sum()) <= SOFT_FRINGE_UNSUPPORTED_TOLERANCE_PX)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rgb_to_lab(rgb):
    s = np.asarray(rgb, dtype=float) / 255
    lin = np.where(s <= .04045, s / 12.92, ((s + .055) / 1.055) ** 2.4)
    xyz = np.einsum('...j,ij->...i', lin, np.array([[.4124564, .3575761, .1804375],
                          [.2126729, .7151522, .0721750],
                          [.0193339, .1191920, .9503041]]))
    v = xyz / np.array([.95047, 1., 1.08883])
    f = np.where(v > (6/29)**3, np.cbrt(v), v / (3*(6/29)**2) + 4/29)
    return np.stack([116*f[...,1]-16, 500*(f[...,0]-f[...,1]),
                     200*(f[...,1]-f[...,2])], axis=-1)


def lab_to_linear_rgb(lab):
    fy = (lab[...,0]+16)/116
    f = np.stack([fy+lab[...,1]/500, fy, fy-lab[...,2]/200], axis=-1)
    xyz = np.where(f > 6/29, f**3, 3*(6/29)**2*(f-4/29))
    xyz *= np.array([.95047, 1., 1.08883])
    lin = np.einsum('...j,ij->...i', xyz, np.linalg.inv(np.array([[.4124564,.3575761,.1804375],
                                      [.2126729,.7151522,.0721750],
                                      [.0193339,.1191920,.9503041]])))
    return lin


def lab_to_rgb(lab):
    lin = np.maximum(lab_to_linear_rgb(lab), 0)
    srgb = np.where(lin <= .0031308, lin*12.92, 1.055*lin**(1/2.4)-.055)
    return np.clip(np.rint(srgb*255), 0, 255).astype(np.uint8)


def quantize_continuous_illumination(lab, source_rgb, region, max_step):
    """Choose only adjacent RGB code values, preserving a source-bound L field.

    Dynamic programming minimizes RGB rounding error within each column. A
    source-material pair prefers a representation whose added L step is within
    the configured limit. If no corner path exists, restart without changing
    the fitted field; ordinary output quality checks still decide acceptance.
    No source texture, float correction, alpha or unlisted pixel is filtered.
    """
    limit=float(max_step);region=np.asarray(region,bool)
    if not np.isfinite(limit) or not 0<limit<=5:
        raise ValueError('Invalid quantization illumination continuity bound')
    if region.shape!=lab.shape[:2] or np.shape(source_rgb)!=lab.shape:
        raise ValueError('Quantization source and material coordinates differ')
    result=lab_to_rgb(lab)
    if not region.any():return result,dict(method='adjacent_RGB_corners_minimum_rounding_cost_v1',changed_pixels=0,infeasible_restarts=0)
    ys,xs=np.nonzero(region);lo,hi=int(ys.min()),int(ys.max())+1;cols=np.unique(xs)
    target=lab[lo:hi,cols];source=rgb_to_lab(np.asarray(source_rgb)[lo:hi,cols]);mask=region[lo:hi,cols]
    lin=np.maximum(lab_to_linear_rgb(target),0)
    srgb=np.clip(np.where(lin<=.0031308,lin*12.92,1.055*lin**(1/2.4)-.055)*255,0,255)
    corners=[];corrections=[];cost=[]
    for code in range(8):
        corner=np.stack([np.ceil(srgb[...,k]) if code&(1<<k) else np.floor(srgb[...,k]) for k in range(3)],axis=-1).astype(np.uint8)
        corners.append(corner);corrections.append(rgb_to_lab(corner)[...,0]-source[...,0]);cost.append(np.sum((corner.astype(float)-srgb)**2,axis=2))
    corners=np.array(corners);corrections=np.array(corrections);cost=np.array(cost)
    h,w=mask.shape;xx=np.arange(w);back=np.zeros((h,8,w),np.uint8);dp=cost[:,0].copy();restarts=0
    for y in range(1,h):
        linked=mask[y]&mask[y-1]
        forbidden=linked[None,None,:]&(abs(corrections[:,y-1][:,None,:]-corrections[:,y][None,:,:])>limit)
        transition=dp[:,None,:]+np.where(forbidden,np.inf,0)
        prev=np.argmin(transition,axis=0);best=np.take_along_axis(transition,prev[None],axis=0)[0]
        invalid=~np.isfinite(best).any(axis=0)
        if invalid.any():
            restarts+=int(invalid.sum());best[:,invalid]=np.min(dp[:,invalid],axis=0);prev[:,invalid]=np.argmin(dp[:,invalid],axis=0)
        back[y]=prev;dp=cost[:,y]+best
    chosen=np.zeros((h,w),np.uint8);chosen[-1]=np.argmin(dp,axis=0)
    for y in range(h-1,0,-1):chosen[y-1]=back[y,chosen[y],xx]
    selected=corners[chosen,np.arange(h)[:,None],xx[None]]
    original=result.copy();roi=result[lo:hi,cols].copy();roi[mask]=selected[mask];result[lo:hi,cols]=roi
    changed=np.any(result!=original,axis=2)
    if np.any(changed&~region) or np.max(abs(result.astype(int)-original.astype(int)))>1:
        raise RuntimeError('Quantization exceeded its source-bound adjacent RGB corners')
    return result,dict(method='adjacent_RGB_corners_minimum_rounding_cost_v1',max_added_step_L=limit,
        changed_pixels=int(changed.sum()),maximum_RGB_code_change=int(abs(result.astype(int)-original.astype(int)).max()),
        infeasible_restarts=restarts,float_field_changed=False,unlisted_pixels_changed=0)


def preserve_chroma_in_gamut(desired, source):
    """Limit only L displacement; source a/b remain unchanged in float Lab."""
    result = np.array(desired, dtype=float, copy=True)
    result[..., 1:] = source[..., 1:]
    linear = lab_to_linear_rgb(result)
    bad = np.any((linear < -1e-7) | (linear > 1+1e-7), axis=-1)
    if bad.any():
        src, dst = source[bad], result[bad]
        delta = dst[..., 0] - src[..., 0]
        lo, hi = np.zeros(len(src)), np.ones(len(src))
        for _ in range(24):
            mid = (lo+hi)/2
            trial = dst.copy()
            trial[..., 0] = src[..., 0] + delta*mid
            rgb = lab_to_linear_rgb(trial)
            valid = np.all((rgb >= -1e-7) & (rgb <= 1+1e-7), axis=-1)
            lo = np.where(valid, mid, lo)
            hi = np.where(valid, hi, mid)
        dst[..., 0] = src[..., 0] + delta*lo
        result[bad] = dst
    return result, int(bad.sum())


def color_mode(config):
    mode = config.get('color_mode', 'source_lightness')
    if mode not in ('source_lightness', 'shared_color'):
        raise ValueError('Unknown color_mode: '+str(mode))
    return mode


def continuous_gamut_limits(desired, safe, source, mask, max_step, seed_mask=None,
                            follow_field=False, follow_source_edges=False,
                            source_edge_weight=None):
    """Propagate actual gamut caps through one material's correction field.

    Only reduce L displacement toward the source. Source texture/chroma are
    never filtered, and an unbounded field is returned exactly unchanged.
    Paper, other materials and opposite-sign corrections block propagation.
    """
    from scipy.ndimage import minimum_filter
    max_step = float(max_step)
    if not np.isfinite(max_step) or max_step <= 0:
        raise ValueError('Gamut continuity step must be finite and positive')
    wanted = desired[..., 0] - source[..., 0]
    if source_edge_weight is None:
        source_edge_weight = np.ones(mask.shape)
    source_edge_weight = np.asarray(source_edge_weight, dtype=float)
    if (source_edge_weight.shape != mask.shape or not np.isfinite(source_edge_weight).all()
            or np.any((source_edge_weight < 0) | (source_edge_weight > 1))):
        raise ValueError('Source-edge weights require finite source-sized values in [0, 1]')
    feature = np.zeros(mask.shape)
    feature_support = np.zeros(mask.shape, bool)
    if follow_source_edges:
        from scipy.ndimage import median_filter
        # Detect narrow source impulses relative to the local source only.
        # Ordinary grain/gradients do not gain a larger propagation budget.
        feature = source[..., 0] - median_filter(source[..., 0], size=3, mode='reflect')
        noise = max(1., float(np.percentile(np.abs(feature[mask]), 75))*3.)
        feature_support = np.abs(feature) > noise
    bounded = safe[..., 0] - source[..., 0]
    limited = mask & (np.abs(bounded) < np.abs(wanted) - 1e-6)
    # Mixed/background-contaminated edge RGB still receives its own safe
    # correction, but must not dim reliable body pixels through propagation.
    if seed_mask is not None:
        seed_mask = np.asarray(seed_mask, dtype=bool)
        if seed_mask.shape != mask.shape:
            raise ValueError('Gamut seed mask dimensions differ from source')
        limited &= seed_mask
    result = safe.copy()
    changed = np.zeros(mask.shape, bool)
    # Four-neighbor paths cannot jump diagonally across a material/label gap.
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
    for sign in (1., -1.):
        region = mask & (sign * wanted > 0)
        seeds = region & limited
        if not seeds.any():
            continue
        magnitude = np.maximum(0., sign * bounded)
        cap = np.where(seeds, magnitude, np.inf)
        # Beyond this radius even a zero cap cannot reduce the largest shift.
        for _ in range(int(np.ceil(float(magnitude[region].max()) / max_step)) + 1):
            if follow_field or follow_source_edges:
                # A narrow photographic shadow needs a steep *smooth* L lift.
                # A fixed displacement slope incorrectly propagates bright
                # pixels' small gamut budgets into that dark rail. Allow the
                # already fitted correction gradient, while spreading only
                # additional gamut reductions. Never filter source RGB/Lab.
                updated = cap.copy()
                for axis in (0, 1):
                    for offset in (-1, 1):
                        neighbor = np.roll(cap, offset, axis)
                        slope = (np.abs(wanted - np.roll(wanted, offset, axis))
                                 if follow_field else 0.)
                        if follow_source_edges:
                            # A real bright scratch can have almost no gamut
                            # headroom while the adjacent shadowed metal has
                            # plenty. Its source contrast bounds the permitted
                            # cap discontinuity; do not turn that tiny feature
                            # into a broad dark halo. Source RGB is untouched.
                            weight = np.minimum(source_edge_weight,
                                np.roll(source_edge_weight, offset, axis))
                            impulse_edge = feature_support | np.roll(feature_support, offset, axis)
                            slope = slope + weight * impulse_edge * np.abs(feature -
                                np.roll(feature, offset, axis))
                        edge = [slice(None), slice(None)]
                        edge[axis] = 0 if offset == 1 else -1
                        neighbor[tuple(edge)] = np.inf
                        updated = np.minimum(updated, neighbor + max_step + slope)
                updated = np.where(region, updated, np.inf)
            else:
                updated = np.where(region, np.minimum(cap, minimum_filter(
                    cap, footprint=cross, mode='constant', cval=np.inf) + max_step), np.inf)
            if np.array_equal(updated, cap):
                break
            cap = updated
        affected = region & (cap < magnitude)
        result[affected, 0] = source[affected, 0] + sign * cap[affected]
        changed |= affected
    return result, dict(max_step_L=max_step, gamut_seed_pixels=int(limited.sum()),
                        adjusted_pixels=int(changed.sum()),
                        method='same_material_gamut_cap_propagation',
                        follows_fitted_correction_gradient=bool(follow_field),
                        follows_source_edges=bool(follow_source_edges))


def gaussian(arr, sigma):
    """Float separable Gaussian; reflection padding avoids dark image borders."""
    out = np.asarray(arr, dtype=float)
    sy, sx = sigma if isinstance(sigma, (list, tuple)) else (sigma, sigma)
    for axis, std in [(0, sy), (1, sx)]:
        if std <= 0:
            continue
        radius = max(1, int(np.ceil(3*std)))
        xx = np.arange(-radius, radius+1)
        kernel = np.exp(-.5*(xx/std)**2)
        kernel /= kernel.sum()
        pad = [(0, 0)]*out.ndim
        pad[axis] = (radius, radius)
        p = np.pad(out, pad, mode='reflect')
        out = np.apply_along_axis(lambda a: np.convolve(a, kernel, mode='valid'), axis, p)
    return out


def shapes_mask(size, shapes):
    im = Image.new('L', size)
    draw = ImageDraw.Draw(im)
    for shape in shapes:
        kind, coordinates = shape['type'], shape['xy']
        if kind == 'rectangle':
            draw.rectangle(coordinates, fill=255)
        elif kind == 'ellipse':
            draw.ellipse(coordinates, fill=255)
        elif kind == 'polygon':
            draw.polygon([tuple(p) for p in coordinates], fill=255)
        else:
            raise ValueError('Unknown shape: '+kind)
    return np.array(im) > 0


def supported_field(lab, fit, editable, sigma, alpha=None):
    """Never extrapolate an unsupported editable component from another surface.

    Shared by CLI and custom batch scripts. The fixed floor is a numerical
    guard, not a guarantee that the selected samples are the right material.
    Values outside reliable support are zero and must not enter smoothing.
    """
    support = gaussian(fit.astype(float), sigma)
    reliable = support >= .008
    missing = editable & ~reliable
    # A soft paper terminal is resolved by its own wider source sigma
    # (printed_label_sigma), not by borrowing a neighbour field: count-only.
    # 2026-11-09: pixels within 3 px of the canvas edge are a body cut (paper
    # ended by the stand cut), not a terminal; only there may a thin fringe
    # borrow the neighbour correction like a rim fringe.
    edge = np.zeros(missing.shape, bool)
    if alpha is not None:
        edge[:3] = edge[-3:] = True; edge[:, :3] = edge[:, -3:] = True
    if not (tolerated_unsupported(missing & ~edge, None, editable & reliable)
            and tolerated_unsupported(missing & edge, alpha, editable & reliable)):
        y, x = np.nonzero(missing)
        raise ValueError(f'Unsupported editable illumination: {len(x)} pixels; '
                         f'bounds [{x.min()}, {y.min()}, {x.max()}, {y.max()}]. '
                         'Split the component and fit its own reviewed source samples; '
                         'nearest-neighbor or median hole filling is not permitted.')
    low = np.stack([gaussian(lab[..., c]*fit, sigma)/np.maximum(support, 1e-9)
                    for c in range(3)], axis=-1)
    low[~reliable] = 0
    return low, support, reliable


def feature_excluded_field(lab, fit, editable, sigma, feature, alpha=None):
    """Estimate illumination around a confirmed albedo mark, not from its ink.

    Enlarge the normalized source convolution only where the excluded feature
    removes local support. Both scales use real surrounding material samples;
    the same 0.008 support gate applies. No median/nearest fill or RGB blur.
    """
    from scipy.ndimage import distance_transform_edt
    if feature.shape != fit.shape or not feature.any() or np.any(feature&fit):
        raise ValueError('Feature field needs a nonempty excluded source feature')
    radius=float(distance_transform_edt(feature).max())
    wide_sigma=np.maximum(np.asarray(sigma,dtype=float),radius*.8).tolist()
    wide,wide_support,_=supported_field(lab,fit,editable,wide_sigma,alpha)
    support=gaussian(fit.astype(float),sigma)
    low,_,_=supported_field(lab,fit,editable&(support>=.008),sigma,alpha)
    t=np.clip((.65-support)/.5,0,1);weight=t*t*(3-2*t)
    # At unrelated domain edges preserve the established field exactly.
    taper=np.clip(4-distance_transform_edt(~feature)/max(wide_sigma),0,1)
    weight*=taper*taper*(3-2*taper)
    low=low*(1-weight[...,None])+wide*weight[...,None]
    support=support*(1-weight)+wide_support*weight
    if np.any(editable&(support<.008)):
        raise ValueError('Unsupported feature-excluded illumination')
    return low,support,support>=.008


def component_offset(source, accepted, component, reference, feather=2):
    """Repair one reviewed component from source texture with a constant Lab offset.

    The accepted image supplies color only in explicitly reviewed stable samples.
    It supplies no texture inside the repair. Outside the component is exact.
    """
    if source.shape != accepted.shape or source.ndim != 3 or source.shape[2] != 4:
        raise ValueError('Source and accepted image must have matching RGBA geometry')
    if not np.array_equal(source[..., 3], accepted[..., 3]):
        raise ValueError('Source and accepted alpha differ')
    component = np.asarray(component, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if component.shape != source.shape[:2] or reference.shape != component.shape:
        raise ValueError('Component masks must match source dimensions')
    if np.any(component & (source[..., 3] == 0)) or np.any(reference & ~component):
        raise ValueError('References must be inside the visible reviewed component')
    if reference.sum() < 20:
        raise ValueError('Insufficient reviewed component reference samples')
    lab = rgb_to_lab(source[..., :3])
    delta = np.median((rgb_to_lab(accepted[..., :3])-lab)[reference], axis=0)
    corrected = lab_to_rgb(lab + delta)
    weight = inward_weight(component, feather)
    dst = accepted.copy()
    mixed = np.clip(np.rint(corrected*weight[..., None]
                    + accepted[..., :3]*(1-weight[..., None])), 0, 255).astype(np.uint8)
    dst[component, :3] = mixed[component]
    return dst, {'method': 'source_texture_constant_component_lab_offset',
                 'offset_lab': delta.tolist(), 'reference_pixels': int(reference.sum()),
                 'component_pixels': int(component.sum()), 'feather': feather}


def inward_weight(mask, radius):
    """Discrete interior distance ramp, exactly zero outside reviewed mask."""
    if radius < 0:
        raise ValueError('Feather radius must be nonnegative')
    if radius == 0:
        return mask.astype(float)
    steps = max(1, int(radius))
    im = Image.fromarray(mask.astype(np.uint8)*255)
    total = mask.astype(float)
    for _ in range(steps):
        im = im.filter(ImageFilter.MinFilter(3))
        total += np.array(im)/255
    x = np.clip(total/(steps+1), 0, 1)
    return (x*x*(3-2*x))*mask


def target_color(job):
    colors = []
    for item in job['views']:
        a = np.array(Image.open(item['source']).convert('RGBA'))
        for box in item['reference_boxes']:
            x0,y0,x1,y1 = box
            roi = a[y0:y1,x0:x1]
            valid = roi[...,3] >= 250
            if not valid.any():
                raise ValueError('Empty reference box')
            colors.append(np.median(rgb_to_lab(roi[...,:3])[valid], axis=0))
    target = np.median(colors, axis=0)
    info = {'source_reference_lab': target.tolist()}
    ref = job.get('ai_reference')
    if ref:
        a = np.array(Image.open(ref['path']).convert('RGBA'))
        x0,y0,x1,y1 = ref['box']
        roi = a[y0:y1,x0:x1]
        valid = roi[...,3] >= 250
        if not valid.any():
            raise ValueError('Empty AI reference box')
        candidate = np.median(rgb_to_lab(roi[...,:3])[valid], axis=0)
        limit = np.asarray(ref.get('max_lab_shift', [8., 5., 5.]))
        delta = np.clip(candidate-target, -limit, limit)
        target = target + float(ref.get('weight', .25))*delta
        info.update(ai_reference_lab=candidate.tolist(), ai_reference_sha256=digest(ref['path']))
    if 'target_lab' in job:
        target = np.array(job['target_lab'], dtype=float)
    info['target_lab'] = target.tolist()
    if not np.isfinite(target).all():
        raise ValueError('Non-finite target color')
    info['target_rgb'] = lab_to_rgb(target).tolist()
    return target, info


def smooth_material_correction(shift, reliable, fit, editable, sigma, mode='legacy'):
    """Smooth an additive field without treating interpolated pixels as source.

    Source fit coverage continuously weights the independent fit estimate. A
    protrusion or soft edge without nearby fit pixels keeps its already
    supported field; it is never assigned zero correction or made uneditable.
    Historical jobs retain their original smoothing unless explicitly opted in.
    """
    denominator=gaussian(reliable.astype(float),sigma)
    original=gaussian(shift*reliable[...,None],sigma)/np.maximum(denominator[...,None],1e-9)
    if mode=='legacy':return original,None
    if mode!='source_fit_coverage':raise ValueError('Unknown correction smoothing mode')
    source_fit=fit&reliable
    fit_denominator=gaussian(source_fit.astype(float),sigma)
    coverage=fit_denominator/np.maximum(denominator,1e-9)
    result=((1-coverage[...,None])*original
            +gaussian(shift*source_fit[...,None],sigma)/np.maximum(denominator[...,None],1e-9))
    if np.any(denominator[editable]<=1e-9) or not np.isfinite(result[editable]).all():
        raise ValueError('Unsupported source-fit correction smoothing')
    no_fit=editable&(fit_denominator==0)
    info=dict(mode=mode,minimum_reliable_support=float(denominator[editable].min()),
              source_fit_coverage_range=[float(coverage[editable].min()),float(coverage[editable].max())],
              zero_local_fit_pixels=int(no_fit.sum()),
              zero_local_fit_max_legacy_difference=float(np.max(np.abs(result[no_fit]-original[no_fit]))) if no_fit.any() else 0.)
    return result,info


def supported_contour_field(lab, fit, editable, alpha, sigma, inset_depth=5., source_inset_limit=None):
    """Tangent/normal source sampling; include curved rims and alpha fringes.

    Requires SciPy (included by phone-texture-preprocessor). No nearest-neighbor
    fill or lowering support threshold when a material is not represented.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates
    tangent, normal = np.asarray(sigma,dtype=float)
    if not np.isfinite([tangent,normal]).all() or tangent<=0 or normal<=0:
        raise ValueError('Contour sigma must be two finite positive values')
    if not np.isfinite(inset_depth) or inset_depth<0:
        raise ValueError('Contour inset depth must be finite and nonnegative')
    inside=alpha>=128
    if source_inset_limit is not None:
        # Compute normals before cropping the transparent exterior. Reflecting
        # an already cropped distance field tilts normals at a tight canvas
        # edge and makes the result depend on incidental transparent margins.
        # The pad covers Gaussian's default four-sigma support plus gradient.
        pad=13;extended=np.pad(inside,pad)
        signed=distance_transform_edt(extended)-distance_transform_edt(~extended)
        gy,gx=np.gradient(gaussian_filter(signed,3))
        dist=signed[pad:-pad,pad:-pad]
        gy,gx=gy[pad:-pad,pad:-pad],gx[pad:-pad,pad:-pad]
    else:
        dist=distance_transform_edt(np.pad(inside,1))[1:-1,1:-1]-distance_transform_edt(~inside)
        gy,gx=np.gradient(gaussian_filter(dist,3))
    norm=np.maximum(np.hypot(gx,gy),1e-8)
    nx,ny=gx/norm,gy/norm;y,x=np.nonzero(editable)
    inset=np.maximum(0,inset_depth-dist[y,x])
    if source_inset_limit is not None:
        if not np.isfinite(source_inset_limit) or not 0<source_inset_limit<=min(alpha.shape)*.1:
            raise ValueError('Invalid source-normal sampling bound')
        # Locate the actual opaque material along each inward normal. A single
        # widest alpha shoulder elsewhere on the rim must not move a thin
        # bottom rim's sampling window across its glass boundary. These are
        # sampling coordinates only, never nearest-pixel RGB replacement.
        found=np.zeros(len(x),bool);inset=np.zeros(len(x));best=np.full(len(x),-1.)
        for offset in np.arange(0,float(source_inset_limit)+1e-12,.5):
            coordinates=[y+ny[y,x]*offset,x+nx[y,x]*offset]
            # Classify the discrete source mask without averaging away a
            # genuine one-pixel metal strip. RGB integration below remains
            # weighted/bilinear and must pass the unchanged .008 support gate.
            supported=map_coordinates(fit.astype(np.uint8),coordinates,order=0,mode='constant',cval=0)>0
            material=map_coordinates(editable.astype(np.uint8),coordinates,order=0,mode='constant',cval=0)>0
            coverage=map_coordinates(alpha.astype(float),coordinates,order=0,mode='constant',cval=0)
            # A short translucent span can use opaque neighbours along the
            # same tangent. Keep its sampling centre inside actual material;
            # the integration must still have >=.008 opaque fit support.
            score=coverage+supported*256.
            take=material&(score>best);inset[take]=offset;best[take]=score[take];found |= material
        if not found.all():
            missing=~found
            raise ValueError(f'No same-material samples on bounded contour normals: {int(missing.sum())} pixels; '
                             f'bounds [{int(x[missing].min())}, {int(y[missing].min())}, {int(x[missing].max())}, {int(y[missing].max())}]')
    # Interpolate against the same zero-valued exterior whether transparent
    # padding is explicit or implicit; `constant` drops fractional border
    # samples wholesale and therefore changes when the canvas is padded.
    sample_mode='grid-constant' if source_inset_limit is not None else 'constant'
    num=np.zeros((len(x),3));den=np.zeros(len(x));total=0.
    for t in np.linspace(-2.5*tangent,2.5*tangent,19):
        for n in np.linspace(-2*normal,2*normal,5):
            coordinates=[y+nx[y,x]*t+ny[y,x]*(n+inset),x-ny[y,x]*t+nx[y,x]*(n+inset)]
            weight=np.exp(-.5*(t/tangent)**2-.5*(n/normal)**2);total+=weight
            den+=weight*map_coordinates(fit.astype(float),coordinates,order=1,mode=sample_mode,cval=0)
            for channel in range(3):
                num[:,channel]+=weight*map_coordinates(lab[...,channel]*fit,coordinates,order=1,mode=sample_mode,cval=0)
    strength=den/total
    missing=strength<.008
    # A few isolated edge pixels without support must not void a whole face.
    # They stay out of `reliable`, so smooth_material_correction fills their
    # correction from neighbouring supported pixels; RGB is never copied.
    missing_mask=np.zeros(alpha.shape,bool);missing_mask[y[missing],x[missing]]=True
    supported_mask=np.zeros(alpha.shape,bool);supported_mask[y[~missing],x[~missing]]=True
    if not tolerated_unsupported(missing_mask,alpha,supported_mask):
        raise ValueError(f'Unsupported editable contour illumination: {int(missing.sum())} pixels; bounds [{int(x[missing].min())}, {int(y[missing].min())}, {int(x[missing].max())}, {int(y[missing].max())}]')
    if not np.isfinite(num).all():raise ValueError('Non-finite contour illumination')
    low=np.zeros_like(lab);support=np.zeros(alpha.shape);ok=~missing
    low[y[ok],x[ok]]=num[ok]/den[ok,None];support[y,x]=strength
    return low,support,support>=.008


def contour_depth_reference(lab, fit, alpha):
    """Source-only normal profile for a rim with different polished facets.

    Bright bevels and dark anodized bands cannot share one scalar target.
    Estimate their continuous normal-depth response from this face, leaving
    tangent illumination correction to the independently supported field.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter1d
    if np.count_nonzero(fit)<60:raise ValueError('Too few independently sampled rim depths')
    depth=distance_transform_edt(np.pad(alpha>=128,1))[1:-1,1:-1]
    bins=np.rint(depth).astype(int);max_bin=int(np.max(bins[fit]))
    positions=[];values=[]
    for i in range(max_bin+1):
        samples=lab[...,0][fit&(bins==i)]
        if len(samples)>=20:positions.append(i);values.append(float(np.percentile(samples,60)))
    if len(positions)<3:raise ValueError('Too few independently sampled rim depths')
    profile=np.interp(np.arange(max_bin+1),positions,values)
    profile=gaussian_filter1d(profile,.75)
    return np.interp(depth,np.arange(len(profile)),profile),dict(depths=positions,source_L60=values)


def blend_correction_boundary(shift, masks, config, source_lab=None):
    """Join additive fields at an internal seam or source material transition.

    Smooth only additive corrections, never RGB/texture. Outer alpha edges,
    paper, glass and other unlisted materials do not become transition targets.
    """
    from scipy.ndimage import binary_dilation, distance_transform_edt, gaussian_filter, minimum_filter
    names=config['domains'];anchor=config['boundary_domain'];sigma=float(config['sigma'])
    kind=config.get('kind','same_material')
    if kind not in ('same_material','source_material_transition'):
        raise ValueError('Unknown correction boundary kind')
    if kind=='source_material_transition' and (len(names)!=2 or sigma>2):
        raise ValueError('Source material transition requires two domains and a narrow sensor-edge blend')
    if len(set(names))!=len(names) or anchor not in names or len(names)<2:
        raise ValueError('Correction blend requires distinct adjacent material domains')
    if not np.isfinite(sigma) or sigma<=0:
        raise ValueError('Correction blend sigma must be finite and positive')
    if any(name not in masks for name in names):
        raise ValueError('Unknown correction blend domain')
    union=np.logical_or.reduce([masks[name] for name in names])
    primary=masks[anchor];other=union&~primary
    seam=(primary&binary_dilation(other))|(other&binary_dilation(primary))
    if not seam.any():raise ValueError('Correction blend has no shared internal boundary')
    distance=distance_transform_edt(~seam)
    t=np.clip(1-distance/(3*sigma),0,1);weight=t*t*(3-2*t)*union
    support=gaussian_filter(union.astype(float),sigma)
    smooth=np.stack([gaussian_filter(shift[...,c]*union,sigma)/np.maximum(support,1e-9)
                     for c in range(3)],axis=-1)
    result=shift+(smooth-shift)*weight[...,None]
    gamut_adjusted=0
    if source_lab is not None and kind=='same_material':
        # Per-pixel gamut clipping can turn a smooth field into speckled steps
        # near saturated highlights. Propagate conservative L limits through
        # this same-metal field, not through source pixels or their texture.
        visible=weight>0
        safe,_=preserve_chroma_in_gamut(source_lab[union]+result[union],source_lab[union])
        safe_shift=result.copy();safe_shift[union,0]=safe[:,0]-source_lab[union,0]
        bounded=safe_shift[...,0].copy()
        for sign in (1.,-1.):
            magnitude=np.maximum(0,sign*safe_shift[...,0])
            cap=np.where(union,magnitude,np.inf)
            # A min-plus envelope only reduces correction magnitude and is
            # at most 0.6 L/pixel in either axis; it never invents RGB detail.
            for _ in range(int(np.ceil(float(magnitude[union].max())/.6))+1):
                updated=np.where(union,np.minimum(cap,minimum_filter(cap,size=3,mode='constant',cval=np.inf)+.6),np.inf)
                if np.array_equal(updated,cap):break
                cap=updated
            same_sign=visible&(sign*safe_shift[...,0]>0)
            bounded[same_sign]=sign*cap[same_sign]
        continuous=safe_shift[...,0]+(bounded-safe_shift[...,0])*weight
        gamut_adjusted=int(np.count_nonzero(visible&(np.abs(continuous-result[...,0])>1e-6)))
        result[visible,0]=continuous[visible]
    return result,dict(domains=names,boundary_domain=anchor,sigma=sigma,
                       transition_pixels=int(np.count_nonzero(weight)),
                       gamut_continuity_pixels=gamut_adjusted,
                       method=('same_material_additive_field_only' if kind=='same_material'
                               else 'source_material_transition_additive_field_only'))


def process_view(item, target, defaults):
    path = Path(item['source'])
    before_hash = digest(path)
    src = np.array(Image.open(path).convert('RGBA'))
    h,w = src.shape[:2]
    protect = shapes_mask((w,h), item.get('protect', []))
    if item.get('protected_mask'):
        external = np.array(Image.open(item['protected_mask']).convert('L'))
        if external.shape != (h,w):
            raise ValueError('Protection mask dimensions differ from source')
        protect |= external > 0
    if item.get('protection_policy', 'reviewed_regions') not in ('sticker_only','reviewed_regions','none'):
        raise ValueError('Unknown protection policy')
    if item.get('protection_policy') == 'sticker_only':
        if item.get('source_sha256') != before_hash or not item.get('sticker_mask'):
            raise ValueError('Sticker-only policy requires source SHA256 and sticker mask')
        paper = np.array(Image.open(item['sticker_mask']).convert('L')) > 0
        if paper.shape != (h,w):
            raise ValueError('Sticker mask dimensions differ')
        visible = src[...,3] > 0
        if np.any((protect ^ paper) & visible):
            raise ValueError('Sticker-only policy rejects protection outside actual paper or unprotected paper')
    if item.get('protection_policy')=='none' and protect.any():raise ValueError('No-zone policy rejects hard protected pixels')
    reviewed = np.ones((h,w), dtype=bool)
    if item.get('editable_mask'):
        external = np.array(Image.open(item['editable_mask']).convert('L'))
        if external.shape != (h,w):
            raise ValueError('Editable mask dimensions differ from source')
        reviewed = external > 0
    required = np.zeros((h,w), dtype=bool)
    if item.get('required_editable_mask'):
        external = np.array(Image.open(item['required_editable_mask']).convert('L'))
        if external.shape != (h,w):
            raise ValueError('Required editable mask dimensions differ from source')
        required = (external > 0) & (src[...,3] > 0)
        if np.any(required & protect):
            raise ValueError('Required editable pixels remain protected; revise obsolete protection')
    exclude = shapes_mask((w,h), item.get('exclude_from_fit', [])) | protect
    lab = rgb_to_lab(src[...,:3])
    mode = color_mode(defaults)
    out_lab = lab.copy()
    allowed_union = np.zeros((h,w), dtype=bool)
    domains = []; domain_masks = {}
    for domain in item['domains']:
        material = (shapes_mask((w,h), domain['shapes']) if domain.get('shapes')
                    else np.ones((h,w), dtype=bool))
        if domain.get('mask'):
            region = np.array(Image.open(domain['mask']).convert('L'))
            if region.shape != (h,w):
                raise ValueError('Material domain mask dimensions differ from source')
            material &= region > 0
        elif not domain.get('shapes'):
            raise ValueError('Material domain requires reviewed shapes or a mask')
        material &= (src[...,3] > 0) & reviewed
        allowed = material & ~protect
        domain_masks[domain.get('name','metal')] = allowed
        if np.any(allowed_union & allowed):
            raise ValueError('Material domains must not overlap')
        allowed_union |= allowed
        fit = material & ~exclude & (src[...,3] >= 240)
        if domain.get('fit_mask'):
            fit_region = np.array(Image.open(domain['fit_mask']).convert('L'))
            if fit_region.shape != (h,w):
                raise ValueError('Fit mask dimensions differ from source')
            fit &= fit_region > 0
        if fit.sum() < 20:
            raise ValueError('Insufficient reviewed material samples')
        sigma = domain.get('sigma', defaults.get('sigma', [18,18]))
        field = domain.get('field','gaussian')
        contour_alpha=src[...,3]
        if domain.get('contour_alpha_mask'):
            if field!='contour':raise ValueError('Body contour coordinates require contour field')
            contour_alpha=np.array(Image.open(domain['contour_alpha_mask']).convert('L'))
            if contour_alpha.shape!=(h,w):raise ValueError('Body contour dimensions differ from source')
            # Fit exclusion is not pixel protection: all original button
            # pixels remain allowed and receive the continuous body lighting.
            fit &= contour_alpha>=128
            if fit.sum()<20:raise ValueError('Insufficient body-contour material samples')
        profile_info=None;field_lab=lab
        if domain.get('reference_profile'):
            if domain['reference_profile'] not in ('contour_depth','contour_depth_lift') or mode!='source_lightness' or field!='contour':
                raise ValueError('Rim normal profile requires source_lightness contour field')
            profile,profile_info=contour_depth_reference(lab,fit,contour_alpha)
            # Fit tangent illumination relative to the measured normal profile.
            # Averaging absolute L across bright/dark facets would attribute
            # their real finish differences to lighting and create dark bands.
            field_lab=lab.copy();field_lab[...,0]-=profile
        if field == 'contour':
            low, support, reliable = supported_contour_field(field_lab, fit, allowed, contour_alpha, sigma,domain.get('contour_inset_depth',5.),domain.get('contour_source_inset_limit'))
        elif field == 'gaussian':
            low, support, reliable = supported_field(lab, fit, allowed, sigma, src[...,3])
        elif field == 'feature_excluded_gaussian':
            if mode!='source_lightness':raise ValueError('Feature field requires source lightness')
            feature=np.array(Image.open(domain['feature_mask']).convert('L'))>0
            low,support,reliable=feature_excluded_field(lab,fit,allowed,sigma,feature,src[...,3])
        else:
            raise ValueError('Unknown illumination field')
        strengths = np.array(domain.get('strength', defaults.get('strength', [.78,.72,.72])))
        if mode == 'source_lightness':
            strengths = strengths.copy()
            strengths[1:] = 0
        # Each material/component uses its own source samples: antennas and
        # button metal cannot inherit the surrounding frame's colour target.
        local_target = target
        if domain.get('reference_boxes'):
            local_target, _ = target_color({'views': [dict(source=str(path), reference_boxes=domain['reference_boxes'])]})
        source_reference=np.array(local_target,copy=True)
        if profile_info is not None:
            local_target=np.array(local_target,copy=True);local_target[0]=0.
        shift = (local_target-low)*strengths
        if domain.get('reference_profile') == 'contour_depth_lift':
            # The old depth profile retained a continuous photographic dark
            # band. Lift its low-frequency L toward this source's measured
            # metal reference, without replacing pixels or flattening texture.
            lift=np.maximum(0.,source_reference[0]-profile)
            shift[...,0]+=lift*strengths[0]
            profile_info=dict(profile_info,lift_reference_L=float(source_reference[0]),
                              maximum_source_profile_lift_L=float(lift[allowed].max()))
        limits = np.asarray(domain.get('max_lab_shift', defaults.get('max_lab_shift', [48,24,24])),dtype=float)
        if limits.shape!=(3,) or not np.isfinite(limits).all() or (limits<0).any():
            raise ValueError('max_lab_shift must have three finite nonnegative values')
        shift = np.clip(shift, -limits, limits)
        # Smooth all correction components, including across excluded features.
        # Protected pixels themselves are reinstated exactly after conversion.
        smooth = domain.get('correction_smooth', 3)
        smoothing_mode=domain.get('correction_smoothing','legacy')
        if smoothing_mode=='source_fit_coverage' and (mode!='source_lightness' or field!='contour'
                or domain.get('reference_profile')!='contour_depth_lift'):
            raise ValueError('Source fit coverage smoothing requires source-lightness contour-depth-lift')
        shift,smoothing_info=smooth_material_correction(shift,reliable,fit,allowed,smooth,smoothing_mode)
        weight = inward_weight(allowed, domain.get('feather', defaults.get('feather', 5)))
        out_lab += shift*weight[...,None]
        domains.append({'name':domain.get('name','metal'), 'editable_pixels':int(allowed.sum()),
                        'fit_pixels':int(fit.sum()),'field':field,'strength':strengths.tolist(),
                        'source_reference_lab':source_reference.tolist(),
                        'source_normal_profile':profile_info,
                        'correction_smoothing':smoothing_info,
                        'contour_coordinate_sha256':digest(domain['contour_alpha_mask']) if domain.get('contour_alpha_mask') else None,
                        'minimum_editable_support':float(support[allowed].min()) if allowed.any() else None,
                        'unsupported_editable_pixels':int(np.count_nonzero(allowed&~reliable))})
    blends=[]
    for config in item.get('correction_blends',[]):
        if mode!='source_lightness':raise ValueError('Correction blends require source_lightness')
        shift,proof=blend_correction_boundary(out_lab-lab,domain_masks,config,lab)
        out_lab=lab+shift;blends.append(proof)
    if np.any(required & ~allowed_union):
        raise ValueError('Required editable pixels missing from material domains or editable mask')
    gamut_limited = 0; gamut_continuity = []
    if mode == 'source_lightness':
        threshold = np.asarray(defaults.get('lightness_deadband', [0, 0]), dtype=float)
        if (threshold.shape != (2,) or not np.isfinite(threshold).all()
                or threshold[0] < 0 or threshold[1] < threshold[0]
                or (threshold[1] > 0 and threshold[1] == threshold[0])):
            raise ValueError('lightness_deadband needs 0 <= start < end, or [0, 0]')
        if threshold[1] > 0:
            delta = out_lab[..., 0]-lab[..., 0]
            t = np.clip((np.abs(delta)-threshold[0])/(threshold[1]-threshold[0]), 0, 1)
            out_lab[..., 0] = lab[..., 0]+delta*t*t*(3-2*t)
        desired = out_lab
        out_lab, gamut_limited = preserve_chroma_in_gamut(desired, lab)
        for domain in item['domains']:
            if 'gamut_continuity_step_L' in domain:
                name = domain.get('name', 'metal')
                seed_mask = None
                if domain.get('gamut_continuity_opaque_seeds', False):
                    seed_mask = src[..., 3] >= 240
                edge_weight = None
                if domain.get('gamut_continuity_source_edge_guard'):
                    from scipy.ndimage import distance_transform_edt
                    guard = float(domain['gamut_continuity_source_edge_guard'])
                    if not np.isfinite(guard) or guard <= 0:
                        raise ValueError('Source-edge material guard must be finite and positive')
                    # Alpha boundary is not a different material. Retain the
                    # old cap continuity near actual paper/plastic/hardware
                    # boundaries; gradually admit source feature contrast only
                    # inside this material, with no hard correction seam.
                    other = (src[..., 3] > 0) & ~domain_masks[name]
                    edge_weight = (np.clip((distance_transform_edt(~other)-guard)/guard, 0, 1)
                                   if other.any() else np.ones((h,w)))
                    # Reserve this relief for the large positive lift needed
                    # by a deep source shadow. Ordinary correction retains its
                    # existing continuity constraints, with a smooth ramp over
                    # the upper half of this domain's configured lift budget.
                    budget = float(domain.get('max_lab_shift', defaults.get('max_lab_shift', [48,24,24]))[0])
                    if budget <= 0:
                        edge_weight *= 0
                    else:
                        edge_weight *= np.clip(2*(desired[...,0]-lab[...,0])/budget-1, 0, 1)
                out_lab, proof = continuous_gamut_limits(desired, out_lab, lab,
                    domain_masks[name], domain['gamut_continuity_step_L'], seed_mask,
                    domain.get('gamut_continuity_follow_field', False),
                    domain.get('gamut_continuity_follow_source_edges', False), edge_weight)
                proof['opaque_seeds_only'] = seed_mask is not None
                gamut_continuity.append(dict(name=name, **proof))
    elif any('gamut_continuity_step_L' in domain for domain in item['domains']):
        raise ValueError('Gamut continuity requires source_lightness')
    rear_shadow=None
    if item.get('rear_shadow_correction'):
        from phone_texture.rear_shadow import correction
        plan=item['rear_shadow_correction']
        if item.get('name')!='back' or mode!='source_lightness':
            raise ValueError('Rear shadow correction requires a source-lightness back')
        if plan['source_sha256']!=before_hash or digest(plan['material_mask'])!=plan['material_sha256']:
            raise ValueError('Rear shadow source/material binding changed')
        material=np.array(Image.open(plan['material_mask']).convert('L'))>0
        if material.shape!=(h,w) or np.any(material&~allowed_union):
            raise ValueError('Rear shadow material is outside the editable source')
        extra,scope,rear_shadow=correction(out_lab[...,0],lab[...,0],material,plan)
        desired=out_lab.copy();desired[...,0]+=extra
        safe,limited=preserve_chroma_in_gamut(desired,lab)
        # Preserve every other material/field exactly, including the logo.
        out_lab[scope]=safe[scope]
        rear_shadow.update(gamut_limited_pixels=limited,scope_outside_material_pixels=int(np.count_nonzero(scope&~material)))
    dst = src.copy()
    converted = lab_to_rgb(out_lab)
    quantization=[]
    for config in item.get('quantization_continuity',[]):
        if mode!='source_lightness':raise ValueError('Illumination quantization requires source lightness')
        region=np.array(Image.open(config['mask']).convert('L'))>0
        if region.shape!=allowed_union.shape or np.any(region&~allowed_union):raise ValueError('Quantization region is not editable source material')
        candidate,proof=quantize_continuous_illumination(out_lab,src[...,:3],region,config['max_added_step_L'])
        converted[region]=candidate[region];quantization.append(proof)
    dst[allowed_union,:3] = converted[allowed_union]
    assert np.array_equal(dst[...,3],src[...,3])
    assert np.array_equal(dst[protect],src[protect])
    assert np.array_equal(dst[~allowed_union],src[~allowed_union])
    if digest(path) != before_hash:
        raise RuntimeError('Source changed during processing')
    return dst, allowed_union, protect, {'source':str(path),'source_sha256':before_hash,
        'size':[w,h], 'domains':domains, 'color_mode':mode,
        'gamut_limited_pixels':gamut_limited,'correction_blends':blends,'rear_shadow_correction':rear_shadow,
        'gamut_continuity':gamut_continuity,'quantization_continuity':quantization,
        'required_editable_pixels':int(required.sum()),
        'required_editable_missing':0,
        'changed_pixels':int(np.any(dst != src,axis=-1).sum()),
        'alpha_changed':0,'protected_changed':0,'outside_changed':0}


def run(job_path):
    job = json.loads(Path(job_path).read_text())
    output = Path(job['output_dir'])
    records = Path(job['records_dir'])
    paths = [Path(v['source']).resolve() for v in job['views']]
    outputs = [output/(v['name']+'.png') for v in job['views']]
    if len(set(outputs)) != len(outputs):
        raise ValueError('Duplicate face names')
    if any(p.exists() for p in outputs):
        raise FileExistsError('Output files already exist; choose a new version directory')
    if any(p.resolve() in paths for p in outputs):
        raise ValueError('Source overwrite is prohibited')
    if records.resolve() == output.resolve() or output.resolve() in records.resolve().parents:
        raise ValueError('Records must live outside the photo delivery directory')
    output.mkdir(parents=True, exist_ok=True)
    records.mkdir(parents=True, exist_ok=True)
    mode = color_mode(job)
    defaults = dict(job.get('defaults', {}), color_mode=mode)
    if mode == 'source_lightness':
        if 'target_lab' in job or 'ai_reference' in job:
            raise ValueError('Shared color targets require explicit color_mode shared_color')
        targets = [target_color({'views': [item]}) for item in job['views']]
        info = {'mode': mode, 'per_view_source_references': {
            item['name']: entry[1] for item, entry in zip(job['views'], targets)}}
    else:
        target, info = target_color(job)
        targets = [(target, info)]*len(job['views'])
        info['mode'] = mode
    report = {'target':info, 'uv_seam_validation':'not_performed_without_uv_correspondence', 'views':{}}
    for item, dest, (target, _) in zip(job['views'], outputs, targets):
        dst, allowed, protect, details = process_view(item,target,defaults)
        src_im = Image.open(item['source'])
        save_args = {'icc_profile':src_im.info['icc_profile']} if 'icc_profile' in src_im.info else {}
        Image.fromarray(dst).save(dest, **save_args)
        saved = np.array(Image.open(dest).convert('RGBA'))
        src = np.array(Image.open(item['source']).convert('RGBA'))
        if saved.shape != src.shape or not np.array_equal(saved[...,3],src[...,3]):
            raise AssertionError('Saved output geometry/alpha differs')
        if not np.array_equal(saved[~allowed],src[~allowed]) or not np.array_equal(saved[protect],src[protect]):
            raise AssertionError('Saved output protection differs')
        if digest(item['source']) != details['source_sha256']:
            raise AssertionError('Source hash changed')
        details.update(output=str(dest),output_sha256=digest(dest))
        report['views'][item['name']] = details
        Image.fromarray(allowed.astype(np.uint8)*255).save(records/(item['name']+'-editable.png'))
        Image.fromarray(protect.astype(np.uint8)*255).save(records/(item['name']+'-protected.png'))
    (records/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',required=True,help='Reviewed JSON job; see material-harmonization reference')
    run(parser.parse_args().job)
