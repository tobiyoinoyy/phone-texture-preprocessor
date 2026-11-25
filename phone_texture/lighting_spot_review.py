"""Source-relative checks along a metal rail or around a measured camera rim."""
import numpy as np
from scipy import ndimage as ndi


def _local_contrast(values, usable, spacing, mode):
    """At the same depth in the material, compare the centre with two flanks."""
    sigma = max(1., spacing / 7)
    density = ndi.gaussian_filter1d(usable.astype(float), sigma, axis=1, mode=mode)
    centre = ndi.gaussian_filter1d(values * usable[..., None], sigma, axis=1, mode=mode)
    centre /= np.maximum(density[..., None], 1e-8)
    left = np.roll(centre, spacing, axis=1)
    right = np.roll(centre, -spacing, axis=1)
    # Require a continuous path of metal to both flanks.
    supported = ndi.minimum_filter1d(density, 2 * spacing + 1, axis=1, mode=mode) > .85
    if mode != 'wrap':
        supported[:, :spacing] = supported[:, -spacing:] = False
    return centre - (left + right) * .5, supported


def _excess(source_contrast, output_contrast):
    # A photographed shadow can be levelled up to the local metal. Penalize
    # a reversal as well as new contrast on an originally uniform surface.
    return output_contrast - np.clip(output_contrast, np.minimum(source_contrast, 0),
                                    np.maximum(source_contrast, 0))


def _review_grid(source, candidate, valid, spacing, mode, propose, read_back, commit):
    reference, supported = _local_contrast(source, valid, spacing, mode)
    tolerance = np.array([2.2, 1.8, 1.8])[:source.shape[-1]]
    def inspect(values):
        contrast, _ = _local_contrast(values, valid, spacing, mode)
        excess = _excess(reference, contrast)
        ratio = np.max(abs(excess) / tolerance, axis=2)
        return excess, supported & (ratio > 1), ratio
    excess, hits, ratio = inspect(candidate)
    evidence = dict(checked_samples=int(supported.sum()), hits_before=int(hits.sum()))
    if hits.any():
        peak = np.unravel_index(np.argmax(np.where(supported, ratio, 0.)), hits.shape)
        evidence['peak_sample'] = [int(v) for v in peak]
        evidence['peak_excess_before'] = excess[peak].round(2).tolist()
        # Work on a copy. Each candidate is resampled in the original
        # coordinates; retain an iteration only when its excess decreases.
        def loss(r):
            return float(np.maximum(r[supported] - 1, 0).sum())
        for _ in range(4):
            strength = np.clip((ratio - .65) / .35, 0, 1) * supported
            reach = ndi.maximum_filter1d(hits, max(3, spacing // 3) | 1, axis=1, mode=mode)
            delta = -excess * (strength * reach)[..., None]
            trial = propose(delta)
            remaining, hit_again, new_ratio = inspect(read_back(trial))
            if loss(new_ratio) >= loss(ratio) - 1e-6:
                break
            commit(trial)
            excess, hits, ratio = remaining, hit_again, new_ratio
            if not hits.any():
                break
        evidence['peak_excess_after'] = excess[peak].round(2).tolist()
    evidence.update(hits_after=int(hits.sum()), status=('insufficient_reference' if not supported.any()
                    else 'residual' if hits.any() else 'corrected' if evidence['hits_before'] else 'clear'))
    return evidence


def side_rail(source, current, safe, alpha, metal):
    """Audit both long edges. Only undo a locally verified gamut-cap propagation."""
    h, w = alpha.shape
    proof = []
    if min(h,w) < 10 or max(h,w) < 3*min(h,w):
        return [dict(status='insufficient_geometry')]
    # Views of the actual arrays, long coordinate last. Contour-relative
    # sampling avoids constant-width assumptions at contact points.
    def horizontal(a):
        return np.swapaxes(a, 0, 1) if h > w else a
    for side in (0, 1):
        raw, out, limit, opacity, mask = [horizontal(a)[::1 if side == 0 else -1]
                                          for a in (source, current, safe, alpha, metal)]
        height, length = mask.shape
        depth = min(height//3, max(5, round(.24*height)))
        x = np.arange(length)[None,:]
        y = np.minimum(np.argmax(opacity >= 240, axis=0) + np.arange(depth)[:,None], height-1)
        valid = (opacity[y,x] >= 128) & mask[y,x]
        original = out[y,x,0].copy()
        ceiling = limit[y,x,0]
        # Compare the rendered contrast to the source on both edges, even
        # where the gamut cap had no effect. Only release demonstrable lost lift.
        def propose(delta):
            return np.clip(out[y,x,0] + delta[...,0], original, np.maximum(original, ceiling))
        def read(trial):
            return trial[...,None]
        def commit(trial):
            changed = valid & (trial != out[y,x,0])
            out[y[changed], np.broadcast_to(x,y.shape)[changed],0] = trial[changed]

        check = _review_grid(raw[y,x,:1], original[...,None], valid, max(4, round(height*.40)),
                             'nearest', propose, read, commit)
        proof.append(check)
    return proof


def camera_arc(raw, current, mask, lens, zone):
    """Audit concentric paths; glass, joint and plateau bevel are outside the domain."""
    cx, cy, radius = lens['centre']
    inner, outer = np.asarray(lens['inner']), np.asarray(lens['outer'])
    # A specular patch can hide the gradient for a short arc. Bound both
    # edges from the nearest confirmed angle on each side; keep only their
    # common metal interior (or their common exterior).
    angles = np.arange(360)
    known = angles[np.asarray(lens['supported'],dtype=bool)]
    if len(known) < 210:
        return dict(status='insufficient_rim')
    extended = np.r_[known-360,known,known+360]
    lower = extended[np.searchsorted(extended, angles,side='right')-1]
    upper = extended[np.searchsorted(extended, angles)]
    left, right = lower % 360, upper % 360
    reliable = ((upper-lower <= 24) & (abs(inner[left]-inner[right]) <= 3)
                & (abs(outer[left]-outer[right]) <= max(5., radius*.055)))
    if zone == 'ring':
        inner = np.maximum(inner,np.maximum(inner[left],inner[right]))
        outer = np.minimum(outer,np.minimum(outer[left],outer[right]))
    else:
        outer = np.maximum(outer,np.maximum(outer[left],outer[right]))
    if zone == 'ring':
        levels = np.linspace(0., 1., 11)
        start, end = inner+1.5, outer-1.
    else:
        levels = np.linspace(0., 1., 17)
        start, end = outer+2., outer+max(7., radius*.23)
    distances = start[None,:] + levels[:,None]*(end-start)
    theta = np.deg2rad(np.arange(360))[None,:]
    coords = [cy + distances*np.sin(theta), cx + distances*np.cos(theta)]
    def sample(a):
        return np.stack([ndi.map_coordinates(v, coords, order=1, mode='nearest')
                         for v in a.transpose(2,0,1)], axis=-1)
    valid = ndi.map_coordinates(mask.astype('f'), coords, order=1, mode='constant') > .98
    valid &= (end-start > 1)[None,:]
    valid &= reliable[None,:]
    pad = int(np.ceil(end.max())) + 3
    y0, y1 = max(0,int(cy)-pad), min(len(mask),int(cy)+pad+1)
    x0, x1 = max(0,int(cx)-pad), min(mask.shape[1],int(cx)+pad+1)
    yy, xx = np.ogrid[y0:y1,x0:x1]
    angle = np.rad2deg(np.arctan2(yy-cy, xx-cx)) % 360
    lo = np.interp(angle,np.arange(360),start,period=360)
    hi = np.interp(angle,np.arange(360),end,period=360)
    distance = np.hypot(yy-cy,xx-cx)
    pos = (distance-lo)/np.maximum(hi-lo,1)
    active = mask[y0:y1,x0:x1] & (pos>0) & (pos<1)
    if not active.any():
        return dict(status='insufficient_reference')
    polar = [pos[active]*(len(levels)-1),angle[active]]
    fade = np.minimum(1.,np.minimum(distance[active]-lo[active],hi[active]-distance[active]))
    tile = current[y0:y1,x0:x1]
    local_coords = [coords[0]-y0,coords[1]-x0]
    def propose(delta):
        trial = tile.copy()
        wrapped = np.concatenate([delta,delta[:,:1]],axis=1)
        change = np.column_stack([ndi.map_coordinates(v,polar,order=1,mode='nearest')
                                  for v in wrapped.transpose(2,0,1)])
        trial[active] += change * fade[:,None]
        return trial
    def read(trial):
        return np.stack([ndi.map_coordinates(v,local_coords,order=1,mode='nearest')
                         for v in trial.transpose(2,0,1)],axis=-1)
    def commit(trial):
        tile[:] = trial

    return _review_grid(sample(raw),sample(current),valid,15,'wrap',propose,read,commit)
