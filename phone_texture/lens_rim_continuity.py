"""Keep two source-resolved lines on the lens rim from trading contrast during lighting."""
import numpy as np
from scipy import ndimage as ndi
from scipy.signal import find_peaks


def _polar(image, coordinates):
    return ndi.map_coordinates(image, coordinates, order=1, mode='nearest')


def _paired_edges(profile, radii, scale, measured_outer):
    """Find two narrow troughs separated by a real, brighter metal ridge."""
    tracks = np.full((360, 2), np.nan)
    contrast = np.zeros((360, 2))
    for angle, line in enumerate(profile):
        peaks, _ = find_peaks(-line, prominence=4., width=(1., 9.), distance=4)
        for left, right in zip(peaks, peaks[1:]):
            if min(abs(radii[[left, right]]-measured_outer[angle])) > 4.5 * scale:
                continue
            if not 2.2 * scale <= radii[right] - radii[left] <= 7.5 * scale:
                continue
            ridge = line[left:right+1].max()
            # Check both shoulders locally: global peak prominence can
            # mistake a long plateau shadow for the second physical line.
            inner = line[max(0,left-7):left].max()
            outer = line[right+1:right+8].max()
            depth = np.minimum(ridge, [inner, outer]) - line[[left,right]]
            if min(depth) >= 4. and min(depth) > min(contrast[angle]):
                tracks[angle] = radii[[left, right]]
                contrast[angle] = depth
    return tracks, contrast


def _corrected_shadow_arc(raw, candidate, coords, flat, tracks, depth, radii, scale, outer):
    """Distinguish a lifted shadow on the plateau from a second rim line."""
    angles = np.arange(len(tracks))
    radius = tracks[:, 1]
    exterior = _polar(flat.astype(np.float32), coords) > .97
    eligible = (radius > outer + 3 * scale) & (depth[:, 1] < .68 * depth[:, 0])
    supported = np.zeros(len(tracks), bool)
    for ang in angles[eligible]:
        peak = np.argmin(abs(radii - radius[ang]))
        reference = (radii > radius[ang] + 4 * scale) & (radii <= radius[ang] + 10 * scale)
        if reference.sum() < 5 or not exterior[ang, peak:np.flatnonzero(reference)[-1]+1].all():
            continue
        before = np.median(raw[ang, reference]) - raw[ang, peak]
        after = np.median(candidate[ang, reference]) - candidate[ang, peak]
        # The same connected material now agrees with the illuminated plateau.
        supported[ang] = before >= 6 and abs(after) <= 6 and before - after >= 4
    density = ndi.uniform_filter1d(supported.astype(float), 9, mode='wrap')
    # A short missing measurement within a corroborated shadow must not
    # acquire a new dark line from angular interpolation at its neighbours.
    return ndi.maximum_filter1d(density >= .44, 9, mode='wrap')


def review(source_lab, lit_lab, metal, lens, flat=None):
    cx, cy, r = lens['centre']
    scale = r / 98.
    radii = r + np.arange(3., 32., .5) * scale
    theta = np.deg2rad(np.arange(360))[:, None]
    coordinates = [cy + np.sin(theta)*radii, cx + np.cos(theta)*radii]
    raw = _polar(source_lab[...,0], coordinates)
    candidate = _polar(lit_lab[...,0], coordinates)
    smooth_raw = ndi.gaussian_filter(raw, (.8, .6), mode=('wrap', 'nearest'))
    tracks, depth = _paired_edges(smooth_raw, radii, scale, lens['outer'])
    visible = np.isfinite(tracks[:,0])
    agreement = np.zeros(360, int)
    for offset in (-3,-2,-1,1,2,3):
        neighbours = np.roll(tracks, offset, axis=0)
        agreement += np.all(abs(tracks-neighbours) <= max(1.3,1.65*scale), axis=1)
    valid = visible & (agreement >= 4)
    shadows = np.zeros(360, bool)
    if flat is not None:
        shadows = _corrected_shadow_arc(raw, candidate, coordinates, flat, tracks, depth,
                                        radii, scale, np.asarray(lens['outer']))
    valid &= ~shadows
    proof = dict(dual_edge_angles=int(valid.sum()), repaired_pixels=0,
                 excluded_shadow_angles=int(shadows.sum()))
    if not valid.any():
        return dict(proof, status='no_supported_pair')
    # Compare depths on the original two tracks, without redetecting a
    # different "strongest" edge on the candidate.
    raw_index = np.zeros_like(tracks, dtype=int)
    raw_index[valid] = np.rint((tracks[valid]-radii[0])/(.5*scale)).astype(int)
    def retained(profile):
        smooth = ndi.gaussian_filter(profile, (.8,.6), mode=('wrap','nearest'))
        ratios = np.ones_like(tracks)
        for ang in np.flatnonzero(valid):
            lo, hi = raw_index[ang]
            line = smooth[ang]
            ridge = line[lo:hi+1].max()
            shoulders = [line[max(0,lo-7):lo].max(), line[hi+1:hi+8].max()]
            ratios[ang] = (np.minimum(ridge, shoulders)-line[[lo,hi]])/depth[ang]
        return ratios
    initial = retained(candidate)
    weak = valid & (np.min(initial,axis=1)<.73)
    proof['weak_angles_before'] = int(weak.sum())
    if not weak.any():
        return dict(proof,status='clear',weak_angles_after=0)

    # Share the local lighting gain across both lines. Use gains already
    # present on the metal shoulders, never interpolate or draw source RGB.
    gains = candidate - raw
    field = np.zeros(360)
    material = _polar(metal.astype('f'), coordinates) > .97
    for ang in np.flatnonzero(valid):
        lo, hi = tracks[ang]
        shoulders = ((radii > lo-5*scale) & (radii < lo-2*scale)
                     | (radii > hi+2*scale) & (radii < hi+5*scale)) & material[ang]
        if shoulders.sum() < 4:
            valid[ang] = False
            continue
        field[ang] = np.median(gains[ang, shoulders])
    excluded_weak = int(np.count_nonzero(weak & ~valid))
    proof['metal_reference_angles'] = int(valid.sum())
    if not np.any(weak & valid):
        return dict(proof,status='insufficient_metal_reference',weak_angles_after=int(weak.sum()))
    support = ndi.gaussian_filter1d(valid.astype(float), 2, mode='wrap')
    field = ndi.gaussian_filter1d(field*valid, 2, mode='wrap')/np.maximum(support,1e-8)
    # Extend only through short, source-supported gaps in the angular measurement.
    angles = np.arange(360)
    def interpolate(values):
        return np.interp(angles,angles[valid],values[valid],period=360)
    low, high = [interpolate(tracks[:,i]) for i in range(2)]
    strength = ndi.maximum_filter1d((weak & valid).astype(float), 13, mode='wrap')
    strength = ndi.gaussian_filter1d(strength, 2, mode='wrap') * np.clip((support-.2)/.5,0,1)
    if shadows.any():
        distance = ndi.distance_transform_edt(np.tile(~shadows, 3))[360:720]
        strength *= np.clip(distance / 4., 0, 1)
    pad = int(np.ceil(radii[-1]))+3
    ys = slice(max(0,int(cy)-pad),min(len(metal),int(cy)+pad+1))
    xs = slice(max(0,int(cx)-pad),min(metal.shape[1],int(cx)+pad+1))
    y,x = np.ogrid[ys,xs]
    phase = np.rad2deg(np.arctan2(y-cy,x-cx))%360
    radius = np.hypot(y-cy,x-cx)
    def angular(values):
        return np.interp(phase,angles,values,period=360)
    outside = np.maximum(angular(low)-radius,radius-angular(high))
    t = np.clip((5.5-outside/scale)/3.5,0,1)
    alpha = t*t*(3-2*t)*angular(strength)*metal[ys,xs]
    roi = lit_lab[ys,xs,0]
    revised = roi + alpha*(source_lab[ys,xs,0]+angular(field)-roi)
    coords = [coordinates[0]-ys.start,coordinates[1]-xs.start]
    trial = _polar(revised, coords)
    after = retained(trial)
    # Accept only an overall recovery of the weaker line without a new
    # confirmed discontinuity on a previously well-resolved pair.
    old_score = np.min(initial,axis=1)
    score = np.min(after,axis=1)
    if (np.sum(np.maximum(.73-score[valid],0)) >= np.sum(np.maximum(.73-old_score[valid],0))
            or np.any(valid & (old_score>=.73) & (score<.55))):
        return dict(proof,status='retained_candidate',weak_angles_after=int(weak.sum()))
    proof['repaired_pixels'] = int(np.count_nonzero(abs(revised-roi)>.01))
    roi[:] = revised
    remaining = excluded_weak + int(np.count_nonzero((score<.73) & valid))
    return dict(proof,status='residual' if remaining else 'corrected',weak_angles_after=remaining)
