"""Register the camera lenses of a standard-size back to fixed model positions.

2026-11-10 user decision: in the 3D viewer the lens rings of some backs
(notably dark blue Pro units) sat 2-4 px down-left of the model's lens rings,
while the flash was correct. The three lens centres are measured on the
standard canvas and the camera area alone is moved onto the model's standard
lens positions: a smooth displacement field inside the camera plateau that
fades to zero at the body outline. The flash is a fixed anchor. Alpha, the
outline and everything below the plateau stay byte-identical.

Standard positions are the medians of light-coloured units that the user
confirmed are correct in the 3D viewer (Pro batch 20261110-104628-6b2a49; Pro
Max batch of 61), as fractions of the standard canvas.
"""
import cv2
import numpy as np
from scipy import ndimage

STANDARD = {
    'iphone17pro': dict(glass_radii=[99.0803, 98.3903, 99.0696], lenses=[[.1951, .0911], [.1975, .2242], [.4497, .1585]], flash=[.8115, .0899]),
    'iphone17promax': dict(glass_radii=[99.5637, 98.8577, 99.1951], lenses=[[.1791, .0824], [.1808, .2049], [.4138, .1441]], flash=[.8264, .0814]),
}
MAX_SHIFT_PX = 8.0       # larger means a misdetection or a different layout, not drift
MIN_SHIFT_PX = .30        # below this the back is left untouched


def _gray(a):
    alpha = a[..., 3:4] / 255.
    return cv2.cvtColor((a[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def _disk_readings(a):
    """Round components measured independently at each source gray threshold."""
    h, w = a.shape[:2]
    roi = _gray(a)[:int(.3 * h), :int(.6 * w)]
    for threshold in range(30, 90, 5):
        binary = cv2.morphologyEx((roi < threshold).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, _, stats, centres = cv2.connectedComponentsWithStats(binary, 8)
        disks = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if .12 * w < bw < .3 * w and .7 < bw / bh < 1.4 and area / (np.pi * bw * bh / 4) >= .8:
                disks.append((centres[i][0], centres[i][1], (bw + bh) / 4))
        yield disks


def _disks(a):
    """Three round dark disks in the camera corner, agreed over several thresholds."""
    readings = []
    for disks in _disk_readings(a):
        if len(disks) == 3:
            # Fixed layout: the right lens is the one furthest right; of the
            # other two, upper then lower.
            right = max(disks, key=lambda v: v[0])
            left = sorted([d for d in disks if d is not right], key=lambda v: v[1])
            readings.append(left + [right])
    if len(readings) < 3:
        return None, len(readings)
    readings = np.asarray(readings)
    # Extreme thresholds can clip a rim highlight or, on dark bodies, merge the
    # lens with its dark bezel. Keep the largest group of readings that agree
    # within 1.5 px with one of them.
    groups = [np.abs(readings[:, :, :2] - ref[:, :2]).max(axis=(1, 2)) <= 1.5 for ref in readings]
    agree = max(groups, key=lambda g: g.sum())
    if agree.sum() < 3:
        return None, len(readings)
    return np.median(readings[agree], axis=0), int(agree.sum())


def lens_centres(a):
    disks, count = _disks(a)
    return (None if disks is None else disks[:, :2]), count


def glass_centres(a, model):
    from .back_geometry import measure_glass_edges
    disks, count = _disks(a)
    if disks is None:
        raise ValueError('Three separate lens disks lack threshold consensus')
    measured = []
    for disk, radius in zip(disks, STANDARD[model]['glass_radii']):
        # Back-camera UV extents identify the glass annulus. Its position must
        # still be measured around each disk on this particular photograph.
        candidates = []
        for estimate in dict.fromkeys([float(disk[2]), radius, radius-1.5, radius+1.5]):
            for width in (8., 4., 3.):
                try:
                    read = measure_glass_edges(a, [[*disk[:2], estimate]], search_width=width)[0]
                except ValueError:
                    continue
                if abs(read[2] - radius) <= 5:
                    candidates.append(read)
        candidates = np.asarray(candidates)
        groups = [candidates[np.max(abs(candidates - ref), axis=1) <= .35] for ref in candidates]
        best = max(groups, key=len, default=[])
        if len(best) < 3:
            raise ValueError('Independent glass annulus readings disagree')
        edge = np.median(best, axis=0)
        measured.append(edge)
    return np.array(measured), count


def _dark_body_plateau_core(L):
    """Fallback for dark bodies where one top bevel window dips below 1.2 L/px.

    Requires zero/one-pixel agreement across all three windows and median >= 1.2
    so noisy/absent edges still fail closed. Used only for the lens-warp taper.
    """
    from .rear_shadow import InsufficientRearShadowEvidence

    h, w = L.shape
    gy, gx = np.gradient(ndimage.gaussian_filter(L, 1))
    definitions = [('left', gx, .035, .085, .06, .24, True),
                   ('right', gx, .90, .97, .10, .20, True),
                   ('top', gy, .015, .05, .40, .65, False),
                   ('bottom', gy, .24, .29, .57, .80, False)]
    edges = {}; proof = {}
    for name, g, lo, hi, aa, bb, vertical in definitions:
        size = w if vertical else h; along = h if vertical else w
        data = g[int(aa*along):int(bb*along), :] if vertical else g[:, int(aa*along):int(bb*along)].T
        signal = np.median(data, axis=0); indices = np.arange(int(lo*size), int(hi*size))
        anchor = indices[np.argmax(abs(signal[indices]))]
        positions = []; strength = []
        for window in np.array_split(data, 3):
            s = np.median(window, axis=0); near = indices[abs(indices-anchor) <= 4]
            position = int(near[np.argmax(abs(s[near]))]); positions.append(position); strength.append(float(abs(s[position])))
        strict_ok = min(strength) >= 1.2 and np.ptp(positions) <= 4
        dark_top_ok = (name == 'top' and np.ptp(positions) <= 1
                       and float(np.median(strength)) >= 1.2 and min(strength) >= 0.6)
        if not (strict_ok or dark_top_ok):
            raise InsufficientRearShadowEvidence('Rear shadow: plateau '+name+' lacks three-window source evidence')
        edges[name] = max(positions)+6 if name in ('left', 'top') else min(positions)-6
        proof[name] = dict(positions=positions, strengths=strength,
                           dark_body_top_fallback=bool(not strict_ok and dark_top_ok))
    bounds = [edges['left']-6, edges['top']-6, edges['right']+6, edges['bottom']+6]
    left, top, right, bottom = bounds; angles = np.linspace(.15, np.pi/2-.15, 100)
    corners = []
    for x, y, sx, sy in [(left, top, 1, 1), (right, top, -1, 1), (left, bottom, 1, -1), (right, bottom, -1, -1)]:
        radii = np.arange(int(w*.025), int(w*.20)); scores = []
        for radius in radii:
            X = x+sx*radius*(1-np.cos(angles)); Y = y+sy*radius*(1-np.sin(angles))
            normal = (ndimage.map_coordinates(gx, [Y, X], order=1)*np.cos(angles)
                      + sx*sy*ndimage.map_coordinates(gy, [Y, X], order=1)*np.sin(angles))
            scores.append(float(np.mean(abs(normal))))
        index = int(np.argmax(scores))
        if scores[index] < 2 or index in (0, len(radii)-1):
            raise InsufficientRearShadowEvidence('Rear shadow: plateau corner lacks source edge evidence')
        corners.append(dict(radius=int(radii[index]), source_edge_strength=scores[index]))
    proof['corners'] = corners
    return bounds, proof


def _plateau_clearance(a, roi, *, plateau_source=None):
    """Distance inside the photographed flat plateau, excluding its bevel and corners."""
    from .rear_shadow import InsufficientRearShadowEvidence, _plateau_core, _ring_material

    def _from_image(img, *, allow_dark_top=False):
        light = cv2.cvtColor(img[..., :3].astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[..., 0]
        bounds, evidence = _dark_body_plateau_core(light) if allow_dark_top else _plateau_core(light)
        height = min(len(a), bounds[3] + 4)
        y, x = np.ogrid[:height, :a.shape[1]]
        flat = _ring_material(a[:height, :, 3] == 255, dict(
            flat_core_bounds=bounds, plateau_source_edges=evidence, lenses=[]), y, x)
        clearance = np.zeros(roi, np.float32)
        rows = min(roi[0], height)
        clearance[:rows] = ndimage.distance_transform_edt(flat)[:rows, :roi[1]]
        return clearance

    last_error = None
    attempts = [(a, False, 'source_measured')]
    if plateau_source is not None and plateau_source.shape[:2] == a.shape[:2]:
        attempts.append((plateau_source, False, 'pre_lighting_source_measured'))
        attempts.append((plateau_source, True, 'pre_lighting_dark_body_source_measured'))
    attempts.append((a, True, 'dark_body_source_measured'))
    for img, allow_dark, label in attempts:
        try:
            return _from_image(img, allow_dark_top=allow_dark), label
        except InsufficientRearShadowEvidence as exc:
            last_error = exc
    raise last_error


def register(a, model, *, plateau_source=None):
    """Register the three photographed glass centres on their model-specific canvas."""
    spec = STANDARD.get(model)
    h, width = a.shape[:2]
    proof = dict(method='three_glass_edges_plateau_bounded_v3', model=model,
                 applied=False, verified=False)
    if spec is None:
        return a, dict(proof, reason='no_standard_for_model')
    target = np.asarray(spec['lenses']) * [width, h]
    proof['target'] = target.tolist()
    try:
        disks, count = glass_centres(a, model)
    except ValueError as exc:
        return a, dict(proof, reason='lenses_not_measured', detail=str(exc))
    shift = target - disks[:, :2]
    initial = np.linalg.norm(shift, axis=1)
    proof.update(threshold_readings=count, measured=disks[:, :2].tolist(),
                 shift=shift.tolist(), max_shift_px=float(np.abs(shift).max()))
    if proof['max_shift_px'] > MAX_SHIFT_PX:
        return a, dict(proof, reason='shift_exceeds_limit')
    if initial.max() <= MIN_SHIFT_PX:
        return a, dict(proof, reason='already_aligned', verified=True, after=disks[:, :2].tolist(),
                       residual_px=float(initial.max()), per_lens_error_px=initial.tolist())
    roi_h = round(.31 * h)
    y, x = np.mgrid[:roi_h, :round(.65 * width)].astype(np.float32)
    field = np.zeros((*x.shape, 2), np.float32)
    from .rear_shadow import InsufficientRearShadowEvidence
    plateau_boundary_label = 'source_measured'
    try:
        clearance, plateau_boundary_label = _plateau_clearance(a, x.shape, plateau_source=plateau_source)
    except InsufficientRearShadowEvidence as exc:
        clearance = None
        proof['plateau_evidence'] = str(exc)

    for (cx, cy), radius, move in zip(target, disks[:, 2], shift):
        distance = np.hypot(x-cx, y-cy) / radius
        t = np.clip((1.52-distance)/.34, 0, 1)
        weight = t*t*(3-2*t)
        if clearance is not None:
            # Both coordinates are bounded: the inverse sampling location can
            # be displaced by |move|. Taper before reaching the actual ledge.
            margin = np.clip(clearance / max(12., 3.1*np.linalg.norm(move)), 0, 1)
            weight = np.minimum(weight, margin*margin*(3-2*margin))
        field += weight[..., None] * move
    mx, my = x-field[..., 0], y-field[..., 1]
    jac = np.gradient(mx,axis=1)*np.gradient(my,axis=0)-np.gradient(mx,axis=0)*np.gradient(my,axis=1)
    active = (abs(field).max(axis=2) > 0)
    source = a[:x.shape[0], :x.shape[1]]
    if clearance is None:
        # A uniformly coloured field outside the three rims has no photographed
        # structure for the taper to distort. Otherwise retain the input.
        surround = active.copy()
        for (cx, cy, r), move in zip(disks, shift):
            surround &= np.hypot(x-cx, y-cy) > 1.24*r + np.linalg.norm(move)
        sample = source[surround, :3]
        if not len(sample) or np.any(sample.max(axis=0) != sample.min(axis=0)):
            return a, dict(proof, reason='registration_not_verified', detail='Plateau boundary cannot be verified')
        proof['plateau_boundary'] = 'uniform_exterior'
    else:
        proof['plateau_boundary'] = plateau_boundary_label

    image = cv2.remap(a, mx, my, cv2.INTER_CUBIC)
    if jac.min() < .5 or jac.max() > 1.55 or np.any(source[active,3]!=255) or np.any(image[active,3]!=255):
        return a, dict(proof, reason='registration_not_verified', detail='Lens warp would affect body contour or geometry')
    out = a.copy()
    out[:x.shape[0], :x.shape[1]][active] = image[active]
    try:
        after, _ = glass_centres(out, model)
    except ValueError as exc:
        return a, dict(proof, reason='registration_not_verified', detail=str(exc))
    errors = np.linalg.norm(after[:, :2]-target, axis=1)
    if errors.max() > MIN_SHIFT_PX:
        # On closely spaced rims cubic ringing can change which photographed
        # gradient wins the glass fit. Retry only that disk, with the same
        # inverse map from the source. Never apply a second geometric move.
        failed = np.flatnonzero(errors > MIN_SHIFT_PX)
        proof['first_pass_errors_px'] = errors.tolist()
        blend = np.zeros_like(x)
        for index in failed:
            cx, cy = target[index]
            distance = np.hypot(x - cx, y - cy) / disks[index, 2]
            taper = np.clip((1.30 - distance) / .14, 0, 1)
            blend = np.maximum(blend, taper * taper * (3 - 2 * taper))
        retry_pixels = active & (blend > 0)
        linear = cv2.remap(a, mx, my, cv2.INTER_LINEAR)
        if np.any(linear[retry_pixels, 3] != 255):
            return a, dict(proof, reason='registration_not_verified', detail='Retry touches transparency')
        patch = out[:x.shape[0], :x.shape[1]]
        original = patch[retry_pixels].astype(np.float32)
        patch[retry_pixels] = np.rint(original + blend[retry_pixels, None] *
                                     (linear[retry_pixels] - original)).astype(np.uint8)
        try:
            after, _ = glass_centres(out, model)
        except ValueError as exc:
            return a, dict(proof, reason='registration_not_verified', detail=str(exc))
        errors = np.linalg.norm(after[:, :2] - target, axis=1)
        radius_drift = abs(after[failed, 2] - disks[failed, 2])
        proof.update(linear_retry_lenses=failed.tolist(), retry_radius_drift_px=radius_drift.tolist())
        if errors.max() > MIN_SHIFT_PX or radius_drift.max() > MIN_SHIFT_PX:
            return a, dict(proof, reason='registration_not_verified', residual_px=float(errors.max()))
    return out, dict(proof, applied=True, verified=True, after=after[:, :2].tolist(),
                     residual_px=float(errors.max()), per_lens_error_px=errors.tolist(),
                     jacobian=[float(jac.min()), float(jac.max())],
                     changed_pixels=int(np.count_nonzero(np.any(out != a, axis=2))))
