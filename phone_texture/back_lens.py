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


def _disks(a):
    """Three round dark disks in the camera corner, agreed over several thresholds."""
    h, w = a.shape[:2]
    roi = _gray(a)[:int(.3 * h), :int(.6 * w)]
    readings = []
    for threshold in range(30, 90, 5):
        binary = cv2.morphologyEx((roi < threshold).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, _, stats, centres = cv2.connectedComponentsWithStats(binary, 8)
        disks = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if .12 * w < bw < .3 * w and .7 < bw / bh < 1.4 and area / (np.pi * bw * bh / 4) >= .8:
                disks.append((centres[i][0], centres[i][1], (bw + bh) / 4))
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


def register(a, model):
    """Register the three photographed glass centres on their model-specific canvas."""
    spec = STANDARD.get(model)
    h, width = a.shape[:2]
    proof = dict(method='three_glass_edges_local_registration_v2', model=model,
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
    for (cx, cy), radius, move in zip(target, disks[:, 2], shift):
        distance = np.hypot(x-cx, y-cy) / radius
        t = np.clip((1.52-distance)/.34, 0, 1)
        field += (t*t*(3-2*t))[..., None] * move
    mx, my = x-field[..., 0], y-field[..., 1]
    jac = np.gradient(mx,axis=1)*np.gradient(my,axis=0)-np.gradient(mx,axis=0)*np.gradient(my,axis=1)
    active = (abs(field).max(axis=2) > 0)
    source = a[:x.shape[0], :x.shape[1]]
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
        return a, dict(proof, reason='registration_not_verified', residual_px=float(errors.max()))
    return out, dict(proof, applied=True, verified=True, after=after[:, :2].tolist(),
                     residual_px=float(errors.max()), per_lens_error_px=errors.tolist(),
                     jacobian=[float(jac.min()), float(jac.max())],
                     changed_pixels=int(np.count_nonzero(np.any(out != a, axis=2))))
