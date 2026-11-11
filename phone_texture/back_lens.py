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
    'iphone17pro': dict(lenses=[[.1951, .0911], [.1975, .2242], [.4497, .1585]], flash=[.8115, .0899]),
    'iphone17promax': dict(lenses=[[.1791, .0824], [.1808, .2049], [.4138, .1441]], flash=[.8264, .0814]),
}
MAX_SHIFT_PX = 8.0       # larger means a misdetection or a different layout, not drift
MIN_SHIFT_PX = .5        # below this the back is left untouched


def _gray(a):
    alpha = a[..., 3:4] / 255.
    return cv2.cvtColor((a[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def lens_centres(a):
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
                disks.append((centres[i][0], centres[i][1]))
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
    groups = [np.abs(readings - ref).max(axis=(1, 2)) <= 1.5 for ref in readings]
    agree = max(groups, key=lambda g: g.sum())
    if agree.sum() < 3:
        return None, len(readings)
    return np.median(readings[agree], axis=0), int(agree.sum())


def register(a, model):
    """Return (image, proof). Untouched when not measurable or already aligned."""
    spec = STANDARD.get(model)
    h, w = a.shape[:2]
    proof = dict(method='lens_centre_local_registration_v1', model=model, applied=False)
    if spec is None:
        return a, dict(proof, reason='no_standard_for_model')
    target = np.asarray(spec['lenses']) * [w, h]
    flash = np.asarray(spec['flash']) * [w, h]
    measured, readings = lens_centres(a)
    proof.update(threshold_readings=readings, target=target.round(2).tolist())
    if measured is None:
        return a, dict(proof, reason='lenses_not_measured')
    shift = target - measured
    proof.update(measured=measured.round(2).tolist(), shift=shift.round(2).tolist(),
                 max_shift_px=float(np.abs(shift).max()))
    if proof['max_shift_px'] > MAX_SHIFT_PX:
        return a, dict(proof, reason='shift_exceeds_limit')
    if proof['max_shift_px'] < MIN_SHIFT_PX:
        return a, dict(proof, reason='already_aligned')
    # Inverse-distance blend of landmark shifts (flash anchored at zero).
    points = np.vstack([target, flash]); moves = np.vstack([shift, [[0, 0]]])
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    num = np.zeros((h, w, 2), np.float32); den = np.zeros((h, w), np.float32)
    soft = (.02 * w) ** 2
    for (px, py), move in zip(points, moves):
        weight = 1 / ((xx - px) ** 2 + (yy - py) ** 2 + soft) ** 1.5
        num += weight[..., None] * move; den += weight
    field = num / den[..., None]
    # Confine to the camera plateau and fade to zero at the body outline.
    x1, y1 = .96 * w, .31 * h; ramp = .04 * w
    inside = np.clip((x1 - xx) / ramp, 0, 1) * np.clip((y1 - yy) / ramp, 0, 1)
    edge = np.clip(ndimage.distance_transform_edt(a[..., 3] >= 250) / (.03 * w), 0, 1)
    taper = inside * edge; taper = taper * taper * (3 - 2 * taper)
    field *= taper[..., None]
    premultiplied = a.astype(np.float32) / 255; premultiplied[..., :3] *= premultiplied[..., 3:4]
    sampled = cv2.remap(premultiplied, xx - field[..., 0], yy - field[..., 1], cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_REPLICATE)
    rgb = sampled[..., :3] / np.maximum(a[..., 3:4] / 255., 1e-6)
    out = a.copy()
    moved = (np.abs(field).max(axis=2) > 1e-3) & (a[..., 3] > 0)
    out[moved, :3] = np.clip(np.rint(rgb[moved] * 255), 0, 255).astype(np.uint8)
    after, _ = lens_centres(out)
    residual = None if after is None else float(np.abs(after - target).max())
    # Verified when every lens lands within 1 px, or at least halves the error.
    if residual is None or residual > max(1., .5 * proof['max_shift_px']):
        return a, dict(proof, reason='registration_not_verified', residual_px=residual)
    if not np.array_equal(out[..., 3], a[..., 3]):
        raise ValueError('Lens registration changed alpha')
    return out, dict(proof, applied=True, residual_px=residual,
                     max_displacement_px=float(np.abs(field).max()),
                     changed_pixels=int(np.count_nonzero(np.any(out != a, axis=2))))
