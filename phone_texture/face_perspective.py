"""Default perspective rectification of the planar front/back faces.

User decision 2026-11-05: front and back are planes, so a camera tilt is
removed from the delivered image itself. Each face is placed and shot on its
own, so every face is measured from its own pixels only.

The reference must lie in the face plane. The outer outline does not: the
side rail stands above the glass and lens bowing curves it. Back uses the
inset glass-panel seam; front uses the black screen-glass edge. Their four
straight middle spans give a quadrilateral that is mapped to a rectangle with
one premultiplied cubic resample. A taper within noise, or any failed check,
keeps the exact input PNG; rectification never blocks generation.
"""
from pathlib import Path
import copy
import shutil

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

from .geometry_review import digest, write

FACES = ('front', 'back')
MIN_TAPER_PX = 3.0      # opposite-side length difference below this is noise
MAX_CORNER_MOVE = .03   # of the reference width; larger means a mis-fit, not tilt
MAX_FIT_RMS_PX = 1.5
MAX_RESIDUAL_TAPER_PX = 2.0
DARK_GLASS_MARGIN = 20.  # L8 above the screen-glass median still counts as glass


class Rejected(ValueError):
    pass


def _robust_line(coords, positions):
    if len(coords) < 40:
        raise Rejected('too_few_straight_reference_samples')
    keep = np.ones(len(coords), bool)
    for _ in range(5):
        fit = np.polyfit(coords[keep], positions[keep], 1)
        residual = positions-np.polyval(fit, coords)
        scale = max(.35, 1.4826*np.median(np.abs(residual[keep])))
        keep = np.abs(residual) <= 3*scale
    rms = float(np.sqrt(np.mean(residual[keep]**2)))
    if keep.mean() < .7 or rms > MAX_FIT_RMS_PX:
        raise Rejected('reference_edge_not_straight')
    return fit, dict(samples=int(keep.sum()), inlier_fraction=float(keep.mean()),
                     rms_px=rms, slope=float(fit[0]))


def _profile_peak(profile, lo, hi, pick, strength=.5):
    """Subpixel location of the strongest (or outermost strong) peak in [lo, hi)."""
    seg = profile[lo:hi]
    if not len(seg) or not np.isfinite(seg).all() or seg.max() <= 0:
        return None
    if pick == 'outer_low':      # first strong peak from the low end
        strong = np.nonzero(seg >= max(.8, strength*seg.max()))[0]
        i = int(strong[0])
        while i+1 < len(seg) and seg[i+1] > seg[i]:
            i += 1
    elif pick == 'outer_high':   # first strong peak from the high end
        strong = np.nonzero(seg >= max(.8, strength*seg.max()))[0]
        i = int(strong[-1])
        while i > 0 and seg[i-1] > seg[i]:
            i -= 1
    else:
        i = int(np.argmax(seg))
    if 0 < i < len(seg)-1:
        y0, y1, y2 = seg[i-1], seg[i], seg[i+1]
        d = y0-2*y1+y2
        offset = .5*(y0-y2)/d if d != 0 else 0.
        return lo+i+float(np.clip(offset, -.5, .5))
    return float(lo+i)


def _signal(a, face):
    """Per-pixel evidence for the in-plane reference edge."""
    gray = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY).astype(float)
    if face == 'back':
        # Thin dark seam between glass panel and aluminium (same as rear_panel).
        return ndimage.gaussian_filter(gray, 3)-ndimage.gaussian_filter(gray, .6), None
    # Screen glass is the large dark region; its edge is the luminance step
    # from the bright bezel/rim into the dark glass, measured inward.
    smooth = ndimage.gaussian_filter(gray, 1.)
    return smooth, np.median(smooth[a[..., 3] >= 250])


def _samples(a, face, side, signal, dark_level, method=None):
    h, w = a.shape[:2]
    opaque = a[..., 3] >= 250
    vertical = side in ('left', 'right')
    if face == 'back':
        # Below the camera plateau; across, skip the rounded corners and the
        # centred QC sticker that covers the lower seam.
        span = (range(int(h*.40), int(h*.85), 3) if vertical else
                list(range(int(w*.20), int(w*.38), 3))+list(range(int(w*.62), int(w*.80), 3)))
    else:
        # Across, skip the centred Dynamic Island and the rounded corners.
        span = (range(int(h*.30), int(h*.80), 3) if vertical else
                list(range(int(w*.20), int(w*.38), 3))+list(range(int(w*.62), int(w*.80), 3)))
    coords, positions = [], []
    for i in span:
        line = signal[i-1:i+2].mean(axis=0) if vertical else signal[:, i-1:i+2].mean(axis=1)
        inside = np.nonzero(opaque[i] if vertical else opaque[:, i])[0]
        if len(inside) < 50:
            continue
        n = len(line)
        if face == 'back':
            # Seam is a few percent of the width inside the outline.
            if side in ('left', 'right'):
                lo, hi = (int(w*.04), int(w*.13)) if side == 'left' else (int(w*.87), int(w*.96))
            else:
                lo, hi = (int(h*.27), int(h*.39)) if side == 'top' else (int(h*.90), int(h*.99))
            # The plateau's lower seam sits just above the panel's top seam:
            # take the lowest strong peak there, as rear_panel does.
            pos = (_profile_peak(line, lo, hi, 'outer_high', strength=.25) if side == 'top'
                   else _profile_peak(line, lo, hi, 'max'))
        else:
            edge = int(inside[0]) if side in ('left', 'top') else int(inside[-1])
            depth = int(max(h, w)*.03)
            pos = (_glass_by_walk(line, n, edge, depth, side, dark_level) if method == 'walk'
                   else _glass_by_gradient(line, n, edge, depth, side, dark_level))
        if pos is not None:
            coords.append(i)
            positions.append(pos)
    return np.asarray(coords, float), np.asarray(positions, float)


def _glass_by_walk(line, n, edge, depth, side, dark_level):
    """Walk outward from inside the dark screen to where it stops being dark.

    Robust to a strong metal-rim gradient outside a weaker glass edge, but a
    rim reflection lying on the glass stops the walk early.
    """
    step = -1 if side in ('left', 'top') else 1
    start = edge+depth if side in ('left', 'top') else edge-depth
    level = dark_level+DARK_GLASS_MARGIN
    if not 0 <= start < n or line[start] > level:
        return None
    j = start
    while 0 <= j+step < n and (j+step-edge)*step <= 0 and line[j+step] <= level:
        j += step
    k = j+step
    if 0 <= k < n and (k-edge)*step <= 0 and line[k] > level:
        return j+step*(level-line[j])/(line[k]-line[j])
    return None


def _glass_by_gradient(line, n, edge, depth, side, dark_level):
    """Innermost strong step into dark glass; a reflection on the glass is
    weaker than this step, but a much stronger rim step can hide it."""
    grad = -np.gradient(line) if side in ('left', 'top') else np.gradient(line)
    lo, hi = (edge+2, min(n, edge+depth)) if side in ('left', 'top') else (max(0, edge-depth), edge-1)
    g = np.where(np.isfinite(grad), grad, 0)
    pos = _profile_peak(g, lo, hi, 'outer_high' if side in ('left', 'top') else 'outer_low')
    if pos is not None:
        probe = int(round(pos))+(4 if side in ('left', 'top') else -4)
        if not 0 <= probe < n or line[probe] > dark_level+25:
            return None
    return pos


def _outline_bottom_seam(a, signal):
    """Lower panel seam searched at a fixed inset above the body outline.

    The fixed h*.90–.99 window can catch the outline itself or a QC-sticker
    edge when the sticker covers part of the seam. The seam sits ~2–6.5 % of
    the width above the outline in every measured SKU, so search only there.
    """
    h, w = a.shape[:2]
    opaque = a[..., 3] >= 250
    coords, positions = [], []
    for i in list(range(int(w*.20), int(w*.38), 3))+list(range(int(w*.62), int(w*.80), 3)):
        inside = np.nonzero(opaque[:, i])[0]
        if len(inside) < 50:
            continue
        edge = int(inside[-1])
        pos = _profile_peak(signal[:, i-1:i+2].mean(axis=1), edge-int(w*.065), edge-int(w*.02), 'max')
        if pos is not None:
            coords.append(i)
            positions.append(pos)
    return np.asarray(coords, float), np.asarray(positions, float)


def _fit_side(a, face, side, signal, dark, bottom_fallback=False):
    """Front edges try both glass detectors and keep the straighter line."""
    if face == 'back':
        if side == 'bottom' and bottom_fallback:
            line, fit = _robust_line(*_outline_bottom_seam(a, signal))
            fit['method'] = 'outline_inset_fallback'
            return line, fit
        return _robust_line(*_samples(a, face, side, signal, dark))
    best, errors = None, []
    for method in ('walk', 'gradient'):
        try:
            line, fit = _robust_line(*_samples(a, face, side, signal, dark, method))
        except Rejected as error:
            errors.append(str(error))
            continue
        fit['method'] = method
        if best is None or fit['rms_px'] < best[1]['rms_px']:
            best = (line, fit)
    if best is None:
        raise Rejected(errors[0])
    return best


def reference_quad(a, face, bottom_fallback=False):
    """In-plane reference lines and their intersections (TL, TR, BL, BR)."""
    signal, dark = _signal(a, face)
    lines, fits = {}, {}
    for side in ('left', 'right', 'top', 'bottom'):
        lines[side], fits[side] = _fit_side(a, face, side, signal, dark, bottom_fallback)

    def meet(vertical, horizontal):     # x = a*y+b, y = c*x+d
        p, q = vertical
        r, t = horizontal
        y = (r*q+t)/(1-r*p)
        return [p*y+q, y]
    quad = np.array([meet(lines['left'], lines['top']), meet(lines['right'], lines['top']),
                     meet(lines['left'], lines['bottom']), meet(lines['right'], lines['bottom'])])
    return quad, fits


def tapers(quad):
    tl, tr, bl, br = quad
    return (float(np.linalg.norm(br-bl)-np.linalg.norm(tr-tl)),
            float(np.linalg.norm(br-tr)-np.linalg.norm(bl-tl)))


def target_rect(quad):
    """Axis-aligned rectangle with the mean side lengths, centred on the quad."""
    tl, tr, bl, br = quad
    width = (np.linalg.norm(tr-tl)+np.linalg.norm(br-bl))/2
    height = (np.linalg.norm(bl-tl)+np.linalg.norm(br-tr))/2
    c = quad.mean(axis=0)
    return np.array([c+[-width/2, -height/2], c+[width/2, -height/2],
                     c+[-width/2, height/2], c+[width/2, height/2]]), width, height


def warp(a, quad, rect):
    """One premultiplied cubic resample, cropped to the visible body plus 1 px."""
    h, w = a.shape[:2]
    matrix = cv2.getPerspectiveTransform(quad.astype(np.float32), rect.astype(np.float32))
    frame = cv2.perspectiveTransform(np.array([[[0, 0], [w, 0], [0, h], [w, h]]], np.float32), matrix)[0]
    origin = np.floor(frame.min(axis=0))-2
    shift = np.array([[1, 0, -origin[0]], [0, 1, -origin[1]], [0, 0, 1]])
    full = shift@matrix
    size = tuple(int(v) for v in np.ceil(frame.max(axis=0)-origin)+2)
    pre = a.astype(np.float32)
    pre[..., :3] *= pre[..., 3:]/255.
    out = cv2.warpPerspective(pre, full, size, flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = np.clip(out[..., 3], 0, 255)
    rgb = np.where(alpha[..., None] > .5, out[..., :3]*255./np.maximum(alpha[..., None], 1e-6), 0)
    result = np.dstack([np.clip(np.rint(rgb), 0, 255), np.rint(alpha)]).astype(np.uint8)
    result[result[..., 3] == 0, :3] = 0
    # Same tight canvas as trim_visible_canvas: body (alpha>=128) plus one
    # antialias pixel; cubic ringing further out is soft-matte padding.
    ys, xs = np.nonzero(result[..., 3] >= 128)
    y0, y1 = max(0, ys.min()-1), min(result.shape[0], ys.max()+2)
    x0, x1 = max(0, xs.min()-1), min(result.shape[1], xs.max()+2)
    crop = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]])
    return result[y0:y1, x0:x1], crop@full


def rectify(a, face):
    """Return (image, proof); image is None when the source is kept."""
    try:
        return _rectify(a, face, False)
    except Rejected:
        if face != 'back':
            raise
    # User decision 2026-11-06: a QC sticker over the lower seam must not
    # drop the whole back, since antenna targets come from this face. The
    # fallback runs only after the normal fit fails, so backs that already
    # rectify keep byte-identical output.
    return _rectify(a, face, True)


def _rectify(a, face, bottom_fallback):
    quad, fits = reference_quad(a, face, bottom_fallback)
    rect, width, height = target_rect(quad)
    move = float(np.max(np.linalg.norm(quad-rect, axis=1)))
    tb, lr = tapers(quad)
    proof = dict(reference='rear_glass_panel_seam' if face == 'back' else 'front_screen_glass_edge',
                 quad=quad.tolist(), target=rect.tolist(), target_size=[width, height],
                 max_corner_move_px=move, edge_fits=fits, taper_top_bottom_px=tb, taper_left_right_px=lr,
                 bottom_seam='outline_inset_fallback' if bottom_fallback else 'fixed_window')
    if max(abs(tb), abs(lr)) < MIN_TAPER_PX:
        return None, dict(proof, action='unchanged_taper_within_noise')
    if move > MAX_CORNER_MOVE*width:
        raise Rejected('corner_move_exceeds_plausible_tilt')
    out, matrix = warp(a, quad, rect)
    after, _ = reference_quad(out, face, bottom_fallback)
    residual = [abs(v) for v in tapers(after)]
    if max(residual) > MAX_RESIDUAL_TAPER_PX or max(residual) >= max(abs(tb), abs(lr)):
        raise Rejected('reference_not_rectangular_after_rectification')
    area = [float(np.count_nonzero(img[..., 3] >= 128)) for img in (a, out)]
    if not .97 <= area[1]/area[0] <= 1.03:
        raise Rejected('body_area_changed_excessively')
    return out, dict(proof, action='rectified', homography=matrix.tolist(),
                     residual_taper_px=dict(top_bottom=residual[0], left_right=residual[1]),
                     output_size=[out.shape[1], out.shape[0]],
                     opaque_area_ratio=area[1]/area[0], resampling='single_premultiplied_cubic')


def prepare(files, geometry, process):
    """Rectify front/back into a new selected folder; geometry rows are re-bound."""
    process = Path(process)
    process.mkdir(parents=True, exist_ok=False)
    selected = process/'selected'
    selected.mkdir()
    before = {f: digest(p) for f, p in files.items()}
    report = dict(version='planar-face-perspective-v1', default_enabled=True,
                  blocks_generation=False, model_uv_verified=False, faces={}, notices=[])
    effective = copy.deepcopy(geometry)
    rows = {item['face']: item for item in effective['images']}
    for face, path in files.items():
        target = selected/(face+'.png')
        if face not in FACES:
            shutil.copy2(path, target)
            continue
        with Image.open(path) as image:
            if image.mode != 'RGBA' or image.format != 'PNG':
                raise ValueError('Perspective input must be RGBA PNG: '+face)
            a = np.asarray(image).copy()
        entry = dict(source=str(path), source_sha256=before[face], applied=False)
        try:
            out, proof = rectify(a, face)
            entry.update(proof)
        except (Rejected, np.linalg.LinAlgError, FloatingPointError) as error:
            out = None
            entry.update(action='kept_source_with_notice', reason=str(error))
            report['notices'].append(dict(face=face, code='perspective_not_applied', severity='review',
                message='未应用透视纠正，保留原图；请核对该面是否倾斜。', reason=str(error)))
        if out is not None:
            Image.fromarray(out).save(target)
            entry.update(applied=True)
            report['notices'].append(dict(face=face, code='perspective_rectified', severity='info',
                message='已按面内参考纠正拍摄透视（%.1f px）。' % entry['max_corner_move_px']))
        else:
            shutil.copy2(path, target)
        entry['output_sha256'] = digest(target)
        report['faces'][face] = entry
        row = rows[face]
        if row['output_sha256'] != before[face]:
            raise ValueError('Geometry provenance mismatch before perspective: '+face)
        if entry['applied']:
            # Evidence built on the unrectified capture (body reference) is
            # validated against this frozen row, never against the new one.
            row['pre_perspective_row'] = copy.deepcopy(row)
            row['pre_perspective_sha256'] = before[face]
            row['output_sha256'] = entry['output_sha256']
            row.setdefault('geometry', {})['output_size'] = entry['output_size']
            row['geometry']['planar_perspective'] = dict(record=str(process/'report.json'),
                source_sha256=before[face], output_sha256=entry['output_sha256'],
                homography=entry['homography'])
    if any(digest(files[f]) != h for f, h in before.items()):
        raise ValueError('Geometry input changed during perspective rectification')
    for face in files:
        if face not in FACES or not report['faces'][face]['applied']:
            if digest(selected/(face+'.png')) != before[face]:
                raise ValueError('Unrectified face is not byte identical: '+face)
    report['applied_faces'] = [f for f in FACES if report['faces'][f]['applied']]
    write(process/'report.json', report)
    write(process/'effective-geometry.json', effective)
    return {f: selected/(f+'.png') for f in files}, effective, report
