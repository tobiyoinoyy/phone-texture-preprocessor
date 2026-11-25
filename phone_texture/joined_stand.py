"""Locate a colored lower rail when the neutral support fills its alpha gaps."""
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import least_squares

from scripts.geometry_postprocess import rotate_alpha_safe


def central_rail(raw, columns, rows):
    from .auto_geometry import consensus_line

    strip = raw[rows[:, None], columns].astype(float)
    chroma = np.ptp(strip, axis=2)
    outside_chroma = np.percentile(chroma[-8:], 80, axis=0)
    colored = chroma > np.maximum(30, outside_chroma + 18)
    continuous = ndimage.minimum_filter1d(colored.astype(np.uint8), 3, axis=0)
    end = np.max(np.where(continuous, np.arange(len(rows))[:, None], -1), axis=0)
    valid = (end > 3) & (end < len(rows) - 6) & (outside_chroma < 28)
    indices = np.flatnonzero(valid)
    offsets = end[valid, None] + np.arange(-2, 4)
    gray = strip.mean(axis=2)
    gradient = np.gradient(ndimage.gaussian_filter1d(gray, .8, axis=0), axis=0)
    strength = gradient[offsets, indices[:, None]]
    peaks = offsets[np.arange(len(indices)), strength.argmax(axis=1)]
    strong = strength.max(axis=1) > 6
    observed = columns[indices[strong]]
    line, inliers, residual = consensus_line(observed + .5, rows[peaks[strong]] + .5,
                                           2., np.tan(np.deg2rad(4)))
    evidence = np.isin(columns, observed[inliers])
    fractions = [float(group.mean()) for group in np.array_split(evidence, 2)]
    if min(fractions) < .60 or residual > 1.5 or abs(line[0]) > np.tan(np.deg2rad(4)):
        raise ValueError('No two-sided colored rail transition at the stand')
    return line, dict(raw_transition_half_coverage=fractions, raw_residual_p95_px=residual)


def corner_drop(raw, distance, width, rail, bottom):
    # Fit each exposed rounded shoulder into the straight rail. Evaluate the
    # fitted shape only on columns that actually extend into the support.
    shoulder = ((distance >= .07) & (distance < .22) &
                (bottom < rail + 2.5) & (bottom >= rail - .055 * width))
    x = np.flatnonzero(shoulder)
    if len(x) < max(16, .05 * width) or np.ptp(x) < .065 * width:
        raise ValueError('Too little exposed corner beside the attached stand')
    y = bottom[x]
    if np.any(y < 4) or np.any(y + 3 >= len(raw)):
        raise ValueError('Corner contact is clipped in the source')
    color_step = np.linalg.norm(raw[y+2, x].astype(float) - raw[y-3, x], axis=1)
    if np.mean(color_step > 18) < .85:
        raise ValueError('Exposed corner lacks raw RGB boundary support')
    sag = rail[x] - y
    def residual(params):
        scale, join = params
        return scale * np.maximum(join - distance[x], 0)**2.5 - sag
    fit = least_squares(residual, [6*width, .19], bounds=([0., .08], [50*width, .34]),
                        loss='soft_l1')
    error = float(np.percentile(abs(fit.fun), 95))
    if error > 2 or np.min(abs(sag)) > 10 or np.ptp(sag) < 6:
        raise ValueError('Exposed corner does not converge on the central rail')
    scale, join = fit.x
    return scale * np.maximum(join - distance, 0)**2.5, dict(
        exposed_columns=[int(x.min()), int(x.max())], samples=len(x), residual_p95_px=error)


def verify_label_end(raw, paper, limit, width):
    if paper is None or not paper.any():
        return
    ys, xs = np.nonzero(paper)
    if np.any(ys > limit[xs] + 1):
        raise ValueError('Paper extends below the inferred rail')
    columns = np.unique(xs[ys >= limit[xs] - .04 * width])
    if not len(columns):
        return
    # A paper seed can stop at the last printed row. Check for continuing
    # paper/ink below it against the actual stand on either side of the label.
    nearby = np.r_[np.arange(columns.min()-12, columns.min()-3),
                   np.arange(columns.max()+4, columns.max()+13)]
    if nearby.min() < 0 or nearby.max() >= len(limit):
        raise ValueError('No surrounding stand beside label')
    def strip(x):
        y = np.rint(limit[x]).astype(int)
        if y.min() < 0 or y.max() + 13 >= len(raw):
            raise ValueError('Label contact lies outside the source photograph')
        return raw[y[None, :] + np.arange(3, 14)[:, None], x].astype(float)
    left, right = np.array_split(nearby, 2)
    # The stand itself can have a broad light gradient across the label width.
    mix = ((columns - left.mean()) / (right.mean()-left.mean()))[None, :, None]
    reference = ((1-mix) * np.median(strip(left), axis=1, keepdims=True)
                 + mix * np.median(strip(right), axis=1, keepdims=True))
    delta = np.linalg.norm(strip(columns) - reference, axis=2)
    if np.percentile(delta, 95) > 24 or np.any((delta > 30).sum(axis=0) >= 5):
        raise ValueError('Photographed label end is unresolved at the support')


def cut_joined_stand(matte, mask, raw, paper, x0, x1, last, axis, stage_sink):
    from .auto_geometry import front_support_coverage

    h, w = mask.shape
    width = x1-x0
    columns = np.arange(int(x0+.28*width), int(x1-.28*width))
    radius = int(.05*width)
    if radius < 8 or last-radius < 0 or last+radius+14 >= h:
        raise ValueError('Insufficient photograph below the rail')
    rows = np.arange(last-radius, last+radius+1)
    if mask[last+radius, columns].mean() < .85:
        raise ValueError('Central matte is not joined to a wide stand')
    line, proof = central_rail(raw, columns, rows)
    x = np.arange(w)
    rail = np.polyval(line, x+.5)
    bottom = h - np.argmax(mask[::-1], axis=0)
    limit = rail.copy()
    corners = []
    for distance in ((x-x0)/width, (x1-x)/width):
        drop, evidence = corner_drop(raw, distance, width, rail, bottom)
        limit -= drop
        corners.append(evidence)
    attached = (bottom > rail + 5) & mask.any(axis=0)
    if not attached.any() or any(np.any(attached[section]) for section in
            (slice(None, corners[0]['exposed_columns'][0]), slice(corners[1]['exposed_columns'][1]+1, None))):
        raise ValueError('Stand extends beyond the witnessed corners')
    # Beyond the shoulder fit the original round silhouette is authoritative.
    contact = ndimage.binary_dilation(attached, iterations=2)
    limit[~contact] = rail[~contact]
    verify_label_end(raw, paper, limit, width)
    a = np.array(matte.convert('RGBA'))
    a[..., 3] = front_support_coverage(a[..., 3], limit[None, :], contact[None, :])
    corrected = Image.fromarray(a)
    rotated = Image.fromarray(rotate_alpha_safe(a, axis['rotation_ccw_degrees']))
    if stage_sink:
        stage_sink('source_matte', matte)
        stage_sink('after_support', corrected)
        stage_sink('after_rotation', rotated)
        stage_sink('after_crop', rotated)
    box = rotated.getchannel('A').getbbox()
    image = rotated.crop(box)
    return image, dict(method='joined_stand_source_rail_and_exposed_corners', axis_evidence=axis,
        bottom_rail=line.tolist(), lower_plane_evidence=dict(**proof, exposed_corners=corners),
        rotation_ccw_degrees=axis['rotation_ccw_degrees'], source_size=list(matte.size),
        output_size=list(image.size), canvas_trim_xyxy=list(box), rgb_painted=False,
        review_reasons=['heuristic_support_cut','complete_outline_requires_review'])
