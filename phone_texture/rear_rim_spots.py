"""Share the illumination of the upper rear metal rail across compact spot halos."""
import cv2
import numpy as np
from scipy import ndimage as ndi


def correct(source, lit, metal, alpha, plateau_top=None):
    """Return an in-gamut candidate target (bounded by the caller) for supported spots."""
    h, w = metal.shape
    height = min(h, round(h * .045))
    if plateau_top is not None:
        height = min(height, int(plateau_top) - max(4, round(w * .006)))
    report = dict(spots=0, pixels=0)
    if height < 8 or w < 80:
        return None, report
    src, dst = source[:height], lit[:height]
    scope = metal[:height] & (alpha[:height] == 255)
    spacing = max(7, round(w * .023))
    # The witnesses must be on a straight portion of this rail, on the same
    # scanline and surface, with solid metal all the way between them.
    top_edge = np.argmax(alpha >= 240, axis=0)
    span = 2 * spacing + 1
    straight = ndi.maximum_filter1d(top_edge, span) - ndi.minimum_filter1d(top_edge, span) <= 2
    valid = ndi.minimum_filter1d(scope.astype('uint8'), span, axis=1, mode='constant') > 0
    valid &= straight & (np.arange(height)[:, None] > top_edge + 3)
    if not valid.any():
        return None, report

    def witnesses(image):
        smooth = ndi.gaussian_filter1d(image, max(1, spacing / 6), axis=1)
        left = np.roll(smooth, spacing, axis=1)
        right = np.roll(smooth, -spacing, axis=1)
        return (left + right) * .5, np.abs(left - right)

    raw_ref, disagreement = witnesses(src)
    lit_ref, _ = witnesses(dst)
    valid &= np.linalg.norm(disagreement, axis=2) < 5
    # Only shadows amplified by processing trigger the correction. A real pit
    # or scratch illuminated like its neighbours is not a failed light field.
    raw_detail = src - raw_ref
    extra_dark = (raw_detail - (dst - lit_ref))[..., 0]
    extra_dark = ndi.gaussian_filter(extra_dark, 1.2)
    seeds = valid & (extra_dark > 2.3)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seeds.astype('uint8'), 8)
    islands = np.zeros(scope.shape, bool)
    for index in range(1, count):
        x, y, width, depth, area = stats[index]
        if area < 5 or width > spacing * 2.3 or depth > spacing * 1.5:
            continue
        if max(width, depth) > 5 * min(width, depth):
            continue
        box = np.s_[y:y + depth, x:x + width]
        component = labels[box] == index
        if abs(raw_detail[box][component, 0]).max() < 2:
            continue
        islands[box] |= component
        report['spots'] += 1
    if not islands.any():
        return None, report
    distance = ndi.distance_transform_edt(~islands)
    weight = np.clip(1 - distance / max(3, spacing * .44), 0, 1)
    weight *= weight * (3 - 2 * weight)
    weight *= valid
    # The local light field comes from nearby processed metal, while real
    # surface marks retain source detail. No texture synthesis or geometry warp.
    target = lit_ref + .72 * raw_detail
    active = weight > 0
    proposal = dst[active] + weight[active, None] * (target[active] - dst[active])
    report['pixels'] = int(active.sum())
    return (active, proposal), report
