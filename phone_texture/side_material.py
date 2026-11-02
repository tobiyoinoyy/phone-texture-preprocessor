"""Source-supported side hardware masks; structure locking boxes are not materials."""
import cv2
import numpy as np
from scipy import ndimage


def hardware_materials(image, face):
    """Resolve this studio profile's complete capsule components independently.

    Recognition may smooth a working signal and bridge short edge gaps. The
    source RGB/alpha are never edited, and every candidate must have source
    edge support on both rails and both caps before it becomes a material.
    """
    if face not in ('left', 'right'):
        raise ValueError('Side hardware requires left or right')
    a = np.asarray(image.convert('RGBA')); h, w = a.shape[:2]
    gray = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), max(1., w*.008)), 4, 12)
    distance = ndimage.distance_transform_edt(edges == 0)
    candidates = []; gy, gx = np.indices((h, w))
    for gap in (0, 1, 2):
        size = 2*gap + 1
        bridged = edges if not gap else cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
        contours, _ = cv2.findContours(bridged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            hull = cv2.convexHull(contour)
            x, y, bw, bh = cv2.boundingRect(hull)
            if not (w*.20 < bw < w*.65 and h*.035 < bh < h*.23
                    and h*.15 < y and y+bh < h*.85
                    and x > w*.15 and x+bw < w*.85
                    and w*.4 < x+bw/2 < w*.6 and 2 < bh/bw < 9):
                continue
            mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(mask, [hull], -1, 1, -1); mask = mask > 0
            corner=max(2,int(bw*.15))
            corner_fill=max(float(part.mean()) for part in (
                mask[y:y+corner,x:x+corner],mask[y:y+corner,x+bw-corner:x+bw],
                mask[y+bh-corner:y+bh,x:x+corner],mask[y+bh-corner:y+bh,x+bw-corner:x+bw]))
            if corner_fill>.25:
                continue  # A long rectangular scratch can have high whole-shape IoU.
            radius = (bw-1)/2
            capsule = ((gx-x-radius)**2
                       + (gy-np.clip(gy, y+radius, y+bh-1-radius))**2 <= radius**2)
            iou = np.count_nonzero(mask & capsule) / np.count_nonzero(mask | capsule)
            if iou < .90 or np.any(mask & (a[..., 3] < 240)):
                continue
            boundary = mask & ~ndimage.binary_erosion(mask)
            yy, xx = np.nonzero(boundary)
            supported = distance[boundary] <= max(1.5, w*.015)
            sides = [xx < x+bw*.25, xx > x+bw*.75,
                     yy < y+bw*.5, yy > y+bh-bw*.5]
            support = [float(supported[s].mean()) for s in sides]
            if min(support) < .75 or supported.mean() < .95:
                continue
            candidates.append((mask, dict(box=[x, y, x+bw, y+bh],
                edge_support=support, boundary_support=float(supported.mean()),
                capsule_iou=float(iou),maximum_corner_fill=corner_fill,
                gap_radius=gap, pixels=int(mask.sum()))))
    selected = []
    for mask, proof in sorted(candidates,
            key=lambda v: (v[1]['pixels'], v[1]['boundary_support']), reverse=True):
        # All candidates have already passed the same source-support gates.
        # Prefer the complete outer contour, even when a short weak rail makes
        # its support slightly lower than an intact inner groove. Ranking a
        # 99%-support flag first silently relabelled that outer ring as body.
        overlaps = [m for m, _ in selected
                    if np.count_nonzero(mask & m) / min(mask.sum(), m.sum()) > .8]
        if overlaps:
            # Only nested/edge-localization duplicates are interchangeable.
            # Substantial crossing boundaries cannot be resolved by area alone.
            if (len(overlaps) != 1 or
                    np.any(mask & (ndimage.distance_transform_edt(~overlaps[0])
                                   > max(1.5, w*.015)))):
                raise ValueError('Ambiguous crossing side hardware contours')
            continue
        selected.append((mask, proof))
    selected.sort(key=lambda entry: entry[1]['box'][1])
    # This is the existing iPhone 17 studio capture profile, not an arbitrary
    # phone detector. Missing/extra hardware is ambiguity, not a blank domain.
    expected = 4 if face == 'left' else 2
    if len(selected) != expected:
        raise ValueError(f'Unresolved {face} hardware materials: {len(selected)}, expected {expected}')
    combined = np.zeros((h, w), bool)
    for mask, proof in selected:
        if np.any(combined & mask):
            raise ValueError('Ambiguous overlapping side hardware materials')
        combined |= mask
    return selected
