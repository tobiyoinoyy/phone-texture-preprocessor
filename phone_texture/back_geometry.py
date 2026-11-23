"""Photograph-based back rail and camera-glass measurements shared by the 17 family."""
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


def measure_glass_edges(a, seeds, *, search_width=8.):
    """Read the glass-to-ring transition and fit it on disjoint angular sectors."""
    gray = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY).astype(float)
    height, width = gray.shape
    end_x = min(width, int(max(x + r + 12 for x, _, r in seeds)) + 2)
    end_y = min(height, int(max(y + r + 12 for _, y, r in seeds)) + 2)
    gray = ndimage.gaussian_filter(gray[:end_y, :end_x], .8)
    angles = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    dx, dy = np.cos(angles), np.sin(angles)
    sectors = (np.arange(len(angles)) // 5) % 2
    measured = []
    for x, y, radius in seeds:
        radii = radius + np.arange(-search_width, search_width + .01, .2)
        samples = ndimage.map_coordinates(
            gray, [y + dy[:, None] * radii, x + dx[:, None] * radii], order=1)
        gradient = np.gradient(samples, .2, axis=1)
        peaks = gradient.argmax(axis=1)
        strength = gradient[np.arange(len(peaks)), peaks]
        valid = (peaks > 1) & (peaks < len(radii) - 2) & (strength > 5)
        edge_x = x + dx * radii[peaks]
        edge_y = y + dy * radii[peaks]

        def ellipse(mask):
            if mask.sum() < 5:
                raise ValueError('Insufficient independent glass perimeter samples')
            points = np.column_stack((edge_x[mask], edge_y[mask])).astype('float32')
            return cv2.fitEllipse(points)

        for _ in range(3):
            if valid.sum() < 180:
                raise ValueError('Insufficient glass perimeter')
            (cx, cy), (diameter_x, diameter_y), degrees = ellipse(valid)
            angle = np.deg2rad(degrees)
            xx, yy = edge_x - cx, edge_y - cy
            u = (xx * np.cos(angle) + yy * np.sin(angle)) / (diameter_x / 2)
            v = (-xx * np.sin(angle) + yy * np.cos(angle)) / (diameter_y / 2)
            valid &= abs((np.hypot(u, v) - 1) * radius) < 2.5
        if min(group.mean() for group in np.array_split(valid, 8)) < .4:
            raise ValueError('Glass lacks edge support across sectors')
        centers = [ellipse(valid & (sectors == index))[0] for index in (0, 1)]
        if np.linalg.norm(np.subtract(*centers)) > .65:
            raise ValueError('Independent glass sector fits disagree')
        (cx, cy), (diameter_x, diameter_y), _ = ellipse(valid)
        measured.append([cx, cy, (diameter_x + diameter_y) / 4])
    return np.array(measured)


def top_rail(alpha, threshold=127.5):
    """Subpixel first alpha crossing on the straight rail, outside rounded ends."""
    width = alpha.shape[1]
    x = np.arange(round(.20 * width), round(.80 * width))
    strip = alpha[:round(.09 * width), x]
    if not (strip.max(axis=0) >= 240).all():
        raise ValueError('Top rail is incomplete between the rounded ends')
    rows = (strip > threshold).argmax(axis=0)
    columns = np.arange(len(x))
    upper = np.where(rows > 0, strip[np.maximum(0, rows-1), columns], 0).astype(float)
    lower = strip[rows, columns].astype(float)
    return x, rows - 1 + (threshold - upper) / np.maximum(lower-upper, 1.)


def top_measure(alpha):
    x, heights = top_rail(alpha)
    ends = max(4, round(.06 * alpha.shape[1]))
    left, right = [float(np.median(a)) for a in (heights[:ends], heights[-ends:])]
    return dict(left_px=left, right_px=right, left_minus_right_px=left-right,
                central_span_p95_px=float(np.percentile(heights, 95)-np.percentile(heights, 5)))


def level_top(image):
    a = np.array(image)
    alpha = a[..., 3]
    x, edge = top_rail(alpha)
    broad = ndimage.gaussian_filter1d(ndimage.median_filter(edge, size=11, mode='nearest'), 6)
    if np.percentile(abs(edge-broad), 95) > 1.25 or np.ptp(broad) > 16:
        raise ValueError('Photographed top rail lacks a stable straight section')
    if np.max(abs(broad - 3.5)) > 14:
        raise ValueError('Top rail would require unsupported displacement')
    # Continue the measured displacement across each native rounded corner.
    # Fade before the lens assembly and side antenna marks.
    offset = np.interp(np.arange(image.width), x, broad - 3.5)
    h = round(.059 * image.width)
    y, xx = np.mgrid[:h, :image.width].astype(np.float32)
    t = np.clip((h-1-y)/(h-14), 0, 1)
    sy = y + offset * (t*t*(3-2*t))
    if np.min(np.diff(sy, axis=0)) < .5:
        raise ValueError('Top alignment would compress the photographed edge')
    clipped = np.arange(16)[:,None] + .5 < sy[0]
    if (alpha[:16][clipped] >= 64).any():
        raise ValueError('Top alignment would clip a photographed corner')
    # Premultiplied interpolation avoids dark fringes around metal on transparency.
    pm = a[:h+2].astype(np.float32)
    pm[..., :3] *= pm[..., 3:] / 255.
    moved = cv2.remap(pm, xx, sy.astype('float32'), cv2.INTER_LINEAR)
    moved[..., :3] *= 255 / np.maximum(moved[..., 3:], 1e-8)
    a[:h] = np.clip(moved.round(), 0, 255).astype('uint8')
    result = top_measure(a[..., 3])
    if abs(result['left_minus_right_px']) > .5 or result['central_span_p95_px'] > 1.1:
        raise ValueError('Final top rail still has unequal headroom')
    return Image.fromarray(a), result
