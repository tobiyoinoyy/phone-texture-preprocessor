"""Conservative body coordinates and measurable uncertainty, without changing pixels."""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage


def _component(alpha, threshold):
    labels, count = ndimage.label(alpha >= threshold, structure=np.ones((3, 3)))
    if not count:
        raise ValueError(f"no foreground at alpha >= {threshold}")
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    return labels == sizes.argmax()


def _edges(mask):
    valid = mask.any(axis=1)
    rows = mask[valid]
    return (np.flatnonzero(valid), np.argmax(rows, axis=1),
            mask.shape[1] - np.argmax(rows[:, ::-1], axis=1))


def _inner_plateau(values, inward_high):
    """Select an inward edge supported by >=15% of rows, rejecting isolated nicks.

    Multiple substantial plateaus are explicitly reported as ambiguity. This is
    a silhouette heuristic, not camera/button recognition.
    """
    values = np.asarray(values, dtype=float)
    tolerance = 1.0
    remaining = np.ones(len(values), dtype=bool)
    groups = []
    while remaining.any():
        v = values[remaining]
        histogram = np.bincount(v.astype(int))
        center = float(histogram.argmax())
        members = remaining & (np.abs(values - center) <= tolerance)
        groups.append((float(np.median(values[members])), int(members.sum())))
        remaining[members] = False
    substantial = [g for g in groups if g[1] >= max(2, .15 * len(values))]
    if not substantial:
        return float(np.median(values)), True, 0.0
    selected = (max if inward_high else min)(substantial, key=lambda g: g[0])
    ambiguous = len(substantial) > 1
    return selected[0], ambiguous, selected[1] / len(values)


def _measure(alpha, threshold):
    mask = _component(alpha, threshold)
    ys, xs = np.nonzero(mask)
    horizontal = np.ptp(xs) > np.ptp(ys)
    if horizontal:
        mask = mask.T
    rows, lows, highs = _edges(mask)
    start, end = int(rows.min()), int(rows.max()) + 1
    central = (rows >= start + .15 * (end-start)) & (rows < end - .15 * (end-start))
    if central.sum() < 2:
        central = np.ones(len(rows), dtype=bool)
    low, ambiguous_low, support_low = _inner_plateau(lows[central], True)
    high, ambiguous_high, support_high = _inner_plateau(highs[central], False)
    if high <= low:
        raise ValueError('inconsistent body edges')
    # Endpoints are measured across the middle half of the body thickness.
    # Thin attached tips do not define the length; corners remain in the image.
    x0 = max(0, int(np.floor(low + .25*(high-low))))
    x1 = min(mask.shape[1], max(x0+1, int(np.ceil(high - .25*(high-low)))))
    columns = mask[:, x0:x1].T
    columns = columns[columns.any(axis=1)]
    top = float(np.median(np.argmax(columns, axis=1)))
    bottom = float(mask.shape[0] - np.median(np.argmax(columns[:, ::-1], axis=1)))
    box = [low, top, high, bottom]
    if horizontal:
        box = [top, low, bottom, high]
    return box, {
        'multiple_edge_plateaus': bool(ambiguous_low or ambiguous_high),
        'edge_support_fraction': [float(support_low), float(support_high)],
    }


def measure_body(image: Image.Image) -> dict:
    """Return body bbox, alpha sensitivity and reasons requiring human review."""
    alpha = np.asarray(image.convert('RGBA'))[..., 3]
    box, evidence = _measure(alpha, 128)
    measurements = {}
    for threshold in (32, 64, 128, 192):
        try:
            measurements[str(threshold)] = _measure(alpha, threshold)[0]
        except ValueError:
            measurements[str(threshold)] = None
    valid = [b for b in measurements.values() if b is not None]
    spread = np.ptp(np.asarray(valid), axis=0)
    width, height = box[2]-box[0], box[3]-box[1]
    limits = np.array([max(2., .01*width), max(2., .01*height)]*2)
    reasons = []
    if evidence['multiple_edge_plateaus']:
        reasons.append('ambiguous_body_edges')
    if any(b is None for b in measurements.values()) or np.any(spread > limits):
        reasons.append('alpha_boundary_unstable')
    if min(width, height) < 8:
        reasons.append('body_too_thin_for_reliable_geometry')
    return {
        'method': 'supported_inner_edges_and_central_endpoints_v1',
        'bbox': box, **evidence,
        'threshold_bboxes': measurements,
        'threshold_edge_spread_px': spread.tolist(),
        'threshold_edge_spread_limits_px': limits.tolist(),
        'review_reasons': reasons,
    }
