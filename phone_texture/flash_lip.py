"""Recover photographed metal contrast immediately outside a measured flash lip."""

import numpy as np
from scipy import ndimage as ndi

from .camera_metal import ANGLES, THETA, _flash
from .rear_shadow import _smoothstep as ease


def correct(source, raw_light, current_light, material):
    """Update L* in place on supported metal. All distances follow the source scale."""
    centre = _flash(source, material.shape)
    if centre is None:
        return dict(status="flash_not_located", changed_pixels=0)
    cx, cy, radius = centre
    scale = radius / 46.0
    radii = radius + np.arange(-5.0, 40.0, 0.25) * scale
    coords = [cy + np.sin(THETA) * radii, cx + np.cos(THETA) * radii]
    original = ndi.map_coordinates(raw_light, coords, order=1)
    profile = ndi.gaussian_filter(original, (2, 1.55), mode=("wrap", "nearest"))
    gradient = np.gradient(profile, 0.25, axis=1)
    window = np.flatnonzero((radii >= radius - 4 * scale) & (radii < radius + 8 * scale))
    peak = window[np.argmin(gradient[:, window], axis=1)]
    edge = radii[peak]
    median_edge = ndi.median_filter(edge, 9, mode="wrap")
    valid = ((-gradient[ANGLES, peak] >= 3.5) & (abs(edge - median_edge) < 2 * scale)
             & (peak > window[0]) & (peak < window[-1]))
    report = dict(edge_angles=int(valid.sum()), changed_pixels=0)
    if valid.sum() < 260:
        return dict(report, status="insufficient_source_lip")
    edge = ndi.gaussian_filter1d(median_edge, 1.4, mode="wrap")
    distance = (radii - edge[:, None]) / scale
    allowed = ndi.map_coordinates(material, coords, order=0)
    gains = ndi.map_coordinates(current_light, coords, order=1) - original
    references = np.zeros(360)
    for i in np.flatnonzero(valid):
        near = gains[i, allowed[i] & (distance[i] >= 15) & (distance[i] < 24)]
        far = gains[i, allowed[i] & (distance[i] >= 24) & (distance[i] <= 34)]
        if len(near) < 23 or len(far) < 25:
            valid[i] = False
            continue
        a, b = np.median(near), np.median(far)
        valid[i] = abs(a - b) <= 3
        references[i] = (a + b) / 2
    report["reference_angles"] = int(valid.sum())
    if valid.sum() < 245:
        return dict(report, status="insufficient_reference_metal")
    gain = np.interp(ANGLES, ANGLES[valid], references[valid], period=360)
    gain = ndi.gaussian_filter1d(gain, 8, mode="wrap")
    trust = ease((ndi.gaussian_filter1d(valid.astype(float), 2, mode="wrap") - 0.60) / 0.30)
    limit = edge.max() + 21 * scale
    y0, x0 = max(0, int(cy-limit)), max(0, int(cx-limit))
    y1, x1 = min(len(material), int(cy+limit)+1), min(material.shape[1], int(cx+limit)+1)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    angle = np.rad2deg(np.arctan2(yy-cy, xx-cx)) % 360
    dr = (np.hypot(yy-cy, xx-cx) - np.interp(angle, ANGLES, edge, period=360)) / scale
    weight = ease((dr-2.6)/2.3) * ease((20-dr)/8) * np.interp(angle, ANGLES, trust, period=360)
    tile = np.s_[y0:y1, x0:x1]
    light = current_light[tile]
    raw = raw_light[tile]
    # Retain the groove and source shading. Only recover missing illumination,
    # respecting the same maximum source-relative lift as the camera metal field.
    target = raw + np.minimum(np.interp(angle, ANGLES, gain, period=360), 34)
    change = weight * np.clip(target-light, 0, 20)
    active = (change > 0.04) & material[tile]
    light[active] += change[active]
    count = int(active.sum())
    return dict(report, status="applied" if count else "no_supported_change", changed_pixels=count)
