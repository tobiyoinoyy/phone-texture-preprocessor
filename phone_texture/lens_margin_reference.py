"""Read a sideways metal reference when a radial ray runs off a raised camera plateau."""

import numpy as np
from scipy import ndimage as ndi


def lateral_reference(raw, lab, flat, quiet, edge_xy, scale, missing):
    indices = np.flatnonzero(missing)
    target = np.zeros((360, 4))
    valid = np.zeros(360, bool)
    if not indices.size:
        return valid, target

    # Each path starts on the editable lip exterior. Both distant windows
    # must be on independently eligible quiet metal, with a continuous metal
    # path all the way there. Never sample through a joint or another lens.
    x, y = edge_xy
    steps = np.arange(0, 141, 2) * scale
    options = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        coords = [y[indices, None] + dy * steps, x[indices, None] + dx * steps]
        path = ndi.map_coordinates(flat, coords, order=0)
        clean = ndi.map_coordinates(quiet, coords, order=0)
        proposed = np.full((len(indices), 4), np.nan)
        # At a nearly tangent angle the ray only reaches quiet metal farther
        # along the ledge. Select the first stable pair with a connected path.
        for start in (34, 50, 66, 82, 98):
            early = (steps >= start * scale) & (steps < (start + 18) * scale)
            late = (steps >= (start + 20) * scale) & (steps < (start + 40) * scale)
            keep = (
                np.isnan(proposed[:, 0])
                & path[:, steps < (start + 40) * scale].all(axis=1)
                & clean[:, early | late].all(axis=1)
            )
            if not keep.any():
                continue
            at = [a[keep][:, early | late] for a in coords]
            values = np.stack(
                [
                    ndi.map_coordinates(c, at, order=1)
                    for c in (*lab.transpose(2, 0, 1), raw[..., 0])
                ],
                axis=-1,
            )
            first = early[early | late]
            a, b = [np.median(values[:, band], axis=1) for band in (first, ~first)]
            stable = (np.linalg.norm(a[:, :3] - b[:, :3], axis=1) < 3.5) & (
                abs(a[:, 3] - b[:, 3]) < 4
            )
            proposed[np.flatnonzero(keep)[stable]] = ((a + b) / 2)[stable]
        options.append(proposed)
    candidates = np.stack(options)
    for j, angle in enumerate(indices):
        values = candidates[:, j]
        values = values[np.isfinite(values[:, 0])]
        if len(values) and np.max(np.ptp(values, axis=0)) < 6:
            target[angle] = np.median(values, axis=0)
            valid[angle] = True
    return valid, target
