"""Transfer a reviewed texture row through two complete geometric seam curves."""
import numpy as np


def transfer_row(back_y, back_height, side_height, back_curve, side_curve):
    """Curves are independent (world longitudinal coordinate, texture v) arrays.

    Include every knot on both surfaces, including unmatched/T-junction vertices.
    Curves must describe the same reviewed physical seam in one world frame.
    Texture pixel centers follow glTF sampling: pixel = v * height - 0.5.
    """
    def curve(values):
        a = np.asarray(values, dtype=float)
        if a.ndim != 2 or a.shape[1] != 2 or len(a) < 2 or not np.isfinite(a).all():
            raise ValueError('Invalid seam curve')
        a = a[np.argsort(a[:, 0])]
        if np.any(np.diff(a[:, 0]) <= 0):
            raise ValueError('Ambiguous seam coordinates')
        dv = np.diff(a[:, 1])
        if not (np.all(dv > 0) or np.all(dv < 0)):
            raise ValueError('Non-monotone seam UV')
        return a
    b, s = curve(back_curve), curve(side_curve)
    by_v = b[np.argsort(b[:, 1])]
    v = (back_y + .5) / back_height
    if v < by_v[0, 1] or v > by_v[-1, 1]:
        raise ValueError('Back antenna outside reviewed seam')
    world = np.interp(v, by_v[:, 1], by_v[:, 0])
    if world < s[0, 0] or world > s[-1, 0]:
        raise ValueError('Side seam does not cover antenna')
    return float(np.interp(world, s[:, 0], s[:, 1])*side_height-.5)
