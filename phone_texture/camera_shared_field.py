"""Transfer the surrounding metal's illumination gain to a verified shadow footprint."""

import numpy as np
from scipy import ndimage as ndi
from .rear_shadow import _smoothstep as ease


def _estimate(gain, support, sigma):
    density = ndi.gaussian_filter(support.astype(float), sigma)
    field = ndi.gaussian_filter(gain * support, sigma) / np.maximum(density, 1e-6)
    return field, density


def apply(raw, lit, support, requested, editable, scale):
    """Estimate one continuous field away from hardware and suspect features.

    The candidate is recomposed from the source plus the shared field. Apply
    both positive and negative local offsets through the entire affected spot.
    """
    gain = lit - raw
    near, density = _estimate(gain, support, (23 * scale, 37 * scale))
    broad, broad_density = _estimate(gain, support, (42 * scale, 65 * scale))
    mix = ease((density - 0.035) / 0.16)
    field = np.clip(near * mix + broad * (1 - mix), -20, 28)
    target = raw + field
    # A coherent area has to be darker than the surrounding correction predicts.
    # Carry the correction across the complete footprint, including any tiny glint.
    reliable = ease((broad_density - 0.012) / 0.075)
    available = (requested > 0.05) & editable & (reliable > 0.7)
    deficit = target - lit
    dark = ndi.gaussian_filter(deficit * available, 2 * scale) / np.maximum(
        ndi.gaussian_filter(available.astype(float), 2 * scale), 1e-6
    )
    triggers = (dark > 2.0) & available
    if not triggers.any():
        return 0
    reach = ndi.distance_transform_edt(~triggers)
    blend = ease((35 * scale - reach) / (16 * scale)) * requested * reliable * editable
    delta = blend * (target - lit)
    lit += delta
    return int(np.count_nonzero(abs(delta) > 0.1))
