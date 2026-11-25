"""Lift coherent cast shade on the photographed plateau immediately outside a lens."""

import numpy as np
from scipy import ndimage as ndi

from .rear_shadow import _smoothstep as ease


def lift(raw, lab, flat, lens, lit_outer, lit_supported, quiet=None):
    angles = np.arange(360)
    cx, cy, radius = lens["centre"]
    scale = radius / 96
    outer = np.asarray(lens["outer"])
    agreed = (
        np.asarray(lens["supported"])
        & lit_supported
        & (abs(outer - lit_outer) <= 2.75 * scale)
    )
    if agreed.sum() < 210:
        return dict(status="insufficient_rim", pixels=0)
    edge = np.maximum(outer, lit_outer)
    offsets = np.arange(0.0, 42.0, 0.5)
    theta = np.deg2rad(angles[:, None])
    radii = edge[:, None] + scale * offsets
    coords = [cy + np.sin(theta) * radii, cx + np.cos(theta) * radii]
    material = ndi.map_coordinates(flat, coords, order=0, mode="constant")
    # Never borrow through another rim, a microphone, or the bevel of the plateau.
    references = [(offsets >= a) & (offsets < b) for a, b in [(20, 29), (29, 40)]]
    radial_room = np.logical_and.reduce(
        [np.mean(material[:, band], axis=1) >= 0.85 for band in references]
    )
    connected = np.all(material[:, (offsets >= 5) & (offsets < 20)], axis=1)
    supported = agreed & radial_room & connected
    indices = np.flatnonzero(supported)
    observed = np.stack(
        [ndi.map_coordinates(c, coords, order=1) for c in lab.transpose(2, 0, 1)],
        axis=-1,
    )
    source = ndi.map_coordinates(raw[..., 0], coords, order=1)

    target = np.zeros((360, 3))
    raw_target = np.zeros(360)
    if indices.size:
        # Smooth angularly within the verified flat-plateau mask so 1-degree
        # anodized-grain noise on dark metal cannot punch holes inside a cast
        # shadow arc and turn it into an isolated dark spot.
        ref_weight = ndi.gaussian_filter(
            material.astype(float), (2.5, 0.0), mode=("wrap", "nearest")
        )
        ref_obs = np.stack(
            [
                ndi.gaussian_filter(
                    observed[..., c] * material, (2.5, 0.0), mode=("wrap", "nearest")
                )
                / np.maximum(ref_weight, 1e-7)
                for c in range(3)
            ],
            axis=-1,
        )
        ref_src = (
            ndi.gaussian_filter(source * material, (2.5, 0.0), mode=("wrap", "nearest"))
            / np.maximum(ref_weight, 1e-7)
        )

        def reference(values, band):
            usable = material[indices][:, band, None]
            return np.nanmedian(
                np.where(usable, values[indices][:, band], np.nan), axis=1
            )

        first, second = [reference(ref_obs, band) for band in references]
        supported[indices] = np.linalg.norm(first - second, axis=-1) < 5
        target[indices] = (first + second) / 2
        raw_target[indices] = reference(ref_src[..., None], references[1])[:, 0]

    lateral = np.zeros(360, bool)
    if quiet is not None:
        from .lens_margin_reference import lateral_reference

        start = edge + 8 * scale
        phase = theta[:, 0]
        xy = (cx + np.cos(phase) * start, cy + np.sin(phase) * start)
        lateral, readings = lateral_reference(
            raw, lab, flat, quiet, xy, scale, agreed & ~radial_room
        )
        target[lateral] = readings[lateral, :3]
        raw_target[lateral] = readings[lateral, 3]
        supported |= lateral
    if supported.sum() < 15:
        return dict(status="insufficient_plateau", pixels=0)
    trust = ndi.gaussian_filter1d(supported.astype(float), 3, mode="wrap")

    def smooth_reference(values):
        total = ndi.gaussian_filter1d(values * supported, 3, mode="wrap")
        return total / np.maximum(trust, 1e-7)

    target = np.column_stack([smooth_reference(c) for c in target.T])
    raw_target = smooth_reference(raw_target)

    def smooth_field(values):
        return ndi.gaussian_filter(values, (2.2, 1.8), mode=("wrap", "nearest"))

    coverage = smooth_field(material.astype(float))

    def normal(values):
        return smooth_field(values * material) / np.maximum(coverage, 1e-7)

    residual = target[:, None, :] - np.stack(
        [normal(c) for c in observed.transpose(2, 0, 1)], axis=-1
    )
    dark = ease((residual[..., 0] - 1.1) / 3)
    source_shadow = ease((raw_target[:, None] - normal(source) - 0.8) / 3)
    strength = (
        dark
        * source_shadow
        * ease((trust[:, None] - 0.55) / 0.38)
        * ease((coverage - 0.45) / 0.50)
    )
    shift = np.clip(residual, [0, -10, -18], [36, 10, 18]) * strength[..., None]
    # Preserve the physical dark joint, then taper seamlessly into normal metal.
    radial = ease((offsets - 1.6) / 2.6) * ease((24 - offsets) / 9)
    shift *= radial[None, :, None]
    y, x = np.ogrid[: len(flat), : flat.shape[1]]
    phase = np.rad2deg(np.arctan2(y - cy, x - cx)) % 360
    dr = (np.hypot(x - cx, y - cy) - np.interp(phase, angles, edge, period=360)) / scale
    selected = flat & (dr > 1.6) & (dr < 24)
    polar = [phase[selected], dr[selected] * 2]
    delta = np.column_stack(
        [
            ndi.map_coordinates(np.vstack([c, c[:1]]), polar, order=1)
            for c in shift.transpose(2, 0, 1)
        ]
    )
    # A deeply shaded pixel can already be close to the base lighting limit.
    # Keep a separate, finite allowance only inside the corroborated shadow.
    available = np.maximum(raw[..., 0][selected] + 44 - lab[..., 0][selected], 0)
    delta *= np.minimum(1, available / np.maximum(delta[:, 0], 1e-7))[:, None]
    lab[selected] += delta
    count = int(np.count_nonzero(delta[:, 0] > 0.1))
    return dict(
        status="applied" if count else "no_supported_change",
        pixels=count,
        reference_angles=int(supported.sum()),
        lateral_reference_angles=int(lateral.sum()),
        unsupported_degrees=np.flatnonzero(~supported).tolist(),
    )
