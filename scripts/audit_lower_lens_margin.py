"""Inspect the lower arc nearest a plateau ledge on a registered Pro-series back PNG."""

import argparse
import json

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from phone_texture import camera_metal
from phone_texture.lighting import engine
from phone_texture.rear_shadow import _plateau_core, _ring_material


def audit(photo, output):
    if output.shape != photo.shape or not np.array_equal(photo[..., 3], output[..., 3]):
        raise ValueError("Expected source and output on the same final canvas")
    render = engine("aluminum")
    source = render.rgb_to_lab(photo[..., :3])[..., 0]
    light = render.rgb_to_lab(output[..., :3])[..., 0]
    bounds, proof = _plateau_core(source)
    lenses = camera_metal.measure(output, "triple")["lenses"]
    yy, xx = np.ogrid[: len(source), : source.shape[1]]
    flat = _ring_material(
        photo[..., 3] == 255,
        dict(flat_core_bounds=bounds, plateau_source_edges=proof, lenses=[]),
        yy,
        xx,
    )
    quiet = flat.copy()
    for lens in lenses:
        _, distance, _, _ = camera_metal._geometry(flat.shape, lens)
        flat &= distance > 1
        quiet &= distance > 0.17 * lens["centre"][2]
    report = []
    for index, lens in enumerate(lenses):
        cx, cy, radius = lens["centre"]
        scale = radius / 96
        outer = np.asarray(lens["outer"])
        if bounds[3] - cy - outer[90] > 42 * scale:
            continue
        for degree in range(45, 136, 5):
            angles = np.arange(degree - 4, degree + 5)
            theta = np.deg2rad(angles[:, None])
            offsets = np.arange(3, 14, 0.5)
            radii = outer[angles, None] + offsets * scale
            y, x = cy + np.sin(theta) * radii, cx + np.cos(theta) * radii
            measured = ndi.map_coordinates(light, [y, x], order=1)
            inside = ndi.map_coordinates(flat, [y, x], order=0)
            # Follow the scanline away from the nearest side wall. Recheck every
            # step and both far windows independently of the production fit.
            sign = 1 if cx < (bounds[0] + bounds[2]) / 2 else -1
            dx = np.arange(0, 98, 2)
            step_x = x[..., None] + sign * dx * scale
            step_y = np.broadcast_to(y[..., None], step_x.shape)
            path = [step_y, step_x]
            inside &= ndi.map_coordinates(flat, path, order=0).all(axis=2)
            distant = (dx >= 68) & (dx < 80) | (dx >= 86)
            path = [v[..., distant] for v in path]
            inside &= ndi.map_coordinates(quiet, path, order=0).all(axis=2)
            far = ndi.map_coordinates(light, path, order=1)
            a, b = [
                np.median(far[..., band], axis=2)
                for band in (dx[distant] < 80, dx[distant] >= 86)
            ]
            inside &= abs(a - b) < 5
            gap = (a + b) / 2 - measured
            entry = dict(lens=index, degrees=degree)
            for lo, hi in ((3, 5), (5, 9), (9, 14)):
                band = (offsets >= lo) & (offsets < hi)
                eligible = inside[:, band]
                entry[f"{lo}-{hi}"] = (
                    round(float(np.median(gap[:, band][eligible])), 2)
                    if eligible.mean() >= 0.5
                    else None
                )
            report.append(entry)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", help="Source photograph registered to the same canvas")
    parser.add_argument("output")
    args = parser.parse_args()
    images = []
    for path in (args.photo, args.output):
        with Image.open(path) as im:
            images.append(np.asarray(im.convert("RGBA")))
    print(json.dumps(audit(*images), indent=2))
