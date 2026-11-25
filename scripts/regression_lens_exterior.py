"""Compare marked lens-shadow sectors on two identically registered, final triple-camera backs."""

import argparse
import json

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from phone_texture.camera_metal import measure


def sector_contrasts(light, lens, direction, scale):
    angles = (direction + np.arange(-16, 17)) % 360
    phase = np.deg2rad(angles[:, None])
    offsets = np.arange(0, 40, 0.5)
    radius = np.asarray(lens["outer"])[angles, None] + scale * offsets
    cx, cy, _ = lens["centre"]
    values = ndi.map_coordinates(
        light, [cy + np.sin(phase) * radius, cx + np.cos(phase) * radius], order=1
    )
    reference = np.median(values[:, (offsets >= 26) & (offsets <= 38)], axis=1)
    return [
        float(
            np.median(reference[:, None] - values[:, (offsets >= lo) & (offsets < hi)])
        )
        for lo, hi in ((4, 10), (10, 17))
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument(
        "--sector",
        nargs=2,
        type=int,
        action="append",
        required=True,
        metavar=("LENS_INDEX", "DEGREES"),
    )
    args = parser.parse_args()
    pair = []
    for path in (args.before, args.after):
        with Image.open(path) as image:
            pair.append(np.asarray(image.convert("RGBA")))
    before, after = pair
    if before.shape != after.shape or not np.array_equal(before[..., 3], after[..., 3]):
        raise ValueError("Canvas or alpha changed")
    lenses = measure(before, "triple")["lenses"]
    lights = [
        cv2.cvtColor(im[..., :3].astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[..., 0]
        for im in pair
    ]
    yy, xx = np.ogrid[: len(before), : before.shape[1]]
    for lens in lenses:
        x, y, radius = lens["centre"]
        glass = np.hypot(xx - x, yy - y) < 0.93 * radius
        if not np.array_equal(before[glass], after[glass]):
            raise ValueError("Glass changed")
    for index, deg in args.sector:
        if not 0 <= index < len(lenses):
            raise ValueError("Unknown lens index")
        old, new = [
            sector_contrasts(light, lenses[index], deg, before.shape[1] / 1000)
            for light in lights
        ]
        print(json.dumps(dict(lens=index, degrees=deg, before=old, after=new)))
        # A shadow fix should match the normal metal without turning it into a bright halo.
        if any(abs(n) > 2.0 for n in new):
            raise ValueError("Residual local shadow or bright halo")


if __name__ == "__main__":
    main()
