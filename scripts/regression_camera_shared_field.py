"""Compare a source-registered shared-field result to the accepted back photo trial."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def contrasts(light, spot, flash, scale):
    x, y = spot
    # Compare neighboring points at the same height, preserving the physical bevel.
    dx = np.arange(-round(42 * scale), round(42 * scale) + 1)
    near = abs(dx) <= 13 * scale
    normal = abs(dx) >= 23 * scale
    values = {}
    for name, (lo, hi) in {"upper": (-23, -4), "joint": (-2, 9), "lower": (10, 24)}.items():
        dy = np.arange(round(lo * scale), round(hi * scale) + 1)
        window = light[np.ix_(y + dy, x + dx)]
        values[name] = float(
            np.median(np.median(window[:, normal], axis=1) - np.median(window[:, near], axis=1))
        )
    cx, cy, radius = flash
    angle = np.deg2rad(np.arange(48, 133))[:, None]
    radii = radius + np.arange(58)[None, :] * scale
    rays = ndi.map_coordinates(light, [cy + np.sin(angle) * radii, cx + np.cos(angle) * radii], order=1)
    values["flash"] = float(
        np.median(np.median(rays[:, 37:58], axis=1) - np.median(rays[:, 9:27], axis=1))
    )
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("photo", "baseline", "accepted", "candidate"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--spot", type=int, nargs=2, required=True, metavar=("X", "Y"))
    parser.add_argument("--flash", type=float, nargs=3, required=True, metavar=("X", "Y", "R"))
    parser.add_argument("--scale", type=float, default=1)
    args = parser.parse_args()
    photos = {
        key: np.array(Image.open(getattr(args, key)).convert("RGBA"))
        for key in ("photo", "baseline", "accepted", "candidate")
    }
    if len({im.shape for im in photos.values()}) != 1 or not np.array_equal(
        photos["candidate"][..., 3], photos["baseline"][..., 3]
    ):
        raise ValueError("Expected registered photos and a candidate with unmodified alpha")
    values = {
        key: contrasts(
            cv2.cvtColor(im[..., :3].astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[..., 0],
            args.spot,
            args.flash,
            args.scale,
        )
        for key, im in photos.items()
    }
    print(json.dumps(values, indent=2))
    yy, xx = np.ogrid[: photos["photo"].shape[0], : photos["photo"].shape[1]]
    x, y, r = args.flash
    lamp = np.hypot(xx - x, yy - y) < r * 0.88
    if not np.array_equal(photos["baseline"][lamp], photos["candidate"][lamp]):
        raise ValueError("The flash face was recoloured")
    for zone, photo in values["photo"].items():
        baseline = abs(values["baseline"][zone] - photo)
        trial = abs(values["candidate"][zone] - photo)
        accepted = abs(values["accepted"][zone] - photo)
        if trial > accepted + 0.75 or (baseline > 2 and trial > baseline * 0.65):
            raise ValueError(f"{zone}: local contrast differs from the original by {trial:.2f} L*")


if __name__ == "__main__":
    main()
