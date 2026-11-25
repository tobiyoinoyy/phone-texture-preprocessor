import numpy as np
import pytest

from phone_texture.lens_exterior import lift


@pytest.mark.parametrize("rotate", [False, True])
def test_narrow_plateau_uses_connected_metal_along_its_edge(rotate):
    y, x = np.mgrid[:330, :340]
    cx = cy = 150
    radius = 96
    edge = np.full(360, float(radius))
    distance = np.hypot(x - cx, y - cy) - radius
    theta = np.arctan2(y - cy, x - cx)
    clearance = 15
    flat = (distance > 2) & (y <= cy + radius + clearance)
    normal = 58 + 0.012 * (y - cy)
    shade = (
        18
        * np.exp(-(((theta - 1.5) / 0.29) ** 4))
        * np.exp(-(((distance - 7) / 4.5) ** 4))
    )
    light = np.where(flat, normal - shade, 18)
    raw = np.stack([light, light * 0, light * 0 - 9], axis=-1)
    if rotate:
        raw = raw.transpose(1, 0, 2).copy()
        flat, distance, shade = flat.T, distance.T, shade.T
    ok = np.ones(360, bool)
    lens = dict(centre=[cx, cy, radius], outer=edge, supported=ok)
    old, new = raw.copy(), raw.copy()
    lift(raw, old, flat, lens, edge, ok)
    proof = lift(raw, new, flat, lens, edge, ok, flat & (distance > 20))
    spot = (shade > 16) & (distance >= 6)
    assert proof["lateral_reference_angles"] > 0
    assert (new[..., 0] - old[..., 0])[spot].min() > 12
    assert np.max(abs((new[..., 0] - raw[..., 0] - shade)[spot])) < 2
    np.testing.assert_array_equal(new[~flat], raw[~flat])

    # A separate island with plausible colour cannot qualify across a gap.
    quiet = flat & (distance > 20)
    disconnected = flat & ((distance < 15) | (distance > 21))
    other = raw.copy()
    proof = lift(raw, other, disconnected, lens, edge, ok, quiet)
    np.testing.assert_array_equal(other[spot], raw[spot])
