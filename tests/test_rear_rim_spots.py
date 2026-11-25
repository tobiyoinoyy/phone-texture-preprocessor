import numpy as np

from phone_texture.rear_rim_spots import correct


def test_upper_rail_uses_neighbouring_field_and_keeps_photographic_marks():
    h, w = 1200, 600
    y, x = np.ogrid[:h, :w]
    source = np.broadcast_to([48., 12., 20.], (h, w, 3)).copy()
    pit = np.exp(-((x - 250)**2 + (y - 21)**2) / 5)
    halo = np.exp(-((x - 250)**2 + (y - 21)**2) / 45)
    source[..., 0] += 6 * pit
    lit = source.copy()
    lit[..., 0] += 12 - 8 * halo
    alpha = np.full((h, w), 255, 'uint8')
    metal = np.ones((h, w), bool)
    change, report = correct(source, lit, metal, alpha, 48)
    mask, proposed = change
    assert report['spots'] == 1
    result = lit.copy()
    result[:len(mask)][mask] = proposed
    assert result[21, 250, 0] > result[21, 275, 0] + 2
    assert abs(result[21, 255, 0] - result[21, 275, 0]) < 1
    assert np.array_equal(result[50:], lit[50:])
    assert np.array_equal(result[:, :190], lit[:, :190])
    # An intervening component invalidates the witnesses. A genuine spot
    # that received the common lift already is not edited a second time.
    metal[:, 250] = False
    assert correct(source, lit, metal, alpha, 48)[0] is None
    assert correct(source, source + [12, 0, 0], ~np.zeros_like(metal), alpha)[0] is None
