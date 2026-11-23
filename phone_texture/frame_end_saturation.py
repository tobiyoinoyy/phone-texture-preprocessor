"""A1.5 terminal paint saturation on the verified A1 colour baseline."""
from pathlib import Path

import numpy as np
from PIL import Image

from . import lighting
from .frame_end_color import smoothstep


def in_gamut_chroma(engine, lab, ratio):
    """Move along the original hue at fixed lightness, staying inside sRGB."""
    low = np.zeros_like(ratio)
    high = np.ones_like(ratio)
    trial = lab.copy()
    for _ in range(9):
        mid = (low + high) * .5
        trial[:, 1:] = lab[:, 1:] * (1 + mid * ratio)[:, None]
        linear = engine.lab_to_linear_rgb(trial)
        valid = (linear.min(axis=1) >= -1e-7) & (linear.max(axis=1) <= 1 + 1e-7)
        low = np.where(valid, mid, low)
        high = np.where(valid, high, mid)
    trial[:, 1:] = lab[:, 1:] * (1 + low * ratio)[:, None]
    return engine.lab_to_rgb(trial)


def enhance(rgba, mask, ends, strength):
    engine = lighting.engine('aluminum')
    vertical = rgba.shape[0] > rgba.shape[1]
    frame = rgba.swapaxes(0, 1) if vertical else rgba
    metal = mask.T if vertical else mask
    height, length = metal.shape
    width = min(.155 * length, 1.65 * height)
    result = frame.copy()
    for end in ends:
        if end['status'] not in ('corrected', 'within_tolerance'):
            continue
        target = np.array(end['reference'][1:])
        reference = np.linalg.norm(target)
        if reference < 9:
            continue
        direction = -1 if end['end'] == 'end' else 1
        limit = int(np.ceil(width))
        image = frame[:, ::direction][:, :limit]
        editable = metal[:, ::direction][:, :limit] & (image[..., 3] > 0)
        rows, columns = np.where(editable)
        if not len(rows):
            continue
        lab = engine.rgb_to_lab(image[rows, columns, :3])
        ab = lab[:, 1:]
        chroma = np.linalg.norm(ab, axis=1)
        cosine = (ab @ target) / np.maximum(reference * chroma, 1e-8)
        gain = np.minimum(strength * chroma, 5.) * smoothstep((width - columns) / (.43 * width))
        gain *= smoothstep((cosine - .63) / .27) * smoothstep((chroma - 4) / 5)
        gain *= smoothstep((reference - 9) / 5)
        dest = result[:, ::direction]
        dest[rows, columns, :3] = in_gamut_chroma(engine, lab, gain / np.maximum(chroma, 1e-8))
    return result.swapaxes(0, 1) if vertical else result


def finish(baseline, records, masks, decisions, chains, sigma):
    from .iphone17_model_finish import _current_seam_readings

    records = Path(records)
    records.mkdir()
    originals = {f: np.array(Image.open(path).convert('RGBA')) for f, path in baseline.items()}
    for strength in (.12, .10, .08, .06, .04, .02):
        selected = dict(baseline)
        changed = []
        for face, source in originals.items():
            result = enhance(source, masks[face], decisions[face], strength)
            if not np.array_equal(result, source):
                path = records / (face + '.png')
                Image.fromarray(result).save(path)
                selected[face] = path
                changed.append(face)
        if not changed:
            return baseline, dict(profile='A1.5', strength=0., changed=[])
        try:
            checks = _current_seam_readings(selected, chains, masks, sigma)
        except ValueError as error:
            return baseline, dict(profile='A1.5', strength=0., changed=[], reason=str(error))
        if len(checks) == 4 and all(c['after_p95_deltaE76'] <= 2. for c in checks):
            return selected, dict(profile='A1.5', strength=strength, changed=changed, seams=checks)
    return baseline, dict(profile='A1.5', strength=0., changed=[], reason='four_corner_colour_limit')
