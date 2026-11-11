"""Fixed per-model back canvas, so stacked backs keep the camera in place.

2026-11-09 user decision: only the back is standardized. Each SKU's back is
photographed at its own distance, so canvases differ by a few percent; the
camera sits consistently in normalized coordinates (UV use) but drifts when
backs are stacked at native size. The final back is resampled once to the
model's standard size (median width, median aspect of real batches).
Antenna, seam and structure work already ran on normalized coordinates.
"""
from pathlib import Path
from PIL import Image

STANDARD_BACK_SIZE = {'iphone17pro': (1000, 2080), 'iphone17promax': (1080, 2254)}


def standardize(source, target, model):
    """Write the standard-size back; premultiplied resampling keeps rim colour.

    The camera lenses are then registered to the model's standard positions
    (back_lens.py); a back whose lenses cannot be verified is kept as resized.
    """
    size = STANDARD_BACK_SIZE.get(model)
    if size is None:
        raise ValueError('No standard back canvas for model: '+str(model))
    with Image.open(source) as image:
        if image.format != 'PNG' or image.mode != 'RGBA':
            raise ValueError('Final back is not RGBA PNG')
        original = image.size
        resized = image.convert('RGBa').resize(size, Image.Resampling.LANCZOS).convert('RGBA')
    import numpy as np
    from .back_lens import register
    registered, lens_proof = register(np.asarray(resized), model)
    Image.fromarray(registered).save(Path(target))
    return dict(method='premultiplied_lanczos_fixed_model_canvas', model=model,
                source_size=list(original), output_size=list(size),
                scale_x=size[0]/original[0], scale_y=size[1]/original[1],
                lens_registration=lens_proof)
