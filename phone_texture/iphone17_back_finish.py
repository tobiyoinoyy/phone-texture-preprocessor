"""Final standard iPhone 17 back canvas, photographed top rail and camera disks."""
import numpy as np
from PIL import Image

from . import workspace as w
from .back_geometry import level_top, measure_glass_edges, top_measure
from .iphone17_model_finish import PROFILE, register_camera, lens_pair


def glass_edges(a):
    return measure_glass_edges(a, lens_pair(a))


def finish_back(image):
    size = tuple(w.read(PROFILE)['camera']['reference_size'])
    source_size = image.size
    if not .96 <= (source_size[0]/source_size[1]) / (size[0]/size[1]) <= 1.04:
        raise ValueError('Back photograph aspect differs from standard model')
    canvas = image.convert('RGBA')
    if canvas.size != size:
        canvas = canvas.resize(size, Image.Resampling.LANCZOS)
    before = top_measure(np.asarray(canvas)[..., 3])
    leveled, top = level_top(canvas)
    finished, camera = register_camera(leveled, center_tolerance=.30, measure=glass_edges)
    return finished, dict(camera, source_size=source_size, canvas=size, top_before=before, top_after=top)
