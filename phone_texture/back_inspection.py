"""Independent acceptance of a delivered PNG in its model's camera coordinates."""
import numpy as np
from PIL import Image

from . import workspace as w
from .back_canvas import STANDARD_BACK_SIZE
from .back_geometry import top_measure
from .back_lens import STANDARD, glass_centres
from .iphone17_back_finish import glass_edges
from .iphone17_model_finish import PROFILE


def inspect_back(path, model):
    if model == 'iphone17':
        camera = w.read(PROFILE)['camera']
        size = tuple(camera['reference_size'])
        targets = np.array(camera['centers_px'])
    else:
        size = STANDARD_BACK_SIZE[model]
        targets = np.array(STANDARD[model]['lenses']) * size
    with Image.open(path) as image:
        if image.format != 'PNG' or image.mode != 'RGBA' or image.size != size:
            raise ValueError('背图格式或画布尺寸与机型基准不符')
        pixels = np.asarray(image)
    disks = glass_edges(pixels) if model == 'iphone17' else glass_centres(pixels, model)[0]
    offsets = disks[:, :2] - targets
    distances = np.linalg.norm(offsets, axis=1)
    rail = top_measure(pixels[..., 3])
    verified = (distances.max() <= .30 and abs(rail['left_minus_right_px']) <= .5
                and rail['central_span_p95_px'] <= 1.1)
    return dict(model=model, verified=bool(verified), size=list(size), sha256=w.sha(path),
                targets=targets.tolist(), centers=disks[:, :2].tolist(), offsets=offsets.tolist(),
                distances_px=distances.tolist(), rail=rail)
