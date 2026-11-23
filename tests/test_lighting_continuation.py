import numpy as np
from PIL import Image

from phone_texture.lighting_continuation import recover


def test_frame_lighting_continuation_is_nonzero_bounded_and_alpha_safe(tmp_path):
    before = np.zeros((100, 900, 4), dtype='uint8')
    before[4:-4, 4:-4, :3] = np.linspace(64, 145, 892, dtype='uint8')[None, :, None]
    before[4:-4, 4:-4, 3] = 255
    before[43:57, 380:510, :3] = 22
    src, dest = tmp_path/'source.png', tmp_path/'light.png'
    Image.fromarray(before).save(src)
    proof = recover(src, 'top', dest, tmp_path)
    with Image.open(dest) as image:
        result = np.asarray(image)
    assert proof['changed_pixels'] > 0
    assert np.array_equal(result[..., 3], before[..., 3])
    assert np.array_equal(result[43:57], before[43:57])
    step = result[..., :3].astype(int) - before[..., :3]
    assert step.min() == 0 and 1 <= step.max() <= 8
    assert np.array_equal(step[..., 0], step[..., 2])
