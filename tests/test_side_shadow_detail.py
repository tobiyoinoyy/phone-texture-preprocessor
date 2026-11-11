"""An illumination repair must keep independently planted surface damage."""
import numpy as np
from PIL import Image, ImageDraw
from phone_texture import auto_material, lighting


def test_side_shadow_reduction_retains_long_scratch_and_local_wear(tmp_path):
    h, w = 700, 120
    yy, xx = np.indices((h, w))
    base = 180 - 110*np.exp(-((xx-4)/15)**2) - 110*np.exp(-((xx-115)/15)**2)
    source = np.full((h, w, 4), 255, np.uint8)
    source[..., :3] = base[..., None]
    source[58:70, :, :3] = 85
    source[630:642, :, :3] = 85
    image = Image.fromarray(source)
    draw = ImageDraw.Draw(image)
    for lo, hi, value in [(200, 290, 178), (390, 480, 75)]:
        draw.rounded_rectangle((42, lo, 78, hi), radius=18,
                               fill=(value, value, value, 255),
                               outline=(35, 35, 35, 255), width=2)
    source = np.array(image)
    source[300:380, 14:16, :3] = np.maximum(source[300:380, 14:16, :3].astype(int)-25, 0)
    source[510:518, 18:24, :3] = np.maximum(source[510:518, 18:24, :3].astype(int)-40, 0)
    path = tmp_path/'right.png'
    Image.fromarray(source).save(path)
    job, _, _, _ = auto_material.propose_job(path, 'right', tmp_path/'materials')
    engine = lighting.engine('aluminum')
    result, allowed, paper, _ = engine.process_view(job['views'][0],
        engine.target_color(job)[0], dict(job['defaults'], color_mode=job['color_mode']))
    src = engine.rgb_to_lab(source[..., :3])[..., 0]
    dst = engine.rgb_to_lab(result[..., :3])[..., 0]
    def contrast(lab, rows, columns, control):
        return float(lab[control, columns].mean()-lab[rows, columns].mean())
    scratch = lambda lab: contrast(lab, slice(325, 355), slice(14, 16), slice(125, 155))
    wear = lambda lab: contrast(lab, slice(510, 518), slice(18, 24), slice(530, 538))
    assert scratch(dst) > .90*scratch(src)
    assert wear(dst) > .98*wear(src)
    reference = float(src[125:155, 50:70].mean())
    before = reference-float(src[125:155, 8:13].mean())
    after = reference-float(dst[125:155, 8:13].mean())
    assert 0 < after < .35*before
    np.testing.assert_array_equal(result[..., 3], source[..., 3])
    np.testing.assert_array_equal(result[~allowed | paper], source[~allowed | paper])
