"""Shallow rims and off-centre stands must not cost native finish or damage."""

import numpy as np
import pytest
from PIL import Image, ImageDraw
from phone_normalizer.frame_strip import normalize_frame_strip
from phone_texture.frame_contact import recover_frame


def studio_frame(camera=True):
    size = (900, 550)
    body = Image.new("L", size)
    draw = ImageDraw.Draw(body)
    draw.rounded_rectangle((50, 180, 849, 279), radius=9, fill=255)
    draw.ellipse((785, 275, 801, 283), fill=255)  # Exposed, real lower-edge damage.
    body = np.asarray(body) > 0
    yy, xx = np.indices(body.shape)
    stand = (xx >= 109) & (xx <= 590) & (yy >= 278)
    island = (xx >= 584) & (xx <= 794) & (yy >= 149) & (yy <= 187) & camera
    raw = np.full((*body.shape, 3), 210, dtype=np.uint8)
    raw[stand] = 105
    raw[island] = 70
    raw[body] = [130, 145, 158]
    raw[body & (yy > 264) & (xx > 781)] = [52, 58, 60]
    rgba = np.dstack([raw, (body | stand | island).astype(np.uint8) * 255])
    return Image.fromarray(rgba), Image.fromarray(raw), body


@pytest.mark.parametrize("face", ["bottom", "top"])
def test_source_contacts_remove_projections_and_preserve_native_damage(face):
    matte, raw, body = studio_frame()
    stages = {}
    _, proof = normalize_frame_strip(
        matte, face=face, source_image=raw, stage_sink=lambda key, im: stages.update({key: im})
    )
    assert proof["method"] == "source_supported_local_frame_contacts"
    source = np.asarray(matte)
    after = np.asarray(stages["after_support"])
    assert np.array_equal(after[..., :3], source[..., :3])
    assert np.array_equal(after[275:286, 780:805], source[275:286, 780:805])
    assert (after[body, 3] >= 128).all()
    assert not after[288:, 100:595, 3].any()
    assert not after[:175, 585:795, 3].any()
    assert proof["transform_checks"]["residual_projection_pixels"] == 0


@pytest.mark.parametrize("conflict", ["missing_contact", "blank_raw", "competing_support", "wrong_size"])
def test_contact_recovery_rejects_unguarded_or_ambiguous_photos(conflict):
    matte, raw, _ = studio_frame()
    a = np.array(matte)
    rgb = np.array(raw)
    if conflict == "blank_raw":
        rgb[:] = 200
    elif conflict == "missing_contact":
        rgb[268:291, 205:320] = 110
    elif conflict == "competing_support":
        a[278:, 645:710, 3] = 255
    else:
        rgb = rgb[1:]
    with pytest.raises(ValueError):
        recover_frame(Image.fromarray(a), Image.fromarray(rgb), "top")


def test_central_projection_is_not_cut_as_a_camera():
    matte, raw, _ = studio_frame(camera=False)
    a, rgb = np.array(matte), np.array(raw)
    a[150:180, 360:520, 3] = 255
    rgb[150:180, 360:520] = 70
    with pytest.raises(ValueError, match="Central upper projection"):
        recover_frame(Image.fromarray(a), Image.fromarray(rgb), "bottom")
