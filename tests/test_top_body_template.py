import numpy as np
import pytest
from PIL import Image

from phone_texture.auto_geometry import frame_geometry
from phone_texture.top_body_template import normalize_top


def tapered_top():
    rows, columns = np.mgrid[:450, :900]
    distance = np.minimum(columns-50, 849-columns)
    upper = 200 + 35*np.exp(-np.maximum(distance,0)/10)
    lower = 300 + .01*(columns-450) - 30*np.exp(-np.maximum(distance,0)/8)
    body = (distance>=0) & (rows>=upper) & (rows<lower)
    camera = (columns>=120) & (columns<750) & (rows>=120) & (rows<230)
    stand = (columns>350) & (columns<600) & (rows>295)
    rgb = np.full((450,900,3),230,np.uint8)
    rgb[stand] = 170
    rgb[camera] = [90,65,40]
    rgb[body] = [180,100,60]
    matte = Image.fromarray(np.dstack([rgb,(body|camera|stand).astype('uint8')*255]))
    return matte,Image.fromarray(rgb)


def test_model_thickness_remap_keeps_camera_and_stand_out_of_the_top():
    matte,raw=tapered_top()
    image,proof=normalize_top(matte,raw,'iphone17pro')
    assert image.size==(1000,127)
    assert proof['source_geometry_checks']['camera_retained_pixels']==0
    assert min(proof['source_lower_rail_evidence']['half_support_fractions'])>=.7
    assert proof['destination_rails']==[[0,1],[0,126]]
    pixels=np.asarray(image)
    interior=pixels[10:-10,150:-150]
    assert np.max(abs(interior[...,:3].astype(float)-[180,100,60]))<=1
    assert np.all(interior[...,3]==255)


def test_uniform_photo_cannot_supply_a_template_edge():
    matte,raw=tapered_top()
    with pytest.raises(ValueError):
        normalize_top(matte,Image.new('RGB',raw.size,'gray'),'iphone17pro')


def test_unknown_model_cannot_borrow_a_thickness():
    matte,raw=tapered_top()
    with pytest.raises(ValueError,match='No calibrated top thickness'):
        normalize_top(matte,raw,'unlisted-model')
