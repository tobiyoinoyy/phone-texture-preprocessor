"""The unified stainless adapter must preserve the original mask invariants."""
import numpy as np
from PIL import Image
from phone_texture import lighting, workspace as w


def test_stainless_adapter_uses_bundled_compositor_and_exports_exact_allowed_mask(tmp_path):
    yy, xx = np.indices((110, 180))
    detail = 2*np.cos(2*np.pi*(xx/8+yy/6)) + 2*np.cos(2*np.pi*(xx/8-yy/6))
    candidate = np.empty((110,180,4), np.uint8)
    candidate[...,:3] = np.rint(np.array([25,30,35])+detail[...,None]); candidate[...,3] = 255
    highlight = np.zeros((110,180), np.uint8); highlight[15:95,55:90] = 255
    source = candidate.copy(); source[highlight>0,:3] = 245
    for name, a in [('source',source),('candidate',candidate),('mask',highlight)]:
        Image.fromarray(a).save(tmp_path/(name+'.png'))
    job = dict(source=str(tmp_path/'source.png'),candidate=str(tmp_path/'candidate.png'),
               highlight_mask=str(tmp_path/'mask.png'),output=str(tmp_path/'output.png'),
               records_dir=str(tmp_path/'records'),confirm_alignment=True,
               confirm_texture_references=True,texture_reference_boxes=[[115,25,165,85]])
    w.save(tmp_path/'job.json',job)
    lighting.run('stainless',tmp_path/'job.json')
    output = np.array(Image.open(tmp_path/'output.png'))
    allowed = np.array(Image.open(tmp_path/'records/editable.png')) > 0
    assert np.array_equal(allowed,highlight>0)
    assert np.array_equal(output[~allowed],source[~allowed])
    assert np.array_equal(output[...,3],source[...,3])
    assert output[allowed,:3].mean()<source[allowed,:3].mean()/2
