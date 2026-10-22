import numpy as np
from PIL import Image
import pytest

from phone_texture.frame_seams import MODELS, profile, resized_chains


def test_model_required_no_silent_shape_guess():
    for model in (None, '', 'iphone17', '../iphone17pro'):
        with pytest.raises(ValueError, match='请选择模型'):
            profile(model)


@pytest.mark.parametrize('model', MODELS)
def test_actual_inset_uv_survives_nonuniform_texture_resizing(tmp_path, model):
    data, _ = profile(model)
    files = {}
    for face, size in [('top',(1400,151)),('bottom',(1300,143)),
                       ('left',(138,2341)),('right',(139,2351))]:
        files[face] = tmp_path/(face+'.png')
        Image.new('RGBA',size,'orange').save(files[face])
    converted = resized_chains(data, files)
    for old, new in zip(data['chains'], converted):
        for a, b in zip(old['sides'], new['sides']):
            old_uv = (np.asarray(a['points'])+.5)/a['image_size']
            new_uv = (np.asarray(b['points'])+.5)/b['image_size']
            np.testing.assert_allclose(old_uv,new_uv,atol=1e-12)
            assert np.min(new_uv) > 0 and np.max(new_uv) < 1
            np.testing.assert_allclose(np.linalg.norm(b['normals'],axis=1),1)
            if b['name'] in ('top','bottom'):
                # All four seams are inside the image, not at PNG edges.
                assert min(np.min(new_uv[:,0]), 1-np.max(new_uv[:,0])) > .02
    assert len(converted)==4
