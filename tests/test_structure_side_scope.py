"""An unresolved side cannot erase the other side's own measured proposal."""
import numpy as np
import pytest
from PIL import Image
from phone_texture.auto_structure import propose
from phone_texture.geometry_review import digest


def sample(tmp_path):
    files={}
    for face,width in [('back',300),('left',100),('right',100)]:
        a=np.full((700,width,4),255,np.uint8);a[...,:3]=180
        for lo,hi in ([(100,112),(600,612)] if face=='back' else [(105,117),(605,617)]):
            a[lo:hi,:,:3]=90
        if face!='back':
            a[220:340,40:60,:3]=110
            a[420:480,40:60,:3]=110
        if face=='right':a[:,:12,:3]=180
        path=tmp_path/(face+'.png');Image.fromarray(a).save(path);files[face]=path
    return files


def test_one_side_keeps_same_phone_rear_reference_without_loading_bad_side(tmp_path):
    files=sample(tmp_path);before={f:digest(p) for f,p in files.items()}
    with pytest.raises(ValueError,match='endpoint.*evidence'):
        propose(files,coordinate_mode='canvas_pixel_centers')
    result=propose(files,coordinate_mode='canvas_pixel_centers',sides=('left',))
    assert set(result)=={'left'}
    assert result['left']['back']['sha256']==before['back']
    assert result['left']['source_sha256']==before['left']
    assert result['left']['back']['physical_side']=='left'
    assert {f:digest(p) for f,p in files.items()}==before
    # The requested side can be measured without the unrelated file at all.
    assert propose({f:p for f,p in files.items() if f!='right'},
        coordinate_mode='canvas_pixel_centers',sides=('left',))==result


@pytest.mark.parametrize('sides',[(),('left','left'),('back',),('left','unknown')])
def test_invalid_side_scope_cannot_silently_skip_structure(sides):
    with pytest.raises(ValueError,match='Explicit unique left/right'):
        propose({},sides=sides)
