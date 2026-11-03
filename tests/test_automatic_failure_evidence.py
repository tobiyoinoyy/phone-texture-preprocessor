"""Failed paper/trim decisions retain the actual coordinate image, not a prior run."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from phone_texture import automatic
from phone_normalizer import api


@pytest.mark.parametrize('failure_stage', ['geometry', 'paper', 'trim'])
def test_geometry_paper_failure_keeps_exact_source_pixels(monkeypatch, tmp_path, failure_stage):
    raw_path=tmp_path/'raw.png'
    Image.new('RGB',(12,10),(64,80,96)).save(raw_path)
    matte=Image.new('RGBA',(12,10),(64,80,96,255))
    geometry=Image.new('RGBA',(8,5),(95,125,150,255))
    geometry.putpixel((0,0),(1,2,3,29))
    item=dict(source=str(raw_path),snapshot=str(raw_path),sha256=automatic.w.sha(raw_path))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(set_num_threads=lambda _:None))
    def intake(source,work,**kwargs):
        Path(work).mkdir(parents=True)
        return dict(skus={'SKU_01':{'bottom':item}})
    monkeypatch.setattr(automatic.w,'intake',intake)
    monkeypatch.setattr(automatic.pipeline,'FACES',('bottom',))

    class Normalizer:
        def __init__(self,*a,**k):
            self.backend=SimpleNamespace(device='cpu',_run_model=lambda *a:None)
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def normalize(self,raw):return matte.copy(),SimpleNamespace(model_input_size=None)
    monkeypatch.setattr(api,'Normalizer',Normalizer)
    def frame(*a,**k):
        if failure_stage=='geometry':raise ValueError('geometry evidence failure')
        return geometry.copy(),{}
    def paper(*a,**k):
        if failure_stage=='paper':raise ValueError('paper evidence failure')
        return np.zeros((5,8),bool),{}
    def trim(*a,**k):raise ValueError('trim evidence failure')
    monkeypatch.setattr(automatic,'frame_geometry',frame)
    monkeypatch.setattr(automatic,'paper_mask',paper)
    monkeypatch.setattr(automatic,'trim_visible_canvas',trim)
    case=tmp_path/'case'
    result=automatic.process_one(('SKU_01',str(tmp_path),str(case),'cpu',1,'iphone17pro'))
    assert result['status']=='failed'
    assert result['error']==failure_stage+' evidence failure'
    assert not (case/'candidate').exists()
    folder=case/'work/geometry-records/failed-source-analysis'
    evidence=json.loads((folder/'bottom.json').read_text())
    assert evidence['source_sha256']==item['sha256']
    np.testing.assert_array_equal(np.array(Image.open(folder/'bottom-matte.png')),np.array(matte))
    expected_stage={'geometry':'geometry','paper':'paper_before_visible_canvas_trim','trim':'visible_canvas_trim'}[failure_stage]
    assert evidence['analysis_stage']==expected_stage
    geometry_path=folder/'bottom-geometry-before-final-trim.png'
    if failure_stage=='geometry':
        assert not geometry_path.exists()
        assert 'geometry_source_image' not in evidence
    else:
        np.testing.assert_array_equal(np.array(Image.open(geometry_path)),np.array(geometry))
        assert evidence['geometry_source_sha256']==automatic.w.sha(geometry_path)
        assert evidence['geometry_source_size']==[8,5]
        assert evidence['coordinate_space']=='post_geometry_before_visible_canvas_trim'
