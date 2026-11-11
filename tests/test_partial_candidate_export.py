import json
from pathlib import Path

import pytest
from PIL import Image

from phone_texture import automatic


def candidate(root,sku):
    folder=root/sku;folder.mkdir(parents=True)
    hashes={}
    for face in automatic.pipeline.FACES:
        path=folder/(face+'.png')
        Image.new('RGBA',(8,12),(100,120,140,255)).save(path)
        hashes[face]=automatic.w.sha(path)
    return dict(sku=sku,candidate=str(root),status='processed_needs_visual_review',
                six_faces_processed=True,final_sha256=hashes)


def test_partial_exports_only_verified_complete_sku(tmp_path):
    (tmp_path/'work').mkdir()
    good=candidate(tmp_path/'candidate','good')
    bad=candidate(tmp_path/'candidate','bad')
    bad.update(status='failed',six_faces_processed=False,error='antenna endpoint uncertain')
    names=automatic.export_complete_candidates([good,bad],tmp_path/'work',tmp_path/'output')
    assert names==['good']
    assert sorted(p.name for p in (tmp_path/'output').iterdir())==['good']
    assert len(list((tmp_path/'output'/'good').glob('*.png')))==6


def test_partial_rejects_changed_complete_image_before_publishing(tmp_path):
    (tmp_path/'work').mkdir()
    good=candidate(tmp_path/'candidate','good')
    Image.new('RGBA',(8,12),'black').save(tmp_path/'candidate'/'good'/'back.png')
    with pytest.raises(ValueError,match='changed before export'):
        automatic.export_complete_candidates([good],tmp_path/'work',tmp_path/'output')
    assert not (tmp_path/'output').exists()


def test_failed_sku_is_not_exported_even_if_stale_six_face_flag(tmp_path):
    bad=candidate(tmp_path/'candidate','bad');bad['status']='failed'
    assert automatic.export_complete_candidates([bad],tmp_path/'work',tmp_path/'output')==[]
    assert not (tmp_path/'output').exists()
