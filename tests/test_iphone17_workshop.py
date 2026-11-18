from PIL import Image
import numpy as np
import pytest

from phone_texture import iphone17_batch as batch, iphone17_material, web_app, workspace as w
from phone_texture.pipeline import FACES


def test_checked_standard_exports_and_seam_scope(tmp_path, monkeypatch):
    source, work, output = [tmp_path/n for n in ('source','work','output')]
    source.mkdir()
    output.mkdir()
    def run(folder, case, dest, single_name, cpu_threads):
        candidate = case/'candidate'
        candidate.mkdir()
        for face in FACES:
            Image.new('RGBA',(4,8),'white').save(candidate/(face+'.png'))
        return dict(candidate=str(candidate), processed=1, seconds=1,
                    status='processed_needs_visual_review',
                    final_sha256={f:w.sha(candidate/(f+'.png')) for f in FACES})
    monkeypatch.setattr(batch.iphone17,'run',run)
    row = batch.process_case((source,work,output,'device'))
    state = web_app.processing_state(row)
    assert state['all_stages_complete'] and state['stage_status']['seam'] == 'complete'
    report = dict(status=row['status'],processed=1,attempts=[row])
    job = dict(output=str(output),names=['device'])
    assert web_app.verified_export_rows(report,job) == [row]
    (output/'device'/'bottom.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='校验'):
        web_app.verified_export_rows(report,job)


def test_unfinished_standard_faces_are_not_complete():
    state = web_app.processing_state(dict(model='iphone17',status='source_preserved_needs_review',
        six_faces_generated=True,stage_status=dict(geometry='partial',lighting='partial',
        structure='blocked',seam='not_applicable'),face_processing={'top':'source_matte'}))
    assert not state['all_stages_complete']
    assert state['retry_required'] and state['face_processing']['top']=='source_matte'


def test_sticker_end_rejects_bevel_running_outside_both_corners():
    h,w = 120,800
    gray = np.full((h,w),195.)
    gray[95:] = 65
    proposal = np.zeros((h,w),bool)
    proposal[:,321:391] = True
    sides = np.ones(h)*12
    sides[95:] = 1
    proof = dict(source_strip_edges=[[0,320],[0,390]])
    with pytest.raises(ValueError,match='endings'):
        iphone17_material.cross_strip_end(proposal,proof,gray,
            np.arange(321,391),[sides,sides])


def test_dark_hardware_uses_closed_source_contours():
    from PIL import ImageDraw
    from phone_texture.iphone17_structure import hardware, capsule_mask
    im = Image.new('RGBA',(110,1940),(68,69,71,255))
    draw = ImageDraw.Draw(im)
    for y,length in ((356,100),(516,155),(704,155),(1225,265)):
        draw.rounded_rectangle((34,y,76,y+length),radius=21,
                               fill=(74,75,78,255),outline=(32,33,35,255),width=3)
    boxes = hardware(im,'left')
    assert len(boxes) == 4
    mask = capsule_mask(im,boxes[0],'left')
    assert mask[400,55] and not mask[355,34]
    with pytest.raises(ValueError,match='expected 4'):
        hardware(Image.new('RGBA',im.size,(32,32,32,255)),'left')
