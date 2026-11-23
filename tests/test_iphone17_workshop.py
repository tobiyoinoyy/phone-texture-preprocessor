from PIL import Image
import numpy as np
import pytest

from phone_texture import iphone17, iphone17_batch as batch, iphone17_material, web_app, workspace as w
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
        from phone_texture.iphone17_model_finish import FRAMES, PROFILE
        hashes = {f:w.sha(candidate/(f+'.png')) for f in FRAMES}
        profile = w.read(PROFILE)
        proof = dict(model_sha256=profile['model_sha256'], profile_sha256=w.sha(PROFILE),
                     output_sha256=hashes, seams=[dict(pair=c['names'], metric_samples=192,
                     total_samples=192, joint_coverage=1., after_p95_deltaE76=.5)
                     for c in profile['chains']])
        return dict(candidate=str(candidate), processed=1, seconds=1, model_frame_finish=proof,
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


def test_bottom_geometry_does_not_use_a_paper_mask(monkeypatch):
    matte = Image.new('RGBA', (900, 120))
    matte.paste((160, 160, 160, 255), (3, 5, 897, 116))
    monkeypatch.setattr(iphone17, 'frame_geometry', lambda *a, **k: (matte, {}))
    def fail(*_):
        raise ValueError('paper sides unresolved')
    monkeypatch.setattr(iphone17, 'paper_mask', fail)
    out, evidence = iphone17.geometry_face(matte, matte, 'bottom', None)
    assert evidence['visible_canvas_trim']['body_pixels_lost'] == 0
    assert out.size == (896, 113)
    with pytest.raises(ValueError, match='paper sides'):
        iphone17.geometry_face(matte, matte, 'top', None)


def test_bottom_lighting_uses_no_label_boundary_and_rejects_an_ink_band(tmp_path, monkeypatch):
    h, width = 120, 1000
    x = np.arange(width)
    lightness = 124 - 66 * np.exp(-np.minimum(x, width-1-x) / 37)
    source = np.full((h, width, 4), 255, np.uint8)
    source[..., :3] = lightness[None, :, None]
    source[:, 420:505, :3] = 225
    source[:, 449:464, :3] = 55
    source[:, 794:809, :3] = 89
    source[42:54, 580:582, :3] = 60
    source[52:54, 580:594, :3] = 60
    png = tmp_path/'bottom.png'
    Image.fromarray(source).save(png)
    def fail(*_, **__):
        raise AssertionError('Standard bottom cannot call the label detector')
    monkeypatch.setattr(iphone17_material,'paper_mask',fail)
    job, report, editable, sticker = iphone17_material.job(png,'bottom',tmp_path/'material',{})
    assert job['views'][0]['protection_policy'] == 'none'
    assert not sticker.any() and editable.all()
    assert abs(report['antenna_columns'][0]-794) <= 2
    assert abs(report['antenna_columns'][1]-809) <= 2
    assert report['paper']['protected_pixels'] == 0
    assert {d['name'] for d in job['views'][0]['domains']} == {'antenna', 'main_surface'}


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
