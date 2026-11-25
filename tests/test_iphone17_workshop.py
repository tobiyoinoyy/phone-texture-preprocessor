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


def test_dark_bottom_apertures_exclude_metal_grain_and_sticker_ink(tmp_path):
    from PIL import ImageDraw
    h, width = 122, 1035
    rng = np.random.default_rng(7)
    base = np.full((h, width, 4), 255, np.uint8)
    # Dark anodized aluminum spanning right across the old global 145 cutoff,
    # with natural darkening at the curved ends and fine grain noise.
    x = np.arange(width)
    metal = 148 - 72 * np.exp(-np.minimum(x, width - 1 - x) / 48)
    grain = np.clip(metal[None, :] + rng.normal(0, 6, (h, width)), 40, 170)
    # Add slightly darker smooth metal patches around gray=136..142 near the holes.
    grain[22:45, 620:670] = np.clip(grain[22:45, 620:670] - 10, 125, 144)
    grain[75:96, 680:740] = np.clip(grain[75:96, 680:740] - 10, 125, 144)
    base[..., :3] = np.rint(grain).astype(np.uint8)[..., None]
    im = Image.fromarray(base)
    draw = ImageDraw.Draw(im)
    # White QC sticker with dark printed characters.
    draw.rectangle((365, 0, 440, 106), fill=(248, 248, 248, 255))
    draw.rectangle((382, 32, 396, 46), fill=(95, 95, 95, 255))
    draw.rectangle((406, 48, 424, 66), fill=(85, 85, 85, 255))
    # Real recessed bottom hardware: USB-C port, screw, 5 left + 3 right speaker holes.
    draw.rounded_rectangle((450, 32, 594, 88), radius=24, fill=(32, 32, 32, 255))
    draw.ellipse((612, 47, 638, 73), fill=(52, 52, 52, 255))
    for cx in (212, 246, 280, 314, 348, 694, 728, 762):
        draw.ellipse((cx - 12, 48, cx + 12, 72), fill=(40, 40, 40, 255))
    # Antenna stripe on the right.
    draw.rectangle((815, 0, 830, h - 1), fill=(76, 76, 76, 255))
    png = tmp_path / 'bottom.png'
    im.save(png)

    job, _, _, _ = iphone17_material.job(png, 'bottom', tmp_path / 'mat', {})
    domains = {d['name']: d for d in job['views'][0]['domains']}
    assert set(domains) == {'apertures', 'antenna', 'main_surface'}
    ap = np.asarray(Image.open(domains['apertures']['mask']).convert('L')) > 0
    # True hardware interiors are in apertures.
    assert ap[60, 520] and ap[60, 625]
    assert all(ap[60, cx] for cx in (212, 246, 280, 314, 348, 694, 728, 762))
    # Smooth dark metal patches and sticker ink stay out of apertures.
    assert not ap[22:45, 620:670].any()
    assert not ap[75:96, 680:740].any()
    assert not ap[:, 365:440].any()

    # Lighting the dark bottom leaves smooth metal free of dark blotches.
    from phone_texture import lighting
    job.update(output_dir=str((tmp_path / 'out').resolve()),
               records_dir=str((tmp_path / 'rec').resolve()))
    jp = tmp_path / 'job.json'
    w.save(jp, job)
    lighting.run('aluminum', jp)
    lit = np.asarray(Image.open(tmp_path / 'out' / 'bottom.png').convert('RGB'))
    # Patch near the screw keeps its natural contrast without dropping into dark blotches.
    src_patch = float(np.median(base[22:45, 620:670, :3]))
    src_ref = float(np.median(base[22:45, 690:760, :3]))
    patch_med = float(np.median(lit[22:45, 620:670]))
    ref_med = float(np.median(lit[22:45, 690:760]))
    assert abs((patch_med - ref_med) - (src_patch - src_ref)) < 4
    assert patch_med > 120


def test_top_rail_never_creates_apertures_domain(tmp_path):
    from PIL import ImageDraw
    h, width = 122, 1035
    im = Image.new('RGBA', (width, h), (138, 140, 142, 255))
    draw = ImageDraw.Draw(im)
    draw.ellipse((490, 44, 525, 79), fill=(48, 48, 48, 255))
    draw.rectangle((815, 0, 830, h - 1), fill=(72, 72, 72, 255))
    png = tmp_path / 'top.png'
    im.save(png)
    job, _, _, _ = iphone17_material.job(png, 'top', tmp_path / 'mat', {})
    names = {d['name'] for d in job['views'][0]['domains']}
    assert 'apertures' not in names
    assert names == {'antenna', 'main_surface'}
