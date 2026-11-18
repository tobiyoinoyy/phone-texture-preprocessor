import numpy as np
import pytest
from PIL import Image,ImageDraw
from phone_texture.iphone17_buttons import rear_buttons,propose,apply,_sample_xy


def back(*,missing=False):
    im=Image.new('RGBA',(1000,2060),(0,0,0,0));d=ImageDraw.Draw(im)
    d.rectangle((10,0,989,2059),fill='#dddddd')
    for i,(start,end) in enumerate([(410,510),(580,735),(775,930)]):
        if missing and i==1:continue
        d.rounded_rectangle((986,start,995,end),radius=4,fill='#888888')
        d.rectangle((986,start+5,989,end-5),fill='#dddddd')
    d.rounded_rectangle((4,630,13,880),radius=4,fill='#888888')
    d.rectangle((10,635,13,875),fill='#dddddd')
    return im


def test_rear_button_correspondence_and_missing_button_rejected():
    assert len(rear_buttons(back(),'left')['rows'])==3
    assert rear_buttons(back(),'left')['rear_image_edge']=='right'
    assert len(rear_buttons(back(),'right')['rows'])==1
    with pytest.raises(ValueError,match='expected 3'):
        rear_buttons(back(missing=True),'left')


def test_perspective_sampling_identity_and_unobserved_pixels():
    a=np.random.default_rng(2).integers(0,256,(20,10,4),dtype=np.uint8);im=Image.fromarray(a)
    y,x=np.mgrid[:20,:10].astype(float)
    assert np.array_equal(a,np.asarray(_sample_xy(im,x,y)))
    with pytest.raises(ValueError,match='unobserved'):_sample_xy(im,x+1,y)


def test_projective_buttons_keep_antenna_rows_sim_and_alpha(tmp_path):
    b=back();rows=np.asarray(rear_buttons(b,'left')['rows']);h=1940
    target=(rows+.5)/b.height*h-.5;norm=target/h
    # Known synthetic perspective: destination = (.97*s+.02)/(.04*s+1).
    sources=(.02-norm)/(.04*norm-.97)*h
    im=Image.new('RGBA',(110,h),'#dddddd');d=ImageDraw.Draw(im)
    for start,end in sources:d.rounded_rectangle((32,round(start),76,round(end)),radius=18,outline='#404040',width=4)
    d.rounded_rectangle((32,1230,76,1490),radius=18,outline='#404040',width=4)
    d.rectangle((0,160,109,175),fill='#888888');d.rectangle((0,1730,109,1745),fill='#888888')
    bp=tmp_path/'back.png';sp=tmp_path/'left.png';b.save(bp);im.save(sp)
    cfg=propose(bp,sp,'left');out,check=apply(im,cfg);a=np.asarray(im);v=np.asarray(out)
    assert check['mapping_mode']=='shared_projective_with_small_smooth_landmark_residual'
    assert np.array_equal(a[...,3],v[...,3])
    assert np.array_equal(a[:250],v[:250])
    assert np.array_equal(a[1150:],v[1150:])
    assert np.max(abs(np.asarray(check['endpoint_error_rear_px'])))<=1.5
    cfg['rear_detection']['rear_image_edge']='left'
    with pytest.raises(ValueError,match='Wrong rear edge'):apply(im,cfg)


def test_button_at_crop_edge_requires_matching_original_neighbours(tmp_path):
    import json
    from phone_texture.geometry_review import digest
    from phone_texture.iphone17_buttons import source_bound_buttons

    original = back()
    canvas = tmp_path/'after-rotation-canvas.png'
    original.save(canvas)
    target = tmp_path/'back.png'
    box = (4, 0, 996, original.height)
    cropped = original.crop(box)
    cropped.save(target)
    record = dict(face='back', output_sha256=digest(target),
                  geometry=dict(canvas_trim_xyxy=[0,0,*original.size], visible_canvas_trim=dict(crop_xyxy=box)),
                  geometry_evidence=[dict(path=str(canvas), sha256=digest(canvas))])
    report = tmp_path/'report.json'
    report.write_text(json.dumps(dict(images=[record])))
    with pytest.raises(ValueError, match='pre-crop evidence required'):
        source_bound_buttons(target, cropped, 'right', None)
    actual = source_bound_buttons(target, cropped, 'right', report)
    expected = rear_buttons(original, 'right')
    np.testing.assert_allclose(actual['rows'], expected['rows'], atol=.1)
    assert actual['pre_crop_sha256'] == digest(canvas)
    record['geometry']['visible_canvas_trim']['crop_xyxy'] = (5,0,997,original.height)
    report.write_text(json.dumps(dict(images=[record])))
    with pytest.raises(ValueError, match='pixels disagree'):
        source_bound_buttons(target, cropped, 'right', report)
    Image.new('RGBA', original.size).save(canvas)
    with pytest.raises(ValueError, match='has changed'):
        source_bound_buttons(target, cropped, 'right', report)


def test_button_texture_preserves_near_opaque_interior_but_rejects_actual_edge():
    from phone_texture.iphone17_buttons import _retain_interior_coverage
    a = np.full((24, 24, 4), 255, dtype=np.uint8)
    a[:, :2, 3] = 0
    a[9:13, 8:14, 3] = 252
    y, x = np.mgrid[:24, :24].astype(float)
    sx = x.copy()
    sx[8:15, 8:15] += .6
    rendered = np.asarray(_sample_xy(Image.fromarray(a), sx, y))
    assert not np.array_equal(rendered[..., 3], a[..., 3])
    out = np.array(_retain_interior_coverage(a, rendered, x, y, sx, y))
    assert np.array_equal(out[..., 3], a[..., 3])
    assert np.array_equal(out[..., :3], rendered[..., :3])
    sx[10, 3] += .5
    with pytest.raises(ValueError, match='body alpha contour'):
        _retain_interior_coverage(a, rendered, x, y, sx, y)
    sx[10, 3] = 3
    sx[10, 12] = 3.1
    with pytest.raises(ValueError, match='sample the body alpha contour'):
        _retain_interior_coverage(a, rendered, x, y, sx, y)
