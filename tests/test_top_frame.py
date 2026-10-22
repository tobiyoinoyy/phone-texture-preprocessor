"""Camera height/width cannot set the top texture plane or its roll."""
import numpy as np
import pytest
from PIL import Image
from phone_texture.top_frame import calibrate_top, verify_top_mask
from phone_texture.auto_geometry import frame_geometry
from phone_texture.geometry_review import guided_geometry, digest
from phone_normalizer.frame_strip import normalize_frame_strip, extract_frame_strip


def scene(camera_top=130, camera_end=770):
    h,w=450,900; yy,xx=np.mgrid[:h,:w];distance=np.minimum(xx-50,849-xx)
    top=200+35*np.exp(-np.maximum(distance,0)/10)
    bottom=330-30*np.exp(-np.maximum(distance,0)/8)
    body=(distance>=0)&(yy>=top)&(yy<bottom)
    camera=(xx>=120)&(xx<camera_end)&(yy>=camera_top)&(yy<230)
    raw=np.full((h,w,3),230,np.uint8);raw[camera]=[90,65,40];raw[body]=[180,100,60]
    return Image.fromarray(np.dstack([raw,(body|camera).astype(np.uint8)*255])),Image.fromarray(raw),body,camera&~body


def test_camera_height_and_extent_do_not_move_body_plane():
    outputs=[]
    for height,end in [(130,770),(80,730),(170,780)]:
        matte,raw,body,camera=scene(height,end);stages={}
        out,proof=frame_geometry(matte,'top',lambda n,i:stages.update({n:i}),raw)
        a=np.array(stages['after_support']);original=np.array(matte)
        assert not a[...,3][camera & (np.indices(camera.shape)[0]<195)].any()
        assert np.array_equal(a[body & (np.indices(body.shape)[0]>205) & (np.indices(body.shape)[0]<320)],
                              original[body & (np.indices(body.shape)[0]>205) & (np.indices(body.shape)[0]<320)])
        assert proof['transform_checks']['status']=='passed'
        outputs.append(np.array(out))
    assert all(np.array_equal(outputs[0],o) for o in outputs[1:])


def test_source_required_and_uniform_photo_cannot_confirm_matte():
    matte,raw,_,_=scene()
    with pytest.raises(ValueError,match='SOURCE_REQUIRED'):normalize_frame_strip(matte,face='top')
    with pytest.raises(ValueError,match='SOURCE_REQUIRED'):extract_frame_strip(matte,face='top')
    with pytest.raises(ValueError,match='RGB_UNSUPPORTED'):
        calibrate_top(np.array(matte),np.full((450,900,3),128,np.uint8))


def test_camera_including_guides_rejected_even_if_reviewed(tmp_path):
    matte,raw,_,_=scene()
    with pytest.raises(ValueError,match='CAMERA_RETAINED'):
        guided_geometry(matte,'top',dict(rotation_ccw_degrees=0,reviewed=True),source_image=raw)
    stages={};_,proof=frame_geometry(matte,'top',lambda n,i:stages.update({n:i}),raw)
    mask=tmp_path/'keep.png';stages['after_support'].getchannel('A').save(mask)
    guide=dict(rotation_ccw_degrees=proof['rotation_ccw_degrees'],retain_mask=dict(path=str(mask),sha256=digest(mask)))
    result,evidence=guided_geometry(matte,'top',guide,source_image=raw)
    assert evidence['top_body_calibration']['transform_checks']['status']=='passed'
    assert result.height<150


def test_single_camera_soft_pixel_or_eroded_body_is_rejected():
    matte,raw,_,_=scene();a=np.array(matte);top,bottom,proof=calibrate_top(a,np.array(raw));stages={}
    frame_geometry(matte,'top',lambda n,i:stages.update({n:i}),raw)
    good=np.array(stages['after_support'])
    bad=good.copy();bad[150,200,3]=1
    with pytest.raises(ValueError,match='CAMERA_RETAINED'):verify_top_mask(a,bad,top,bottom,proof)
    bad=good.copy();bad[250,200,3]=0
    with pytest.raises(ValueError,match='BODY_ERODED'):verify_top_mask(a,bad,top,bottom,proof)


def test_bad_corner_and_bad_angle_are_not_silently_accepted():
    matte,raw,_,_=scene();a=np.array(matte);a[140:240,55:80,3]=255
    with pytest.raises(ValueError):calibrate_top(a,np.array(raw))
    with pytest.raises(ValueError,match='RESIDUAL_TILT'):
        from phone_texture.top_frame import verify_top_transform
        stages={};out,p=frame_geometry(matte,'top',lambda n,i:stages.update({n:i}),raw)
        verify_top_transform(np.array(matte),stages['after_rotation'],out,np.array(p['top_rail']),
            np.array(p['bottom_rail']),2,p['canvas_trim_xyxy'])


def test_top_rigid_deskew_is_default_and_uses_same_entry():
    matte,raw,_,_=scene()
    matte=matte.rotate(1.3,resample=Image.Resampling.BICUBIC,expand=True)
    raw=raw.rotate(1.3,resample=Image.Resampling.BICUBIC,expand=True,fillcolor=(230,230,230))
    out,proof=normalize_frame_strip(matte,face='top',source_image=raw)
    assert abs(proof['rotation_ccw_degrees']+1.3)<.15
    assert max(proof['transform_checks']['residual_rail_degrees'])<.15
    assert out.height<150


def test_old_approved_original_top_report_is_not_grandfathered(tmp_path):
    from geometry_helpers import approved_fixture
    from phone_texture.geometry_review import verify_report,write
    import json
    folder=tmp_path/'images/A';folder.mkdir(parents=True)
    for face in ('front','back','left','right','top','bottom'):
        Image.new('RGBA',(16,32),(100,80,70,255)).save(folder/(face+'.png'))
    path=approved_fixture(folder.parent,tmp_path/'records')
    value=json.loads(path.read_text());value['mode']='standardize';write(path,value)
    with pytest.raises(ValueError,match='CALIBRATION_STALE'):verify_report(path)
