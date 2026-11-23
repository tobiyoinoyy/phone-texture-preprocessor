import numpy as np
import pytest
from PIL import Image,ImageDraw
from phone_texture.auto_geometry import front_lower_closure,principal


def fixture():
 h,w=700,400;im=Image.new('L',(w,h));ImageDraw.Draw(im).rounded_rectangle((80,50,320,550),radius=30,fill=255);body=np.array(im)>0;yy,xx=np.indices((h,w));clips=(((xx>=105)&(xx<135))|((xx>=270)&(xx<300)))&(yy>=530)&(yy<590)
 mask=body|clips;raw=np.full((h,w,3),230,np.uint8);raw[(xx>=100)&(xx<305)&(yy>=545)]=210;raw[clips]=200;raw[body]=20;raw[body&(yy>=546)]=170
 widths=mask.sum(axis=1);wide=np.flatnonzero(widths>widths.max()*.8);return mask,raw,body,int(wide.max())


def test_front_source_alpha_keeps_all_known_body_pixels_despite_bright_rim():
 mask,raw,body,last=fixture();before=raw.copy();_,line,_,_,_,proof=front_lower_closure(mask,raw,80,321,last)
 yy,xx=np.indices(mask.shape);removed=yy>np.polyval(line,xx+.5)+1
 assert not np.any(removed&body)
 assert np.any(removed&mask&~body)
 assert np.array_equal(raw,before)
 assert proof['RGB_does_not_set_inward_cut']
 assert min(proof['dark_glass_closure_fractions'])>=.9


def test_front_alpha_without_actual_rgb_closure_is_rejected():
 mask,raw,body,last=fixture();raw[:]=20
 with pytest.raises(ValueError):front_lower_closure(mask,raw,80,321,last)


def test_source_boundary_shifted_away_from_alpha_is_not_confirmed():
 mask,raw,body,last=fixture();raw[520:]=230
 with pytest.raises(ValueError):front_lower_closure(mask,raw,80,321,last)


def test_narrow_support_cannot_define_front_body_end():
 mask,raw,body,last=fixture();mask[550:650,140:260]=True;raw[550:650,140:260]=20
 with pytest.raises(ValueError):front_lower_closure(mask,raw,80,321,last)


def test_disconnected_bottom_border_residue_cannot_extend_phone_conservation():
    mask, raw, body, last = fixture()
    alpha = mask.astype(np.uint8) * 255
    baseline = front_lower_closure(mask, raw, 80, 321, last, alpha)
    # A separate strip at the photograph boundary reproduced the Pro failure.
    alpha[-1, 180:220] = 255
    phone = principal(alpha)
    actual = front_lower_closure(phone, raw, 80, 321, last, alpha)
    np.testing.assert_array_equal(actual[1], baseline[1])
    assert actual[-1]['source_conservation_outward_px'] == baseline[-1]['source_conservation_outward_px']
    assert actual[-1]['source_conservation_component'] == 'principal_phone'
    rows, columns = np.indices(mask.shape)
    removed = rows > np.polyval(actual[1], columns + .5) + 1
    assert not np.any(removed & body)


def test_standard_front_closure_rejects_white_clips_as_the_phone_lower_edge(monkeypatch):
    from phone_texture.auto_geometry import plate_geometry
    import phone_texture.auto_geometry as geometry

    mask, raw, body, last = fixture()
    mask = body.copy()
    raw[:] = 230
    raw[body] = 20
    for left, right in [(128, 154), (247, 273)]:
        mask[530:590, left:right] = True
        raw[550:590, left:right] = 200
    matte = Image.fromarray(np.dstack([raw, mask.astype('uint8') * 255]))
    def no_symmetric_corner(*args):
        raise ValueError('Plate sides are not a parallel studio capture')
    monkeypatch.setattr(geometry, 'plate_plane_from_own_corners', no_symmetric_corner)
    stages = {}
    image, proof = plate_geometry(
        matte, 'front', source_image=Image.fromarray(raw), prefer_front_closure=True,
        stage_sink=lambda name, result: stages.update({name: result}))
    source = np.asarray(stages['after_support'])
    assert not source[552:590, 128:154, 3].any()
    assert not source[552:590, 247:273, 3].any()
    np.testing.assert_array_equal(source[body], np.asarray(matte)[body])
    assert proof['lower_plane_evidence']['dark_glass_closure_fractions'] == [1., 1.]
    assert image.height <= 502


def test_attached_bright_stand_cannot_pass_as_a_completed_front():
    from phone_texture.auto_geometry import check_front_support_residue
    mask,raw,body,last=fixture()
    raw[mask & ~body]=220
    image=Image.fromarray(np.dstack([raw,mask.astype('uint8')*255]))
    with pytest.raises(ValueError,match='stand projection'):
        check_front_support_residue(image)
    clean=Image.fromarray(np.dstack([raw,body.astype('uint8')*255]))
    assert not check_front_support_residue(clean)['bright_support_residue']
