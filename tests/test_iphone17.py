"""Independent layout, holdout and source-integrity checks for standard 17."""
import numpy as np
import pytest
from PIL import Image,ImageDraw
from phone_texture.iphone17_port import DRAWING_HOLES,WIDTH_MM,mapping,apply
from phone_texture.iphone17_structure import hardware,capsule_mask


def bottom(*,port_shift=0,missing=False):
    im=Image.new('RGBA',(1030,120),(220,224,228,255));d=ImageDraw.Draw(im)
    project=lambda x:14.8*x+3
    holes=np.sort(WIDTH_MM-DRAWING_HOLES)
    for i,x in enumerate(holes):
        if missing and i==1:continue
        cx=project(x);d.ellipse((round(cx-10),47,round(cx+10),67),fill='#353535')
    pc=project(WIDTH_MM/2)+port_shift
    d.rounded_rectangle((round(pc-65),35,round(pc+65),78),radius=18,fill='#202020')
    cx=project(42.65);d.ellipse((round(cx-11),46,round(cx+11),68),fill='#353535')
    return im


def test_standard_asymmetric_holes_register_without_fitting_port():
    image=bottom();mx,proof=mapping(image)
    assert proof['port_used_for_fitting'] is False
    assert abs(proof['port_after_x']-proof['canvas_center_x'])<1.5
    assert proof['holdout_holes_max_error_px']<1.5
    result=apply(image,mx);a=np.asarray(image);b=np.asarray(result)
    cap=proof['native_endcaps_fixed_px']
    assert np.array_equal(a[:,:cap],b[:,:cap])
    assert np.array_equal(a[:,-cap:],b[:,-cap:])
    assert a.shape==b.shape


def test_independent_port_holdout_rejects_wrong_nominal_alignment():
    with pytest.raises(ValueError,match='USB-C holdout'):
        mapping(bottom(port_shift=8))


def test_missing_or_pro_style_holes_cannot_pass_as_standard():
    with pytest.raises(ValueError,match='asymmetric'):
        mapping(bottom(missing=True))


def test_hardware_requires_four_standard_left_capsules():
    im=Image.new('RGBA',(110,1940),(225,228,230,255));d=ImageDraw.Draw(im)
    for y,length in [(356,100),(516,155),(704,155),(1225,265)]:
        d.rounded_rectangle((32,y,77,y+length),radius=16,outline='#404040',width=3)
    assert len(hardware(im,'left'))==4
    first=hardware(im,'left')[0]
    mask=capsule_mask(im,first)
    assert not mask[first[1],first[0]]
    assert mask[first[1]+40,first[0]+20]
    with pytest.raises(ValueError,match='expected 2'):
        hardware(im,'right')


def test_blank_hardware_is_rejected():
    with pytest.raises(ValueError,match='capsule'):
        hardware(Image.new('RGBA',(110,1940),(220,220,220,255)),'left')


def rear_rim(*,shift=0,bands=True,glass_only=False):
    im=Image.new('RGBA',(1000,2060),(0,0,0,0));d=ImageDraw.Draw(im)
    d.rounded_rectangle((10,1,989,2058),radius=100,fill='#404040')
    d.rounded_rectangle((18,9,981,2050),radius=96,fill='#dddddd')
    if bands:
        for x0,x1,rows in [(10,15,[195,211,1810,1825]),(984,989,[217,232,1830,1846])]:
            if glass_only:x0,x1=100,130
            for lo,hi in zip(rows[::2],rows[1::2]):
                d.rectangle((x0,lo+shift,x1,hi+shift-1),fill='#999999')
    return im


def test_rear_rim_measures_independent_bands_and_rear_view_orientation():
    from phone_texture.iphone17_structure import rear_antennas
    actual=rear_antennas(rear_rim())
    for side,expected in [('right',[195,211,1810,1825]),('left',[217,232,1830,1846])]:
        assert np.max(abs(np.array(actual[side]['rows'])-expected))<=1
    assert actual['right']['rear_image_edge']=='left'
    shifted=rear_antennas(rear_rim(shift=12))
    for side in actual:
        assert np.max(abs(np.array(shifted[side]['rows'])-actual[side]['rows']-12))<=1


@pytest.mark.parametrize('kwargs',[{'bands':False},{'glass_only':True}])
def test_missing_rear_rim_cannot_fall_back_to_side_bands(kwargs):
    from phone_texture.iphone17_structure import rear_antennas
    with pytest.raises(ValueError,match='Rear rim antenna'):
        rear_antennas(rear_rim(**kwargs))


def test_old_native_side_only_config_is_rejected():
    from phone_texture.iphone17_structure import apply as align
    with pytest.raises(ValueError,match='Measured rear rim'):
        align(Image.new('RGBA',(110,1940)),{'basis':'same_sku_back_canvas_scale_native_side_bands','back_antenna_present':False})


def test_each_side_rejects_common_opposite_rear_edge():
    from phone_texture.iphone17_structure import apply as align
    cfg={'basis':'same_sku_back_rim_measured_pixel_center_ratio','back_antenna_present':True,
         'side':'right','reference_policy':'corresponding_rear_edge','rear_detection':{'rear_image_edge':'right'}}
    with pytest.raises(ValueError,match='own corresponding rear edge'):
        align(Image.new('RGBA',(110,1940)),cfg)


def test_subpixel_rear_gradient_is_not_rounded_to_integer():
    from phone_texture.iphone17_structure import gradient_peak
    x=np.arange(50);g=np.exp(-((x-20.35)/2)**2)
    assert abs(gradient_peak(g,20,1)-20.35)<.03


def test_rendered_bottom_residual_cannot_pass_with_correct_upper():
    from phone_texture.iphone17_structure import validate_render_checks
    check={'side':'right','reference_policy':'corresponding_rear_edge','rear_image_edge':'left',
           'refined_indices':[2,3],'rendered_error_rear_px':[[0,0,0,0]]*3,'rendered_width_error_rear_px':[[0,0]]*3}
    validate_render_checks(check)
    check['rendered_error_rear_px']=[[0,0,0,0],[0,0,0,0],[0,0,.9,.9]]
    with pytest.raises(ValueError,match='residual'):validate_render_checks(check)
    check['rendered_error_rear_px']=[[0,0,-.5,.5]]*3
    check['rendered_width_error_rear_px']=[[0,1.]]*3
    with pytest.raises(ValueError,match='width'):validate_render_checks(check)


def test_rear_side_height_difference_is_preserved_not_averaged():
    from phone_texture.iphone17_structure import rear_antennas
    detection=rear_antennas(rear_rim())
    assert detection['left']['rows'][2]-detection['right']['rows'][2]>18


def test_same_rear_ridge_can_interrupt_across_only_three_native_tracks():
    from phone_texture.iphone17_structure import _coherent_rim_stripe
    im = Image.new('RGBA', (988, 2058), (80, 80, 80, 255))
    draw = ImageDraw.Draw(im)
    for x, level in [(7, 120), (8, 180), (9, 120)]:
        draw.line((x, 0, x, im.height-1), fill=(level, level, level, 255))
        draw.line((x, 211, x, 225), fill=(90, 90, 90, 255))
    pair, proof = _coherent_rim_stripe(im, 'right', 'upper')
    assert np.allclose(pair, [210.5, 225.5], atol=.2)
    assert proof['track_offsets_px'] == [7, 8, 9]
    # A stripe reaching across the glass is not independent rear-rim evidence.
    draw.line((26, 211, 26, 225), fill=(20, 20, 20, 255))
    with pytest.raises(ValueError):
        _coherent_rim_stripe(im, 'right', 'upper')


def test_bilinear_half_coverage_is_measured_on_the_interpolated_profile():
    from phone_texture.iphone17_structure import _warp_along_rail
    rgba = np.full((100, 36, 4), 255, dtype=np.uint8)
    rgba[40, :3, 3] = [101, 165, 218]
    rgba[41, :3, 3] = [123, 180, 226]
    image = Image.fromarray(rgba)
    src, dst = [2, 40.36, 97], [2, 50, 97]
    previous = _warp_along_rail(image, src, dst)
    current = _warp_along_rail(image, src, dst, interpolate_profile=True)
    assert np.asarray(previous)[50, 0, 3] == 127
    assert np.asarray(current)[50, 0, 3] == 128
    assert np.array_equal(np.asarray(current)[[0, 99]], rgba[[0, 99]])


def test_two_individually_visible_stripes_at_unrelated_positions_do_not_make_a_rear_pair():
    from phone_texture.iphone17_structure import _paired_rim_stripes
    im = Image.new('RGBA', (988, 2058), (80, 80, 80, 255))
    draw = ImageDraw.Draw(im)
    for x, level in [(7, 125), (8, 180), (9, 125)]:
        draw.line((x, 0, x, im.height-1), fill=(level, level, level, 255))
        for y in (211, 1900):
            draw.line((x, y, x, y+14), fill=(90, 90, 90, 255))
    with pytest.raises(ValueError, match='opposite ends'):
        _paired_rim_stripes(im, 'right')


def test_native_span_warp_keeps_the_photographed_silhouette_and_locked_cap():
    from phone_texture.iphone17_structure import _warp_inside_native_rail
    rgba=np.zeros((110,60,4),dtype=np.uint8)
    for y in range(110):
        left=3+int(35<y<60)
        rgba[y,left:55,:3]=[50+y,80,105]
        rgba[y,left:55,3]=255
        rgba[y,left,3]=110
    im=Image.fromarray(rgba)
    out=np.asarray(_warp_inside_native_rail(im,[5,40,70,104],[5,47,70,104]))
    assert np.array_equal(out[...,3],rgba[...,3])
    assert np.array_equal(out[:6],rgba[:6])
    assert np.array_equal(out[70:],rgba[70:])
    assert out[47,30,0]==rgba[40,30,0]


def test_partner_material_reference_cannot_fill_a_missing_break(monkeypatch):
    from phone_texture import iphone17_structure as structure
    image=Image.new('RGBA',(988,2058),(80,80,80,255))
    draw=ImageDraw.Draw(image)
    draw.rectangle((1,1821,12,1835),fill=(150,150,150,255))
    def anchor(_,side,band):
        if band=='upper':
            raise ValueError('missing')
        return [1820.5,1835.5],dict(rows=[1820.5,1835.5])
    monkeypatch.setattr(structure,'_confirmed_rim_pair',anchor)
    with pytest.raises(ValueError,match='No independent optical match'):
        structure._matched_partner(image,'right')


def test_ridge_peak_and_area_require_two_real_interruptions():
    from phone_texture.iphone17_structure import _ridge_height_and_area
    rgba=np.full((2058,988,4),255,dtype=np.uint8)
    rgba[...,:3]=45
    for col,bright in [(5,80),(6,130),(7,190),(8,130),(9,80)]:
        rgba[:,col,:3]=bright
        for start in (211,1832):
            rgba[start:start+15,col,:3]=55
    result=_ridge_height_and_area(Image.fromarray(rgba),'right')
    assert np.allclose(result[0][0],[210.5,225.5],atol=.5)
    assert np.allclose(result[1][0],[1831.5,1846.5],atol=.5)
    rgba[1832:1847]=rgba[1831]
    with pytest.raises(ValueError):
        _ridge_height_and_area(Image.fromarray(rgba),'right')


def test_source_and_export_choose_separate_levels_and_check_every_witness(tmp_path):
    from phone_texture import iphone17_port as port, workspace

    image = bottom()
    mx, proof = mapping(image)
    rendered = apply(image, mx)
    # Relight both metal and dark details to a disjoint grayscale range.
    a = np.array(rendered)
    a[..., :3] = np.rint(a[..., :3].astype(float)*.22+175)
    path = tmp_path/'bottom.png'
    Image.fromarray(a).save(path)
    port.verify_export(path, proof)
    assert proof['final_verified_sha256'] == workspace.sha(path)
    assert max(proof['selected_thresholds']) < min(proof['rendered_hole_thresholds'])
    assert len(proof['rendered_hole_errors_px']) == 8
    assert any(c['status']=='confirmed' for c in proof['rendered_screw_checks'])
    assert set(proof['rendered_port_contour']['selected']['support']) == {'left_end','right_end','top','bottom'}
    # Even with a good port centre and all eight holes, a missing screw fails.
    x = round(42.65 / WIDTH_MM * (image.width-1))
    a[:, x-18:x+19] = a[:, :1]
    Image.fromarray(a).save(path)
    with pytest.raises(ValueError):
        port.verify_export(path, dict(proof))
