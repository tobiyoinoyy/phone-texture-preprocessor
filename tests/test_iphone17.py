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
