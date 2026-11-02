"""A scratch contour is not a rectangular hardware material/color island."""
import numpy as np
from PIL import Image, ImageDraw
import pytest

from phone_texture import auto_material, lighting, lighting_review
from phone_texture.side_material import hardware_materials


def side_image():
    h,w=1000,80;yy,xx=np.indices((h,w))
    value=175+20*np.sin(yy/180)+np.where((xx+yy)%2,1,-1)
    a=np.full((h,w,4),255,np.uint8);a[...,:3]=np.rint(value[...,None]+np.array([0,3,5]))
    a[:,:2,3]=0;a[:,-2:,3]=0;a[:,2,3]=128;a[:,-3,3]=128
    image=Image.fromarray(a);draw=ImageDraw.Draw(image)
    draw.rectangle((2,80,77,88),fill=(100,102,104,255))
    draw.rectangle((2,900,77,908),fill=(100,102,104,255))
    draw.rounded_rectangle((26,300,54,430),radius=14,fill=(188,191,193,255),outline=(65,67,69,255),width=3)
    draw.rounded_rectangle((26,600,54,730),radius=14,fill=(90,93,95,255),outline=(45,47,49,255),width=3)
    draw.rectangle((15,230,22,270),outline=(145,147,149,255),width=1)
    return image


def test_complete_components_exclude_thin_scratches_and_rectangle_corners():
    image=side_image();original=np.asarray(image).copy()
    components=hardware_materials(image,'right')
    assert len(components)==2
    mask=np.logical_or.reduce([m for m,_ in components])
    assert not mask[230:271,15:23].any()
    assert mask[350,40] and mask[650,40]
    assert not mask[300,26]  # Empty capsule corner cannot become a color square.
    assert all(p['boundary_support']>=.95 and min(p['edge_support'])>=.75 for _,p in components)
    np.testing.assert_array_equal(image,original)


def test_missing_or_extra_hardware_is_rejected_instead_of_becoming_background():
    image=side_image();draw=ImageDraw.Draw(image)
    draw.rectangle((20,590,60,740),fill=(180,183,185,255))
    with pytest.raises(ValueError,match='Unresolved right hardware'):
        hardware_materials(image,'right')
    image=side_image();draw=ImageDraw.Draw(image)
    draw.rounded_rectangle((26,460,54,550),radius=14,fill=(190,193,195,255),outline=(60,63,65,255),width=3)
    with pytest.raises(ValueError,match='Unresolved right hardware'):
        hardware_materials(image,'right')


def test_weak_outer_rail_cannot_select_an_intact_inner_groove():
    image=Image.new('RGBA',(100,1200),(180,183,185,255));draw=ImageDraw.Draw(image)
    for y in (300,700):
        draw.rounded_rectangle((30,y,70,y+180),radius=20,fill=(180,183,185,255),
            outline=(80,83,85,255),width=2)
        draw.rounded_rectangle((35,y+5,65,y+175),radius=15,fill=(165,168,170,255),
            outline=(90,93,95,255),width=2)
    complete=hardware_materials(image,'right')
    for y in (300,700):
        # Seven weak source pixels leave the outer contour above every existing
        # support gate, but below the old 99% preference for the inner groove.
        draw.rectangle((27,y+50,33,y+56),fill=(180,183,185,255))
    weakened=np.asarray(image).copy()
    components=hardware_materials(image,'right')
    assert len(components)==len(complete)==2
    for (mask,proof),(outer,outer_proof) in zip(components,complete):
        assert .95<=proof['boundary_support']<.99
        assert min(proof['edge_support'])>=.75
        assert proof['box']==outer_proof['box']
        np.testing.assert_array_equal(mask,outer)
    np.testing.assert_array_equal(image,weakened)


def test_independent_hardware_references_keep_dark_control_and_source_detail(tmp_path):
    image=side_image();source=tmp_path/'source.png';image.save(source)
    job,proof,allowed,paper=auto_material.propose_job(source,'right',tmp_path/'material')
    module=lighting.engine('aluminum');target,_=module.target_color(job)
    result,_,_,report=module.process_view(job['views'][0],target,dict(job['defaults'],color_mode='source_lightness'))
    output=tmp_path/'output.png';Image.fromarray(result).save(output)
    result=np.asarray(Image.open(output));original=np.asarray(image)
    domains={d['name']:d for d in job['views'][0]['domains']}
    assert 'hardware' not in domains and {'hardware_0','hardware_1'}<=set(domains)
    assert domains['hardware_0']['reference_boxes']!=domains['hardware_1']['reference_boxes']
    assert result[650,40,:3].mean()<result[350,40,:3].mean()-60
    np.testing.assert_array_equal(result[...,3],original[...,3])
    np.testing.assert_array_equal(result[~allowed|paper],original[~allowed|paper])
    # Genuine thin scratch remains darker than its nearby metal, at the same pixels.
    assert result[245,15,:3].mean()<result[245,19,:3].mean()-10
    assert report['source_sha256']==lighting_review.digest(source)
    scope=auto_material.lighting_scope(source,'right',job,tmp_path/'scope')
    metrics=lighting_review.measure_local(scope,source,output,allowed,paper)
    assert any(k.startswith('side_metal_fragment_') for k in metrics)
    # An introduced old-style rectangle must still fail the unchanged 1.25 L gate.
    bad=result.copy();bad[230:271,15:23,:3]=np.maximum(bad[230:271,15:23,:3].astype(int)-35,0)
    Image.fromarray(bad).save(tmp_path/'block.png')
    with pytest.raises(ValueError,match='discontinuity'):
        lighting_review.measure_local(scope,source,tmp_path/'block.png',allowed,paper)


def test_hardware_rectangle_requires_complete_component_checks(tmp_path):
    source=tmp_path/'source.png';side_image().save(source)
    job,_,allowed,paper=auto_material.propose_job(source,'right',tmp_path/'material')
    module=lighting.engine('aluminum');target,_=module.target_color(job)
    result,_,_,_=module.process_view(job['views'][0],target,
        dict(job['defaults'],color_mode='source_lightness'))
    scope=auto_material.lighting_scope(source,'right',job,tmp_path/'scope')
    domains={d['name']:d for d in job['views'][0]['domains']}
    hardware=np.asarray(Image.open(domains['hardware_1']['mask']))>0
    block=np.zeros(hardware.shape,bool);block[625:705,32:49]=True
    assert np.all(hardware[block])
    bad=result.copy();bad[block,:3]=np.minimum(bad[block,:3].astype(int)+35,255)
    output=tmp_path/'hardware-block.png';Image.fromarray(bad).save(output)
    # Surrounding metal can remain perfect while the button itself is damaged.
    old_checks={k:v for k,v in scope['local_checks'].items() if not k.startswith('hardware_')}
    assert any(k.startswith('side_metal_fragment_') for k in old_checks)
    lighting_review.measure_local(dict(scope,local_checks=old_checks),source,output,allowed,paper)
    for direction in ('length','width'):
        name='hardware_1_'+direction
        with pytest.raises(ValueError,match='discontinuity at '+name):
            lighting_review.measure_local(dict(scope,local_checks={name:scope['local_checks'][name]}),
                source,output,allowed,paper)


def test_pixel_continuity_rejects_small_flattened_texture_hidden_by_line_percentile(tmp_path):
    image=side_image().resize((120,1500),Image.Resampling.NEAREST)
    source_pixels=np.asarray(image).copy();yy,xx=np.indices((30,8))
    source_pixels[950:980,56:64,:3]=np.array([90,93,95])+np.where((yy+xx)%2,20,-20)[...,None]
    source=tmp_path/'source.png';Image.fromarray(source_pixels).save(source)
    job,_,allowed,paper=auto_material.propose_job(source,'right',tmp_path/'materials')
    scope=auto_material.lighting_scope(source,'right',job,tmp_path/'scope')
    checks={k:v for k,v in scope['local_checks'].items() if k in ('hardware_1_length','hardware_1_width')}
    assert len(checks)==2 and all(c['max_pixel_added_step_L']==1.25 for c in checks.values())
    # Unchanged source texture is valid, however strong its original gradients.
    metrics=lighting_review.measure_local(dict(scope,local_checks=checks),source,source,allowed,paper)
    assert all(m['max_pixel_added_step_L']==0 for m in metrics.values())
    flat=source_pixels.copy()
    flat[950:980,56:64,:3]=np.median(flat[950:980,56:64,:3],axis=(0,1)).astype(np.uint8)
    assert lighting_review.lightness(source_pixels[950:980,56:64,:3]).std()>5
    assert lighting_review.lightness(flat[950:980,56:64,:3]).std()<1e-10
    output=tmp_path/'flattened.png';Image.fromarray(flat).save(output)
    # The 8x30 erased patch is diluted below 25% of either component axis.
    old_checks={k:{p:v for p,v in c.items() if p!='max_pixel_added_step_L'} for k,c in checks.items()}
    old=lighting_review.measure_local(dict(scope,local_checks=old_checks),source,output,allowed,paper)
    assert all(m['max_added_step_L']<=1.25 for m in old.values())
    for name,check in checks.items():
        with pytest.raises(ValueError,match='pixel correction discontinuity at '+name):
            lighting_review.measure_local(dict(scope,local_checks={name:check}),source,output,allowed,paper)
    # The optional peak gate is active on its own, not silently dependent on
    # whether a caller also configures the older coherent-line gate.
    standalone={k:{p:v for p,v in c.items() if p!='max_added_step_L'} for k,c in checks.items()}
    with pytest.raises(ValueError,match='pixel correction discontinuity'):
        lighting_review.measure_local(dict(scope,local_checks=standalone),source,output,allowed,paper)


def test_component_reference_rejects_uniform_wrong_hardware_target(tmp_path):
    source=tmp_path/'source.png';side_image().save(source)
    job,_,allowed,paper=auto_material.propose_job(source,'right',tmp_path/'materials')
    scope=auto_material.lighting_scope(source,'right',job,tmp_path/'scope')
    domains={d['name']:d for d in job['views'][0]['domains']}
    check=scope['local_checks']['hardware_1_reference']
    assert check['percentile']==50
    assert check['min_L']==max(0.,min(check['source_before_L'],check['source_reference_L'])-5.)
    assert check['max_L']==min(100.,max(check['source_before_L'],check['source_reference_L'])+5.)
    np.testing.assert_array_equal(np.asarray(Image.open(check['mask']))>0,
        np.asarray(Image.open(domains['hardware_1']['fit_mask']))>0)
    source_pixels=np.asarray(Image.open(source));hardware=np.asarray(Image.open(domains['hardware_1']['mask']))>0
    module=lighting.engine('aluminum');lab=module.rgb_to_lab(source_pixels[...,:3])
    lifted=lab.copy();lifted[hardware,0]+=15
    bad=source_pixels.copy();bad[hardware,:3]=module.lab_to_rgb(lifted)[hardware]
    output=tmp_path/'wrong-target.png';Image.fromarray(bad).save(output)
    actual=lighting_review.lightness(bad[...,:3])-lighting_review.lightness(source_pixels[...,:3])
    assert 14.5<float(np.median(actual[hardware]))<15.5
    # A uniform wrong target remains continuous and retains source texture.
    continuity={k:v for k,v in scope['local_checks'].items() if k in ('hardware_1_length','hardware_1_width')}
    lighting_review.measure_local(dict(scope,local_checks=continuity),source,output,allowed,paper)
    with pytest.raises(ValueError,match='Local feature effect failed: hardware_1_reference'):
        lighting_review.measure_local(dict(scope,local_checks={'hardware_1_reference':check}),
            source,output,allowed,paper)


def test_fit_mask_is_only_estimation_support_and_cannot_bypass_support_gate(tmp_path):
    module=lighting.engine('aluminum');source=tmp_path/'source.png'
    Image.new('RGBA',(80,100),(160,165,170,255)).save(source)
    fit=np.zeros((100,80),np.uint8);fit[40:60,35:45]=255
    fit_path=tmp_path/'fit.png';Image.fromarray(fit).save(fit_path)
    item=dict(source=str(source),domains=[dict(shapes=[dict(type='rectangle',xy=[0,0,79,99])],
        fit_mask=str(fit_path),sigma=[1,1],feather=0)])
    with pytest.raises(ValueError,match='Unsupported editable illumination'):
        module.process_view(item,np.array([70,0,0]),{})
    Image.new('L',(10,10),255).save(fit_path)
    with pytest.raises(ValueError,match='Fit mask dimensions'):
        module.process_view(item,np.array([70,0,0]),{})
