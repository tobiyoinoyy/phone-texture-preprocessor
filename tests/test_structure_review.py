import copy
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from phone_texture import structure_review as s
from test_six_view_workflow import w, scope
from geometry_helpers import approved_fixture


def fixture(tmp):
    root=tmp/'source';folder=root/'A';folder.mkdir(parents=True)
    for face in w.FACES:
        a=np.full((256,60,4),255,np.uint8);a[...,:3]=150
        for lo,hi in ([(40,52),(200,212)] if face in ('left','right') else [(60,72),(180,192)]):a[lo:hi,:,:3]=70
        a[110:145,25:35,:3]=90  # independent button witness
        Image.fromarray(a).save(folder/(face+'.png'))
    files=s.batch_files(root);spec={'A':{}}
    for side in ('left','right'):
        spec['A'][side]=dict(source_sha256=s.digest(folder/(side+'.png')),body_y=[0,255],
            source_lines=[10,39.5,51.5,100,155,199.5,211.5,245],antenna_indices=[1,2,5,6],
            button_ranges=[[110,145]],button_basis='Reviewed complete synthetic button',
            back=dict(face='back',sha256=s.digest(folder/'back.png'),body_y=[0,255],antenna_rows=[59.5,71.5,179.5,191.5],
                x_range=[5,15],search_windows=[[54,65],[66,77],[174,185],[186,197]],signs=[-1,1,-1,1]),
            measurements={name:dict(x_range=x,search_windows=[[54,65],[66,77],[174,185],[186,197]],signs=[-1,1,-1,1])
                          for name,x in [('left_end',[4,12]),('center',[25,35]),('right_end',[48,56])]})
    specfile=tmp/'spec.json';s.write(specfile,spec)
    return root,files,spec,specfile


def approved(tmp):
    root,files,spec,path=fixture(tmp);process=tmp/'structure';s.prepare(root,path,process)
    review=s.read(process/'review-template.json');review['reviewer']='Test visual inspector'
    for side in ('left','right'):review['items']['A'][side]={**dict.fromkeys(s.CHECKS,True),'notes':'Reviewed both antennas, endpoints and full button'}
    s.write(process/'review.json',review)
    return root,files,spec,path,process


def test_real_pixels_align_and_unaligned_control_is_rejected(tmp_path):
    root,files,spec,path,process=approved(tmp_path)
    after,proof=s.verify_delivery(files,process,process/'review.json')
    assert proof['status']=='image_space_passed'
    assert proof['model_uv_verified'] is False
    assert np.array_equal(s.pixels(files['A']['left'])[110:145],s.pixels(after['A']['left'])[110:145])
    with pytest.raises(ValueError,match='remapping'):s.verify(files,files,spec)


@pytest.mark.parametrize('failure',['candidate','source','spec','review','evidence','companion'])
def test_stale_or_incomplete_structure_proof_is_rejected(tmp_path,failure):
    root,files,spec,path,process=approved(tmp_path)
    if failure in ('candidate','companion','source'):
        p=files['A']['left'] if failure=='source' else process/'candidate/A'/('back.png' if failure=='companion' else 'left.png')
        a=s.pixels(p);a[120,30,0]+=1;Image.fromarray(a).save(p)
    elif failure=='spec':
        sp=s.read(process/'spec.json');sp['A']['left']['back']['antenna_rows'][0]+=5;s.write(process/'spec.json',sp)
    elif failure=='review':
        r=s.read(process/'review.json');r['items']['A']['right']['lower_edges_and_width']=False;s.write(process/'review.json',r)
    else:
        p=Path(s.read(process/'report.json')['artifacts'][0]['path']);p.write_bytes(b'changed')
    with pytest.raises(ValueError):s.verify_delivery(files,process,process/'review.json')


@pytest.mark.parametrize('failure',['missing_side','other_back','fake_landmarks','wrong_endpoint','buttons','bad_windows','unsupported_basis'])
def test_invalid_calibration_cannot_pass(tmp_path,failure):
    root,files,spec,path=fixture(tmp_path);c=spec['A']['left']
    if failure=='missing_side':del spec['A']['right']
    elif failure=='other_back':c['back']['sha256']='wrong'
    elif failure=='fake_landmarks':c['back']['antenna_rows'][0]+=8
    elif failure=='wrong_endpoint':c['measurements']['left_end']['x_range']=[25,35]
    elif failure=='buttons':c['button_ranges']=[[30,90]]
    elif failure=='bad_windows':c['back']['search_windows'][0]=[54,55]
    else:c['basis']='model_uv'
    s.write(path,spec)
    with pytest.raises(ValueError):s.prepare(root,path,tmp_path/'structure')


def test_full_publication_requires_structure_and_publishes_aligned_pixels(tmp_path):
    root,files,spec,path,process=approved(tmp_path)
    work=tmp_path/'work';w.intake(root,work)
    proof=approved_fixture(root,work);w.checkpoint(work,root,proof)
    plan={'A':{f:{'action':'copy','reason':'Synthetic fixture; no lighting requested'} for f in w.FACES}}
    lighting={'A':{f:dict(action='copy',mode='unchanged',reason='Synthetic fixture',source_sha256=w.sha(p),
        protection_policy='reviewed_regions',protection_basis='No lighting requested') for f,p in files['A'].items()}}
    for f in ('front','back'):lighting['A'][f]['perimeter_exemption']=dict(reason='user_scope_excludes_perimeter',basis='No lighting requested',source_sha256=w.sha(files['A'][f]))
    w.save(work/'scope.json',lighting);w.freeze_scope(work,work/'scope.json');w.save(work/'plan.json',plan)
    with pytest.raises(ValueError,match='require structure'):w.publish(work,work/'plan.json',tmp_path/'out')
    assert not (tmp_path/'out').exists()
    w.publish(work,work/'plan.json',tmp_path/'out',process,process/'review.json')
    assert w.sha(tmp_path/'out/A/left.png')==w.sha(process/'candidate/A/left.png')
    validation=w.read(work/'validation.json');assert validation['structure']['status']=='image_space_passed'
    assert validation['skus']['A']['left']['structure_changed']


def test_photos_exemption_requires_scope_and_legacy_intake_fails_closed(tmp_path):
    root,files,spec,path=fixture(tmp_path)
    with pytest.raises(ValueError,match='scope reason'):w.intake(root,tmp_path/'work',delivery_goal='photos')
    w.intake(root,tmp_path/'work');m=w.read(tmp_path/'work/intake.json');del m['delivery_goal'];w.save(tmp_path/'work/intake.json',m)
    with pytest.raises(ValueError,match='require structure'):w.publish(tmp_path/'work',tmp_path/'none.json',tmp_path/'out')


def test_width_error_is_detected_even_when_mapping_report_is_consistent(tmp_path):
    root,files,spec,path=fixture(tmp_path)
    # Reviewed lower source edge is wrong: remapping runs, but actual stripe becomes too thin.
    spec['A']['left']['source_lines'][2]+=5
    s.write(path,spec)
    with pytest.raises(ValueError,match='position/width mismatch'):
        s.prepare(root,path,tmp_path/'structure')


def test_front_back_targets_keep_both_ends_independent(tmp_path):
    root,files,spec,path=fixture(tmp_path)
    a=s.pixels(files['A']['front']);a[...,:3]=150;a[64:76,:,:3]=70;a[184:196,:,:3]=70;Image.fromarray(a).save(files['A']['front'])
    for side in ('left','right'):
        c=spec['A'][side];c['basis']='front_back_images';c['front_at']='left'
        c['front']=dict(c['back'],face='front',sha256=s.digest(files['A']['front']),antenna_rows=[63.5,75.5,183.5,195.5],
            search_windows=[[58,69],[70,81],[178,189],[190,201]])
        for m in c['measurements'].values():m['search_windows']=[[54,68],[69,81],[174,188],[189,201]]
    s.write(path,spec);process=tmp_path/'structure';s.prepare(root,path,process)
    checks=s.read(process/'report.json')['checks']['A']['left']['checks']
    assert checks['left_end']['target_rows'][0]>checks['right_end']['target_rows'][0]+2
