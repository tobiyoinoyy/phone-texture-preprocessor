"""Fixed-reference gate regressions with labelled synthetic image coordinates."""
import copy
from pathlib import Path
import pytest
from phone_texture import model_reference as m, workspace as w
from test_structure_review import fixture


def model_fixture(tmp):
    source, files, spec, specfile = fixture(tmp)
    glb = tmp / 'fixed.glb'; glb.write_bytes(b'synthetic model hash fixture')
    rows = [59.5, 71.5, 179.5, 191.5]
    profile = dict(schema='phone-model-uv-reference-v1', model=dict(glb='fixed.glb', glb_sha256=w.sha(glb)),
        reference_texture_directory=str(source / 'A'), reference_images={f: dict(file=f+'.png', size=[60,256], sha256=w.sha(p)) for f,p in files['A'].items()},
        material_image_mapping={**{f:f+'.png' for f in w.FACES}, 'back-camera':'back.png', 'backdetal':'back.png'},
        antenna_rows=dict(front=dict(image_left=rows,image_right=rows), back=dict(image_left=rows,image_right=rows), left=rows,right=rows),
        lens_centers_back_px=[[22,100]],button_outline_boxes_px=dict(action=dict(face='left',box=[25,110,35,145])))
    path=tmp/'profile.json';w.save(path,profile)
    reviewpath=tmp/'review.json';m.template(files,path,reviewpath);review=w.read(reviewpath)
    review['reviewer']='Labelled synthetic fixture'
    for face,item in review['items']['A'].items():
        item.update(**dict.fromkeys(m.CHECKS,True),notes='Known synthetic labelled coordinates')
        points={}
        for name in item['reference_points']:
            if name.startswith('body:'):
                points[name]=[[1,1],[58,1],[1,254],[58,254]][int(name.split(':')[1])]
            elif name.startswith('antenna:'):
                _,end,i=name.split(':');points[name]=[5 if end in ('image_left','left') else 54,rows[int(i)]]
            elif name.startswith('lens:'):points[name]=[22,100]
            elif name.startswith('button:'):points[name]=[25,110] if name.endswith(':tl') else [35,145]
            else:points[name]=[20,120] if name.endswith(':0') else [40,120]
        item['reference_points']=points;item['output_points']=copy.deepcopy(points)
        item['evidence']=[dict(path=str(files['A'][face]),sha256=w.sha(files['A'][face]),scale=i) for i in (1,4)]
    w.save(reviewpath,review)
    return files,path,reviewpath,review


def test_known_coordinates_pass_but_do_not_claim_live_model_verification(tmp_path):
    files,path,reviewpath,review=model_fixture(tmp_path)
    proof=m.verify(files,path,reviewpath)
    assert proof['status']=='reference_image_coordinates_passed'
    assert proof['model_uv_verified'] is False
    m.verify_proof(files,dict(path=str(path),sha256=w.sha(path)),proof)


@pytest.mark.parametrize('fault',['position','fake_target','model','image','evidence','missing','slant'])
def test_fixed_reference_cannot_be_satisfied_by_arbitrary_or_stale_proof(tmp_path,fault):
    files,path,reviewpath,review=model_fixture(tmp_path)
    item=review['items']['A']['left']
    if fault=='position':item['output_points']['button:action:tl'][1]+=8
    elif fault=='fake_target':
        item['reference_points']['button:action:tl'][1]+=8
        item['output_points']['button:action:tl'][1]+=8
    elif fault=='model':(tmp_path/'fixed.glb').write_bytes(b'changed')
    elif fault=='image':files['A']['back'].write_bytes(b'changed')
    elif fault=='evidence':item['evidence'][0]['sha256']='stale'
    elif fault=='missing':del review['items']['A']['bottom']
    else:item['output_points']['antenna:right:0'][1]+=4
    w.save(reviewpath,review)
    with pytest.raises(ValueError):m.verify(files,path,reviewpath)


def test_plain_model_texture_publication_cannot_bypass_linked_fixed_reference(tmp_path):
    from test_workflow_runner import batch, approve_structure
    work,output,config,files,spec,specfile=batch(tmp_path)
    approve_structure(work,config,specfile)
    assert __import__('phone_texture.workflow',fromlist=['run']).run(work)['status']=='completed'
    # Register an unresolved fixed reference in a fresh lower-level publication
    # request; no image-space report alone can satisfy it.
    manifest=w.read(work/'intake.json')
    manifest['workflow_contract']['model_reference']=dict(path=str(tmp_path/'missing-profile.json'),sha256='missing')
    w.save(work/'intake.json',manifest)
    with pytest.raises(ValueError,match='独立的图片坐标'):
        w.publish(work,work/'delivery-plan.json',tmp_path/'other-output',work/'structure-records',work/'structure-review.json')
    assert not (tmp_path/'other-output').exists()
