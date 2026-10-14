import hashlib
import numpy as np
from PIL import Image
import pytest
from texture_alignment.image_reference_targets import build_targets
from texture_alignment.align_textures import warp_rows


def guide(tmp):
    p=tmp/'front.png';q=tmp/'back.png'
    Image.new('RGBA',(20,201),'orange').save(p);Image.new('RGBA',(20,401),'orange').save(q)
    def view(p,body,rows):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),body_y=body,antenna_rows=rows)
    return dict(front=view(p,[0,200],[20,24,180,184]),back=view(q,[0,400],[48,56,352,360]),
                side_body_y=[0,100],front_at='left',button_rows=[30,40],button_basis='Reviewed button template')


def test_both_edges_follow_own_reference_and_buttons_stay_horizontal(tmp_path):
    g=guide(tmp_path);a=np.array(build_targets(g,101,5))
    assert np.allclose(a[0],[10,10.5,11,11.5,12])
    assert np.allclose(a[1,[0,-1]],[12,14])
    assert np.all(a[2]==30) and np.all(a[3]==40)
    assert np.allclose(a[-2,[0,-1]],[90,88])
    g['front_at']='right';assert np.array_equal(build_targets(g,101,5),a[:,::-1])


def test_warp_matches_both_texture_edges_without_color_generation(tmp_path):
    g=guide(tmp_path);target=build_targets(g,101,5)
    source=[15,18,30,40,85,89]
    a=np.zeros((101,5,4),np.uint8);a[...,3]=255
    for y in range(101):a[y,:,:3]=[y,y+1,y+2]
    b=np.array(warp_rows(Image.fromarray(a),source,target))
    assert np.array_equal(b[10,0],a[15,0]) and np.array_equal(b[12,-1],a[15,-1])
    assert np.array_equal(b[30],a[30]) and np.array_equal(b[40],a[40])


@pytest.mark.parametrize('fault',['hash','orientation','crossing','bounds','button_basis'])
def test_bad_reference_rejected(tmp_path,fault):
    g=guide(tmp_path)
    if fault=='hash':g['front']['sha256']='old'
    elif fault=='orientation':g.pop('front_at')
    elif fault=='crossing':g['button_rows']=[11]
    elif fault=='bounds':g['back']['body_y']=[0,500]
    else:g.pop('button_basis')
    with pytest.raises(ValueError):build_targets(g,101,5)


def test_image_reference_pipeline_applies_without_scene_record(tmp_path):
    import json
    from phone_texture import pipeline
    source=tmp_path/'input'/'SKU_01';source.mkdir(parents=True)
    for face in pipeline.FACES:Image.new('RGBA',(20,32),'orange').save(source/(face+'.png'))
    work=tmp_path/'work';pipeline.prepare(source.parent,work,reference='SKU_01',height=64,width=10)
    g=guide(tmp_path)
    for face,rows in [('front',[3,4,27,28]),('back',[4,5,26,27])]:
        p=source/(face+'.png');g[face]=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),body_y=[0,31],antenna_rows=rows)
    g.update(side_body_y=[0,63],button_rows=[24,32])
    config=dict(reference='SKU_01',height=64,alignment_basis='front_back_images',
        points={'SKU_01':{s:[6,9,24,32,54,57] for s in ('left','right')}},
        targets_by_sku={'SKU_01':{}},image_reference_guides={'SKU_01':{}},
        reference_files=[{k:g[f][k] for k in ('path','sha256')} for f in ('front','back')])
    import copy
    for side in ('left','right'):
        own=copy.deepcopy(g);own['front_at']='left' if side=='left' else 'right'
        config['image_reference_guides']['SKU_01'][side]=own
        config['targets_by_sku']['SKU_01'][side]=build_targets(own,64,10)
    p=tmp_path/'landmarks.json';p.write_text(json.dumps(config))
    report=pipeline.apply(work,p,tmp_path/'output',tmp_path/'records')
    assert report['uv_validation']=='not_checked_image_space_only'
    assert len(list((tmp_path/'output/SKU_01').glob('*.png')))==6
    config['targets_by_sku']['SKU_01']['left'][0][0]+=1;p.write_text(json.dumps(config))
    with pytest.raises(ValueError,match='Targets differ'):
        pipeline.apply(work,p,tmp_path/'bad-output',tmp_path/'bad-records')
    assert not (tmp_path/'bad-output').exists()
