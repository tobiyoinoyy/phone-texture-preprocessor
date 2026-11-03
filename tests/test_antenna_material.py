import numpy as np
import pytest
from PIL import Image, ImageDraw
from phone_texture.antenna_material import trace_boundaries
from phone_texture import auto_material, auto_structure, lighting, lighting_review


def crossover_photo():
    h,w=500,100;x=np.arange(w);metal=np.full(w,180.)
    metal[:18]=np.linspace(45,180,18);metal[-18:]=np.linspace(180,45,18)
    a=np.full((h,w,4),255,np.uint8);a[...,:3]=metal[None,:,None]
    a[42:51,:,:3]=90;a[450:459,:,:3]=90
    # Same-sign distractors cannot replace a jointly reversed material pair.
    a[443:446,-9:,:3]=135;a[461:464,:9,:3]=135
    return Image.fromarray(a)


def test_reversed_rim_requires_both_edges_and_preserves_band_width():
    image=crossover_photo();before=np.asarray(image).copy()
    rows,_,_=auto_structure.antenna_rows(image,[40,60]);lines,proof=trace_boundaries(image,rows)
    for index,target in enumerate((41.5,50.5,449.5,458.5)):
        assert np.max(abs(lines[index]-target))<1.5
    assert min(len(b['opposite_polarity_columns']) for b in proof['bands'])>5
    assert np.max(abs((lines[3]-lines[2])-9))<1.5
    np.testing.assert_array_equal(image,before)


@pytest.mark.parametrize('width',[3,8,12,23])
def test_missing_endpoint_is_rejected_instead_of_extrapolated_from_center(width):
    a=np.array(crossover_photo());a[:,:width,:3]=80
    with pytest.raises(ValueError,match='endpoint.*evidence'):
        trace_boundaries(Image.fromarray(a),[42,51,450,459])


def test_old_hash_bound_but_wrong_endpoint_material_is_rejected(tmp_path):
    image=crossover_photo();p=tmp_path/'right.png';image.save(p)
    rows,signs,proof=auto_structure.antenna_rows(image,[40,60]);lines,_=trace_boundaries(image,rows)
    lines[2,-10:]-=6
    cfg={'right':dict(source_sha256=auto_material.digest(p),source_lines=lines.tolist(),antenna_indices=[0,1,2,3],source_detection=proof)}
    with pytest.raises(ValueError,match='Frozen antenna boundaries disagree'):
        auto_material.propose_job(p,'right',tmp_path/'materials',landmarks=cfg)


def test_one_translucent_edge_cannot_hide_an_unsupported_opaque_endpoint():
    a=np.array(crossover_photo());a[:,:8,:3]=80;a[448:453,:8,3]=239
    with pytest.raises(ValueError,match='endpoint.*evidence'):
        trace_boundaries(Image.fromarray(a),[42,51,450,459],bands=(1,))


def test_normal_auto_scope_rejects_a_narrow_endpoint_block(tmp_path):
    im=crossover_photo();draw=ImageDraw.Draw(im)
    for lo,hi,color in [(150,240,(178,178,178,255)),(300,390,(75,75,75,255))]:
        draw.rounded_rectangle((34,lo,66,hi),radius=16,fill=color,outline=(35,35,35,255),width=2)
    path=tmp_path/'right.png';im.save(path)
    job,_,allowed,paper=auto_material.propose_job(path,'right',tmp_path/'materials')
    scope=auto_material.lighting_scope(path,'right',job,tmp_path/'scope')
    assert job['views'][0]['antenna_boundary_bands']==[0,1]
    assert {'antenna_1_left_metal','antenna_1_right_metal'}<=set(scope['local_checks'])
    engine=lighting.engine('aluminum');result,_,_,_=engine.process_view(job['views'][0],engine.target_color(job)[0],dict(job['defaults'],color_mode=job['color_mode']))
    output=tmp_path/'fixed.png';Image.fromarray(result).save(output)
    lighting_review.measure_local(scope,path,output,allowed,paper)
    result[435:444,2:8,:3]=np.maximum(0,result[435:444,2:8,:3].astype(int)-35)
    Image.fromarray(result).save(output)
    with pytest.raises(ValueError,match='antenna_1_left_metal'):
        lighting_review.measure_local(scope,path,output,allowed,paper)
