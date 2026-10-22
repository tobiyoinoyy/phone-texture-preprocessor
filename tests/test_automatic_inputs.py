"""Original-only proposals: physical features, rejection, and source invariants."""
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import pytest
from phone_texture.auto_material import paper_mask,reference_box
from phone_texture.auto_structure import antenna_rows,per_column_boundaries,hardware_ranges
from phone_texture.auto_geometry import frame_rails,consensus_line
from phone_texture.automatic import measure_masks,run


def test_printed_paper_is_separate_from_touching_bright_frame_and_wear():
    a=np.full((120,700,4),255,np.uint8);a[...,:3]=[25,35,58]
    a[:3,:,:3]=220  # Specular metal rail touches the real label.
    a[:,290:330,:3]=220
    for y in range(12,110,10):a[y:y+4,298:322,:3]=25
    a[70:76,500:510,:3]=225  # An isolated pale wear mark is not paper.
    paper,proof=paper_mask(Image.fromarray(a),'bottom')
    assert proof['components']
    assert paper[30,310] and paper[0,310] and paper[119,310]
    assert not paper[0,100] and not paper[72,505]
    assert not paper[:,260].any() and not paper[:,350].any()


def test_unprinted_flash_or_highlight_is_not_pixel_protection():
    a=np.full((240,160,4),255,np.uint8);a[...,:3]=[60,65,80]
    a[160:220,70:88,:3]=230
    mask,_=paper_mask(Image.fromarray(a),'back')
    assert not mask.any()


def test_antenna_pairs_require_two_opaque_supported_ends():
    a=np.full((500,80,4),255,np.uint8);a[...,:3]=170
    a[42:49,:,:3]=65;a[450:457,:,:3]=65
    rows,signs,_=antenna_rows(Image.fromarray(a),[30,50])
    assert np.max(np.abs(np.array(rows)-[41.5,48.5,449.5,456.5]))<=1
    assert signs==[-1,1,-1,1]
    lines=per_column_boundaries(Image.fromarray(a),rows)
    assert lines.shape==(4,80) and np.all(np.diff(lines,axis=0)>2)
    a[430:,:,3]=0
    with pytest.raises(ValueError):antenna_rows(Image.fromarray(a),[30,50])


def test_blank_or_unidentified_hardware_does_not_get_invented():
    image=Image.new('RGBA',(80,500),(100,110,120,255))
    with pytest.raises(ValueError,match='hardware'):hardware_ranges(image)


def test_reference_is_entirely_in_current_opaque_material():
    a=np.full((70,40,4),255,np.uint8);a[...,:3]=130
    mask=np.zeros((70,40),bool);mask[10:60,4:8]=True
    box=reference_box(a,mask);x0,y0,x1,y1=box
    assert mask[y0:y1,x0:x1].all() and (x1-x0)*(y1-y0)>=20
    with pytest.raises(ValueError):reference_box(a,np.zeros(mask.shape,bool))


def test_lighting_cannot_relabel_changed_paper_or_geometry_as_complete(tmp_path):
    a=np.full((20,20,4),255,np.uint8);a[...,:3]=100
    src=tmp_path/'source.png';out=tmp_path/'output.png';Image.fromarray(a).save(src)
    allowed=np.ones((20,20),bool);paper=np.zeros_like(allowed);paper[4:8,4:8]=True;allowed[paper]=False
    a[5,5,0]+=1;Image.fromarray(a).save(out)
    with pytest.raises(ValueError,match='paper'):measure_masks(src,out,allowed,paper)
    a[5,5,0]-=1;a[10,10,3]=200;Image.fromarray(a).save(out)
    with pytest.raises(ValueError,match='geometry'):measure_masks(src,out,allowed,paper)


def test_30_is_not_silently_accepted_as_30_independent_model_processes(tmp_path):
    src=tmp_path/'input';src.mkdir()
    with pytest.raises(ValueError,match='1–4'):run(src,tmp_path/'process',tmp_path/'out',jobs=30)
    assert not (tmp_path/'process').exists() and not (tmp_path/'out').exists()


def test_independent_raw_photo_is_required_for_rail_validation():
    a=np.zeros((200,400,4),np.uint8);a[80:110,20:380]=255
    with pytest.raises(ValueError):frame_rails(a)


def test_source_edge_consensus_rejects_unrelated_edges_and_resists_glints():
    rng=np.random.default_rng(17);x=np.arange(240,dtype=float)
    y=.015*x+80+rng.normal(0,.25,len(x));y[::8]+=20
    line,keep,error=consensus_line(x,y,2.5,.08)
    assert error<1 and keep.mean()>.8
    assert np.max(np.abs(np.polyval(line,[0,239])-np.array([80,83.585])))<.2
    with pytest.raises(ValueError):consensus_line(x,rng.uniform(0,150,len(x)),2.5,.08)


def test_label_tail_is_not_replaced_by_an_oversized_convex_protection_hull():
    a=np.full((260,400,4),255,np.uint8);a[...,:3]=[230,130,65]
    a[150:215,145:190,:3]=220;a[215:250,158:178,:3]=220
    for y in range(160,205,8):a[y:y+3,152:182,:3]=20
    for y in range(220,245,6):a[y:y+2,162:173,:3]=20
    mask,_=paper_mask(Image.fromarray(a),'back')
    assert mask[230,167]
    assert not mask[230,150] and not mask[230,185]


def test_bottom_label_does_not_freeze_metal_below_its_actual_end():
    a=np.full((120,700,4),255,np.uint8);a[...,:3]=[25,35,58]
    a[:95,290:330,:3]=220
    for y in range(12,76,10):a[y:y+4,298:322,:3]=25
    cv2=pytest.importorskip('cv2');cv2.circle(a,(323,82),2,(205,40,35,255),-1)
    mask,proof=paper_mask(Image.fromarray(a),'bottom')
    assert mask[90,310] and not mask[100,310]
    assert 93<=proof['source_paper_end']<=96


def test_qr_seed_recovers_paper_on_pale_metal_without_freezing_neighbours():
    cv2=pytest.importorskip('cv2')
    if not hasattr(cv2,'QRCodeEncoder_create'):pytest.skip('OpenCV QR encoder unavailable')
    a=np.full((800,400,4),255,np.uint8);a[...,:3]=[220,220,235]
    a[480:680,140:260,:3]=[199,200,217];a[680:790,172:223,:3]=[199,200,217]
    a[495:545,160:230,:3]=[210,50,45]
    qr=cv2.resize(cv2.QRCodeEncoder_create().encode('original-only-paper-regression'),(90,90),interpolation=cv2.INTER_NEAREST)
    block=np.zeros((90,90,3),np.uint8);block[:]=[199,200,217];block[qr<128]=20
    a[565:655,155:245,:3]=block
    for y in range(690,780,8):a[y:y+3,180:215,:3]=20
    mask,proof=paper_mask(Image.fromarray(a),'back')
    assert proof['cue']=='detected_QR_quadrilateral'
    assert mask[600,200] and mask[720,200]
    assert not mask[:,100].any() and not mask[730,150]


def test_front_rim_reaches_a_tightly_cropped_canvas_without_editing_glass():
    from phone_texture.auto_material import front_metal
    # Four different photographed widths, plus a white inner glass reflection.
    a=np.zeros((900,450,4),np.uint8);a[:]=[170,90,45,255]
    a[12:-10,15:-14,:3]=[12,15,20]
    a[12:-10,20:23,:3]=[195,195,200]
    mask,proof=front_metal(Image.fromarray(a))
    assert mask[-1,100:350].all() and mask[0,100:350].all()
    assert mask[300:600,0].all() and mask[300:600,-1].all()
    assert not mask[100:800,23:425].any()
    assert len(proof['edge_measurements_px'])==4


def test_source_corner_geometry_removes_local_stand_and_preserves_paper_tail():
    from phone_texture.auto_geometry import plate_geometry
    body=Image.new('L',(400,620));ImageDraw.Draw(body).rounded_rectangle((70,20,330,540),radius=35,fill=255)
    raw=Image.new('RGB',body.size,(230,230,230));raw.paste((190,100,45),(0,0,400,620),body)
    matte=raw.convert('RGBA');alpha=np.array(body);alpha[530:,180:250]=255
    # Paper crossing the lower edge is independent from the stand.
    alpha[520:570,150:166]=255;paper=np.zeros(alpha.shape,bool);paper[520:570,150:166]=True
    matte.putalpha(Image.fromarray(alpha));out,evidence=plate_geometry(matte,'back',paper,source_image=raw)
    stages={};out,evidence=plate_geometry(matte,'back',paper,lambda n,i:stages.update({n:i.copy()}),source_image=raw)
    cut=np.array(stages['after_support'])
    assert cut[560:610,180:250,3].sum()==0
    assert np.all(cut[545:568,150:166,3]==255)
    assert np.array_equal(cut[30:525,:,3],alpha[30:525])
    assert evidence['own_corner_evidence']['raw_rgb_supported_fraction']>=.6


def test_plate_angle_uses_the_long_body_axis_even_with_a_slanted_stand():
    from phone_texture.auto_geometry import plate_axis,principal
    from scipy import ndimage
    body=Image.new('L',(400,620));ImageDraw.Draw(body).rounded_rectangle((70,20,330,540),radius=35,fill=255)
    raw=Image.new('RGB',body.size,(230,230,230));raw.paste((190,100,45),(0,0,400,620),body)
    matte=raw.convert('RGBA');alpha=np.array(body);stand=Image.fromarray(alpha)
    ImageDraw.Draw(stand).polygon([(175,530),(255,552),(255,619),(175,619)],fill=255)
    matte.putalpha(stand);a=np.array(matte);m=principal(a[...,3]);rows=np.arange(80,480)
    slope,proof=plate_axis(a,np.array(raw),rows)
    assert abs(proof['rotation_ccw_degrees'])<.05
    assert abs(slope)<.001


def test_extreme_concurrency_is_explicit_and_bounded_before_loading_models(tmp_path):
    from phone_texture.automatic import concurrency_limits
    assert concurrency_limits(4)==2
    assert concurrency_limits(10,2,True)==2
    assert concurrency_limits(4,3,True)==3
    for jobs,slots,experimental in [(6,None,False),(4,3,False),(10,5,True),(4,5,True),(30,2,True),(True,1,True)]:
        with pytest.raises(ValueError):concurrency_limits(jobs,slots,experimental)
    source=tmp_path/'source';source.mkdir()
    with pytest.raises(ValueError):run(source,tmp_path/'work',tmp_path/'out',jobs=10)
    assert not (tmp_path/'work').exists() and not (tmp_path/'out').exists()


def test_front_rim_ignores_stronger_translucent_outer_shoulder():
    from phone_texture.auto_material import front_metal
    # White background bleed at a soft alpha boundary has a stronger falling
    # gradient than the actual opaque metal/glass interface farther inside.
    a=np.zeros((900,450,4),np.uint8);a[:]=[170,90,45,255]
    a[16:-10,15:-14,:3]=[12,15,20]
    a[:6,:,:3]=255;a[6,:,:3]=80
    for y,alpha in enumerate([0,1,8,60,130,200,246,253]):a[y,:,3]=alpha
    mask,proof=front_metal(Image.fromarray(a))
    assert mask[10,100:350].all()  # Visible metal is not lost to the false edge.
    assert not mask[25:800,30:420].any()  # Glass must not enter the metal field.
    assert proof['opaque_boundary_support'][2]==1


def test_front_rim_rejects_boundary_without_opaque_material():
    from phone_texture.auto_material import front_metal
    a=np.zeros((900,450,4),np.uint8);a[:]=[170,90,45,230]
    a[16:-10,15:-14,:3]=[12,15,20]
    with pytest.raises(ValueError,match='opaque support'):
        front_metal(Image.fromarray(a))


def test_rear_panel_uses_source_seam_and_curved_corners_not_height_cut():
    from phone_texture.auto_material import rear_panel
    image=Image.new('RGBA',(1000,2100),(180,110,65,255));draw=ImageDraw.Draw(image)
    draw.rounded_rectangle((45,15,955,610),radius=100,outline=(20,20,20),width=3)
    draw.rounded_rectangle((63,700,937,2040),radius=125,fill=(205,145,95),outline=(25,25,25),width=3)
    panel,proof=rear_panel(image)
    assert abs(proof['bounds'][1]-700)<=3
    assert panel[750,500] and not panel[650,500]
    assert not panel[710,70] and panel[850,70]
    assert not panel[2030,70] and panel[2000,500]
    assert all(abs(c['radius']-125)<8 for c in proof['corners'])


def test_rear_panel_rejects_source_without_supported_material_seams():
    from phone_texture.auto_material import rear_panel
    with pytest.raises(ValueError,match='Rear panel'):
        rear_panel(Image.new('RGBA',(600,1200),(180,110,65,255)))
