from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from phone_texture import source_preservation as sp,automatic,cli



def exported(path, source, face, model='iphone17pro'):
    """2026-11-09: the back is resampled once to the model's standard canvas."""
    if face != 'back':
        return path.read_bytes() == source.read_bytes()
    from phone_texture.back_canvas import standardize, STANDARD_BACK_SIZE
    import tempfile
    from PIL import Image as _I
    expected = Path(tempfile.mkdtemp())/'back.png'; standardize(source, expected, model)
    return path.read_bytes() == expected.read_bytes() and _I.open(path).size == STANDARD_BACK_SIZE[model]

def test_unresolved_paper_material_scope_cannot_be_treated_as_effect_warning():
 with pytest.raises(ValueError,match='Bottom paper material boundary unresolved') as rejected:
  sp.require_material_boundary({'views':[]},{'source_paper_material_boundary':{'material_boundary_resolution':'unresolved_no_source_cross_edge'}})
 assert sp.failure_kind(rejected.value)=='material_classification'

def setup_case(tmp_path,monkeypatch):
 case=tmp_path/'case';files={};items=[]
 for i,face in enumerate(sp.pipeline.FACES):
  p=case/'work/geometry-records/staged/sku'/(face+'.png');p.parent.mkdir(parents=True,exist_ok=True)
  a=np.full((80,40,4),255,np.uint8);a[...,:3]=60+i*20;a[20:24,4:8,:3]=[150,20,60];Image.fromarray(a).save(p)
  files[face]=p;items.append(dict(face=face,output_sha256=sp.digest(p),geometry={'recorded_transform':'unchanged_test_coordinates'}))
 def no_supported_effect(*a,**k):raise ValueError('Unsupported editable illumination: intentional unsupported source')
 monkeypatch.setattr(sp.auto_material,'propose_job',no_supported_effect)
 return case,files,dict(candidates_complete=True,images=items)

def test_uncertain_antenna_exports_exact_geometry_pixels_without_claiming_alignment(tmp_path,monkeypatch):
 case,files,geometry=setup_case(tmp_path,monkeypatch)
 r=sp.preserve(case,'sku',geometry,files,{},'iphone17pro',ValueError('Opaque antenna endpoint lacks paired source evidence'),'original trace')
 assert r['six_faces_generated'] and not r['six_faces_processed']
 assert not r['quality_checks_complete'] and not r['structural_alignment_applied'] and not r['seam_accepted']
 for face,p in files.items():
  assert exported(case/'candidate/sku'/(face+'.png'),p,face)
  assert r['preserved_faces'][face]['RGB_changed_pixels']==0
 assert r['blocked_check']['traceback']=='original trace'
 assert r['frame_uv_seam_checks']['status']=='measured_only_not_uv_accepted'
 assert len(r['frame_uv_seam_checks']['observations'])==4

def test_unknown_failure_is_not_converted_to_an_approved_candidate(tmp_path,monkeypatch):
 case,files,geometry=setup_case(tmp_path,monkeypatch)
 with pytest.raises(ValueError,match='not an optional'):
  sp.preserve(case,'sku',geometry,files,{},'iphone17pro',ValueError('Original source hash changed'),'trace')
 assert not (case/'candidate').exists()

def test_changed_geometry_is_rejected_even_for_a_reviewable_failure(tmp_path,monkeypatch):
 case,files,geometry=setup_case(tmp_path,monkeypatch);Image.new('RGBA',(40,80),'red').save(files['left'])
 with pytest.raises(ValueError,match='Geometry source changed'):
  sp.preserve(case,'sku',geometry,files,{},'iphone17pro',ValueError('Opaque antenna endpoint lacks paired source evidence'),'trace')
 assert not (case/'candidate').exists()

def test_uncertain_antenna_does_not_reuse_modified_back_pixels(tmp_path,monkeypatch):
 case,files,geometry=setup_case(tmp_path,monkeypatch);light=case/'work/lighting/sku/back.png';light.parent.mkdir(parents=True);Image.new('RGBA',(40,80),'red').save(light)
 checks={'back':dict(source_sha256=sp.digest(files['back']),output_sha256=sp.digest(light),source_relative_effect_checks={'checked':True})}
 r=sp.preserve(case,'sku',geometry,files,checks,'iphone17pro',ValueError('Original independent source edges lack evidence'),'trace')
 assert r['preserved_faces']['back']['selected_sha256']==sp.digest(files['back'])
 assert exported(case/'candidate/sku/back.png',files['back'],'back')

def test_checked_lighting_cannot_change_alpha_or_position_in_fallback(tmp_path,monkeypatch):
 case,files,geometry=setup_case(tmp_path,monkeypatch);light=case/'work/lighting/sku/front.png';light.parent.mkdir(parents=True);Image.new('RGBA',(40,80),(60,60,60,100)).save(light)
 checks={'front':dict(source_sha256=sp.digest(files['front']),output_sha256=sp.digest(light),source_relative_effect_checks={'checked':True})}
 with pytest.raises(ValueError,match='changed source position or alpha'):
  sp.preserve(case,'sku',geometry,files,checks,'iphone17pro',ValueError('UV seam correction field discontinuity'),'trace')
 assert not (case/'candidate').exists()

def test_review_candidate_exports_with_distinct_incomplete_quality_state(tmp_path,monkeypatch):
 case,files,geometry=setup_case(tmp_path,monkeypatch);r=sp.preserve(case,'sku',geometry,files,{},'iphone17pro',ValueError('Opaque antenna endpoint lacks paired source evidence'),'trace');r['sku']='sku'
 work=tmp_path/'batch';work.mkdir();assert automatic.export_complete_candidates([r],work,tmp_path/'export')==['sku']
 assert len(list((tmp_path/'export/sku').glob('*.png')))==6
 assert not r['six_faces_processed']

def test_public_auto_cli_enables_explicit_source_preservation(monkeypatch,tmp_path):
 seen={}
 def run(*a,**k):seen.update(k);return dict(status='review_candidates_needs_visual_review',processed=0,generated=1,total=1)
 monkeypatch.setattr(automatic,'run',run)
 assert cli.main(['auto',str(tmp_path),'--model','iphone17pro'])==2
 assert seen['protection_policy']=='none' and seen['review_fallback']=='preserve-source'


def test_source_mutation_during_optional_processing_is_not_silently_preserved(tmp_path,monkeypatch):
 case,files,geometry=setup_case(tmp_path,monkeypatch)
 def corrupt(*a,**k):
  Image.new('RGBA',(40,80),'red').save(files['left'])
  raise ValueError('Unsupported editable illumination: test')
 monkeypatch.setattr(sp.auto_material,'propose_job',corrupt)
 with pytest.raises(ValueError,match='changed during optional'):
  sp.preserve(case,'sku',geometry,files,{},'iphone17pro',ValueError('Opaque antenna endpoint lacks paired source evidence'),'trace')
 assert not (case/'candidate').exists()


def test_mixed_frame_effect_that_worsens_source_seams_is_retained_for_review(tmp_path,monkeypatch):
 case,files,geometry=setup_case(tmp_path,monkeypatch);light=case/'work/lighting/sku/top.png';light.parent.mkdir(parents=True);Image.new('RGBA',(40,80),(255,0,0,255)).save(light)
 checks={'top':dict(source_sha256=sp.digest(files['top']),output_sha256=sp.digest(light),source_relative_effect_checks={'checked':True})}
 r=sp.preserve(case,'sku',geometry,files,checks,'iphone17pro',ValueError('UV seam correction field discontinuity'),'trace')
 assert not r['rejected_partial_effects']
 assert (case/'candidate/sku/top.png').read_bytes()==light.read_bytes()
 for face in ('left','right','bottom'):
  assert (case/'candidate/sku'/(face+'.png')).read_bytes()==files[face].read_bytes()
 assert not r['frame_uv_seam_checks']['source_baseline_nonregression']
 assert r['frame_uv_seam_checks']['lighting_retained_for_independent_review']
 assert not r['seam_accepted']


def test_unavailable_optional_front_fit_keeps_pixels_and_error_instead_of_blocking_six(tmp_path,monkeypatch):
 case,files,geometry=setup_case(tmp_path,monkeypatch)
 def no_front(*a,**k):raise ValueError('Front rim lacks nearby opaque material support')
 monkeypatch.setattr(sp.auto_material,'propose_job',no_front)
 r=sp.preserve(case,'sku',geometry,files,{},'iphone17pro',ValueError('Opaque antenna endpoint lacks paired source evidence'),'trace')
 assert r['six_faces_generated'] and not r['six_faces_processed']
 assert any('Front rim lacks' in e['error'] for e in r['additional_optional_failures'])
 for face,p in files.items():assert exported(case/'candidate/sku'/(face+'.png'),p,face)


@pytest.mark.parametrize('family',['single_edge','same_sign_edges','wrong_width'])
def test_rejected_source_pair_is_not_reconstructed_or_counted_as_aligned(family,tmp_path,monkeypatch):
 from phone_texture.antenna_material import trace_boundaries
 case,files,geometry=setup_case(tmp_path,monkeypatch)
 a=np.full((500,100,4),255,np.uint8);a[...,:3]=180;a[42:51,:,:3]=90;a[450:459,:,:3]=90
 a[:250,:12,:3]=180
 if family=='single_edge':a[42:250,:12,:3]=90
 elif family=='same_sign_edges':a[42:250,:12,:3]=120;a[51:250,:12,:3]=60
 else:a[42:61,:12,:3]=90
 with pytest.raises(ValueError) as failed:trace_boundaries(Image.fromarray(a),[42,51,450,459])
 Image.fromarray(a).save(files['left'])
 for item in geometry['images']:
  if item['face']=='left':item['output_sha256']=sp.digest(files['left'])
 r=sp.preserve(case,'sku',geometry,files,{},'iphone17pro',failed.value,'actual paired detector rejected input')
 assert r['six_faces_generated'] and not r['six_faces_processed'] and not r['mandatory_program_checks_complete']
 assert (case/'candidate/sku/left.png').read_bytes()==files['left'].read_bytes()
 assert not r['structural_alignment_applied'] and not r['seam_accepted']


def test_same_direction_central_edges_are_preserved_not_rebuilt_or_marked_aligned(tmp_path,monkeypatch):
 from phone_texture.antenna_material import trace_boundaries
 case,files,geometry=setup_case(tmp_path,monkeypatch)
 a=np.full((500,100,4),255,np.uint8);a[...,:3]=180;a[42:250,:,:3]=120;a[51:250,:,:3]=60;a[450:459,:,:3]=90
 with pytest.raises(ValueError,match='opposing central') as failed:trace_boundaries(Image.fromarray(a),[42,51,450,459])
 Image.fromarray(a).save(files['left'])
 for item in geometry['images']:
  if item['face']=='left':item['output_sha256']=sp.digest(files['left'])
 r=sp.preserve(case,'sku',geometry,files,{},'iphone17pro',failed.value,'actual same-direction central edges rejected')
 assert r['six_faces_generated'] and not r['six_faces_processed'] and not r['structural_alignment_applied']
 assert (case/'candidate/sku/left.png').read_bytes()==files['left'].read_bytes()
