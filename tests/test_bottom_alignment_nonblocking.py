import importlib.util,json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from phone_texture import bottom_alignment as ba
from phone_texture.geometry_review import digest

def source_fixture(tmp_path):
 from port_source_fixture import fixture
 a,_=fixture('ip17pro');files={}
 for f in ['front','back','left','right','top','bottom']:
  p=tmp_path/(f+'.png');Image.fromarray(a).save(p);files[f]=p
 geo={'candidates_complete':True,'images':[{'face':f,'output_sha256':digest(p),'geometry':{}} for f,p in files.items()]}
 return a,files,geo

def test_actual_alignment_updates_provenance_without_changing_input(tmp_path):
 a,files,geo=source_fixture(tmp_path);old=json.dumps(geo,sort_keys=True)
 selected,effective,r=ba.prepare(files,geo,'iphone17pro',tmp_path/'process')
 assert r['applied'] and r['blocks_generation'] is False and r['status']=='applied_with_notice'
 assert json.dumps(geo,sort_keys=True)==old
 assert effective['images'][-1]['output_sha256']==digest(selected['bottom'])!=digest(files['bottom'])
 assert all(digest(selected[f])==digest(files[f]) for f in files if f!='bottom')
 assert r['integrity']['endcaps_exact']

@pytest.mark.parametrize('fault',['reject','fold','crop','stretch','endcap'])
def test_bad_transform_keeps_exact_original(tmp_path,monkeypatch,fault):
 a,files,geo=source_fixture(tmp_path)
 def bad(image,model):
  if fault=='reject':raise ba.Rejected('port_layer_uncertain')
  h,w=image.shape[:2];out=image.copy();mx=np.broadcast_to(np.arange(w,dtype='float32'),(h,w)).copy()
  if fault=='fold':mx[:,w//2:w//2+3]=mx[:,w//2:w//2+3][:,::-1]
  if fault=='crop':out=out[:,:-1]
  if fault=='stretch':mx[:,w//2]+=3
  if fault=='endcap':out[0,0,0]^=255
  return out,{'action':'corrected_needs_visual_review'},mx,np.ones((h,w),bool)
 monkeypatch.setattr(ba,'rectify',bad)
 selected,_,r=ba.prepare(files,geo,'iphone17pro',tmp_path/'process')
 assert r['status']=='kept_source_with_notice' and not r['blocks_generation']
 assert r['notices'] and r['fallback_byte_identical']
 assert digest(selected['bottom'])==digest(files['bottom'])

def test_unknown_model_does_not_guess_profile(tmp_path):
 _,files,geo=source_fixture(tmp_path);_,_,r=ba.prepare(files,geo,'unknown',tmp_path/'process')
 assert r['reason']=='unsupported_model_no_inferred_profile' and r['fallback_byte_identical']

def test_explicit_off_keeps_all_files_and_no_warning(tmp_path):
 _,files,geo=source_fixture(tmp_path);selected,_,r=ba.prepare(files,geo,'iphone17pro',tmp_path/'process',enabled=False)
 assert r['status']=='disabled' and not r['notices'] and not r['applied']
 assert all(digest(selected[f])==digest(files[f]) for f in files)
