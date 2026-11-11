"""Programmatic processing can edit label RGB without inventing its geometry."""
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from phone_texture import auto_material,lighting
from phone_texture.automatic import no_zone_effects
from test_unmarked_paper_closure import source


def test_unresolved_protection_end_is_not_a_fit_only_material_failure():
 a,_,_,_=source(end=140);white=np.all(a[...,:3]==[220,225,230],axis=2);a[...,:3][white]=[140,145,150];image=Image.fromarray(a)
 with pytest.raises(ValueError):auto_material.paper_mask(image,'bottom')
 fit,proof=auto_material.paper_mask(image,'bottom',resolve_end=False)
 assert fit.any() and proof['usage']=='material_fit_exclusion_only'
 assert not proof['pixel_protection']


def test_programmatic_no_zone_edits_label_pixels_but_retains_source_and_alpha(tmp_path):
 a,_,_,_=source(end=140);original=a.copy();yy=np.arange(len(a))[:,None,None]
 outside=np.all(a[...,:3]==[25,35,58],axis=2)
 gradient=np.broadcast_to(np.rint(np.array([70,90,120])[None,None,:]*(.65+.5*yy/len(a))),(a.shape[0],a.shape[1],3)).astype('uint8')
 a[...,:3][outside]=gradient[outside];path=tmp_path/'bottom.png';Image.fromarray(a).save(path)
 job,proof,allowed,paper=auto_material.propose_job(path,'bottom',tmp_path/'materials',protection_policy='none')
 assert not paper.any() and proof['protected_zone_pixels']==0
 # Bottom paper is lit with the frame metal; its rough location is only a fit exclusion.
 label,_=auto_material.paper_mask(Image.fromarray(a),'bottom',resolve_end=False)
 assert label.any() and proof['paper']['method']=='bottom_paper_lit_with_frame_metal'
 assert np.all(allowed[label])
 module=lighting.engine('aluminum');target,_=module.target_color(job);out,_,protected,report=module.process_view(job['views'][0],target,dict(job['defaults'],color_mode='source_lightness'))
 assert not protected.any();np.testing.assert_array_equal(out[...,3],a[...,3]);assert np.any(np.any(out[...,:3]!=a[...,:3],axis=2)&label)
 np.testing.assert_array_equal(np.array(Image.open(path)),a)
 assert job['views'][0]['protection_policy']=='none' and report['protected_changed']==0


def test_no_zone_policy_rejects_a_hidden_legacy_protection_mask(tmp_path):
 a=np.full((32,32,4),255,np.uint8);a[...,:3]=150;path=tmp_path/'source.png';Image.fromarray(a).save(path)
 mask=tmp_path/'protected.png';Image.fromarray(np.ones((32,32),np.uint8)*255).save(mask)
 item=dict(source=str(path),protection_policy='none',protected_mask=str(mask),domains=[])
 with pytest.raises(ValueError,match='No-zone'):lighting.engine('aluminum').process_view(item,np.zeros(3),{})


def test_unknown_policy_is_rejected_before_processing(tmp_path):
 a,_,_,_=source();p=tmp_path/'bottom.png';Image.fromarray(a).save(p)
 with pytest.raises(ValueError,match='Unknown'):auto_material.propose_job(p,'bottom',tmp_path/'materials',protection_policy='invalid')
