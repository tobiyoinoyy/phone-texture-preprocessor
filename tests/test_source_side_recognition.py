from pathlib import Path
from unittest.mock import patch
from phone_texture import view_recognition as vr
from phone_texture.source_side_recognition import classify_canonical
from PIL import Image


def test_complete_hardware_left_and_reverse():
 def parts(image,face):
  if face=='right':raise ValueError('Missing two complete components')
  boxes=[[30,200,60,280],[30,350,60,490],[30,520,60,660],[30,1200,60,1380]]
  if image.getpixel((0,0))[0]:boxes=[[x,1600-y1,x1,1600-y] for x,y,x1,y1 in boxes[::-1]]
  return [(None,dict(box=b)) for b in boxes]
 with patch('phone_texture.source_side_recognition.hardware_materials',parts):
  assert classify_canonical(Image.new('RGBA',(100,1600),(0,0,0,255)))['face']=='left'
  assert classify_canonical(Image.new('RGBA',(100,1600),(1,0,0,255)))['face'] is None


def test_blank_has_no_complete_side_evidence():
 assert classify_canonical(Image.new('RGBA',(100,1600),(150,150,150,255)))['face'] is None


def test_unchanged_atlas_acceptance_does_not_load_model():
 old=dict(face='left',suggested_face='left')
 with patch.object(vr,'_recognize_atlas',return_value=old),patch('phone_texture.source_side_recognition.recognize_source_side') as source:
  assert vr.recognize(Path('arbitrary'))==old
  source.assert_not_called()


def test_weak_coarse_silhouette_cannot_enter_source_fallback():
 old=dict(face=None,suggested_face='right',scores=dict(left=.7,right=.8))
 with patch.object(vr,'_recognize_atlas',return_value=old),patch('phone_texture.source_side_recognition.recognize_source_side') as source:
  assert vr.recognize(Path('arbitrary'))['face'] is None
  source.assert_not_called()


def test_ambiguous_side_requires_unique_complete_hardware_proof():
 old=dict(face=None,suggested_face='right',scores=dict(left=.9,right=.91))
 with patch.object(vr,'_recognize_atlas',return_value=old.copy()),patch('phone_texture.source_side_recognition.recognize_source_side',return_value=dict(face=None)):
  assert vr.recognize(Path('random-file'))['face'] is None
 with patch.object(vr,'_recognize_atlas',return_value=old.copy()),patch('phone_texture.source_side_recognition.recognize_source_side',return_value=dict(face='left')):
  result=vr.recognize(Path('different-file'))
  assert result['face']=='left' and result['method']=='source_complete_ordered_hardware'
