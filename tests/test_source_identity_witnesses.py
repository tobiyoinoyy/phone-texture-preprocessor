import cv2
import numpy as np
import pytest
from phone_texture.source_plate_recognition import camera_triad
from phone_texture.source_cap_recognition import classify


def plate(kind='back',tone=150):
 a=np.full((384,384,3),230,np.uint8);a[40:360,128:264]=tone
 if kind=='front':a[55:345,135:257]=15
 elif kind!='empty':
  centers=[(160,80),(160,125),(205,102)]
  if kind=='mirror':centers=[(384-x,y) for x,y in centers]
  for i,c in enumerate(centers):
   if kind=='missing' and i==2:continue
   cv2.circle(a,c,13,(10,10,10),-1)
 return a

@pytest.mark.parametrize('tone',[90,150,220])
def test_source_three_lenses_are_visible_at_multiple_thresholds(tone):
 p=camera_triad(plate(tone=tone),np.array([[1,0,0],[0,1,0]],np.float32))
 assert p['accepted'] and len(p['consistent_thresholds'])>=2

@pytest.mark.parametrize('kind',['front','empty','mirror','missing'])
def test_absence_wrong_order_or_missing_lens_is_not_back_evidence(kind):
 assert not camera_triad(plate(kind),np.array([[1,0,0],[0,1,0]],np.float32))['accepted']


def cap(kind='bottom'):
 a=np.full((800,800,3),220,np.uint8);a[370:440,110:690]=90
 if kind=='top':a[355:370,170:650]=90
 elif kind not in ['plain','no_port']:
  cv2.rectangle(a,(385,392),(455,415),(10,10,10),-1)
 if kind not in ['top','plain']:
  for x in [240,260,280,300,320]+([] if kind=='one_group' else [510,530,550,570,590]):cv2.circle(a,(x,402),6,(10,10,10),-1)
 return a

@pytest.mark.parametrize('kind',['bottom','top'])
def test_cap_has_positive_source_hardware_or_projection_evidence(kind):
 assert classify(cap(kind))['face']==kind

@pytest.mark.parametrize('kind',['plain','no_port','one_group'])
def test_cap_does_not_infer_missing_features_by_elimination(kind):
 assert classify(cap(kind))['face'] is None


def test_rotated_or_blank_cap_does_not_form_horizontal_body_witness():
 assert classify(np.rot90(cap()))['face'] is None
 assert classify(np.full((800,800,3),255,np.uint8))['face'] is None
