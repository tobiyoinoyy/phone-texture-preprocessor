from pathlib import Path
import json, hashlib
import numpy as np
from PIL import Image,ImageDraw
from phone_texture.geometry_review import digest,write,guided_geometry,check_boundary_pixels
from phone_texture.pipeline import process_batch
from scripts.geometry_postprocess import rotate_alpha_safe
from phone_normalizer.uv import normalize_face

import argparse
parser=argparse.ArgumentParser(description='Local real-photo regression: masks, boundary witnesses, all six faces. Does not approve/publish.')
parser.add_argument('--source',type=Path,required=True,help='Six named original JPEGs')
parser.add_argument('--mattes',type=Path,required=True,help='Cached full-canvas BiRefNet_dynamic PNGs')
parser.add_argument('--previous',type=Path,required=True,help='Previous six-face geometry for unchanged-face comparison')
parser.add_argument('--process',type=Path,required=True,help='Fresh regression directory')
args=parser.parse_args()
ROOT=args.process.resolve();ROOT.mkdir(parents=True,exist_ok=False)
REPO=Path(__file__).resolve().parents[1]
SOURCE=args.source.resolve();MATTES=args.mattes.resolve();PREVIOUS=args.previous.resolve()
CONFIG=json.loads((REPO/'tests/fixtures/iphone17promax-boundaries.json').read_text())
(ROOT/'masks').mkdir()
# Cache visible pixels must retain source RGB; transparent RGB may be zeroed. All bytes are recorded.
provenance={}
for face in ('front','back','left','right','top','bottom'):
 src=Image.open(SOURCE/(face+'.jpg')).convert('RGB');matte=Image.open(MATTES/(face+'.png')).convert('RGBA')
 if src.size!=matte.size or not np.array_equal(np.asarray(src)[np.asarray(matte)[...,3]>0],np.asarray(matte)[...,:3][np.asarray(matte)[...,3]>0]):
  raise ValueError('Cached matte is not from the unchanged source: '+face)
 provenance[face]={'source_sha256':digest(SOURCE/(face+'.jpg')),'matte_sha256':digest(MATTES/(face+'.png'))}
write(ROOT/'input-provenance.json',provenance)

def save_mask(a,name):
 p=ROOT/'masks'/(name+'.png');Image.fromarray(a.astype(np.uint8)).save(p)
 return {'path':str(p),'sha256':digest(p)}

def witnesses(size,rects):
 im=Image.new('L',size)
 for x0,y0,x1,y1 in rects:ImageDraw.Draw(im).rectangle((x0,y0,x1-1,y1-1),fill=255)
 return np.array(im)

def rotate_mask(a,angle):
 rgba=np.zeros((*a.shape,4),np.uint8);rgba[...,:3]=255;rgba[...,3]=a
 # Keep only unambiguous solid witness interiors after resampling.
 return np.where(rotate_alpha_safe(rgba,angle)[...,3]>=254,255,0).astype(np.uint8)

class CachedBiRefNet:
 name='birefnet';model_name='ZhengPeng7/BiRefNet_dynamic';device='cached-source-bound'
 def __init__(self):
  self.cache={hashlib.sha256(Image.open(p).convert('RGB').tobytes()).hexdigest():p.stem for p in SOURCE.glob('*.jpg')}
 def segment(self,image):
  face=self.cache[hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()]
  return Image.open(MATTES/(face+'.png')).copy()
 def close(self):pass

if __name__=='__main__':
 guides={};audit={}
 for face in ('top','front','back'):
  c=CONFIG[face];p=SOURCE/(face+'.jpg');assert digest(p)==c['source_sha256']
  matte=Image.open(MATTES/(face+'.png'));w,h=matte.size
  points=np.array(c['support_boundary']);xs=np.arange(w);ys=np.arange(h)[:,None]
  boundary=np.interp(xs,points[:,0],points[:,1]);domain=(xs>=points[0,0])&(xs<=points[-1,0])
  # Subpixel coverage at the original photograph's contact; no global erosion.
  cut=np.clip(boundary[None,:]-ys+.5,0,1)
  retain=np.full((h,w),255,np.uint8);retain[:,domain]=np.rint(cut[:,domain]*255).astype(np.uint8)
  remove=(domain[None,:]&(ys>boundary[None,:]+.5)).astype(np.uint8)*255
  protect=np.zeros((h,w),np.uint8)
  if 'paper_polygon' in c:
   paper=Image.fromarray(protect);ImageDraw.Draw(paper).polygon([tuple(v) for v in c['paper_polygon']],fill=255);protect=np.array(paper)
   retain[protect>0]=255;remove[protect>0]=0
  if 'frame_upper_boundary' in c:
   pts=np.array(c['frame_upper_boundary']);line=np.interp(xs,pts[:,0],pts[:,1]);upper=np.clip(ys-line[None,:]+.5,0,1)
   retain=np.minimum(retain,np.rint(upper*255).astype(np.uint8))
  guide={'source_sha256':digest(p),'rotation_ccw_degrees':c['angle'],'retain_mask':save_mask(retain,face+'-retain'),'support_remove_mask':save_mask(remove,face+'-remove')}
  if protect.any():guide['protect_mask']=save_mask(protect,face+'-paper')
  guides[face]=guide
 write(ROOT/'geometry-guides.json',{'iphone17promax':guides})
 backend=CachedBiRefNet()
 import shutil
 batch=ROOT/'input'/'iphone17promax';batch.mkdir(parents=True)
 for face in provenance:shutil.copy2(SOURCE/(face+'.jpg'),batch/(face+'.jpg'))
 report=process_batch(batch.parent,ROOT/'geometry-approved',ROOT/'candidates',mode='standardize',frame_guides=ROOT/'geometry-guides.json',backend_instance=backend)
 for face in ('top','front','back'):
  c=CONFIG[face];size=Image.open(SOURCE/(face+'.jpg')).size
  fg=witnesses(size,c['body_witnesses']);bg=witnesses(size,c['support_witnesses'])
  checks={};stage_dir=ROOT/'candidates/geometry-review/iphone17promax'/face
  for stage in ('after-support-source-canvas','after-rotation-canvas','after-crop-canvas'):
   a,b=(fg,bg) if stage.startswith('after-support') else (rotate_mask(fg,c['angle']),rotate_mask(bg,c['angle']))
   out=Image.open(stage_dir/(stage+'.png'))
   checks[stage]=check_boundary_pixels(out,a,b)
  # Old automatic cut is stable but wrong: witness tests must reject it.
  if face in ('front','back'):
   oldstages={};normalize_face(Image.open(SOURCE/(face+'.jpg')),face,backend=backend,stage_sink=lambda n,v:oldstages.update({n:Image.fromarray(v) if isinstance(v,np.ndarray) else v}))
   try:check_boundary_pixels(oldstages['after_support'],fg,bg)
   except ValueError as exc:checks['old_failure_detected']=str(exc)
   else:raise AssertionError('Old stand residue was not detected')
  else:
   # Reproduce original pre-rotation + rectangle on original-space witnesses.
   old=Image.fromarray(rotate_alpha_safe(np.array(Image.open(MATTES/'top.png')),c['angle']))
   clipped=Image.new('RGBA',old.size);clipped.paste(old.crop((180,774,1410,896)),(180,774))
   try:check_boundary_pixels(clipped,rotate_mask(fg,c['angle']),rotate_mask(bg,c['angle']))
   except ValueError as exc:checks['old_failure_detected']=str(exc)
   else:raise AssertionError('Old top over-crop was not detected')
  audit[face]=checks
 for face in ('left','right','bottom'):
  a=PREVIOUS/(face+'.png');b=ROOT/'candidates/staged/iphone17promax'/(face+'.png')
  assert np.array_equal(np.array(Image.open(a)),np.array(Image.open(b))),face
  audit[face]={'rgba_exact_match_to_previous_geometry':True,'sha256':digest(b)}
 write(ROOT/'regression-results.json',audit)
 print(json.dumps(audit,indent=2))
