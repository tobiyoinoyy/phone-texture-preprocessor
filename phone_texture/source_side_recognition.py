"""Independent complete hardware geometry, no filenames or class elimination."""
from phone_texture.side_material import hardware_materials

def classify_canonical(image):
 accepted=[];proof={}
 for face in ['left','right']:
  try:
   parts=hardware_materials(image,face);boxes=[p[1]['box'] for p in parts];heights=[b[3]-b[1] for b in boxes];widths=[b[2]-b[0] for b in boxes];centers=[(b[1]+b[3])/2/image.height for b in boxes]
   if face=='left':
    valid=(len(parts)==4 and max(centers[:3])<.55 and centers[3]>.60 and heights[0]<.8*min(heights[1:3]) and .8<heights[1]/heights[2]<1.25 and heights[3]>1.5*heights[0] and centers[3]-centers[2]>2*(centers[2]-centers[1]))
   else:
    # All raw photos use the same canonical 90deg rotation; right sides
    # therefore put the shorter observed camera capsule above the longer power key.
    # Require a difference exceeding both caps localization-error budgets.
    valid=(len(parts)==2 and .20<centers[0]<.55 and .55<centers[1]<.80 and heights[1]-heights[0]>4*max(1.5,image.width*.015) and .7<heights[0]/heights[1]<1.4)
   proof[face]=dict(accepted=bool(valid),boxes=boxes,widths=widths,heights=heights,centers=centers)
   if valid:accepted.append(face)
  except ValueError as e:proof[face]=dict(accepted=False,reason=str(e))
 return dict(face=accepted[0] if len(accepted)==1 else None,proof=proof,method='complete_capsule_count_and_ordered_hardware_geometry')


def recognize_source_side(path, device='auto'):
 """A bounded source proof for atlas-ambiguous studio sides.

 All measurements come from the current photo. This never uses a filename,
 missing-class elimination or incoming samples as atlas references.
 """
 from pathlib import Path
 import hashlib
 from PIL import Image, ImageOps
 from phone_normalizer.api import Normalizer, NormalizerConfig
 from .auto_geometry import frame_geometry
 path=Path(path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
 with Image.open(path) as opened:raw=ImageOps.exif_transpose(opened).convert('RGB')
 with Normalizer(NormalizerConfig(remove_support=False,deskew=False,tight_crop=False,device=device)) as normalizer:
  matte,_=normalizer.normalize(raw)
 canonical,geometry=frame_geometry(matte,'left',source_image=raw)
 result=classify_canonical(canonical)
 result.update(source_sha256=digest,geometry=geometry,
  source_orientation='common_90_degree_rotation; complete ordered capsules required')
 if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
  raise ValueError('Source changed during complete side recognition')
 return result
