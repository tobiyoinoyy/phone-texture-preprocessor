"""Build image-space side antenna targets from reviewed front/back landmarks.

Front/back silhouette coordinates are measured independently. This does not
claim UV seam validation; button anchors remain independently supplied.
"""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image


def rows_from_view(view, height, body_y):
    p=Path(view['path'])
    if hashlib.sha256(p.read_bytes()).hexdigest()!=view['sha256']:
        raise ValueError('Front/back reference image changed')
    image_h=Image.open(p).height
    ends=np.asarray(view['body_y'],float)
    rows=np.asarray(view['antenna_rows'],float)
    target=np.asarray(body_y,float)
    if ends.shape!=(2,) or rows.shape!=(4,) or target.shape!=(2,) or not np.isfinite(np.r_[ends,rows,target]).all():
        raise ValueError('Need finite body endpoints and four antenna boundaries')
    if not 0<=ends[0]<ends[1]<image_h or not 0<=target[0]<target[1]<height:
        raise ValueError('Body endpoints outside image')
    if np.any(np.diff(rows)<=0) or rows[0]<=ends[0] or rows[-1]>=ends[1]:
        raise ValueError('Antenna boundaries must be ordered inside the reviewed body')
    return target[0]+(rows-ends[0])/(ends[1]-ends[0])*(target[1]-target[0])


def build_targets(guide, height, width):
    if width<2 or guide.get('front_at') not in ('left','right'):
        raise ValueError('Explicit side orientation required')
    f=rows_from_view(guide['front'],height,guide['side_body_y'])
    b=rows_from_view(guide['back'],height,guide['side_body_y'])
    t=np.linspace(0,1,width)
    if guide['front_at']=='right':t=1-t
    antenna=f[:,None]*(1-t)+b[:,None]*t
    buttons=np.asarray(guide.get('button_rows',[]),float)
    if buttons.ndim!=1 or not np.isfinite(buttons).all():raise ValueError('Invalid independent button anchors')
    if buttons.size and not guide.get('button_basis'):
        raise ValueError('Button targets need an independent reviewed basis')
    lines=np.concatenate([antenna,np.repeat(buttons[:,None],width,axis=1)],axis=0)
    lines=lines[np.argsort(lines[:,0])]
    if np.any(np.diff(lines,axis=0)<=0) or np.any(lines<=0) or np.any(lines>=height-1):
        raise ValueError('Front/back/button targets cross; review pose and markings')
    return lines.tolist()
