"""Source-only screening for opt-in review; never a geometry quality verdict."""
from pathlib import Path
import math

from . import source_body_reference as sbr

DISTANCE_RAW_PIXELS = 12


def inspect(raw_path, back_path, geometry_item):
    """Only a remote in-shaft alpha extension enables the new reference.

    The 12px distance comes from auto_geometry's remote support search. Reusing
    it as an applicability policy is new and does not certify body geometry.
    Outside-shaft extensions and smaller extensions are outside this screen.
    """
    import numpy as np
    from PIL import Image
    raw=sbr._binding(raw_path); back=sbr._binding(back_path)
    sbr._require(geometry_item.get('face')=='back' and
        geometry_item['source_sha256']==raw['sha256'] and
        geometry_item['output_sha256']==back['sha256'], 'Applicability source identity differs')
    geometry=geometry_item['geometry']
    with Image.open(back_path) as image:
        sbr._require(image.mode=='RGBA' and image.format=='PNG', 'Applicability requires immutable RGBA PNG')
        sbr._require(list(image.size)==geometry['output_size'], 'Applicability geometry dimensions differ')
        alpha=np.asarray(image)[...,3]
    h,w=alpha.shape
    audit=dict(status='source_tail_applicability_unknown', applicable=None,
        source=back,raw_source=raw,geometry_item_sha256=sbr._canonical(geometry_item),
        implementation_sha256=sbr._sha(__file__),source_reference_implementation_sha256=sbr._sha(sbr.__file__),
        policy=dict(distance_raw_pixels=DISTANCE_RAW_PIXELS,alpha_threshold=128,
            distance_source='phone_texture.auto_geometry: far_start / remote support limit + 12',
            original_geometry_implementation_sha256=sbr._sha(Path(__file__).with_name('auto_geometry.py')),
            new_applicability_policy=True,quality_certification=False,
            scope='within_source_shaft_only; outside-shaft or <=12px extension not assessed'),
        fit_uses_side_antenna_or_hardware=False,geometry_changed=False,
        reference_certified=False,visual_approved=False,publishable=False)
    if geometry.get('method')!=sbr.GEOMETRY_METHOD or any(geometry.get(k) is None for k in ('bottom_rail','alpha_lower_rail','body_x')):
        audit['reason']='No supported original source RGB/alpha rail pair'
        return audit
    # Generic mapping/identity errors remain hard. Missing bilateral semantic
    # provenance may keep applicability unknown, not create an acceptance.
    try:
        sbr._geometry_contract(geometry)
    except ValueError as error:
        if str(error).startswith(('Exposed two-flank rail evidence required',
                                  'Too few original exposed rail samples',
                                  'Raw RGB rail no longer agrees')):
            audit['reason']=str(error)
            return audit
        raise
    forward,_,mapping=sbr.coordinate_map(geometry)
    opaque=alpha>=128
    sample=opaque[int(h*.25):int(h*.65)]
    if not len(sample) or not sample.any(axis=1).all():
        audit['reason']='No complete source shaft rows'
        return audit
    left=int(np.median(np.argmax(sample,axis=1)))
    right=int(np.median(w-1-np.argmax(sample[:,::-1],axis=1)))
    if right-left<=16:
        audit['reason']='No usable independent source shaft width'
        return audit
    lines={}
    for name,key in (('rgb','bottom_rail'),('alpha','alpha_lower_rail')):
        rail=geometry[key]
        for distance in (0,DISTANCE_RAW_PIXELS):
            points=[forward(x+.5,rail[0]*(x+.5)+rail[1]+distance) for x in geometry['body_x']]
            sbr._require(abs(points[1][0]-points[0][0])>1,'Degenerate source applicability rail')
            slope=(points[1][1]-points[0][1])/(points[1][0]-points[0][0])
            intercept=points[0][1]-slope*points[0][0]
            sbr._require(math.isfinite(slope) and math.isfinite(intercept),'Nonfinite applicability rail')
            lines[name+('_remote' if distance else '')]=[float(slope),float(intercept)]
    xx=np.arange(left,right+1)[None,:];yy=np.arange(h)[:,None]
    rgb,matte=lines['rgb_remote'],lines['alpha_remote']
    tail=opaque[:,left:right+1]&(yy>rgb[0]*xx+rgb[1])&(yy>matte[0]*xx+matte[1])
    rows,cols=np.nonzero(tail); count=int(tail.sum())
    audit.update(status='source_remote_tail_detected' if count else 'no_in_shaft_remote_extension_detected',
        applicable=bool(count),shaft_x=[left,right],mapped_planes=lines,coordinate_map=mapping,
        remote_alpha128_pixel_count=count,
        remote_alpha128_bbox_xyxy=([int(cols.min()+left),int(rows.min()),int(cols.max()+left)+1,int(rows.max())+1] if count else None),
        remote_alpha128_row_counts=[[int(y),int(tail[y].sum())] for y in np.unique(rows)])
    sbr._require(sbr._sha(raw_path)==raw['sha256'] and sbr._sha(back_path)==back['sha256'],
                 'Applicability source changed during measurement')
    return audit
