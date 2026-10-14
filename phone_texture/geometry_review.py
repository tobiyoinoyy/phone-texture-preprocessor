"""Source-space geometry editing and mandatory, content-bound review evidence.

Review is an explicit agent/human decision. Hashes bind the decision to evidence;
they do not prove that segmentation or a manually drawn boundary is correct.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

CHECKS = ('source_overlay', 'four_corners', 'complete_lower_edge',
          'contact_before_rotation', 'contact_after_rotation', 'dark_light_backgrounds')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def read_mask(spec, size):
    path = Path(spec['path'])
    if digest(path) != spec['sha256']:
        raise ValueError('Geometry mask hash changed')
    with Image.open(path) as im:
        if im.size != size or im.mode != 'L' or im.format != 'PNG':
            raise ValueError('Geometry masks must be source-size grayscale PNGs')
        return np.array(im)


def guided_geometry(matte, face, guide, stage_sink=None):
    """Mask in SOURCE coordinates, rotate once, trim only transparent canvas.

    crop_xyxy is an enclosing work area, never an alternative to a support mask.
    Legacy rectangles that cut visible content now fail even if called reviewed.
    """
    from scripts.geometry_postprocess import rotate_alpha_safe
    src = np.array(matte.convert('RGBA'))
    h, w = src.shape[:2]
    keep = read_mask(guide['retain_mask'], (w,h)) if guide.get('retain_mask') else None
    remove = read_mask(guide['support_remove_mask'], (w,h)) > 0 if guide.get('support_remove_mask') else np.zeros((h,w),bool)
    protect = read_mask(guide['protect_mask'], (w,h)) > 0 if guide.get('protect_mask') else np.zeros((h,w),bool)
    if np.any(remove & protect) or (keep is not None and np.any(remove & (keep > 0))):
        raise ValueError('Support removal overlaps retained phone or protected paper')
    if keep is not None and np.any(protect & (keep == 0) & (src[...,3] > 0)):
        raise ValueError('Retained outline excludes protected content')
    work = src.copy()
    if keep is not None:
        work[...,3] = np.minimum(work[...,3], keep)
    work[...,3][remove] = 0
    work[protect] = src[protect]
    if stage_sink:
        stage_sink('source_matte', Image.fromarray(src))
        stage_sink('after_support', Image.fromarray(work))
    bounds = guide.get('crop_xyxy', [0,0,w,h])
    if len(bounds) != 4 or not all(type(v) is int for v in bounds):
        raise ValueError('Invalid enclosing crop coordinates')
    x0,y0,x1,y1 = bounds
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError('Enclosing crop outside source')
    outside = np.ones((h,w),bool);outside[y0:y1,x0:x1] = False
    retained = work[...,3] > 0
    if keep is not None:
        retained |= keep > 0
    if np.any(outside & retained):
        raise ValueError('Enclosing crop intersects retained phone contour; use a local support mask, not a crop line')
    angle = guide['rotation_ccw_degrees']
    if isinstance(angle,bool) or not np.isfinite(angle) or abs(angle)>4:
        raise ValueError('Reviewed rotation must be finite and within 4 degrees')
    # Rotate the full canvas, never a pre-clipped rectangle.
    rotated = Image.fromarray(rotate_alpha_safe(work, angle))
    if stage_sink:
        stage_sink('after_rotation', rotated)
    box = rotated.getchannel('A').getbbox()
    if box is None:
        raise ValueError('Empty retained phone contour')
    result = rotated.crop(box)
    if stage_sink:
        stage_sink('after_crop', rotated)
    if face in ('left','right'):
        result = result.transpose(Image.Transpose.ROTATE_90 if face=='left' else Image.Transpose.ROTATE_270)
    return result, {'method':'source_mask_geometry_v2', 'source_size':[w,h],
        'reviewed_source_crop':guide, 'canvas_trim_xyxy':list(box), 'rotation_ccw_degrees':angle,
        'output_size':list(result.size), 'resampling_passes':int(bool(angle)), 'rgb_painted':False,
        'review_reasons':['manual_geometry', 'complete_outline_requires_review'],
        'operation_order':['retain_outline','local_support_mask','rotate','transparent_canvas_trim','face_quarter_turn']}


def review_reasons(value):
    reasons = set()
    if isinstance(value,dict):
        reasons.update(value.get('review_reasons',[]))
        if value.get('support_applied'):
            reasons.add('heuristic_support_cut')
        if value.get('needs_visual_review'):
            reasons.add('extrapolated_frame_edges')
        if 'reviewed_source_crop' in value:
            reasons.add('manual_geometry')
        for item in value.values():
            if isinstance(item,(dict,list)):
                reasons.update(review_reasons(item))
    elif isinstance(value,list):
        for item in value:
            reasons.update(review_reasons(item))
    return sorted(reasons)


def background(im, color):
    canvas = Image.new('RGBA', im.size, color)
    return Image.alpha_composite(canvas, im.convert('RGBA')).convert('RGB')


def make_evidence(source, result, stages, folder):
    """Persist full-resolution boundaries, four corners and complete contact band.

    Source overlays are in source coordinates; post-rotation views are explicitly
    labelled in their own canvas. Full lower bands include all possible contacts,
    avoiding reliance on a potentially wrong heuristic contact-point estimate.
    """
    folder = Path(folder);folder.mkdir(parents=True,exist_ok=True)
    artifacts = []
    def save(im, name):
        path = folder/(name+'.png');im.save(path)
        artifacts.append({'path':str(path.resolve()),'sha256':digest(path)})
    raw = ImageOps.exif_transpose(Image.open(source)).convert('RGB')
    pre = stages.get('after_support', stages.get('source_matte'))
    if pre is None or pre.size != raw.size:
        raise ValueError('Source-space geometry evidence missing')
    a = pre.getchannel('A')
    boundary = np.array(a.filter(ImageFilter.MaxFilter(3))) != np.array(a.filter(ImageFilter.MinFilter(3)))
    overlay = np.array(raw);overlay[boundary] = [255,30,80]
    overlay = Image.fromarray(overlay)
    save(overlay,'source-boundary-overlay')
    save(pre,'after-support-source-canvas')
    post = stages.get('after_rotation', result)
    save(post,'after-rotation-canvas')
    save(stages.get('after_crop',post),'after-crop-canvas')
    # All four corners and the full lower edge, at 1x and 4x, both backgrounds.
    for label, image in [('source',overlay.convert('RGBA')), ('pre',pre), ('post',post), ('output',result)]:
        if label == 'source':
            box=a.getbbox()
        else:
            box=image.getchannel('A').getbbox()
        if not box:
            raise ValueError('Cannot review empty geometry')
        x0,y0,x1,y1=box
        rw=min(150,max(24,(x1-x0)//4));rh=min(150,max(24,(y1-y0)//4))
        rects={'top-left':(x0-4,y0-4,x0+rw,y0+rh),'top-right':(x1-rw,y0-4,x1+4,y0+rh),
               'bottom-left':(x0-4,y1-rh,x0+rw,y1+4),'bottom-right':(x1-rw,y1-rh,x1+4,y1+4),
               'complete-lower-edge':(x0-4,y1-rh,x1+4,y1+5)}
        for name,rect in rects.items():
            crop=image.crop(rect)
            for color,cname in [('#222222','dark'),('#eeeeee','light')]:
                rendered=background(crop,color)
                save(rendered,f'{label}-{name}-{cname}-1x')
                save(rendered.resize((rendered.width*4,rendered.height*4),Image.Resampling.NEAREST),f'{label}-{name}-{cname}-4x')
    # Compact contact sheet for discovery, not a replacement for the 1x/4x files.
    thumbs=[]
    for label,im in [('source',overlay),('before rotation',pre),('after rotation',post),('output',result)]:
        if label=='source': b=a.getbbox()
        else:b=im.getchannel('A').getbbox()
        thumb=background(im.convert('RGBA').crop((max(0,b[0]-6),max(0,b[3]-160),min(im.width,b[2]+6),min(im.height,b[3]+6))),'#eeeeee')
        thumb.thumbnail((1200,170))
        tile=Image.new('RGB',(1220,200),'#ddd');ImageDraw.Draw(tile).text((10,5),label,fill='black');tile.paste(thumb,(10,25));thumbs.append(tile)
    sheet=Image.new('RGB',(1220,800));
    for i,im in enumerate(thumbs):sheet.paste(im,(0,i*200))
    save(sheet,'contact-sheet')
    return artifacts


def verify_report(report_path, records=None):
    report_path=Path(report_path)
    report=json.loads(report_path.read_text())
    if report.get('geometry_status') != 'approved' or not report.get('geometry_review'):
        raise ValueError('Geometry review missing or unresolved; lighting constraints cannot approve geometry')
    review=report['geometry_review']
    if not review.get('reviewer') or not review.get('items'):
        raise ValueError('Geometry reviewer/items missing')
    entries=report['images']
    if not report.get('candidates_complete') or not entries:
        raise ValueError('Incomplete geometry candidates')
    if records is not None:
        expected={(sku,face) for sku,faces in records.items() for face in faces}
        if {(e['sku'],e['face']) for e in entries} != expected or len(entries)!=len(expected):
            raise ValueError('Geometry report must cover exactly all six faces')
    for entry in entries:
        key=entry['sku']+'/'+entry['face']
        checked=review['items'].get(key,{})
        if not checked.get('notes') or any(checked.get(k) is not True for k in CHECKS):
            raise ValueError('Incomplete geometry review: '+key)
        if digest(entry['source']) != entry['source_sha256']:
            raise ValueError('Geometry source changed after review')
        if records is not None and digest(records[entry['sku']][entry['face']]) != entry['output_sha256']:
            raise ValueError('Geometry output changed after review')
        artifacts=entry.get('geometry_evidence',[])
        required={'source-boundary-overlay.png','after-support-source-canvas.png','after-rotation-canvas.png','after-crop-canvas.png','contact-sheet.png'}
        required |= {f'{stage}-{part}-{bg}-{scale}x.png'
            for stage in ('source','pre','post','output')
            for part in ('top-left','top-right','bottom-left','bottom-right','complete-lower-edge')
            for bg in ('dark','light') for scale in (1,4)}
        if not required.issubset({Path(a['path']).name for a in artifacts}):
            raise ValueError('Geometry boundary evidence missing')
        for artifact in artifacts:
            if digest(artifact['path']) != artifact['sha256']:
                raise ValueError('Geometry evidence changed after review')
        verify_boundary_assertions(entry, checked)
    return report


def check_boundary_pixels(image, foreground, background_mask):
    """Independent, source-observed interior witnesses, NOT editing masks.

    Thresholds are fixed: confirmed solid body must remain >=128 alpha; confirmed
    support must be completely absent. Witnesses exclude uncertain mixed pixels.
    Full boundary/antialias judgement remains an explicit visual review.
    """
    a = np.asarray(image.convert('RGBA'))[...,3]
    fg, bg = foreground > 0, background_mask > 0
    if fg.shape != a.shape or bg.shape != a.shape or not fg.any() or not bg.any() or np.any(fg & bg):
        raise ValueError('Boundary witnesses must be nonempty, disjoint and match the image')
    lost = int(np.count_nonzero(fg & (a < 128)))
    residual = int(np.count_nonzero(bg & (a > 0)))
    if lost or residual:
        raise ValueError(f'Boundary regression: lost_body_pixels={lost}, residual_support_pixels={residual}')
    return {'lost_body_pixels':lost,'residual_support_pixels':residual,
            'body_witness_pixels':int(fg.sum()),'support_witness_pixels':int(bg.sum())}


def verify_boundary_assertions(entry, checked):
    risk = (set(entry.get('review_reasons',[])) | set(review_reasons(entry.get('geometry',{})))) & {
        'manual_geometry','heuristic_support_cut','ambiguous_body_edges','extrapolated_frame_edges'}
    assertions = checked.get('boundary_assertions',{})
    required = {'after-support-source-canvas.png','after-rotation-canvas.png','after-crop-canvas.png'} if risk else set()
    if not required.issubset(assertions):
        raise ValueError('Risky geometry requires independent body/support witnesses before and after rotation')
    artifacts={Path(a['path']).name:a['path'] for a in entry['geometry_evidence']}
    guide=entry.get('geometry',{}).get('reviewed_source_crop',{})
    editing_hashes={guide[k]['sha256'] for k in ('retain_mask','support_remove_mask','protect_mask') if k in guide}
    for name,spec in assertions.items():
        if name not in artifacts:
            raise ValueError('Unknown geometry boundary artifact')
        if any(spec[k]['sha256'] in editing_hashes for k in ('foreground','background')):
            raise ValueError('Boundary witnesses must be independent of editing/protection masks')
        with Image.open(artifacts[name]) as im:
            check_boundary_pixels(im,read_mask(spec['foreground'],im.size),read_mask(spec['background'],im.size))


def approve(process, review_path, output):
    from .pipeline import batch_files,publish,separate_paths,fresh
    process,output=separate_paths(process,output);fresh(output)
    path=process/'report.json';report=json.loads(path.read_text())
    review=json.loads(Path(review_path).read_text())
    if review.get('report_sha256')!=digest(path):
        raise ValueError('Review is stale for this candidate report')
    if not report.get('candidates_complete'):
        raise ValueError('Cannot approve an incomplete geometry batch')
    report['geometry_status']='approved';report['geometry_review']=review
    # Validate before publication; keep the pending report unchanged on failure.
    temp=process/'approval-check.json';write(temp,report)
    try:
        records=batch_files(process/'staged')
        for entry in report['images']:
            separate_paths(Path(entry['source']).parent,output)
        verify_report(temp,records)
        publish(process/'staged',output)
    finally:
        temp.unlink(missing_ok=True)
    report['success']=True;report['output']=str(output);write(path,report)
    return report
