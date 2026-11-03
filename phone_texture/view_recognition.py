"""Conservative image-only view recognition for the validated studio capture.

The atlas contains raw-image multiscale edge descriptors, not masks, calibrations or
processed textures. It never uses the incoming filename, EXIF label or SKU ID.
Low support, duplicate faces and missing views are rejected, not assigned by
elimination. Similarity scores are not calibrated probabilities.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from PIL import Image, ImageOps

FACES = ('front', 'back', 'left', 'right', 'top', 'bottom')
DATA = Path(__file__).with_name('recognition_data')
PROFILE = 'iphone17-studio-views-v3-outline-registration'


def read_rgb(path):
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert('RGB')
        im.thumbnail((800, 800))
        return np.array(im)


def edge_feature(rgb):
    a = cv2.resize(rgb, (192, 192), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    g = cv2.createCLAHE(2., (8, 8)).apply(g)
    g = cv2.GaussianBlur(g, (5, 5), 1).astype(np.float32)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    mag = cv2.GaussianBlur(np.minimum(mag, 100), (9, 9), 2)
    # This is the capture profile, not a crop or mask applied to output images.
    # Downweight the support and printed paper in the same region for all views.
    mag[140:] = 0; mag[80:, 83:109] = 0
    vector = cv2.resize(mag, (48, 48)).ravel()
    return vector / (np.linalg.norm(vector) + 1e-8)


def side_feature(rgb):
    # Preserve fine button/antenna edges while excluding the lower support.
    # Coordinates belong to the validated studio framing, not a body crop.
    image = cv2.resize(rgb, (384, 384), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.createCLAHE(2., (8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 1).astype(np.float32)
    magnitude = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0),
                              cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    magnitude = cv2.GaussianBlur(np.minimum(magnitude, 100), (5, 5), 1)
    magnitude[:100] = 0
    magnitude[230:] = 0
    magnitude[160:, 166:218] = 0
    vector = cv2.resize(magnitude, (96, 96)).ravel()
    return vector / (np.linalg.norm(vector) + 1e-8)


@lru_cache(maxsize=1)
def atlas():
    meta = json.loads((DATA/'atlas.json').read_text())
    with np.load(DATA/'atlas.npz', allow_pickle=False) as z:
        arrays = {k:z[k] for k in z.files}
    return meta, arrays


def hardware_feature(feature):
    """Independent interior-button witness within the studio framing.

    Excludes the long silhouette rails and stand dominating whole-side scores.
    Uses the existing frozen descriptors, without adding incoming samples.
    """
    grid = feature.reshape(96, 96)
    vector = grid[48:52, 16:80].ravel().copy()
    return vector / (np.linalg.norm(vector) + 1e-8)


def _plate_regions():
    """Disjoint pose and identity witnesses in the frozen studio atlas space."""
    y,x=np.indices((48,48))
    # Leave a gap even under the bounded candidate transforms: the camera/
    # screen witness cannot enter the cap/rail estimator's sampling footprint.
    outer=((((x>=11)&(x<19))|((x>=29)&(x<37)))&(y>=23)&(y<33))
    outer |= (y>=3)&(y<7)&(x>=19)&(x<29)
    interior=(y>=12)&(y<19)&(x>=18)&(x<30)
    return outer,interior


def _outline_registration(feature, references):
    """Estimate pose using only the cap and long rails, never class scores.

    One common transform is evaluated against every class afterwards. The
    existing front descriptors provide an unoccluded silhouette; no incoming
    image becomes a reference and camera/screen pixels cannot select the fit.
    """
    outer,_=_plate_regions()
    target=np.median(np.stack(references),axis=0).reshape(48,48)[outer]
    norm=float(np.linalg.norm(target))
    if norm<=1e-8:raise ValueError('Frozen atlas lacks plate outline support')
    target=target/norm;grid=np.asarray(feature).reshape(48,48)
    best=(-1.,None,None)
    def consider(scale,dx,dy):
        nonlocal best
        if not .90<=scale<=1.10 or abs(dx)>3 or abs(dy)>3:return
        matrix=np.array([[scale,0,23.5*(1-scale)+dx],
                         [0,scale,23.5*(1-scale)+dy]],np.float32)
        aligned=cv2.warpAffine(grid,matrix,(48,48),flags=cv2.INTER_LINEAR)
        values=aligned[outer];support=float(np.linalg.norm(values))
        score=float(values@target/(support+1e-8))
        if score>best[0]:best=(score,(float(scale),float(dx),float(dy)),matrix)
    for scale in np.linspace(.90,1.10,11):
        for dx in np.linspace(-3,3,13):
            for dy in np.linspace(-3,3,13):consider(scale,dx,dy)
    _,(scale,dx,dy),_=best
    for s in np.linspace(scale-.01,scale+.01,5):
        for x in np.linspace(dx-.375,dx+.375,7):
            for y in np.linspace(dy-.375,dy+.375,7):consider(s,x,y)
    score,(scale,dx,dy),matrix=best
    proof=dict(outline_score=score,scale=scale,translation_cells=[dx,dy],
        descriptor_size=48,scale_bounds=[.90,1.10],translation_bound_cells=3.,
        estimation='frozen_front_outer_cap_and_rails_only')
    if score<.90:raise ValueError('Outer silhouette support is insufficient')
    # A boundary optimum can indicate that the actual pose lies outside the
    # declared capture tolerance; do not accept a clipped best-effort fit.
    if min(scale-.90,1.10-scale)<1e-6 or max(abs(dx),abs(dy))>=3-1e-6:
        raise ValueError('Plate framing lies outside bounded registration')
    return matrix,proof


def _plate_source_support(rgb,matrix,face):
    """Independently require visible closure, plus front glass in source RGB."""
    small=cv2.resize(rgb,(48,48),interpolation=cv2.INTER_AREA).astype(np.float32)
    aligned=cv2.warpAffine(small,matrix,(48,48),flags=cv2.INTER_LINEAR)
    valid=cv2.warpAffine(np.ones((48,48),np.float32),matrix,(48,48),flags=cv2.INTER_LINEAR)>.999
    gray=aligned.mean(axis=2);gy=np.abs(np.gradient(gray,axis=0))
    # The atlas intentionally suppresses the stand/lower label. Verify the
    # lower body on both sides of that label from the uncensored source RGB.
    bottoms=[]
    for lo,hi in ((17,20),(28,31)):
        # The same studio profile places the lower corners in rows 42..45.
        # Earlier edges can be a truncated body; padded warp borders are not
        # source evidence and must not compete with the actual lower edge.
        rows=[row for row in range(42,46) if valid[row-1:row+2,lo:hi].all()]
        if not rows:
            raise ValueError('Plate bottom closure is unsupported or clipped')
        signal=np.array([np.median(gy[row,lo:hi]) for row in rows])
        row=rows[int(np.argmax(signal))]
        if signal.max()<4:
            raise ValueError('Plate bottom closure is unsupported or clipped')
        bottoms.append(row)
    if abs(bottoms[0]-bottoms[1])>2:
        raise ValueError('Plate lower corners do not support one upright body')
    proof=dict(bottom_edge_rows=bottoms)
    if face=='front':
        glass=gray[12:31,18:30];background=np.concatenate([gray[12:31,3:10].ravel(),gray[12:31,38:45].ravel()])
        if not valid[12:31,3:45].all():raise ValueError('Front glass witness leaves source image')
        contrast=float(np.median(background)-np.median(glass))
        fraction=float(np.mean(glass<np.median(background)-40))
        if contrast<40 or fraction<.90:
            raise ValueError('Outer outline lacks independent dark screen evidence')
        proof.update(screen_background_contrast=contrast,dark_screen_fraction=fraction)
    return proof


def register_plate(rgb,feature,refs,arrays):
    """Conservative framing fallback; unchanged atlas/score/margin gates."""
    proof=dict(accepted=False,scope='upright front/back studio framing only')
    try:
        matrix,fit=_outline_registration(feature,[arrays[r['id']+'_edge'] for r in refs if r['face']=='front'])
        proof.update(fit)
        aligned=cv2.warpAffine(np.asarray(feature).reshape(48,48),matrix,(48,48),flags=cv2.INTER_LINEAR).ravel()
        aligned=aligned/(np.linalg.norm(aligned)+1e-8)
        scores={face:max(float(aligned@arrays[r['id']+'_edge']) for r in refs if r['face']==face) for face in FACES}
        ranked=sorted(scores,key=scores.get,reverse=True);face=ranked[0]
        proof.update(scores=scores,face=face,score=scores[face],margin=scores[face]-scores[ranked[1]])
        if face not in ('front','back') or scores[face]<.90 or proof['margin']<.025:
            raise ValueError('Registered view fails unchanged class score/margin gates')
        _,interior=_plate_regions();inside=aligned.reshape(48,48)[interior]
        energy=float(np.linalg.norm(inside))
        relevant=[r for r in refs if r['face']==face]
        reference_energy=[float(np.linalg.norm(arrays[r['id']+'_edge'].reshape(48,48)[interior])) for r in relevant]
        if energy<min(reference_energy)*.5:
            raise ValueError('Plate interior has insufficient independent structure')
        inside=inside/(energy+1e-8)
        interior_scores={f:max(float(inside@(arrays[r['id']+'_edge'].reshape(48,48)[interior]/
            (np.linalg.norm(arrays[r['id']+'_edge'].reshape(48,48)[interior])+1e-8)))
            for r in refs if r['face']==f) for f in ('front','back')}
        proof.update(interior_scores=interior_scores,interior_energy=energy)
        other='back' if face=='front' else 'front'
        if interior_scores[face]<.90 or interior_scores[face]-interior_scores[other]<.025:
            raise ValueError('Independent plate interior disagrees with registered outline')
        proof['source_support']=_plate_source_support(rgb,matrix,face)
        proof['accepted']=True
    except ValueError as exc:
        proof['reason']=str(exc)
    return proof


def recognize(path, exclude_group=None):
    rgb = read_rgb(path); feature = edge_feature(rgb)
    meta, arrays = atlas()
    refs = [r for r in meta['references'] if r['group'] != exclude_group]
    scores = {face:max(float(feature @ arrays[r['id']+'_edge']) for r in refs if r['face']==face) for face in FACES}
    ranked = sorted(scores, key=scores.get, reverse=True)
    best = ranked[0]; score = scores[best]; margin = score-scores[ranked[1]]
    result = dict(face=None, suggested_face=best, score=round(score,5), margin=round(margin,5),
                  method='edge_atlas', profile=PROFILE, scores=scores)
    if score < .90:
        if best in ('front','back'):
            registration=register_plate(rgb,feature,refs,arrays)
            result['registration']=registration
            if registration['accepted']:
                result.update(face=registration['face'],suggested_face=registration['face'],
                    method='outline_registered_plate',raw_scores=scores,
                    scores=registration['scores'],score=round(registration['score'],5),
                    margin=round(registration['margin'],5),reason=None)
                return result
        result['reason'] = '画面与当前拍摄规范差异较大'; return result
    if best in ('left', 'right'):
        # Fine edges resolve buttons/antenna layout that a coarse silhouette
        # loses. Compare both orientations to reject mirrored side photos.
        feature = side_feature(rgb)
        mirror = side_feature(rgb[:, ::-1].copy())
        detail = {face:max(float(feature @ arrays[r['id']+'_side']) for r in refs if r['face']==face)
                  for face in ('left', 'right')}
        mirrored = max(float(mirror @ arrays[r['id']+'_side']) for r in refs if r['face'] in ('left','right'))
        order = sorted(detail, key=detail.get, reverse=True)
        a,b = detail[order[0]],detail[order[1]]
        result.update(method='multiscale_side_edges', side_scores=detail, mirror_score=mirrored)
        if a-mirrored < .015:
            hardware = hardware_feature(feature); flipped = hardware_feature(mirror)
            fine = {face:max(float(hardware @ hardware_feature(arrays[r['id']+'_side']))
                            for r in refs if r['face']==face) for face in ('left','right')}
            flipped_score = max(float(flipped @ hardware_feature(arrays[r['id']+'_side']))
                                for r in refs if r['face'] in ('left','right'))
            result['hardware_orientation'] = dict(scores=fine, mirror_score=flipped_score)
            # Only resolve a near tie with a second, agreeing pixel witness.
            # A mirror winning either comparison remains rejected.
            if (a <= mirrored or max(fine,key=fine.get) != order[0]
                    or fine[order[0]] < .90 or fine[order[0]]-flipped_score < .01):
                result['reason'] = '疑似镜像或侧面方向不确定'; return result
        if a < .90 or a-b < .035:
            result['reason'] = '左右侧面特征不足'; return result
        best = order[0]
    elif margin < .025:
        result['reason'] = '方向识别不确定'; return result
    result.update(face=best, reason=None)
    return result


def inspect_images(paths):
    images = []; errors = []
    if len(paths) != 6:
        return dict(images=[], complete=False, errors=[f'需要 6 张原图，当前 {len(paths)} 张'])
    for path in sorted(paths):
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            proof = recognize(path)
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError('Image changed during recognition')
            images.append(dict(file=str(path), name=path.name, sha256=digest, **proof))
            if proof['face'] is None:
                errors.append(path.name+'：'+proof['reason'])
        except (OSError, ValueError, cv2.error):
            errors.append(path.name+'：图片无法识别')
    found = [r['face'] for r in images if r['face']]
    duplicates = [f for f in FACES if found.count(f)>1]
    if duplicates:
        errors.append('同一方向出现多张：'+', '.join(duplicates))
    complete = not errors and set(found)==set(FACES) and len(found)==6
    if not complete and not errors:
        errors.append('未识别出完整六面')
    return dict(images=images, complete=complete, errors=errors)
