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
PROFILE = 'iphone17-studio-views-v2-hardware-orientation'


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
