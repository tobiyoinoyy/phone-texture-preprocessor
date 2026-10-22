"""Optional fixed-model image-coordinate gate; never claims live Blender proof.

Landmarks are independently reviewed on both the accepted reference and the new
images. This gate detects coordinate mismatches, not semantic landmarks.
"""
from pathlib import Path
import numpy as np
from PIL import Image

from .geometry_review import digest, write
from .pipeline import FACES
from .workspace import read

TOLERANCE = 2.5  # measured in accepted reference-image pixels
CHECKS = ('body_outline', 'materials_and_details', 'native_and_4x', 'level_side_antennas')


def load(path):
    path = Path(path).expanduser().resolve()
    profile = read(path)
    if profile.get('schema') != 'phone-model-uv-reference-v1':
        raise ValueError('需要固定模型坐标清单，不是旧版 --uv-profile 配置')
    if set(profile['reference_images']) != set(FACES):
        raise ValueError('固定模型参考必须包含六面')
    glb = path.parent / profile['model']['glb']
    if digest(glb) != profile['model']['glb_sha256']:
        raise ValueError('固定模型 GLB 与参考指纹不一致')
    folder = Path(profile['reference_texture_directory'])
    for face, entry in profile['reference_images'].items():
        image = folder / entry['file']
        if digest(image) != entry['sha256'] or list(Image.open(image).size) != entry['size']:
            raise ValueError('固定模型参考图片发生变化：' + face)
    expected = {f: f + '.png' for f in FACES}
    expected.update({'back-camera': 'back.png', 'backdetal': 'back.png'})
    if profile.get('material_image_mapping') != expected:
        raise ValueError('固定模型六图材质映射不完整')
    return profile


def required_points(profile, face):
    labels = ['body:' + str(i) for i in range(4)]
    if face in ('front', 'back', 'left', 'right'):
        rows = profile['antenna_rows'][face]
        if isinstance(rows, dict):
            labels += ['antenna:' + end + ':' + str(i) for end in ('image_left', 'image_right') for i in range(4)]
        else:
            labels += ['antenna:' + end + ':' + str(i) for end in ('left', 'right') for i in range(4)]
    if face == 'back':
        labels += ['lens:' + str(i) for i in range(len(profile['lens_centers_back_px']))]
    for name, button in profile.get('button_outline_boxes_px', {}).items():
        if button['face'] == face:
            labels += ['button:' + name + ':' + corner for corner in ('tl', 'br')]
    if face in ('top', 'bottom'):
        labels += ['port:' + str(i) for i in range(2)]
    return labels


def template(files, profile_path, output):
    profile = load(profile_path)
    output = Path(output)
    if output.exists():
        raise ValueError('模型复核模板已存在，请另选文件')
    value = dict(profile_sha256=digest(profile_path), reviewer='', items={})
    for sku, faces in files.items():
        value['items'][sku] = {}
        for face, image in faces.items():
            labels = required_points(profile, face)
            value['items'][sku][face] = dict(output_sha256=digest(image), notes='',
                **dict.fromkeys(CHECKS, False), reference_points=dict.fromkeys(labels),
                output_points=dict.fromkeys(labels), evidence=[])
    write(output, value)
    return value


def verify(files, profile_path, review_path):
    profile = load(profile_path); review = read(review_path)
    if review.get('profile_sha256') != digest(profile_path) or not review.get('reviewer'):
        raise ValueError('固定模型复核缺失或已过期')
    if set(review['items']) != set(files):
        raise ValueError('固定模型复核 SKU 不一致')
    for sku, faces in files.items():
        if set(review['items'][sku]) != set(FACES):
            raise ValueError('固定模型须复核完整六面')
        for face, image in faces.items():
            item = review['items'][sku][face]
            if item['output_sha256'] != digest(image) or not item.get('notes') or any(item.get(c) is not True for c in CHECKS):
                raise ValueError('固定模型逐面复核不完整或成品已改变')
            labels = required_points(profile, face)
            points = []
            for key, size in [('reference_points', profile['reference_images'][face]['size']),
                              ('output_points', Image.open(image).size)]:
                if set(item[key]) != set(labels):
                    raise ValueError('固定模型结构锚点缺项')
                xy = np.asarray([item[key][name] for name in labels], float)
                if xy.shape != (len(labels), 2) or not np.isfinite(xy).all() or np.any(xy < 0) or np.any(xy >= np.asarray(size)):
                    raise ValueError('固定模型结构锚点无效')
                points.append((xy + .5) / np.asarray(size))
            reference = np.asarray(profile['reference_images'][face]['size'])
            errors = np.abs((points[0] - points[1]) * reference)
            if errors.max() > TOLERANCE:
                raise ValueError(f'固定模型结构位置不匹配：{sku}/{face}，请先配准图片')
            # The coordinate inventory itself supplies known targets; a reviewer
            # cannot shift both lists together to make a mismatch pass.
            rp = item['reference_points']
            for name in labels:
                target = None
                if name.startswith('lens:'):
                    target = profile['lens_centers_back_px'][int(name.split(':')[1])]
                elif name.startswith('button:'):
                    _, button, corner = name.split(':')
                    box = profile['button_outline_boxes_px'][button]['box']
                    target = box[:2] if corner == 'tl' else box[2:]
                if target is not None and np.max(np.abs(np.asarray(rp[name]) - target)) > TOLERANCE:
                    raise ValueError('复核锚点不符合固定坐标清单')
                if name.startswith('antenna:'):
                    _, end, index = name.split(':'); rows = profile['antenna_rows'][face]
                    row = rows[end][int(index)] if isinstance(rows, dict) else rows[int(index)]
                    if abs(rp[name][1] - row) > TOLERANCE:
                        raise ValueError('天线目标不符合固定坐标清单')
            if face in ('left', 'right'):
                op = item['output_points']
                if any(abs(op['antenna:left:' + str(i)][1] - op['antenna:right:' + str(i)][1])
                       * reference[1] / Image.open(image).height > TOLERANCE for i in range(4)):
                    raise ValueError('固定模型侧面天线须横向平直')
            evidence = item.get('evidence', [])
            if {e.get('scale') for e in evidence} != {1, 4}:
                raise ValueError('固定模型复核需要原尺寸和 4 倍证据')
            for e in evidence:
                if digest(e['path']) != e['sha256']:
                    raise ValueError('固定模型视觉证据已改变')
    return dict(status='reference_image_coordinates_passed', profile_path=str(Path(profile_path).resolve()),
                profile_sha256=digest(profile_path), review_path=str(Path(review_path).resolve()),
                review_sha256=digest(review_path), model_uv_verified=False)


def verify_proof(files, reference, proof):
    if not proof or proof.get('profile_sha256') != reference['sha256'] or digest(reference['path']) != reference['sha256']:
        raise ValueError('固定模型需要独立的图片坐标复核，图片空间对齐不能替代')
    actual = verify(files, reference['path'], proof['review_path'])
    if actual != proof:
        raise ValueError('固定模型复核证明发生变化')
