"""One entry for three stages; publish only complete six-face batches.

Staging, diagnostic images and metadata always live outside deliverables.
The original geometry engines are retained, not reimplemented.
"""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import json
import math
import shutil

import numpy as np
from PIL import Image, ImageDraw

from texture_alignment import align_textures as alignment

FACES = ('front', 'back', 'left', 'right', 'top', 'bottom')
FRAMES = ('left', 'right', 'top', 'bottom')
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff', '.bmp'}


def batch_files(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'输入目录不存在：{root}')
    folders = sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith('.'))
    if not folders:
        raise ValueError('输入应为包含 SKU 子文件夹的批次目录')
    records = {}
    for folder in folders:
        faces = {}
        for path in folder.iterdir():
            face = path.stem.lower()
            if path.is_file() and path.suffix.lower() in EXTENSIONS and face in FACES:
                if face in faces:
                    raise ValueError(f'{folder.name}/{face} 存在多份同名图片')
                faces[face] = path
        missing = set(FACES) - set(faces)
        if missing:
            raise ValueError(f'{folder.name} 缺少六面图：{", ".join(sorted(missing))}')
        for path in faces.values():
            with Image.open(path) as image:
                image.verify()
        records[folder.name] = faces
    return records


def separate_paths(*paths):
    paths = [Path(p).expanduser().resolve() for p in paths]
    for i, first in enumerate(paths):
        for second in paths[i + 1:]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError('输入、成品、过程目录必须互相独立，不能互相包含')
    return paths


def fresh(path):
    if path.exists() or path.is_symlink():
        raise ValueError(f'请使用新的目录，避免覆盖：{path}')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def publish(staged, output):
    """Validate all six faces before creating the deliverable directory."""
    records = batch_files(staged)
    fresh(output)
    created = False
    try:
        output.mkdir(parents=True)
        created = True
        for sku, faces in records.items():
            (output / sku).mkdir()
            for path in faces.values():
                shutil.copy2(path, output / sku / path.name)
    except Exception:
        # Only our new incomplete output is removed; staging remains for diagnosis.
        if created:
            shutil.rmtree(output)
        raise


def overview(before, after, destination, faces=FRAMES):
    """Single sheet: every left together, then right, top and bottom."""
    cell_w, cell_h, columns = 240, 470, 8
    panels = []
    for face in faces:
        names = sorted(after)
        for start in range(0, len(names), columns):
            group = names[start:start + columns]
            panel = Image.new('RGB', (cell_w * len(group), cell_h), '#242424')
            draw = ImageDraw.Draw(panel)
            for index, sku in enumerate(group):
                draw.text((index * cell_w + 10, 8), f'{face} / {sku}', fill='white')
                for j, (label, source) in enumerate((('before', before), ('after', after))):
                    with Image.open(source[sku][face]) as opened:
                        image = opened.convert('RGBA')
                    image.thumbnail((105, cell_h - 65), Image.Resampling.LANCZOS)
                    x = index * cell_w + 10 + j * 115
                    draw.text((x, 28), label, fill='white')
                    panel.paste(image, (x + (105 - image.width) // 2, 55), image)
            panels.append(panel)
    sheet = Image.new('RGB', (max(p.width for p in panels), sum(p.height for p in panels)), '#242424')
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    sheet.save(destination)


def deskew_image(image, face):
    from phone_normalizer.frame_strip import deskew_cropped_frame
    if image.mode != 'RGBA' or image.getextrema()[3][0] == 255:
        raise ValueError(f'{face} 需要已抠图的透明 RGBA 图片')
    # Exact quarter-turns expose horizontal rails without interpolation.
    horizontal = image.transpose(Image.Transpose.ROTATE_90) if face in ('left', 'right') else image
    result, info = deskew_cropped_frame(horizontal, face='top' if face == 'top' else 'bottom')
    if face in ('left', 'right'):
        result = result.transpose(Image.Transpose.ROTATE_270)
    info.update(face=face, source_size=list(image.size), output_size=list(result.size))
    return result, info


def guided_frame(matte, face, guide):
    """Apply reviewed source-space geometry without a second automatic rail fit."""
    from scripts.geometry_postprocess import rotate_alpha_safe
    result = matte.crop(tuple(guide['crop_xyxy'])).convert('RGBA')
    angle = guide['rotation_ccw_degrees']
    if angle:
        result = Image.fromarray(rotate_alpha_safe(np.asarray(result), angle))
    box = result.getchannel('A').getbbox()
    if box is None:
        raise ValueError('标定区域内没有可见像素')
    result = result.crop(box)
    if face in ('left', 'right'):
        result = result.transpose(Image.Transpose.ROTATE_90 if face == 'left' else Image.Transpose.ROTATE_270)
    return result, {'method': 'reviewed_frame_geometry', 'reviewed_source_crop': guide,
                    'output_size': list(result.size), 'resampling_passes': int(bool(angle)),
                    'rgb_painted': False}


def process_batch(input_dir, output_dir, process_dir, *, mode, faces=FRAMES, device='auto', backend_instance=None, frame_guides=None):
    source, output, work = separate_paths(input_dir, output_dir, process_dir)
    fresh(output)
    fresh(work)
    records = batch_files(source)
    guides = json.loads(Path(frame_guides).read_text()) if frame_guides else {}
    for sku, entries in guides.items():
        if sku not in records:
            raise ValueError(f'边框裁切标定包含未知 SKU：{sku}')
        for face, guide in entries.items():
            if face not in FRAMES or guide['source_sha256'] != alignment.digest(records[sku][face]):
                raise ValueError(f'{sku}/{face} 原图与裁切标定不一致')
            with Image.open(records[sku][face]) as image:
                x0,y0,x1,y1 = guide['crop_xyxy']
                if not all(type(v) is int for v in (x0,y0,x1,y1)) or not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
                    raise ValueError(f'{sku}/{face} 裁切坐标无效')
            angle = guide.get('rotation_ccw_degrees')
            if not isinstance(angle, (int,float)) or not math.isfinite(angle) or abs(angle) > 4:
                raise ValueError(f'{sku}/{face} 需要经核对的摆正角度（不超过 4 度）')
    if mode not in ('standardize', 'deskew') or not faces or set(faces) - set(FRAMES):
        raise ValueError('无效处理模式或边框面名称')
    work.mkdir(parents=True)
    staged = work / 'staged'
    report = {'mode': mode, 'success': False, 'uv_validation': 'requires_current_blender_scene', 'images': []}
    context = nullcontext(None)
    if mode == 'standardize':
        from phone_normalizer.api import Normalizer, NormalizerConfig
        from phone_normalizer.uv import normalize_face
        from phone_normalizer.frame_strip import normalize_frame_strip
        context = Normalizer(NormalizerConfig(remove_support=False, deskew=False, tight_crop=False, device=device), backend_instance=backend_instance)
    try:
        with context as normalizer:
            for sku, files in records.items():
                folder = staged / sku
                folder.mkdir(parents=True)
                for face, path in files.items():
                    print(f'{mode}: {sku}/{face}', flush=True)
                    info = {'sku': sku, 'face': face, 'source': str(path), 'source_sha256': alignment.digest(path)}
                    if mode == 'deskew' and face not in faces:
                        dest = folder / path.name
                        shutil.copy2(path, dest)
                        info['operation'] = 'byte_exact_copy'
                    else:
                        dest = folder / f'{face}.png'
                        with Image.open(path) as opened:
                            if mode == 'deskew':
                                result, geometry = deskew_image(opened, face)
                            elif face in FRAMES:
                                matte, _ = normalizer.normalize(opened)
                                guide = guides.get(sku, {}).get(face)
                                if guide:
                                    result, geometry = guided_frame(matte, face, guide)
                                else:
                                    result, geometry = normalize_frame_strip(matte, face=face, deskew=True)
                            else:
                                result, geometry = normalize_face(opened, face, backend=normalizer.backend,
                                    config=NormalizerConfig(crop_threshold=128, device=device))
                        result.save(dest)
                        info['geometry'] = geometry
                    info['output_sha256'] = alignment.digest(dest)
                    report['images'].append(info)
        overview(records, batch_files(staged), work / 'overview.png')
        publish(staged, output)
        report['success'] = True
        report['output'] = str(output)
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write_json(work / 'report.json', report)
    return report


def prepare(input_dir, process_dir, *, reference, width=None, height=None):
    source, work = separate_paths(input_dir, process_dir)
    batch_files(source)
    alignment.prepare(SimpleNamespace(input=source, output=work, reference=reference, width=width, height=height))


def validate_calibration(work, config):
    """No implicit shared antenna fallback in the merged workflow."""
    from texture_alignment.blender_scene_guard import require_match
    manifest = json.loads((work / 'manifest.json').read_text())
    if not config.get('targets_by_sku'):
        raise ValueError('需要 targets_by_sku：天线必须按各自背板与当前 UV 标定')
    references = {str(Path(item['path']).expanduser().resolve()): item['sha256']
                  for item in config.get('reference_files', [])}
    for item in manifest['companions']:
        if Path(item['name']).stem.lower() == 'back':
            if references.get(str(Path(item['source']).resolve())) != item['sha256']:
                raise ValueError(f'{item["sku"]} 缺少对应背板的标定哈希')
    scene = Path(config.get('scene_record', '')).expanduser().resolve()
    if not scene.is_file() or references.get(str(scene)) != alignment.digest(scene):
        raise ValueError('需要 scene_record：当前 Blender 场景标定记录及其 reference_files 哈希')
    captured = json.loads(scene.read_text())
    require_match(captured, captured)
    for item in manifest['images']:
        alignment.target_points(config, item['sku'], item['side'])
    return manifest


def apply(work_dir, landmarks, output_dir, process_dir):
    work, output, process = separate_paths(work_dir, output_dir, process_dir)
    fresh(output)
    fresh(process)
    config = json.loads(Path(landmarks).read_text())
    manifest = validate_calibration(work, config)
    # Protect original six-face inputs as well as the prepared batch directory.
    for item in manifest['images'] + manifest['companions']:
        separate_paths(Path(item['source']).parent, output, process)
    process.mkdir(parents=True)
    staged = process / 'staged'
    alignment.apply(SimpleNamespace(work=work, landmarks=Path(landmarks), output=staged))
    after = batch_files(staged)
    before = {sku: dict(files) for sku, files in after.items()}
    for item in manifest['images']:
        before[item['sku']][item['side']] = work / item['file']
    overview(before, after, process / 'overview.png', faces=('left', 'right'))
    # Reports and old per-page previews are process artifacts, never deliverables.
    for path in staged.iterdir():
        if path.is_file():
            shutil.move(str(path), process / path.name)
    report = json.loads((process / 'alignment-report.json').read_text())
    report['uv_validation'] = 'requires_current_blender_scene'
    write_json(process / 'alignment-report.json', report)
    publish(staged, output)
    return report
