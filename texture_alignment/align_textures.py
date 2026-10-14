#!/usr/bin/env python3
"""Normalize cropped side textures, then align marked button edges at fixed height."""
import argparse
import base64
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SIDES = ('left', 'right')


def target_points(config, name, side):
    """Explicit per-SKU seam targets must never fall back to a shared SKU."""
    if 'targets_by_sku' in config:
        try:
            return config['targets_by_sku'][name][side]
        except KeyError as exc:
            raise ValueError(f'{name}/{side}: 缺少该 SKU 的背板接缝目标') from exc
    return config.get('targets', config['points'][config['reference']])[side]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def image_size(path):
    with Image.open(path) as im:
        return im.size


def validate_points(points, height, label):
    p = np.asarray(points, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all():
        raise ValueError(f'{label}: 需要至少一个有效标记点')
    if p[0] <= 0 or p[-1] >= height - 1 or np.any(np.diff(p) <= 0):
        raise ValueError(f'{label}: 标记必须从上到下递增，且位于首尾像素之间')
    return np.r_[0., p, height - 1.]


def anchor_matrix(points, height, width, label):
    a=np.asarray(points,dtype=float)
    if a.ndim==1:a=np.repeat(a[:,None],width,axis=1)
    if a.ndim!=2 or a.shape[1]!=width:
        raise ValueError(f'{label}: anchor columns differ from texture width')
    return np.stack([validate_points(a[:,x],height,f'{label} column {x}') for x in range(width)],axis=1)


def warp_rows(image, source, target):
    """Inverse mapping with independently reviewed front/back edge targets."""
    image=image.convert('RGBA');h=image.height;w=image.width
    src=anchor_matrix(source,h,w,'source');dst=anchor_matrix(target,h,w,'target')
    if src.shape!=dst.shape:raise ValueError('源图与目标标记数量必须相同')
    if np.array_equal(src,dst):return image.copy()
    rows=np.stack([np.interp(np.arange(h),dst[:,x],src[:,x]) for x in range(w)],axis=1)
    return sample_rows(image,rows)


def sample_rows(image, rows):
    h = image.height
    lo = np.floor(rows).astype(int)
    hi = np.minimum(lo + 1, h - 1)
    t = (rows - lo).astype(np.float32)[..., None]
    rgba = np.asarray(image).astype(np.float32)
    premul = rgba.copy()
    premul[..., :3] *= rgba[..., 3:] / 255.
    x = np.broadcast_to(np.arange(image.width), rows.shape)
    result = premul[lo, x] * (1 - t) + premul[hi, x] * t
    result[..., :3] = np.divide(result[..., :3] * 255., result[..., 3:],
                                out=np.zeros_like(result[..., :3]), where=result[..., 3:] > 0)
    result = np.clip(np.rint(result), 0, 255).astype(np.uint8)
    # Exact integer samples (including both endpoints) preserve all RGBA bytes.
    exact = np.abs(rows - np.rint(rows)) < 1e-9
    result[exact] = np.asarray(image)[np.rint(rows[exact]).astype(int), x[exact]]
    return Image.fromarray(result)


def source_anchors(config, name, side, width):
    points = np.asarray(config['points'][name][side], dtype=float)
    slopes = config.get('antenna_slopes', {}).get(name, {}).get(side)
    if slopes is None:
        return points
    slopes = np.asarray(slopes, dtype=float)
    if slopes.shape != points.shape or not np.isfinite(slopes).all():
        raise ValueError('天线倾斜参数与定位点数量不一致')
    return points[:, None] + slopes[:, None] * (np.arange(width) - (width - 1) / 2)


def prepare(args):
    root, out = args.input.resolve(), args.output.resolve()
    if out == root or root in out.parents:
        raise ValueError('输出目录必须位于输入批次目录之外')
    if out.exists():
        raise ValueError('请使用尚不存在的输出目录，防止覆盖已有结果')
    records = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        files = {}
        for side in SIDES:
            matches = [p for p in folder.iterdir() if p.is_file() and p.stem.lower() == side
                       and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff')]
            if len(matches) > 1:
                raise ValueError(f'{folder.name}/{side}: 存在重名图片，请只保留一份')
            if matches:
                files[side] = matches[0]
        if files and len(files) != 2:
            raise ValueError(f'{folder.name}: 缺少 left 或 right')
        if files:
            records.append((folder.name, files))
    reference = next((files for name, files in records if name == args.reference), None)
    if reference is None:
        raise ValueError(f'未找到基准目录 {args.reference} 的 left/right 图片')
    ref_sizes = {s: image_size(reference[s]) for s in SIDES}
    height = args.height or ref_sizes['left'][1]
    if height < 2:
        raise ValueError('高度必须至少为 2 像素')
    sizes = {s: [args.width or max(1, round(ref_sizes[s][0] * height / ref_sizes[s][1])), height]
             for s in SIDES}
    if any(size[0] < 1 for size in sizes.values()):
        raise ValueError('宽度必须为正整数')
    # Validate all inputs before creating output.
    for _, files in records:
        for path in files.values():
            with Image.open(path) as im:
                im.verify()
    out.mkdir(parents=True)
    manifest = {'reference': args.reference, 'height': height, 'sizes': sizes, 'images': [], 'companions': []}
    for name, files in records:
        for side in ('front', 'back', 'top', 'bottom'):
            matches = [p for p in (root / name).iterdir() if p.is_file() and p.stem.lower() == side
                       and p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff')]
            for path in matches:
                manifest['companions'].append({'sku': name, 'name': path.name,
                    'source': str(path), 'sha256': digest(path)})
        for side, path in files.items():
            dest = out / 'normalized' / name / f'{side}.png'
            dest.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as original:
                im = original.convert('RGBA')
                original_size = list(im.size)
                if im.size != tuple(sizes[side]):
                    im = im.resize(sizes[side], Image.Resampling.LANCZOS)
                im.save(dest)
            manifest['images'].append({'sku': name, 'side': side, 'source': str(path),
                'original_size': original_size, 'source_sha256': digest(path),
                'file': str(dest.relative_to(out)), 'sha256': digest(dest)})
    write_json(out / 'manifest.json', manifest)
    annotations = {'reference': args.reference, 'height': height, 'points': {
        name: {s: [] for s in SIDES} for name, _ in records}}
    write_json(out / 'landmarks.json', annotations)
    make_annotator(out, manifest, annotations)
    print(f'已统一 {len(records) * 2} 张贴图，尺寸 {sizes}；打开 {out / "标注按钮.html"}')


def make_annotator(out, manifest, annotations):
    data = []
    for item in manifest['images']:
        width, height = image_size(out / item['file'])
        # Embedded full-resolution images avoid local browser permission problems.
        encoded = base64.b64encode((out / item['file']).read_bytes()).decode('ascii')
        data.append({**item, 'width': width, 'height': height,
                     'url': 'data:image/png;base64,' + encoded})
    template = Path(__file__).with_name('annotator.html').read_text(encoding='utf-8')
    payload = json.dumps({'images': data, 'annotations': annotations}, ensure_ascii=False).replace('<', '\\u003c')
    (out / '标注按钮.html').write_text(template.replace('/*PAYLOAD*/', payload), encoding='utf-8')


def apply(args):
    work = args.work.resolve()
    manifest = json.loads((work / 'manifest.json').read_text())
    config = json.loads(args.landmarks.read_text())
    h, ref = manifest['height'], manifest['reference']
    if config['reference'] != ref or config['height'] != h:
        raise ValueError('标记文件与本批次 sku0 / 高度不一致')
    output = args.output.resolve() if args.output else work / 'aligned'
    if output.exists():
        raise ValueError('结果目录已存在，请用 --output 指定新目录')
    ref_points = config.get('targets', config['points'][ref])
    for reference in config.get('reference_files', []):
        if digest(Path(reference['path'])) != reference['sha256']:
            raise ValueError('背板或 UV 基准文件已变化，需重新标定: '+reference['path'])
    for item in manifest.get('companions', []):
        if digest(Path(item['source'])) != item['sha256']:
            raise ValueError(f'{item["sku"]}/{item["name"]}: 原有贴图已变化，请重新准备批次')
    for side in SIDES:
        anchor_matrix(target_points(config, ref, side), h, manifest['sizes'][side][0], f'{ref}/{side}')
    # Check the whole batch before emitting any images.
    for item in manifest['images']:
        name, side = item['sku'], item['side']
        if digest(work / item['file']) != item['sha256']:
            raise ValueError(f'{name}/{side}: 标准化图片已经变化，请重新标注')
        src = validate_points(config['points'][name][side], h, f'{name}/{side}')
        dst = anchor_matrix(target_points(config, name, side), h, manifest['sizes'][side][0], 'reference')
        if len(src) != len(dst):
            raise ValueError(f'{name}/{side}: 标记数量与 sku0 不一致')
        anchors = source_anchors(config, name, side, manifest['sizes'][side][0])
        if anchors.ndim == 2:
            for x in (0, anchors.shape[1] - 1):
                validate_points(anchors[:, x], h, f'{name}/{side}/column {x}')
    output.mkdir(parents=True)
    report = {'reference': ref, 'height': h, 'method': 'piecewise_linear_y_fixed_endpoints',
              'targets': ref_points, 'targets_by_sku':config.get('targets_by_sku'),
              'reference_files':config.get('reference_files', []), 'images': []}
    for item in manifest['images']:
        name, side = item['sku'], item['side']
        original = Image.open(work / item['file']).convert('RGBA')
        source, target = config['points'][name][side], target_points(config, name, side)
        result = warp_rows(original, source_anchors(config, name, side, original.width), target)
        dest = output / name / f'{side}.png'
        dest.parent.mkdir(parents=True, exist_ok=True)
        result.save(dest)
        assert result.size == original.size
        src = anchor_matrix(source_anchors(config,name,side,original.width), h, original.width, 'source')
        dst = anchor_matrix(target, h, original.width, 'target')
        report['images'].append({'sku': name, 'side': side, 'size': list(result.size),
            'source_y': source, 'target_y': target,
            'source_slopes': config.get('antenna_slopes', {}).get(name, {}).get(side),
            'max_shift_px': float(np.max(np.abs(src - dst))),
            'segment_stretch': (np.diff(dst,axis=0) / np.diff(src,axis=0)).tolist(),
            'sha256': digest(dest)})
    report['companions'] = []
    for item in manifest.get('companions', []):
        dest = output / item['sku'] / item['name']
        shutil.copy2(item['source'], dest)
        report['companions'].append({**item, 'output_sha256': digest(dest)})
    write_json(output / 'alignment-report.json', report)
    write_json(output / 'landmarks-used.json', config)
    for side in SIDES:
        make_review(work, output, manifest, config, side)
    print(f'已映射 {len(report["images"])} 张；所有输出高度仍为 {h} px。结果：{output}')


def make_review(work, output, manifest, config, side):
    items = [v for v in manifest['images'] if v['side'] == side]
    h = manifest['height']
    display_h = min(h, 1200)
    display_w = max(20, round(manifest['sizes'][side][0] * display_h / h))
    cell = max(130, display_w * 2 + 36)
    for start in range(0, len(items), 8):
        batch = items[start:start + 8]
        board = Image.new('RGB', (cell * len(batch), display_h + 65), '#242424')
        draw = ImageDraw.Draw(board)
        for col, item in enumerate(batch):
            name = item['sku']
            draw.text((col * cell + 8, 8), name, fill='white')
            for j, (label, path) in enumerate((('before', work / item['file']),
                    ('after', output / name / f'{side}.png'))):
                im = Image.open(path).convert('RGBA').resize((display_w, display_h), Image.Resampling.LANCZOS)
                x = col * cell + 8 + j * (display_w + 18)
                draw.text((x, 28), label, fill='white')
                board.paste(im, (x, 55), im)
                for row in np.asarray(target_points(config, name, side)):
                    ends=np.atleast_1d(row)
                    y0=55+round(float(ends[0])*(display_h-1)/(h-1))
                    y1=55+round(float(ends[-1])*(display_h-1)/(h-1))
                    draw.line((x,y0,x+display_w,y1),fill='#ff5e58')
        board.save(output / f'review-{side}-{start // 8 + 1}.jpg', quality=94)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='统一尺寸并创建按钮标注页')
    p.add_argument('input', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', default='sku0')
    p.add_argument('--height', type=int)
    p.add_argument('--width', type=int, help='可选：两侧统一宽度；默认各侧使用 sku0 宽高比')
    p.set_defaults(func=prepare)
    p = sub.add_parser('apply', help='按标注执行定高映射')
    p.add_argument('work', type=Path)
    p.add_argument('--landmarks', type=Path, required=True)
    p.add_argument('--output', type=Path)
    p.set_defaults(func=apply)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(2, f'错误：{exc}\n')


if __name__ == '__main__':
    main()
