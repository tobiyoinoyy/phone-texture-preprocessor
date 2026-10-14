"""Source-bound lighting scope and independent perimeter acceptance.

This is an effect gate, separate from geometry and edit-boundary invariants.
Masks and thresholds must be reviewed from the source, never fitted to output.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ZONES = ('top', 'bottom', 'left', 'right', 'top_left', 'top_right', 'bottom_left', 'bottom_right')
CHECKS = ('whole_perimeter', 'four_corners', 'material_boundaries', 'paper_edges',
          'no_new_blocks_or_gray_rim', 'source_details_retained')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def pixels(path):
    return np.array(Image.open(path).convert('RGBA'))


def witness(path, shape):
    m = np.array(Image.open(path).convert('L')) > 0
    if m.shape != shape or not m.any():
        raise ValueError('Perimeter witness must be nonempty and source-sized')
    return m


def lightness(rgb):
    s = rgb.astype(float) / 255
    linear = np.where(s <= .04045, s / 12.92, ((s + .055) / 1.055) ** 2.4)
    y = linear @ np.array([.2126729, .7151522, .0721750])
    return np.where(y > (6/29)**3, 116*np.cbrt(y)-16, (29/3)**3*y)


def validate_spec(spec, source):
    if spec['source_sha256'] != digest(source):
        raise ValueError('Lighting scope source mismatch')
    if spec['action'] not in ('copy', 'aluminum', 'stainless') or not spec.get('reason'):
        raise ValueError('Explicit per-face lighting action and reason required')
    if spec['mode'] not in ('unchanged', 'regional', 'perimeter'):
        raise ValueError('Unknown lighting scope')
    if (spec['mode'] == 'unchanged') != (spec['action'] == 'copy'):
        raise ValueError('Copy cannot fulfill a required lighting correction')
    if spec['action']=='aluminum' and 'protection_policy' not in spec:
        raise ValueError('Explicit aluminum protection policy required; do not inherit old masks')
    if spec.get('protection_policy')=='reviewed_regions' and not spec.get('protection_basis'):
        raise ValueError('Additional protection requires an explicit user/task basis')
    if spec.get('protection_policy', 'reviewed_regions') not in ('sticker_only','reviewed_regions'):
        raise ValueError('Unknown protection policy')
    if spec.get('protection_policy') == 'sticker_only':
        p = spec.get('sticker_mask')
        if not p or np.array(Image.open(p).convert('L')).shape != pixels(source).shape[:2]:
            raise ValueError('Sticker-only scope requires a source-sized sticker mask')
    for name in spec.get('local_checks', {}):
        if not name or name in ZONES or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name):
            raise ValueError('Local check requires a unique safe name')
    if spec['mode'] != 'perimeter':
        return
    if spec['action'] != 'aluminum' or not spec.get('reference_basis'):
        raise ValueError('Perimeter needs aluminum action and reviewed reference basis')
    a = pixels(source)
    required = witness(spec['required_metal_mask'], a.shape[:2])
    if np.any(required & (a[..., 3] == 0)):
        raise ValueError('Required metal must be visible source metal')
    if set(spec['zones']) != set(ZONES):
        raise ValueError('Perimeter requires four edges and four corners')
    yy, xx = np.nonzero(a[..., 3] > 0)
    x0, x1, y0, y1 = xx.min(), xx.max(), yy.min(), yy.max()
    for name, zone in spec['zones'].items():
        m = witness(zone['mask'], a.shape[:2])
        if np.any(m & ((a[..., 3] < 240) | ~required)):
            raise ValueError('Brightness witnesses must be opaque required metal')
        wy, wx = np.nonzero(m)
        cx, cy = (np.median(wx)-x0)/max(1,x1-x0), (np.median(wy)-y0)/max(1,y1-y0)
        if (('top' in name and cy > .25) or ('bottom' in name and cy < .75)
                or ('left' in name and cx > .25) or ('right' in name and cx < .75)):
            raise ValueError('Perimeter witness is not in its declared edge/corner')
        bounds = np.array([zone['min_L'], zone['max_L']], dtype=float)
        if not np.isfinite(bounds).all() or not 0 < bounds[0] < bounds[1] <= 100:
            raise ValueError('Finite, source-reviewed L bounds required')


def freeze_scope(work, path):
    work = Path(work)
    dest = work/'lighting-scope.json'
    if dest.exists():
        raise ValueError('Lighting scope already frozen; use a new review workspace to revise scope')
    baseline, specs = read(work/'geometry.json'), read(path)
    if set(specs) != set(baseline):
        raise ValueError('Lighting scope SKU set differs')
    assets = {}
    for sku, faces in baseline.items():
        if set(specs[sku]) != set(faces):
            raise ValueError('Explicit lighting scope required for all six faces')
        aluminum = any(v['action'] == 'aluminum' for v in specs[sku].values())
        for face, item in faces.items():
            spec = specs[sku][face]
            if aluminum and face in ('front', 'back') and spec['mode'] != 'perimeter':
                exemption = spec.get('perimeter_exemption', {})
                if (exemption.get('reason') not in ('user_scope_excludes_perimeter', 'no_visible_metal', 'already_accepted')
                        or not exemption.get('basis') or exemption.get('source_sha256') != item['sha256']):
                    raise ValueError('Front/back aluminum perimeter requires explicit source-bound exemption')
            if digest(item['path']) != item['sha256']:
                raise ValueError('Geometry changed before lighting scope')
            spec = specs[sku][face]
            validate_spec(spec, item['path'])
            if spec.get('protection_policy') == 'sticker_only':
                assets[str(Path(spec['sticker_mask']).resolve())] = digest(spec['sticker_mask'])
            paths = [z['mask'] for z in spec.get('local_checks', {}).values()]
            if spec['mode'] == 'perimeter':
                paths += [spec['required_metal_mask']] + [z['mask'] for z in spec['zones'].values()]
            assets.update({str(Path(p).resolve()): digest(p) for p in paths})
    write(dest, {'specs': specs, 'assets': assets})
    write(work/'lighting-scope-proof.json', {'sha256': digest(dest)})
    return specs


def load_scope(work):
    work = Path(work)
    path = work/'lighting-scope.json'
    if not path.exists() or not (work/'lighting-scope-proof.json').exists():
        raise ValueError('Frozen lighting scope required before publication')
    if digest(path) != read(work/'lighting-scope-proof.json')['sha256']:
        raise ValueError('Lighting scope changed after review')
    result = read(path)
    for path, sha in result['assets'].items():
        if digest(path) != sha:
            raise ValueError('Perimeter witness changed after scope freeze')
    return result['specs']


def measure_perimeter(spec, source, output, allowed, protected):
    validate_spec(spec, source)
    a, b = pixels(source), pixels(output)
    if a.shape != b.shape or not np.array_equal(a[..., 3], b[..., 3]):
        raise ValueError('Perimeter output changed geometry')
    required = witness(spec['required_metal_mask'], a.shape[:2])
    if np.any(required & (~allowed | protected)):
        raise ValueError('Required perimeter metal frozen or omitted')
    before_L, after_L = lightness(a[..., :3]), lightness(b[..., :3])
    reports = {}
    for name, zone in spec['zones'].items():
        m = witness(zone['mask'], a.shape[:2])
        before, after = float(np.median(before_L[m])), float(np.median(after_L[m]))
        if not zone['min_L'] <= after <= zone['max_L']:
            raise ValueError(f'Perimeter effect failed at {name}: L={after:.2f}, expected {zone["min_L"]}..{zone["max_L"]}')
        reports[name] = {'before_L': before, 'after_L': after, 'witness_sha256': digest(zone['mask'])}
    reports.update(measure_local(spec, source, output, allowed, protected))
    return reports


def measure_local(spec, source, output, allowed, protected):
    validate_spec(spec, source)
    a, b = pixels(source), pixels(output)
    if a.shape != b.shape or not np.array_equal(a[..., 3], b[..., 3]):
        raise ValueError('Local output changed geometry')
    after_L = lightness(b[..., :3])
    reports = {}
    for name, zone in spec.get('local_checks', {}).items():
        m = witness(zone['mask'], a.shape[:2])
        if np.any(m & ((a[...,3] < 240) | ~allowed | protected)):
            raise ValueError(f'Local feature omitted or frozen: {name}')
        q = float(zone.get('percentile', 10))
        bounds = np.array([q, zone['min_L'], zone['max_L']],dtype=float)
        if not np.isfinite(bounds).all() or not 0 <= q <= 100 or not 0 <= bounds[1] < bounds[2] <= 100:
            raise ValueError('Invalid local feature threshold')
        value = float(np.percentile(after_L[m],q))
        if not bounds[1] <= value <= bounds[2]:
            raise ValueError(f'Local feature effect failed: {name}: L={value:.2f}')
        reports[name] = {'after_percentile_L':value,'percentile':q,'witness_sha256':digest(zone['mask'])}
    return reports


def evidence_images(spec, source, output, folder):
    """Deterministic identical-coordinate 1x/4x comparisons on light/dark backgrounds."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    a, b = Image.open(source).convert('RGBA'), Image.open(output).convert('RGBA')
    assets = {}
    for name, zone in {**spec.get('zones', {}), **spec.get('local_checks', {})}.items():
        y, x = np.nonzero(witness(zone['mask'], (a.height, a.width)))
        box = (max(0,int(x.min())-12), max(0,int(y.min())-12),
               min(a.width,int(x.max())+13), min(a.height,int(y.max())+13))
        crops = [im.crop(box) for im in (a,b)]
        w,h = crops[0].size
        for bg, color in [('dark','#222222'), ('light','#eeeeee')]:
            for scale in (1,4):
                sheet = Image.new('RGB', (w*scale*2,h*scale), color)
                for i, crop in enumerate(crops):
                    c = crop.resize((w*scale,h*scale), Image.Resampling.NEAREST)
                    sheet.paste(c, (i*w*scale,0), c)
                p = folder/f'{name}-{bg}-{scale}x.png'
                sheet.save(p)
                assets[str(p.resolve())] = digest(p)
    review = {'source_sha256': digest(source), 'output_sha256': digest(output),
              'spec_sha256': hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest(),
              'evidence': assets, 'checks': {c: False for c in CHECKS}}
    write(folder/'review.json', review)
    return review


def check_review(spec, source, output, path):
    review = read(path)
    expected = hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    if (review['source_sha256'] != digest(source) or review['output_sha256'] != digest(output)
            or review['spec_sha256'] != expected):
        raise ValueError('Perimeter visual review belongs to another source/output/scope')
    if any(review.get('checks', {}).get(c) is not True for c in CHECKS):
        raise ValueError('Perimeter visual review incomplete')
    names = {Path(p).name for p in review['evidence']}
    needed = {f'{z}-{bg}-{s}x.png' for z in (*((ZONES) if spec['mode'] == 'perimeter' else ()), *spec.get('local_checks', {})) for bg in ('dark','light') for s in (1,4)}
    if not needed <= names:
        raise ValueError('Perimeter visual evidence incomplete')
    for p, sha in review['evidence'].items():
        if digest(p) != sha:
            raise ValueError('Perimeter evidence changed after review')
    return digest(path)
