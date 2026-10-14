"""Read-only intake, geometry checkpoint and fail-closed six-face delivery.

Geometry and lighting stay in their maintained tools. This helper never infers
material masks or treats a copied face as a successful lighting correction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from PIL import Image, ImageDraw
from phone_texture.pipeline import FACES, EXTENSIONS, batch_files, separate_paths, fresh
from phone_texture.geometry_review import verify_report
from phone_texture.protection_policy import check_sticker_only
from phone_texture.lighting_review import freeze_scope, load_scope, measure_perimeter, measure_local, check_review, evidence_images


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def face_files(folder):
    files = {}
    for path in folder.iterdir():
        if path.is_file() and path.suffix.lower() in EXTENSIONS and path.stem.lower() in FACES:
            face = path.stem.lower()
            if face in files:
                raise ValueError(f'Duplicate face: {folder.name}/{face}')
            with Image.open(path) as im:
                im.verify()
            files[face] = path
    if set(files) != set(FACES):
        raise ValueError(f'Missing faces: {folder.name}: {set(FACES) - set(files)}')
    return files


def intake(source, work, skus=None, single_name='SKU_01', delivery_goal='model_textures', goal_reason=None):
    if delivery_goal not in ('model_textures', 'photos') or (delivery_goal == 'photos' and not goal_reason):
        raise ValueError('Photos-only delivery needs an explicit user scope reason')
    source, work = separate_paths(source, work)
    fresh(work)
    is_single = any(p.is_file() and p.stem.lower() in FACES and p.suffix.lower() in EXTENSIONS for p in source.iterdir())
    if is_single:
        if Path(single_name).name != single_name or single_name in ('.', '..'):
            raise ValueError('Unsafe single SKU name')
        selected = {single_name: source}
    else:
        folders = {p.name: p for p in source.iterdir() if p.is_dir() and not p.name.startswith('.')}
        if skus and set(skus) - set(folders):
            raise ValueError(f'Unknown SKU: {set(skus) - set(folders)}')
        selected = {k: folders[k] for k in (skus or sorted(folders))}
    if not selected or (is_single and skus and set(skus) != {single_name}):
        raise ValueError('Empty selection or invalid single-folder selection')
    records = {sku: face_files(folder) for sku, folder in selected.items()}
    manifest = {'source_root': str(source), 'delivery_goal': delivery_goal, 'goal_reason': goal_reason, 'skus': {}}
    work.mkdir(parents=True)
    for sku, faces in records.items():
        (work/'input'/sku).mkdir(parents=True)
        manifest['skus'][sku] = {}
        for face, path in faces.items():
            digest = sha(path)
            dest = work/'input'/sku/(face + path.suffix.lower())
            shutil.copy2(path, dest)
            if sha(path) != digest or sha(dest) != digest:
                raise ValueError('Source changed while copying')
            manifest['skus'][sku][face] = {'source': str(path), 'snapshot': str(dest), 'sha256': digest}
    save(work/'intake.json', manifest)
    return manifest


def check_sources(manifest):
    for faces in manifest['skus'].values():
        for item in faces.values():
            if sha(item['source']) != item['sha256'] or sha(item['snapshot']) != item['sha256']:
                raise ValueError('Original or intake snapshot changed')


def checkpoint(work, geometry, geometry_report=None):
    work, geometry = Path(work).resolve(), Path(geometry).resolve()
    fresh(work/'geometry.json')
    manifest = read(work/'intake.json')
    check_sources(manifest)
    records = batch_files(geometry)
    if set(records) != set(manifest['skus']):
        raise ValueError('Geometry SKU set differs from intake')
    if geometry_report is None:
        raise ValueError('Approved geometry report required before freezing lighting baseline')
    approved = verify_report(geometry_report, records)
    for entry in approved['images']:
        if entry['source_sha256'] != manifest['skus'][entry['sku']][entry['face']]['sha256']:
            raise ValueError('Geometry proof is not from this intake source')
    result = {}
    for sku, faces in records.items():
        result[sku] = {}
        for face, path in faces.items():
            with Image.open(path) as im:
                if im.format != 'PNG' or im.mode != 'RGBA':
                    raise ValueError('Geometry must be RGBA PNG')
                result[sku][face] = {'path': str(path), 'sha256': sha(path), 'size': list(im.size)}
    save(work/'geometry.json', result)
    save(work/'geometry-proof.json', {'report':str(Path(geometry_report).resolve()),
                                     'sha256':sha(geometry_report)})
    return result


def mask(path, shape):
    with Image.open(path) as im:
        arr = np.array(im.convert('L'))
    if arr.shape != shape:
        raise ValueError('Mask dimensions differ')
    return arr > 0


def check_protection_boundaries(entry, source, allowed, protected):
    """Check independently reviewed feature/metal masks, not the output diff.

    The agent traces these from the immutable source before correction. This
    guard catches oversized protection even when all protected pixels are
    perfectly unchanged. It does not infer materials from image brightness.
    """
    checks = entry.get('boundary_checks', [])
    if entry.get('contains_sticker') and not checks:
        raise ValueError('Sticker requires source-bound paper/adjacent-metal checks')
    reports = []
    for check in checks:
        if check['source_sha256'] != sha(source):
            raise ValueError('Protection boundary review belongs to another source')
        feature = mask(check['feature_mask'], allowed.shape)
        metal = mask(check['adjacent_metal_mask'], allowed.shape)
        alpha = np.array(Image.open(source).convert('RGBA'))[..., 3] > 0
        if not feature.any() or not metal.any() or np.any(feature & metal):
            raise ValueError('Boundary review needs disjoint nonempty feature/metal masks')
        feature &= alpha
        metal &= alpha
        if not feature.any() or not metal.any():
            raise ValueError('Boundary review must cover visible source pixels')
        frozen_metal = metal & (protected | ~allowed)
        exposed_feature = feature & (~protected | allowed)
        if frozen_metal.any():
            raise ValueError(f'Oversized protection or missed adjacent metal: {int(frozen_metal.sum())} pixels')
        if exposed_feature.any():
            raise ValueError(f'Protected feature coverage missing: {int(exposed_feature.sum())} pixels')
        reports.append({'feature_mask_sha256': sha(check['feature_mask']),
                        'adjacent_metal_mask_sha256': sha(check['adjacent_metal_mask']),
                        'feature_pixels': int(feature.sum()), 'metal_pixels': int(metal.sum()),
                        'frozen_metal_pixels': 0, 'exposed_feature_pixels': 0})
    return reports



def delivery_overview(before, after, path):
    """Group by face with horizontal strips readable instead of tiny thumbnails."""
    names = sorted(after)
    panels = []
    for face in ('left', 'right', 'top', 'bottom', 'front', 'back'):
        strip = face in ('top', 'bottom')
        height = 180 if strip else 700 if face in ('left', 'right') else 450
        for start in range(0, len(names), 4):
            group = names[start:start + 4]
            panel = Image.new('RGB', (1440, height), '#303030')
            draw = ImageDraw.Draw(panel)
            for i, sku in enumerate(group):
                draw.text((i*360+8, 8), f'{face} / {sku} / before - after', fill='white')
                for j, records in enumerate((before, after)):
                    with Image.open(records[sku][face]) as opened:
                        im = opened.convert('RGBA')
                    im.thumbnail((340, 60) if strip else (166, height-40), Image.Resampling.LANCZOS)
                    x = i*360+(10 if strip else j*178+6)
                    y = 30+j*72 if strip else 32
                    panel.paste(im, (x, y), im)
            panels.append(panel)
    sheet = Image.new('RGB', (1440, sum(p.height for p in panels)), '#303030')
    y = 0
    for panel in panels:
        sheet.paste(panel, (0, y))
        y += panel.height
    sheet.save(path)


def publish(work, plan_path, output, structure_process=None, structure_review=None):
    work, output = Path(work).resolve(), Path(output).resolve()
    manifest = read(work/'intake.json')
    goal = manifest.get('delivery_goal', 'model_textures')
    if goal not in ('model_textures', 'photos'):
        raise ValueError('Unknown delivery goal')
    if goal == 'photos' and not manifest.get('goal_reason'):
        raise ValueError('Photos-only delivery lacks user scope reason')
    if goal == 'model_textures' and (not structure_process or not structure_review):
        raise ValueError('Model textures require structure alignment and visual review before publication')
    separate_paths(work, output, manifest['source_root'])
    fresh(output)
    baseline, plan = read(work/'geometry.json'), read(plan_path)
    scope = load_scope(work)
    if set(scope) != set(baseline):
        raise ValueError('Lighting scope SKU set differs')
    proof_path = work/'geometry-proof.json'
    if not proof_path.is_file():
        raise ValueError('Geometry approval missing; pixel checks cannot approve geometry')
    proof = read(proof_path)
    if sha(proof['report']) != proof['sha256']:
        raise ValueError('Geometry approval changed after checkpoint')
    verify_report(proof['report'], {s:{f:Path(v['path']) for f,v in faces.items()} for s,faces in baseline.items()})
    if set(baseline) != set(plan) or set(baseline) != set(manifest['skus']):
        raise ValueError('Delivery SKU set differs')
    check_sources(manifest)
    before, after, report = {}, {}, {}
    for sku, faces in baseline.items():
        if set(plan[sku]) != set(FACES):
            raise ValueError('Delivery must explicitly account for all six faces')
        before[sku], after[sku], report[sku] = {}, {}, {}
        for face, item in faces.items():
            src = Path(item['path'])
            separate_paths(output, src.parent)
            if sha(src) != item['sha256']:
                raise ValueError('Geometry changed after checkpoint')
            entry = plan[sku][face]
            action = entry['action']
            intended = scope[sku][face]
            if action != intended['action'] or intended['source_sha256'] != item['sha256']:
                raise ValueError('Delivery does not fulfill frozen lighting scope')
            if action not in ('copy', 'aluminum', 'stainless') or not entry.get('reason'):
                raise ValueError('Unknown action or missing processing reason')
            before[sku][face] = src
            dst = src if action == 'copy' else Path(entry['output']).resolve()
            separate_paths(output, dst.parent)
            a = np.array(Image.open(src).convert('RGBA'))
            with Image.open(dst) as im:
                if im.format != 'PNG' or im.mode != 'RGBA':
                    raise ValueError('Delivery must be RGBA PNG')
                b = np.array(im)
            if a.shape != b.shape or not np.array_equal(a[..., 3], b[..., 3]):
                raise ValueError('Lighting changed geometry or alpha')
            changed = np.any(a != b, axis=-1)
            record = {'action': action, 'reason': entry['reason'], 'source_sha256': sha(src),
                      'output_sha256': sha(dst), 'changed_pixels': int(changed.sum()),
                      'alpha_changed': 0, 'outside_changed': 0, 'protected_changed': 0}
            if action != 'copy':
                allowed = mask(entry['allowed_mask'], a.shape[:2]) & (a[..., 3] > 0)
                protected = mask(entry['protected_mask'], a.shape[:2])
                if np.any(changed & (~allowed | protected)):
                    raise ValueError('Lighting changed protected or out-of-mask pixels')
                record['protection_policy'] = check_sticker_only(intended, src, protected, allowed)
                record['boundary_checks'] = check_protection_boundaries(entry, src, allowed, protected)
                record.update(job_sha256=sha(entry['job']), allowed_mask_sha256=sha(entry['allowed_mask']),
                              protected_mask_sha256=sha(entry['protected_mask']))
            if intended['mode'] == 'perimeter':
                record['perimeter_effect'] = measure_perimeter(intended, src, dst, allowed, protected)
                if not entry.get('perimeter_review'):
                    raise ValueError('Perimeter visual review required')
                record['perimeter_review_sha256'] = check_review(intended, src, dst, entry['perimeter_review'])
                record['perimeter_status'] = 'effect_and_visual_review_passed'
            elif intended.get('local_checks'):
                local_allowed = allowed if action != 'copy' else a[..., 3] > 0
                local_protected = protected if action != 'copy' else np.zeros(a.shape[:2], bool)
                record['local_effect'] = measure_local(intended, src, dst, local_allowed, local_protected)
                if not entry.get('local_review'):
                    raise ValueError('Local feature visual review required')
                record['local_review_sha256'] = check_review(intended, src, dst, entry['local_review'])
                record['local_status'] = 'effect_and_visual_review_passed'
            after[sku][face], report[sku][face] = dst, record
    structure = {'status': 'not_requested_photos_only'}
    if goal == 'model_textures':
        from phone_texture.structure_review import verify_delivery
        separate_paths(output, structure_process)
        after, structure = verify_delivery(after, structure_process, structure_review)
        for sku, faces in after.items():
            for face, path in faces.items():
                record = report[sku][face]
                record['lighting_output_sha256'] = record['output_sha256']
                record['pixel_constraints_stage'] = 'lighting_before_structure'
                record['output_sha256'] = sha(path)
                record['structure_changed'] = record['output_sha256'] != record['lighting_output_sha256']
    # Recheck at publication, then copy to a sibling staging directory. A failed
    # validation never creates a misleading complete-looking delivery folder.
    check_sources(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix='.six-view-', dir=output.parent))
    try:
        for sku, faces in after.items():
            (staged/sku).mkdir()
            for face, src in faces.items():
                dest = staged/sku/(face+'.png')
                shutil.copy2(src, dest)
                if sha(dest) != report[sku][face]['output_sha256']:
                    raise ValueError('Output changed during publication')
        fresh(output)
        staged.rename(output)
    finally:
        if staged.exists():
            shutil.rmtree(staged)
    save(work/'validation.json', {'delivery_goal':goal, 'structure':structure, 'geometry_status':'approved', 'lighting_status': 'pixel_constraints_passed',
                                 'lighting_scope_sha256': sha(work/'lighting-scope.json'),
                                 'perimeter_status': 'passed' if any(v.get('perimeter_status') for fs in report.values() for v in fs.values()) else 'not_requested',
                                 'model_seams': 'not_checked', 'skus': report})
    delivery_overview(before, after, work/'overview.png')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('intake')
    p.add_argument('--input', required=True)
    p.add_argument('--work', required=True)
    p.add_argument('--skus', nargs='+')
    p.add_argument('--single-name', default='SKU_01')
    p.add_argument('--delivery-goal', choices=('model_textures','photos'), default='model_textures')
    p.add_argument('--goal-reason', help='User-authorized photos-only scope; never use to bypass failed alignment')
    p = sub.add_parser('checkpoint')
    p.add_argument('--work', required=True)
    p.add_argument('--geometry', required=True)
    p.add_argument('--geometry-report', required=True)
    p = sub.add_parser('lighting-scope')
    p.add_argument('--work', required=True)
    p.add_argument('--spec', required=True)
    p = sub.add_parser('perimeter-evidence')
    p.add_argument('--work', required=True)
    p.add_argument('--sku', required=True)
    p.add_argument('--face', choices=('front', 'back'), required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', required=True)
    p = sub.add_parser('align-structure')
    p.add_argument('--input', required=True, help='Complete six-face lighting candidates')
    p.add_argument('--spec', required=True)
    p.add_argument('--process', required=True)
    p = sub.add_parser('publish')
    p.add_argument('--work', required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--structure-process')
    p.add_argument('--structure-review')
    p.add_argument('--output', required=True)
    a = parser.parse_args()
    if a.command == 'intake':
        result = intake(a.input, a.work, a.skus, a.single_name, a.delivery_goal, a.goal_reason)
    elif a.command == 'align-structure':
        from phone_texture.structure_review import prepare
        result = prepare(a.input, a.spec, a.process)
    elif a.command == 'checkpoint':
        result = checkpoint(a.work, a.geometry, a.geometry_report)
    elif a.command == 'lighting-scope':
        result = freeze_scope(a.work, a.spec)
    elif a.command == 'perimeter-evidence':
        scope = load_scope(a.work)[a.sku][a.face]
        src = read(Path(a.work)/'geometry.json')[a.sku][a.face]['path']
        result = evidence_images(scope, src, a.candidate, a.output)
    else:
        result = publish(a.work, a.plan, a.output, a.structure_process, a.structure_review)
    print(json.dumps({'command': a.command, 'status': result.get('status', 'passed'), 'groups': len(result.get('skus', result))}))


if __name__ == '__main__':
    main()
