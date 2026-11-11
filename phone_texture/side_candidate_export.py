"""Create a movable evidence package from verified current-job selections."""
import copy
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from . import workspace as w, pipeline


def create(row, case, manifest, geometry_files, options):
    from .automatic import implementation_manifest
    from .review_bundle import package, verify

    case = Path(case).resolve(strict=True)
    if row.get('side_candidate_review_complete') is not True:
        raise ValueError('Only completed current side-review candidates can create this package')
    w.check_sources(manifest)
    faces = {}
    for face in pipeline.FACES:
        source = Path(geometry_files[face]).resolve(strict=True)
        target = Path(row['candidate'])/row['sku']/(face+'.png')
        info = row['per_face_stages'][face]
        if (w.sha(source) != info['source_sha256'] or w.sha(target) != row['final_sha256'][face]
                or info['selected_sha256'] != row['final_sha256'][face]):
            raise RuntimeError('Portable candidate current source/selection changed: '+face)
        with Image.open(source) as a, Image.open(target) as b:
            if a.format != 'PNG' or b.format != 'PNG' or a.mode != 'RGBA' or b.mode != 'RGBA':
                raise RuntimeError('Portable candidate source/output is not RGBA PNG')
            before, after = np.asarray(a), np.asarray(b)
            if before.shape != after.shape or not np.array_equal(before[..., 3], after[..., 3]):
                raise RuntimeError('Portable candidate changed source geometry: '+face)
        checks = dict(source_identity_verified=True, pixel_integrity_verified=True,
            output_selection_verified=True, hard_failures=[],
            alpha_changed_pixels=0, source_and_output_size_equal=True,
            source_geometry_byte_identical=w.sha(source) == w.sha(target),
            final_canvas_checks=copy.deepcopy(row['final_canvas_checks'][face]))
        if info['selected_stage'] == 'structure_review':
            checks['lighting'] = copy.deepcopy(row['lighting_review_checks'][face])
            checks['structure'] = copy.deepcopy(row['structure_review_checks'][face])
        elif info['selected_stage'] == 'lighting':
            original = row['lighting_checks'][face]
            if (original['source_sha256'] != w.sha(source) or original['output_sha256'] != w.sha(target)
                    or any(original[name] != 0 for name in ('alpha_changed', 'paper_changed', 'outside_material_changed'))):
                raise RuntimeError('Portable normal-lighting checks differ: '+face)
            checks['current_lighting_checks'] = copy.deepcopy(original)
        elif w.sha(source) != w.sha(target):
            raise RuntimeError('Portable face lacks a verified selected stage: '+face)
        faces[face] = dict(source_sha256=w.sha(source), output_sha256=w.sha(target), checks=checks)
    # Current case files are a closed evidence inventory. Original source paths
    # in JSON remain provenance; the package copies this job's intake snapshots.
    assets = {str(p.resolve()): w.sha(p) for p in (case/'work').rglob('*') if p.is_file()}
    repo = Path(__file__).resolve().parents[1]
    assets.update({str(repo/name): digest for name, digest in implementation_manifest(repo).items()})
    row['portable_review_evidence'] = dict(schema='current_review_evidence/v1',
        options=dict(options), assets=assets, faces=faces)
    folder = case/'review-bundle'
    descriptor = package(row, case, folder)
    sources = {face: Path(item['source']) for face, item in manifest['skus'][row['sku']].items()}
    verify(folder, sources, sku=row['sku'], options=options,
           expected_manifest_sha256=descriptor['manifest_sha256'])
    archive_path = case/'review-bundle.zip'
    if archive_path.exists():
        raise ValueError('Portable review archive already exists')
    files = sorted(p for p in folder.rglob('*') if p.is_file())
    digests = {p.relative_to(folder).as_posix(): w.sha(p) for p in files}
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_STORED) as archive:
        for path in files:
            archive.write(path, path.relative_to(folder).as_posix())
    import hashlib
    with zipfile.ZipFile(archive_path) as archive:
        if len(archive.namelist()) != len(digests) or set(archive.namelist()) != set(digests):
            raise RuntimeError('Portable review ZIP file inventory differs')
        for name, digest in digests.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise RuntimeError('Portable review ZIP member changed: '+name)
    w.check_sources(manifest)
    for path, expected in assets.items():
        if w.sha(path) != expected:
            raise RuntimeError('Current evidence changed during portable packaging: '+path)
    descriptor.update(zip_path=str(archive_path), zip_sha256=w.sha(archive_path),
        historical_paths_required_on_replay=False, current_sources_verified=True,
        archive_files=len(files), image_processing_executed_for_package=False)
    return descriptor
