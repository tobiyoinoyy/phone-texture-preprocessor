"""Deterministic color continuity at the supplied models' inset frame UV seams.

Model UV geometry is fixed; all colors and material masks belong to this SKU.
Run after structure alignment, without resampling or changing the final alpha.
"""
import contextlib
import importlib
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from . import structure_review as sr
from .geometry_review import digest, write
from .pipeline import FACES, fresh

MODELS = ('iphone17pro', 'iphone17promax')


def profile(model, model_sha256=None):
    if model not in MODELS:
        raise ValueError('请选择模型：iPhone 17 Pro 或 iPhone 17 Pro Max')
    directory = Path(__file__).with_name('seam_profiles')
    path = directory / (model + '-viewer.json')
    data = json.loads(path.read_text())
    if model_sha256 is not None and data['model_sha256'] != model_sha256:
        path = directory / (model + '.json')
        data = json.loads(path.read_text())
        if data['model_sha256'] != model_sha256:
            raise ValueError('接缝坐标与所选模型文件不匹配')
    expected = {frozenset(p) for p in [('top','left'), ('top','right'),
                                      ('bottom','left'), ('bottom','right')]}
    if data['schema'] != 'phone-frame-uv-seams-v1' or len(data['chains']) != 4 or {
        frozenset(c['names']) for c in data['chains']} != expected:
        raise ValueError('Model profile must cover all four frame corners')
    return data, path


def resized_chains(data, files):
    """Normalized UV is invariant; correctly transform inward normals too."""
    chains = json.loads(json.dumps(data['chains']))
    for chain in chains:
        for side in chain['sides']:
            size = np.asarray(Image.open(files[side['name']]).size)
            scale = size / np.asarray(side['image_size'])
            points = np.asarray(side['points']); normal = np.asarray(side['normals'])
            new_normal = normal / scale
            new_normal /= np.linalg.norm(new_normal, axis=1)[:, None]
            side['points'] = ((points + .5) * scale - .5).tolist()
            side['normals'] = new_normal.tolist()
            side['px_per_unit'] *= float(np.median(np.sum(normal * scale * new_normal, axis=1)))
            side['image_size'] = size.tolist()
    return chains


def seam_engine():
    scripts = Path(__file__).resolve().parents[1]/'skills/phone-aluminum-frame-restore/scripts'
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    return importlib.import_module('match_uv_seams')


def terminal_metal(work, face, source_sha256):
    folder = Path(work)/'material-inputs'/face
    mask = folder/'seam-terminal-metal.png'
    proof_path = mask.with_suffix('.json')
    if face not in ('left','right') or not mask.is_file() or not proof_path.is_file():
        return None
    proof = json.loads(proof_path.read_text())
    if (proof.get('scope') != 'only_outside_both_source_antennas' or proof.get('face') != face
            or proof.get('source_sha256') != source_sha256 or proof.get('mask_sha256') != digest(mask)):
        raise ValueError('Terminal seam material evidence differs from the selected source')
    return mask


def run(work, sku, model, output, records, *, model_sha256=None, terminal_sources=None):
    data, profile_path = profile(model, model_sha256)
    work, output, records = Path(work), Path(output), Path(records)
    fresh(output); fresh(records); records.mkdir(parents=True)
    files = {f: work/'structure-records/candidate'/sku/(f+'.png') for f in FACES}
    light = {f: work/'lighting'/sku/(f+'.png') for f in FACES}
    spec = json.loads((work/'structure-spec.json').read_text())[sku]
    views = {}
    for face in ('left','right','top','bottom'):
        # Source-derived classification travels through the exact same
        # source-to-target mapping as the image, never reclassified by output.
        material = work/'material-inputs'/face
        mask_path = material/'main_surface.png'
        if not mask_path.is_file() and face in ('left','right'):
            if digest(light[face]) != spec[face]['source_sha256']:
                raise ValueError('Terminal metal mapping differs from its aligned lighting source')
            mask_path = terminal_metal(work, face, (terminal_sources or {}).get(face))
        if mask_path is None or not mask_path.is_file():
            raise ValueError('Missing source-verified frame metal: '+face)
        mask = Image.open(mask_path).convert('RGBA')
        if face in ('left','right'):
            source, target, _, _ = sr.mapping(light, face, spec[face])
            mask = sr.warp_rows(mask, source, target)
        mask_array = np.asarray(mask)[...,0] >= 254
        mask_path = records/(face+'-metal.png')
        Image.fromarray(mask_array.astype('uint8')*255).save(mask_path)
        views[face] = dict(source=str(files[face]), editable_mask=str(mask_path),
                           protected_mask=str(material/'paper.png'))
    chains_path = records/'chains.json'
    write(chains_path, dict(chains=resized_chains(data, files)))
    # Explicit paired UVs don't need a second mesh extraction at runtime.
    write(records/'scene.json', {})
    job = dict(scene=str(records/'scene.json'), seam_chains=str(chains_path),
        views=views, output_dir=str(records/'matched-frames'), records_dir=str(records/'engine'),
        width_units=.20, sigma=3, guard_pixels=18, protection_feather=0,
        max_lab_shift=[25,15,15], minimum_joint_coverage=.98,
        maximum_p95_deltaE76=2.0, sampling_refinements=3 if terminal_sources else 2,
        maximum_correction_step_deltaE76=3.0,
        # 2026-11-09: rounded rim ends can make a steep step; smooth along the
        # seam first, else apply with a hint up to 8 (user: effect issues hint).
        continuity_retry_sigmas=[2, 4], continuity_warning_limit_deltaE76=8.0)
    write(records/'job.json', job)
    module = seam_engine()
    with contextlib.redirect_stdout(io.StringIO()):
        report = module.run(records/'job.json')
    from .frame_end_color import finish as finish_end_color
    masks = {f: (np.array(Image.open(v['editable_mask'])) > 0)
                & (np.array(Image.open(v['protected_mask']).convert('L')) == 0) for f,v in views.items()}
    end_color = finish_end_color(records/'matched-frames', output, records/'end-color', masks, resized_chains(data, files),
                                 sigma=job['sigma'], width_units=job['width_units'])
    if end_color['status'] == 'applied':
        report['initial_seams'] = report['seams']
        report['seams'] = end_color['seams']
        resolution = end_color.get('continuity_resolution') or {}
        if resolution.get('warning_limit'):
            report['initial_continuity_resolution'] = report.get('continuity_resolution')
            report['continuity_resolution'] = resolution
    report['end_color'] = end_color
    for face in ('front','back'):
        shutil.copy2(files[face], output/(face+'.png'))
    # Local RGB correction cannot touch antennas, hardware, paper or alpha.
    for face in ('left','right'):
        before = np.asarray(Image.open(files[face]).convert('RGBA'))
        after = np.asarray(Image.open(output/(face+'.png')).convert('RGBA'))
        for lo, hi in spec[face]['button_ranges']:
            if not np.array_equal(before[lo:hi], after[lo:hi]):
                raise ValueError('UV seam correction reached button pixels')
    report.update(model=model, model_sha256=data['model_sha256'],
        profile_sha256=digest(profile_path), profile_path=str(profile_path),
        source_sha256={f:digest(p) for f,p in files.items()},
        output_sha256={f:digest(output/(f+'.png')) for f in FACES},
        status='four_frame_uv_color_seams_passed',
        scope='Actual inset UV of the bundled model; other model UVs require a new profile',
        model_geometry_changed=False, model_uv_changed=False)
    write(records/'report.json', report)
    return report
