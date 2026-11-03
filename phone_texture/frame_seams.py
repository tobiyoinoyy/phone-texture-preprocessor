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


def profile(model):
    if model not in MODELS:
        raise ValueError('请选择模型：iPhone 17 Pro 或 iPhone 17 Pro Max')
    path = Path(__file__).with_name('seam_profiles') / (model + '.json')
    data = json.loads(path.read_text())
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


def run(work, sku, model, output, records):
    data, profile_path = profile(model)
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
        mask = Image.open(material/'main_surface.png').convert('RGBA')
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
        views=views, output_dir=str(output), records_dir=str(records/'engine'),
        width_units=.20, sigma=3, guard_pixels=18, protection_feather=0,
        max_lab_shift=[25,15,15], minimum_joint_coverage=.98,
        maximum_p95_deltaE76=2.0, sampling_refinements=2,
        maximum_correction_step_deltaE76=3.0)
    write(records/'job.json', job)
    scripts = Path(__file__).resolve().parents[1]/'skills/phone-aluminum-frame-restore/scripts'
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    module = importlib.import_module('match_uv_seams')
    with contextlib.redirect_stdout(io.StringIO()):
        report = module.run(records/'job.json')
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
