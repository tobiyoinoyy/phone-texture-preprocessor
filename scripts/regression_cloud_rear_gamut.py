"""Replay one saved cloud back-lighting failure without segmentation or publishing.

The case contains process/cases/SKU_01/work and 02_lighting_intermediates/back.png.
Inputs are read-only; the output directory must be new. No inference/network calls.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from phone_texture.auto_material import REAR_GAMUT_CONTINUITY_STEP_L
from phone_texture.lighting import dependencies, engine
from phone_texture.lighting_review import (digest, evidence_images, lightness,
                                           measure_perimeter, read, write)


def relocate(value, work):
    if isinstance(value, dict):
        return {key: relocate(child, work) for key, child in value.items()}
    if isinstance(value, list):
        return [relocate(child, work) for child in value]
    marker = '/process/cases/SKU_01/work/'
    if isinstance(value, str) and marker in value:
        return str(work / value.split(marker, 1)[1])
    return value


def added_steps(source_L, output_L, spec):
    result = {}
    for name, zone in spec['local_checks'].items():
        if 'max_added_step_L' not in zone:
            continue
        mask = np.asarray(Image.open(zone['mask']).convert('L')) > 0
        axis = zone.get('step_axis', 0)
        pairs = (mask[1:] & mask[:-1]) if axis == 0 else (mask[:, 1:] & mask[:, :-1])
        diff = np.abs(np.diff(output_L - source_L, axis=axis))
        if axis == 1:
            pairs, diff = pairs.T, diff.T
        rows = [(i, float(np.percentile(d[m], 75)), int(m.sum()))
                for i, (d, m) in enumerate(zip(diff, pairs)) if m.sum() >= 10]
        line, value, count = max(rows, key=lambda entry: entry[1])
        result[name] = dict(max_line_p75_L=value, line=line, line_pair_count=count,
                            limit_L=zone['max_added_step_L'])
    return result


def run(case, output):
    case, output = Path(case).resolve(), Path(output).resolve()
    work = case / 'process/cases/SKU_01/work'
    output.mkdir(parents=True, exist_ok=False)
    old_job = relocate(read(work / 'material-inputs/back/job.json'), work)
    spec = relocate(read(work / 'material-inputs/back/lighting-scope.json'), work)
    source = Path(old_job['views'][0]['source'])
    baseline = case / '02_lighting_intermediates/back.png'
    inputs = dependencies(old_job) | dependencies(spec)
    for face in ('front', 'back', 'left', 'right', 'top', 'bottom'):
        path = case / '01_geometry_intermediates' / (face + '.png')
        inputs[str(path)] = digest(path)
    inputs[str(baseline)] = digest(baseline)
    job = copy.deepcopy(old_job)
    view = job['views'][0]
    for domain in view['domains']:
        if domain['name'] in ('camera_surround', 'main_surface'):
            domain['gamut_continuity_step_L'] = REAR_GAMUT_CONTINUITY_STEP_L
    # These are replay parameters, not a workflow release or auto resume.
    job.pop('output_dir', None); job.pop('records_dir', None)
    write(output / 'replay-job.json', job)
    write(output / 'unchanged-scope.json', spec)
    module = engine('aluminum')
    target, _ = module.target_color(job)
    candidate, allowed, protected, proof = module.process_view(
        view, target, dict(job['defaults'], color_mode=job['color_mode']))
    path = output / 'back-candidate.png'
    Image.fromarray(candidate).save(path)
    new = np.asarray(Image.open(path).convert('RGBA'))
    src = np.asarray(Image.open(source).convert('RGBA'))
    old = np.asarray(Image.open(baseline).convert('RGBA'))
    assert np.array_equal(src[..., 3], new[..., 3])
    assert np.array_equal(src[protected], new[protected])
    assert np.array_equal(src[~allowed], new[~allowed])
    masks = {d['name']: np.asarray(Image.open(d['mask'])) > 0 for d in view['domains']}
    changed_domains = masks['camera_surround'] | masks['main_surface']
    assert np.array_equal(old[~changed_domains], new[~changed_domains])
    try:
        measure_perimeter(spec, source, baseline, allowed, protected)
    except ValueError as error:
        rejection = str(error)
        assert 'rear_upper_join_center' in rejection and 'discontinuity' in rejection
    else:
        raise AssertionError('Original cloud candidate unexpectedly passed')
    # The production gate and its frozen source masks/limits stay unchanged.
    metrics = measure_perimeter(spec, source, path, allowed, protected)
    source_L = lightness(src[..., :3])
    report = dict(case='SKU_01-cloud-20261102', source_sha256=digest(source),
        baseline_sha256=digest(baseline), candidate_sha256=digest(path),
        original_rejected=rejection, candidate_numeric_checks_passed=True,
        before=added_steps(source_L, lightness(old[..., :3]), spec),
        after=added_steps(source_L, lightness(new[..., :3]), spec),
        alpha_changed=0, protected_changed=0, outside_material_changed=0,
        outside_camera_and_main_changed_vs_baseline=0,
        changed_pixels_vs_baseline=int(np.any(old != new, axis=-1).sum()),
        metrics=metrics, processing=proof, input_hashes=inputs,
        visual_acceptance='pending user review', model_uv_verified=False,
        full_pipeline_rerun=False, published=False)
    evidence_images(spec, source, path, output / 'source-vs-candidate')
    joins = copy.deepcopy(spec)
    joins['zones'] = {}
    joins['local_checks'] = {k: v for k, v in spec['local_checks'].items() if k.startswith('rear_upper_join_')}
    evidence_images(joins, baseline, path, output / 'baseline-vs-candidate')
    assert all(digest(p) == sha for p, sha in inputs.items()), 'Read-only input changed'
    report['all_input_hashes_unchanged'] = True
    write(output / 'regression.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = run(args.case, args.output)
    print(json.dumps({key: report[key] for key in ('before', 'after', 'all_input_hashes_unchanged')}, indent=2))
