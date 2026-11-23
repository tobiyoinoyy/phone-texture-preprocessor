"""Same-device frame-end paint correction on registered metal, before final UV QA."""
import contextlib
import io
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d

from . import lighting


def smoothstep(value):
    value = np.clip(value, 0., 1.)
    return value * value * (3 - 2 * value)


def correct(rgba, metal):
    """Estimate only broad end illumination from opaque, central-width metal strips."""
    engine = lighting.engine('aluminum')
    vertical = rgba.shape[0] > rgba.shape[1]
    pixels = rgba.swapaxes(0, 1) if vertical else rgba
    metal = metal.T if vertical else metal
    height, length = metal.shape
    if pixels.shape[:2] != metal.shape:
        raise ValueError('Frame-end material coordinates differ from image')
    lab = engine.rgb_to_lab(pixels[..., :3])
    fit = metal & (pixels[..., 3] >= 250)
    fit[:int(.22 * height)] = False
    fit[int(.78 * height):] = False
    # Equal votes per longitudinal window prevent one large unoccluded patch
    # from dominating the reference. Localized hardware and wear are outliers.
    block = max(4, round(min(.018 * length, .15 * height)))
    centers, colors = [], []
    for x in range(0, length, block):
        zone = fit[:, x:x + block]
        if zone.sum() < max(16, .14 * height * block):
            continue
        centers.append(x + (zone.shape[1] - 1) / 2)
        colors.append(np.median(lab[:, x:x + block][zone], axis=0))
    centers, colors = np.array(centers), np.array(colors)
    out = pixels.copy()
    report = []
    width = min(.155 * length, 1.65 * height)
    for reverse in (False, True):
        sign = 'end' if reverse else 'start'
        distance = length - 1 - centers if reverse else centers
        ref = (distance >= width * 1.12) & (distance <= .49 * length)
        near = (distance <= width * .62)
        if ref.sum() < 4 or near.sum() < 3:
            report.append(dict(end=sign, status='insufficient_metal'))
            continue
        reference = np.median(colors[ref], axis=0)
        stable = ref & (np.linalg.norm(colors - reference, axis=1) <= 8.)
        if stable.sum() < max(4, ref.sum() * .55):
            report.append(dict(end=sign, status='inconsistent_reference'))
            continue
        reference = np.median(colors[stable], axis=0)
        end = np.median(colors[near], axis=0)
        loss = reference[0] - end[0]
        tint = np.linalg.norm(reference[1:] - end[1:])
        info = dict(end=sign, reference=reference.tolist(), before=end.tolist(),
                    reference_windows=int(stable.sum()))
        # A bright silver end with small white-balance variations is already good.
        if loss < 7 and (tint < 7 or np.linalg.norm(end[1:]) > np.linalg.norm(reference[1:]) + 2):
            report.append(dict(info, status='within_tolerance'))
            continue
        # Require several contiguous metal observations all the way to the end.
        local = distance < width * 1.2
        points = np.sort(distance[local])
        if points[0] > block * 2 or np.max(np.diff(points), initial=0) > block * 3:
            report.append(dict(info, status='incomplete_end_support'))
            continue
        position = np.arange(length)
        d = length - 1 - position if reverse else position
        profile = np.column_stack([np.interp(position, centers, colors[:, c]) for c in range(3)])
        profile = gaussian_filter1d(profile, max(2., block * .80), axis=0, mode='nearest')
        offset = reference - profile
        offset[:, 0] = np.clip(offset[:, 0] - 2.5, 0, 24) * .84
        offset[:, 1:] *= np.minimum(1., 20 / np.maximum(1., np.linalg.norm(offset[:, 1:], axis=1)))[:, None] * .88
        offset *= smoothstep((width - d) / (.42 * width))[:, None]
        # A smooth RGB transfer preserves steps between nearby source values
        # even on saturated paint. Limit headroom compression near gamut limits.
        base = engine.lab_to_rgb(profile).astype(float)
        goal = engine.lab_to_rgb(profile + offset).astype(float)
        delta = goal - base
        rgb = pixels[..., :3].astype(float)
        # Distinct exposed material should not acquire the surrounding paint.
        chroma_distance = np.linalg.norm(lab[..., 1:] - profile[None, :, 1:], axis=2)
        weight = smoothstep((30 - chroma_distance) / 12)
        active = (d < width)[None, :] & metal & (pixels[..., 3] > 0)
        scale = np.where(delta >= 0, (255 - rgb) / np.maximum(255 - base, 1),
                         rgb / np.maximum(base, 1))
        rgb += weight[..., None] * delta * np.clip(scale, 0, 1.2)
        out[active, :3] = np.clip(np.rint(rgb[active]), 0, 255).astype('uint8')
        report.append(dict(info, status='corrected', extent_px=float(width)))
    return (out.swapaxes(0, 1) if vertical else out), report


def prepare_views(views, records):
    """Never replace source evidence; return seam-engine inputs and per-end decisions."""
    records = Path(records)
    records.mkdir(parents=True, exist_ok=True)
    prepared, decisions = {}, {}
    for face, view in views.items():
        rgba = np.array(Image.open(view['source']).convert('RGBA'))
        metal = np.array(Image.open(view['editable_mask']).convert('L')) > 0
        metal &= np.array(Image.open(view['protected_mask']).convert('L')) == 0
        candidate, decisions[face] = correct(rgba, metal)
        prepared[face] = dict(view)
        if not np.array_equal(candidate, rgba):
            path = records / (face + '.png')
            Image.fromarray(candidate).save(path)
            prepared[face]['source'] = str(path)
    return prepared, decisions


def finish(source, output, records, masks, chains, *, sigma, width_units):
    """Generate a separate four-face set; the first, verified seam pass stays immutable."""
    from . import workspace as w

    records.mkdir(parents=True, exist_ok=True)
    views = {}
    for face, mask in masks.items():
        path = source / (face + '.png')
        mp = records / (face + '-metal.png')
        pp = records / (face + '-excluded.png')
        Image.fromarray(mask.astype('uint8') * 255).save(mp)
        Image.fromarray((~mask).astype('uint8') * 255).save(pp)
        views[face] = dict(source=str(path), editable_mask=str(mp), protected_mask=str(pp))
    prepared, decisions = prepare_views(views, records / 'corrected')
    selected = {face: Path(view['source']) for face, view in views.items()}
    proof = dict(ends=decisions, baseline_sha256={f: w.sha(p) for f, p in selected.items()},
                 status='unchanged')
    if views != prepared:
        try:
            selected, checks = _verify_candidate(prepared, views, records, masks, chains,
                                                 decisions, sigma, width_units)
            proof.update(checks, status='applied')
        except ValueError as exc:
            # The first pass has already passed the model seam limits. A failed
            # optional colour refinement must not replace that verified set.
            proof.update(status='retained_baseline', reason=str(exc))
    if proof['status'] != 'retained_baseline':
        from .frame_end_saturation import finish as saturate
        proof['a1_sha256'] = {f: w.sha(p) for f, p in selected.items()}
        selected, saturation = saturate(selected, records/'saturation', masks, decisions, chains, sigma)
        proof['saturation'] = saturation
        if saturation['changed']:
            proof.update(status='applied', seams=saturation['seams'])
    output.mkdir(parents=True, exist_ok=False)
    for face, candidate in selected.items():
        shutil.copy2(candidate, output / (face + '.png'))
    proof['output_sha256'] = {f: w.sha(output / (f + '.png')) for f in views}
    return proof


def _verify_candidate(prepared, views, records, masks, chains, decisions, sigma, width_units):
    from . import workspace as w
    from .frame_seams import seam_engine
    from .iphone17_model_finish import _current_seam_readings

    proof = {}
    selected = {face: Path(view['source']) for face, view in prepared.items()}
    readings = _current_seam_readings(selected, chains, masks, sigma)
    needed = [chain for chain, check in zip(chains, readings) if check['after_p95_deltaE76'] > 2.]
    if needed:
        w.save(records / 'chains.json', dict(chains=needed))
        w.save(records / 'scene.json', {})
        destination = records / 'matched'
        job = dict(views=prepared, scene=str(records / 'scene.json'),
                   seam_chains=str(records / 'chains.json'), output_dir=str(destination),
                   records_dir=str(records / 'seam-engine'), width_units=width_units,
                   sigma=sigma, guard_pixels=18, protection_feather=0,
                   max_lab_shift=[25, 15, 15], minimum_joint_coverage=.98,
                   maximum_p95_deltaE76=2., sampling_refinements=3,
                   maximum_correction_step_deltaE76=3.,
                   continuity_retry_sigmas=[2, 4], continuity_warning_limit_deltaE76=8.)
        w.save(records / 'job.json', job)
        with contextlib.redirect_stdout(io.StringIO()):
            match = seam_engine().run(records / 'job.json')
        proof['continuity_resolution'] = match['continuity_resolution']
        selected = {f: destination / (f + '.png') for f in views}
        readings = _current_seam_readings(selected, chains, masks, sigma)
    if any(r['after_p95_deltaE76'] > 2. for r in readings):
        raise ValueError('Endpoint correction failed final four-corner colour verification')
    helper = lighting.engine('aluminum')
    for face, candidate in selected.items():
        before = np.array(Image.open(views[face]['source']).convert('RGBA'))
        after = np.array(Image.open(candidate).convert('RGBA'))
        mask = masks[face] & (before[..., 3] > 0)
        if not np.array_equal(before[..., 3], after[..., 3]) or not np.array_equal(before[~mask], after[~mask]):
            raise ValueError('Frame-end correction changed geometry or nonmetal: ' + face)
        if before.shape[0] > before.shape[1]:
            before, after, mask = before.swapaxes(0, 1), after.swapaxes(0, 1), mask.T
        h, length = mask.shape
        width = min(.155 * length, 1.65 * h)
        for entry in decisions[face]:
            if entry['status'] != 'corrected':
                continue
            x = np.arange(length)
            d = length - 1 - x if entry['end'] == 'end' else x
            y = np.arange(h)[:, None]
            zone = mask & (before[..., 3] >= 250) & (d > .12 * width) & (d < .65 * width) & (y > .22*h) & (y < .78*h)
            if zone.sum() < 16:
                raise ValueError('Insufficient independent endpoint samples: ' + face)
            ref = np.array(entry['reference'])
            for label, image in [('before', before), ('after', after)]:
                patch = np.median(helper.rgb_to_lab(image[zone, :3]), axis=0)
                entry['distance_' + label] = float(np.linalg.norm(patch - ref))
                entry['sample_' + label] = patch.tolist()
            if entry['distance_after'] >= entry['distance_before']:
                raise ValueError('Endpoint did not approach its same-metal reference: ' + face)
    proof['seams'] = readings
    return selected, proof


def effect_notes(proof):
    if proof.get('status') == 'retained_baseline':
        return [dict(code='frame_end_color_retained', face='frame', error=proof['reason'],
                     message='端部校色复测未通过，沿用已通过接缝检查的边框')]
    notes = []
    reason = proof.get('saturation', {}).get('reason')
    if reason:
        notes.append(dict(code='frame_saturation_retained', face='frame', error=reason,
                          message='端部增色未通过四角复测，沿用 A1 颜色'))
    for face, ends in proof.get('ends', {}).items():
        for end in ends:
            if end['status'] in ('inconsistent_reference', 'insufficient_metal', 'incomplete_end_support'):
                label = ('左端', '右端') if face in ('bottom', 'top') else ('上端', '下端')
                notes.append(dict(face=face, code='frame_end_reference_uncertain', error=end['status'],
                                  message=(label[1] if end['end'] == 'end' else label[0]) + '同材质取色依据不足，沿用原端部颜色'))
    return notes
