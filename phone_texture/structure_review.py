"""Source-bound image-space antenna remapping and final delivery verification.

Image landmarks are reviewed input, not automatic semantic recognition or UV proof.
All final checks are rerun from pixels; a successful old report is insufficient.
"""
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from texture_alignment.align_textures import anchor_matrix, warp_rows
from texture_alignment.image_reference_targets import rows_from_view
from .pipeline import batch_files, separate_paths, fresh, overview
from .geometry_review import digest, write

TOLERANCE = 2.5
CHECKS = ('upper_edges_and_width', 'lower_edges_and_width', 'both_endpoints',
          'buttons_and_card_tray', 'body_outline', 'native_and_4x')


def read(path):
    return json.loads(Path(path).read_text())


def pixels(path):
    with Image.open(path) as im:
        if im.format != 'PNG' or im.mode != 'RGBA':
            raise ValueError('Structure input/output must be RGBA PNG')
        return np.array(im)


def intervals(value, height):
    a = np.asarray(value, float)
    if a.ndim != 2 or a.shape[1] != 2 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Reviewed nonempty button ranges required')
    if np.any(a != np.floor(a)) or np.any(a[:, 0] < 0) or np.any(a[:, 1] > height) or np.any(a[:, 0] >= a[:, 1]):
        raise ValueError('Invalid button ranges')
    return a.astype(int)


def detect(path, x_range, windows, signs):
    a = pixels(path); h, w = a.shape[:2]
    if len(x_range) != 2 or any(type(v) is not int for v in x_range):
        raise ValueError('Integer measurement columns required')
    x0, x1 = x_range
    if not 0 <= x0 < x1 <= w or x1-x0 < 3:
        raise ValueError('Measurement columns outside image')
    boxes = np.asarray(windows, float)
    if boxes.shape != (4, 2) or not np.isfinite(boxes).all() or np.any(boxes != np.floor(boxes)):
        raise ValueError('Four integer antenna boundary search windows required')
    boxes = boxes.astype(int)
    if np.any(boxes[:, 0] < 1) or np.any(boxes[:, 1] >= h-1) or np.any(boxes[:, 1]-boxes[:, 0] < 3) or np.any(boxes[1:, 0] < boxes[:-1, 1]):
        raise ValueError('Boundary search windows must be ordered, nonoverlapping and inside image')
    if len(signs) != 4 or any(v not in (-1, 1) for v in signs):
        raise ValueError('Reviewed edge polarities required')
    signal = np.median(a[:, x0:x1, :3].astype(float).mean(axis=2), axis=1)
    g = np.gradient(gaussian_filter1d(signal, 1.0))
    rows = []
    for (lo, hi), sign in zip(boxes, signs):
        row = int(lo + np.argmax(sign*g[lo:hi]))
        if row in (lo, hi-1) or sign*g[row] < 0.5 or np.median(a[row, x0:x1, 3]) < 128:
            raise ValueError('Ambiguous antenna edge; review native pixels/search region')
        rows.append(row)
    return np.asarray(rows, float)


def reference(files, ref, side_height, body):
    face = ref['face']
    if face not in ('front', 'back') or digest(files[face]) != ref['sha256']:
        raise ValueError('Reference must bind this SKU current front/back image')
    view = dict(ref, path=str(files[face]))
    target = rows_from_view(view, side_height, body)
    measured = detect(files[face], ref['x_range'], ref['search_windows'], ref['signs'])
    if np.max(np.abs(measured-np.asarray(ref['antenna_rows']))) > TOLERANCE:
        raise ValueError('Front/back landmarks disagree with actual reference pixels')
    return target


def mapping(files, side, cfg):
    src = files[side]; a = pixels(src); h, w = a.shape[:2]
    if digest(src) != cfg['source_sha256']:
        raise ValueError('Side landmarks belong to another source')
    source = np.asarray(cfg['source_lines'], float)
    # anchor_matrix verifies ordered finite per-column coordinates.
    anchor_matrix(source, h, w, 'source')
    if source.ndim == 1:
        source = np.repeat(source[:, None], w, axis=1)
    indices = cfg['antenna_indices']
    if len(indices) != 4 or any(type(v) is not int for v in indices) or sorted(set(indices)) != indices or indices[0] < 0 or indices[-1] >= len(source):
        raise ValueError('Four ordered antenna indices required')
    back = cfg['back']
    if back.get('face') != 'back':
        raise ValueError('Each side requires its own SKU back reference')
    rows = reference(files, back, h, cfg['body_y'])
    target_rows = np.repeat(rows[:, None], w, axis=1)
    basis = cfg.get('basis', 'back_image')
    if basis == 'front_back_images':
        front = cfg['front']
        if front.get('face') != 'front' or cfg.get('front_at') not in ('left', 'right'):
            raise ValueError('Explicit front/back side orientation required')
        f = reference(files, front, h, cfg['body_y'])
        t = np.linspace(0, 1, w)
        if cfg['front_at'] == 'right':
            t = 1-t
        target_rows = f[:, None]*(1-t)+rows[:, None]*t
    elif basis != 'back_image':
        raise ValueError('This gate verifies image space; model UV needs separate scene validation')
    target = source.copy(); target[indices] = target_rows
    anchor_matrix(target, h, w, 'target')
    buttons = intervals(cfg['button_ranges'], h)
    if any(lo <= max(source[indices[1]].max(), target[indices[1]].max()) or hi >= min(source[indices[2]].min(), target[indices[2]].min()) for lo, hi in buttons):
        raise ValueError('Button ranges must sit between the two antennas')
    if not cfg.get('button_basis'):
        raise ValueError('Independent button review basis required')
    return source, target, target_rows, buttons


def verify(before, after, spec):
    if set(before) != set(after) or set(spec) != set(before):
        raise ValueError('Structure SKU set differs')
    result = {}
    for sku, files in before.items():
        if set(spec[sku]) != {'left', 'right'}:
            raise ValueError('Both sides need independent structure checks')
        result[sku] = {}
        for face in ('front', 'back', 'top', 'bottom'):
            if digest(files[face]) != digest(after[sku][face]):
                raise ValueError('Structure remapping changed companion face: '+face)
        for side in ('left', 'right'):
            cfg = spec[sku][side]; source, target, targets, buttons = mapping(files, side, cfg)
            a, b = pixels(files[side]), pixels(after[sku][side]); h, w = a.shape[:2]
            if a.shape != b.shape:
                raise ValueError('Structure changed canvas dimensions')
            expected = np.asarray(warp_rows(Image.fromarray(a), source, target))
            if not np.array_equal(expected, b):
                raise ValueError('Output does not match reviewed source-pixel remapping')
            for lo, hi in buttons:
                if not np.array_equal(a[lo:hi], b[lo:hi]):
                    raise ValueError('Structure remapping changed button/card tray pixels')
            measurements = cfg['measurements']
            if set(measurements) != {'left_end', 'center', 'right_end'}:
                raise ValueError('Both endpoints and center need pixel checks')
            checks = {}
            for name, m in measurements.items():
                x0, x1 = m['x_range']
                if (name == 'left_end' and x1 > w*.25) or (name == 'right_end' and x0 < w*.75) or (name == 'center' and not x0 <= w/2 <= x1):
                    raise ValueError('Endpoint/center witness in wrong location')
                measured = detect(after[sku][side], m['x_range'], m['search_windows'], m['signs'])
                desired = np.median(targets[:, x0:x1], axis=1)
                error = np.abs(measured-desired)
                width = np.abs(np.diff(measured.reshape(2, 2), axis=1)-np.diff(desired.reshape(2, 2), axis=1)).ravel()
                if max(error) > TOLERANCE or max(width) > TOLERANCE:
                    raise ValueError(f'Antenna position/width mismatch: {sku}/{side}/{name}')
                checks[name] = dict(measured_rows=measured.tolist(), target_rows=desired.tolist(), error_px=error.tolist(), width_error_px=width.tolist())
            result[sku][side] = dict(input_sha256=digest(files[side]), output_sha256=digest(after[sku][side]),
                                     checks=checks, button_pixels_unchanged=True)
    return result


def evidence(before, after, spec, process):
    artifacts = []
    for sku in before:
        for side in ('left', 'right'):
            _, _, targets, _ = mapping(before[sku], side, spec[sku][side])
            for band, rows in enumerate((targets[:2], targets[2:])):
                for bg in ('#222222', '#eeeeee'):
                    crops = []
                    for files in (before, after):
                        im = Image.open(files[sku][side]).convert('RGBA')
                        # Include original AND target boundaries in the same source-coordinate crop.
                        src_lines=np.asarray(spec[sku][side]['source_lines'])
                        idx=spec[sku][side]['antenna_indices'][band*2:band*2+2]
                        lo=int(min(rows.min(),src_lines[idx].min()))-24
                        hi=int(max(rows.max(),src_lines[idx].max()))+25
                        crop=im.crop((0,max(0,lo),im.width,min(im.height,hi)))
                        tile=Image.new('RGB',crop.size,bg);tile.paste(crop,mask=crop.getchannel('A'));crops.append(tile)
                    for scale in (1,4):
                        sheet=Image.new('RGB',((crops[0].width+crops[1].width)*scale+12,crops[0].height*scale),bg)
                        for i,c in enumerate(crops):sheet.paste(c.resize((c.width*scale,c.height*scale),Image.Resampling.NEAREST),(i*(c.width*scale+12),0))
                        dest=process/f'{sku}-{side}-{band}-{bg[1:]}-{scale}x.png';sheet.save(dest)
                        artifacts.append(dict(path=str(dest),sha256=digest(dest)))
    overview(before, after, process/'overview.png', faces=('left','right'))
    artifacts.append(dict(path=str(process/'overview.png'),sha256=digest(process/'overview.png')))
    return artifacts


def prepare(input_dir, spec_path, process):
    input_dir, process = separate_paths(input_dir, process);fresh(process)
    before = batch_files(input_dir);spec=read(spec_path)
    # Validate all sources before creating candidates.
    if set(spec)!=set(before):raise ValueError('Structure SKU set differs')
    for sku,files in before.items():
        if set(spec[sku])!={'left','right'}:raise ValueError('Both sides required')
        for side in ('left','right'):mapping(files,side,spec[sku][side])
    process.mkdir(parents=True);write(process/'spec.json',spec)
    for sku, files in before.items():
        folder=process/'candidate'/sku;folder.mkdir(parents=True)
        for face,src in files.items():
            if face in ('left','right'):
                source,target,_,_=mapping(files,face,spec[sku][face])
                warp_rows(Image.open(src),source,target).save(folder/(face+'.png'))
            else:shutil.copy2(src,folder/(face+'.png'))
    after=batch_files(process/'candidate');checks=verify(before,after,spec)
    artifacts=evidence(before,after,spec,process)
    report=dict(status='needs_visual_review',spec_sha256=digest(process/'spec.json'),
        inputs={s:{f:dict(path=str(p),sha256=digest(p)) for f,p in fs.items()} for s,fs in before.items()},
        outputs={s:{f:digest(p) for f,p in fs.items()} for s,fs in after.items()},checks=checks,artifacts=artifacts,model_uv_verified=False)
    write(process/'report.json',report)
    write(process/'review-template.json',dict(report_sha256=digest(process/'report.json'),reviewer='',
        items={s:{side:{**{c:False for c in CHECKS},'notes':''} for side in ('left','right')} for s in before}))
    return report


def verify_delivery(before, process, review_path):
    process=Path(process).resolve();report=read(process/'report.json');spec=read(process/'spec.json');review=read(review_path)
    if digest(process/'spec.json')!=report['spec_sha256'] or review['report_sha256']!=digest(process/'report.json') or not review.get('reviewer'):
        raise ValueError('Structure review is missing or stale')
    if set(report['inputs'])!=set(before) or set(review['items'])!=set(before):raise ValueError('Structure review SKU set differs')
    for sku,files in before.items():
        for f,p in files.items():
            if digest(p)!=report['inputs'][sku][f]['sha256']:
                raise ValueError('Structure was not run on these lighting outputs')
        for side in ('left','right'):
            r=review['items'][sku][side]
            if not r.get('notes') or any(r.get(c) is not True for c in CHECKS):raise ValueError('Incomplete structure visual review')
    for item in report['artifacts']:
        if digest(item['path'])!=item['sha256']:raise ValueError('Structure visual evidence changed')
    after=batch_files(process/'candidate')
    if {s:{f:digest(p) for f,p in fs.items()} for s,fs in after.items()}!=report['outputs']:raise ValueError('Structure output changed after review')
    checks=verify(before,after,spec)
    return after,dict(status='image_space_passed',checks=checks,report_sha256=digest(process/'report.json'),
                      review_sha256=digest(review_path),model_uv_verified=False)
