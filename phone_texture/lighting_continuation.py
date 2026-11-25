"""Bounded, source-only lighting continuation for a verified six-view crop."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy import ndimage

from . import workspace as w

FRAMES = {'left', 'right', 'top', 'bottom'}


def _mask(path, shape):
    if not path.is_file():
        return None
    try:
        with Image.open(path) as image:
            mask = np.asarray(image.convert('L')) > 0
    except UnidentifiedImageError as exc:
        raise ValueError('Lighting fallback material mask is unreadable: ' + path.name) from exc
    if mask.shape != shape:
        raise ValueError('Lighting fallback material mask does not match the source crop')
    return mask


def _rear_exterior(source, output, material):
    from . import camera_metal
    from .lighting import engine

    if material is None or not material.any():
        return dict(status='source_evidence_unavailable', reason='缺少本张背图的凸台金属区域', applied_pixels=0)
    try:
        plan = camera_metal.measure(source, 'triple')
    except ValueError as error:
        return dict(status='source_evidence_unavailable', reason=str(error), applied_pixels=0)
    render = engine('aluminum')
    raw = render.rgb_to_lab(source[..., :3])
    lit = render.rgb_to_lab(output[..., :3])
    target, active, report = camera_metal.apply(source, raw, lit, material, plan, exterior_only=True)
    band = output[:len(active), :, :3]
    proposed, limited = render.bounded_lab_offset(lit[:len(active)][active], target[active])
    encoded = render.lab_to_rgb(proposed)
    changed = int(np.count_nonzero(np.any(encoded != band[active], axis=1)))
    band[active] = encoded
    return dict(report, status='applied' if changed else 'no_supported_change',
                applied_pixels=changed, gamut_limited_pixels=limited)


def recover(source, face, output, assets, *, metal_back=False, review_camera=False):
    """Apply mild correction only within reliable same-material regions."""
    source, output, assets = map(Path, (source, output, assets))
    if output.exists():
        raise ValueError('Lighting fallback needs a new output path')
    digest = w.sha(source)
    with Image.open(source) as image:
        a = np.asarray(image.convert('RGBA'))
    h, width = a.shape[:2]
    rgb = a[..., :3]
    opaque = a[..., 3] >= 240
    names = ('rear_perimeter', 'main_surface') if face == 'back' and metal_back else (
        ('rear_perimeter',) if face == 'back' else ('main_surface',))
    masks = [mask for name in names if (mask := _mask(assets/(name+'.png'), opaque.shape)) is not None]
    origin = 'current_photo_material_masks'
    if not masks and face == 'back' and metal_back:
        provisional = _mask(assets/'rear_perimeter-provisional.png',opaque.shape)
        if provisional is not None:
            masks = [provisional]
            origin = 'current_photo_verified_rear_rim_without_panel_corner'
    if not masks and face == 'front':
        from .auto_material import front_metal
        metal, _ = front_metal(Image.fromarray(a), allow_partial=True)
        masks = [metal]
        origin = 'current_photo_individually_confirmed_front_edges'
    if not masks and face == 'back' and not metal_back:
        from .iphone17_material import glass_back_rim
        from .auto_material import paper_mask
        photograph = Image.fromarray(a)
        paper_guess, _ = paper_mask(photograph, face, resolve_end=False, source_tail=False)
        metal, _, _ = glass_back_rim(photograph, paper_guess, allow_partial=True)
        masks = [metal]
        origin = 'current_photo_individually_confirmed_rear_edges'
    review_camera = review_camera and metal_back and face == 'back'
    camera = _mask(assets/'camera_surround.png', opaque.shape) if review_camera else None
    camera_basis = 'source_material_mask'
    if review_camera and camera is None:
        # A failed rear-panel or antenna check may precede creation of the
        # normal material mask. Independently measure the plateau on this crop.
        from .rear_shadow import _plateau_core, _ring_material, InsufficientRearShadowEvidence
        from .auto_material import rear_camera_glass
        camera_basis = 'source_plateau_edges'
        if min(h, width) >= 100 and opaque.any():
            light = cv2.cvtColor(rgb.astype('float32') / 255, cv2.COLOR_RGB2LAB)[..., 0]
            try:
                bounds, edges = _plateau_core(light)
            except InsufficientRearShadowEvidence:
                pass
            else:
                glass, _ = rear_camera_glass(a, opaque)
                yy, xx = np.ogrid[:h, :width]
                camera = _ring_material(opaque & ~glass, dict(flat_core_bounds=bounds,
                    plateau_source_edges=edges, lenses=[]), yy, xx)
    if not masks and not (camera is not None and camera.any()):
        if face not in FRAMES or min(h, width) < 18 or not 3 <= max(h, width)/min(h, width) <= 24:
            raise ValueError('No independently supported fallback metal area')
        # Both rails lie inside the opaque crop, away from the central keys,
        # sockets and speaker holes. Local color/edge checks reject mixed areas.
        cross = h if face in ('top', 'bottom') else width
        length = width if face in ('top', 'bottom') else h
        cross_axis = np.arange(cross)/max(1, cross-1)
        long_axis = np.arange(length)/max(1, length-1)
        rails = ((.13 < cross_axis) & (cross_axis < .37)) | ((.63 < cross_axis) & (cross_axis < .87))
        interior = (.07 < long_axis) & (long_axis < .93)
        strip = rails[:, None] & interior if face in ('top', 'bottom') else interior[:, None] & rails
        masks = [strip]
        origin = 'same_frame_opaque_photographic_rails'
    paper = _mask(assets/'paper.png', opaque.shape)
    if paper is None:
        paper = _mask(assets/'printed-label-material.png', opaque.shape)
    if paper is None and face in ('back', 'bottom'):
        # No end geometry is required from the optional print cue. It never
        # blocks the primary processing route or invents a paper boundary.
        from .auto_material import paper_mask
        paper, _ = paper_mask(Image.fromarray(a), face, resolve_end=False, source_tail=False)
    paper = np.zeros(opaque.shape, bool) if paper is None else paper
    paper = ndimage.binary_dilation(paper, iterations=2)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype('float32')
    sigma = (max(2., h*.018), max(2., width*.018))
    narrow_edges = origin in ('current_photo_individually_confirmed_front_edges',
                              'current_photo_individually_confirmed_rear_edges',
                              'current_photo_verified_rear_rim_without_panel_corner')
    if narrow_edges:
        # Blur along a known rail, not across the narrow metal into glass.
        middle = ((np.arange(h)>.17*h) & (np.arange(h)<.83*h))[:,None]
        fields = [(mask & middle,(sigma[0],1.2)) for mask in masks]
        fields += [(mask & ~middle,(1.2,sigma[1])) for mask in masks]
    else:
        fields = [(mask,sigma) for mask in masks]
    out = a.copy()
    changed_total = 0
    claimed = np.zeros(opaque.shape, bool)
    measurements = []
    for mask, smoothing in fields:
        region = mask & opaque & ~paper & ~claimed
        depth = ndimage.distance_transform_edt(region)
        safe = depth >= (1 if narrow_edges else 2)
        if safe.sum() < 100:
            continue
        field_weight = ndimage.gaussian_filter(safe.astype('float32'), smoothing)
        field = ndimage.gaussian_filter(np.where(safe, gray, 0), smoothing)/np.maximum(field_weight, 1e-5)
        valid = safe & (field_weight >= .28) & (abs(gray-field) <= 20)
        if valid.sum() < 100:
            continue
        pts = field[valid]
        lo, hi = np.percentile(pts, [20, 72])
        if hi-lo < 2.5:
            continue
        fade = np.clip(depth/4 if narrow_edges else (depth-1)/5, 0, 1)
        lift = np.clip((hi-field)*.40, 0, 8)*fade
        lift[~valid] = 0
        room = 255 - rgb.max(axis=2)
        delta = np.minimum(np.rint(lift), room).astype('uint8')
        changed = delta > 0
        if changed.sum() < 80:
            continue
        out[..., :3][changed] = rgb[changed] + delta[changed, None]
        claimed |= changed
        changed_total += int(changed.sum())
        measurements.append(dict(samples=int(valid.sum()), edited=int(changed.sum()),
                                 measured_source_range=float(hi-lo), max_lift=int(delta.max())))
    extra = {}
    if review_camera:
        before_camera = out.copy()
        region = camera & opaque & ~paper if camera is not None else None
        camera_review = _rear_exterior(a, out, region)
        if region is not None and np.any(out[~region] != before_camera[~region]):
            raise RuntimeError('Camera lighting escaped its verified metal region')
        camera_review['material_basis'] = camera_basis
        extra['camera_metal'] = camera_review
        changed_total = int(np.count_nonzero(np.any(a != out, axis=2)))
    if not changed_total:
        raise ValueError('Fallback found no support for a nonzero lighting correction')
    if not np.array_equal(a[..., 3], out[..., 3]) or np.any(out[paper] != a[paper]) or w.sha(source) != digest:
        raise RuntimeError('Lighting fallback failed source and pixel integrity checks')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as f:
        Image.fromarray(out).save(f, format='PNG')
    with Image.open(output) as image:
        if not np.array_equal(np.asarray(image), out):
            raise RuntimeError('Lighting fallback PNG changed on round trip')
    return dict(method=origin, status='degraded_lighting', changed_pixels=changed_total,
                max_rgb_lift=int(np.max(out[..., :3].astype(np.int16) - rgb)) if review_camera else 8,
                measurements=measurements, **extra, original_alpha_unchanged=True,
                source_sha256=digest, lighting_sha256=w.sha(output), visual_review_passed=False)
