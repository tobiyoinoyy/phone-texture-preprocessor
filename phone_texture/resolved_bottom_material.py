"""Bind resolved bottom material evidence to current pixels, without approvals."""
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image

RESOLUTIONS=('original_source_closure','source_aligned_metal_window','source_contour_frame_domain')

def validate(view,evidence,scope=None):
    from .geometry_review import digest
    if view.get('name')!='bottom' or view.get('protection_policy')!='none':
        raise ValueError('Bottom material review requires an explicit unprotected bottom view')
    source=Path(view['source'])
    if digest(source)!=view.get('source_sha256'):raise ValueError('Bottom material source identity changed')
    if scope is not None and scope.get('source_sha256')!=digest(source):raise ValueError('Bottom material scope source identity changed')
    resolution=evidence.get('material_boundary_resolution')
    if resolution=='source_contour_frame_domain':
        from .source_material_bridge import validate_source_contour_material
        validate_source_contour_material(view,evidence,scope=scope)
        return dict(resolution=resolution,source_sha256=digest(source),binding='source_contour_material_validator')
    if resolution not in RESOLUTIONS:raise ValueError('Unresolved bottom material cannot enable normal review')
    with Image.open(source) as image:
        if image.mode!='RGBA' or image.format!='PNG':raise ValueError('Bottom material source must be RGBA PNG')
        a=np.asarray(image).copy()
    with Image.open(view['source_label_fit_exclusion_mask']) as image:label=np.asarray(image.convert('L'))>0
    if label.shape!=a.shape[:2] or not label.any():raise ValueError('Bottom material label mask must be source-sized and nonempty')
    aligned=evidence.get('aligned_source_closure',{})
    closure=evidence.get('unmarked_source_closure',{})
    raw_hash=aligned.get('original_source_rgba_sha256') or closure.get('source_rgba_sha256')
    mask_hash=aligned.get('original_mask_sha256') or closure.get('mask_sha256')
    if raw_hash!=hashlib.sha256(a.tobytes()).hexdigest():raise ValueError('Bottom closure evidence belongs to different source pixels')
    if mask_hash!=hashlib.sha256(label.tobytes()).hexdigest():raise ValueError('Bottom closure evidence belongs to a different material mask')
    domains={d['name']:d for d in view.get('domains',[])}
    if set(domains)!={'printed_label','main_surface'}:raise ValueError('Resolved bottom requires separate paper and frame domains')
    for name,expected in [('printed_label',label&(a[...,3]>0)),('main_surface',~label&(a[...,3]>0))]:
        with Image.open(domains[name]['mask']) as image:actual=np.asarray(image.convert('L'))>0
        if actual.shape!=expected.shape or not np.array_equal(actual,expected):raise ValueError('Resolved bottom domain differs from source closure: '+name)
    return dict(resolution=resolution,source_sha256=digest(source),source_RGBA_sha256=raw_hash,
        label_mask_sha256=mask_hash,binding='current_RGBA_and_material_mask_match_independent_source_closure',
        material_classification_recomputed=False,visual_approved=False)
