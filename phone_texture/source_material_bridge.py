"""Isolated pipeline adapter draft: source-closed paper on an aluminum bottom.

Consumes a frozen, source-bound v3 report. This is NOT a lighting validator,
does not change geometry, protect paper pixels, assign a paper lift budget, or
grant export/visual acceptance. Frozen r2/r3 code is not modified on import.
The new resolution requires validate_source_contour_material explicitly.
"""
import hashlib
import json
import math
from pathlib import Path

RESOLUTION = 'source_contour_frame_domain'
OLD_RESOLUTIONS = ('original_source_closure', 'source_aligned_metal_window')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def fail(message, evidence=None):
    error = ValueError('Bottom paper material boundary unresolved: '+message)
    error.material_boundary_evidence = evidence or {}
    raise error


def clip_segment_to_box(p, q, box):
    """Exact segment intersection; endpoints need not lie inside the box."""
    x0, y0, x1, y1 = map(float, box)
    px, py = map(float, p); qx, qy = map(float, q)
    dx, dy = qx-px, qy-py
    lo, hi = 0., 1.
    for direction, distance in [(-dx, px-x0), (dx, x1-px), (-dy, py-y0), (dy, y1-py)]:
        if abs(direction) < 1e-12:
            if distance < 0: return None
            continue
        ratio = distance/direction
        if direction < 0: lo = max(lo, ratio)
        else: hi = min(hi, ratio)
        if lo > hi: return None
    return [[px+lo*dx, py+lo*dy], [px+hi*dx, py+hi*dy]]


def _overlap_fraction(a, b):
    x0, y0, x1, y1 = a; u0, v0, u1, v1 = b
    area = max(0., min(x1, u1)-max(x0, u0))*max(0., min(y1, v1)-max(y0, v0))
    return area/max(1., (x1-x0)*(y1-y0))


def source_feature_role(feature, raw_contours):
    """Keep independent closed features; recognize source rail segments.

Uses the original compact-contour shape criteria, not an expanded peer mask.
Unknown Canny responses remain peer observations, never semantic material gates.
Independent closed source regions take priority over a long crossing rail.
"""
    box = list(map(float, feature['bbox']))
    closed = []
    rails = []
    for original in raw_contours:
        x, y, width, height = original['bbox']
        source_box = [x, y, x+width, y+height]
        # A real source-enclosed local feature is not erased by a rail crossing.
        if (width > 0 and height > 0 and .45 < width/height < 6
                and float(original['area']) >= max(12., width*height*.45)
                and _overlap_fraction(source_box, box) >= .75):
            closed.append(original['id'])
        points = original['points']
        for start, end in zip(points, points[1:]):
            dx, dy = end[0]-start[0], end[1]-start[1]
            if abs(dx) < 1e-12 or abs(dy/dx) > .15: continue
            if min(start[0], end[0]) > box[0]-2.5 or max(start[0], end[0]) < box[2]+2.5: continue
            clipped = clip_segment_to_box(start, end, [box[0], box[1], box[2]-1, box[3]-1])
            if clipped is not None:
                rails.append(dict(source_contour_id=original['id'], source_segment=[start, end],
                                  intersection=clipped, raw_source_area=original['area']))
    if feature.get('source_paper_edge_association_supported'):
        role = 'source_paper_boundary_response'
        closed = []  # Enclosed print in a source-proven paper domain is paper.
    elif closed:
        role = 'independent_closed_source_feature'
    elif rails:
        role = 'source_frame_rail_or_shadow_response'
    else:
        role = 'peer_only_unclassified_edge_response'
    return dict(role=role, independent_closed_source_ids=closed, source_rail_segments=rails,
                peer_avoidance_may_be_retained=True, peer_mask_is_semantic_material_gate=False)


def _check_v3(report, source_hash):
    if report.get('source_sha256') != source_hash or not report.get('input_dependencies_unchanged'):
        fail('Frozen source contour report belongs to another or changed source')
    if report.get('method') != 'source_gradient_connected_paper_contour_diagnostic_v3':
        fail('Expected the reviewed v3 source contour method')
    if not report.get('closed_paper_contour_supported') or report.get('failures'):
        fail('Source paper contour is not closed by the reviewed predicate', report)
    thresholds = report.get('thresholds', {})
    if any(thresholds.get(key) != value for key, value in [('gradient', .7), ('support', .75), ('residual_px', 2.5)]):
        fail('Source contour thresholds differ')
    for name in ('left', 'right'):
        side = report.get('sides', {}).get(name, {})
        corner = report.get('corners', {}).get(name, {})
        if (side.get('connected_source_supported_fraction', 0) < .75
                or not corner.get('route_passes_positive_turn')):
            fail('Missing actual connected source side/corner: '+name)
    if report.get('connected_transverse', {}).get('support', 0) < .75:
        fail('Insufficient source transverse edge support')
    if not report.get('source_U_path') or any(p.get('alpha', 0) < 240 for p in report['source_U_path']):
        fail('Source contour path lacks opaque source evidence')
    proposal_binding = report.get('material_cross_edge_proposal')
    if proposal_binding:
        if sha(proposal_binding['path']) != proposal_binding['sha256']:
            fail('Material cross-edge proposal evidence changed')
        proposal = read(proposal_binding['path'])
        from . import material_cross_edge
        material_cross_edge.validate_proposal_record(proposal, source_hash, report.get('cross_edge'))
        hint_hashes = [digest for path, digest in report.get('input_sha256_start', {}).items()
                       if Path(path).name == 'source-label-location-hint.png']
        if hint_hashes != [proposal.get('hint_sha256')]:
            fail('Material cross-edge proposal binds a different locator file')



def build_source_contour_material(source, hint_path, report_path, folder, *,
                                  surface_prior='aluminum_bottom_frame', legacy_error=None):
    """Return (paper material mask, verifiable evidence), never a rendered image.

This adapter is ready for isolated r4 calls, not automatically installed in the
pipeline. The original material gate must continue to reject the new enum until
its caller explicitly invokes the validator below.
"""
    import cv2
    import numpy as np
    from PIL import Image
    from scipy import ndimage
    if surface_prior != 'aluminum_bottom_frame': fail('Unsupported source surface prior')
    source, hint_path, report_path, folder = map(Path, (source, hint_path, report_path, folder))
    source_hash = sha(source); report = read(report_path)
    _check_v3(report, source_hash)
    bound = report.get('input_sha256_start', {})
    if sha(hint_path) not in {digest for path, digest in bound.items() if Path(path).name == 'source-label-location-hint.png'}:
        fail('Paper location hint is not bound to the source contour report')
    a = np.array(Image.open(source).convert('RGBA'))
    hint = np.array(Image.open(hint_path).convert('L')) > 0
    if hint.shape != a.shape[:2]: fail('Paper hint has another canvas')
    h, w = hint.shape
    line = report['cross_edge']['source_end_line']
    yy, xx = np.mgrid[:h, :w]
    paper = hint & (yy <= np.floor(np.polyval(line, xx)+.5)+1) & (a[..., 3] > 0)
    released = hint & ~paper & (a[..., 3] >= 240)
    if not paper.any() or not released.any(): fail('No supported source paper termination to apply')
    provenance = report.get('source_feature_provenance', {})
    raw = provenance.get('raw_Canny_contours', [])
    if not provenance.get('every_original_analysis_pixel_reproduced'):
        fail('Missing original source contour provenance')
    originals = {value['id']: value for value in raw}
    aperture = np.zeros(hint.shape, np.uint8)
    roles = []
    for feature in provenance.get('components', []):
        role = source_feature_role(feature, raw)
        role.update(analysis_contour=feature['component'], bbox=feature['bbox'])
        roles.append(role)
        for source_id in role['independent_closed_source_ids']:
            contour = np.asarray(originals[source_id]['points'], np.int32).reshape(-1, 1, 2)
            cv2.drawContours(aperture, [contour], -1, 1, cv2.FILLED)
    aperture = aperture > 0
    # The user's none strategy remains none. This is a material-evidence mask,
    # not a frozen RGB/protection zone and not an instruction to remove a hole.
    if np.any(released & aperture): fail('Released paper-tail proposal overlaps an independent closed source feature')
    xs = np.flatnonzero(hint.any(axis=0))
    gap = max(3, int((xs[-1]-xs[0]+1)*.04)); count = max(4, int((xs[-1]-xs[0]+1)*.09))
    xl = np.arange(max(0, int(xs[0])-gap-count), max(0, int(xs[0])-gap))
    xr = np.arange(min(w, int(xs[-1])+gap+1), min(w, int(xs[-1])+gap+count+1))
    if not len(xl) or not len(xr): fail('Missing same-source lateral frame anchors')
    y0 = max(0, report['cross_edge']['source_end_peak']-12)
    frame = ((a[..., 3] >= 240) & ~paper & ~aperture & (yy >= y0)
             & (xx >= xl.min()) & (xx <= xr.max()))
    labels, _ = ndimage.label(frame, np.ones((3, 3), bool))
    left_ids = set(int(v) for v in np.unique(labels[:, xl]) if v)
    right_ids = set(int(v) for v in np.unique(labels[:, xr]) if v)
    connected = np.isin(labels, list(left_ids & right_ids))
    if np.any(released & ~connected): fail('Released source tail does not join both known frame sides')
    folder.mkdir(parents=True, exist_ok=False)
    masks = {}
    for name, mask in [('paper-material', paper), ('released-frame', released),
                       ('independent-source-apertures', aperture), ('connected-source-frame', connected)]:
        path = folder/(name+'.png')
        Image.fromarray(mask.astype('uint8')*255).save(path)
        masks[name] = dict(path=str(path.resolve()), sha256=sha(path), pixels=int(mask.sum()))
    evidence = dict(material_boundary_resolution=RESOLUTION, material_domain_verified=True,
        source_sha256=source_hash, source_size=[w, h], source_geometry_coordinate_system='immutable_geometry_RGBA_pixel_centres',
        surface_prior=surface_prior, source_contour_report=str(report_path.resolve()), source_contour_report_sha256=sha(report_path),
        hint_mask=str(hint_path.resolve()), hint_sha256=sha(hint_path),
        paper_material_mask=masks['paper-material']['path'], paper_material_mask_sha256=masks['paper-material']['sha256'],
        source_end_line=line, inclusion_formula='hint & (y <= floor(line(x)+0.5)+1) & (alpha>0)',
        classification_masks=masks, feature_roles=roles, semantic_frame_connected=True,
        peer_exclusion_mask_used_as_material_gate=False,
        old_three_row_material_window=dict(status='not_satisfied' if legacy_error else 'not_evaluated_by_this_adapter', original_error=legacy_error),
        implementation_path=str(Path(__file__).resolve()), implementation_sha256=sha(__file__),
        supplementary_implementation_hashes=report.get('source_implementation_hashes', {}),
        evidence_generation='frozen_source_report_replay_not_general_runtime_detection',
        normal_coverage=None, lighting_effect_verified=False, visual_approved=False, publishable=False,
        pixel_protection=False, paper_lighting_budget_set=False,
        native_raw_topology_verified=False, integration_status='isolated_r4_adapter_requires_explicit_validator')
    if sha(source) != source_hash: fail('Source changed while preparing material evidence')
    (folder/'material-evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
    return paper, evidence


def resolve_current_source_material(source, hint, proof, folder, *,
                                    surface_prior='aluminum_bottom_frame', legacy_error=None):
    """General source-only entry point; no SKU, coordinates or old result flags.

Reuses the reviewed pure v3 algorithms on the CURRENT source and locator.
Only the source paper locator/proof and existing aluminum-bottom prior are
inputs. The resulting evidence files belong to this execution and source hash.
"""
    import importlib.util
    import numpy as np
    from PIL import Image
    from phone_texture.bottom_paper_audit import _source_cross_edge
    source, folder = Path(source), Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    source_start = sha(source)
    core_path = Path(__file__).with_name('source_contour_core.py')
    core_start = sha(core_path)
    dependency_hashes = {str(Path(__file__).with_name(name).resolve()): sha(Path(__file__).with_name(name))
        for name in ('auto_material.py', 'bottom_paper_audit.py', 'bottom_paper_closure.py')}
    a = np.array(Image.open(source).convert('RGBA'))
    hint = np.asarray(hint, dtype=bool)
    if hint.shape != a.shape[:2]: fail('Current locator has another source canvas')
    spec = importlib.util.spec_from_file_location('_source_contour_material_core', core_path)
    core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
    cross_edge_proposal = None
    cross_edge_implementation = None
    try:
        line, _, _, edge = _source_cross_edge(a, hint, proof)
        report, _ = core.topology(a, hint, proof, edge)
    except ValueError as cross_error:
        # Material-only candidate generation; the shared raw geometry detector
        # and its thresholds remain unchanged. Frame validation still follows.
        from . import material_cross_edge
        cross_edge_implementation = Path(material_cross_edge.__file__).resolve()
        cross_edge_hash = sha(cross_edge_implementation)
        try:
            edge, report, cross_edge_proposal = material_cross_edge.select_material_cross_edge(
                a, hint, proof, legacy_error=str(cross_error))
        except ValueError as proposal_error:
            rejected = dict(source_sha256=source_start,
                evidence=getattr(proposal_error, 'material_boundary_evidence', {}),
                error=str(proposal_error), material_certified=False, publishable=False)
            (folder/'rejected-material-cross-edge.json').write_text(json.dumps(rejected, ensure_ascii=False, indent=2)+'\n')
            raise
        if sha(cross_edge_implementation) != cross_edge_hash:
            fail('Material cross-edge implementation changed during proposal')
        line = edge['source_end_line']
    yy, xx = np.mgrid[:hint.shape[0], :hint.shape[1]]
    paper = hint & (yy <= np.floor(np.polyval(line, xx)+.5)+1) & (a[..., 3] > 0)
    feature_report, _ = core.feature_provenance(a, None, hint, paper, proof, report)
    if (sha(source) != source_start or sha(core_path) != core_start
            or any(sha(path) != digest for path, digest in dependency_hashes.items())):
        fail('Source or contour implementation changed during material detection')
    hint_path = folder/'source-label-location-hint.png'
    Image.fromarray(hint.astype('uint8')*255).save(hint_path)
    source_hash = sha(source)
    implementations = {str(core_path.resolve()): sha(core_path), **dependency_hashes}
    if cross_edge_proposal is not None:
        implementations[str(cross_edge_implementation)] = cross_edge_hash
        proposal_path = folder/'material-cross-edge-proposal.json'
        cross_edge_proposal.update(source_sha256=source_hash,
            hint_sha256=sha(hint_path), implementation_path=str(cross_edge_implementation),
            implementation_sha256=cross_edge_hash)
        proposal_path.write_text(json.dumps(cross_edge_proposal, ensure_ascii=False, indent=2)+'\n')
        report['material_cross_edge_proposal'] = dict(path=str(proposal_path.resolve()), sha256=sha(proposal_path))
    report.update(source_sha256=source_hash, source=str(source.resolve()), cross_edge=edge,
                  source_feature_provenance=feature_report, input_dependencies_unchanged=True,
                  source_implementation_hashes=implementations,
                  input_sha256_start={str(source.resolve()): source_hash, str(hint_path.resolve()): sha(hint_path)})
    report_path = folder/'current-source-contour-report.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    result, evidence = build_source_contour_material(source, hint_path, report_path, folder/'material',
        surface_prior=surface_prior, legacy_error=legacy_error)
    evidence['evidence_generation'] = 'current_source_only_no_SKU_or_external_result'
    (folder/'material/material-evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
    return result, evidence


def validate_source_contour_material(view, evidence, scope=None):
    """Verify actual source/mask/report bindings; no string-only allowlisting."""
    import numpy as np
    from PIL import Image
    if view.get('name') != 'bottom' or evidence.get('material_boundary_resolution') != RESOLUTION:
        fail('This source contour adapter applies only to the bottom face')
    if view.get('protection_policy') != 'none' or evidence.get('pixel_protection') is not False:
        fail('Source contour materials must not create pixel protection zones')
    source = Path(view['source'])
    if sha(source) != evidence.get('source_sha256') or view.get('source_sha256') != evidence.get('source_sha256'):
        fail('Source contour material source identity differs')
    if not evidence.get('material_domain_verified') or not evidence.get('semantic_frame_connected'):
        fail('Source contour material evidence is incomplete')
    bindings = [(evidence['source_contour_report'], evidence['source_contour_report_sha256']),
                (evidence['hint_mask'], evidence['hint_sha256']),
                (evidence['paper_material_mask'], evidence['paper_material_mask_sha256']),
                (evidence['implementation_path'], evidence['implementation_sha256'])]
    bindings += [(item['path'], item['sha256']) for item in evidence['classification_masks'].values()]
    bindings += list(evidence.get('supplementary_implementation_hashes', {}).items())
    if any(sha(path) != digest for path, digest in bindings): fail('Frozen source material evidence changed')
    report = read(evidence['source_contour_report']); _check_v3(report, evidence['source_sha256'])
    if evidence['source_end_line'] != report['cross_edge']['source_end_line']:
        fail('Material boundary differs from the actual source cross edge')
    a = np.array(Image.open(source).convert('RGBA'))
    hint = np.array(Image.open(evidence['hint_mask']).convert('L')) > 0
    if report.get('material_cross_edge_proposal'):
        from . import material_cross_edge
        proposal = read(report['material_cross_edge_proposal']['path'])
        material_cross_edge.validate_proposal_record(proposal, evidence['source_sha256'],
            report['cross_edge'], rgba=a, hint=hint)
    paper = np.array(Image.open(evidence['paper_material_mask']).convert('L')) > 0
    h, w = a.shape[:2]
    if [w, h] != evidence['source_size']: fail('Source material canvas differs')
    expected = hint & (np.arange(h)[:, None] <= np.floor(np.polyval(evidence['source_end_line'], np.arange(w))+.5)+1) & (a[..., 3] > 0)
    if not np.array_equal(paper, expected): fail('Paper material mask changed from the source boundary')
    actual = np.array(Image.open(view['source_label_fit_exclusion_mask']).convert('L')) > 0
    actual_hint = np.array(Image.open(view['source_label_location_hint_mask']).convert('L')) > 0
    if not np.array_equal(actual, paper) or not np.array_equal(actual_hint, hint): fail('View uses another source paper/hint domain')
    if view.get('source_label_material_evidence') != evidence: fail('View does not bind this material evidence')
    if scope is not None and (scope.get('source_paper_material_boundary') != evidence
                              or scope.get('source_sha256') != evidence['source_sha256']):
        fail('Scope and view use different source material evidence')
    return dict(material_boundary_resolution=RESOLUTION, material_domain_verified=True,
                source_sha256=evidence['source_sha256'], normal_coverage=None,
                lighting_effect_verified=False, visual_approved=False, publishable=False)


def require_material_boundary_adapter(job, scope=None, *, legacy_validator):
    """Explicit one-face r4 hook; all existing resolutions keep the old gate."""
    views = job.get('views', [])
    fresh = [v for v in views if v.get('source_label_material_evidence', {}).get('material_boundary_resolution') == RESOLUTION]
    if not fresh: return legacy_validator(job, scope)
    if len(views) != 1 or len(fresh) != 1: fail('Expected the pipeline single-face bottom material job')
    return validate_source_contour_material(fresh[0], fresh[0]['source_label_material_evidence'], scope)
