"""Generate source-bound antenna-position review candidates in the current job.

No historical sample IDs, templates, observations or output paths are inputs.
Observation and rendering are separate so the lighting route can use the same
current source evidence. A review candidate never certifies strict endpoints,
material classification, physical reference, visual approval or UV seams.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import traceback

from .side_candidate_structure_math import ReviewRejected, require

OBSERVATION_SCHEMA = 'phone_texture_current_side_observation/v1'
VERSION = 'side_candidate_structure_v1'
FACES = ('front', 'back', 'left', 'right', 'top', 'bottom')
MAX_HORIZONTAL_DISPLACEMENT = 1.0


class SideCandidateIntegrityError(RuntimeError):
    def __init__(self, message, result=None, path=None):
        super().__init__(message)
        self.result = result
        self.path = str(path) if path is not None else None
        self.result_path = self.path


def require_integrity(value, message):
    if not value:
        raise SideCandidateIntegrityError(message)


class SideCandidateStructureFailure(ReviewRejected):
    def __init__(self, message, result, path):
        super().__init__(message, dict(result_path=str(path)))
        self.result = result
        self.path = str(path)
        self.result_path = str(path)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, 'tolist'):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def canonical(value):
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def verify_bindings(bindings):
    issues = []
    for path, expected in bindings.items():
        try:
            actual = sha(path)
            if actual != expected:
                issues.append(dict(path=path, expected=expected, actual=actual))
        except OSError as error:
            issues.append(dict(path=path, error=str(error)))
    if issues:
        error = SideCandidateIntegrityError('Current source or implementation binding changed')
        error.evidence = dict(issues=issues)
        raise error
    return dict(files=len(bindings), unchanged=True)


def bind(bindings, path, expected=None):
    path = Path(path).resolve(strict=True)
    actual = sha(path)
    require_integrity(expected is None or actual == expected, 'Current file binding differs: ' + str(path))
    require_integrity(str(path) not in bindings or bindings[str(path)] == actual,
                      'Conflicting source binding: ' + str(path))
    bindings[str(path)] = actual
    return path


def code_bindings():
    folder = Path(__file__).resolve().parent
    names = ('side_candidate_structure.py', 'side_candidate_structure_math.py',
             'side_candidate_structure_observation.py', 'auto_structure.py', 'antenna_material.py',
             'source_antenna_evidence.py', 'side_material.py', 'structure_review.py',
             'source_body_reference.py', 'lighting_review.py')
    return {str(folder / name): sha(folder / name) for name in names}


def flags():
    return dict(strict_endpoint_evidence_complete=False, strict_structure_verified=False,
        structural_alignment_certified=False, material_classification_complete=False,
        quality_checks_complete=False, visual_approved=False, model_uv_verified=False,
        selected_for_delivery=False, six_faces_processed=False, seam_executed=False,
        reference_certified=False, reference_review_required=True)


def read_record(value, path_key, bindings):
    if isinstance(value, (str, Path)):
        path = bind(bindings, value)
        try:
            result = json.loads(path.read_text())
        except (ValueError, OSError) as error:
            raise SideCandidateIntegrityError('Current report unreadable: ' + str(path)) from error
    else:
        require(isinstance(value, dict), 'A saved current-job report is required')
        path = value.get(path_key) or value.get('artifacts', {}).get('result')
        require(path is not None, 'Current report lacks its saved path: ' + path_key)
        path = bind(bindings, path)
        try:
            result = json.loads(path.read_text())
        except (ValueError, OSError) as error:
            raise SideCandidateIntegrityError('Current report unreadable: ' + str(path)) from error
        require_integrity(isinstance(result, dict), 'Current report must be a JSON object')
        # Convenience path/SHA fields are added after the report is saved and
        # cannot include their own digest without a recursive self-reference.
        for key, item in result.items():
            require_integrity(key in value and value[key] == item, 'Passed current report differs from saved bytes: ' + key)
    require_integrity(isinstance(result, dict), 'Current report must be a JSON object')
    return result, path


def _rgba(path):
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        require(image.mode == 'RGBA' and image.format == 'PNG', 'Unchanged RGBA PNG required')
        return np.asarray(image).copy()


def _finish(folder, filename, result, inputs, codes, error=None):
    try:
        result['inputs_unchanged'] = verify_bindings(inputs)['unchanged']
        result['code_unchanged'] = verify_bindings(codes)['unchanged']
        result['implementation_unchanged'] = result['code_unchanged']
    except Exception as integrity_error:
        error = error or integrity_error
        result['hard_failures'].append(dict(stage='final_integrity', error=str(integrity_error),
            evidence=getattr(integrity_error, 'evidence', {})))
        result['inputs_unchanged'] = False
        result['code_unchanged'] = False
        result['implementation_unchanged'] = False
    if result['hard_failures']:
        result['eligible_for_visual_review'] = False
        result['structural_alignment_applied'] = False
        result['selected_for_review_export'] = False
    result['artifact_sha256'] = {str(path.resolve()): sha(path) for path in folder.rglob('*')
                               if path.is_file() and path.name != filename and not path.name.endswith('.tmp')}
    path = folder / filename
    result['artifacts']['result'] = str(path)
    result['result_path'] = str(path)
    if filename == 'observation.json':
        result['observation_path'] = str(path)
    save(path, result)
    result = clean(result)
    if filename == 'observation.json':
        result['observation_sha256'] = sha(path)
    if (not result['inputs_unchanged'] or not result['code_unchanged'] or
            isinstance(error, (json.JSONDecodeError, UnicodeError)) or
            (error is not None and not isinstance(error, ValueError))):
        raise SideCandidateIntegrityError(str(error or 'Current input integrity failed'), result, path) from error
    if error is not None or result['hard_failures']:
        raise SideCandidateStructureFailure(str(error or result['hard_failures'][0]['error']), result, path)
    return result


def observe_current(source, face, folder, *, raw_context, geometry_files,
                    geometry_report, body_reference, sku=None):
    """Observe current raw pairs/back/controls once, without rendering a candidate.

The raw context must be generated from this job's source snapshot. Existing
strict material errors are recorded; predicted or missing columns remain
unconfirmed. A failure never creates an eligible lighting/structure input.
"""
    require(face in ('left', 'right'), 'Only independent left/right side faces are supported')
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    inputs, codes = {}, code_bindings()
    result = dict(**flags(), schema=OBSERVATION_SCHEMA, face=face, sku=sku,
        key=sku or 'current-source', input_sha256=inputs, code_sha256=codes,
        hard_failures=[], artifacts={}, candidate_generated=False,
        eligible_for_visual_review=False, structural_alignment_applied=False,
        processing_state='preflight', source=str(Path(source).resolve()))
    stage = 'source_identity'
    error = None
    try:
        import numpy as np
        from PIL import Image
        from scipy.ndimage import gaussian_filter1d
        from . import auto_structure as auto, structure_review as gate
        from . import source_antenna_evidence as original
        from .side_material import hardware_materials
        from .side_candidate_structure_observation import trace_band, reference_contract

        require_integrity(set(geometry_files) == set(FACES), 'Current six-face geometry set required')
        if isinstance(geometry_report, (str, Path)):
            report_path = bind(inputs, geometry_report)
            report = json.loads(report_path.read_text())
        else:
            require(isinstance(geometry_report, dict), 'Current geometry report required')
            report = copy.deepcopy(geometry_report)
            report_path = folder / 'geometry-report.json'
            save(report_path, report)
            bind(inputs, report_path)
        require_integrity(report.get('candidates_complete') is True, 'Geometry report is incomplete')
        rows = report.get('images', [])
        require_integrity(len(rows) == 6 and {row.get('face') for row in rows} == set(FACES), 'Ambiguous geometry face rows')
        by_face = {row['face']: row for row in rows}
        identities = {row.get('sku') for row in rows}
        require_integrity(len(identities) == 1 and None not in identities, 'Geometry contains different SKU identities')
        actual_sku = next(iter(identities))
        require_integrity(sku is None or actual_sku == sku, 'Requested SKU differs from current geometry')
        result.update(sku=actual_sku, key=actual_sku)
        files = {f: bind(inputs, geometry_files[f], by_face[f]['output_sha256']) for f in FACES}
        source = bind(inputs, source, by_face[face]['output_sha256'])
        require_integrity(source == files[face], 'Side source is not the selected geometry face')
        require(isinstance(raw_context, dict), 'Current original RGB context required')
        # Photometric recovery is a separate hypothesis. Observe the original
        # saved RGB reference even if a prior strict attempt added that field.
        context = {k: copy.deepcopy(v) for k, v in raw_context.items()
                   if k not in ('photometric_path', 'photometric_sha256', 'photometric_proof', 'normalization_rows')}
        require_integrity(context.get('method') == 'original_rgb_before_alpha_premultiplication_v1',
                          'Unexpected original RGB context method')
        require_integrity(context.get('face') == face and context.get('geometry') == by_face[face]['geometry'],
                'Raw context belongs to different side geometry')
        require_integrity(context.get('raw_source_sha256') == by_face[face]['source_sha256'],
                'Raw context belongs to a different capture')
        bind(inputs, context['path'], context['sha256'])
        bind(inputs, context['raw_source'], context['raw_source_sha256'])
        raw = original.load(context, source, face)
        geometry_pixels = _rgba(source)
        raw_pixels = np.asarray(raw.convert('RGBA'))
        require_integrity(raw_pixels.shape == geometry_pixels.shape and
                np.array_equal(raw_pixels[..., 3], geometry_pixels[..., 3]), 'Original RGB changed geometry or alpha')
        require(isinstance(body_reference, dict) and body_reference.get('enabled') is True,
                'Explicit current back-reference audit is required')
        reference = copy.deepcopy(body_reference)
        for path, digest in reference.get('bound_assets_sha256', {}).items():
            bind(inputs, path, digest)
        require(reference.get('bound_assets_sha256'), 'Current reference lacks source bindings')
        back_source = reference.get('applicability', {}).get('source', {})
        # A perspective-rectified back keeps its frozen pre-rectification row;
        # the body reference was screened on that capture, not the new pixels.
        reference_row = by_face['back'].get('pre_perspective_row', by_face['back'])
        require_integrity(back_source.get('sha256') == reference_row['output_sha256'],
                'Reference belongs to another back geometry')
        bind(inputs, back_source['path'], back_source['sha256'])
        # Optional bottom alignment makes byte-identical companion copies.
        # Their paths differ while the immutable back row and bytes stay bound.
        require_integrity(reference.get('geometry_report_row_sha256') == canonical(reference_row),
                'Reference back geometry row differs')
        reference_review = reference_contract(reference, reference_row['geometry'])
        result.update(source=str(source), source_sha256=sha(source), source_context=context,
            geometry_report_path=str(report_path), geometry_files={f: str(p) for f, p in files.items()},
            geometry_sha256={f: by_face[f]['output_sha256'] for f in FACES},
            reference_review=reference_review, original_reference_status=reference_review['original_status'],
            body_reference=reference)
        stage = 'current_raw_and_back_measurements'
        back_pixels = _rgba(files['back'])
        back = Image.fromarray(back_pixels)
        geo = Image.fromarray(geometry_pixels)
        h, w = geometry_pixels.shape[:2]
        bh, bw = back_pixels.shape[:2]
        require([bw, bh] == by_face['back']['geometry']['output_size'], 'Own-back dimensions differ')
        opaque = back_pixels[int(bh * .25):int(bh * .75), :, 3] >= 128
        require(opaque.shape[0] > 0 and opaque.any(axis=1).all(), 'Own-back lateral edge lacks support')
        edge = (int(np.median(bw - 1 - np.argmax(opaque[:, ::-1], axis=1))) if face == 'left'
                else int(np.median(np.argmax(opaque, axis=1))))
        rear_rows, rear_signs, rear_columns, rear_proof = auto._rear_reference(back, face, edge, None, 3, per_band=True)
        require(rear_proof['agreeing_columns'] >= 3, 'Own-back rows lack independent neighbouring support')
        expected_width = float(np.mean([rear_rows[1] - rear_rows[0], rear_rows[3] - rear_rows[2]]) * h / bh)
        xr = [max(1, int(w * .4)), min(w - 1, int(w * .6))]
        seeds, _, central = auto.antenna_rows(geo, xr, expected_width)
        raw_rows, _, raw_central = auto.antenna_rows(raw, xr, expected_width)
        agreements = []
        for band in (0, 1):
            candidates = central['supported_central_candidates'][band]
            distance = min(float(np.max(np.abs(np.asarray(raw_rows[2*band:2*band+2]) - pair))) for pair in candidates)
            require(distance <= 2.5, 'Raw central material differs from current geometry candidates')
            agreements.append(dict(band=band, minimum_distance_px=distance, required_maximum_px=2.5,
                                   does_not_certify_endpoint_support=True))
        gradient = np.gradient(gaussian_filter1d(auto.gray_signal(raw_pixels, xr), 1.))
        signs = [int(np.sign(gradient[int(y)])) for y in seeds]
        require(all(v in (-1, 1) for v in signs) and signs[0] == -signs[1] and signs[2] == -signs[3],
                'Source central material polarity is ambiguous')
        back_measured = gate.detect(files['back'], rear_columns, auto.windows(rear_rows, bh), rear_signs).tolist()
        require(max(abs(a - b) for a, b in zip(back_measured, rear_rows)) <= gate.TOLERANCE,
                'Own-back seeds disagree with current source pixels')
        target = [(y + .5) / bh * h - .5 for y in rear_rows]
        result.update(central_rows=seeds, central_signs=signs, central_candidates=central,
            raw_RGB_own_central_rows=raw_rows, raw_RGB_central_candidates=raw_central,
            raw_seed_to_geometry_candidate_agreement=agreements, expected_width=expected_width,
            rear_reference=dict(rows=rear_rows, signs=rear_signs, columns=rear_columns, proof=rear_proof,
                actual_rows=back_measured, expected_side_width=expected_width, canvas_center_targets=target,
                physical_side=face, edge_in_back_image='right' if face == 'left' else 'left',
                view_orientation='unmirrored_rear'), bands={})
        for band in (0, 1):
            measured, observations = trace_band(raw, seeds, band)
            require(observations is not None and measured['observer_reached'],
                    'Current raw paired observations unavailable', dict(band=band, observation=clean(measured)))
            supported = [item for item in measured['columns'] if item['paired_source_supported'] and
                         xr[0] <= item['column'] < xr[1]]
            require(len(supported) >= max(3, math.ceil((xr[1] - xr[0]) * .6)),
                    'Current central paired coverage is insufficient')
            result['bands'][str(band)] = dict(original_RGB_geometry_seeds=measured,
                assessment=dict(current_original_RGB_gate_passed=measured['actual_gate_passed'],
                    full_endpoint_recovery_certified=False, observed_pairs_are_source_bound=True,
                    missing_or_predicted_pairs_are_not_material_evidence=True))
        stage = 'current_hardware_reservation'
        ranges, boxes = auto.hardware_ranges(geo)
        middle = [min(v[0] for v in ranges) - 3, max(v[1] for v in ranges) + 3]
        components = [proof for _, proof in hardware_materials(geo, face)]
        for component in components:
            require(middle[0] <= component['box'][1] < component['box'][3] <= middle[1],
                    'Independent hardware component escapes the fixed middle')
        result['hardware'] = dict(coarse_ranges=ranges, coarse_boxes=boxes, middle_reserved_range=middle,
            independent_material_components=components, contours_not_removed_or_reclassified=True,
            no_remap_or_hardware_edit_performed=True)
        result.update(processing_state='completed_diagnostic_only', source_geometry_preserved=True,
            complete_source_endpoint_proof=all(v['original_RGB_geometry_seeds']['actual_gate_passed']
                                               for v in result['bands'].values()),
            observations_computed_from_current_source=True, historical_observations_reused=False)
    except Exception as caught:
        error = caught
        result.update(processing_state=stage + '_rejected')
        result['hard_failures'].append(dict(stage=stage, error=str(caught), type=type(caught).__name__,
            evidence=getattr(caught, 'evidence', {}), traceback=traceback.format_exc()))
    return _finish(folder, 'observation.json', result, inputs, codes, error)


def generate(lighting_result, observation, output_dir):
    """Apply one current-source plan to current relit RGB, retaining alpha.

Eligible ordinary lighting can supply a current_stage_lighting_v1 wrapper
with its real mask/effect checks; this does not trigger another lighting render.
"""
    folder = Path(output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    inputs, codes = {}, code_bindings()
    result = dict(**flags(), version=VERSION, input_sha256=inputs, code_sha256=codes,
        hard_failures=[], checks={}, artifacts={}, candidate_generated=False,
        eligible_for_visual_review=False, structural_alignment_applied=False,
        selected_for_review_export=False, remap_executed=False, lighting_reprocessed=False,
        execution_state='preflight', state='contour_relative_sampling_assumption_review')
    stage = 'current_report_binding'
    error = None
    try:
        import numpy as np
        from PIL import Image
        from . import auto_structure as auto, structure_review as gate
        from .side_candidate_structure_math import build_mapping, remap_array, shoulder_diagnostics, edge_uncertainty
        observed, observed_path = read_record(observation, 'observation_path', inputs)
        lighting, lighting_path = read_record(lighting_result, 'result_path', inputs)
        require(observed.get('schema') == OBSERVATION_SCHEMA and
                observed.get('processing_state') == 'completed_diagnostic_only' and
                observed.get('observations_computed_from_current_source') is True and
                observed.get('historical_observations_reused') is False and not observed.get('hard_failures'),
                'Current original-RGB observation did not finish')
        require(observed.get('inputs_unchanged') is True and observed.get('code_unchanged') is True,
                'Current observation lacks intact provenance')
        require(lighting.get('version') in ('side_candidate_lighting_v1', 'current_stage_lighting_v1'),
                'Unsupported current lighting evidence schema')
        require(lighting.get('eligible_for_visual_review') is True and lighting.get('hard_failures') == [] and
                lighting.get('source_geometry_preserved') is True and lighting.get('inputs_unchanged') is True and
                lighting.get('implementation_unchanged') is True, 'Current lighting candidate failed its hard checks')
        require(isinstance(lighting.get('checks'), dict) and lighting['checks'], 'Actual lighting checks are missing')
        if lighting['version'] == 'current_stage_lighting_v1':
            require(lighting.get('normal_stage_checks_complete') is True, 'Ordinary lighting checks are incomplete')
        else:
            required_checks = {'source_contract', 'original_masks', 'original_and_tangent_effects', 'field_and_hardware_diagnostics'}
            require(required_checks <= set(lighting['checks']), 'Shared lighting checks are incomplete')
            origin = lighting.get('current_observation', {})
            require_integrity(origin.get('sha256') == sha(observed_path) and Path(origin['path']).resolve() == observed_path,
                    'Lighting used another source observation')
        face = observed['face']
        require_integrity(face in ('left', 'right') and lighting.get('face') == face, 'Lighting/observation faces differ')
        for record in (observed, lighting):
            for field in ('input_sha256', 'code_sha256', 'implementation_sha256'):
                for path, digest in record.get(field, {}).items():
                    bind(inputs, path, digest)
        geometry = bind(inputs, observed['source'], observed['source_sha256'])
        require_integrity(lighting.get('source_sha256') == observed['source_sha256'] and
                Path(lighting['source']).resolve() == geometry, 'Lighting belongs to another geometry source')
        source = bind(inputs, lighting['candidate'], lighting['candidate_sha256'])
        pixels, geo = _rgba(source), _rgba(geometry)
        require_integrity(pixels.shape == geo.shape and np.array_equal(pixels[..., 3], geo[..., 3]),
                'Lighting changed immutable geometry/alpha')
        result.update(face=face, sku=observed['sku'], source=str(source), source_sha256=sha(source),
            geometry_source=str(geometry), geometry_sha256=sha(geometry),
            current_observation=dict(path=str(observed_path), sha256=sha(observed_path)),
            current_lighting=dict(path=str(lighting_path), sha256=sha(lighting_path)),
            reference_review=copy.deepcopy(observed['reference_review']),
            original_reference_status=observed['original_reference_status'],
            source_lighting_assumption_retained=lighting['version'] == 'side_candidate_lighting_v1',
            source_geometry_preserved=True)
        stage = 'source_plan'
        h, w = pixels.shape[:2]
        seeds, signs = observed['central_rows'], observed['central_signs']
        columns = observed['central_candidates']['signal_columns']
        before = gate.detect(source, columns, auto.windows(seeds, h), signs).tolist()
        require(max(abs(a - b) for a, b in zip(before, seeds)) <= gate.TOLERANCE,
                'Current lighting no longer supports source central rows')
        target = observed['rear_reference']['canvas_center_targets']
        measured, witnesses = [], []
        for band in (0, 1):
            evidence = observed['bands'][str(band)]['original_RGB_geometry_seeds']
            supported = [item for item in evidence['columns'] if item['paired_source_supported']]
            central = [item for item in supported if columns[0] <= item['column'] < columns[1]]
            require(len(central) >= max(3, math.ceil((columns[1] - columns[0]) * .6)),
                    'Current central pair coverage differs')
            pair = np.median(np.asarray([item['measured_pair'] for item in central]), axis=0).tolist()
            bounds = [min(item['measured_pair'][0] for item in supported),
                      max(item['measured_pair'][1] for item in supported)]
            measured.append(dict(source_pair=pair, observed_bounds=bounds, target_pair=target[band*2:band*2+2]))
            witnesses.append(dict(band=band, fixed_supported_columns=[item['column'] for item in supported],
                unknown_columns=sorted(set(evidence['opaque_missing_columns']) | set(evidence['terminal_missing_columns'])),
                strict_source_error=evidence.get('error'), actual_strict_gate_passed=evidence['actual_gate_passed'],
                strict_endpoint_evidence_complete=False))
        plan = build_mapping(h, w, measured, observed['hardware']['middle_reserved_range'])
        prep = dict(face=face, sku=observed['sku'], source=str(source), source_sha256=sha(source),
            geometry_source=str(geometry), geometry_sha256=sha(geometry), mapping=plan,
            source_central_rows=before, source_central_columns=columns, source_signs=signs,
            source_witnesses=witnesses, reference_review=observed['reference_review'],
            current_observation=result['current_observation'], current_lighting=result['current_lighting'],
            horizontal_displacement_limit_px=MAX_HORIZONTAL_DISPLACEMENT,
            horizontal_limit_basis='conservative maximum observed in prior candidate validation; no retry or relaxation',
            strict_structure_verified=False, visual_approved=False, model_uv_verified=False)
        prep_path = folder / 'preparation.json'
        save(prep_path, prep)
        bind(inputs, prep_path)
        result['artifacts']['preparation'] = str(prep_path)
        result['prepared_longitudinal_mapping'] = plan
        stage = 'actual_donor_remap'
        candidate, proof, transport = remap_array(pixels, plan, return_transport=True)
        result['checks']['contour_sampling'] = proof
        require(max(abs(proof['dx_min']), abs(proof['dx_max'])) <= MAX_HORIZONTAL_DISPLACEMENT + 1e-9,
                'Current contour would exceed the bounded horizontal review scope',
                dict(dx_min=proof['dx_min'], dx_max=proof['dx_max'], limit=MAX_HORIZONTAL_DISPLACEMENT))
        output = folder / 'candidate.png'
        Image.fromarray(candidate).save(output)
        coordinates = folder / 'sampling-coordinates.npz'
        np.savez_compressed(coordinates, **transport)
        uncertainty = folder / 'edge-uncertainty-not-material.png'
        Image.fromarray(edge_uncertainty(pixels[..., 3]).astype(np.uint8) * 255).save(uncertainty)
        result.update(remap_executed=True, candidate_generated=True,
            candidate=str(output), candidate_sha256=sha(output),
            original_opacity_contract_passed=proof['original_opacity_contract_passed'],
            opacity_review_required=bool(proof['opacity_review_events']))
        result['artifacts'].update(candidate=str(output), sampling_coordinates=str(coordinates), edge_uncertainty=str(uncertainty))
        stage = 'final_saved_pixel_measurements'
        saved = _rgba(output)
        require(np.array_equal(saved, candidate), 'Saved PNG differs from the audited construction')
        expected = [y + plan['bands'][i // 2]['delta'] for i, y in enumerate(before)]
        final_rows = gate.detect(output, columns, auto.windows(expected, h, 6), signs).tolist()
        require(max(abs(a - b) for a, b in zip(final_rows, expected)) <= gate.TOLERANCE,
                'Final central edges differ from source translation')
        final = []
        for band, item in enumerate(plan['bands']):
            pair = final_rows[band*2:band*2+2]
            source_width = before[band*2+1] - before[band*2]
            width = pair[1] - pair[0]
            center_error = sum(pair) / 2 - sum(item['target_pair']) / 2
            require(abs(center_error) <= gate.TOLERANCE and abs(width - source_width) <= gate.TOLERANCE,
                    'Final source width or own-back center exceeds the existing tolerance')
            final.append(dict(band=band, measured_rows=pair, center_error_px=center_error,
                source_width=source_width, final_width=width, tolerance_px=gate.TOLERANCE))
        result['checks']['final_central_measurements'] = final
        result['checks']['shoulder_diagnostics'] = shoulder_diagnostics(pixels, saved, plan)
        result.update(execution_state='candidate_generated_needs_contour_and_visual_review',
            eligible_for_visual_review=True, structural_alignment_applied=True, selected_for_review_export=True,
            canvas_dimensions_and_alpha_preserved=True, structural_pixels_repositioned=True,
            structure_operation='band_center_contour_relative_sampling_review',
            remaining_checks=['natural endpoint/shoulder appearance and horizontal detail compression',
                              'source material and physical back-reference certification', 'actual user and UV/seam review'])
    except Exception as caught:
        error = caught
        result.update(execution_state=stage + '_rejected')
        result['hard_failures'].append(dict(stage=stage, error=str(caught), type=type(caught).__name__,
            evidence=getattr(caught, 'evidence', {}), traceback=traceback.format_exc()))
    return _finish(folder, 'result.json', result, inputs, codes, error)
