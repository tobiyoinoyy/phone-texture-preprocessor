"""Generate one review-only side candidate from this job's fresh observations.

No SKU identifiers, prior results or external experiment modules are consulted.
The ordinary successful lighting path remains the caller's first choice.
"""
import contextlib
import hashlib
import json
import math
from pathlib import Path
import time
import traceback


VERSION = 'side_candidate_lighting_v1'
OBSERVATION_SCHEMA = 'phone_texture_current_side_observation/v1'
FALSE_FLAGS = ('strict_endpoint_evidence_complete', 'material_classification_complete',
    'quality_checks_complete', 'structural_alignment_applied', 'visual_approved',
    'model_uv_verified', 'selected_for_delivery', 'strict_structure_verified',
    'structural_alignment_certified', 'six_faces_processed', 'seam_executed')


class SideCandidateLightingFailure(ValueError):
    def __init__(self, result, result_path):
        self.result = result
        self.result_path = str(result_path)
        self.side_candidate_lighting_evidence = dict(path=str(result_path),
            sha256=sha(result_path), hard_failures=result['hard_failures'])
        super().__init__('Side lighting candidate rejected: '+
            '; '.join(item['error'] for item in result['hard_failures']))


class SideCandidateIntegrityError(RuntimeError):
    """The caller must fail the case; this is not a local review fallback."""
    def __init__(self, message, result=None, result_path=None):
        self.result = result
        self.result_path = str(result_path) if result_path is not None else None
        super().__init__(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if hasattr(value, 'tolist'):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def verify(bindings):
    for path, digest in bindings.items():
        if not (isinstance(digest, str) and len(digest) == 64 and sha(path) == digest):
            raise SideCandidateIntegrityError('Side candidate bound input changed: '+path)


def merge_bindings(target, values):
    for path, digest in values.items():
        path = str(Path(path).resolve(strict=True))
        if path in target and target[path] != digest:
            raise SideCandidateIntegrityError('Conflicting side candidate input binding: '+path)
        target[path] = digest


def run(source, face, folder, *, endpoint_report, protection_policy='none',
        lighting_profile='accepted-shadow-lift-v2', raw_context=None):
    """Render once, check actual pixels, and return an unapproved candidate.

    endpoint_report is a newly written current-observation report. Its original
    raw geometry pixels determine known material samples; unresolved endpoint
    pixels receive only the shared-field assumption, never a material label.
    Any failed hard check writes its evidence and raises without selection.
    """
    require(face in ('left', 'right'), 'Side candidate face must be left or right')
    require(protection_policy == 'none', 'Side review requires explicit no-zone policy')
    require(lighting_profile == 'accepted-shadow-lift-v2', 'Unsupported side review profile')
    source = Path(source).resolve(strict=True)
    endpoint_report = Path(endpoint_report).resolve(strict=True)
    folder = Path(folder).resolve()
    require(not folder.exists(), 'Side candidate output folder already exists')
    require(not source.is_relative_to(folder) and not endpoint_report.is_relative_to(folder),
            'Side candidate output cannot contain its inputs')
    folder.mkdir(parents=True, exist_ok=False)
    result_path = folder/'result.json'
    started = time.perf_counter()
    inputs, implementation = {}, {}
    result = dict(version=VERSION, mode='current_source_shared_field_review', face=face,
        source=str(source), source_sha256=sha(source), result_path=str(result_path),
        current_observation=dict(path=str(endpoint_report), sha256=sha(endpoint_report)),
        state='preparing', render_attempts=0, render_completed=False,
        candidate_generated=False, candidate=None, candidate_sha256=None,
        source_geometry_preserved=False, eligible_for_visual_review=False,
        historical_candidate_reused=False, historical_observation_reused=False,
        segmentation_executed=False, retry_performed=False, profile_applications=0,
        input_sha256=inputs, implementation_sha256=implementation, checks={},
        hard_failures=[], artifacts=dict(result=str(result_path)),
        input_dependencies_unchanged=False, inputs_unchanged=False,
        implementation_unchanged=False, integrity_failure=False, **dict.fromkeys(FALSE_FLAGS, False))
    stage = 'preflight'

    def checkpoint():
        save(result_path, result)

    def failure(name, error):
        entry = dict(stage=name, type=type(error).__name__, error=str(error),
                     traceback=traceback.format_exc())
        for attribute in ('issues', 'reports', 'failures'):
            if hasattr(error, attribute):
                entry[attribute] = getattr(error, attribute)
        result['hard_failures'].append(entry)

    try:
        import cv2
        import numpy as np
        from PIL import Image
        from . import lighting, lighting_review, shadow_lift_profile as profile
        from .automatic import implementation_manifest, measure_masks, no_zone_effects
        from . import side_candidate_material as material
        from . import side_candidate_field as field
        from .side_candidate_diagnostics import measure as crosspairs
        require(cv2.getNumThreads() == 1, 'Side candidate requires the caller single-thread OpenCV limit')
        repo = Path(__file__).resolve().parents[1]
        implementation.update({str(repo/name): digest for name, digest in implementation_manifest(repo).items()})
        verify(implementation)
        observation = json.loads(endpoint_report.read_text())
        require(observation.get('schema') == OBSERVATION_SCHEMA and observation.get('face') == face,
                'A current same-face observation is required')
        require(observation.get('processing_state') == 'completed_diagnostic_only' and
                observation.get('observations_computed_from_current_source') is True and
                observation.get('historical_observations_reused') is False and
                observation.get('hard_failures') == [],
                'Current observation did not finish its measurements')
        require(observation.get('inputs_unchanged') is True and observation.get('code_unchanged') is True,
                'Current observation did not finish with intact bindings')
        require(observation.get('source_sha256') == result['source_sha256'], 'Current geometry identity differs')
        merge_bindings(inputs, observation['input_sha256'])
        merge_bindings(inputs, {str(source): result['source_sha256'], str(endpoint_report): sha(endpoint_report)})
        observed_code = observation['code_sha256']
        require(bool(observed_code), 'Current observation lacks implementation bindings')
        require(all(str(Path(path).resolve()) in implementation and
                    implementation[str(Path(path).resolve())] == digest for path, digest in observed_code.items()),
                'Observation was not produced by this installed implementation')
        verify(inputs)
        context = observation['source_context']
        if raw_context is not None:
            require(isinstance(raw_context, dict) and
                    raw_context.get('path') == context.get('path') and
                    raw_context.get('sha256') == context.get('sha256') and
                    raw_context.get('geometry_sha256') == context.get('geometry_sha256'),
                    'Supplied raw context differs from the current observation')
        result['source_context'] = context
        checkpoint()
        stage = 'current_partial_material_preparation'
        assets = folder/'material-inputs'
        job, analysis, _, _ = material.propose_job(source, face, assets,
            protection_policy=protection_policy, partial_side_evidence=endpoint_report)
        job, profile_proof = profile.apply(job, face, assets, lighting_profile)
        result['profile_applications'] = 1
        analysis['lighting_profile'] = profile_proof
        scope = material.lighting_scope(source, face, job, assets/'effect-source-witnesses')
        require(job['views'][0].get('required_processing_finalize') ==
                'partial_side_lighting.finalize_candidate_then_independent_checks' and
                scope.get('requires_partial_scope_finalization') is True,
                'Unexpected partial preparation contract')
        save(assets/'job.json', job)
        save(assets/'analysis.json', analysis)
        save(assets/'lighting-scope.json', scope)
        merge_bindings(inputs, {str(path): sha(path) for path in assets.rglob('*') if path.is_file()})
        verify(inputs)
        result['artifacts']['analysis'] = str(assets/'analysis.json')
        stage = 'shared_field_preparation'
        prepared = folder/'prepared'
        current_bindings = dict(implementation)
        merge_bindings(current_bindings, inputs)
        render_job, render_scope, preparation = field.prepare(assets/'job.json',
            assets/'lighting-scope.json', prepared, input_bindings=current_bindings)
        merge_bindings(inputs, preparation['prepared_assets_sha256'])
        merge_bindings(inputs, {str(prepared/'preparation.json'): sha(prepared/'preparation.json')})
        result['preparation'] = preparation
        result['artifacts'].update(preparation=str(prepared/'preparation.json'),
            job=str(prepared/'job.json'), scope=str(prepared/'scope.json'), lighting_log=str(folder/'lighting.log'))
        candidate = prepared/'render'/(face+'.png')
        stage = 'lighting_render'
        result.update(state=stage, render_attempts=1)
        checkpoint()
        with (folder/'lighting.log').open('x') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            lighting.run('aluminum', prepared/'job.json')
        require(candidate.is_file(), 'Side render produced no candidate')
        result.update(render_completed=True, candidate_generated=True,
            candidate=str(candidate), candidate_sha256=sha(candidate))
        result['artifacts']['candidate'] = str(candidate)
        stage = 'actual_pixel_checks'
        result['checks']['source_contract'] = field.verify_source_contract(prepared/'preparation.json', candidate)
        result['source_geometry_preserved'] = result['checks']['source_contract']['source_and_alpha_contract_passed']
        with Image.open(preparation['allowed_mask']) as image:
            allowed = np.asarray(image.convert('L')) > 0
        with Image.open(preparation['protected_mask']) as image:
            protected = np.asarray(image.convert('L')) > 0
        result['checks']['original_masks'] = measure_masks(source, candidate, allowed, protected)
        result['checks']['no_zone_effects'] = no_zone_effects(source, candidate, analysis)
        result['checks']['original_and_tangent_effects'] = profile.measure_effect(
            lighting_review.measure_local, render_scope, source, candidate, allowed, protected, lighting_profile)
        result['checks']['field_and_hardware_diagnostics'] = field.field_and_hardware_diagnostics(
            prepared/'preparation.json', candidate)
        result['checks']['pure_cross_hardware_diagnostics'] = crosspairs(render_job, job,
            source, candidate, lightness=lighting_review.lightness, require=require)
        require(cv2.getNumThreads() == 1, 'Side candidate changed OpenCV thread limit')
        result['opencv_threads'] = cv2.getNumThreads()
        result['state'] = 'illumination_assumption_review'
    except Exception as error:
        failure(stage, error)
        # Only a source-verified algorithm/check ValueError may preserve the
        # previous selected face. Bad input identity, I/O or code faults must
        # fail the whole case instead of looking like a harmless candidate miss.
        if stage == 'preflight' or not isinstance(error, ValueError):
            result['integrity_failure'] = True
    finally:
        for name, bindings in (('input', inputs), ('implementation', implementation)):
            try:
                verify(bindings)
                result['inputs_unchanged' if name == 'input' else 'implementation_unchanged'] = bool(bindings)
            except Exception as error:
                failure(name+'_integrity', error)
                result['integrity_failure'] = True
        result['input_dependencies_unchanged'] = result['inputs_unchanged'] and result['implementation_unchanged']
        if not result['input_dependencies_unchanged']:
            result['integrity_failure'] = True
        result['eligible_for_visual_review'] = (result['render_completed'] and
            result['source_geometry_preserved'] and result['input_dependencies_unchanged'] and
            not result['hard_failures'])
        if result['hard_failures']:
            result['state'] = 'side_lighting_candidate_rejected'
        result['seconds'] = time.perf_counter()-started
        result['artifact_sha256'] = {str(path): sha(path) for path in folder.rglob('*')
            if path.is_file() and path != result_path}
        result['portable_assets_sha256'] = {path: digest for path, digest in inputs.items()
            if path not in implementation}
        merge_bindings(result['portable_assets_sha256'], result['artifact_sha256'])
        result = clean(result)
        checkpoint()
    if not result['eligible_for_visual_review']:
        if result['integrity_failure']:
            raise SideCandidateIntegrityError('Side candidate input, implementation or I/O integrity failed',
                                             result, result_path)
        raise SideCandidateLightingFailure(result, result_path)
    return result
