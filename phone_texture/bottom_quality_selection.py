"""Record existing checks for a bottom-only comparison; no new visual limits."""
from pathlib import Path
import json
import math
import re
from .geometry_review import digest, write


def local_rejection_code(error, phase):
    """Small fail-closed list of measured local rejections, never broad phases.

    Provenance/shape/configuration ValueErrors can occur in every phase.
    Unknown messages (including wrapped material identity errors) propagate.
    """
    if type(error) is not ValueError:
        return None
    text = str(error)
    if phase == 'material':
        unresolved = {
            'Source paper contour is not closed by the reviewed predicate',
            'Insufficient source transverse edge support',
            'Source contour path lacks opaque source evidence',
            'No supported source paper termination to apply',
            'Released paper-tail proposal overlaps an independent closed source feature',
            'Missing same-source lateral frame anchors',
            'Released source tail does not join both known frame sides',
            'unresolved_no_source_cross_edge', 'missing source boundary evidence',
        }
        prefix = 'Bottom paper material boundary unresolved: '
        if text.startswith(prefix) and text[len(prefix):] in unresolved:
            return 'known_source_material_support_rejection'
        if text in {'Too few opaque material pixels', 'No complete opaque source reference patch for material domain',
                    'No reliable material domains',
                    'This QC studio profile requires a resolved printed paper label on back/bottom'}:
            return 'known_material_fit_support_rejection'
        if re.fullmatch(r'Material region has no reliable fit samples: (printed_label|main_surface)', text):
            return 'known_material_fit_support_rejection'
    elif phase == 'scope':
        if text in {'Effect witness has no opaque source material', 'Too few independent source effect pixels'}:
            return 'known_source_effect_support_rejection'
    elif phase == 'render':
        if text in {'Insufficient reviewed material samples', 'Insufficient body-contour material samples',
                    'Too few independently sampled rim depths'}:
            return 'known_render_fit_support_rejection'
        bounds = r'\[[0-9]+, [0-9]+, [0-9]+, [0-9]+\]'
        if re.fullmatch(r'Unsupported editable illumination: [1-9][0-9]* pixels; bounds '+bounds+
                r'\. Split the component and fit its own reviewed source samples; '
                r'nearest-neighbor or median hole filling is not permitted\.', text):
            return 'known_render_field_support_rejection'
        if re.fullmatch(r'Unsupported editable contour illumination: [1-9][0-9]* pixels; bounds '+bounds, text):
            return 'known_render_field_support_rejection'
    elif phase == 'effect':
        if text in {'Too few source-normal continuity pairs', 'Too few internal boundary pairs',
                    'Too few same-material continuity pairs', 'Too few material continuity samples per line'}:
            return 'known_effect_sampling_rejection'
        if re.fullmatch(r'Insufficient same-row normal metal peers at [A-Za-z0-9_]+: row [0-9]+', text):
            return 'known_effect_sampling_rejection'
        number = r'[0-9]+(?:\.[0-9]+)?'
        if re.fullmatch(r'Lighting (?:normal peer|boundary|pixel correction|correction) discontinuity at '
                       r'[A-Za-z0-9_]+: '+number+r' L(?: \(peak '+number+r' L\))?', text):
            return 'known_measured_effect_discontinuity'
    return None


def effect_fully_passed(scope, effect):
    """Require explicit evidence for every scoped check, never absence of error.

    Successful measure_local historically returns named reports without a
    top-level success flag. This derives a comparison certificate from those
    actual reports while leaving the original effect object unchanged.
    """
    def within(value, limit):
        return (type(value) in (int, float) and type(limit) in (int, float)
                and math.isfinite(value) and math.isfinite(limit) and 0 <= value <= limit)

    def sampled(report):
        return type(report.get('pair_count')) is int and report['pair_count'] >= 20

    zones = scope.get('local_checks')
    if (not isinstance(zones, dict) or not zones or not isinstance(effect, dict)
            or effect.get('original_effect_check_passed') is False
            or effect.get('passed') is False
            or any(effect.get(k) for k in ('warnings', 'effect_failures'))):
        return False
    for name, zone in zones.items():
        report = effect.get(name)
        if (not isinstance(report, dict) or report.get('effect_check_passed') is not True
                or report.get('quantization_review')):
            return False
        if 'normal_peer_mask' in zone:
            coverage = report.get('normal_coverage')
            if coverage is not None:
                if (not isinstance(coverage, dict) or coverage.get('complete') is not True
                        or coverage.get('original_threshold_passed') is not True
                        or report.get('normal_check_passed') is not True
                        or coverage.get('normal_pairs', 0) < 20
                        or coverage.get('qualified_target_pairs', 0) < 20
                        or coverage.get('limit_L') != zone['max_normal_peer_step_L']
                        or not within(coverage.get('max_observed_residual_L75'), zone['max_normal_peer_step_L'])):
                    return False
            elif not (sampled(report) and report.get('limit_L') == zone['max_normal_peer_step_L']
                      and within(report.get('max_normal_peer_step_L'), zone['max_normal_peer_step_L'])):
                return False
        for requirement, measured in (
                ('max_added_step_L', 'max_added_step_L'),
                ('max_pixel_added_step_L', 'max_pixel_added_step_L')):
            if requirement in zone and not (sampled(report) and within(report.get(measured), zone[requirement])):
                return False
        if 'boundary_peer_mask' in zone:
            limit = zone['max_boundary_step_L']
            if not (sampled(report) and within(report.get('boundary_step_L99'), limit)
                    and within(report.get('boundary_peak_L'), limit*1.5)):
                return False
    return True


def record_attempt(source, candidate, assets, checked, scope):
    """Persist complete checks alongside each untouched candidate tree."""
    assets = Path(assets)
    source_sha256, candidate_sha256 = digest(source), digest(candidate)
    if (checked.get('source_sha256') != source_sha256
            or checked.get('output_sha256') != candidate_sha256):
        raise RuntimeError('Bottom pixel checks do not bind the current source/candidate')
    path = assets/'selection-checks.json'
    write(path, checked)
    effect = checked['source_relative_effect_checks']
    return dict(status='measured', source=str(source), source_sha256=source_sha256,
        candidate=str(candidate), candidate_sha256=candidate_sha256, assets=str(assets),
        job=str(assets/'job.json'), job_sha256=digest(assets/'job.json'),
        scope=str(assets/'lighting-scope.json'), scope_sha256=digest(assets/'lighting-scope.json'),
        checks=str(path), checks_sha256=digest(path),
        input_sha256=bind_inputs(json.loads((assets/'job.json').read_text()), scope),
        original_effect_check_passed=effect_fully_passed(scope, effect),
        original_effect_top_level_flag=effect.get('original_effect_check_passed'),
        certificate_basis='all_actual_scoped_effect_reports_complete_and_passed',
        required_checks=sorted(scope.get('local_checks', {})),
        original_effect_error=effect.get('error'), visual_approved=False)


def verify_attempt(record):
    verify_bindings(record['input_sha256'])
    for field in ('source', 'candidate', 'job', 'scope', 'checks'):
        if digest(record[field]) != record[field+'_sha256']:
            raise RuntimeError('Bottom comparison evidence changed: '+field)
    scope = json.loads(Path(record['scope']).read_text())
    checks = json.loads(Path(record['checks']).read_text())
    if (checks.get('source_sha256') != record['source_sha256']
            or checks.get('output_sha256') != record['candidate_sha256']):
        raise RuntimeError('Bottom pixel checks differ from the bound source/candidate')
    if effect_fully_passed(scope, checks['source_relative_effect_checks']) is not record['original_effect_check_passed']:
        raise RuntimeError('Bottom comparison success certificate differs from actual checks')


def bind_inputs(*values):
    """Freeze existing source/mask/job dependencies before the trial runs."""
    result = {}
    def visit(value):
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif isinstance(value, (str, Path)):
            path = Path(value)
            if path.is_absolute() and path.is_file():
                result[str(path)] = digest(path)
    for value in values:
        visit(value)
    return result


def verify_bindings(bindings):
    if any(digest(path) != expected for path, expected in bindings.items()):
        raise RuntimeError('Bottom trial source or input assets changed')
