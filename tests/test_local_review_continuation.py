"""R6 static test draft -- PREPARED ONLY, NOT EXECUTED by batch_audit.

Run only when root schedules this file under the global processing guard.
Imports must resolve to integrated-r6; do not put this in a frozen runtime.

Design contract:
* Default measure_normal(..., contextual_review=False) retains hard failures.
* A source/material-bound bottom may record normal context as review: preserve
  the original 1.25 L limit, every measured/exceeded/insufficient row and counts.
  ``complete`` means coverage, NOT original_threshold_passed or visual approval.
* Pixel review varies only candidate 8-bit RGB by +/-0.5 per channel, clipped to
  [0,255]; source RGB/L is fixed. Every raw over-limit pair must have a possible
  added-step minimum <= the unchanged limit. This proves compatibility only,
  not quantization causation or a simultaneously feasible image for all pairs.
* A coherent-line, source, alpha, material binding or later-zone hard failure
  wins over any earlier review issue. Reviews cannot short-circuit the loop.
* No thresholds, masks, candidates or source pixels may be modified by review.

The scalar interval oracle below is independent of the helper under test and
evaluates monotone sRGB-to-Y-to-L endpoints. It never opens real batch images.
Tiny synthetic fixture images are created ONLY when root executes these tests.

API snapshot: local_effect_review.EffectLocalReviewRequired;
quantization_pixel_review(source_rgba,candidate_rgba,witness_bool,axis,limit);
normal_review_state.measure_normal(...,contextual_review=...); CONTEXT_POLICY;
spec.local_review_policy='quantization_bounded_pixel_review_v1'.

Still needed outside this unit draft: root's saved-candidate replay must show
retained lighting bytes proceed to structure/seam/export with review flags.
Do not count these prepared tests as passed or as another E2E run.
"""
import copy
import hashlib
import math

import numpy as np
from PIL import Image
import pytest

from phone_texture import lighting_review as review
from phone_texture import local_effect_review as local
from phone_texture import normal_review_state as normal
from phone_texture import shadow_lift_profile as profile


LIMIT = 1.25


def _hard(call):
    with pytest.raises(ValueError) as captured:
        call()
    assert not isinstance(captured.value, (
        local.EffectLocalReviewRequired, normal.EffectCoverageUnavailable,
        review.EffectThresholdFailure,
    )), "An invariant/mandatory failure must not become a review exception"
    return captured.value


def _normal_fixture():
    # Different adjacent-row intersections: [20,8,8,0,0].
    target = np.zeros((6, 60), bool)
    peer = np.zeros_like(target)
    target[:, :24] = True
    for y, count in enumerate((20, 20, 8, 20, 0, 20)):
        peer[y, 30:30 + count] = True
    delta = np.zeros(target.shape, float)
    delta[:, :24] = np.array([0, .5, 2.5, 2.75, 4.75, 5])[:, None]
    return delta, target, peer


def test_context_normal_keeps_all_rows_after_first_shortage_and_exceedance():
    delta, target, peer = _normal_fixture()
    before = (delta.copy(), target.copy(), peer.copy())
    result = normal.measure_normal(delta, target, peer, LIMIT, contextual_review=True)
    assert [r['row'] for r in result['rows']] == [0, 1, 2, 3, 4]
    assert [r['peer_pairs'] for r in result['rows']] == [20, 8, 8, 0, 0]
    assert [r['witness_pairs'] for r in result['rows']] == [24] * 5
    assert [r['row'] for r in result['missing_rows']] == [1, 2, 3, 4]
    assert result['threshold_exceeded_rows'] == [1]
    assert result['rows'][1]['added_gradient_residual_L75'] == 2
    assert result['rows'][2]['added_gradient_residual_L75'] == .25
    for row in result['rows'][3:]:
        assert row.get('added_gradient_residual_L75') is None
        assert row['peer_columns'] == []
    assert result['limit_L'] == LIMIT
    assert result['minimum_pairs_per_row'] == 10
    assert result['minimum_total_pairs'] == 20
    assert result['original_threshold_passed'] is False
    assert result['complete'] is False
    for original, current in zip(before, (delta, target, peer)):
        np.testing.assert_array_equal(original, current)


def test_context_normal_preserves_every_exceeded_row_including_the_last():
    delta, target, peer = _normal_fixture()
    peer[:, 30:50] = True
    delta[:, :24] = np.array([0, 2, 2.25, 5, 5.5, 8])[:, None]
    result = normal.measure_normal(delta, target, peer, LIMIT, contextual_review=True)
    assert result['threshold_exceeded_rows'] == [0, 2, 4]
    assert [r['added_gradient_residual_L75'] for r in result['rows']] == [2, .25, 2.75, .5, 2.5]
    assert result['complete'] is True  # Coverage only; never a quality pass.
    assert result['original_threshold_passed'] is False
    assert result['max_observed_residual_L75'] == 2.75


@pytest.mark.parametrize('contextual', [False, True])
def test_normal_equal_to_original_threshold_is_not_exceeded(contextual):
    delta = np.zeros((2, 60), float)
    target = np.zeros_like(delta, bool); target[:, :24] = True
    peer = np.zeros_like(target); peer[:, 30:50] = True
    delta[1, :24] = LIMIT
    result = normal.measure_normal(delta, target, peer, LIMIT, contextual_review=contextual)
    assert result['threshold_exceeded_rows'] == []
    assert result['original_threshold_passed'] is True
    assert result['limit_L'] == LIMIT


@pytest.mark.parametrize('peer_count', [8, 20])
def test_default_normal_still_hard_for_measurable_exceedance(peer_count):
    delta = np.zeros((2, 60), float); delta[1, :24] = 2
    target = np.zeros_like(delta, bool); target[:, :24] = True
    peer = np.zeros_like(target); peer[:, 30:30 + peer_count] = True
    with pytest.raises(ValueError, match='normal peer discontinuity'):
        normal.measure_normal(delta, target, peer, LIMIT)


def test_empty_context_is_unmeasured_not_zero_or_pass():
    empty = np.zeros((2, 60), bool)
    result = normal.measure_normal(np.zeros(empty.shape), empty, empty, LIMIT,
                                   contextual_review=True)
    assert result['rows'] == []
    assert result['max_observed_residual_L75'] is None
    assert result['original_threshold_passed'] is False
    assert result['not_applicable_certified'] is False
    assert result['physical_transition_support_certified'] is False


def _scalar_L(rgb):
    linear = []
    for value in rgb:
        s = float(value) / 255
        linear.append(s / 12.92 if s <= .04045 else ((s + .055) / 1.055) ** 2.4)
    y = sum(c * v for c, v in zip((.2126729, .7151522, .0721750), linear))
    value = 116 * y ** (1 / 3) - 16 if y > (6 / 29) ** 3 else (29 / 3) ** 3 * y
    return min(100., max(0., value))


def _pair_interval_minimum(source_pair, candidate_pair, *, widen_source=False):
    intervals = []
    for source_rgb, candidate_rgb in zip(source_pair, candidate_pair):
        source_lo = _scalar_L([max(0, x - .5) for x in source_rgb]) if widen_source else _scalar_L(source_rgb)
        source_hi = _scalar_L([min(255, x + .5) for x in source_rgb]) if widen_source else _scalar_L(source_rgb)
        lo = _scalar_L([max(0, x - .5) for x in candidate_rgb]) - source_hi
        hi = _scalar_L([min(255, x + .5) for x in candidate_rgb]) - source_lo
        intervals.append((lo, hi))
    low = intervals[1][0] - intervals[0][1]
    high = intervals[1][1] - intervals[0][0]
    return 0. if low <= 0 <= high else min(abs(low), abs(high))


def _pixel_fixture(start=100, end=104, *, all_columns=False):
    source = np.full((2, 24, 4), 100, np.uint8); source[..., 3] = 255
    candidate = source.copy(); candidate[..., :3] = start
    candidate[1, 0, :3] = end
    if all_columns:
        candidate[1, :, :3] = end
    return source, candidate, np.ones(source.shape[:2], bool)


@pytest.mark.parametrize('axis', [0, 1])
def test_quantization_review_is_candidate_only_bounded_and_never_quality_pass(axis):
    source, candidate, witness = _pixel_fixture()
    if axis:
        source, candidate, witness = (np.swapaxes(a, 0, 1).copy() for a in (source, candidate, witness))
    before = source.copy(), candidate.copy(), witness.copy()
    result = local.quantization_pixel_review(source, candidate, witness, axis, LIMIT)
    assert result is not None
    assert result['kind'] == 'quantization_compatible_pixel_warning'
    assert result['over_limit_pairs'] == 1 and result['pair_count'] == 24
    assert result['measured_max_step_L'] > LIMIT
    assert result['minimum_possible_steps_with_candidate_quantization_L'][0] <= LIMIT
    assert result['limit_L'] == LIMIT
    assert result['source_pixels_held_fixed'] is True
    assert result['candidate_RGB_uncertainty_per_channel'] == .5
    assert result['original_threshold_passed'] is False
    assert result['rounding_cause_proven'] is False
    assert result['quality_checks_complete'] is False
    for original, current in zip(before, (source, candidate, witness)):
        np.testing.assert_array_equal(original, current)


def test_one_incompatible_pair_makes_entire_pixel_review_hard():
    source, candidate, witness = _pixel_fixture()
    candidate[1, 23, :3] = 110
    assert local.quantization_pixel_review(source, candidate, witness, 0, LIMIT) is None


def test_source_uncertainty_must_not_be_added_to_candidate_uncertainty():
    source, candidate, witness = _pixel_fixture(end=106)
    source[1, 0, :3] = 101
    s = source[:, 0, :3].tolist(); c = candidate[:, 0, :3].tolist()
    assert _pair_interval_minimum(s, c) > LIMIT
    assert _pair_interval_minimum(s, c, widen_source=True) <= LIMIT
    assert local.quantization_pixel_review(source, candidate, witness, 0, LIMIT) is None


@pytest.mark.parametrize('start,end', [(0, 5), (251, 255), (104, 100)])
def test_quantization_interval_clips_code_endpoints_and_handles_negative_step(start, end):
    source, candidate, witness = _pixel_fixture(start, end)
    expected = _pair_interval_minimum(source[:, 0, :3].tolist(), candidate[:, 0, :3].tolist())
    result = local.quantization_pixel_review(source, candidate, witness, 0, LIMIT)
    assert result is not None
    assert result['minimum_possible_steps_with_candidate_quantization_L'][0] == pytest.approx(expected, abs=1e-12)


def test_quantization_uses_exact_unrounded_interval_boundary_without_threshold_padding():
    source, candidate, witness = _pixel_fixture()
    # Same physical endpoint, obtained from the actual lightness implementation
    # only to remove oracle floating-roundoff from this strict-boundary test.
    lower = review.lightness(candidate[..., :3].astype(float) - .5) - review.lightness(source[..., :3])
    upper = review.lightness(candidate[..., :3].astype(float) + .5) - review.lightness(source[..., :3])
    boundary = float(lower[1, 0] - upper[0, 0])
    assert boundary > 0
    assert local.quantization_pixel_review(source, candidate, witness, 0, boundary) is not None
    assert local.quantization_pixel_review(source, candidate, witness, 0,
                                           math.nextafter(boundary, -math.inf)) is None


def _mask(path, data):
    Image.fromarray(np.asarray(data, np.uint8) * 255).save(path)
    return str(path)


def _bound_local_case(tmp_path, *, resolution='source_aligned_metal_window', peer_count=20):
    source = np.full((4, 100, 4), 100, np.uint8); source[..., 3] = 255
    label = np.zeros(source.shape[:2], bool); label[:, 90:96] = True
    source[label, :3] = 200
    source_path = tmp_path/'bottom.png'; Image.fromarray(source).save(source_path)
    target = np.zeros_like(label); target[:, :24] = True
    peer = np.zeros_like(label); peer[:, 32:32 + peer_count] = True
    label_path = _mask(tmp_path/'label.png', label)
    source_hash = hashlib.sha256(source.tobytes()).hexdigest()
    label_hash = hashlib.sha256(label.tobytes()).hexdigest()
    evidence = dict(material_boundary_resolution=resolution)
    if resolution == 'original_source_closure':
        evidence['unmarked_source_closure'] = dict(source_rgba_sha256=source_hash, mask_sha256=label_hash)
    else:
        evidence['aligned_source_closure'] = dict(original_source_rgba_sha256=source_hash,
                                                  original_mask_sha256=label_hash)
    view = dict(name='bottom', source=str(source_path), source_sha256=review.digest(source_path),
        protection_policy='none', source_label_fit_exclusion_mask=label_path,
        source_label_location_hint_mask=label_path, source_label_material_evidence=evidence,
        domains=[dict(name='printed_label', mask=label_path),
                 dict(name='main_surface', mask=_mask(tmp_path/'metal.png', ~label))])
    spec = dict(mode='regional', action='aluminum', reason='Synthetic R6 continuation control',
        protection_policy='none', source_sha256=review.digest(source_path),
        normal_coverage_policy=normal.CONTEXT_POLICY, source_material_view=view,
        source_paper_material_boundary=evidence, local_review_policy=local.POLICY,
        local_checks={'bottom_paper_metal_normal':dict(mask=_mask(tmp_path/'target.png', target),
            normal_peer_mask=_mask(tmp_path/'peer.png', peer), min_L=0, max_L=100,
            max_normal_peer_step_L=LIMIT)})
    return source_path, source, spec, np.ones(label.shape, bool), np.zeros(label.shape, bool)


def _run(case, candidate, tmp_path):
    source_path, _, spec, allowed, protected = case
    output = tmp_path/'candidate.png'; Image.fromarray(candidate).save(output)
    return profile.measure_effect(review.measure_local, spec, source_path, output,
                                  allowed, protected, profile.NAME)


@pytest.mark.parametrize('resolution', ['source_aligned_metal_window', 'original_source_closure'])
def test_source_bound_legacy_bottom_context_keeps_rendered_candidate_as_review(tmp_path, resolution):
    case = _bound_local_case(tmp_path, resolution=resolution)
    assert normal.review_enabled(case[2], case[0]) is True
    candidate = case[1].copy(); candidate[1:, :24, :3] = 104
    result = _run(case, candidate, tmp_path)
    assert result['status'] == 'effect_review_required'
    assert not result['passed'] and not result['original_effect_check_passed']
    assert not result['quality_checks_complete']
    coverage = result['effect_measurements']['bottom_paper_metal_normal']['normal_coverage']
    assert coverage['threshold_exceeded_rows'] == [0]
    assert coverage['limit_L'] == LIMIT


@pytest.mark.parametrize('fault', ['source_hash', 'view_source', 'label_bytes', 'domain_bytes', 'resolution', 'face'])
def test_material_or_source_binding_fault_is_hard_even_with_context_policy(tmp_path, fault):
    case = _bound_local_case(tmp_path); spec = case[2]; view = spec['source_material_view']
    if fault == 'source_hash': spec['source_sha256'] = 'not-the-source'
    elif fault == 'view_source': view['source'] = str(tmp_path/'another-source.png')
    elif fault == 'label_bytes': Image.new('L', (100, 4), 255).save(view['source_label_fit_exclusion_mask'])
    elif fault == 'domain_bytes': Image.new('L', (100, 4), 255).save(view['domains'][1]['mask'])
    elif fault == 'resolution': spec['source_paper_material_boundary']['material_boundary_resolution'] = 'unresolved'
    elif fault == 'face': view['name'] = 'left'
    _hard(lambda: _run(case, case[1].copy(), tmp_path))


def test_context_policy_reuses_independent_source_closed_proof_and_rejects_mask_tampering(tmp_path):
    from phone_texture import source_material_bridge as bridge
    # Same synthetic scene shape used by the independent material-bridge suite;
    # actually resolve it when scheduled, never fabricate a material-pass flag.
    source = np.full((96, 160, 4), 100, np.uint8); source[..., 3] = 255
    source[:70, 50:101, :3] = 200
    field = np.rint(80 - 45 * np.exp(-.5 * ((np.arange(160) - 75) / 55) ** 2)).astype('uint8')
    source[70:, :, :3] = field[None, :, None]
    path = tmp_path/'bottom.png'; Image.fromarray(source).save(path)
    hint = np.zeros(source.shape[:2], bool); hint[:, 50:101] = True
    proof = dict(source_strip_edges=[[0, 49.5], [0, 100.5]], seed_box=[54, 10, 97, 60])
    _, evidence = bridge.resolve_current_source_material(path, hint, proof, tmp_path/'fresh')
    view = dict(name='bottom', source=str(path), source_sha256=bridge.sha(path), protection_policy='none',
        source_label_fit_exclusion_mask=evidence['paper_material_mask'],
        source_label_location_hint_mask=evidence['hint_mask'], source_label_material_evidence=evidence)
    spec = dict(mode='regional', source_sha256=bridge.sha(path), normal_coverage_policy=normal.CONTEXT_POLICY,
        source_material_view=view, source_paper_material_boundary=evidence)
    assert normal.review_enabled(spec, path) is True
    assert evidence['lighting_effect_verified'] is False and evidence['visual_approved'] is False
    Image.new('L', (160, 96), 255).save(evidence['paper_material_mask'])
    _hard(lambda: normal.review_enabled(spec, path))


def test_alpha_change_stays_hard_with_both_review_policies_enabled(tmp_path):
    case = _bound_local_case(tmp_path)
    candidate = case[1].copy(); candidate[1:, :24, :3] = 104; candidate[0, 0, 3] = 254
    _hard(lambda: _run(case, candidate, tmp_path))


def test_normal_review_does_not_skip_same_zone_coherent_line_failure(tmp_path):
    case = _bound_local_case(tmp_path)
    case[2]['local_checks']['bottom_paper_metal_normal'].update(max_added_step_L=LIMIT, step_axis=0)
    candidate = case[1].copy(); candidate[1:, :24, :3] = 104
    error = _hard(lambda: _run(case, candidate, tmp_path))
    assert 'correction discontinuity' in str(error)


@pytest.mark.parametrize('warning_first', [True, False])
def test_quantization_review_does_not_hide_other_zone_hard_failure(tmp_path, warning_first):
    case = _bound_local_case(tmp_path); spec = case[2]
    first = np.zeros(case[1].shape[:2], bool); first[:, :24] = True
    second = np.zeros_like(first); second[:, 32:56] = True
    pixel = dict(mask=_mask(tmp_path/'pixel.png', first), min_L=0, max_L=100,
                 max_pixel_added_step_L=LIMIT, step_axis=0)
    hard = dict(mask=_mask(tmp_path/'hard.png', second), min_L=0, max_L=100,
                max_added_step_L=LIMIT, step_axis=0)
    pairs = [('pixel_warning', pixel), ('coherent_hard', hard)]
    spec['local_checks'] = dict(pairs if warning_first else reversed(pairs))
    candidate = case[1].copy(); candidate[1, 0, :3] = 104; candidate[1:, 32:56, :3] = 110
    _hard(lambda: _run(case, candidate, tmp_path))


def test_coherent_line_remains_hard_even_if_every_pixel_is_quantization_compatible(tmp_path):
    case = _bound_local_case(tmp_path); spec = case[2]
    zone = copy.deepcopy(spec['local_checks']['bottom_paper_metal_normal'])
    zone.pop('normal_peer_mask'); zone.pop('max_normal_peer_step_L')
    zone.update(max_added_step_L=LIMIT, max_pixel_added_step_L=LIMIT, step_axis=0)
    spec['local_checks'] = {'combined': zone}
    candidate = case[1].copy(); candidate[1:, :24, :3] = 104
    _hard(lambda: _run(case, candidate, tmp_path))


def test_quantization_review_requires_explicit_policy_opt_in(tmp_path):
    case = _bound_local_case(tmp_path); spec = case[2]
    zone = copy.deepcopy(spec['local_checks']['bottom_paper_metal_normal'])
    zone.pop('normal_peer_mask'); zone.pop('max_normal_peer_step_L')
    zone.update(max_pixel_added_step_L=LIMIT, step_axis=0)
    spec['local_checks'] = {'pixel_only': zone}
    spec.pop('local_review_policy')
    candidate = case[1].copy(); candidate[1, 0, :3] = 104
    _hard(lambda: _run(case, candidate, tmp_path))


def test_normal_quantization_and_brightness_reviews_are_all_retained(tmp_path):
    case = _bound_local_case(tmp_path, peer_count=8); spec = case[2]
    quant = np.zeros(case[1].shape[:2], bool); quant[:, 60:84] = True
    brightness = np.zeros_like(quant); brightness[:, 86:88] = True
    spec['local_checks']['quantized_pixel'] = dict(mask=_mask(tmp_path/'quant.png', quant),
        min_L=0, max_L=100, max_pixel_added_step_L=LIMIT, step_axis=0)
    spec['local_checks']['brightness_warning'] = dict(mask=_mask(tmp_path/'brightness.png', brightness),
        min_L=90, max_L=100)
    candidate = case[1].copy(); candidate[1:, :24, :3] = 104; candidate[1, 60, :3] = 104
    result = _run(case, candidate, tmp_path)
    kinds = {w['kind'] for w in result['warnings']}
    assert {'normal_continuity_warning', 'normal_coverage_unavailable',
            'quantization_compatible_pixel_warning', 'lighting_effect_target_warning'} <= kinds
    assert set(result['effect_measurements']) == set(spec['local_checks'])
    assert result['passed'] is False and result['quality_checks_complete'] is False


def test_baseline_profile_does_not_silently_opt_into_local_review(tmp_path):
    case = _bound_local_case(tmp_path); source_path, source, spec, allowed, protected = case
    candidate = source.copy(); candidate[1:, :24, :3] = 104
    output = tmp_path/'candidate.png'; Image.fromarray(candidate).save(output)
    with pytest.raises(local.EffectLocalReviewRequired):
        profile.measure_effect(review.measure_local, spec, source_path, output,
                               allowed, protected, 'baseline')


# Remaining integration controls, deliberately not represented by fake passes:
# 1. Root's guarded Max12/28/53/61 saved-candidate replay: confirm all normal rows
#    and every >1.25 row preserved; Max61 left 652 pairs / one raw over-limit pair
#    uses the unrounded 1.290815983482922 measurement, not displayed '1.29'.
# 2. A review-only lighting result must retain the original attempt bytes, proceed
#    to independently validated structure/seam, and export without claiming human
#    visual or UV approval. Source/alpha/material failure must still reject it.
