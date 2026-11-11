"""Source-fitted lighting target approved on Pro14/34; other SKUs need review.

This changes illumination proposals, never source RGB, alpha, fit support floors,
antenna positions or effect-check verdicts. Printed paper has its own small
illumination field rather than a protected rectangle or borrowed metal field.
"""
import copy
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from .lighting_review import EffectThresholdFailure
from .normal_review_state import EffectCoverageUnavailable, effect_review_result
from .local_effect_review import EffectLocalReviewRequired, effect_result as local_effect_result
from .label_transition_migration import migrate_label_blends, rebuild_label_blends

NAME = 'accepted-shadow-lift-v2'
PROFILES = ('baseline', 'accepted-shadow-lift-v1', NAME)


def printed_label_sigma(a, label, face):
    """Size a vertical paper field from its own opaque source cross-section.

    The old whole-image fraction made a ~114 px paper strip use a 7 px
    transverse field. Its narrowing, partly transparent terminal then lost
    support even though the same paper had ample source samples. A quarter
    strip-width Gaussian estimates this material's low-frequency illumination;
    it is chosen once from source geometry, never enlarged until a check passes.
    Unresolved or disconnected hints retain the original proposal and guards.
    """
    h, w = label.shape
    sigma = [max(8, h*.007), max(3, w*.007)]
    proof = dict(method='original_image_fraction', sigma=list(sigma),
                 support_floor=.008, source_geometry_only=True)
    if face != 'back' or not label.any():
        return sigma, proof
    fit = label & (a[..., 3] >= 240)
    _, count = ndimage.label(label, np.ones((3, 3), dtype=bool))
    _, fit_count = ndimage.label(fit, np.ones((3, 3), dtype=bool))
    if count != 1 or fit_count != 1 or np.count_nonzero(fit) < 20:
        return sigma, dict(proof, reason='not_one_opaque_supported_paper_component')
    widths = []
    rows = []
    for y in np.flatnonzero(fit.sum(axis=1) >= 20):
        xx = np.flatnonzero(fit[y])
        if len(xx) != xx[-1]-xx[0]+1:
            continue
        widths.append(int(xx[-1]-xx[0]+1))
        rows.append(int(y))
    if len(rows) < 20:
        return sigma, dict(proof, reason='insufficient_source_cross_sections')
    width = float(np.median(widths))
    height = rows[-1]-rows[0]+1
    if not (width <= w*.18 and height >= width*1.5):
        return sigma, dict(proof, reason='not_a_supported_vertical_paper_strip')
    sigma[1] = max(sigma[1], width/4.)
    return sigma, dict(proof, method='opaque_vertical_paper_cross_section',
                      sigma=list(sigma), source_width_median_px=width,
                      opaque_row_count=len(rows), opaque_height_px=height,
                      opaque_connected_components=fit_count,
                      cross_section_sigma_fraction=.25,
                      unresolved_pixels_ignored=False,
                      source_fit_support_threshold_unchanged=True)


def reference(a, domain, engine):
    lab = engine.rgb_to_lab(a[..., :3])
    mask = np.array(Image.open(domain.get('fit_mask', domain['mask'])).convert('L')) > 0
    mask &= a[..., 3] >= 250
    percentile = 75 if 'antenna' in domain['name'] else 85
    for hh, ww in ((7, 7), (3, 12), (12, 3), (1, 24), (24, 1)):
        support = ndimage.uniform_filter(mask.astype(float), (hh, ww), mode='constant') >= 1-1e-6
        yy, xx = np.nonzero(support)
        if not len(xx):
            continue
        means = ndimage.uniform_filter(lab[..., 0], (hh, ww), mode='nearest')
        squares = ndimage.uniform_filter(lab[..., 0]**2, (hh, ww), mode='nearest')
        deviation = np.sqrt(np.maximum(0, squares-means**2))
        target = np.percentile(means[yy, xx], percentile)
        index = int(np.argmin(abs(means[yy, xx]-target)+.15*deviation[yy, xx]))
        y0, x0 = int(yy[index])-hh//2, int(xx[index])-ww//2
        box = [x0, y0, x0+ww, y0+hh]
        if not mask[y0:y0+hh, x0:x0+ww].all():
            raise ValueError('Reference patch crosses a source material boundary')
        return box, dict(percentile=percentile, patch_xyxy=box,
            patch_source_median_L=float(np.median(lab[y0:y0+hh, x0:x0+ww, 0])))
    raise ValueError('No complete opaque reference patch '+domain['name'])


def apply(job, face, folder, profile=NAME):
    if profile not in PROFILES:
        raise ValueError('Unknown lighting profile')
    if profile == 'baseline':
        return job, dict(profile=profile)
    from .lighting import engine
    render = engine('aluminum')
    result = copy.deepcopy(job)
    folder = Path(folder)
    view = result['views'][0]
    if result.get('color_mode') != 'source_lightness':
        raise ValueError('Shadow lift requires per-source lightness mode')
    if view['name'] != face or render.digest(view['source']) != view['source_sha256']:
        raise ValueError('Shadow lift source identity differs')
    a = np.array(Image.open(view['source']).convert('RGBA'))
    h, w = a.shape[:2]
    changes = []
    for domain in view['domains']:
        previous = copy.deepcopy(domain)
        if profile == NAME and face == 'back' and domain['name'] == 'rear_perimeter':
            # This contour model already lifts photographic depth shadows.
            # Raising its source reference and budget also clips bright button
            # metal, introducing new tangent steps. Retain the generic source-
            # fitted perimeter model while other materials use this profile.
            # Generation and acceptance have separate references: retaining
            # the established render must not lower V2's shadow requirements.
            box, acceptance_proof = reference(a, previous, render)
            domain['acceptance_reference_boxes'] = [box]
            changes.append(dict(domain=domain['name'],
                retained_generic_source_contour_model=True, parameters=previous,
                acceptance_reference_boxes=[box], acceptance_reference=acceptance_proof,
                acceptance_basis='Original V2 source target; render reference retained separately',
                acceptance_thresholds_relaxed=False))
            continue
        box, proof = reference(a, domain, render)
        domain.update(reference_boxes=[box],
            strength=[.90 if 'antenna' in domain['name'] else .98, 0, 0],
            max_lab_shift=[60, 0, 0])
        if face == 'front' and domain['name'] == 'main_surface':
            domain['reference_profile'] = 'contour_depth_lift'
        if face in ('top', 'bottom') and domain['name'] == 'main_surface':
            domain.update(sigma=[4.5, max(30, w*.045)], gamut_continuity_follow_field=True)
        if face in ('left', 'right') and domain['name'] == 'main_surface':
            domain['sigma'] = [max(36, h*.025), max(4, w*.035)]
        changes.append(dict(domain=domain['name'], old=previous, new=copy.deepcopy(domain), reference=proof))
    if view.get('protection_policy') == 'none' and view.get('source_label_fit_exclusion_mask'):
        label = (np.array(Image.open(view['source_label_fit_exclusion_mask']).convert('L')) > 0) & (a[..., 3] > 0)
        union = np.zeros(label.shape, bool)
        retained = []
        removed_label_components = []
        old_blends = copy.deepcopy(view.get('correction_blends', []))
        before_masks, after_masks = {}, {}
        effective = a[..., 3] > 0
        if view.get('editable_mask'):
            editable = np.array(Image.open(view['editable_mask']).convert('L'))
            if editable.shape != effective.shape:
                raise ValueError('Editable mask dimensions differ from source')
            effective &= editable > 0
        for domain in view['domains']:
            mask = (np.array(Image.open(domain['mask']).convert('L')) > 0) & (a[..., 3] > 0)
            before_masks[domain['name']] = mask & effective
            union |= mask
            revised = mask & ~label
            if domain['name'].startswith('label_component_') and revised.any() and np.any(mask & label):
                raise ValueError('Partial paper-component migration is unsupported')
            if not revised.any():
                if not domain['name'].startswith('label_component_'):
                    raise ValueError('Non-paper material entirely overlaps label hint: '+domain['name'])
                removed_label_components.append(domain['name'])
                continue
            path = folder/(domain['name']+'-without-label.png')
            Image.fromarray(revised.astype('uint8')*255).save(path)
            domain['mask'] = str(path)
            retained.append(domain)
            after_masks[domain['name']] = revised & effective
        view['domains'] = retained
        if removed_label_components:
            changes.append(dict(removed_redundant_paper_components=removed_label_components,reason='replaced by independently source-fitted printed_label, not frozen or omitted'))
        label &= union
        if label.any():
            path, fit = folder/'printed-label-material.png', folder/'printed-label-fit.png'
            Image.fromarray(label.astype('uint8')*255).save(path)
            Image.fromarray((label & (a[..., 3] >= 240)).astype('uint8')*255).save(fit)
            sigma, sigma_proof = printed_label_sigma(a, label, face) if profile == NAME else (
                [max(8, h*.007), max(3, w*.007)], dict(method='original_image_fraction'))
            domain = dict(name='printed_label', mask=str(path), fit_mask=str(fit),
                sigma=sigma, strength=[.8, 0, 0],
                max_lab_shift=[6, 0, 0], feather=0, correction_smooth=1.5,
                gamut_continuity_step_L=.2, gamut_continuity_follow_field=True)
            box, proof = reference(a, domain, render)
            domain['reference_boxes'] = [box]
            view['domains'].append(domain)
            changes.append(dict(domain='printed_label', new=domain, reference=proof,
                hard_pixel_protection=False, source_fit_support_threshold=.008,
                source_field_scale=sigma_proof))
            after_masks['printed_label'] = label & effective
        try:
            migrated, transition_proof = migrate_label_blends(
                old_blends, before_masks, after_masks, removed_label_components)
        except ValueError as error:
            # Exact inheritance only holds when the refined paper equals the
            # old components pixel for pixel. Otherwise rebuild the narrow
            # paper transitions on the refined masks instead of skipping the face.
            migrated, transition_proof = rebuild_label_blends(
                old_blends, after_masks, removed_label_components, str(error))
        if 'correction_blends' in view or migrated:
            view['correction_blends'] = migrated
        changes.append(dict(printed_label_transition_migration=transition_proof))
    from .closed_paper_lighting import apply as apply_closed_paper
    closed_paper = apply_closed_paper(view, face)
    if closed_paper is not None:
        changes.append(dict(source_closed_paper_illumination=closed_paper))
    view['reference_boxes'] = view['domains'][0]['reference_boxes']
    result['lighting_profile'] = profile
    return result, dict(profile=profile,
        approved_sample_scope=['ip17pro/sku14', 'ip17pro/sku34'] if profile=='accepted-shadow-lift-v1' else [],
        reference_samples_from_prior_profile=['ip17pro/sku14', 'ip17pro/sku34'],
        material_boundary_owner='source_verified_automatic_materials_no_secondary_label_reclassification',
        other_SKUs_user_visual_approved=False, changes=changes,
        source_fit_support_threshold=.008, geometry_changed=False)


def measure_effect(function, scope, source, output, allowed, protected, profile):
    """Keep aesthetic failures visible while allowing remaining stages to run."""
    try:
        return function(scope, source, output, allowed, protected)
    except EffectLocalReviewRequired as error:
        if profile not in PROFILES[1:]:
            raise
        return local_effect_result(error)
    except EffectCoverageUnavailable as error:
        if profile not in PROFILES[1:]:
            raise
        return effect_review_result(error)
    except EffectThresholdFailure as error:
        if profile not in PROFILES[1:]:
            raise
        return dict(passed=False, original_effect_check_passed=False,
            status='effect_warning_not_quality_passed', error=str(error),
            effect_failures=error.failures, effect_measurements=error.reports,
            non_aesthetic_scope_checks_complete=True,
            action='continue_remaining_stages_with_warning_no_raw_fallback')
