"""Isolated R4 recipe for a separately verified, source-closed bottom paper.

The old 6 L setting is a reproducibility parameter, not a quality limit. Paper
keeps its own source reference and field. A narrow transition smooths only the
additive illumination field; no source RGB, print shape or alpha is blurred.
"""
import copy


def apply(view, face):
    evidence = view.get('source_label_material_evidence', {})
    if (face != 'bottom' or view.get('protection_policy') != 'none'
            or evidence.get('material_boundary_resolution') not in ('source_contour_frame_domain','source_aligned_metal_window')):
        return None
    from .resolved_bottom_material import validate
    material_binding=validate(view,evidence)
    domains = {d['name']: d for d in view['domains']}
    if set(domains) != {'printed_label', 'main_surface'}:
        raise ValueError('Closed bottom paper recipe requires its independent paper and frame domains')
    paper = domains['printed_label']
    previous = copy.deepcopy(paper)
    if paper.get('strength') != [.8, 0, 0] or not paper.get('reference_boxes'):
        raise ValueError('Closed paper recipe requires its source-fitted lightness field')
    blend = dict(domains=['printed_label', 'main_surface'], boundary_domain='printed_label',
                 sigma=1.25, kind='source_material_transition')
    existing = view.get('correction_blends', [])
    if existing and existing != [blend]:
        raise ValueError('Closed paper recipe cannot overwrite another material transition')
    paper['max_lab_shift'] = [24, 0, 0]
    view['correction_blends'] = [blend]
    return dict(method='source_closed_bottom_independent_paper_v1', material_binding=material_binding, previous=previous,
        selected=copy.deepcopy(paper), correction_blend=copy.deepcopy(blend),
        old_6L_is_not_a_user_quality_limit=True, paper_reference_and_fit_unchanged=True,
        budget_is_recipe_parameter_not_final_pixel_acceptance_limit=True,
        blend_support_is_algorithmic_not_physical_material_width=True,
        final_gamut_adjustment_may_propagate_beyond_direct_blend_support=True,
        prior_diagnostic_samples=['ip17pro/sku1', 'ip17pro/sku11'],
        other_samples_not_regression_verified=True, visual_approved=False)
