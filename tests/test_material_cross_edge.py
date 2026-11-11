"""Small material-only controls, prepared statically; root runs under guard."""
import copy

import numpy as np
from PIL import Image
import pytest

from phone_texture import auto_material, bottom_paper_audit, bottom_paper_closure
from phone_texture import material_cross_edge, source_material_bridge


def sloped_scene():
    # Synthetic dimensions only. Runtime implementation contains no SKU/ROI.
    h, w = 120, 220
    a = np.full((h, w, 4), 80, dtype=np.uint8)
    a[..., 3] = 255
    yy, xx = np.mgrid[:h, :w]
    region = (xx >= 60) & (xx <= 140) & (yy < 80 + .14*(xx-100))
    a[region, :3] = 220
    hint = (xx >= 60) & (xx <= 140)
    proof = dict(source_strip_edges=[[0, 59.5], [0, 140.5]],
                 seed_box=[65, 10, 135, 65])
    return a, hint, proof


def legacy_rejection(a, hint, proof):
    with pytest.raises(ValueError) as caught:
        bottom_paper_audit._source_cross_edge(a, hint, proof)
    assert str(caught.value) in (
        'Paper end is not supported across the strip',
        'Paper end does not support a bounded source cross-edge')
    return str(caught.value)


def test_continuous_sloped_source_uses_original_search_and_requires_topology():
    a, hint, proof = sloped_scene()
    original = a.copy(), hint.copy(), copy.deepcopy(proof)
    error = legacy_rejection(a, hint, proof)
    edge, topology, evidence = material_cross_edge.select_material_cross_edge(
        a, hint, proof, legacy_error=error)
    assert edge['method'] == 'material_continuous_source_cross_edge_v2'
    assert abs(edge['source_end_line'][0]) <= .15
    assert edge['source_end_support'] >= .75
    assert topology['closed_paper_contour_supported'] and topology['failures'] == []
    assert topology['connected_transverse']['support'] >= .75
    assert all(topology['corners'][side]['route_passes_positive_turn'] for side in ('left', 'right'))
    assert evidence['closed_candidate_count'] == 1
    assert evidence['numeric_thresholds_unchanged'] and evidence['original_y_search_unchanged']
    assert not evidence['material_certified'] and not evidence['publishable']
    np.testing.assert_array_equal(a, original[0])
    np.testing.assert_array_equal(hint, original[1])
    assert proof == original[2]


def test_closed_material_binds_new_source_line_and_proposal_hash(tmp_path):
    a, hint, proof = sloped_scene()
    legacy_rejection(a, hint, proof)
    source = tmp_path / 'bottom.png'
    Image.fromarray(a).save(source)
    paper, evidence = source_material_bridge.resolve_current_source_material(
        source, hint, proof, tmp_path / 'material')
    view = dict(name='bottom', source=str(source), source_sha256=evidence['source_sha256'],
        protection_policy='none', source_label_fit_exclusion_mask=evidence['paper_material_mask'],
        source_label_location_hint_mask=evidence['hint_mask'], source_label_material_evidence=evidence)
    scope = dict(source_paper_material_boundary=evidence, source_sha256=evidence['source_sha256'])
    checked = source_material_bridge.validate_source_contour_material(view, evidence, scope)
    assert checked['material_domain_verified'] and not checked['lighting_effect_verified']
    assert not evidence['pixel_protection'] and not evidence['paper_lighting_budget_set']
    report = source_material_bridge.read(evidence['source_contour_report'])
    binding = report['material_cross_edge_proposal']
    proposal = source_material_bridge.read(binding['path'])
    assert proposal['source_sha256'] == evidence['source_sha256']
    assert proposal['selected_edge'] == report['cross_edge']
    assert report['cross_edge']['source_end_line'] == evidence['source_end_line']
    assert any(path.endswith('/material_cross_edge.py') for path in evidence['supplementary_implementation_hashes'])
    from pathlib import Path
    Path(binding['path']).write_text('{}\n')
    with pytest.raises(ValueError, match='proposal evidence changed'):
        source_material_bridge.validate_source_contour_material(view, evidence, scope)


def test_unresolved_bottom_paper_no_longer_enters_material_fallback(tmp_path, monkeypatch):
    a, hint, proof = sloped_scene()
    source = tmp_path / 'bottom.png'
    Image.fromarray(a).save(source)
    monkeypatch.setattr(auto_material, 'paper_mask', lambda *a, **k: (hint.copy(), proof.copy()))
    unresolved = dict(proof, material_boundary_resolution='unresolved_no_source_cross_edge',
                      material_boundary_warning='Paper end is not supported across the strip')
    monkeypatch.setattr(bottom_paper_closure, 'resolve_material_strip_end',
                        lambda *a: (hint.copy(), unresolved))
    observed = []

    class ReachedMaterialFallback(RuntimeError):
        pass

    def detect(actual_source, actual_hint, actual_proof, folder, **kwargs):
        observed.append((actual_source, actual_proof, kwargs))
        np.testing.assert_array_equal(actual_hint, hint)
        raise ReachedMaterialFallback('routing observed; no downstream computation')

    monkeypatch.setattr(source_material_bridge, 'resolve_current_source_material', detect)
    # Bottom paper is lit with the frame metal: neither closure nor fallback runs.
    job, *_ = auto_material.propose_job(source, 'bottom', tmp_path / 'assets', protection_policy='none')
    assert observed == [] and job['views'][0]['domains'][0]['name'] == 'main_surface'


def test_no_gradient_support_is_not_promoted_to_material():
    a, hint, proof = sloped_scene()
    a[..., :3] = 100
    with pytest.raises(ValueError, match='Bottom paper material boundary unresolved'):
        material_cross_edge.select_material_cross_edge(a, hint, proof,
            legacy_error='Paper end is not supported across the strip')


def test_printed_cross_band_with_continuing_paper_is_not_closed(monkeypatch):
    a, hint, proof = sloped_scene()
    a[:, 60:141, :3] = 220
    a[78:82, 60:141, :3] = 40
    # Scope this control to candidate/topology selection; a horizontal printed
    # band can pass the old detector and must never certify the continuing sheet.
    original = material_cross_edge.trace_source

    def trace(*args):
        report, arrays = original(*args)
        report['exact_mirrored_error'] = 'Paper end is not supported across the strip'
        return report, arrays

    monkeypatch.setattr(material_cross_edge, 'trace_source', trace)
    with pytest.raises(ValueError, match='no unique source-closed contour') as caught:
        material_cross_edge.select_material_cross_edge(a, hint, proof,
            legacy_error='Paper end is not supported across the strip')
    evidence = caught.value.material_boundary_evidence
    assert evidence['closed_candidate_count'] == 0
    assert not evidence['material_certified']


def test_multiple_source_closed_candidates_are_ambiguous(monkeypatch):
    # Do not use contrast ranking to hide a second complete closure candidate.
    a, hint, proof = sloped_scene()
    monkeypatch.setattr(material_cross_edge, 'trace_source', lambda *a: (
        {'exact_mirrored_error': 'Paper end is not supported across the strip', 'search': [60, 118]}, {}))
    monkeypatch.setattr(material_cross_edge, 'continuous_candidates', lambda *a: dict(
        connected_components=[], eligible_components=2, topology_attempt_limit=3,
        topology=[{'report': {'closed_paper_contour_supported': True, 'failures': []}} for _ in range(2)]))
    with pytest.raises(ValueError, match='no unique source-closed contour') as caught:
        material_cross_edge.select_material_cross_edge(a, hint, proof,
            legacy_error='Paper end is not supported across the strip')
    assert caught.value.material_boundary_evidence['closed_candidate_count'] == 2


def test_unrelated_legacy_failure_cannot_enable_material_fallback():
    a, hint, proof = sloped_scene()
    with pytest.raises(ValueError, match='outside the material fallback scope'):
        material_cross_edge.select_material_cross_edge(a, hint, proof,
            legacy_error='Paper closure lacks opaque source/background samples')


def extension_trace():
    # Gate-only source observation fixture; it grants exploration, not material.
    trace = dict(search=[2, 4], actual_side_corridor_observations={
        'left': [dict(y=y, source_pixels=[dict(x=2, alpha=255, signed_gx=2., forward=True)])
                 for y in (3, 4, 5)],
        'right': []})
    original = dict(connected_components=[dict(component=1, line=[0., 4.],
        ridge=[dict(x=2, y=4), dict(x=3, y=4)])], eligible_components=0, topology=[])
    return trace, original


@pytest.mark.parametrize('missing', ['touch', 'side', 'adjacency'])
def test_extension_requires_real_boundary_touch_and_connected_side_pixels(missing):
    trace, original = extension_trace()
    assert material_cross_edge.extension_gate(trace, original)['allowed']
    if missing == 'touch':
        for p in original['connected_components'][0]['ridge']:
            p['y'] = 3
    elif missing == 'side':
        trace['actual_side_corridor_observations']['left'][1]['source_pixels'] = []
    else:
        trace['actual_side_corridor_observations']['left'][2]['source_pixels'][0]['x'] = 6
    assert not material_cross_edge.extension_gate(trace, original)['allowed']


def bound_extended_record():
    """Binding fixture only; existing core/frame validation is separately tested."""
    import hashlib
    from pathlib import Path
    a = np.full((10, 6, 4), 80, dtype=np.uint8)
    a[..., 3] = 255
    a[9, :, 3] = 0
    hint = np.zeros((10, 6), bool)
    hint[:, 2:4] = True
    trace, original = extension_trace()
    trace['thresholds'] = dict(gradient=.7, support=.75, residual_px=2.5,
        maximum_abs_slope=.15, sigma=.8, physical_transition_width_estimated=False)
    trace['row_observations'] = [dict(y=y, outside_left=[0], outside_right=[5]) for y in range(10)]
    trace['column_observations'] = [dict(x=2), dict(x=3)]
    method = 'material_continuous_source_cross_edge_v2'
    input_edge = dict(source_end_line=[0., 7.], source_end_peak=7, source_end_search=[2, 8], source_end_support=1.)
    edge = dict(input_edge, method=method)
    component = dict(component=1, eligible_for_existing_topology_diagnostic=True, line=[0., 7.],
        expected_columns=2, within_residual_support=1., source_nodes_yx=[[7, 2], [7, 3]],
        ridge=[dict(x=x, y=7, alpha=255, raw_down=2., specific=2.) for x in (2, 3)])
    selected = dict(eligible_components=1, topology_attempt_limit=3, connected_components=[component],
        original_y_search_unchanged=False, old_search_lower_bound_unchanged=True,
        topology=[dict(component=1, edge=input_edge,
                       report=dict(closed_paper_contour_supported=True, failures=[]))])
    rows = [dict(y=y, locator_pixels=2, locator_opaque_fraction=1. if y < 9 else 0.,
        source_lateral_references_opaque=y < 9, source_row_eligible=y < 9) for y in range(2, 10)]
    extension = dict(extension_performed=True,
        extension_gate=material_cross_edge.extension_gate(trace, original),
        marker_used_as_search_prior_only=True, source_pixels_extrapolated=False,
        thresholds=dict(gradient=.7, support=.75, residual_px=2.5, maximum_abs_slope=.15, alpha_minimum=240),
        opaque_support_extent=rows, last_supported_source_row=8, extended_search=[2, 8],
        continuous_source_candidates=selected, unique_existing_U_closure_supported=True)
    module_path = Path(material_cross_edge.__file__).resolve()
    proposal = dict(method=method, source_sha256='bound-source', selected_edge=edge,
        selected_component=1, closed_candidate_count=1, numeric_thresholds_unchanged=True,
        implementation_path=str(module_path), implementation_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
        source_size=[6, 10], source_array=dict(shape=[10, 6, 4], dtype='uint8', byte_order='uint8_not_applicable', storage_order='C'),
        locator_array=dict(shape=[10, 6], dtype='bool', byte_order='bool_not_applicable', storage_order='C'),
        source_rgba_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
        source_locator_array_sha256=hashlib.sha256(hint.tobytes()).hexdigest(),
        legacy_trace=trace, original_continuous_source_candidates=original,
        continuous_source_candidates=selected, search_extension=extension, original_y_search_unchanged=False)
    return a, hint, proposal, edge


def test_expanded_search_is_explicitly_bound_to_actual_opaque_source_rows():
    a, hint, proposal, edge = bound_extended_record()
    checked = material_cross_edge.validate_proposal_record(proposal, 'bound-source', edge, rgba=a, hint=hint)
    assert checked['search_extension_verified']
    assert not checked['material_certified'] and not checked['lighting_effect_verified']
    changed = a.copy()
    changed[8, 0, 3] = 0
    with pytest.raises(ValueError, match='array binding changed'):
        material_cross_edge.validate_proposal_record(proposal, 'bound-source', edge, rgba=changed, hint=hint)


@pytest.mark.parametrize('damage', ['false_unchanged', 'extra_row', 'opaque_claim', 'disconnected_track'])
def test_extended_metadata_cannot_waive_its_coverage_obligation(damage):
    a, hint, proposal, edge = bound_extended_record()
    if damage == 'false_unchanged':
        proposal['original_y_search_unchanged'] = True
    elif damage == 'extra_row':
        proposal['search_extension']['last_supported_source_row'] = 9
    elif damage == 'opaque_claim':
        proposal['search_extension']['opaque_support_extent'][-1]['source_row_eligible'] = True
    else:
        proposal['legacy_trace']['actual_side_corridor_observations']['left'][1]['source_pixels'] = []
    with pytest.raises(ValueError):
        material_cross_edge.validate_proposal_record(proposal, 'bound-source', edge, rgba=a, hint=hint)


def test_extended_graph_keeps_invalid_rows_in_place(monkeypatch):
    a, hint, proposal, edge = bound_extended_record()
    # A missing opaque row stays a NaN row in the full coordinate grid. It
    # must not disappear and falsely make graph nodes on rows 5 and 7 adjacent.
    a[6, 0, 3] = 0
    trace = proposal['legacy_trace']
    arrays = dict(cols=np.array([2, 3]), rows=np.arange(2, 5), down=np.ones((10, 6)),
                  alpha=a[..., 3], left=1, right=4)
    seen = []

    def observe(source, locator, proof, fresh):
        seen.append(fresh)
        assert fresh['rows'].tolist() == list(range(2, 9))
        assert np.isnan(fresh['specific'][4]).all()  # Coordinate y=6.
        assert np.isfinite(fresh['specific'][3]).all()
        assert np.isfinite(fresh['specific'][5]).all()
        return dict(eligible_components=0, topology=[], original_y_search_unchanged=True)

    monkeypatch.setattr(material_cross_edge, 'continuous_candidates', observe)
    result = material_cross_edge.extended_candidates(a, hint, {}, trace, arrays,
        proposal['original_continuous_source_candidates'])
    assert seen and result['extended_search'] == [2, 8]
    assert not result['unique_existing_U_closure_supported']
