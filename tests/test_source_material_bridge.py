"""Small controls; run only through the shared root-scheduled guard."""
import importlib.util
from pathlib import Path
import sys
import pytest


def module():
    from phone_texture import source_material_bridge
    return source_material_bridge


def rail_fixture():
    feature = dict(component=99, bbox=[30, 50, 55, 61], source_paper_edge_association_supported=False)
    raw = [dict(id=71, points=[[5, 59], [90, 59]], bbox=[5, 59, 86, 1], area=0.)]
    return feature, raw


def test_long_source_segment_with_both_endpoints_outside_is_recognized():
    m = module(); feature, raw = rail_fixture()
    value = m.source_feature_role(feature, raw)
    assert value['role'] == 'source_frame_rail_or_shadow_response'
    import numpy as np
    np.testing.assert_allclose(value['source_rail_segments'][0]['intersection'],
                               [[30., 59.], [54., 59.]], rtol=0, atol=1e-12)
    assert value['peer_mask_is_semantic_material_gate'] is False


def test_independent_closed_aperture_wins_over_crossing_long_rail():
    feature, raw = rail_fixture()
    raw.append(dict(id=900, points=[[35, 52], [43, 52], [43, 58], [35, 58]], bbox=[35, 52, 9, 7], area=48.))
    value = module().source_feature_role(feature, raw)
    assert value['role'] == 'independent_closed_source_feature'
    assert value['independent_closed_source_ids'] == [900]


def test_morphology_only_compact_response_is_not_semantic_hardware():
    feature, _ = rail_fixture()
    raw = [dict(id=8, points=[[34, 53], [45, 54]], bbox=[34, 53, 12, 2], area=0.)]
    value = module().source_feature_role(feature, raw)
    assert value['role'] == 'peer_only_unclassified_edge_response'
    assert value['peer_mask_is_semantic_material_gate'] is False


def current_scene(tmp_path):
    import numpy as np
    from PIL import Image
    a = np.full((96, 160, 4), 100, dtype=np.uint8); a[..., 3] = 255
    a[:70, 50:101, :3] = 200
    field = np.rint(80-45*np.exp(-.5*((np.arange(160)-75)/55)**2)).astype('uint8')
    a[70:, :, :3] = field[None, :, None]
    source = tmp_path/'bottom.png'; Image.fromarray(a).save(source)
    hint = np.zeros(a.shape[:2], bool); hint[:, 50:101] = True
    proof = dict(source_strip_edges=[[0, 49.5], [0, 100.5]], seed_box=[54, 10, 97, 60])
    return source, hint, proof


def test_current_source_closed_paper_can_create_bound_material_evidence(tmp_path):
    m = module(); source, hint, proof = current_scene(tmp_path)
    paper, evidence = m.resolve_current_source_material(source, hint, proof, tmp_path/'fresh')
    view = dict(name='bottom', source=str(source), source_sha256=m.sha(source), protection_policy='none',
        source_label_fit_exclusion_mask=evidence['paper_material_mask'],
        source_label_location_hint_mask=evidence['hint_mask'], source_label_material_evidence=evidence)
    scope = dict(source_paper_material_boundary=evidence, source_sha256=evidence['source_sha256'])
    result = m.validate_source_contour_material(view, evidence, scope)
    assert result['material_domain_verified'] is True
    assert result['lighting_effect_verified'] is False and result['visual_approved'] is False
    assert evidence['normal_coverage'] is None
    assert not evidence['pixel_protection'] and not evidence['paper_lighting_budget_set']
    assert evidence['evidence_generation'] == 'current_source_only_no_SKU_or_external_result'
    with pytest.raises(ValueError, match='Scope and view'):
        m.validate_source_contour_material(view, evidence, dict(source_paper_material_boundary={}))
    source.write_bytes(b'changed source')
    with pytest.raises(ValueError, match='source identity'):
        m.validate_source_contour_material(view, evidence, scope)


def test_printed_band_with_actual_paper_continuation_cannot_get_resolution(tmp_path):
    import numpy as np
    from PIL import Image
    m = module(); source, hint, proof = current_scene(tmp_path)
    a = np.array(Image.open(source).convert('RGBA'))
    a[70:73, 50:101, :3] = 40; a[73:, 50:101, :3] = 200
    Image.fromarray(a).save(source)
    with pytest.raises(ValueError):
        m.resolve_current_source_material(source, hint, proof, tmp_path/'rejected')
