"""Saved review metadata only: no render, model, image creation or decoding."""
import copy
import json

import pytest

from phone_texture import web_app as web
from phone_texture.candidate_manifest import candidate_manifest_item


def saved_review_row():
    faces = {face: dict(lighting='complete', selected_stage='lighting') for face in web.FACES}
    for face in ('left', 'right'):
        faces[face].update(lighting='review_applied', structure='review_required',
            selected_stage='structure_review', structure_operation_applied=True,
            structure_review_required=True, illumination_assumption_review=True,
            strict_structure_verified=False, strict_endpoint_evidence_complete=False,
            material_classification_complete=False, reference_review_required=True,
            source_reference_status='source_tail_applicability_unknown',
            reference_review=dict(original_status='source_tail_applicability_unknown',
                original_applicable=None, original_unknown_preserved=True,
                mode='existing_canvas_own_corner_assumption_review', reference_certified=False),
            structure_review_policy='row_alpha_mass_center_contour_sampling_review_v3',
            whole_core_rgb_rigid=False, alpha_outline_unchanged=True,
            selected_sha256='a'*64, selected_stage_file='/bound/'+face+'/candidate.png',
            review_evidence=dict(structure_result='/bound/'+face+'/result.json',
                structure_result_sha256='b'*64, historical_same_x_failed=True,
                actual_positive_xy_donors_checked=True, quality_certification=False))
    return dict(sku='sku10', status='source_preserved_needs_review',
        saved_candidate_review_assembly=True, assembly_mode=web.SAVED_REVIEW_ASSEMBLY_MODE,
        review_continuation_saved=True, automatic_retry_eligible=False,
        six_faces_generated=True, six_faces_processed=False,
        quality_checks_complete=False, visual_review_passed=False,
        structural_alignment_applied=True, structural_alignment_certified=False,
        strict_structure_verified=False, strict_endpoint_evidence_complete=False,
        reference_review_required=True, structural_review_applied_faces=['left', 'right'],
        stage_status=dict(geometry='complete', lighting='review_required', structure='review_required', seam='blocked'),
        per_face_stages=faces, final_sha256={f:'a'*64 for f in web.FACES},
        source_body_reference=dict(status='source_tail_applicability_unknown',
            reference_supported=False, reference_review_required=True),
        historical_blocked_checks=[dict(stage='lighting', face='left', error='original strict endpoint failed')],
        historical_review_warnings=[dict(face='left', code='lighting_blocked', message='original failure')],
        source_baseline_evidence=dict(path='/bound/R6-result.json', sha256='c'*64,
            blocked_checks_sha256='d'*64, review_warnings_sha256='e'*64,
            role='historical_strict_checks_not_reclassified_as_pass'),
        blocked_checks=[dict(stage='structure', kind='strict_structure_certification_pending',
            error='原严格端点证据不足；已应用待核对候选'),
            dict(stage='seam', kind='review_candidate_seam_unverified', error='接缝处理未完成')],
        item_effect_warnings=[dict(face=face, kind='existing_canvas_own_corner_assumption_review',
            message='本机画布参考仍待核对', original_status='source_tail_applicability_unknown',
            reference_certified=False, evidence_path='/bound/'+face+'/result.json',
            evidence_sha256='b'*64) for face in ('left', 'right')])


def test_applied_review_is_shown_without_claiming_strict_or_complete_processing():
    state = web.processing_state(saved_review_row())
    assert state['stage_status']['lighting'] == 'review_required'
    assert state['stage_status']['structure'] == 'review_required'
    assert '六面已应用调光候选，部分待核对' in state['stage_summary']
    assert '天线位置已调整，待核对' in state['stage_summary']
    assert '天线未对齐' not in state['stage_summary']
    assert state['structural_alignment_applied'] is True
    assert state['structure_operation_status'] == 'applied_needs_review'
    assert state['strict_structure_verified'] is False
    assert state['all_stages_complete'] is False
    # review_only still means all program stages finished; saved review has its own caption.
    # Saved review assemblies only suppress automatic retry, not stage obligations.
    assert state['review_only'] is False
    assert state['automatic_retry_eligible'] is False and state['retry_required'] is False


def test_top_level_applied_list_cannot_invent_a_per_face_operation():
    row = saved_review_row()
    row['per_face_stages']['right'].update(structure_operation_applied=False, selected_stage='lighting')
    state = web.processing_state(row)
    assert state['structural_review_applied_faces'] == ['left']
    assert '左侧天线位置已调整，另一侧未调整' in state['stage_summary']


def test_six_files_alone_do_not_mean_processing_or_saved_review_completion():
    row = dict(status='source_preserved_needs_review', six_faces_generated=True,
        six_faces_processed=False, final_sha256={f:'a'*64 for f in web.FACES},
        stage_status=dict(geometry='complete', lighting='failed', structure='blocked', seam='blocked'))
    state = web.processing_state(row)
    assert state['all_stages_complete'] is False and state['structural_alignment_applied'] is False
    assert state['structure_operation_status'] == 'not_applied'
    assert state['retry_required'] is True and state['review_continuation_saved'] is False


@pytest.mark.parametrize('change', [
    {'saved_candidate_review_assembly':False}, {'review_continuation_saved':False},
    {'assembly_mode':'unknown'}, {'six_faces_generated':False}, {'status':'failed'},
    {'final_sha256':{'left':'a'*64}}, {'quality_checks_complete':True},
])
def test_only_closed_saved_review_metadata_disables_default_retry(change):
    row = saved_review_row(); row.update(change)
    state = web.processing_state(row)
    assert state['review_continuation_saved'] is False
    assert state['retry_required'] is True


def test_real_retry_names_skips_saved_review_but_keeps_failed_and_unfinished(tmp_path):
    rows = {'saved':saved_review_row(), 'failed':dict(status='failed'),
            'unfinished':dict(status='source_preserved_needs_review', six_faces_generated=True,
                stage_status=dict(geometry='complete', lighting='partial', structure='blocked', seam='blocked'))}
    for name, row in rows.items():
        path = tmp_path/'cases'/name; path.mkdir(parents=True)
        (path/'result.json').write_text(json.dumps(row))
    app = object.__new__(web.App)
    assert app.retry_names(dict(names=list(rows), work=str(tmp_path))) == ['failed', 'unfinished']


def test_manifest_keeps_bound_operation_evidence_and_unknown_reference_without_mutation():
    row = saved_review_row(); before = copy.deepcopy(row)
    item = candidate_manifest_item(row)
    assert row == before
    assert item['status'] in web.GENERATED_STATUSES
    assert item['generated'] is True and item['all_program_checks_complete'] is False
    assert item['quality_checks_complete'] is False and item['strict_structure_verified'] is False
    assert item['structural_alignment_applied'] is True and item['structural_alignment_certified'] is False
    assert item['structural_review_applied_faces'] == ['left', 'right']
    assert item['final_sha256'] == row['final_sha256']
    assert item['item_effect_warnings'] == row['item_effect_warnings']
    for face in ('left', 'right'):
        actual = item['per_face_stages'][face]; original = row['per_face_stages'][face]
        assert actual['review_evidence'] == original['review_evidence']
        assert actual['selected_sha256'] == original['selected_sha256']
        assert actual['source_reference_status'] == 'source_tail_applicability_unknown'
        assert actual['reference_review'] == original['reference_review']
        assert actual['whole_core_rgb_rigid'] is False
    assert item['source_body_reference_review']['status'] == 'source_tail_applicability_unknown'


def test_ordinary_complete_stage_and_compact_manifest_behavior_remain():
    row = dict(sku='ordinary', status='processed_needs_visual_review', six_faces_processed=True,
        per_face_stages=dict(bottom=dict(lighting='effect_warning', selected_stage='seam', selected_sha256='old')))
    state = web.processing_state(row)
    assert state['all_stages_complete'] is True and state['retry_required'] is False
    assert state['structural_alignment_applied'] is True
    assert state['structure_operation_status'] == 'complete'
    item = candidate_manifest_item(row)
    assert item['per_face_stages']['bottom'] == dict(lighting='effect_warning', selected_stage='seam')


def test_historical_failures_are_retained_in_manifest_not_current_ui_reasons():
    row = saved_review_row()
    item = candidate_manifest_item(row)
    for key in ('historical_blocked_checks', 'historical_review_warnings', 'source_baseline_evidence', 'blocked_checks'):
        assert item[key] == row[key]
    state = web.processing_state(row)
    assert state['review_continuation_saved'] is True
    assert not any('original' in str(reason) for reason in state['review_reasons'])
    assert any('接缝' in str(reason) for reason in state['review_reasons'])


def test_ordinary_manifest_does_not_invent_new_failed_strict_certification():
    row = dict(sku='ordinary', status='processed_needs_visual_review', six_faces_processed=True,
        stage_status=dict(geometry='complete', lighting='complete', structure='complete', seam='complete'),
        structural_alignment_applied=True)
    assert web.processing_state(row)['strict_structure_verified'] is True
    item = candidate_manifest_item(row)
    assert item['structural_alignment_applied'] is True
    assert 'strict_structure_verified' not in item
    assert 'structural_alignment_certified' not in item
