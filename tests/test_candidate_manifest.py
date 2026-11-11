"""Exercise the real automatic manifest construction with no image processing."""
import copy
import json
from concurrent.futures import Future
from types import SimpleNamespace

from phone_texture.candidate_manifest import candidate_manifest_item


def warning_row():
    return dict(sku='sku12', status='full_stages_effect_warning_needs_review',
        six_faces_generated=True, six_faces_processed=False,
        quality_checks_complete=False, visual_review_passed=False,
        review_warnings=[],
        # Actual stage result schema from prepare_source_body_reference and
        # process_stages: warning, reference audit, and input opt-in are distinct.
        item_effect_warnings=[dict(face='back',kind='source_body_reference_review',
            code='source_body_reference_review',scope='structure_reference',
            message='天线参考采用本机源图机身边界，最终位置待核对')],
        source_body_reference_review_enabled=True,reference_review_required=True,
        source_body_reference=dict(enabled=True,status='source_body_reference_review_required',
            selected_coordinate_mode='source_body_reference_review',reference_supported=True,
            reference_review_required=True,body_y=[12.5,1698.25],remap_verified=True,
            selected_reference_remap_verified=True,visual_approved=False,publishable=False,
            evidence_path='/Users/private/body-reference.json',
            evidence_file_sha256='source-evidence-file-hash',source_body_reference_sha256='source-proof-hash'),
        stage_status=dict(geometry='complete',lighting='complete',structure='complete',seam='complete'),
        per_face_stages=dict(bottom=dict(lighting='effect_warning',selected_stage='seam',
            lighting_file='/Users/private/work/bottom.png',selected_sha256='abc')),
        lighting_effect_warnings=[dict(face='bottom',kind='local_effect_review_required',
            status='effect_review_required',passed=False,original_effect_check_passed=False,
            quality_checks_complete=False,error='Local review required',
            warnings=[dict(kind='normal_continuity_warning',check='bottom_paper_metal_normal',
                original_threshold_passed=False,limit_L=1.25,measured_max_residual_L75=1.396,
                normal_coverage=dict(complete=False,qualified_target_pairs=82,minimum_pairs_per_row=10,
                    missing_rows=[dict(row=140,witness_pairs=82,peer_pairs=0)],
                    physical_transition_support_certified=False)),
                dict(kind='quantization_compatible_pixel_warning',check='hardware_1_length',
                    original_threshold_passed=False,limit_L=1.25,measured_max_step_L=1.29,
                    pair_count=120,over_limit_pairs=2,coordinates_yx=[[1,2],[2,2]],
                    measured_steps_L=[1.28,1.29],
                    minimum_possible_steps_with_candidate_quantization_L=[1.12,1.14],
                    rounding_cause_proven=False,quality_checks_complete=False,
                    new_joint_constraint_metric=17)],
            effect_measurements=dict(check=dict(mask='/Users/private/mask.png',normal_check_passed=False)),
            effect_failures=[dict(kind='local_brightness',measured_L=37.35,min_L=50,max_L=80)])])


def test_real_automatic_manifest_preserves_each_warning_and_face_state(monkeypatch,tmp_path):
    from phone_texture import automatic, frame_seams
    source=tmp_path/'raw';(source/'sku12').mkdir(parents=True)
    row=warning_row()
    class Pool:
        def __init__(self,*args,**kwargs):pass
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def submit(self,*args):
            future=Future();future.set_result(copy.deepcopy(row));return future
    monkeypatch.setattr(frame_seams,'profile',lambda _:({'model_sha256':'test-model'},None))
    monkeypatch.setattr(automatic.w,'face_files',lambda _: {})
    monkeypatch.setattr(automatic,'implementation_manifest',lambda _: {})
    monkeypatch.setattr(automatic.multiprocessing,'get_context',
        lambda _:SimpleNamespace(Semaphore=lambda _:None))
    monkeypatch.setattr(automatic.concurrent.futures,'ProcessPoolExecutor',Pool)
    monkeypatch.setattr(automatic,'export_complete_candidates',lambda *args:['sku12'])
    work=tmp_path/'work'
    report=automatic.run(source,work,tmp_path/'output',jobs=1,cpu_threads=1,
        model='iphone17promax',protection_policy='none',skus=['sku12'])
    assert report['status']=='review_candidates_needs_visual_review'
    item=json.loads((work/'partial-candidate-manifest.json').read_text())['items'][0]
    assert item['all_program_checks_complete'] is False
    assert item['quality_checks_complete'] is False and item['visual_review_passed'] is False
    assert item['per_face_stages']['bottom']==dict(lighting='effect_warning',selected_stage='seam')
    effect=item['lighting_effect_warnings'][0]
    assert effect['original_effect_check_passed'] is False
    assert [v['kind'] for v in effect['warnings']]==['normal_continuity_warning','quantization_compatible_pixel_warning']
    normal,quantization=effect['warnings']
    assert normal['normal_coverage']['missing_rows'][0]['peer_pairs']==0
    assert normal['measured_max_residual_L75']==1.396 and normal['limit_L']==1.25
    assert quantization['rounding_cause_proven'] is False
    assert quantization['over_limit_pairs']==2 and quantization['new_joint_constraint_metric']==17
    assert effect['effect_failures'][0]['measured_L']==37.35
    assert item['item_effect_warnings']==row['item_effect_warnings']
    assert item['source_body_reference_review']==row['source_body_reference']
    assert effect['effect_measurements']['check']['mask']=='/Users/private/mask.png'


def test_dense_evidence_is_explicitly_condensed_and_bound_without_mutating_result():
    row=warning_row();before=copy.deepcopy(row)
    item=candidate_manifest_item(row)
    quantization=item['lighting_effect_warnings'][0]['warnings'][1]
    assert quantization['coordinates_yx']['item_count']==2
    assert len(quantization['coordinates_yx']['evidence_sha256'])==64
    assert quantization['measured_steps_L']['maximum']==1.29
    assert item['portable_review_evidence']['all_warning_entries_retained']
    assert item['portable_review_evidence']['condensed_fields']
    assert len(item['lighting_effect_warnings_sha256'])==64
    assert row==before


def test_complete_error_warning_and_path_text_survives_traceback_condensation():
    row=warning_row()
    row['error']='Cannot open /Users/private/My Project/source.png'
    row['blocked_check']=dict(face='bottom',stage='material',
        error=r'Failed at C:\private\source.png',traceback='File "/Users/private/runtime.py", line 2')
    item=candidate_manifest_item(row)
    assert item['error']==row['error']
    assert item['blocked_check']['error']==row['blocked_check']['error']
    assert item['item_effect_warnings']==row['item_effect_warnings']
    assert item['source_body_reference_review']==row['source_body_reference']
    assert item['blocked_check']['traceback']['omitted_from_portable_summary'] is True
    assert len(item['blocked_check']['traceback']['evidence_sha256'])==64
    assert item['portable_review_evidence']['error_warning_and_path_text_retained'] is True


def test_no_warning_entry_or_decision_is_invented_for_complete_row():
    row=dict(sku='sku1',status='processed_needs_visual_review',six_faces_processed=True)
    item=candidate_manifest_item(row)
    assert item['lighting_effect_warnings']==[] and item['per_face_stages']=={}
    assert item['item_effect_warnings']==[] and item['source_body_reference_review'] is False
    assert item['all_program_checks_complete'] is True
    assert item['quality_checks_complete'] is False and item['visual_review_passed'] is False


def test_reference_summary_comes_from_stage_audit_not_opt_in_or_old_alias():
    row=warning_row()
    row['source_body_reference_review']=False  # Old erroneous lookup must be ignored.
    summary=candidate_manifest_item(row)['source_body_reference_review']
    assert summary==row['source_body_reference']
    assert summary['reference_review_required'] is True
    assert summary['body_y']==[12.5,1698.25]
    assert summary['selected_coordinate_mode']=='source_body_reference_review'
    assert summary['remap_verified'] is True
    assert summary['visual_approved'] is False and summary['publishable'] is False


def test_unavailable_reference_preserves_original_mode_and_distinct_remap_flags():
    row=warning_row()
    row['source_body_reference']=dict(enabled=True,status='kept_original_reference_with_notice',
        selected_coordinate_mode='canvas_pixel_centers',reference_supported=False,
        reference_review_required=True,remap_verified=False,selected_reference_remap_verified=True,
        visual_approved=False,publishable=False,
        source_reference_error='Missing native source witnesses at /Users/private/body-reference.json')
    row['item_effect_warnings']=[dict(face='back',kind='source_body_reference_unavailable',
        code='source_body_reference_unavailable',scope='structure_reference',
        message='背板存在机身外延伸，新天线参考证据不足，保留原参考待核对',
        error=row['source_body_reference']['source_reference_error'])]
    item=candidate_manifest_item(row)
    assert item['source_body_reference_review']==row['source_body_reference']
    assert item['item_effect_warnings']==row['item_effect_warnings']
    assert item['source_body_reference_review']['remap_verified'] is False
    assert item['source_body_reference_review']['selected_reference_remap_verified'] is True
    assert item['all_program_checks_complete'] is False


def test_input_opt_in_without_an_audit_does_not_invent_reference_evidence():
    row=dict(sku='sku1',status='source_preserved_needs_review',six_faces_processed=False,
        source_body_reference_review_enabled=True)
    assert candidate_manifest_item(row)['source_body_reference_review'] is False
