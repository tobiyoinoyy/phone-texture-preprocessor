"""Review evidence for downloaded six-face candidate manifests.

Keep complete warning/error/path text. Condense only dense point/sample
arrays and tracebacks, with explicit counts and source-evidence hashes.
"""
import hashlib
import json

_DENSE_FIELDS = {
    'coordinates_yx', 'witness_columns', 'peer_columns',
    'target_added_gradient_L', 'peer_added_gradient_L', 'measured_steps_L',
    'minimum_possible_steps_with_candidate_quantization_L',
}


def _sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()


def candidate_manifest_item(row):
    """Retain review reasons and source-reference state without a quality pass."""
    condensed = []

    def summarize(value, key='', location=''):
        if key == 'traceback' or key in _DENSE_FIELDS:
            summary = dict(omitted_from_portable_summary=True,
                item_count=len(value) if hasattr(value, '__len__') else None,
                evidence_sha256=_sha(value))
            if isinstance(value, list) and value and all(
                    isinstance(x, (int, float)) and not isinstance(x, bool) for x in value):
                summary.update(minimum=min(value), maximum=max(value))
            condensed.append(location)
            return summary
        if isinstance(value, dict):
            return {name:summarize(child, name, location+'.'+str(name))
                    for name,child in value.items()}
        if isinstance(value, (list, tuple)):
            return [summarize(item, location=location+'['+str(i)+']')
                    for i,item in enumerate(value)]
        return value

    warnings = row.get('lighting_effect_warnings', [])
    item_warnings = row.get('item_effect_warnings', [])
    per_face = {}
    for face, info in row.get('per_face_stages', {}).items():
        fields = ('lighting', 'structure', 'seam', 'selected_stage', 'material_status')
        # Existing ordinary manifests retain their compact schema. Explicit
        # saved review selections additionally carry their operation/provenance.
        if (info.get('structure_operation_applied') is True
                or info.get('illumination_assumption_review')
                or info.get('structure_review_required') or info.get('review_evidence')):
            fields += ('structure_operation_applied', 'structure_review_required',
                'illumination_assumption_review', 'strict_structure_verified',
                'strict_endpoint_evidence_complete', 'material_classification_complete',
                'reference_review_required', 'source_reference_status', 'reference_review',
                'structure_review_policy', 'whole_core_rgb_rigid', 'alpha_outline_unchanged',
                'source_sha256', 'lighting_sha256', 'structure_sha256',
                'selected_sha256', 'selected_stage_file', 'review_evidence')
        per_face[face] = {key:info[key] for key in fields if key in info}
    selected = dict(sku=row['sku'],
        generated=bool(row.get('six_faces_generated', row.get('six_faces_processed'))),
        all_program_checks_complete=bool(row.get('six_faces_processed')),
        status=row['status'], error=row.get('error'), blocked_check=row.get('blocked_check'),
        review_warnings=row.get('review_warnings', []),
        lighting_effect_warnings=warnings, lighting_effect_warnings_sha256=_sha(warnings),
        item_effect_warnings=item_warnings, item_effect_warnings_sha256=_sha(item_warnings),
        # Stage output holds the audit here; the similarly named input flag is not evidence.
        source_body_reference_review=row.get('source_body_reference', False),
        stage_status=row.get('stage_status', {}),
        quality_checks_complete=row.get('quality_checks_complete', False),
        visual_review_passed=row.get('visual_review_passed', False),
        per_face_stages=per_face)
    # Carry explicit facts without inventing failed certification for older rows.
    for key in ('structural_alignment_applied', 'structural_review_applied_faces',
            'structural_alignment_certified', 'strict_structure_verified',
            'strict_endpoint_evidence_complete', 'reference_review_required',
            'saved_candidate_review_assembly', 'assembly_mode',
            'review_continuation_saved', 'automatic_retry_eligible', 'final_sha256',
            'blocked_checks', 'historical_blocked_checks', 'historical_review_warnings',
            'source_baseline_evidence', 'current_side_review', 'side_candidate_review_complete',
            'side_candidate_review', 'source_hashes_unchanged', 'portable_review_package'):
        if key in row:
            selected[key] = row[key]
    result = summarize(selected)
    result['portable_review_evidence'] = dict(
        all_warning_entries_retained=True, original_effect_evidence_sha256=_sha(warnings),
        original_item_effect_warnings_sha256=_sha(item_warnings),
        condensed_fields=condensed, error_warning_and_path_text_retained=True,
        full_dense_evidence_location='Original local result.json; identify by evidence SHA-256',
        quality_approval_created=False)
    return result
