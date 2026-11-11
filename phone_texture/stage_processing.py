"""Independent, source-bound stages; failed transforms never erase good stages."""
import copy
import shutil
import time
import traceback
from pathlib import Path

from PIL import Image

from . import pipeline, workspace as w, structure_review as sr, lighting, frame_seams
from . import auto_material, auto_structure, lighting_review
from .geometry_review import write
from .shadow_lift_profile import apply as apply_profile, measure_effect


def geometry_identity(geometry, files, sku):
    items = {entry['face']: entry for entry in geometry['images']}
    if not geometry.get('candidates_complete') or set(items) != set(pipeline.FACES) or len(geometry['images']) != len(items):
        raise ValueError('Complete six-face geometry checkpoint required')
    for face, path in files.items():
        item = items[face]
        if item.get('sku', sku) != sku or w.sha(path) != item['output_sha256']:
            raise ValueError('Geometry checkpoint source identity changed: '+face)
        with Image.open(path) as image:
            if image.mode != 'RGBA' or image.format != 'PNG':
                raise ValueError('Geometry checkpoint must contain RGBA PNG')
            expected = item.get('geometry', {}).get('output_size')
            if expected is not None and list(image.size) != expected:
                raise ValueError('Geometry checkpoint dimensions changed')
    return items


def resume_geometry(old_case, case, sku, manifest):
    """Copy a complete, hash-verified checkpoint into a new case, never approve it.

    A failed case without a completed report is eligible for fresh segmentation.
    An existing complete checkpoint that fails identity checks is a hard error.
    """
    old_case, case = Path(old_case), Path(case)
    report_path = old_case/'work/geometry-records/report.json'
    if not report_path.exists():
        return None
    geometry = w.read(report_path)
    if not geometry.get('candidates_complete'):
        return None
    old_manifest = w.read(old_case/'work/intake.json')
    if set(old_manifest['skus']) != {sku} or set(manifest['skus']) != {sku}:
        raise ValueError('Geometry resume belongs to another SKU')
    w.check_sources(old_manifest)
    w.check_sources(manifest)
    old_files = {f: old_case/'work/geometry-records/staged'/sku/(f+'.png') for f in pipeline.FACES}
    items = geometry_identity(geometry, old_files, sku)
    for face in pipeline.FACES:
        if items[face].get('sku') != sku:
            raise ValueError('Geometry resume entry belongs to another SKU')
        old = old_manifest['skus'][sku][face]
        new = manifest['skus'][sku][face]
        if old['sha256'] != new['sha256'] or items[face]['source_sha256'] != new['sha256']:
            raise ValueError('Geometry resume does not match current SKU source: '+face)
        if w.sha(items[face]['source']) != old['sha256']:
            raise ValueError('Geometry evidence source changed: '+face)
    records = case/'work/geometry-records'
    folder = records/'staged'/sku
    folder.mkdir(parents=True)
    geometry = copy.deepcopy(geometry)
    for entry in geometry['images']:
        face = entry['face']
        shutil.copy2(old_files[face], folder/(face+'.png'))
        entry['source'] = manifest['skus'][sku][face]['snapshot']
    geometry['resume_provenance'] = dict(
        previous_case=str(old_case.resolve()), report_sha256=w.sha(report_path),
        reused_geometry_only=True, source_hashes_verified=True,
        visual_approval_added=False, previous_geometry_status=geometry.get('geometry_status'))
    files = {f: folder/(f+'.png') for f in pipeline.FACES}
    geometry_identity(geometry, files, sku)
    write(records/'report.json', geometry)
    from .geometry_review import CHECKS
    write(records/'review-template.json', dict(report_sha256=w.sha(records/'report.json'), reviewer='',
        items={sku+'/'+face: {**dict.fromkeys(CHECKS, False), 'notes': '', 'boundary_assertions': {}} for face in pipeline.FACES}))
    return geometry, files


def prepare_source_body_reference(case, sku, files, manifest, metadata):
    """Screen once from immutable back; never use sides to choose a length."""
    from . import source_body_reference as sbr
    from .source_body_reference_applicability import inspect
    if not isinstance(metadata, dict):
        raise ValueError('Explicit immutable back reference metadata required')
    raw, back, report = (Path(metadata[k]).resolve() for k in ('raw_path','back_path','geometry_report_path'))
    expected = manifest['skus'][sku]['back']['sha256']
    original = w.read(report)
    rows = [entry for entry in original.get('images', []) if entry.get('face') == 'back']
    if (not original.get('candidates_complete') or len(rows) != 1 or
            rows[0] != metadata['geometry_item'] or rows[0].get('sku', sku) != sku):
        raise ValueError('Immutable back report row differs from requested source reference')
    item = rows[0]
    rectified = metadata.get('perspective_output_sha256')
    selected_back = rectified if rectified else item['output_sha256']
    if (w.sha(raw) != expected or item['source_sha256'] != expected or
            w.sha(back) != item['output_sha256'] or w.sha(files['back']) != selected_back):
        raise ValueError('Immutable back reference source identity differs')
    bindings = {str(path): w.sha(path) for path in (raw, back, report)}
    folder = Path(case)/'work/source-body-reference'
    folder.mkdir(parents=True, exist_ok=False)
    audit = dict(enabled=True, selected_coordinate_mode='canvas_pixel_centers',
        reference_supported=False, reference_review_required=False, remap_verified=False,
        visual_approved=False, publishable=False, bound_assets_sha256=bindings,
        geometry_report_path=str(report), geometry_report_row_sha256=sbr._canonical(item),
        applicability=inspect(raw, back, item))
    options = dict(coordinate_mode='canvas_pixel_centers')
    warnings = []
    if audit['applicability']['applicable'] is True and rectified:
        # Body evidence is measured in pre-perspective back coordinates; it
        # must not steer antenna targets on the rectified back.
        audit.update(status='kept_original_reference_with_notice', reference_review_required=True,
            source_reference_error='back_perspective_rectified_reference_coordinates_differ')
        warnings.append(dict(face='back', kind='source_body_reference_unavailable',
            code='source_body_reference_unavailable', scope='structure_reference',
            message='背面已做透视纠正，机身外延伸参考坐标不一致，保留原参考待核对'))
    elif audit['applicability']['applicable'] is True:
        try:
            evidence = sbr.build(raw, back, item)
        except ValueError as error:
            audit.update(status='kept_original_reference_with_notice',
                reference_review_required=True, source_reference_error=str(error),
                source_reference_traceback=traceback.format_exc())
            warnings.append(dict(face='back', kind='source_body_reference_unavailable',
                code='source_body_reference_unavailable', scope='structure_reference',
                message='背板存在机身外延伸，新天线参考证据不足，保留原参考待核对', error=str(error)))
        else:
            evidence_path = folder/'evidence.json'
            write(evidence_path, evidence)
            options = dict(coordinate_mode=sbr.MODE, body_reference=evidence)
            audit.update(status='source_body_reference_review_required',
                selected_coordinate_mode=sbr.MODE, reference_supported=True,
                reference_review_required=True, body_y=evidence['body_y'],
                evidence_path=str(evidence_path), evidence_file_sha256=w.sha(evidence_path),
                source_body_reference_sha256=evidence['evidence_sha256'])
            warnings.append(dict(face='back', kind='source_body_reference_review',
                code='source_body_reference_review', scope='structure_reference',
                message='天线参考采用本机源图机身边界，最终位置待核对'))
    else:
        audit['status'] = audit['applicability']['status']
    if any(w.sha(path) != value for path, value in bindings.items()):
        raise ValueError('Source reference input changed during screening/build')
    write(folder/'report.json', audit)
    return options, audit, warnings


def process_stages(case, sku, geometry, files, manifest, model, protection_policy,
                   lighting_profile, progress, stage, *, allow_partial=True, bottom_fallback=None,
                   source_body_reference_review=False, source_body_reference_metadata=None,
                   bottom_quality=None, side_candidate_review=False):
    """Run every independent face, then both-side structure and eligible seams.

    Geometry-derived landmarks are measured before illumination, but failure to
    prove one side does not prevent independent materials or other faces.
    Structure always runs the existing both-side verification, including when
    some inputs are unchanged, trusted geometry instead of corrected lighting.
    """
    from .automatic import measure_masks, no_zone_effects
    from .source_antenna_evidence import create as create_antenna_evidence
    from .source_preservation import source_seam_observations, failure_kind, require_material_boundary

    if type(source_body_reference_review) is not bool:
        raise ValueError('source_body_reference_review must be an explicit bool')
    if type(side_candidate_review) is not bool:
        raise ValueError('side_candidate_review must be an explicit bool')
    if side_candidate_review and (not source_body_reference_review or protection_policy != 'none'
                                 or lighting_profile != 'accepted-shadow-lift-v2'):
        raise ValueError('Side candidate review requires current source reference, none protection and accepted-shadow-lift-v2')
    if bottom_quality is not None and (not isinstance(bottom_quality, dict)
            or set(bottom_quality) != {'source', 'commit'}
            or not all(callable(value) for value in bottom_quality.values())):
        raise ValueError('Bottom quality comparison requires source and commit callbacks')
    case = Path(case)
    items = geometry_identity(geometry, files, sku)
    checks, jobs, analyses, per_face, blocked, proposal = {}, {}, {}, {}, [], {}
    selected = dict(files)
    selected_hashes = {f: w.sha(p) for f, p in selected.items()}
    state = dict(geometry='complete', lighting='failed', structure='blocked', seam='blocked')
    reference_options = dict(coordinate_mode='canvas_pixel_centers')
    reference_audit = None
    item_effect_warnings = []
    quality_bindings = {}
    raw_evidence = {}
    side_review, review_lighting, review_structure = None, {}, {}
    historical_strict_blocks = []

    def verify_sources():
        w.check_sources(manifest)
        geometry_identity(geometry, files, sku)
        if any(w.sha(path) != value for path, value in quality_bindings.items()):
            raise RuntimeError('Immutable bottom comparison source changed')
        if reference_audit:
            bindings = dict(reference_audit['bound_assets_sha256'])
            if reference_audit.get('evidence_path'):
                bindings[reference_audit['evidence_path']] = reference_audit['evidence_file_sha256']
            if any(w.sha(path) != value for path, value in bindings.items()):
                raise ValueError('Immutable source body reference evidence changed')

    def failure(name, face, error):
        verify_sources()
        entry = dict(stage=name, face=face, kind=failure_kind(error) or name,
            error=str(error), traceback=traceback.format_exc())
        if getattr(error, 'antenna_evidence_failures', None):
            entry['antenna_evidence_failures'] = error.antenna_evidence_failures
        if getattr(error, 'hardware_contour_conflict', None):
            entry['hardware_contour_conflict'] = error.hardware_contour_conflict
        if hasattr(error, 'material_boundary_evidence'):
            entry['material_boundary_evidence'] = error.material_boundary_evidence
        if hasattr(error, 'normal_coverage_evidence'):
            entry['normal_coverage_evidence'] = error.normal_coverage_evidence
        blocked.append(entry)
        if not allow_partial:
            raise error
        return entry

    if source_body_reference_review:
        begin = time.perf_counter()
        verify_sources()
        reference_options, reference_audit, item_effect_warnings = prepare_source_body_reference(
            case, sku, files, manifest, source_body_reference_metadata)
        verify_sources()
        stage('source_only_body_reference_screen', begin)

    begin = time.perf_counter()
    for side in ('left', 'right'):
        try:
            evidence = create_antenna_evidence(manifest['skus'][sku][side]['snapshot'], files[side],
                items[side]['geometry'], side, case/'work/geometry-records/raw-antenna-evidence'/(side+'.png'))
            raw_evidence[side] = evidence
            proposal.update(auto_structure.propose(files, **reference_options,
                antenna_evidence={side: evidence}, sides=(side,)))
        except ValueError as error:
            failure('structure_proposal', side, error)
    write(case/'work/geometry-structure-proposal.json', proposal)
    stage('fresh_landmark_measurements', begin)

    begin = time.perf_counter()
    light = case/'work/lighting'/sku
    light.mkdir(parents=True)
    for face in pipeline.FACES:
        progress('lighting', face, 6+len(per_face))
        rejection = None
        quality_first, quality_files, quality_audit = None, None, None
        for attempt in range(2):
            attempt_files = quality_files if quality_files is not None else files
            source = attempt_files[face]
            row = dict(lighting='failed', selected_stage='geometry', source_sha256=w.sha(source))
            per_face[face] = row
            attempt_name = (face+'-port-candidate' if face=='bottom' and bottom_fallback is not None else face) if attempt == 0 else face+'-source-fallback'
            if quality_first is not None:
                attempt_name = 'bottom-quality-original'
            assets = case/'work/material-inputs'/attempt_name
            # A side may use its independently verified proposal. Back requires
            # both sides; no other SKU or guessed endpoint substitutes for them.
            landmarks = proposal if (face in proposal or set(proposal) == {'left', 'right'}) else None
            attempt_phase = 'material'
            attempt_bindings = {}
            try:
                job, analysis, allowed, paper = auto_material.propose_job(source, face, assets,
                    companions=attempt_files, landmarks=landmarks, protection_policy=protection_policy)
                # Keep the unrendered proposal for diagnosing unresolved material.
                # A full-height paper-location hint is not a verified paper domain.
                write(assets/'material-proposal.json', job)
                write(assets/'analysis.json', analysis)
                require_material_boundary(job)
                attempt_phase = 'profile'
                job, proof = apply_profile(job, face, assets, lighting_profile)
                analysis['lighting_profile'] = proof
                attempt_phase = 'scope'
                scope = auto_material.lighting_scope(source, face, job, assets/'effect-source-witnesses')
                write(assets/'lighting-scope.json', scope)
                require_material_boundary(job, scope)
                # Failed outputs remain diagnostic files, separate from selected
                # lighting inputs used by structure.
                output = case/'work/lighting-attempts'/attempt_name
                job.update(output_dir=str(output), records_dir=str(case/'work/lighting-records'/attempt_name))
                path = assets/'job.json'
                write(path, job)
                write(assets/'analysis.json', analysis)
                if quality_first is not None:
                    from .bottom_quality_selection import bind_inputs
                    attempt_bindings = bind_inputs(job, scope, path, assets/'lighting-scope.json')
                attempt_phase = 'render'
                lighting.run('aluminum', path)
                candidate = output/(face+'.png')
                attempt_phase = 'pixel_integrity'
                checked = measure_masks(source, candidate, allowed, paper)
                if protection_policy == 'none':
                    checked['actual_no_zone_effects'] = no_zone_effects(source, candidate, analysis)
                attempt_phase = 'effect'
                effect = measure_effect(lighting_review.measure_perimeter if face in ('front', 'back') else lighting_review.measure_local,
                    scope, source, candidate, allowed, paper, lighting_profile)
                checked['source_relative_effect_checks'] = effect
                attempt_phase = 'evidence'
                lighting_review.evidence_images(scope, source, candidate, assets/'effect-visual-review')
                verify_sources()
                if attempt_bindings:
                    from .bottom_quality_selection import verify_bindings
                    verify_bindings(attempt_bindings)
                checks[face], jobs[face], analyses[face] = checked, str(path), analysis
                selected[face] = candidate
                row.update(lighting='effect_warning' if effect.get('original_effect_check_passed') is False else 'complete',
                    selected_stage='lighting', lighting_sha256=w.sha(candidate), lighting_file=str(candidate))
                if (face == 'bottom' and attempt == 0 and bottom_quality is not None
                        and bottom_fallback is not None
                        and effect.get('original_effect_check_passed') is False):
                    from .bottom_quality_selection import record_attempt
                    quality_first = dict(row=copy.deepcopy(row), checked=checked, job_path=str(path),
                        analysis=analysis, candidate=candidate, assets=assets)
                    quality_audit = dict(version='bottom_quality_selection_v1',
                        comparison_reason='optional_port_candidate_effect_warning',
                        port=record_attempt(source, candidate, assets, checked, scope),
                        selected='pending', visual_approved=False, warning_reclassified_as_error=False)
                    original_source = Path(bottom_quality['source']())
                    quality_bindings[str(original_source)] = w.sha(original_source)
                    quality_bindings[str(source)] = w.sha(source)
                    verify_sources()
                    quality_files = dict(files, bottom=original_source)
                    continue
                if quality_first is not None:
                    from .bottom_quality_selection import record_attempt
                    quality_audit['original'] = record_attempt(source, candidate, assets, checked, scope)
            except ValueError as error:
                # The optional remap may invalidate material/lighting evidence. Only
                # that bottom attempt may retry the verified original; source/I/O
                # failures still fail closed and other faces are never recomputed.
                verify_sources()
                if quality_first is not None:
                    from .bottom_quality_selection import verify_bindings, local_rejection_code
                    verify_bindings(attempt_bindings)
                    # A local trial rejection keeps the measured first candidate.
                    # Pixel integrity, input identity, I/O and runtime errors must
                    # not become a quiet quality-selection decision.
                    rejection_code = local_rejection_code(error, attempt_phase)
                    if rejection_code is None:
                        raise
                    quality_audit['original'] = dict(status='locally_rejected',
                        source=str(source), source_sha256=w.sha(source), assets=str(assets),
                        original_effect_check_passed=False, phase=attempt_phase,
                        local_rejection_code=rejection_code,
                        classifier='bottom_quality_selection.local_rejection_code.v1',
                        error=str(error), error_type=type(error).__name__, traceback=traceback.format_exc())
                    for evidence_name in ('material_boundary_evidence', 'normal_coverage_evidence'):
                        if hasattr(error, evidence_name):
                            quality_audit['original'][evidence_name] = getattr(error, evidence_name)
                    diagnostic_files = [assets/'job.json', assets/'lighting-scope.json',
                        case/'work/lighting-attempts'/attempt_name/'bottom.png']
                    quality_audit['original']['diagnostic_sha256'] = {
                        str(p): w.sha(p) for p in diagnostic_files if p.is_file()}
                    assets.mkdir(parents=True, exist_ok=True)
                    write(assets/'quality-trial-failure.json', quality_audit['original'])
                    break
                if face == 'bottom' and attempt == 0 and bottom_fallback is not None:
                    rejection = dict(error=str(error),kind=failure_kind(error) or 'lighting',
                        candidate_source_sha256=w.sha(source),traceback=traceback.format_exc())
                    for evidence_name in ('antenna_evidence_failures','hardware_contour_conflict',
                            'material_boundary_evidence','normal_coverage_evidence'):
                        if hasattr(error,evidence_name):
                            rejection[evidence_name]=getattr(error,evidence_name)
                    restored = Path(bottom_fallback(error))
                    verify_sources()
                    if restored != Path(files[face]):
                        raise RuntimeError('Bottom fallback did not update effective geometry sources')
                    selected[face] = restored
                    selected_hashes[face] = w.sha(restored)
                    continue
                material_failure = failure_kind(error) == 'material_classification'
                failure('material' if material_failure else 'lighting', face, error)
                row['lighting_error'] = str(error)
                if material_failure:
                    row['material_status'] = 'unresolved'
                    if hasattr(error, 'material_boundary_evidence'):
                        row['material_boundary_evidence'] = error.material_boundary_evidence
            break
        if quality_first is not None:
            from .bottom_quality_selection import verify_attempt
            verify_sources()
            verify_attempt(quality_audit['port'])
            original_attempt = quality_audit['original']
            if original_attempt['status'] == 'measured':
                verify_attempt(original_attempt)
            prefer_original = original_attempt['original_effect_check_passed'] is True
            quality_audit.update(selected='original' if prefer_original else 'port',
                reason='original_all_effect_checks_passed' if prefer_original else 'original_not_all_effect_checks_passed',
                other_faces_reprocessed=False)
            audit_path = case/'work/bottom-quality-selection/report.json'
            audit_path.parent.mkdir(parents=True, exist_ok=False)
            write(audit_path, quality_audit)
            if prefer_original:
                restored = Path(bottom_quality['commit'](audit_path))
                verify_sources()
                if restored != Path(files[face]) or w.sha(restored) != original_attempt['source_sha256']:
                    raise RuntimeError('Bottom quality commit did not bind the selected original source')
            else:
                row = quality_first['row']
                per_face[face] = row
                checks[face], jobs[face], analyses[face] = (
                    quality_first['checked'], quality_first['job_path'], quality_first['analysis'])
                selected[face], assets = quality_first['candidate'], quality_first['assets']
            row['optional_bottom_quality_comparison'] = dict(report=str(audit_path),
                report_sha256=w.sha(audit_path), selected=quality_audit['selected'],
                original_effect_check_passed=prefer_original)
        if rejection is not None:
            row['optional_bottom_candidate_rejection'] = rejection
        if face in checks and assets.name != face:
            # Seam processing consumes canonical material paths. Copy only the
            # final accepted source's complete assets; keep both attempt trees
            # and all JSON references valid for audit.
            canonical=case/'work/material-inputs'/face
            shutil.copytree(assets,canonical)
            hashes={}
            for original_asset in assets.rglob('*'):
                if original_asset.is_file():
                    relative=original_asset.relative_to(assets)
                    if w.sha(original_asset)!=w.sha(canonical/relative):
                        raise RuntimeError('Final material alias changed during copy')
                    hashes[str(relative)]=w.sha(original_asset)
            write(canonical/'material-selection.json',dict(source_sha256=w.sha(files[face]),
                selected_assets=str(assets),selected_job=jobs[face],copied_assets_sha256=hashes))
        shutil.copy2(selected[face], light/(face+'.png'))
        if w.sha(light/(face+'.png')) != w.sha(selected[face]):
            raise ValueError('Selected lighting stage changed during copy')
        selected[face] = light/(face+'.png')
        selected_hashes[face] = w.sha(selected[face])
    write(case/'work/lighting-checks.json', checks)
    write(case/'work/lighting-jobs.json', {sku: jobs})
    write(case/'work/lighting-stage.json', dict(faces=per_face, blocks=blocked))
    state['lighting'] = 'complete' if len(checks) == len(pipeline.FACES) else 'partial' if checks else 'failed'
    stage('independent_material_lighting_and_checks', begin)

    structural, seam_report = None, None
    if set(proposal) == {'left', 'right'}:
        begin = time.perf_counter()
        progress('alignment', completed=12)
        try:
            if any(w.sha(p) != selected_hashes[f] for f, p in selected.items()):
                raise RuntimeError('Selected lighting source changed before structure')
            spec = {sku: auto_structure.bind_lighting_sources(proposal, files, selected)}
            spec_path = case/'work/structure-spec.json'
            write(spec_path, spec)
            structural = sr.prepare(case/'work/lighting', spec_path, case/'work/structure-records')
            verify_sources()
            selected = {f: case/'work/structure-records/candidate'/sku/(f+'.png') for f in pipeline.FACES}
            selected_hashes = structural['outputs'][sku]
            state['structure'] = 'complete'
            for side in ('left', 'right'):
                per_face[side].update(selected_stage='structure', structure_sha256=w.sha(selected[side]))
                weak = proposal[side].get('source_boundary_detection', {}).get('endpoint_shape_weak') or []
                if weak:
                    item_effect_warnings.append(dict(face=side, kind='antenna_endpoint_shape_weak',
                        code='antenna_endpoint_shape_weak', bands=weak,
                        message='天线端部条带信号偏弱，已按测得曲线继续处理，请核对天线两端'))
        except ValueError as error:
            failure('structure', None, error)
            state['structure'] = 'failed'
        stage('fresh_landmarks_structure_and_checks', begin)

    if side_candidate_review and state['structure'] != 'complete':
        from .side_candidate_pipeline import continue_sides
        begin = time.perf_counter()
        verify_sources()
        side_review, review_lighting, review_structure, review_failures = continue_sides(
            case, sku, geometry, files, raw_evidence, reference_audit, selected,
            selected_hashes, per_face, checks, progress, verify_sources)
        continued = set(review_structure)
        relit = set(review_lighting)
        historical_strict_blocks = [copy.deepcopy(entry) for entry in blocked
            if (entry.get('face') in relit and entry['stage'] in ('lighting', 'material'))
            or (entry.get('face') in continued and entry['stage'] == 'structure_proposal')
            or (entry.get('face') is None and entry['stage'] == 'structure' and len(continued) == 2)]
        blocked[:] = [entry for entry in blocked if entry not in historical_strict_blocks]
        if continued:
            state['structure'] = 'review_required'
            state['seam'] = 'blocked'
            for face in sorted(continued):
                item_effect_warnings.append(dict(face=face, kind='current_side_candidate_review',
                    code='current_side_candidate_review', scope='lighting_and_structure',
                    message='本次作业已生成调光及天线候选，结构、材质与视觉效果仍待核对'))
        if any(item.get('lighting', {}).get('via') == 'new_review_render'
               and item.get('selected_lighting') for item in side_review['faces'].values()):
            state['lighting'] = 'review_required'
        blocked.extend(review_failures)
        verify_sources()
        stage('current_side_candidate_generation_and_checks', begin)

    if state['structure'] == 'complete' and all(face in checks for face in pipeline.FRAMES):
        begin = time.perf_counter()
        progress('checking', completed=13)
        try:
            if any(w.sha(p) != selected_hashes[f] for f, p in selected.items()):
                raise RuntimeError('Verified structural source changed before seam processing')
            seam_report = frame_seams.run(case/'work', sku, model,
                case/'work/seam-candidates'/sku, case/'work/seam-records')
            verify_sources()
            selected = {f: case/'work/seam-candidates'/sku/(f+'.png') for f in pipeline.FACES}
            selected_hashes = seam_report['output_sha256']
            state['seam'] = 'complete'
            resolution = seam_report.get('continuity_resolution') or {}
            if resolution.get('method') == 'applied_with_steep_step_warning':
                item_effect_warnings.append(dict(face=resolution.get('face'), kind='seam_correction_step_warning',
                    code='seam_correction_step_warning',
                    message='四角接缝校色在圆角端过渡偏陡（%.1f，常规上限 %.1f），已照常应用' % (
                        resolution['max_step_deltaE76'], resolution['limit'])))
            for face in pipeline.FRAMES:
                per_face[face].update(selected_stage='seam', seam_sha256=w.sha(selected[face]))
        except ValueError as error:
            failure('seam', None, error)
            state['seam'] = 'failed'
        stage('frame_seams_and_checks', begin)

    # This is evidence only. A worse unregistered UV comparison is never a
    # reason to delete successful lighting or a verified structural stage.
    if seam_report is None:
        try:
            seam_report = source_seam_observations(selected, files, model)
            seam_report.update(alignment_applied=state['structure'] == 'complete' or bool(review_structure),
                strict_alignment_certified=state['structure'] == 'complete',
                status='measured_only_not_uv_accepted')
        except ValueError as error:
            failure('seam_observation', None, error)
            seam_report = dict(status='observation_unavailable', error=str(error))

    verify_sources()
    from .auto_geometry import trim_visible_canvas
    final_canvas_checks = {}; back_canvas_proof = None
    final_staged = case/'work/selected-candidate'/sku
    final_staged.mkdir(parents=True)
    for face, source in selected.items():
        if w.sha(source) != selected_hashes[face]:
            raise RuntimeError('Verified selected stage changed before final output: '+face)
        with Image.open(source) as image:
            if image.format != 'PNG' or image.mode != 'RGBA':
                raise ValueError('Final stage is not RGBA PNG')
            paper = None if protection_policy == 'none' else auto_material.paper_mask(image, face)[0]
            _, trim = trim_visible_canvas(image, paper)
            if any(trim['removed_margins_ltrb']):
                raise ValueError('Final canvas contains excess soft-matte padding: '+face)
            final_canvas_checks[face] = trim
        dest = final_staged/(face+'.png')
        if face == 'back':
            from .back_canvas import standardize
            back_canvas_proof = standardize(source, dest, model)
            back_canvas_proof['source_sha256'] = w.sha(source)
            lens = back_canvas_proof.get('lens_registration') or {}
            if not lens.get('applied') and lens.get('reason') in ('lenses_not_measured', 'shift_exceeds_limit',
                                                                   'registration_not_verified'):
                item_effect_warnings.append(dict(face='back', kind='lens_registration_not_applied',
                    code='lens_registration_not_applied',
                    message='镜头位置未自动校准（%s），保留标准尺寸背面' % {
                        'lenses_not_measured': '未识别到三颗镜头',
                        'shift_exceeds_limit': '偏差 %.1fpx 超过 8px 上限' % lens.get('max_shift_px', 0),
                        'registration_not_verified': '校准后复测未通过'}[lens['reason']]))
        else:
            shutil.copy2(source, dest)
            if w.sha(dest) != w.sha(source):
                raise ValueError('Final selected stage changed during copy')
        per_face[face]['selected_sha256'] = w.sha(dest)
        per_face[face]['selected_stage_file'] = str(source)
    candidate = case/'candidate'
    pipeline.publish(final_staged.parent, candidate)
    final = pipeline.batch_files(candidate)[sku]
    verify_sources()
    warnings = [dict(face=f, **c['source_relative_effect_checks']) for f, c in checks.items()
        if c['source_relative_effect_checks'].get('original_effect_check_passed') is False]
    all_stages = all(value == 'complete' for value in state.values())
    processed = all_stages and not warnings and not item_effect_warnings
    status = 'processed_needs_visual_review' if processed else 'full_stages_effect_warning_needs_review' if all_stages else 'source_preserved_needs_review'
    review = [dict(face=b.get('face') or 'all', code=b['stage']+'_blocked', message=b['error']) for b in blocked]
    review.extend(item_effect_warnings)
    for face, checked in checks.items():
        if checked.get('actual_no_zone_effects', {}).get('review_warnings'):
            review.append(dict(face=face, code='identifier_readability_unconfirmed', message='标识可读性待核对'))
    result = dict(status=status, six_faces_generated=True, six_faces_processed=processed,
        mandatory_program_checks_complete=processed, quality_checks_complete=False,
        stage_status=state, lighting_complete=state['lighting'] == 'complete',
        structural_alignment_applied=state['structure'] == 'complete' or bool(review_structure), seam_accepted=state['seam'] == 'complete',
        model_uv_verified=False, visual_review_passed=False, per_face_stages=per_face,
        face_processing={f: row['selected_stage'] for f, row in per_face.items()},
        blocked_checks=blocked, review_warnings=review, lighting_effect_warnings=warnings,
        item_effect_warnings=item_effect_warnings,
        candidate=str(candidate), lighting_checks=checks, structure_checks=structural['checks'] if structural else {},
        frame_uv_seam_checks=seam_report, source_hashes_unchanged=True,
        final_canvas_checks=final_canvas_checks, back_standard_canvas=back_canvas_proof,
        final_sha256={f: w.sha(p) for f, p in final.items()})
    if side_review is not None:
        result.update(current_side_review=side_review,
            side_candidate_review_complete=side_review['complete'] and all(
                per_face[face]['lighting'] in ('complete', 'effect_warning')
                for face in ('front', 'back', 'top', 'bottom')),
            side_candidate_review_attempted=True,
            side_candidate_generation_mode='fresh_current_sources',
            structural_review_applied_faces=sorted(review_structure),
            strict_structure_verified=False, structural_alignment_certified=False,
            strict_endpoint_evidence_complete=False, material_classification_complete=False,
            visual_approved=False, selected_for_delivery=False,
            seam_executed=False, saved_review_reused=False,
            lighting_review_checks=review_lighting, structure_review_checks=review_structure,
            historical_blocked_checks=historical_strict_blocks,
            source_reference_status_by_face={face: per_face[face].get('source_reference_status')
                                            for face in review_structure},
            reference_review_required=bool(review_structure),
            structure_operation_status='applied_needs_review' if review_structure else 'not_applied')
    if reference_audit is not None:
        reference_audit.update(remap_verified=reference_audit['reference_supported'] and state['structure']=='complete',
            selected_reference_remap_verified=state['structure']=='complete')
        result['source_body_reference'] = reference_audit
        result['reference_review_required'] = reference_audit['reference_review_required'] or bool(review_structure)
        if reference_audit['reference_review_required']:
            for side in ('left','right'):
                per_face[side].update(reference_review_required=True,
                    reference_coordinate_mode=reference_audit['selected_coordinate_mode'])
        write(case/'work/source-body-reference/report.json', reference_audit)
    if blocked:
        result['blocked_check'] = blocked[0]
    write(case/'work/stage-report.json', result)
    pipeline.overview({sku: files}, {sku: final}, case/'overview.png', faces=pipeline.FACES)
    return result
