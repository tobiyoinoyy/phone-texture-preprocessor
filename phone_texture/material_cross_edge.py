"""Material-only connected cross-edge proposal; raw geometry is unchanged.

The original interval is retained unless measured censoring and source side
continuation permit exploring existing opaque pixels. Numeric gates stay fixed. A source
line alone never certifies materials: existing U-contour and frame validation
remain mandatory. No RGB, alpha, hint or geometry array is edited.
"""
import hashlib
import traceback


def _require(condition, message):
    if not condition:
        _reject(message)


def _reject(message, evidence=None):
    error = ValueError('Bottom paper material boundary unresolved: '+message)
    error.material_boundary_evidence = evidence or {}
    raise error


def plain(value):
    """Convert diagnostic scalar types and nonfinite observations to JSON."""
    import math
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def trace_source(a, paper, proof):
    """Mirror frozen _source_cross_edge, retaining observations before raises.

    The full-search graph is a separate diagnostic; it cannot override the
    original function's result. All source gradient/coverage gates stay fixed.
    """
    import cv2
    import numpy as np
    from scipy import ndimage
    from phone_texture.auto_material import red_markers

    h, w = paper.shape
    rows = np.flatnonzero(paper.any(axis=1))
    _require(len(rows) >= h * .4, 'Locator has insufficient source rows')
    left = np.array([np.flatnonzero(paper[y])[0] for y in rows])
    right = np.array([np.flatnonzero(paper[y])[-1] for y in rows])
    lf, rf = np.polyfit(rows, left, 1), np.polyfit(rows, right, 1)
    radius = max(6, int(np.ceil(h * .07)))
    components = proof.get('components', [])
    markers = [v for v in red_markers(a)
               if paper[min(h-1, int(v[0])), min(w-1, int(v[1]))]]
    if components:
        lower = max(c['box'][3] for c in components)
        markers = [v for v in markers if v[0] < lower + 1]
        predicted = float(lower - 1)
        censored = lower >= h - max(2, int(h * .10)) - 1
        lo, hi = max(int(h*.45), lower-radius), min(h-2, lower+radius)
        if censored:
            hi = h-2
        anchor = 'source_white_component_censored' if censored else 'source_white_component'
    elif markers:
        cy, cx, bw, bh = max(markers)
        predicted = cy + h*.14
        lo, hi = max(int(cy+bh), int(predicted-radius)), min(h-2, int(predicted+radius))
        anchor = 'source_print_marker_search_only'
    else:
        predicted, lo, hi = None, int(h*.5), h-2
        anchor = 'independent_strip_closure_search'
    _require(hi-lo >= 3, 'No independent source search interval')
    gray = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY).astype(float)
    down = -np.gradient(ndimage.gaussian_filter1d(gray, .8, axis=0), axis=0)
    gx = np.gradient(ndimage.gaussian_filter1d(gray, .8, axis=1), axis=1)
    foreground, background, contrasts, outside, row_observations = [], [], [], [], []
    for y in range(h):
        l, r = int(round(np.polyval(lf, y))), int(round(np.polyval(rf, y)))
        span = r-l+1
        gap, count = max(3, int(span*.04)), max(4, int(span*.09))
        inner = np.r_[np.arange(l+gap, l+gap+count), np.arange(r-gap-count+1, r-gap+1)]
        xl, xr = np.arange(l-gap-count, l-gap), np.arange(r+gap+1, r+gap+count+1)
        _require(xl.min() >= 0 and xr.max() < w, 'Missing lateral source samples')
        coverage = float(np.mean(a[y, np.r_[inner, xl, xr], 3] >= 128))
        if coverage < .8:
            foreground.append(np.nan); background.append(np.nan); contrasts.append(np.nan)
        else:
            foreground.append(float(np.median(down[y, inner])))
            background.append(float(np.median(down[y, np.r_[xl, xr]])))
            contrasts.append(float(np.median(gray[y, inner])-np.median(gray[y, np.r_[xl, xr]])))
        outside.append((float(np.median(down[y, xl])), float(np.median(down[y, xr]))))
        row_observations.append(dict(y=y, left=l, right=r, inner_columns=inner,
            outside_left=xl, outside_right=xr, alpha128_fraction=coverage,
            foreground_down=foreground[-1], background_down=background[-1],
            foreground_minus_background=foreground[-1]-background[-1],
            gray_inside_minus_outside=contrasts[-1]))
    foreground, background, outside = map(np.asarray, (foreground, background, outside))
    signal = foreground-background
    candidates = np.arange(lo, hi+1)
    candidates = candidates[np.isfinite(signal[candidates])]
    _require(len(candidates) > 0, 'No opaque source search samples')
    peak = int(candidates[np.argmax(signal[candidates])])
    l, r = int(round(np.polyval(lf, peak))), int(round(np.polyval(rf, peak)))
    cols = np.arange(l+3, r-2)
    local_rows = np.arange(max(lo, peak-3), min(hi, peak+3)+1)
    all_rows = np.arange(lo, hi+1)
    blend = (cols-l)/max(1, r-l)

    def corrected(ys):
        return down[ys[:, None], cols]-((1-blend)*outside[ys, 0, None]+blend*outside[ys, 1, None])

    local = corrected(local_rows)
    full = corrected(all_rows)
    index = np.argmax(local, axis=0)
    measured = local_rows[index]
    strong = (local[index, np.arange(len(cols))] > .7) & (down[measured, cols] > .7)
    pre_error = None
    if peak in (lo, hi) or foreground[peak] < .7 or signal[peak] < .7:
        pre_error = 'Physical paper end lacks independent strip-specific contrast'
    legacy_error = pre_error
    line, good, residual = None, np.zeros(len(cols), bool), None
    if not legacy_error and strong.mean() < .75:
        legacy_error = 'Paper end is not supported across the strip'
    if not legacy_error:
        line = np.polyfit(cols[strong], measured[strong], 1)
        residual = abs(np.polyval(line, cols)-measured)
        good = strong & (residual <= 2.5)
        if good.mean() < .75 or abs(line[0]) > .15:
            legacy_error = 'Paper end does not support a bounded source cross-edge'
        else:
            line = np.polyfit(cols[good], measured[good], 1)
    column_observations = []
    for i, x in enumerate(cols):
        j = int(np.argmax(full[:, i])); y = int(all_rows[j])
        column_observations.append(dict(x=int(x), fixed_peak_y=int(measured[i]),
            fixed_raw_down=float(down[measured[i], x]), fixed_specific=float(local[index[i], i]),
            fixed_alpha=int(a[measured[i], x, 3]), fixed_strong=bool(strong[i]),
            full_peak_y=y, full_raw_down=float(down[y, x]), full_specific=float(full[j, i]),
            full_alpha=int(a[y, x, 3]), full_peak_outside_fixed_window=bool(y not in local_rows),
            full_search_samples=[dict(y=int(yy), raw_down=float(down[yy, x]),
                specific=float(full[k, i]), alpha=int(a[yy, x, 3])) for k, yy in enumerate(all_rows)]))
    side_tracks = {}
    for name, fit, sign in [('left', proof['source_strip_edges'][0], 1),
                            ('right', proof['source_strip_edges'][1], -1)]:
        records = []
        for y in range(max(0, lo-12), h):
            predicted_x = float(np.polyval(fit, y+.5))
            xs = np.flatnonzero(abs(np.arange(w)-predicted_x) <= 2.5)
            records.append(dict(y=y, predicted_x=predicted_x,
                source_pixels=[dict(x=int(x), alpha=int(a[y, x, 3]), signed_gx=float(sign*gx[y, x]),
                    down=float(down[y, x]), forward=bool(sign*gx[y, x] > .7 and a[y, x, 3] >= 240),
                    reverse=bool(-sign*gx[y, x] > .7 and a[y, x, 3] >= 240)) for x in xs]))
        side_tracks[name] = records
    report = dict(search_anchor=anchor, predicted=predicted, search=[lo, hi], peak=peak,
        fixed_search_rows=local_rows, fitted_locator_sides=[lf, rf], source_proof_sides=proof['source_strip_edges'],
        thresholds=dict(gradient=.7, support=.75, residual_px=2.5, maximum_abs_slope=.15,
                        sigma=.8, physical_transition_width_estimated=False),
        fixed_strong_fraction=float(strong.mean()), fixed_supported_columns=cols[strong],
        fixed_rejected_columns=cols[~strong], exact_mirrored_error=legacy_error,
        fixed_line=line, fixed_residuals=residual, row_observations=row_observations,
        column_observations=column_observations, actual_side_corridor_observations=side_tracks)
    return report, dict(cols=cols, rows=all_rows, specific=full, down=down,
        alpha=a[..., 3], peak=peak, left=l, right=r)


def continuous_candidates(a, hint, proof, arrays):
    """Observe connected source nodes in the original search range, not masks.

    No closing/dilation/gap fill, new threshold, colour edit or wider Y interval.
    A component's ridge must still fit the frozen slope/residual/support limits.
    Run the existing topology predicate on at most three strongest source-only
    proposals; every individual failure is retained and grants no acceptance.
    """
    import numpy as np
    from scipy import ndimage
    from phone_texture import source_contour_core as core
    xs, ys = arrays['cols'], arrays['rows']
    down, specific = arrays['down'][ys[:, None], xs], arrays['specific']
    nodes = (down > .7) & (specific > .7) & (arrays['alpha'][ys[:, None], xs] >= 240)
    labels, count = ndimage.label(nodes, np.ones((3, 3), bool))
    components, usable = [], []
    for ident in range(1, count+1):
        iy, ix = np.nonzero(labels == ident)
        unique = np.unique(ix)
        ridge = []
        for col in unique:
            options = np.flatnonzero(labels[:, col] == ident)
            row = int(options[np.argmax(specific[options, col])])
            ridge.append(dict(x=int(xs[col]), y=int(ys[row]), specific=float(specific[row, col]),
                              raw_down=float(down[row, col]), alpha=int(arrays['alpha'][ys[row], xs[col]])))
        item = dict(component=ident, node_count=len(ix), supported_columns=len(unique),
            expected_columns=len(xs), support=float(len(unique)/max(1, len(xs))),
            source_nodes_yx=[[int(ys[y]), int(xs[x])] for y, x in zip(iy, ix)], ridge=ridge,
            direct_source_connectivity=True, interpolated_or_filled_nodes=0)
        if len(ridge) >= 2:
            xx, yy = np.array([p['x'] for p in ridge]), np.array([p['y'] for p in ridge])
            line = np.polyfit(xx, yy, 1); residual = abs(np.polyval(line, xx)-yy)
            support = float(np.count_nonzero(residual <= 2.5)/max(1, len(xs)))
            eligible = bool(support >= .75 and abs(line[0]) <= .15)
            item.update(line=line, residuals=residual, within_residual_support=support,
                        eligible_for_existing_topology_diagnostic=eligible)
            if eligible:
                usable.append(item)
        components.append(item)
    # Selection is solely geometric support then measured source contrast.
    usable.sort(key=lambda x: (-x['within_residual_support'],
        -sum(p['specific'] for p in x['ridge'])/len(x['ridge']), x['component']))
    topology = []
    for item in usable[:3]:
        center = (arrays['left']+arrays['right'])/2
        edge = dict(source_end_line=item['line'],
            source_end_peak=int(round(float(np.polyval(item['line'], center)))),
            source_end_search=[int(ys[0]), int(ys[-1])],
            source_end_support=item['within_residual_support'],
            diagnostic_candidate_only=True)
        try:
            measured, _ = core.topology(a, hint, proof, edge)
            topology.append(dict(component=item['component'], edge=edge, report=measured))
        except Exception as error:
            topology.append(dict(component=item['component'], edge=edge,
                status='diagnostic_error', error=str(error), traceback=traceback.format_exc()))
    return dict(method='connected_strong_nodes_within_original_source_search_v1',
        original_y_search_unchanged=True, connected_components=components,
        eligible_components=len(usable), topology_attempt_limit=3, topology=topology,
        continuous_sloped_source_proposal_found=bool(usable),
        any_existing_topology_supported=any(x.get('report', {}).get('closed_paper_contour_supported') for x in topology),
        material_certified=False, material_mask_created=False, publishable=False)



def extension_gate(trace, continuous):
    """Require an actual censored component and a connected source side track.

    The locator is only a search corridor. Positive source pixels at three
    consecutive rows (hi-1, hi, hi+1), linked by 8-neighbour adjacency, prove
    the side continues beyond the old search. No fitted side extends pixels.
    This is permission to investigate only, not a material acceptance gate.
    """
    hi = trace['search'][1]
    touching = []
    for item in continuous['connected_components']:
        if (len(item['ridge']) >= 2 and item.get('line') is not None
                and abs(item['line'][0]) <= .15
                and any(p['y'] == hi for p in item['ridge'])):
            touching.append(item['component'])
    tracks = {}
    for side, observations in trace['actual_side_corridor_observations'].items():
        rows = {row['y']: row for row in observations}
        reachable = None
        recorded = []
        for y in (hi-1, hi, hi+1):
            pixels = rows.get(y, {}).get('source_pixels', [])
            xs = {p['x'] for p in pixels if p['forward'] and p['alpha'] >= 240
                  and p['signed_gx'] > .7}
            if reachable is not None:
                xs = {x for x in xs if any(abs(x-old) <= 1 for old in reachable)}
            recorded.append(dict(y=y, connected_source_columns=sorted(xs)))
            reachable = xs
        tracks[side] = dict(source_connected_across_old_search_end=bool(reachable), rows=recorded)
    allowed = bool(touching and any(x['source_connected_across_old_search_end'] for x in tracks.values()))
    return dict(allowed=allowed, original_search_end=hi,
        boundary_touching_source_components=touching, actual_side_tracks=tracks,
        reason='old_search_censored_and_actual_side_continues' if allowed else
               'no_joint_source_censoring_and_side_continuation_evidence',
        material_acceptance=False)


def extended_candidates(a, hint, proof, trace, arrays, original):
    """Search only existing opaque source pixels; keep every old threshold.

    Source columns and original lower Y limit are unchanged. The upper limit
    comes from actual locator/alpha support, never a projected cross-edge or
    marker offset. Rows without opaque lateral references are unusable graph
    nodes. All candidate components are retained and ambiguity is explicit.
    """
    import numpy as np
    gate = extension_gate(trace, original)
    result = dict(extension_gate=gate, extension_performed=False,
        material_certified=False, material_mask_created=False, publishable=False,
        visual_approved=False, source_pixels_extrapolated=False,
        thresholds=dict(gradient=.7, support=.75, residual_px=2.5,
                        maximum_abs_slope=.15, alpha_minimum=240),
        original_search=list(trace['search']), marker_used_as_search_prior_only=True,
        unique_continuous_cross_edge=False, unique_existing_U_closure_supported=False)
    if not gate['allowed']:
        result['status'] = 'not_applicable_original_search_not_source_censored'
        return result
    h, w = hint.shape
    xs = arrays['cols']
    down = arrays['down']
    row_observations = {row['y']: row for row in trace['row_observations']}
    evidence = []
    valid_rows = []
    for y in range(trace['search'][0], h):
        row = row_observations[y]
        xl, xr = np.asarray(row['outside_left'], int), np.asarray(row['outside_right'], int)
        active = np.flatnonzero(hint[y])
        inside = bool(len(active) and 0 <= active[0] <= active[-1] < w)
        locator_opacity = float(np.mean(a[y, active, 3] >= 240)) if inside else 0.
        references_opaque = bool(len(xl) and len(xr)
            and np.all(a[y, np.r_[xl, xr], 3] >= 240))
        valid = bool(inside and locator_opacity >= .75 and references_opaque)
        valid_rows.append(valid)
        evidence.append(dict(y=y, locator_pixels=len(active),
            locator_opaque_fraction=locator_opacity,
            source_lateral_references_opaque=references_opaque,
            source_row_eligible=valid))
    usable = [v['y'] for v in evidence if v['source_row_eligible']]
    new_hi = max(usable, default=trace['search'][1])
    result['opaque_support_extent'] = evidence
    result['last_supported_source_row'] = new_hi
    if new_hi <= trace['search'][1]:
        result['status'] = 'not_applicable_no_opaque_source_below_old_search'
        return result
    ys = np.arange(trace['search'][0], new_hi+1)
    blend = (xs-arrays['left']) / max(1, arrays['right']-arrays['left'])
    specific = np.full((len(ys), len(xs)), np.nan)
    for index, y in enumerate(ys):
        if not valid_rows[index]:
            continue
        row = row_observations[int(y)]
        left = float(np.median(down[y, row['outside_left']]))
        right = float(np.median(down[y, row['outside_right']]))
        specific[index] = down[y, xs] - ((1-blend)*left+blend*right)
    fresh = dict(arrays, rows=ys, specific=specific)
    candidates = continuous_candidates(a, hint, proof, fresh)
    candidates['method'] = 'connected_source_nodes_to_actual_opaque_support_end_v1'
    candidates['original_y_search_unchanged'] = False
    candidates['old_search_lower_bound_unchanged'] = True
    # A single source-supported component is required before even describing a
    # unique closure. Multiple fitted components cannot be selected by output.
    unique = candidates['eligible_components'] == 1
    accepted = [v for v in candidates['topology']
                if v.get('report', {}).get('closed_paper_contour_supported')]
    result.update(status='extended_diagnostic_complete', extension_performed=True,
        extended_search=[int(ys[0]), int(ys[-1])],
        continuous_source_candidates=candidates,
        unique_continuous_cross_edge=unique,
        unique_existing_U_closure_supported=bool(unique and len(accepted) == 1),
        ambiguity_rejected=candidates['eligible_components'] > 1,
        limitations=['Diagnostic only: no material masks, lighting or visual acceptance.',
            'Source geometry pixels are observed directly; no independent native-raw topology claim.',
            'Opaque locator support bounds exploration; the locator itself never certifies paper.',
            'A marker prediction can be censored and is not a physical paper-tail coordinate.'])
    return result



def _closed_candidates(candidates):
    return [x for x in candidates['topology']
            if x.get('report', {}).get('closed_paper_contour_supported')
            and not x['report'].get('failures')]


def select_material_cross_edge(a, hint, proof, *, legacy_error):
    """Use one source-closed line; source censoring can permit investigation.

    Original-window success never expands the search. Missing or ambiguous
    evidence never becomes a material mask by this function alone.
    """
    allowed = ('Paper end is not supported across the strip',
               'Paper end does not support a bounded source cross-edge')
    if str(legacy_error) not in allowed:
        _reject('Legacy cross-edge failure is outside the material fallback scope',
                dict(legacy_error=str(legacy_error)))
    trace, arrays = trace_source(a, hint, proof)
    _require(trace['exact_mirrored_error'] == str(legacy_error),
             'Material cross-edge trace differs from the frozen legacy rejection')
    original = continuous_candidates(a, hint, proof, arrays)
    candidates = original
    extension = dict(status='not_needed', extension_performed=False)
    passed = _closed_candidates(original)
    ambiguous = original['eligible_components'] > 1
    if not passed and not ambiguous:
        extension = extended_candidates(a, hint, proof, trace, arrays, original)
        if extension['extension_performed']:
            candidates = extension['continuous_source_candidates']
            passed = _closed_candidates(candidates)
            ambiguous = candidates['eligible_components'] != 1
    expanded = extension['extension_performed']
    evidence = plain(dict(method='material_continuous_source_cross_edge_v2',
        legacy_error=str(legacy_error), source_rgba_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
        source_locator_array_sha256=hashlib.sha256(hint.tobytes()).hexdigest(),
        source_size=[a.shape[1], a.shape[0]],
        source_array=dict(shape=list(a.shape), dtype=str(a.dtype), byte_order='uint8_not_applicable', storage_order='C'),
        locator_array=dict(shape=list(hint.shape), dtype=str(hint.dtype), byte_order='bool_not_applicable', storage_order='C'),
        legacy_trace=trace, original_continuous_source_candidates=original,
        continuous_source_candidates=candidates, search_extension=extension,
        numeric_thresholds_unchanged=True, original_y_search_unchanged=not expanded,
        raw_geometry_detector_modified=False, source_pixels_interpolated=False,
        source_geometry_modified=False, material_certified=False,
        lighting_effect_verified=False, visual_approved=False, publishable=False))
    if ambiguous or len(passed) != 1:
        evidence['closed_candidate_count'] = len(passed)
        _reject('Continuous material cross-edge has no unique source-closed contour', evidence)
    selected = passed[0]
    edge = dict(selected['edge'], method='material_continuous_source_cross_edge_v2')
    evidence.update(selected_component=selected['component'], selected_edge=plain(edge),
                    closed_candidate_count=1, topology_supported=True)
    return plain(edge), selected['report'], evidence


def validate_proposal_record(proposal, source_sha256, edge, *, rgba=None, hint=None):
    """Validate the explicit search obligation, never a boolean waiver.

    Source arrays, when supplied by the existing material validator, also
    verify the exact decoded-source/locator binding and every opaque limit row.
    """
    from pathlib import Path
    _require(proposal.get('method') == 'material_continuous_source_cross_edge_v2',
             'Unexpected material cross-edge proposal version')
    _require(proposal.get('source_sha256') == source_sha256
             and proposal.get('selected_edge') == edge
             and proposal.get('closed_candidate_count') == 1
             and proposal.get('numeric_thresholds_unchanged') is True,
             'Material cross-edge proposal does not bind this source line')
    _require(proposal.get('implementation_path') == str(Path(__file__).resolve())
             and proposal.get('implementation_sha256') == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'Material cross-edge implementation binding differs')
    trace = proposal['legacy_trace']; original = proposal['original_continuous_source_candidates']
    selected = proposal['continuous_source_candidates']; extension = proposal['search_extension']
    expected_thresholds = dict(gradient=.7, support=.75, residual_px=2.5,
        maximum_abs_slope=.15, sigma=.8, physical_transition_width_estimated=False)
    _require(trace.get('thresholds') == expected_thresholds, 'Material source thresholds differ')
    passed = _closed_candidates(selected)
    _require(len(passed) == 1 and passed[0]['component'] == proposal['selected_component']
             and dict(passed[0]['edge'], method=proposal['method']) == edge,
             'Selected material edge differs from the unique stored contour')
    actual_eligible = [c for c in selected['connected_components']
                       if c.get('eligible_for_existing_topology_diagnostic') is True]
    _require(len(actual_eligible) == selected['eligible_components'] == 1
             and actual_eligible[0]['component'] == proposal['selected_component'],
             'Selected material contour differs from the unique eligible source component')
    component = actual_eligible[0]
    expected_columns = {p['x'] for p in trace['column_observations']}
    ridge = component['ridge']; line = component['line']
    nodes = {tuple(p) for p in component['source_nodes_yx']}
    _require(line == edge['source_end_line'] and len(line) == 2 and abs(line[0]) <= .15
             and component['expected_columns'] == len(expected_columns) and expected_columns
             and len({p['x'] for p in ridge}) == len(ridge),
             'Material edge line or source support denominator differs')
    _require(all(p['x'] in expected_columns and (p['y'], p['x']) in nodes
                 and p['alpha'] >= 240 and p['raw_down'] > .7 and p['specific'] > .7
                 and edge['source_end_search'][0] <= p['y'] <= edge['source_end_search'][1]
                 for p in ridge), 'Material ridge lacks actual strong opaque component nodes')
    support = sum(abs(line[0]*p['x']+line[1]-p['y']) <= 2.5 for p in ridge)/len(expected_columns)
    _require(support >= .75 and support == component['within_residual_support']
             and support == edge['source_end_support'], 'Material ridge residual support differs')
    expanded = extension.get('extension_performed')
    _require(isinstance(expanded, bool) and proposal.get('original_y_search_unchanged') is (not expanded),
             'Material search extension flag is inconsistent')
    if not expanded:
        _require(selected == original and selected.get('original_y_search_unchanged') is True
                 and edge['source_end_search'] == trace['search'],
                 'Unexpanded material search changed its original domain')
        _require(selected['eligible_components'] == 1,
                 'Original material edge is not the unique eligible source candidate')
    else:
        gate = extension_gate(trace, original)
        _require(gate['allowed'] and extension.get('extension_gate') == gate,
                 'Material extension lacks actual censoring and side-track evidence')
        _require(not _closed_candidates(original) and original['eligible_components'] <= 1,
                 'Material search expanded despite an original closed or ambiguous contour')
        _require(extension.get('marker_used_as_search_prior_only') is True
                 and extension.get('source_pixels_extrapolated') is False,
                 'Material extension does not preserve source observation semantics')
        _require(extension.get('thresholds') == dict(gradient=.7, support=.75, residual_px=2.5,
                 maximum_abs_slope=.15, alpha_minimum=240), 'Extended material thresholds differ')
        rows = extension['opaque_support_extent']
        _require([v['y'] for v in rows] == list(range(trace['search'][0], proposal['source_size'][1])),
                 'Material extension omitted opaque coverage rows')
        valid = []
        for row in rows:
            eligible = bool(row['locator_pixels'] > 0 and row['locator_opaque_fraction'] >= .75
                            and row['source_lateral_references_opaque'])
            _require(row['source_row_eligible'] is eligible, 'Material extension opacity claim differs')
            if eligible: valid.append(row['y'])
        last = max(valid, default=trace['search'][1])
        _require(last > trace['search'][1]
                 and extension['last_supported_source_row'] == last
                 and extension['extended_search'] == [trace['search'][0], last]
                 and edge['source_end_search'] == extension['extended_search'],
                 'Material extension exceeds its actual opaque source support')
        _require(selected == extension['continuous_source_candidates']
                 and selected.get('original_y_search_unchanged') is False
                 and selected.get('old_search_lower_bound_unchanged') is True
                 and selected['eligible_components'] == 1
                 and extension['unique_existing_U_closure_supported'] is True,
                 'Extended material edge is not uniquely source-closed')
    if rgba is not None or hint is not None:
        import numpy as np
        _require(rgba is not None and hint is not None, 'Both source and locator arrays are required')
        _require(proposal.get('source_array') == dict(shape=list(rgba.shape), dtype=str(rgba.dtype),
                 byte_order='uint8_not_applicable', storage_order='C')
                 and proposal.get('locator_array') == dict(shape=list(hint.shape), dtype=str(hint.dtype),
                 byte_order='bool_not_applicable', storage_order='C')
                 and [rgba.shape[1], rgba.shape[0]] == proposal['source_size']
                 and hint.shape == rgba.shape[:2]
                 and hashlib.sha256(rgba.tobytes()).hexdigest() == proposal['source_rgba_sha256']
                 and hashlib.sha256(hint.tobytes()).hexdigest() == proposal['source_locator_array_sha256'],
                 'Material proposal source or locator array binding changed')
        if expanded:
            source_rows = {v['y']: v for v in trace['row_observations']}
            for row in extension['opaque_support_extent']:
                y = row['y']; active = np.flatnonzero(hint[y]); old = source_rows[y]
                fraction = float(np.mean(rgba[y, active, 3] >= 240)) if len(active) else 0.
                left, right = old['outside_left'], old['outside_right']
                _require(len(left) > 0 and len(right) > 0
                         and all(isinstance(v, int) and 0 <= v < rgba.shape[1] for v in left+right),
                         'Extended material requires both actual source reference sides')
                refs = np.r_[left, right].astype(int)
                _require(len(refs) > 0 and np.all((refs >= 0) & (refs < rgba.shape[1])),
                         'Extended material reference columns leave source canvas')
                opaque = bool(np.all(rgba[y, refs, 3] >= 240))
                _require(row['locator_pixels'] == len(active)
                         and row['locator_opaque_fraction'] == fraction
                         and row['source_lateral_references_opaque'] is opaque,
                         'Stored material opaque bound differs from current source pixels')
    return dict(source_line_bound=True, search_extension_verified=expanded,
                material_certified=False, lighting_effect_verified=False)
