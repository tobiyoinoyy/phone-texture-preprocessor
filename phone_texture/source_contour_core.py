"""Source-only contour core copied from frozen v3; no driver, SKU or path state."""
from collections import deque

def topology(a, hint, proof, edge):
    """Measure U contour, actual side continuation, and opaque source bounds.

The 0.7 gradient, 75% per-side/per-cross support and 2.5 px search/residual
limits are inherited independently for every part. Analysis sigma .8 defines
derivatives only, not any physical paper/metal transition thickness.
"""
    import cv2
    import numpy as np
    from scipy import ndimage
    h, w = hint.shape
    threshold, minimum, tolerance = .7, .75, 2.5
    sigma = .8
    support = int(np.ceil(4*sigma))
    gray = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY).astype(float)
    # Same directional source derivatives used by the preceding measurement.
    gx = np.gradient(ndimage.gaussian_filter1d(gray, sigma, axis=1), axis=1)
    gy = np.gradient(ndimage.gaussian_filter1d(gray, sigma, axis=0), axis=0)
    down = -gy
    yy, xx = np.mgrid[:h, :w]
    left, right = [np.asarray(v, float) for v in proof['source_strip_edges']]
    lx, rx = np.polyval(left, yy+.5), np.polyval(right, yy+.5)
    line = np.asarray(edge['source_end_line'], float)
    cross_y = np.polyval(line, xx)
    opaque = a[..., 3] >= 240
    peak = edge['source_end_peak']
    corners = {}
    for name, fit in [('left', left), ('right', right)]:
        y = float(peak)
        for _ in range(4):
            x = float(np.polyval(fit, y+.5))
            y = float(np.polyval(line, x))
        corners[name] = [x, y]
    top = max(0, int(np.floor(min(v[1] for v in corners.values())))-3*support)
    # Source-specific transverse derivative: compare the outside gradients at
    # each geometry row, not RGB values or a remote photographic brightness.
    background = np.full((h, w), np.nan)
    lateral_reference = {'left': {}, 'right': {}}
    exterior = []
    for y in range(top, h):
        l = int(round(np.polyval(left, y+.5)))
        r = int(round(np.polyval(right, y+.5)))
        span = r-l+1
        gap = max(3, int(span*.04)); count = max(4, int(span*.09))
        xl = np.arange(l-gap-count, l-gap)
        xr = np.arange(r+gap+1, r+gap+count+1)
        valid = xl.min() >= 0 and xr.max() < w
        valid = bool(valid and opaque[y, np.r_[xl, xr]].all())
        exterior.append(dict(row=y, all_reference_alpha_at_least_240=valid))
        if valid:
            vl, vr = float(np.median(down[y, xl])), float(np.median(down[y, xr]))
            background[y] = vl+(np.arange(w)-l)/max(1, r-l)*(vr-vl)
            for name, columns, orientation in [('left', xl, 1), ('right', xr, -1)]:
                values = orientation*gx[y, columns]
                median = float(np.median(values))
                lateral_reference[name][y] = dict(median=median,
                    noise=max(threshold, float(np.quantile(abs(values-median), .99))))
    specific = down-background
    cross_band = ((abs(yy-cross_y) <= tolerance) & (xx >= lx-tolerance)
                  & (xx <= rx+tolerance) & opaque & (down > threshold)
                  & (specific > threshold))
    side_bands = {}
    upright_bands = {}
    side_reports = {}
    actual_tracks = {}
    continued = {}
    for name, fit, predicted, signed in [('left', left, lx, gx), ('right', right, rx, -gx)]:
        cx, cy = corners[name]
        search = (abs(xx-predicted) <= tolerance) & (yy >= top) & opaque
        positive = search & (signed > threshold)
        # An upright edge must retain a mostly lateral normal. A real corner
        # rotates toward the transverse normal; its blurred response is not
        # an upright paper side continuing indefinitely below the corner.
        # The camera's strong along-frame lighting does not erase a lateral
        # paper edge. Require actual signed gx peaks and geometric continuity;
        # raw gy dominance is not an edge-existence criterion.
        upright = positive
        reverse_upright = search & (-signed > threshold)
        upright_bands[name] = upright
        before_rows = list(range(top, min(h, int(np.floor(cy))-support+1)))
        track = []
        for y in before_rows:
            xs = np.flatnonzero(upright[y])
            if len(xs):
                x = int(xs[np.argmax(signed[y, xs])])
                track.append(dict(x=x, y=y, signed_gx=float(signed[y, x]),
                                  gy=float(gy[y, x]), alpha=int(a[y, x, 3]),
                                  residual_px=float(abs(x-predicted[y, x]))))
        fraction = len(track)/max(1, len(before_rows))
        side_reports[name] = dict(expected_rows=before_rows, source_supported_fraction=fraction,
                                  required_support=minimum, supported=fraction >= minimum,
                                  tracked_source_pixels=track)
        actual_tracks[name] = track
        side_bands[name] = positive & (yy <= cy+tolerance)
        labels, count = ndimage.label(upright, np.ones((3, 3), bool))
        ids = {int(labels[p['y'], p['x']]) for p in track if labels[p['y'], p['x']]}
        from_top = np.isin(labels, list(ids)) if ids else np.zeros(hint.shape, bool)
        py, px = np.nonzero(from_top)
        max_depth = float(np.max(py-np.polyval(line, px))) if len(px) else None
        # Inspect every remaining opaque row for reappearing upright source
        # sides, including disconnected branches. Do not ignore a later band.
        after_rows = []
        for y in range(max(0, int(np.floor(cy))+1), h):
            domain = search[y] & (y > np.polyval(line, np.arange(w))+tolerance)
            if domain.any():
                xs = np.flatnonzero(domain & upright[y])
                ref = lateral_reference[name].get(y)
                lateral_specific = signed[y]-ref['median'] if ref else None
                supported_forward = np.flatnonzero(domain & (signed[y] > threshold) & (lateral_specific > ref['noise'])) if ref else []
                supported_reverse = np.flatnonzero(domain & (-signed[y] > threshold) & (-lateral_specific > ref['noise'])) if ref else []
                after_rows.append(dict(row=y, opaque_search_pixels=int(domain.sum()),
                                       upright_pixels=xs.tolist(),
                                       reverse_upright_pixels=np.flatnonzero(domain & reverse_upright[y]).tolist(),
                                       source_lateral_gradient_reference=ref,
                                       source_specific_forward_pixels=list(map(int, supported_forward)),
                                       source_specific_reverse_pixels=list(map(int, supported_reverse))))
        continued[name] = dict(connected_from_actual_upper_track=bool(ids),
                              maximum_signed_distance_below_cross_px=max_depth,
                              endpoint_within_existing_residual_limit=(max_depth is not None and max_depth <= tolerance),
                              actual_source_upright_paths=[dict(x=int(x), y=int(y), signed_gx=float(signed[y, x]),
                                                              gy=float(gy[y, x]), alpha=int(a[y, x, 3])) for y, x in zip(py, px)],
                              after_rows=after_rows)

    # Join only source gradient pixels: no dilation, closing, interpolated
    # bridge, nearest-fill or straight segment is inserted in this graph.
    nodes = cross_band | side_bands['left'] | side_bands['right']
    labels, count = ndimage.label(nodes, np.ones((3, 3), bool))
    anchors = {}
    for name, track in actual_tracks.items():
        anchors[name] = (track[len(track)//2]['y'], track[len(track)//2]['x']) if track else None
    left_anchor, right_anchor = anchors['left'], anchors['right']
    same_component = bool(left_anchor and right_anchor and labels[left_anchor]
                          and labels[left_anchor] == labels[right_anchor])
    route = []
    # Topological connection is distinct from ridge localization. Force the
    # route through actual positive-turn source pixels at BOTH corners, then
    # along transverse-support nodes; a side branch in the same component is
    # not enough. No graph pixel is later used to locate the paper mask edge.
    corner_nodes = {}
    for name, signed in [('left', gx), ('right', -gx)]:
        cx, cy = corners[name]
        corner_nodes[name] = ((abs(xx-cx) <= tolerance) & (abs(yy-cy) <= tolerance)
                              & (signed > threshold) & (down > threshold) & cross_band)
    if same_component:
        initial = (*left_anchor, 0)
        final = (*right_anchor, 2)
        queue = deque([initial]); previous = {initial: None}
        while queue and final not in previous:
            y, x, phase = queue.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y+dy, x+dx
                    if not (0 <= ny < h and 0 <= nx < w):
                        continue
                    if phase == 0:
                        if dy < 0 or not side_bands['left'][ny, nx]: continue
                        next_phase = 1 if corner_nodes['left'][ny, nx] else 0
                    elif phase == 1:
                        if dx < 0 or not cross_band[ny, nx]: continue
                        next_phase = 2 if corner_nodes['right'][ny, nx] else 1
                    else:
                        if dy > 0 or not side_bands['right'][ny, nx]: continue
                        next_phase = 2
                    state = (ny, nx, next_phase)
                    if state in previous: continue
                    previous[state] = (y, x, phase); queue.append(state)
        if final in previous:
            here = final
            while here is not None:
                y, x, phase = here
                route.append(dict(x=x, y=y, gx=float(gx[y, x]), gy=float(gy[y, x]),
                                  down_specific=float(specific[y, x]) if np.isfinite(specific[y, x]) else None,
                                  alpha=int(a[y, x, 3]), topology_phase=phase))
                here = previous[here]
            route.reverse()
    component = labels == labels[left_anchor] if same_component else np.zeros_like(nodes)
    for name, signed in [('left', gx), ('right', -gx)]:
        expected = side_reports[name]['expected_rows']
        connected_track = []
        for y in expected:
            xs = np.flatnonzero(upright_bands[name][y] & component[y])
            if len(xs):
                x = int(xs[np.argmax(signed[y, xs])])
                connected_track.append(dict(x=x, y=y, signed_gx=float(signed[y, x]),
                                            gy=float(gy[y, x]), alpha=int(a[y, x, 3])))
        fraction = len(connected_track)/max(1, len(expected))
        side_reports[name].update(connected_source_supported_fraction=fraction,
            connected_source_supported=fraction >= minimum,
            source_pixels_in_final_U_component=connected_track)
    corner_reports = {}
    for name, signed in [('left', gx), ('right', -gx)]:
        cx, cy = corners[name]
        zone = (abs(xx-cx) <= tolerance) & (abs(yy-cy) <= tolerance)
        # Positive mixed normal connects the measured lateral edge to the
        # downward intensity step; line intersections alone are insufficient.
        turning = corner_nodes[name] & component
        py, px = np.nonzero(turning)
        on_route = [p for p in route if turning[p['y'], p['x']]]
        corner_reports[name] = dict(intersection_search_center=[cx, cy],
            connected_positive_turn=bool(len(px)), route_passes_positive_turn=bool(on_route),
            route_turn_pixels=on_route, source_turn_pixels=[dict(x=int(x), y=int(y),
                signed_gx=float(signed[y, x]), down=float(down[y, x]), alpha=int(a[y, x, 3])) for y, x in zip(py, px)])

    columns = np.arange(int(np.ceil(corners['left'][0]))+3, int(np.floor(corners['right'][0]))-2)
    cross_trace = []
    source_cross_opacity = []
    for x in columns:
        measured_row = int(round(float(np.polyval(line, x))))
        source_cross_opacity.append(bool(0 <= measured_row < h and opaque[measured_row, x]))
        ys = np.flatnonzero(cross_band[:, x] & component[:, x])
        if len(ys):
            y = int(ys[np.argmax(specific[ys, x])])
            cross_trace.append(dict(x=int(x), y=y, down=float(down[y, x]),
                                    specific=float(specific[y, x]), alpha=int(a[y, x, 3]),
                                    residual_px=float(abs(y-np.polyval(line, x)))))
    cross_fraction = len(cross_trace)/max(1, len(columns))
    # A full-width printed stripe commonly has a valid U-shaped leading edge
    # AND paper sides returning below it. Any simultaneous upright side pair
    # below the residual band is a counter-evidence, not accepted metal texture.
    after = {name: {v['row']: v for v in value['after_rows']} for name, value in continued.items()}
    paired_reappearance = [y for y in sorted(set(after['left']) & set(after['right']))
                           if after['left'][y]['source_specific_forward_pixels'] and after['right'][y]['source_specific_forward_pixels']]
    reverse_pairs = [y for y in sorted(set(after['left']) & set(after['right']))
                     if after['left'][y]['reverse_upright_pixels'] and after['right'][y]['reverse_upright_pixels']]
    # Opposite polarity can be a cast shadow OR dark print on continuing paper.
    # Record all such rows. A consecutive branch longer than the existing
    # geometric uncertainty band is not explained by corner localization alone.
    reverse_runs = []
    for y in reverse_pairs:
        if reverse_runs and y == reverse_runs[-1][-1]+1: reverse_runs[-1].append(y)
        else: reverse_runs.append([y])
    def supported_runs(polarity):
        field = 'source_specific_'+polarity+'_pixels'
        runs, current, reachable = [], [], None
        for y in sorted(set(after['left']) & set(after['right'])):
            values = [set(after[name][y][field]) for name in ('left', 'right')]
            connected = None
            if current and y == current[-1]+1:
                connected = [{x for x in xs if any(abs(x-p) <= 1 for p in previous)}
                             for xs, previous in zip(values, reachable)]
            if connected and all(connected):
                current.append(y); reachable = connected
            else:
                if current: runs.append(current)
                current = [y] if all(values) else []; reachable = values if current else None
        if current: runs.append(current)
        return runs
    forward_supported_runs = supported_runs('forward')
    reverse_supported_runs = supported_runs('reverse')
    forward_continuation = any(rows[-1]-rows[0] > tolerance for rows in forward_supported_runs)
    reverse_ambiguous = any(rows[-1]-rows[0] > tolerance for rows in reverse_supported_runs)
    last_observed = max(set(after['left']) & set(after['right']), default=None)
    reverse_censored = bool(reverse_pairs and last_observed == reverse_pairs[-1])
    failures = []
    for name in ('left', 'right'):
        if not side_reports[name]['supported']: failures.append(name+'_actual_side_support_insufficient')
        if not side_reports[name]['connected_source_supported']:
            failures.append(name+'_same_component_side_support_below_75_percent')
        if not corner_reports[name]['connected_positive_turn']: failures.append(name+'_corner_not_connected')
        if not corner_reports[name]['route_passes_positive_turn']:
            failures.append(name+'_actual_route_does_not_pass_positive_turn')
        if not continued[name]['endpoint_within_existing_residual_limit']:
            failures.append(name+'_upright_track_continues_or_unresolved')
    if not route: failures.append('no_actual_source_U_path')
    if cross_fraction < minimum: failures.append('connected_transverse_support_below_75_percent')
    cross_opaque_fraction = sum(source_cross_opacity)/max(1, len(source_cross_opacity))
    if cross_opaque_fraction < minimum: failures.append('source_transverse_edge_not_opaque_foreground')
    if forward_continuation: failures.append('source_specific_continuous_paper_sides_reappear_possible_print_T_junction')
    bright_contour = not failures
    if reverse_ambiguous:
        failures.append('reverse_polarity_sides_shadow_or_continuing_print_ambiguous')
    accepted = not failures
    report = dict(method='source_gradient_connected_paper_contour_diagnostic_v3',
        closed_paper_contour_supported=accepted, material_certified=False, publishable=False,
        visual_approved=False, closed_bright_region_contour_supported=bright_contour,
        source_coordinate_system='immutable_geometry_RGBA_pixel_centres',
        native_raw_topology_verified=False, source_rgb_interpolated_by_this_diagnostic=False,
        thresholds=dict(gradient=threshold, support=minimum, residual_px=tolerance, analysis_sigma=sigma,
                        analysis_kernel_support_used_as_physical_transition=False),
        corners=corner_reports, sides=side_reports, side_termination=continued,
        connected_transverse=dict(support=cross_fraction, expected_columns=columns.tolist(), measured_path=cross_trace,
                                  measured_source_edge_opaque_fraction=cross_opaque_fraction),
        source_U_path=route, source_U_path_length=len(route), source_U_path_is_material_boundary=False,
        source_U_path_usage='Topology connection only; not a maximum-gradient ridge or paper mask cutoff',
        paired_reappearance_rows=paired_reappearance, source_specific_forward_runs=forward_supported_runs,
        source_specific_reverse_runs=reverse_supported_runs, reverse_polarity_pair_rows=reverse_pairs,
        reverse_polarity_pair_runs=reverse_runs, reverse_polarity_pair_observed=bool(reverse_pairs),
        reverse_polarity_continuation_exceeds_residual_band=reverse_ambiguous,
        reverse_polarity_observation_censored_at_visible_end=reverse_censored,
        reverse_observation_class=('supported_continuation' if reverse_ambiguous else
            'isolated_or_short_unconfirmed_observation' if reverse_pairs else 'not_observed'),
        shadow_or_print_ambiguous=reverse_ambiguous,
        isolated_censored_reverse_is_not_proof_of_continuing_print=True,
        outside_gradient_alpha=exterior, failures=failures,
        limitations=['This is an experimental contour predicate, not production material certification',
                     'Native raw connected ridge still requires independent validation',
                     'Outside-paper aluminum and apertures are not certified by this graph',
                     'Photometrically invisible continuations remain ambiguous',
                     'No physical pure/mixed material band is inferred from analysis blur'])
    return report, dict(source_gradient_nodes=nodes, connected_contour_component=component,
                        transverse_supported=cross_band)

def feature_provenance(a, features, hint, proposed, proof, contour):
    """Attribute compact-contour responses using positive source paper evidence.

The input is a peer-avoidance mask, not semantic hardware. Its known 3px
dilation is an analysis operation, not a physical material transition band.
No material is proved by taking the complement of main_surface.
"""
    import numpy as np
    import cv2
    from scipy import ndimage
    h, w = hint.shape
    yy, xx = np.mgrid[:h, :w]
    boxes = [v['box'] for v in proof.get('components', [])]
    if proof.get('seed_box'): boxes.append(proof['seed_box'])
    source_seed = np.zeros_like(hint)
    for x0, y0, x1, y1 in boxes:
        source_seed[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = True
    source_seed &= hint
    before_rows = sorted({y for side in contour['sides'].values() for y in side['expected_rows']})
    # Existing paper with enclosed print plus actual connected source sides;
    # the anchor stops above the transverse analysis band.
    seed_rows = np.flatnonzero(source_seed.any(axis=1))
    anchor = source_seed.copy()
    if len(seed_rows) and before_rows and contour['closed_bright_region_contour_supported']:
        anchor |= hint & (yy >= int(seed_rows.min())) & (yy <= max(before_rows))
    anchor &= a[..., 3] >= 240
    analysis_envelope = ndimage.binary_dilation(proposed, iterations=3)
    left, right = [np.asarray(v, float) for v in proof['source_strip_edges']]
    lateral = ((xx >= np.polyval(left, yy+.5)-3) & (xx <= np.polyval(right, yy+.5)+3))
    ridge = np.zeros_like(hint)
    for p in contour['connected_transverse']['measured_path']: ridge[p['y'], p['x']] = True
    ridge_count = int(ridge.sum())
    # Reproduce the existing analysis exactly, retaining contour identity
    # BEFORE the extra dilation merges neighboring paper and aperture masks.
    gray = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY)
    edges_raw = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), 1.), 30, 60)
    edges_closed = cv2.morphologyEx(edges_raw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    raw_contours, _ = cv2.findContours(edges_raw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    raw_records = [dict(id=i, points=c.reshape(-1, 2).tolist(), bbox=list(map(int, cv2.boundingRect(c))),
                        area=float(cv2.contourArea(c))) for i, c in enumerate(raw_contours)]
    retained = np.zeros_like(hint); associated = np.zeros_like(hint); unknown = np.zeros_like(hint)
    reproduced = np.zeros_like(hint); individual_masks = {}
    records = []
    for index, raw_contour in enumerate(contours):
        x, y, cw, ch = cv2.boundingRect(raw_contour)
        area = cv2.contourArea(raw_contour)
        if not (max(3, h*.07) <= cw < w*.22 and max(3, h*.07) <= ch < h*.65
                and .45 < cw/ch < 6 and area >= max(12, cw*ch*.45)):
            continue
        filled = np.zeros(hint.shape, np.uint8)
        cv2.drawContours(filled, [raw_contour], -1, 1, cv2.FILLED)
        component = filled > 0
        expanded = ndimage.binary_dilation(component, iterations=3)
        reproduced |= expanded
        individual_masks[f'contour-{index:03d}-before-extra-dilation'] = component
        individual_masks[f'contour-{index:03d}-own-extra-dilation'] = expanded
        ys, xs = np.nonzero(component)
        size = int(component.sum())
        support = int((component & anchor).sum())
        edge_support = int((expanded & ridge).sum())
        inside = float((component & proposed).sum()/max(1, size))
        side_inside = float((component & lateral).sum()/max(1, size))
        fraction = edge_support/max(1, ridge_count)
        outside_analysis_envelope = int((expanded & ~analysis_envelope).sum())
        paper_related = bool(contour['closed_paper_contour_supported'] and support >= 20
                             and ridge_count and fraction >= .75 and inside >= .75 and side_inside >= .75)
        if paper_related:
            role = 'source_paper_edge_and_existing_analysis_dilation'
            associated |= expanded
        elif (component & proposed).any():
            role = 'unresolved_paper_overlapping_source_feature'; unknown |= expanded; retained |= expanded
        else:
            role = 'independent_external_source_contour_exclusion'; retained |= expanded
        predecessor_ids = [i for i, c in enumerate(raw_contours)
                           if component[c[:, 0, 1], c[:, 0, 0]].any()]
        records.append(dict(component=index, bbox=[int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1)],
            pixels=size, own_dilation_pixels=int(expanded.sum()), points=raw_contour.reshape(-1, 2).tolist(),
            pre_close_source_contour_ids=predecessor_ids, source_paper_anchor_pixels=support,
            measured_transverse_ridge_pixels=edge_support, measured_ridge_coverage=fraction,
            fraction_inside_original_cross_edge_paper_proposal=inside,
            fraction_inside_source_side_corridor_with_existing_3px_analysis_dilation=side_inside,
            outside_paper_plus_existing_analysis_dilation_pixels=outside_analysis_envelope,
            source_paper_edge_association_supported=paper_related, role=role,
            excluded_from_external_connectivity=not paper_related))
    if features is not None and not np.array_equal(reproduced, features):
        raise ValueError('Reproduced per-contour analysis differs from frozen source feature mask')
    return dict(method='source_Canny_close_contour_individual_dilation_provenance_v3',
        substrate_prior='Same aluminum-bottom capture prior as the existing bottom material classifier',
        source_seed_boxes=boxes, source_seed_pixels=int(source_seed.sum()),
        source_paper_anchor_pixels=int(anchor.sum()), required_paper_anchor_pixels=20,
        minimum_ridge_and_region_support=.75, original_feature_mask_has_semantic_hole_labels=False,
        paper_anchor_basis='Existing source paper classification + its connected actual side tracks, alpha>=240; not a new white-pixel classifier',
        component_attribution_is_diagnostic_and_conditional=True,
        pre_dilation_contour_identity_not_reconstructed=False,
        every_original_analysis_pixel_reproduced=True,
        frozen_peer_mask_compared=features is not None, source_operators_recomputed=True,
        raw_Canny_contours=raw_records,
        operators=dict(source_blur_sigma=1., Canny=[30, 60], closing_kernel=[3, 3], extra_dilation_iterations=3),
        analysis_operator_footprints_are_not_physical_paper_expansion=True,
        retained_contour_union_cannot_be_erased_by_overlapping_paper_contour=True,
        main_surface_complement_used=False, components=records,
        paper_associated_pixels=int(associated.sum()),
        retained_independent_or_unresolved_feature_pixels=int(retained.sum()),
        unknown_feature_pixels=int(unknown.sum())), dict(individual_masks,
            source_Canny_edges=edges_raw > 0, source_after_closing_edges=edges_closed > 0,
            source_paper_anchor=anchor, source_transverse_ridge=ridge,
            source_paper_associated_feature_components=associated,
            retained_source_feature_exclusions=retained,
            unresolved_paper_overlapping_features=unknown)
