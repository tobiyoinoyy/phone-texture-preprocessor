"""Source-supported side hardware masks; structure locking boxes are not materials."""
import cv2
import numpy as np
from scipy import ndimage


def refine_capsule_boundary(distance, alpha, box, original, tolerance):
    """Fit a one-pixel contour localization error, retaining all source gates.

    A convex hull over granular metal can stand outside a long rail. Search
    only the same already-supported capsule; never bridge a missing cap/rail.
    """
    h,w=alpha.shape;gy,gx=np.indices((h,w));x0,y0,bw0,bh0=box;best=None
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            for dw in (-1,0,1):
                for dh in (-1,0,1):
                    x,y,bw,bh=x0+dx,y0+dy,bw0+dw,bh0+dh;r=(bw-1)/2
                    mask=(gx-x-r)**2+(gy-np.clip(gy,y+r,y+bh-1-r))**2<=r*r
                    iou=np.count_nonzero(mask&original)/np.count_nonzero(mask|original)
                    if iou<.90 or np.any(mask&(alpha<240)):continue
                    boundary=mask&~ndimage.binary_erosion(mask);yy,xx=np.nonzero(boundary)
                    supported=distance[boundary]<=tolerance
                    sides=[xx<x+bw*.25,xx>x+bw*.75,yy<y+bw*.5,yy>y+bh-bw*.5]
                    support=[float(supported[s].mean()) for s in sides]
                    if min(support)<.75 or supported.mean()<.95:continue
                    score=(float(supported.mean()),min(support),-sum(abs(z) for z in (dx,dy,dw,dh)))
                    if best is None or score>best[0]:
                        best=(score,mask,dict(box=[x,y,x+bw,y+bh],edge_support=support,
                            boundary_support=float(supported.mean()),capsule_iou=iou,
                            boundary_method='bounded_source_capsule_fit',localization_adjustment=[dx,dy,dw,dh]))
    return None if best is None else best[1:]


def _hardware_material_step(a, box, tolerance):
    """Validate a dark interior against adjacent metal without edge scores.

    The first samples lie beyond the existing localization tolerance. A deeper
    central band must show the same material difference, so a wide dark groove
    around otherwise unchanged metal cannot masquerade as a filled control.
    Source variation supplies the noise scale; no output RGB is inspected."""
    (x, y, x1, y1) = box
    rx = x1 - 1
    by = y1 - 1
    span = rx - x
    height = by - y + 1
    start = int(np.ceil(tolerance)) + 1
    offsets = np.arange(start, 3 * int(np.ceil(tolerance)) + 1)
    rows = np.arange(y + height // 4, y + height - height // 4)
    cx = (x + rx) / 2
    cols = np.arange(int(round(cx)) - max(1, int(span * 0.05)), int(round(cx)) + max(1, int(span * 0.05)) + 1)

    def compare(inside, outside, ia, oa):
        if not np.all(ia >= 240) or not np.all(oa >= 240):
            return dict(accepted=False, reason='nonopaque material witnesses')
        im = np.median(inside, axis=1)
        om = np.median(outside, axis=1)
        delta = om - im
        noise = np.r_[np.linalg.norm(inside - im[:, None, :], axis=2).ravel(), np.linalg.norm(outside - om[:, None, :], axis=2).ravel()]
        limit = max(1.0, float(np.quantile(noise, 0.99)))
        magnitudes = np.linalg.norm(delta, axis=1)
        luminance = delta @ np.array([0.299, 0.587, 0.114])
        supported = (magnitudes > limit) & (luminance > 0)
        return dict(accepted=bool(supported.mean() >= 0.75), supported_fraction=float(supported.mean()), source_noise_rgb=limit, inside_rgb=np.median(im, axis=0).tolist(), outside_rgb=np.median(om, axis=0).tolist(), delta_rgb=np.median(delta, axis=0).tolist(), median_delta_gray=float(np.median(luminance)))
    checks = {}
    core_half = max(1, int(span * 0.125))
    core_cols = np.arange(int(round(cx)) - core_half, int(round(cx)) + core_half + 1)
    for (label, bx, sign) in [('left', x, -1), ('right', rx, 1)]:
        ix = bx - sign * offsets
        ox = bx + sign * offsets
        if min(ix.min(), ox.min()) < 0 or max(ix.max(), ox.max()) >= a.shape[1]:
            return dict(accepted=False, reason='material witnesses outside source')
        checks[label] = compare(a[rows[:, None], ix, :3].astype(float), a[rows[:, None], ox, :3].astype(float), a[rows[:, None], ix, 3], a[rows[:, None], ox, 3])
        checks['deep_' + label] = compare(a[rows[:, None], core_cols, :3].astype(float), a[rows[:, None], ox, :3].astype(float), a[rows[:, None], core_cols, 3], a[rows[:, None], ox, 3])
    for (label, by0, sign) in [('top', y, -1), ('bottom', by, 1)]:
        iy = by0 - sign * offsets
        oy = by0 + sign * offsets
        if min(iy.min(), oy.min()) < 0 or max(iy.max(), oy.max()) >= a.shape[0]:
            return dict(accepted=False, reason='material witnesses outside source')
        checks[label] = compare(a[iy[None, :], cols[:, None], :3].astype(float), a[oy[None, :], cols[:, None], :3].astype(float), a[iy[None, :], cols[:, None], 3], a[oy[None, :], cols[:, None], 3])
    return dict(accepted=all((v['accepted'] for v in checks.values())), source_sample_offsets=offsets.tolist(), checks=checks)

def _independent_hardware_proposals(image, face, existing=()):
    """Propose missing dark controls from paired source rails and cap transitions.

    Existing accepted components are excluded. Native filled contours supply
    shape evidence; their noisy convex hulls only bound the search. Enumerate
    every supported rail and cap peak, then retain the original source support,
    opacity, shape and one-pixel localization checks plus independent material
    witnesses. This conservative fallback does not infer missing boundaries."""
    if face not in ('left', 'right'):
        raise ValueError('Side hardware requires left or right')
    a = np.asarray(image.convert('RGBA'))
    (h, w) = a.shape[:2]
    gray = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(gray, (0, 0), max(1.0, w * 0.008))
    edges = cv2.Canny(smooth, 4, 12)
    dist = ndimage.distance_transform_edt(edges == 0)
    dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    tol = max(1.5, w * 0.015)
    all_entries = []
    proposals = []
    (gy, gx) = np.indices((h, w))
    for gap in (0, 1, 2):
        joined = edges if gap == 0 else cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((gap * 2 + 1,) * 2, np.uint8))
        (contours, _) = cv2.findContours(joined, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for (ci, contour) in enumerate(contours):
            hull = cv2.convexHull(contour)
            (x, y, bw, bh) = cv2.boundingRect(hull)
            if not (w * 0.2 < bw < w * 0.65 and h * 0.035 < bh < h * 0.23 and (h * 0.15 < y) and (y + bh < h * 0.85) and (x > w * 0.15) and (x + bw < w * 0.85) and (w * 0.4 < x + bw / 2 < w * 0.6) and (2 < bh / bw < 9)):
                continue
            prior = np.zeros((h, w), np.uint8)
            cv2.drawContours(prior, [contour], -1, 1, -1)
            prior = prior > 0
            if any((np.count_nonzero(prior & m) / min(prior.sum(), m.sum()) > 0.8 for (m, _) in existing)):
                continue
            source_corner = max(2, int(bw * 0.15))
            source_corner_fill = max((float(part.mean()) for part in (prior[y:y + source_corner, x:x + source_corner], prior[y:y + source_corner, x + bw - source_corner:x + bw], prior[y + bh - source_corner:y + bh, x:x + source_corner], prior[y + bh - source_corner:y + bh, x + bw - source_corner:x + bw])))
            if source_corner_fill > 0.25:
                continue
            y0 = y + bh // 4
            y1 = y + bh - bh // 4
            roi = edges[y0:y1, x:x + bw] > 0
            normal = np.abs(dx[y0:y1, x:x + bw]) > np.abs(dy[y0:y1, x:x + bw])
            xx = np.arange(bw)
            left = (roi & normal & (dx[y0:y1, x:x + bw] < 0)).sum(axis=0)
            right = (roi & normal & (dx[y0:y1, x:x + bw] > 0)).sum(axis=0)
            left[xx >= bw * 0.33] = 0
            right[xx <= bw * 0.67] = 0
            left_choices = np.flatnonzero(left / (y1 - y0) >= 0.75) + x
            right_choices = np.flatnonzero(right / (y1 - y0) >= 0.75) + x
            for lx in left_choices.tolist():
                for rx in right_choices.tolist():
                    lv = float(left[lx - x] / (y1 - y0))
                    rv = float(right[rx - x] / (y1 - y0))
                    span = rx - lx
                    if span <= w * 0.2:
                        continue
                    cx = (lx + rx) / 2
                    half = max(1, int(round(span * 0.05)))
                    cols = np.arange(int(round(cx)) - half, int(round(cx)) + half + 1)
                    # A connected source scratch can extend the contour box
                    # beyond a real cap. The box bounds candidate search only;
                    # every cap still needs the unchanged source gates below.
                    top_y = np.arange(y, y + bh // 3)
                    bottom_y = np.arange(y + bh * 2 // 3, y + bh)
                    top_score = np.median(-dy[top_y[:, None], cols], axis=1)
                    bottom_score = np.median(dy[bottom_y[:, None], cols], axis=1)
                    top_choices = top_y[(top_score >= 12) & (top_score == ndimage.maximum_filter1d(top_score, 3))]
                    bottom_choices = bottom_y[(bottom_score >= 12) & (bottom_score == ndimage.maximum_filter1d(bottom_score, 3))]
                    for ty in top_choices.tolist():
                        for by in bottom_choices.tolist():
                            radius = span / 2
                            mask = (gx - cx) ** 2 + (gy - np.clip(gy, ty + radius, by - radius)) ** 2 <= radius ** 2
                            boundary = mask & ~ndimage.binary_erosion(mask)
                            (yy, xxx) = np.nonzero(boundary)
                            s = dist[boundary] <= tol
                            parts = [xxx < lx + span * 0.25, xxx > lx + span * 0.75, yy < ty + span * 0.5, yy > by - span * 0.5]
                            support = [float(s[v].mean()) for v in parts]
                            iou = float((mask & prior).sum() / (mask | prior).sum())
                            cp = max(2, int((span + 1) * 0.15))
                            cf = max((float(part.mean()) for part in [mask[ty:ty + cp, lx:lx + cp], mask[ty:ty + cp, rx - cp + 1:rx + 1], mask[by - cp + 1:by + 1, lx:lx + cp], mask[by - cp + 1:by + 1, rx - cp + 1:rx + 1]]))
                            entry = dict(search_component=[x, y, bw, bh], gap=gap, contour_index=ci, rail_x=[lx, rx], rail_vote=[lv, rv], source_apex=[ty, by], cap_gradient=[float(top_score[np.flatnonzero(top_y == ty)[0]]), float(bottom_score[np.flatnonzero(bottom_y == by)[0]])], box=[lx, ty, rx + 1, by + 1], support=float(s.mean()), edge_support=support, source_component_iou=iou, corner_fill=cf, alpha_ok=not np.any(mask & (a[..., 3] < 240)))
                            entry['source_corner_fill'] = source_corner_fill
                            entry['proposal_method'] = 'independent_signed_source_rails_and_caps_v1'
                            entry['accepted'] = bool(s.mean() >= 0.95 and min(support) >= 0.75 and (iou >= 0.9) and (cf <= 0.25) and entry['alpha_ok'])
                            if not entry['accepted'] and min(support) >= 0.75 and (iou >= 0.9) and entry['alpha_ok']:
                                refined = refine_capsule_boundary(dist, a[..., 3], [lx, ty, span + 1, by - ty + 1], mask, tol)
                                if refined is not None:
                                    (refined_mask, fit) = refined
                                    source_iou = float((refined_mask & prior).sum() / (refined_mask | prior).sum())
                                    if source_iou >= 0.9:
                                        mask = refined_mask
                                        entry.update(fit)
                                        entry['source_component_iou'] = source_iou
                                        entry['support'] = fit['boundary_support']
                                        entry['accepted'] = True
                            if entry['accepted']:
                                material = _hardware_material_step(a, entry['box'], tol)
                                entry['independent_material'] = material
                                entry['accepted'] = material['accepted']
                            entry['pixels'] = int(mask.sum())
                            entry['boundary_support'] = float(entry['support'])
                            entry.setdefault('capsule_iou', float(entry['source_component_iou']))
                            all_entries.append(entry)
                            if entry['accepted']:
                                proposals.append((mask, entry))
    return (proposals, all_entries)

def _distinct_hardware_proposals(proposals, tolerance):
    """Validate the entire overlap group before choosing a representative.

    Localization agreement is not transitive: a middle candidate must not hide
    two incompatible extremes. All group members must agree pairwise within the
    existing tolerance; area ordering only chooses among equivalent solutions."""

    def key(item):
        (ys, xs) = np.nonzero(item[0])
        return (-int(item[0].sum()), int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    proposals = sorted(proposals, key=key)
    neighbors = [set() for _ in proposals]
    distances = {}
    for (i, (mask, _)) in enumerate(proposals):
        for j in range(i):
            other = proposals[j][0]
            if np.count_nonzero(mask & other) / min(mask.sum(), other.sum()) > 0.8:
                neighbors[i].add(j)
                neighbors[j].add(i)
    visited = set()
    selected = []
    for i in range(len(proposals)):
        if i in visited:
            continue
        group = set()
        pending = [i]
        while pending:
            k = pending.pop()
            if k in group:
                continue
            group.add(k)
            pending.extend(neighbors[k] - group)
        indices = sorted(group)
        for (pos, j) in enumerate(indices):
            for k in indices[:pos]:
                for idx in (j, k):
                    if idx not in distances:
                        distances[idx] = ndimage.distance_transform_edt(~proposals[idx][0])
                if np.any(proposals[j][0] & (distances[k] > tolerance)) or np.any(proposals[k][0] & (distances[j] > tolerance)):
                    raise ValueError('Ambiguous competing independent material boundaries')
        visited.update(group)
        selected.append(proposals[indices[0]])
    return selected

def _resolve_missing_hardware(image, face, partial):
    """Add only independently certified unexplained material components.

 Call only after the existing detector has exhausted its original candidates.
 Its accepted masks have priority and are returned unchanged. This helper does
 not waive the final count, overlap, opacity, source-support or shape checks.
 """
    expected = 4 if face == 'left' else 2
    if len(partial) >= expected:
        if len(partial) > expected:
            raise ValueError('Unresolved existing hardware count exceeds profile')
        return (partial, {'used_fallback': False})
    (proposals, entries) = _independent_hardware_proposals(image, face, existing=partial)
    tol = max(1.5, image.width * 0.015)
    proposals = _distinct_hardware_proposals(proposals, tol)
    extras = []
    for (mask, proof) in proposals:
        if any((np.count_nonzero(mask & m) / min(mask.sum(), m.sum()) > 0.8 for (m, _) in partial)):
            continue
        extras.append((mask, proof))
    result = sorted(partial + extras, key=lambda v: v[1]['box'][1])
    expected = 4 if face == 'left' else 2
    if len(result) != expected:
        raise ValueError(f'Unresolved {face} hardware materials: {len(result)}, expected {expected}')
    combined = np.zeros((image.height, image.width), bool)
    for (mask, _) in result:
        if np.any(combined & mask):
            raise ValueError('Ambiguous independent/legacy hardware overlap')
        combined |= mask
    return (result, dict(used_fallback=True, legacy_partial=len(partial), independent_unique=len(proposals), added=len(extras), proposals=entries))


def _select_hardware_candidates(candidates, width, allow_nested=False):
    """Keep the existing selection gates and expose rejected source candidates.

    A crossing is a disagreement between inferred masks, not a claim that two
    physical buttons intersect. Evidence is diagnostic only: it cannot choose
    a mask, alter a boundary, or relax the localization tolerance.
    """
    selected=[]
    tolerance=max(1.5,width*.015)
    for mask,proof in sorted(candidates,
            key=lambda v:(v[1]['pixels'],v[1]['boundary_support']),reverse=True):
        overlaps=[(m,p) for m,p in selected
            if np.count_nonzero(mask&m)/min(mask.sum(),m.sum())>.8]
        if overlaps:
            distance=ndimage.distance_transform_edt(~overlaps[0][0])
            outside=mask&(distance>tolerance)
            # Pro Max camera control: the button inside a pale chamfer seen at an
            # angle, so the button capsule may cross the irregular chamfer hull by
            # a few px where the chamfer is foreshortened. Last resort only (after
            # independent resolution): of the nested pair keep the cleaner capsule,
            # which is the button outline used on every other sample; a real
            # crossing still fails.
            nested=(allow_nested and len(overlaps)==1 and np.count_nonzero(mask&overlaps[0][0])/mask.sum()>=.95
                    and not np.any(mask&(distance>2*tolerance)))
            if nested and np.any(outside):
                if proof['capsule_iou']>overlaps[0][1]['capsule_iou']:
                    selected[[id(m) for m,_ in selected].index(id(overlaps[0][0]))]=(mask,dict(proof,nested_contour_fallback=True))
                continue
            if len(overlaps)!=1 or np.any(outside):
                comparisons=[]
                for other,other_proof in overlaps:
                    delta=ndimage.distance_transform_edt(~other)
                    protruding=mask&(delta>tolerance)
                    yy,xx=np.nonzero(protruding)
                    comparisons.append(dict(selected_candidate=dict(other_proof),
                        overlap_fraction=float(np.count_nonzero(mask&other)/min(mask.sum(),other.sum())),
                        incoming_outside_tolerance_pixels=int(protruding.sum()),
                        maximum_incoming_distance_px=float(delta[mask].max()),
                        outside_bounds_xyxy=([int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1] if len(xx) else None),
                        witnesses_truncated=bool(len(xx)>24),
                        source_witnesses_xy=[[int(x),int(y)] for x,y in zip(xx[:24],yy[:24])]))
                error=ValueError('Ambiguous crossing side hardware contours')
                error.hardware_contour_conflict=dict(
                    meaning='source_candidate_boundary_disagreement_not_confirmed_image_defect',
                    tolerance_px=float(tolerance),incoming_candidate=dict(proof),
                    overlap_count=len(overlaps),comparisons=comparisons,
                    accepted_candidate_count_before_failure=len(selected))
                raise error
            continue
        selected.append((mask,proof))
    return selected


def _hardware_candidate_groups(candidates):
    """Group overlapping hypotheses without merging their source masks."""
    neighbors=[set() for _ in candidates]
    for i,(mask,_) in enumerate(candidates):
        for j in range(i):
            other=candidates[j][0]
            if np.count_nonzero(mask&other)/min(mask.sum(),other.sum())>.8:
                neighbors[i].add(j);neighbors[j].add(i)
    groups=[];visited=set()
    for i in range(len(candidates)):
        if i in visited:continue
        group=set();pending=[i]
        while pending:
            index=pending.pop()
            if index in group:continue
            group.add(index);pending.extend(neighbors[index]-group)
        visited.update(group);groups.append(sorted(group))
    return groups


def _resolve_hardware_conflicts(image,face,candidates,original_error):
    """Replace only uniquely source-certified legacy conflict groups.

    Accepted nonconflicting masks/proofs retain identity. The independent
    detector certifies source rails, caps, shape and interior material; all of
    its competing candidates must agree pairwise within the old tolerance.
    A non-equivalent legacy boundary with its own material witnesses vetoes
    replacement. Nothing here changes source RGB or the final delivery gates.
    """
    tolerance=max(1.5,image.width*.015);unchanged=[];conflicts=[]
    original_conflict=getattr(original_error,'hardware_contour_conflict',None)
    if original_conflict is None:raise original_error
    for indices in _hardware_candidate_groups(candidates):
        group=[candidates[i] for i in indices]
        try:
            unchanged.extend(_select_hardware_candidates(group,image.width))
        except ValueError as error:
            conflict=getattr(error,'hardware_contour_conflict',None)
            if conflict is None:raise
            conflicts.append((indices,group,conflict))
    if not conflicts:raise original_error
    independent,attempts=_independent_hardware_proposals(image,face,existing=unchanged)
    pixels=np.asarray(image.convert('RGBA'));resolved=[]

    def fail(reason,review,cause=None):
        error=ValueError('Ambiguous crossing side hardware contours')
        error.hardware_contour_conflict=dict(original_conflict,
            independent_disambiguation=dict(review,accepted=False,reason=reason))
        if cause is None:raise error
        raise error from cause

    for indices,group,conflict in conflicts:
        matching=[(mask,proof) for mask,proof in independent
            if any(np.count_nonzero(mask&other)/min(mask.sum(),other.sum())>.8
                   for other,_ in group)]
        review=dict(method='unique_independent_source_material_conflict_resolution_v1',
            source_size=[image.width,image.height],original_conflict=conflict,
            legacy_candidate_indices=indices,legacy_candidates=[dict(proof) for _,proof in group],
            independent_accepted_candidates=[dict(proof) for _,proof in matching],
            independent_attempts=attempts,tolerance_px=float(tolerance),
            acceptance_gates_unchanged=True)
        try:
            unique=_distinct_hardware_proposals(matching,tolerance)
        except ValueError as error:
            fail('Competing independently certified source boundaries',review,error)
        if len(unique)!=1:
            review['independent_distinct_count']=len(unique)
            fail('Exactly one independent material boundary required',review)
        selected,proof=unique[0]
        distance=ndimage.distance_transform_edt(~selected);comparisons=[];competitors=[]
        for index,(other,legacy_proof) in zip(indices,group):
            reverse=ndimage.distance_transform_edt(~other)
            forward_max=float(distance[other].max());reverse_max=float(reverse[selected].max())
            equivalent=max(forward_max,reverse_max)<=tolerance
            material=_hardware_material_step(pixels,legacy_proof['box'],tolerance)
            comparisons.append(dict(legacy_index=index,
                maximum_legacy_distance_px=forward_max,
                maximum_independent_distance_px=reverse_max,
                equivalent_within_original_tolerance=equivalent,
                independent_material_witnesses=material))
            if material['accepted'] and not equivalent:competitors.append(index)
        review['legacy_material_comparisons']=comparisons
        if competitors:
            review['competing_legacy_indices']=competitors
            fail('Non-equivalent legacy boundary has independent material evidence',review)
        review.update(accepted=True,independent_distinct_count=1,
            unchanged_other_components=len(unchanged))
        # Copy only the new proof. Do not mutate an attempt entry referenced in
        # the audit, or any accepted legacy mask/proof in another component.
        resolved.append((selected,dict(proof,hardware_conflict_resolution=review)))
    return unchanged+resolved


def hardware_materials(image, face):
    """Resolve this studio profile's complete capsule components independently.

    Recognition may smooth a working signal and bridge short edge gaps. The
    source RGB/alpha are never edited, and every candidate must have source
    edge support on both rails and both caps before it becomes a material.
    """
    if face not in ('left', 'right'):
        raise ValueError('Side hardware requires left or right')
    a = np.asarray(image.convert('RGBA')); h, w = a.shape[:2]
    gray = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), max(1., w*.008)), 4, 12)
    distance = ndimage.distance_transform_edt(edges == 0)
    candidates = []; refinements=[]; blemished=[]; gy, gx = np.indices((h, w))
    for gap in (0, 1, 2):
        size = 2*gap + 1
        bridged = edges if not gap else cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, np.ones((size, size), np.uint8))
        contours, _ = cv2.findContours(bridged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour_index,contour in enumerate(contours):
            hull = cv2.convexHull(contour)
            x, y, bw, bh = cv2.boundingRect(hull)
            if not (w*.20 < bw < w*.65 and h*.035 < bh < h*.23
                    and h*.15 < y and y+bh < h*.85
                    and x > w*.15 and x+bw < w*.85
                    and w*.4 < x+bw/2 < w*.6 and 2 < bh/bw < 9):
                continue
            mask = np.zeros((h, w), np.uint8)
            cv2.drawContours(mask, [hull], -1, 1, -1); mask = mask > 0
            corner=max(2,int(bw*.15))
            corner_fills=[float(part.mean()) for part in (
                mask[y:y+corner,x:x+corner],mask[y:y+corner,x+bw-corner:x+bw],
                mask[y+bh-corner:y+bh,x:x+corner],mask[y+bh-corner:y+bh,x+bw-corner:x+bw])]
            corner_fill=max(corner_fills)
            if sorted(corner_fills)[-2]>.25:
                continue  # A long rectangular scratch can have high whole-shape IoU.
            # One filled corner is a blemish touching the rim, not a rectangle:
            # only a fitted capsule (never the bulging hull) may stand for it.
            blemish_corner=corner_fill>.25
            radius = (bw-1)/2
            capsule = ((gx-x-radius)**2
                       + (gy-np.clip(gy, y+radius, y+bh-1-radius))**2 <= radius**2)
            iou = np.count_nonzero(mask & capsule) / np.count_nonzero(mask | capsule)
            if iou < .90 or np.any(mask & (a[..., 3] < 240)):
                continue
            boundary = mask & ~ndimage.binary_erosion(mask)
            yy, xx = np.nonzero(boundary)
            supported = distance[boundary] <= max(1.5, w*.015)
            sides = [xx < x+bw*.25, xx > x+bw*.75,
                     yy < y+bw*.5, yy > y+bh-bw*.5]
            support = [float(supported[s].mean()) for s in sides]
            entry=dict(box=[x, y, x+bw, y+bh],
                edge_support=support, boundary_support=float(supported.mean()),
                capsule_iou=float(iou),maximum_corner_fill=corner_fill,
                gap_radius=gap,contour_index=contour_index,
                source_contour_area=float(cv2.contourArea(contour)),
                source_hull_area=float(cv2.contourArea(hull)),pixels=int(mask.sum()))
            if min(support) < .75 or supported.mean() < .95 or blemish_corner:
                # A rail/cap without independent source evidence remains a
                # failure. Refine only the localization of a complete capsule.
                if min(support)<.75:continue
                (blemished if blemish_corner else refinements).append((mask,entry));continue
            candidates.append((mask,entry))
    for mask,proof in sorted(refinements,key=lambda item:item[1]['pixels'],reverse=True):
        # An independently supported original contour takes precedence. Do
        # not add a fitted alternative over an already-resolved component.
        if any(np.count_nonzero(mask&m)/min(mask.sum(),m.sum())>.8 for m,_ in candidates):continue
        x,y,x1,y1=proof['box']
        refined=refine_capsule_boundary(distance,a[...,3],[x,y,x1-x,y1-y],mask,max(1.5,w*.015))
        if refined is None:continue
        mask,new_proof=refined;proof=dict(proof,**new_proof);proof['pixels']=int(mask.sum())
        candidates.append((mask,proof))
    # Keep complete outer contours ahead of nested grooves. Resolve an old
    # conflict only with a unique, independently certified source material.
    try:
        selected=_select_hardware_candidates(candidates,w)
    except ValueError as error:
        if getattr(error,'hardware_contour_conflict',None) is None:raise
        try:
            selected=_resolve_hardware_conflicts(image,face,candidates,error)
        except ValueError as unresolved:
            if getattr(unresolved,'hardware_contour_conflict',None) is None:raise
            try:selected=_select_hardware_candidates(candidates,w,allow_nested=True)
            except ValueError:raise unresolved from None
    # This is the existing iPhone 17 studio capture profile, not an arbitrary
    # phone detector. Missing/extra hardware is ambiguity, not a blank domain.
    expected = 4 if face == 'left' else 2
    selected.sort(key=lambda entry: entry[1]['box'][1])
    if len(selected) < expected:
        try:
            selected, proposal_review = _resolve_missing_hardware(image, face, selected)
            # Preserve every already-valid mask and proof. Attach the full competing
            # proposal audit only to new entries, using copies to avoid proof cycles.
            selected = [
                (mask, dict(proof, independent_proposal_review=proposal_review))
                if proof.get('proposal_method') == 'independent_signed_source_rails_and_caps_v1'
                else (mask, proof)
                for mask, proof in selected
            ]
        except ValueError as missing:
            # A blemished rim is the last resort for a control nothing else
            # explains; only its fitted capsule (never the bulging hull) is used.
            for mask,proof in sorted(blemished,key=lambda item:item[1]['pixels'],reverse=True):
                if len(selected)>=expected:break
                if any(np.count_nonzero(mask&m)/min(mask.sum(),m.sum())>.8 for m,_ in selected):continue
                x,y,x1,y1=proof['box']
                refined=refine_capsule_boundary(distance,a[...,3],[x,y,x1-x,y1-y],mask,max(1.5,w*.015))
                if refined is None:continue
                mask,new_proof=refined
                if any(np.any(mask&m) for m,_ in selected):continue
                selected.append((mask,dict(proof,**new_proof,pixels=int(mask.sum()),blemished_rim_fallback=True)))
            if len(selected)!=expected:raise missing
            selected.sort(key=lambda entry: entry[1]['box'][1])
    if len(selected) != expected:
        raise ValueError(f'Unresolved {face} hardware materials: {len(selected)}, expected {expected}')
    combined = np.zeros((h, w), bool)
    for mask, proof in selected:
        if np.any(combined & mask):
            raise ValueError('Ambiguous overlapping side hardware materials')
        combined |= mask
    return selected
