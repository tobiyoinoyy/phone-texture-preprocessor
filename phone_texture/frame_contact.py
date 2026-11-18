"""Recover a studio frame locally where the pedestal or camera hides its silhouette."""

import numpy as np
from PIL import Image
from scipy import ndimage

from scripts.geometry_postprocess import rotate_alpha_safe
from .auto_geometry import principal


def _span(edge, guide, sign, width):
    # A small undulation can be finish damage. Only a substantial connected
    # projection qualifies; isolated outliers never set its endpoints.
    excursion = sign * (edge - guide)
    strong = excursion > max(8, 0.009 * width)
    joined = ndimage.binary_closing(strong, structure=np.ones(max(3, int(0.012 * width))))
    spans = [
        s[0]
        for s in ndimage.find_objects(ndimage.label(joined)[0])
        if s[0].stop - s[0].start >= 0.025 * width
    ]
    if not spans:
        return None
    if len(spans) != 1:
        raise ValueError("Frame contact has competing projections")
    left, right = spans[0].start, spans[0].stop
    # Include the transition into the projection, without following a
    # gently curved, exposed body edge for hundreds of columns.
    pad = max(3, int(0.015 * width))
    outer_left, outer_right = max(0, left - pad), min(width, right + pad)
    while left > outer_left and excursion[left - 1] > 2:
        left -= 1
    while right < outer_right and excursion[right] > 2:
        right += 1
    return left, right


def _contact(edge, x, span, gradient, threshold_edges, sign):
    left, right = span
    width = len(x)
    margin = max(8, int(0.028 * width))
    flanks = [
        np.arange(max(int(0.008 * width), left - margin), left),
        np.arange(right, min(width - int(0.008 * width), right + margin)),
    ]
    if min(map(len, flanks)) < 8:
        raise ValueError("Frame contact lacks two exposed ends")
    anchors = []
    for idx in flanks:
        y = edge[idx]
        flank_line = np.polyfit(x[idx] + 0.5, y, 1)
        if np.percentile(abs(y - np.polyval(flank_line, x[idx] + 0.5)), 90) > 2 or any(
            np.percentile(abs(other[idx] - y), 90) > 2 for other in threshold_edges
        ):
            raise ValueError("Frame contact ends disagree across silhouette samples")
        rows = np.clip(y[:, None] + np.arange(-3, 4), 0, gradient.shape[0] - 1)
        if (gradient[rows, x[idx, None]].max(axis=1) > 3).mean() < 0.8:
            raise ValueError("Frame contact ends lack original RGB evidence")
        anchors.append([np.median(x[idx] + 0.5), np.median(y)])
    line = np.polyfit(*np.asarray(anchors).T, 1)
    cols = x[left:right]
    predicted = np.polyval(line, cols + 0.5)
    offsets = np.arange(-7, 8)
    rows = np.clip(np.rint(predicted[:, None]).astype(int) + offsets, 0, gradient.shape[0] - 1)
    strength = gradient[rows, cols[:, None]]
    # A metal bevel often gives a stronger second peak. Select a resolved
    # local maximum near the two measured flanks, never the edge of the search.
    peaks = strength == ndimage.maximum_filter1d(strength, size=3, axis=1)
    peaks &= strength >= np.maximum(3.0, 0.35 * strength.max(axis=1, keepdims=True))
    peaks[:, [0, -1]] = False
    peak = np.argmin(np.where(peaks, abs(rows + 0.5 - predicted[:, None]), np.inf), axis=1)
    observed = rows[np.arange(len(cols)), peak] + 0.5
    # Refine the sampled gradient maximum to its subpixel centre. A step
    # between two pixel rows gives two equal maxima half a pixel apart.
    indices = np.arange(len(cols))
    left_strength = strength[indices, np.maximum(0, peak - 1)]
    right_strength = strength[indices, np.minimum(len(offsets) - 1, peak + 1)]
    curvature = left_strength + right_strength - 2 * strength[indices, peak]
    subpixel = np.divide(
        0.5 * (left_strength - right_strength),
        curvature,
        out=np.zeros_like(observed),
        where=curvature < -1e-6,
    )
    observed += np.clip(subpixel, -0.5, 0.5)
    valid = peaks.any(axis=1) & (abs(observed - predicted) <= 6)
    # Fit on alternating columns, then check disjoint columns in every
    # quarter of the contact. A short highlight cannot stand in for a rail.
    train = valid & (np.arange(len(cols)) % 2 == 0)
    if train.sum() < 8:
        raise ValueError("Frame contact missing source transition")
    # Fit the low-curvature photographed rail. Switching per column between
    # parallel bevel highlights would cut a sawtooth into the silhouette.
    # Use medians of non-overlapping windows and a smooth interpolant to
    # follow mild perspective curvature without reproducing pixel noise.
    windows = np.array_split(np.flatnonzero(train), max(2, len(cols) // 24))
    anchors_x = [float(np.median(cols[group])) for group in windows if len(group) >= 3]
    offsets_y = [float(np.median((observed - predicted)[group])) for group in windows if len(group) >= 3]
    if len(anchors_x) < 2 or np.max(np.diff(anchors_x)) > 48:
        raise ValueError("RGB contact contains an unsupported gap")
    edge_y = predicted + ndimage.gaussian_filter1d(np.interp(cols, anchors_x, offsets_y), 4)
    if np.max(abs(edge_y - predicted)) > 6:
        raise ValueError("RGB contact leaves its bracketed silhouette")
    held = np.arange(len(cols)) % 2 == 1
    # Held-out columns must contain a local peak at the fitted transition;
    # the immediately adjacent bevel can be stronger than the outer rail.
    supported = (peaks & (abs(rows + 0.5 - edge_y[:, None]) <= 2)).any(axis=1)
    fractions = [float(supported[k].mean()) for k in np.array_split(np.flatnonzero(held), 4)]
    if min(fractions) < 0.8:
        raise ValueError("Frame contact lacks consistent held-out RGB across the rail")
    pixels = np.r_[x[flanks[0]], cols, x[flanks[1]]]
    # Continue the measured rail through the soft ends of the occluder.
    # The min with native alpha later keeps the original round corners.
    boundary = np.interp(pixels, cols, edge_y)
    for flank, idx in [(flanks[0], slice(0, len(flanks[0]))), (flanks[1], slice(-len(flanks[1]), None))]:
        near = 0 if idx.start == 0 else -1
        origin = cols[near]
        slope = line[0]
        extension = edge_y[near] + slope * (x[flank] - origin)
        blend = np.minimum(1.0, abs(x[flank] - origin) / len(flank))
        boundary[idx] = extension * (1 - blend) + (edge[flank] + sign) * blend
    witness_cols = cols[held & supported]
    nearest = np.argmin(np.where(peaks, abs(rows + 0.5 - edge_y[:, None]), np.inf), axis=1)
    witness_rows = rows[np.arange(len(cols)), nearest][held & supported]
    return (
        pixels,
        boundary,
        (witness_cols, witness_rows),
        dict(
            source_span=[int(cols[0]), int(cols[-1]) + 1],
            anchor_line=line.tolist(),
            held_out_rgb_support=fractions,
        ),
    )


def recover_frame(matte, raw_image, face, stage_sink=None, deskew=True):
    if raw_image is None or raw_image.size != matte.size:
        raise ValueError("Frame contact needs its original RGB photograph")
    rgba = np.asarray(matte.convert("RGBA"))
    raw = np.asarray(raw_image.convert("RGB"))
    alpha = rgba[..., 3]
    mask = principal(alpha)
    _, all_x = np.nonzero(mask)
    x = np.arange(all_x.min(), all_x.max() + 1)
    width = len(x)
    if width < 350:
        raise ValueError("Frame contact resolution is insufficient")

    def edges(m):
        return np.argmax(m[:, x], axis=0), len(m) - np.argmax(m[::-1, x], axis=0)

    top, bottom = edges(mask)
    alternate = [edges(principal(alpha, t)) for t in (64, 192)]
    tips = np.r_[int(0.015 * width) : int(0.025 * width), int(0.975 * width) : int(0.985 * width)]
    lines = [np.polyfit(x[tips] + 0.5, edge[tips], 1) for edge in (top, bottom)]
    endpoints = x[[0, -1]] + 0.5
    thickness = np.polyval(lines[1], endpoints) - np.polyval(lines[0], endpoints)
    if (
        min(thickness) < 0.035 * width
        or max(thickness) > 0.25 * width
        or max(abs(line[0]) for line in lines) > 0.07
    ):
        raise ValueError("Frame contact end thickness or tilt is unsupported")
    slopes = np.degrees(np.arctan([line[0] for line in lines]))
    angle = float(np.mean(slopes)) if deskew else 0.0
    if np.ptp(slopes) > 0.3:
        raise ValueError("Exposed ends disagree on frame roll")
    cap, stand = [
        _span(edge, np.polyval(line, x + 0.5), sign, width)
        for edge, line, sign in zip((top, bottom), lines, (-1, 1))
    ]
    if (
        stand is None
        or not 0.18 * width <= stand[1] - stand[0] <= 0.75 * width
        or np.median(bottom[slice(*stand)] - np.polyval(lines[1], x[slice(*stand)] + 0.5)) < 0.2 * width
    ):
        raise ValueError("No distinct pedestal bracketed by exposed frame")
    if cap is not None and cap[0] < width / 2 < cap[1]:
        raise ValueError("Central upper projection cannot be identified as the studio camera island")
    if face == "top" and (cap is None or not 0.12 * width <= cap[1] - cap[0] <= 0.45 * width):
        raise ValueError("No independently bracketed top camera island")
    gradient = np.linalg.norm(
        np.gradient(ndimage.gaussian_filter1d(raw.astype(np.float32), 0.8, axis=0), axis=0), axis=2
    )
    result = rgba.copy()
    y = np.arange(len(alpha))[:, None]
    proof = {}
    body = np.zeros_like(mask)
    excluded = np.zeros_like(mask)
    opaque_core = ndimage.binary_erosion(alpha >= 250, iterations=3)
    for index, (key, edge, span, sign) in enumerate(
        zip(("camera", "pedestal"), (top, bottom), (cap, stand), (-1, 1))
    ):
        if span is None:
            continue
        cols, bound, witness, details = _contact(
            edge, x, span, gradient, [s[index] for s in alternate], sign
        )
        coverage = np.clip(bound - y if sign == 1 else y + 1 - bound, 0, 1)
        result[:, cols, 3] = np.minimum(result[:, cols, 3], np.rint(coverage * 255).astype(np.uint8))
        witness_x, witness_y = witness
        distance = sign * (y - witness_y)
        body[:, witness_x] |= opaque_core[:, witness_x] & (distance <= -6) & (distance >= -12)
        excluded[:, witness_x] |= distance >= 6
        proof[key] = details
    if (
        not body.any()
        or not (excluded & (alpha >= 128)).any()
        or result[..., 3][excluded].any()
        or not np.array_equal(result[body], rgba[body])
    ):
        raise ValueError("Frame recovery lost body pixels or retained projection")
    # Reject additional disconnected objects, tall supports or a second
    # unmodelled camera instead of letting their alpha enlarge the output.
    ys, xs = np.nonzero(result[..., 3] >= 64)
    if (
        xs.min() < x[0] - 3
        or xs.max() > x[-1] + 3
        or np.any(ys < np.polyval(lines[0], xs + 0.5) - max(8, 0.008 * width))
        or np.any(ys > np.polyval(lines[1], xs + 0.5) + max(8, 0.008 * width))
    ):
        raise ValueError("Residual foreground outside the measured body")

    if stage_sink:
        stage_sink("source_matte", matte)
        stage_sink("after_support", Image.fromarray(result))
    rotated = (
        Image.fromarray(rotate_alpha_safe(result, angle))
        if abs(angle) > 1e-5
        else Image.fromarray(result)
    )
    projected = []
    for region in (body, excluded):
        witness = Image.fromarray(region.astype(np.uint8))
        if abs(angle) > 1e-5:
            witness = witness.rotate(angle, resample=Image.Resampling.NEAREST, expand=True)
        projected.append(np.asarray(witness) > 0)
    inside, outside = projected
    rotated_alpha = np.asarray(rotated.getchannel("A"))
    if not inside.any() or (rotated_alpha[inside] < 128).any() or rotated_alpha[outside].any():
        raise ValueError("Frame contact fails body or projection checks after deskew")
    box = rotated.getchannel("A").getbbox()
    out = rotated.crop(box)
    x0, y0, x1, y1 = box
    if inside[y0:y1, x0:x1].sum() != inside.sum():
        raise ValueError("Tight crop lost a verified frame pixel")
    if stage_sink:
        stage_sink("after_rotation", rotated)
        stage_sink("after_crop", out)
    if face in ("left", "right"):
        out = out.transpose(Image.Transpose.ROTATE_90 if face == "left" else Image.Transpose.ROTATE_270)
    return out, dict(
        method="source_supported_local_frame_contacts",
        face=face,
        contacts=proof,
        rotation_ccw_degrees=angle,
        canvas_trim_xyxy=list(box),
        automatic_checks=dict(
            status="passed",
            body_witness_pixels=int(body.sum()),
            projection_witness_pixels=int((excluded & (alpha >= 128)).sum()),
        ),
        transform_checks=dict(
            status="passed",
            body_lost_after_rotation_pixels=0,
            body_lost_after_crop_pixels=0,
            residual_projection_pixels=0,
        ),
        camera_policy="exclude_entire_camera_island_and_lenses"
        if face == "top"
        else "exclude_projection",
        deskew_rotation_ccw_degrees=angle,
        orientation_quarter_turns_ccw={"left": 1, "right": -1}.get(face, 0),
        source_size=list(matte.size),
        output_size=list(out.size),
        rgb_painted=False,
        review_reasons=["heuristic_support_cut", "complete_outline_requires_review"],
    )
