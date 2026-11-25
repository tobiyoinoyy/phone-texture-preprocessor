"""Source-measured camera metal lighting shared by the dual and triple rear cameras."""

import cv2
import numpy as np
from scipy import ndimage as ndi

from .rear_shadow import _smoothstep as ease

ANGLES = np.arange(360)
THETA = np.deg2rad(ANGLES)[:, None]


def _scan(light, centre, offsets):
    x, y, radius = centre
    distance = radius + offsets
    return ndi.map_coordinates(
        light, [y + np.sin(THETA) * distance, x + np.cos(THETA) * distance], order=1, mode="nearest"
    )


def _independent_seeds(boosted):
    from .back_lens import _disk_readings, STANDARD

    h, w = boosted.shape[:2]
    profiles = np.array([spec['lenses'] for spec in STANDARD.values()]) * [w, h]
    windows = np.stack((profiles.min(axis=0) - .045*w, profiles.max(axis=0) + .045*w), axis=1)
    readings = list(_disk_readings(boosted))
    seeds = []
    for low, high in windows:
        candidates = []
        for disks in readings:
            inside = [d for d in disks if np.all(d[:2] >= low) and np.all(d[:2] <= high)]
            if len(inside) == 1:
                candidates.append(inside[0])
        points = np.array(candidates)
        groups = [points[np.max(abs(points - p), axis=1) <= 1.5] for p in points]
        best = max(groups, key=len, default=[])
        seeds.append(np.median(best, axis=0) if len(best) >= 3 else None)
    return seeds, windows


def _verified_glass(source, seed):
    from .back_geometry import measure_glass_edges

    x, y, radius = seed
    candidates = []
    for fraction in (0., -.03, .03):
        trial = [[x, y, radius * (1 + fraction)]]
        try:
            first = measure_glass_edges(source, trial, search_width=6.)[0]
            second = measure_glass_edges(source, trial, search_width=9.)[0]
        except ValueError:
            continue
        if np.max(abs(first - second)) <= .65:
            reading = (first + second) / 2
            if fraction == 0.:
                return reading
            candidates.append(reading)
    if candidates and np.max(np.ptp(candidates, axis=0)) <= .65:
        return np.mean(candidates, axis=0)
    return None


def measure(source, layout):
    """Keep the measured glass centre and both metal edges for each supported arc."""
    from .back_lens import _disks
    from .iphone17_model_finish import lens_pair

    location = "source_gray"
    exclusions = []
    slots = None
    if layout == "dual":
        centres = lens_pair(source)
    elif layout == "triple":
        centres, _ = _disks(source)
        if centres is None:

            gray = cv2.cvtColor(source[..., :3], cv2.COLOR_RGB2GRAY)
            boosted = source.copy()
            roi = gray[: round(len(gray) * 0.32), : round(gray.shape[1] * 0.65)]
            enhanced = cv2.createCLAHE(clipLimit=2.1, tileGridSize=(8, 8)).apply(roi)
            boosted[: roi.shape[0], : roi.shape[1], :3] = enhanced[..., None]
            seeds, _ = _disks(boosted)
            windows = None
            if seeds is None:
                seeds, windows = _independent_seeds(boosted)
            centres, slots = [], []
            for slot, seed in enumerate(seeds):
                measured = _verified_glass(source, seed) if seed is not None else None
                if measured is not None:
                    centres.append(measured)
                    slots.append(slot)
                    continue
                if seed is not None:
                    x, y, r = seed
                    pad = 1.65*r
                    box = [x-pad, y-pad, x+pad, y+pad]
                else:
                    low, high = windows[slot]
                    pad = .18 * source.shape[1]
                    box = [*(low-pad), *(high+pad)]
                exclusions.append(dict(slot=slot, box=box))
            if not centres:
                raise ValueError("三枚镜头均缺少原图分区边界依据")
            location = "enhanced_gray_verified_on_source"
    else:
        raise ValueError("Unknown camera layout")
    light = cv2.cvtColor(source[..., :3].astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[..., 0]
    lenses = []
    for centre in centres:
        lo, hi, stable = _edges(light, centre)
        lenses.append(
            dict(centre=centre.tolist(), inner=lo.tolist(), outer=hi.tolist(), supported=stable.tolist())
        )
    return dict(layout=layout, location=location, lenses=lenses,
                slots=list(range(len(lenses))) if slots is None else slots, exclusions=exclusions)


def _edges(light, centre):
    """Locate glass-to-metal and metal-to-joint gradients independently per direction."""
    r = centre[2]
    offsets = np.arange(-0.095 * r, 0.27 * r, 0.35)
    scan = ndi.gaussian_filter(_scan(light, centre, offsets), (2, 1.4), mode=("wrap", "nearest"))
    gradient = np.gradient(scan, 0.35, axis=1)
    inner = np.flatnonzero((offsets > -0.08 * r) & (offsets < 0.07 * r))
    outer = np.flatnonzero((offsets > 0.075 * r) & (offsets < 0.245 * r))
    rising = inner[np.argmax(gradient[:, inner], axis=1)]
    falling = outer[np.argmin(gradient[:, outer], axis=1)]
    lo, hi = r + offsets[rising], r + offsets[falling]
    width = hi - lo
    stable = (
        (gradient[ANGLES, rising] > 2)
        & (gradient[ANGLES, falling] < -1.45)
        & (np.abs(hi - ndi.median_filter(hi, 15, mode="wrap")) < 2.5)
        & (width > 0.048 * r)
        & (width < 0.25 * r)
    )
    return lo, hi, stable


def _geometry(shape, lens):
    y, x = np.ogrid[: shape[0], : shape[1]]
    cx, cy, r = lens["centre"]
    angle = np.rad2deg(np.arctan2(y - cy, x - cx)) % 360
    radius = np.hypot(x - cx, y - cy)
    lo, hi = [np.interp(angle, ANGLES, lens[key], period=360) for key in ("inner", "outer")]
    stable = np.asarray(lens["supported"])
    weight = ndi.gaussian_filter1d(stable.astype(float), 3, mode="wrap")
    trust = ease((np.interp(angle, ANGLES, weight, period=360) - 0.62) / 0.34)
    if stable.sum() < 210:
        trust[:] = 0
    ring = ease((radius - lo - 1.2) / 1.8) * ease((hi - radius - 0.8) / 1.8) * trust
    return ring, radius - hi, angle, trust


def dual_scope(shape, plan):
    """The hull of the two rims lies inside the photographed capsule. Exclude both glasses."""
    h, w = shape
    rims = []
    glasses = np.zeros((h, w), bool)
    for lens in plan["lenses"]:
        cx, cy, _ = lens["centre"]
        outer = np.asarray(lens["outer"])
        rims.extend(
            np.column_stack((cx + np.cos(THETA[:, 0]) * outer, cy + np.sin(THETA[:, 0]) * outer))
        )
        yy, xx = np.ogrid[:h, :w]
        angle = np.rad2deg(np.arctan2(yy - cy, xx - cx)) % 360
        inner = np.interp(angle, ANGLES, lens["inner"], period=360)
        glasses |= np.hypot(yy - cy, xx - cx) <= inner + 1
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.rint(rims).astype("int32")), 1)
    return (mask > 0) & ~glasses


def _filtered(values, mask, sigma):
    weight = ndi.gaussian_filter(mask.astype(float), sigma)
    return ndi.gaussian_filter(values * mask, sigma) / np.maximum(weight, 1e-7)


def _flash(source, shape):
    """Identify the source flash by circularity, size and multi-threshold agreement."""
    h, w = source.shape[:2]
    gray = cv2.cvtColor(source[: shape[0], :, :3], cv2.COLOR_RGB2GRAY)
    x0, x1 = int(0.70 * w), int(0.91 * w)
    y0, y1 = int(0.034 * h), int(0.142 * h)
    patch = gray[y0:y1, x0:x1]
    if min(patch.shape) < 8:
        return None
    readings = []
    for threshold in np.linspace(np.percentile(patch, 50), np.percentile(patch, 98), 9):
        outlines, _ = cv2.findContours(
            (patch > threshold).astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        found = []
        for contour in outlines:
            (x, y), r = cv2.minEnclosingCircle(contour)
            if (
                0.027 * w < r < 0.055 * w
                and x - r > 1
                and y - r > 1
                and x + r < patch.shape[1] - 1
                and y + r < patch.shape[0] - 1
                and cv2.contourArea(contour) > 0.84 * np.pi * r * r
            ):
                found.append([x + x0, y + y0, r])
        if len(found) == 1:
            readings.extend(found)
    if len(readings) < 3:
        return None
    points = np.array(readings)
    group = max((points[np.max(abs(points - p), axis=1) < 2] for p in points), key=len)
    return np.median(group, axis=0) if len(group) >= 3 else None


def _align_exterior(lab, flat, centre, distance, angle, trust):
    """Compare each direction to stable metal beyond its shade, on the same surface."""
    r = centre[2]
    reference = flat & (distance > 0.22 * r) & (distance < 0.36 * r)
    local = flat & (distance > 0.028 * r) & (distance < 0.24 * r) & (trust > 0)
    if not local.any():
        return 0
    bins = (angle / 5).astype(int)
    medians = np.zeros((72, 3))
    stable = np.zeros(72, bool)
    for i in np.unique(bins[reference]):
        samples = lab[reference & (bins == i)]
        if len(samples) < max(12, 0.0025 * r * r):
            continue
        median = np.median(samples, axis=0)
        if np.percentile(np.linalg.norm(samples - median, axis=1), 80) < 5.5:
            stable[i] = True
            medians[i] = median
    weight = ndi.gaussian_filter1d(stable.astype(float), 1.2, mode="wrap")
    refs = ndi.gaussian_filter1d(medians, 1.2, axis=0, mode="wrap") / np.maximum(weight[:, None], 1e-6)
    phase = (angle[local] / 5) - 0.5
    confidence = np.interp(phase, np.arange(72), weight, period=72)
    target = np.column_stack([np.interp(phase, np.arange(72), v, period=72) for v in refs.T])
    smooth = np.column_stack([_filtered(v, flat, 2.4)[local] for v in lab.transpose(2, 0, 1)])
    residual = target - smooth
    d = distance[local] / r
    blend = (
        ease((confidence - 0.72) / 0.24)
        * trust[local]
        * ease((d - 0.028) / 0.041)
        * ease((0.24 - d) / 0.10)
    )
    blend *= ease((residual[:, 0] - 0.9) / 3.0)
    shift = np.clip(residual, [-0.0, -7.0, -10.0], [14.0, 7.0, 10.0])
    lab[local] += blend[:, None] * shift
    return int(np.count_nonzero(blend > 0.03))


def _cast_shadows(raw, lab, flat, geometry, lenses, lit_edges):
    from .lens_exterior import lift as lift_cast_shadow

    margin_reference = flat.copy()
    for centre, distance, _, _ in geometry:
        margin_reference &= distance > 0.20 * centre[2]
    return [
        lift_cast_shadow(raw, lab, flat, lens, hi, ok, margin_reference)
        for lens, (hi, ok) in zip(lenses, lit_edges)
    ]


def apply(source, source_lab, output_lab, allowed, plan, plateau=None, *, exterior_only=False):
    """Work in the source geometry before resizing. Recover detail from this photograph."""
    from .rear_shadow import _plateau_core, _ring_material, InsufficientRearShadowEvidence

    lenses = plan["lenses"]
    bottom = min(len(source), int(max(y + 2.1 * r for x, y, r in (v["centre"] for v in lenses))) + 1)
    lab = output_lab[:bottom].copy()
    raw = source_lab[:bottom]
    metal = allowed[:bottom] & (source[:bottom, :, 3] == 255)
    shape = metal.shape
    for entry in plan.get('exclusions', []):
        x0, y0, x1, y1 = entry['box']
        metal[max(0, int(y0)):int(np.ceil(y1))+1, max(0, int(x0)):int(np.ceil(x1))+1] = False
    flat = metal.copy()
    if plan["layout"] == "triple":
        try:
            if not plateau or "flat_core_bounds" not in plateau:
                edges, proof = _plateau_core(source_lab[..., 0])
                plateau = dict(flat_core_bounds=edges, plateau_source_edges=proof)
            yy, xx = np.ogrid[:bottom, : shape[1]]
            flat = _ring_material(flat, dict(plateau, lenses=[]), yy, xx)
        except InsufficientRearShadowEvidence:
            flat[:] = False
    rings = np.zeros(shape)
    geometry = []
    confirmed_angles = []
    lit_edges = []
    for lens in lenses:
        ring, distance, angle, trust = _geometry(shape, lens)
        # A re-read of the lit image catches apparent displaced edges. Only
        # the overlap of both observed metal bands receives the shared colour.
        lit_lo, lit_hi, lit_ok = _edges(lab[..., 0], lens["centre"])
        lit_edges.append((lit_hi, lit_ok))
        lo = np.asarray(lens["inner"])
        hi = np.asarray(lens["outer"])
        corroborated = dict(
            lens,
            inner=np.maximum(lit_lo, lo),
            outer=np.minimum(lit_hi, hi),
            supported=lit_ok & lens["supported"] & (abs(lit_lo - lo) < 2.8) & (abs(lit_hi - hi) < 2.8),
        )
        confirmed_angles.append(int(np.sum(corroborated["supported"])))
        common_ring = _geometry(shape, corroborated)[0]
        rings = np.maximum(rings, np.minimum(ring, common_ring) * metal)
        flat &= distance > 2
        geometry.append((lens["centre"], distance, angle, trust))
    for lens, (centre, distance, _, _) in zip(lenses, geometry):
        if sum(lens["supported"]) < 210:
            if plan["layout"] == "dual":
                flat[:] = False
            else:
                # Keep a weak rim and its shade out of every shared metal sample.
                flat &= distance > .55 * centre[2]
    flash = _flash(source, shape) if plan["layout"] == "triple" and flat.any() else None
    if flash is None and plan["layout"] == "triple":
        # When the rim cannot be measured, omit the complete flash search window.
        fh, fw = source.shape[:2]
        flat[int(0.026 * fh) : int(0.148 * fh), int(0.69 * fw) : int(0.93 * fw)] = False
    if flash is not None:
        yy, xx = np.ogrid[:bottom, : shape[1]]
        x, y, r = flash
        # Allow for the transition between the reflective face and its outer bezel.
        distance = np.hypot(xx - x, yy - y) - (r + max(3, 0.10 * r))
        flat &= distance > 2
    # Remove isolated dark hardware (mic and LiDAR) from the flat-metal field.
    if flat.any():
        middle = np.median(raw[..., 0][flat])
        hard = (raw[..., 0] < min(22.0, middle - 20)) & flat
        n, components, stats, _ = cv2.connectedComponentsWithStats(hard.astype("uint8"), 8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if w > 3 and h > 3 and 0.72 < w / h < 1.4 and area > 0.63 * w * h:
                flat &= ~ndi.binary_dilation(components == i, iterations=3)
    report = dict(
        layout=plan["layout"],
        location=plan["location"],
        measured_slots=plan.get("slots", list(range(len(lenses)))),
        excluded_slots=[e["slot"] for e in plan.get("exclusions", [])],
        ring_confirmed_angles=confirmed_angles,
        supported_angles=[int(np.sum(v["supported"])) for v in lenses],
    )
    from .camera_shared_field import apply as apply_shared_field

    scale = shape[1] / 1000.0
    yy, xx = np.ogrid[:bottom, : shape[1]]
    quiet = flat.copy()
    eligible = flat.copy()
    requested = np.zeros(shape)
    for centre, d, _, _ in geometry:
        quiet &= d > 32 * scale
        requested = np.maximum(requested, ease((55 * scale - d) / (24 * scale)))
    if flash is not None:
        quiet &= distance > 38 * scale
        requested = np.maximum(requested, ease((76 * scale - distance) / (28 * scale)))
    seam = np.zeros(shape)
    if flat.any() and plateau and "flat_core_bounds" in plateau:
        x0, _, x1, y1 = plateau["flat_core_bounds"]
        straight = ease((xx - x0 - 0.22 * (x1 - x0)) / (0.06 * (x1 - x0))) * ease(
            (x1 - xx - 0.18 * (x1 - x0)) / (0.06 * (x1 - x0))
        )
        seam = straight * ease((53 * scale - abs(yy - y1)) / (17 * scale))
        eligible |= metal & (seam > 0)
        # Sample on both sides of the joint. The mark and the physical bevel
        # receive that field from the photo; neither determines its shape.
        quiet &= abs(yy - y1) > 20 * scale
        quiet |= metal & (straight > 0.9) & (yy > y1 + 20 * scale) & (yy < y1 + 48 * scale)
        requested = np.maximum(requested, seam)
    previous = lab[..., 0].copy()
    report["shared_field_pixels"] = apply_shared_field(
        raw[..., 0], lab[..., 0], quiet, requested, eligible, scale
    )
    report["lower_seam_pixels"] = int(
        np.count_nonzero((abs(previous - lab[..., 0]) > 0.1) & (seam > 0.01))
    )
    exterior = []
    for centre, d, angle, trust in geometry:
        exterior.append(_align_exterior(lab, flat, centre, d, angle, trust))
    report["lens_exterior_pixels"] = exterior
    if exterior_only:
        # A continuation has authority only for the verified exterior shadow.
        # Reuse the same field and source/target rim tests as the ordinary path.
        report["lens_cast_shadow"] = _cast_shadows(raw, lab, flat, geometry, lenses, lit_edges)
        weight = np.zeros(shape)
        for centre, distance, _, trust in geometry:
            # The measured falling edge can be on the INNER side of the dark
            # physical joint. Keep that joint intact on a fallback photograph.
            radial = distance / centre[2]
            weight = np.maximum(weight, trust * ease((radial - .052) / .025)
                                * ease((.30 - radial) / .05))
        lab = output_lab[:bottom] + weight[..., None] * (lab - output_lab[:bottom])
        active = flat & (weight > 0) & np.any(abs(lab - output_lab[:bottom]) > 1e-6, axis=-1)
        return lab, active, report
    report["flash_exterior_pixels"] = 0
    if flash is not None:
        angle = np.rad2deg(np.arctan2(yy - flash[1], xx - flash[0])) % 360
        report["flash_exterior_pixels"] = _align_exterior(
            lab, flat, flash, distance, angle, np.ones(shape)
        )
    # Separate medium-scale dark mottling from fine grain on warm, saturated metal.
    report["warm_metal_pixels"] = 0
    if flat.sum() > 500:
        colour = np.median(raw[flat], axis=0)
        if colour[1] > 12 and colour[2] > 18:
            light = lab[..., 0]
            small = _filtered(light, flat, 2)
            broad = _filtered(light, flat, 10)
            original_mid = _filtered(raw[..., 0], flat, 2) - _filtered(raw[..., 0], flat, 10)
            mid = small - broad
            fine = light - small
            delta = np.clip(-mid - 0.3, 0, 5) * 0.70 + np.clip(original_mid - mid, 0, 3) * 0.25
            delta -= fine * 0.55 * ease((9 - abs(fine)) / 4)
            # Keep the measured joint and its just-corrected neighbourhood intact.
            fade = ease((ndi.distance_transform_edt(flat) - 8) / 10)
            light += fade * delta
            report["warm_metal_pixels"] = int(np.count_nonzero(abs(fade * delta) > 0.1))
    from .lighting_spot_review import camera_arc
    spot_checks = []
    def review(zone, stage, mask):
        checks = [camera_arc(raw, lab, mask, lens, zone) for lens in lenses]
        spot_checks.append(dict(stage=stage, zone=zone, lenses=checks))

    review('exterior', 'shared_field_and_exterior', flat)
    inside = rings > 0.9
    report["ring_pixels"] = 0
    if inside.sum() > 100:
        lo, hi = np.percentile(lab[..., 0][inside], [24, 72])
        sample = inside & (lab[..., 0] >= lo) & (lab[..., 0] <= hi)
        target = np.median(lab[sample], axis=0)
        centre = np.median(raw[inside], axis=0)
        use = rings > 0.01
        low = np.column_stack([_filtered(v, use, 2.8)[use] for v in raw.transpose(2, 0, 1)])
        limits = np.array([9.0, 5.0, 5.0])
        wanted = (
            target
            + 0.15 * (low - centre)
            + [0.79, 0.74, 0.74] * limits * np.tanh((raw[use] - low) / limits)
        )
        lab[use] += rings[use, None] * (wanted - lab[use])
        report["ring_pixels"] = int(use.sum())
    review('ring', 'ring_lighting', metal)
    from .rear_shadow import _restore_secondary_arc

    report["secondary_arc_guard"] = []
    for lens in lenses:
        if sum(lens["supported"]) < 210:
            report["secondary_arc_guard"].append(dict(status="insufficient_angles"))
            continue
        cx, cy, r = lens["centre"]
        measured = dict(ellipse=[cx, cy, 2 * r, 2 * r, 0], hardware_radii=np.asarray(lens["outer"]) / r)
        valid = flat & (_geometry(shape, lens)[3] > 0.5)
        report["secondary_arc_guard"].append(
            _restore_secondary_arc(output_lab[:bottom], raw, lab, valid, measured)
        )
    changed = np.max(abs(lab - output_lab[:bottom]), axis=-1) > 1e-6
    # Bound the base illumination relative to the photo before the independent cast-shadow pass.
    lab[..., 0][changed] = np.clip(
        lab[..., 0][changed], raw[..., 0][changed] - 34, raw[..., 0][changed] + 34
    )
    report["lens_cast_shadow"] = _cast_shadows(raw, lab, flat, geometry, lenses, lit_edges)
    review('exterior', 'cast_shadow', flat)
    from .lens_rim_continuity import review as review_rim_continuity
    report['rim_continuity'] = [review_rim_continuity(raw, lab, metal, lens, flat) for lens in lenses]
    report['spot_review'] = spot_checks
    changed = np.max(abs(lab - output_lab[:bottom]), axis=-1) > 1e-6
    return lab, changed, report
