"""New diagnostic helper, no imports or work on images until explicitly called.

Caller must hold the global guard and validate all source/job/mask bindings.
No acceptance decision; independent materials can have real contrast changes.
"""


def measure(job, original_job, source, candidate, *, lightness, require):
    import numpy as np
    from PIL import Image
    with Image.open(source) as image:
        require(image.mode == 'RGBA' and image.format == 'PNG', 'Crosspair source format differs')
        a = np.asarray(image).copy()
    with Image.open(candidate) as image:
        require(image.mode == 'RGBA' and image.format == 'PNG', 'Crosspair candidate format differs')
        b = np.asarray(image).copy()
    require(a.shape == b.shape and np.array_equal(a[..., 3], b[..., 3]), 'Crosspair geometry/alpha differs')
    domains = {d['name']: d for d in job['views'][0]['domains']}
    original = {d['name']: d for d in original_job['views'][0]['domains']}
    require(len(domains) == len(job['views'][0]['domains']), 'Duplicate crosspair domain')

    def mask(path):
        with Image.open(path) as image:
            m = np.asarray(image.convert('L')) > 0
        require(m.shape == a.shape[:2], 'Crosspair mask dimensions differ')
        return m

    carrier = mask(domains['main_surface']['mask'])
    hardware = {n: d for n, d in domains.items() if n.startswith('hardware_')}
    require(set(hardware) == {n for n in original if n.startswith('hardware_')}, 'Hardware set differs')
    require(all(d == original[n] for n, d in hardware.items()), 'Hardware field or mask changed')
    before = lightness(a[..., :3]); after = lightness(b[..., :3]); opaque = a[..., 3] >= 240
    results = {}
    for name, domain in hardware.items():
        h = mask(domain['mask']); require(not np.any(h & carrier), 'Hardware and carrier overlap')
        axes = {}
        for axis in (0, 1):
            p, q = ((slice(None, -1), slice(None)), (slice(1, None), slice(None))) if axis == 0 else (
                (slice(None), slice(None, -1)), (slice(None), slice(1, None)))
            h_to_c = h[p] & carrier[q]; c_to_h = carrier[p] & h[q]
            raw = h_to_c | c_to_h; crossing = raw & opaque[p] & opaque[q]
            yy, xx = np.nonzero(crossing)
            orientation = np.where(h_to_c[crossing], -1.0, 1.0)
            old = (before[q]-before[p])[crossing]*orientation
            new = (after[q]-after[p])[crossing]*orientation
            added = new-old; magnitude = np.abs(added)
            flips = (np.abs(old) > 1.25) & (old*new < 0)
            resolved_flips = flips & (np.abs(new) > 1.25)
            pairs = []
            for i in range(len(yy)):
                py, px = int(yy[i]), int(xx[i]); qy, qx = py+(axis == 0), px+(axis == 1)
                hw, metal = ([px, py], [qx, qy]) if h_to_c[py, px] else ([qx, qy], [px, py])
                pairs.append(dict(hardware_xy=hw, carrier_xy=metal,
                    source_hardware_minus_carrier_L=float(old[i]),
                    candidate_hardware_minus_carrier_L=float(new[i]), added_contrast_L=float(added[i]),
                    sign_reversal=bool(flips[i]),
                    sign_reversal_both_magnitudes_above_1_25_L=bool(resolved_flips[i])))
            axes[str(axis)] = dict(raw_cross_pair_count=int(raw.sum()),
                excluded_alpha_below_240_pairs=int((raw & ~crossing).sum()), pair_count=len(pairs),
                added_L_step_max=float(magnitude.max()) if len(magnitude) else None,
                added_L_step_p75=float(np.percentile(magnitude, 75)) if len(magnitude) else None,
                added_L_step_p95=float(np.percentile(magnitude, 95)) if len(magnitude) else None,
                pairs_above_1_25_L=int(np.sum(magnitude > 1.25)),
                source_contrast_sign_reversals_above_1_25_L=int(flips.sum()),
                sign_reversals_both_magnitudes_above_1_25_L=int(resolved_flips.sum()),
                same_domain_pairs_included=0, pair_evidence=pairs, automatic_acceptance=False)
        results[name] = dict(axes=axes, hardware_field_unchanged=True)
    return dict(status='diagnostic_only', hardware_domain_count=len(hardware), measurements=results,
        pairing='H[p]&C[q] or C[p]&H[q], both endpoints alpha>=240',
        contrast_orientation='hardware minus carrier',
        no_hardware_detected=not bool(hardware),
        threshold_semantics='1.25 L descriptive comparison only, not a new cross-material gate',
        semantic_material_certification=False, automatic_acceptance=False,
        quality_checks_complete=False, visual_approved=False, selected_for_delivery=False)
