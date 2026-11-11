"""Resolve an unmarked printed strip from independent source closure evidence.

The overprotection audit is not a mask generator. This resolver separately
requires a transverse paper edge, both paper margins before it, broad metal
after it, and no later reappearance of those source paper margins.
"""
import hashlib
import json
import numpy as np

from .bottom_paper_audit import _source_cross_edge


def resolve_unmarked_strip_end(a, paper, proof, *, after_distance=4):
    if not proof.get('source_strip_edges'):
        raise ValueError('Unmarked closure requires independently fitted paper sides')
    line, lf, rf, edge = _source_cross_edge(a, paper, proof)
    h, w = paper.shape
    center = float(edge['source_end_peak'])

    def row_samples(y, local=False):
        l, r = int(round(np.polyval(lf, y))), int(round(np.polyval(rf, y)))
        span = r-l+1
        width = max(4, int(np.ceil(span*.1))) if local else max(8, int(np.ceil(span*.2)))
        gap = max(3, int(np.ceil(span*.04))) if local else max(4, int(np.ceil(span*.08)))
        xl = np.arange(l-gap-width, l-gap)
        xr = np.arange(r+gap+1, r+gap+width+1)
        inner = np.arange(l+2, r-1)
        if xl.min() < 0 or xr.max() >= w or len(inner) < 10:
            raise ValueError('Unmarked closure lacks independent lateral metal')
        xs = np.r_[xl, xr]
        if not np.all(a[y, np.r_[xs, inner], 3] >= 240):
            raise ValueError('Unmarked closure lacks opaque paper/metal samples')
        left, right = np.median(a[y, xl, :3], axis=0), np.median(a[y, xr, :3], axis=0)

        def background(x):
            t = (np.asarray(x)-np.median(xl))/(np.median(xr)-np.median(xl))
            return left+t[..., None]*(right-left)

        noise = max(1., float(np.quantile(np.linalg.norm(
            a[y, xs, :3].astype(float)-background(xs), axis=1), .99)))
        residual = a[y, inner, :3].astype(float)-background(inner)
        margin = max(4, int(np.ceil(len(inner)*.1)))
        return inner, residual, noise, margin, background

    # The preceding blank margins must differ from independently sampled metal.
    before = max(0, int(np.floor(center))-1)
    _, residual, noise, margin, _ = row_samples(before)
    direction = np.median(np.r_[residual[:margin], residual[-margin:]], axis=0)
    norm = float(np.linalg.norm(direction))
    direction /= max(norm, 1e-12)
    paper_signal = residual @ direction > noise
    broad_witness = dict(noise_rgb=noise, contrast_rgb=norm,
        margin_fractions=[float(paper_signal[:margin].mean()),float(paper_signal[-margin:].mean())])
    margin_sampling = 'broad'
    if norm <= noise or min(paper_signal[:margin].mean(), paper_signal[-margin:].mean()) < .75:
        # A wide lateral window can include the charging-port bevel. Check
        # the closer source metal independently, without loosening the RGB
        # contrast or 75% coverage gates. The following metal witnesses and
        # continuation guard still use the original broad sampling windows.
        _, residual, noise, margin, _ = row_samples(before, local=True)
        direction = np.median(np.r_[residual[:margin], residual[-margin:]], axis=0)
        norm = float(np.linalg.norm(direction))
        if norm <= noise:
            raise ValueError('Unmarked closure lacks source paper colour before edge')
        direction /= norm
        paper_signal = residual @ direction > noise
        if min(paper_signal[:margin].mean(), paper_signal[-margin:].mean()) < .75:
            raise ValueError('Unmarked closure lacks both source paper margins')
        margin_sampling = 'local_before_broad_after'
    preceding_witness = dict(row=before, sampling=margin_sampling, noise_rgb=noise,
        margin_fractions=[float(paper_signal[:margin].mean()),float(paper_signal[-margin:].mean())],
        contrast_rgb=norm, broad_reference=broad_witness)

    # Four pixels exceed the gradient detector's sigma=.8 support. Demand
    # broad source metal and both margins, rather than the audit's short run.
    witnesses = []
    for y in range(int(np.ceil(center))+after_distance, int(np.ceil(center))+after_distance+3):
        if y >= h:
            raise ValueError('Unmarked closure has no complete post-edge witness')
        _, residual, noise, margin, _ = row_samples(y)
        metal = np.linalg.norm(residual, axis=1) <= noise
        witnesses.append(dict(row=y, metal_fraction=float(metal.mean()),
            left_margin_fraction=float(metal[:margin].mean()),
            right_margin_fraction=float(metal[-margin:].mean()), noise_rgb=noise))
    for key in ('metal_fraction', 'left_margin_fraction', 'right_margin_fraction'):
        if np.mean([v[key] for v in witnesses]) < .75:
            raise ValueError('Unmarked closure lacks broad source metal after edge')
    local_witnesses = []
    if margin_sampling != 'broad':
        # The closer reference must also see exposed metal. A distant dark
        # port can otherwise inflate broad-window noise on both sides of a
        # printed stroke and hide continuing paper.
        for witness in witnesses:
            _, residual, noise, margin, _ = row_samples(witness['row'], local=True)
            metal = np.linalg.norm(residual, axis=1) <= noise
            local_witnesses.append(dict(row=witness['row'], noise_rgb=noise,
                fractions=[float(metal.mean()),float(metal[:margin].mean()),float(metal[-margin:].mean())]))
        if not any(min(v['fractions']) >= .75 for v in local_witnesses):
            raise ValueError('Unmarked closure lacks local source metal after edge')

    # A full-width printed stroke can mimic the first edge. Reject if source
    # paper margins reappear below it, including after a dark printed band.
    checked_rows = []
    for y in range(int(np.ceil(center))+7, h):
        try:
            _, residual, noise, margin, _ = row_samples(y)
        except ValueError:
            continue  # alpha fringe is not an opaque material witness
        signal = residual @ direction > noise
        if min(signal[:margin].mean(), signal[-margin:].mean()) >= .75:
            raise ValueError('Source paper margins continue below proposed closure')
        if margin_sampling != 'broad':
            _, residual, noise, margin, _ = row_samples(y, local=True)
            signal = residual @ direction > noise
            if min(signal[:margin].mean(), signal[-margin:].mean()) >= .75:
                raise ValueError('Source paper margins continue below proposed closure')
        checked_rows.append(y)
    if not checked_rows:
        raise ValueError('Unmarked closure has no independent lower continuation check')

    # The one-pixel sensor transition matches the existing marked-strip rule.
    # Remove an extra transition pixel only when it is quantization-identical
    # to independently measured metal; do not erode uncertain paper RGB.
    cutoff = np.floor(np.polyval(line, np.arange(w))+.5).astype(int)+1
    result = paper & (np.arange(h)[:, None] <= cutoff[None, :])
    terminal = cutoff.copy()
    active_columns = paper.any(axis=0)
    for _ in range(3):
        for y in np.unique(terminal[active_columns]):
            if not 0 <= y < h:
                raise ValueError('Unmarked closure falls outside source canvas')
            _, _, _, _, background = row_samples(int(y))
            xs = np.flatnonzero(paper[y] & (terminal == y))
            residual = a[y, xs, :3].astype(float)-background(xs)
            xs = xs[np.linalg.norm(residual, axis=1) <= 1.]
            result[y, xs] = False
            terminal[xs] -= 1
    if not result.any() or np.any(result & ~paper):
        raise ValueError('Unmarked closure cannot add paper outside measured sides')
    info = dict(proof, pixels=int(result.sum()), source_paper_end='independent_unmarked_source_closure',
        unmarked_source_closure=dict(edge, method='paper_margins_cross_edge_metal_closure_v1',
            source_rgba_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
            source_size=[w, h], post_edge_witnesses=witnesses,
            local_post_edge_witnesses=local_witnesses,
            preceding_margin_witness=preceding_witness,
            lower_continuation_checked_rows=checked_rows,
            removed_extrapolated_pixels=int(np.count_nonzero(paper & ~result)),
            mask_sha256=hashlib.sha256(result.tobytes()).hexdigest()))
    return result, info


def resolve_aligned_strip_end(a,paper,proof, *, after_distance=4):
    """Measure margins at one distance from a sloped source cross-edge.

    Integer column translations select original pixels without interpolation.
    All original source closure and later overprotection checks still apply.
    """
    if not proof.get('source_strip_edges'):
        raise ValueError('Aligned closure requires independently measured paper sides')
    line,_,_,edge=_source_cross_edge(a,paper,proof)
    h,w=paper.shape;xx=np.arange(w);center=edge['source_end_peak']
    shift=np.rint(np.polyval(line,xx)-center).astype(int)
    yy=np.arange(h)[:,None]+shift[None,:];valid=(yy>=0)&(yy<h)
    aligned=a[np.clip(yy,0,h-1),xx[None,:]].copy();aligned[~valid]=0
    aligned_paper=paper[np.clip(yy,0,h-1),xx[None,:]]&valid
    result,info=resolve_unmarked_strip_end(aligned,aligned_paper,proof,after_distance=after_distance)
    inverse=np.arange(h)[:,None]-shift[None,:];valid_inverse=(inverse>=0)&(inverse<h)
    mapped=result[np.clip(inverse,0,h-1),xx[None,:]]&valid_inverse&paper
    # Resolve only the end, preserving source-mask rows above the measured edge.
    active=np.flatnonzero(paper.any(axis=0))
    tail_start=max(0,int(np.floor(np.min(np.polyval(line,active))))-4)
    original=paper.copy();original[tail_start:]=mapped[tail_start:]
    if not original.any():raise ValueError('Aligned closure leaves no measured paper')
    info=dict(info,pixels=int(original.sum()),aligned_source_closure=dict(
        method='integer_column_source_edge_registration_v1',
        original_source_rgba_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
        original_cross_edge=edge,column_translations=shift.tolist(),
        tail_only_start_row=tail_start,
        source_rgb_interpolated=False,source_rgba_edited=False,
        removed_extrapolated_pixels=int((paper&~original).sum()),
        original_mask_sha256=hashlib.sha256(original.tobytes()).hexdigest()))
    return original,info


def resolve_material_strip_end(a, strip, proof):
    """Resolve a lighting material boundary without changing geometry rules.

    A fitted full-height strip is only a paper-location hint.  A real paper
    edge can cast a short shadow across the first post-edge sample rows, so
    search the first complete three-row metal witness window.  Every trial
    retains the original 75% coverage, noise, opacity and lower-continuation
    gates; in particular, the continuation scan is NOT moved past the window.
    """
    from .auto_material import trim_strip_to_source_end
    from .bottom_paper_audit import guard_bottom_paper_end
    try:
        result, info = trim_strip_to_source_end(a, strip, proof)
        # A numeric marker-based cutoff measures a local paper edge only.
        # The overprotection audit is rejection-only: unresolved means no
        # still-protected metal was proved, not that all released pixels were
        # positively certified. Only an independent full-width closure can
        # bypass the strict source-metal window search below.
        if (info.get('source_paper_end')=='independent_unmarked_source_closure'
                and info.get('unmarked_source_closure')):
            return result, dict(info, material_boundary_resolution='original_source_closure')
        if isinstance(info.get('source_paper_end'), (int,float)):
            original_error = ('Marker-based paper end is a local edge proposal; '
                'independent full-width source material closure is required')
        else:
            original_error = 'Original source closure did not resolve a physical paper end'
    except ValueError as strict_error:
        original_error = str(strict_error)
    try:
        _, _, _, edge = _source_cross_edge(a, strip, proof)
    except ValueError as edge_error:
        # No source transverse end means no released metal may be invented.
        # Keep this unresolved state visible; it is not a visual acceptance.
        return strip, dict(proof, material_boundary_resolution='unresolved_no_source_cross_edge',
            material_boundary_warning=str(edge_error), strict_closure_error=original_error)
    trials=[]
    # Four pixels exceed the source gradient's sigma=.8, truncate=4 support.
    for distance in range(4, max(4, a.shape[0]-int(np.ceil(edge['source_end_peak']))-2)):
        try:
            result, info = resolve_aligned_strip_end(a, strip, proof, after_distance=distance)
            result, info = guard_bottom_paper_end(a, result, info)
            return result, dict(info, material_boundary_resolution='source_aligned_metal_window',
                material_end_witness_search=dict(first_post_edge_distance=4,
                    selected_post_edge_distance=distance, consecutive_metal_rows=3,
                    minimum_mean_support=.75, earlier_trials=trials,
                    lower_continuation_scan_unchanged=True), strict_closure_error=original_error)
        except ValueError as error:
            trials.append(dict(post_edge_distance=distance, error=str(error)))
    # Persist the source evidence in stage error/analysis records. A paper-like
    # strip near the opaque canvas end may lack room for an independent metal
    # witness; that is uncertainty, not proof of a damaged image or paper
    # continuing outside the visible source.
    diagnostic=dict(source_size=[a.shape[1],a.shape[0]],source_cross_edge=edge,
        first_post_edge_distance=4,required_consecutive_metal_rows=3,
        post_edge_trials=trials,material_boundary_resolution='unresolved_no_source_metal_window')
    raise ValueError('Bottom paper material boundary unresolved: '+original_error+
        '; no independent post-edge metal window passed; source_evidence='+
        json.dumps(diagnostic,sort_keys=True))
