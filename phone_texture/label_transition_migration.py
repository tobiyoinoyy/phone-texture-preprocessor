"""DRAFT: migrate existing paper transitions; never infer a new source boundary.

Not installed or executed. Arrays are effective editable/visible domain masks.
Existing material certification, source identity and effect checks remain the
caller's responsibility. The support below describes the blend operation only,
not subsequent gamut processing or final RGB changes.
"""
import copy
import hashlib
import numpy as np
from scipy import ndimage


def _support(config, masks):
    names = config.get('domains', [])
    anchor = config.get('boundary_domain')
    sigma = float(config['sigma'])
    kind = config.get('kind', 'same_material')
    if (len(names) < 2 or len(set(names)) != len(names) or anchor not in names
            or any(name not in masks for name in names)
            or not np.isfinite(sigma) or sigma <= 0
            or kind not in ('same_material', 'source_material_transition')
            or (kind == 'source_material_transition' and (len(names) != 2 or sigma > 2))):
        raise ValueError('Invalid inherited correction blend')
    union = np.logical_or.reduce([masks[name] for name in names])
    primary = masks[anchor]
    other = union & ~primary
    seam = ((primary & ndimage.binary_dilation(other))
            | (other & ndimage.binary_dilation(primary)))
    if not seam.any():
        raise ValueError('Inherited correction blend has no actual adjacency')
    direct = union & (ndimage.distance_transform_edt(~seam) < 3*sigma)
    # scipy Gaussian's default truncate=4 uses a separable square footprint.
    # Include points outside union: adding another paper component there can
    # change the normalized convolution denominator as well as its numerator.
    radius = int(4*sigma+.5)
    read = ndimage.maximum_filter(direct, size=2*radius+1, mode='constant')
    return union, direct, read


def migrate_label_blends(blends, before, after, removed):
    """Conservatively rename/merge an already declared, single-peer transition.

    No declaration means no invented blend. Unsupported material combinations
    raise instead of returning a claim of complete transition coverage.
    """
    removed = set(removed)
    relevant = [i for i, config in enumerate(blends)
                if removed.intersection(config.get('domains', []))]
    proof = dict(method='inherit_existing_source_material_transition_only',
                 source_boundary_certified_here=False,
                 final_rgb_support_guaranteed=False)
    if not relevant:
        return copy.deepcopy(blends), dict(proof, status='no_inherited_transition',
            reconstructed_pairs=0, complete_transition_coverage_claimed=False)
    if 'printed_label' not in after:
        raise ValueError('Removed paper transition has no replacement domain')
    old_paper = np.logical_or.reduce([before[name] for name in removed])
    paper = after['printed_label']
    if not np.array_equal(old_paper, paper):
        raise ValueError('Replacement paper extends beyond inherited transition material')
    templates, peers, covered, old_supports, old_reads = [], set(), set(), [], []
    for index in relevant:
        config = blends[index]
        names = config.get('domains', [])
        old_names = removed.intersection(names)
        if config.get('kind') != 'source_material_transition' or len(names) != 2 or len(old_names) != 1:
            raise ValueError('Paper must not inherit a same-material or multi-material blend')
        old_name = next(iter(old_names))
        if old_name in covered:
            raise ValueError('Duplicate paper transition declaration is unsupported')
        peer = next(name for name in names if name != old_name)
        if peer not in after or not np.array_equal(before[peer], after[peer]):
            raise ValueError('Inherited paper peer changed during material refinement')
        _, direct, read = _support(config, before)
        if np.any(read & (paper & ~before[old_name])):
            raise ValueError('Merging paper components would change a Gaussian read footprint')
        if any(np.any(direct & prior_read) or np.any(prior_direct & read)
               for prior_direct, prior_read in zip(old_supports, old_reads)):
            raise ValueError('Inherited paper pairs interact through their shared peer')
        old_supports.append(direct)
        old_reads.append(read)
        peers.add(peer)
        covered.add(old_name)
        templates.append({key: value for key, value in config.items()
                          if key not in ('domains', 'boundary_domain')})
    if covered != removed or len(peers) != 1 or any(t != templates[0] for t in templates[1:]):
        raise ValueError('Paper migration requires complete, compatible declarations for one peer')
    peer = next(iter(peers))
    neighbors = {name for name, mask in after.items() if name != 'printed_label'
                 and np.any(ndimage.binary_dilation(paper) & mask)}
    if neighbors != {peer}:
        raise ValueError('Paper has unsupported additional or missing material adjacency')
    new = dict(templates[0], domains=['printed_label', peer], boundary_domain='printed_label')
    union, direct, read = _support(new, after)
    if not np.array_equal(direct, np.logical_or.reduce(old_supports)):
        raise ValueError('Replacement paper changes inherited direct transition support')
    # Existing blends execute sequentially. Keep unrelated ones verbatim,
    # but reject any interaction rather than make their order significant.
    for index, config in enumerate(blends):
        if index in relevant:
            continue
        other_union, other_direct, other_read = _support(config, after)
        if config.get('kind', 'same_material') == 'same_material':
            # Its internal gamut envelope can spread beyond its direct seam.
            collision = np.any(union & other_union)
        else:
            collision = np.any(direct & other_read) or np.any(other_direct & read)
        if collision:
            raise ValueError('Inherited paper transition interacts with another blend')
    result = []
    for index, config in enumerate(blends):
        if index == relevant[0]:
            result.append(new)
        elif index not in relevant:
            result.append(copy.deepcopy(config))
    return result, dict(proof, status='inherited_pair_migrated',
        old_paper_components=sorted(removed), peer=peer,
        reconstructed_pairs=1, sigma=new['sigma'],
        direct_support_pixels=int(direct.sum()),
        direct_support_shape=list(direct.shape),
        direct_support_sha256=hashlib.sha256(direct.astype('uint8').tobytes()).hexdigest(),
        all_final_effect_gates_still_required=True)


def rebuild_label_blends(blends, after, removed, reason):
    """Fallback when exact migration is rejected: rebuild paper transitions.

    Unrelated blends are kept verbatim. Blends touching removed paper
    components are dropped, and the refined printed_label gets one narrow
    source-material transition per adjacent domain, as propose_job declares
    for paper components. Only additive fields are blended; RGB is untouched.
    """
    removed = set(removed)
    result = [copy.deepcopy(config) for config in blends
              if not removed.intersection(config.get('domains', []))]
    paper = after.get('printed_label')
    peers = []
    if paper is not None and paper.any():
        grown = ndimage.binary_dilation(paper)
        peers = sorted(name for name, mask in after.items()
                       if name != 'printed_label' and np.any(grown & mask))
        for peer in peers:
            result.append(dict(domains=['printed_label', peer], boundary_domain='printed_label',
                               sigma=1.25, kind='source_material_transition'))
    return result, dict(method='rebuild_paper_transitions_on_refined_masks',
        status='rebuilt_after_exact_migration_rejected', exact_migration_error=reason,
        removed_paper_components=sorted(removed), peers=peers, reconstructed_pairs=len(peers),
        source_boundary_certified_here=False, all_final_effect_gates_still_required=True)
