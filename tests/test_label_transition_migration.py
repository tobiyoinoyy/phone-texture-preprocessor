"""DRAFT, NOT RUN. Synthetic mask tests require root's serial test window."""
import copy
import numpy as np
import pytest
from phone_texture.label_transition_migration import migrate_label_blends


def pair(paper, peer='metal', **updates):
    config = dict(domains=[paper, peer], boundary_domain=paper,
                  sigma=1.25, kind='source_material_transition')
    config.update(updates)
    return config


def fixture(components=1):
    paper = np.zeros((60, 220), bool)
    first = paper.copy(); first[10:45, 35:50] = True
    masks = {'label_component_0': first}
    if components == 2:
        second = paper.copy(); second[10:45, 140:155] = True
        masks['label_component_1'] = second
    paper = np.logical_or.reduce(list(masks.values()))
    masks['metal'] = ~paper
    after = {'printed_label': paper.copy(), 'metal': masks['metal'].copy()}
    removed = [name for name in masks if name.startswith('label_')]
    return masks, after, removed, [pair(name) for name in removed]


@pytest.mark.parametrize('count', [1, 2])
def test_existing_paper_pairs_merge_without_changing_materials(count):
    before, after, removed, blends = fixture(count)
    frozen = copy.deepcopy((before, after, blends))
    actual, proof = migrate_label_blends(blends, before, after, removed)
    assert actual == [pair('printed_label')]
    assert proof['old_paper_components'] == removed
    assert proof['reconstructed_pairs'] == 1
    assert proof['all_final_effect_gates_still_required'] is True
    assert proof['source_boundary_certified_here'] is False
    assert proof['final_rgb_support_guaranteed'] is False
    for current, old in zip((before, after), frozen[:2]):
        for name in current:
            np.testing.assert_array_equal(current[name], old[name])
    assert blends == frozen[2]


def test_unrelated_same_material_blend_is_preserved_verbatim_and_in_order():
    before, after, removed, blends = fixture()
    # These two independent materials do not share the paper/metal union.
    left = np.zeros((60, 220), bool); left[:, 185:200] = True
    right = np.zeros((60, 220), bool); right[:, 200:] = True
    for masks in (before, after):
        masks['metal'] &= ~(left | right)
        masks.update(other_a=left, other_b=right)
    unrelated = dict(domains=['other_a', 'other_b'], boundary_domain='other_a', sigma=3.)
    actual, _ = migrate_label_blends([unrelated] + blends, before, after, removed)
    assert actual == [unrelated, pair('printed_label')]


def test_missing_prior_declaration_never_becomes_new_sensor_evidence():
    before, after, removed, _ = fixture()
    actual, proof = migrate_label_blends([], before, after, removed)
    assert actual == []
    assert proof['status'] == 'no_inherited_transition'
    assert proof['complete_transition_coverage_claimed'] is False


def test_paper_cannot_be_migrated_into_same_material_smoothing():
    before, after, removed, blends = fixture()
    blends[0].pop('kind')
    with pytest.raises(ValueError, match='same-material'):
        migrate_label_blends(blends, before, after, removed)


def test_extended_paper_cannot_inherit_old_boundary_evidence():
    before, after, removed, blends = fixture()
    after['printed_label'][10:45, 50] = True
    after['metal'][10:45, 50] = False
    with pytest.raises(ValueError, match='extends beyond'):
        migrate_label_blends(blends, before, after, removed)


def test_one_paper_with_two_neighbors_rejects_even_if_only_one_pair_declared():
    before, after, removed, blends = fixture()
    hardware = np.zeros((60, 220), bool); hardware[20:35, 50:58] = True
    for masks in (before, after):
        masks['metal'] &= ~hardware
        masks['hardware'] = hardware
    with pytest.raises(ValueError, match='additional or missing material adjacency'):
        migrate_label_blends(blends, before, after, removed)


def test_declared_but_nonadjacent_pair_is_rejected():
    before, after, removed, blends = fixture()
    # A real empty gutter, rather than renaming an adjacent material.
    for masks in (before, after):
        masks['metal'][8:47, 33:52] = False
    with pytest.raises(ValueError, match='no actual adjacency'):
        migrate_label_blends(blends, before, after, removed)


def test_incompatible_component_sigmas_are_not_silently_reselected():
    before, after, removed, blends = fixture(2)
    blends[1]['sigma'] = 1.
    with pytest.raises(ValueError, match='compatible declarations'):
        migrate_label_blends(blends, before, after, removed)


def test_duplicate_old_pair_is_not_silently_coalesced():
    before, after, removed, blends = fixture()
    with pytest.raises(ValueError, match='Duplicate paper transition'):
        migrate_label_blends(blends + blends, before, after, removed)


def test_nearby_disconnected_paper_components_are_not_silently_coupled():
    before, after, removed, blends = fixture()
    second = np.zeros((60, 220), bool); second[10:45, 52:59] = True
    before['label_component_1'] = second
    before['metal'] &= ~second
    after['metal'] &= ~second
    after['printed_label'] |= second
    with pytest.raises(ValueError, match='Gaussian read footprint'):
        migrate_label_blends(blends + [pair('label_component_1')], before, after,
                             removed + ['label_component_1'])


def test_two_sequential_transition_footprints_must_not_interact():
    before, after, removed, blends = fixture()
    # Separate antenna never touches paper, but its smoothing neighborhood
    # overlaps the paper transition through the intervening metal.
    antenna = np.zeros((60, 220), bool); antenna[10:45, 55:60] = True
    for masks in (before, after):
        masks['metal'] &= ~antenna
        masks['antenna'] = antenna
    with pytest.raises(ValueError, match='interacts with another blend'):
        migrate_label_blends(blends + [pair('antenna')], before, after, removed)


def test_separate_paper_bodies_cannot_hide_sequential_interaction_in_metal():
    before, after, removed, blends = fixture()
    # Paper 2 begins beyond paper 1's read footprint. Their direct/read
    # supports still intersect in the shared metal between the two papers.
    second = np.zeros((60, 220), bool); second[10:45, 60:75] = True
    before['label_component_1'] = second
    before['metal'] &= ~second
    after['metal'] &= ~second
    after['printed_label'] |= second
    with pytest.raises(ValueError, match='interact through their shared peer'):
        migrate_label_blends(blends + [pair('label_component_1')], before, after,
                             removed + ['label_component_1'])


def test_shared_same_material_envelope_is_explicitly_unsupported():
    before, after, removed, blends = fixture()
    other = np.zeros((60, 220), bool); other[:, 190:] = True
    for masks in (before, after):
        masks['metal'] &= ~other
        masks['other'] = other
    same_material = dict(domains=['metal', 'other'], boundary_domain='other', sigma=3.)
    with pytest.raises(ValueError, match='interacts with another blend'):
        migrate_label_blends(blends + [same_material], before, after, removed)
