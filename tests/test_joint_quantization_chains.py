"""Joint interval feasibility, including constraints on raw-under-limit edges.

Apply the companion module patch before scheduling this suite. These tests
were prepared statically and must run only in the parent's compute window.
"""
import math

import numpy as np
import pytest

from phone_texture import local_effect_review as local
from phone_texture.lighting_review import lightness


def oriented(values, axis):
    result = np.asarray(values, float)[:, None]
    return result if axis == 0 else result.T


def intervals(centers, radius, axis):
    centers = oriented(centers, axis)
    return centers-radius, centers+radius, np.ones(centers.shape, bool)


@pytest.mark.parametrize('axis', [0, 1])
def test_three_nodes_can_pass_every_pair_but_have_no_joint_assignment(axis):
    low, high, witness = intervals([0., 1.25, 2.5], .125, axis)
    # Every edge alone has minimum distance exactly 1; the endpoints must
    # nevertheless differ by >=2.25 across only two <=1 edges. This oracle
    # follows from the triangle inequality, independent of the propagation.
    assert 1.25-2*.125 == 1.
    assert 2.5-2*.125 > 2*1.
    assert local._joint_chain_intervals(low, high, witness, axis, 1.) is None


@pytest.mark.parametrize('axis', [0, 1])
def test_raw_under_limit_middle_edge_must_join_the_constraints(axis):
    low, high, witness = intervals([0., 1.25, 2.25, 3.5], .125, axis)
    centers = oriented([0., 1.25, 2.25, 3.5], axis)
    measured = np.abs(np.diff(centers, axis=axis)).ravel()
    assert measured.tolist() == [1.25, 1., 1.25]
    # Both exceeded pairs have a feasible assignment. Dropping the middle
    # edge would falsely allow those assignments to contradict each other.
    for nodes in ([0., 1.25], [2.25, 3.5]):
        lo, hi, active = intervals(nodes, .125, axis)
        assert local._joint_chain_intervals(lo, hi, active, axis, 1.) is not None
    assert 3.5-2*.125 > 3*1.
    assert local._joint_chain_intervals(low, high, witness, axis, 1.) is None


@pytest.mark.parametrize('axis', [0, 1])
def test_joint_alternating_chain_has_one_consistent_assignment(axis):
    low, high, witness = intervals([0., 1.25, 0., 1.25], .125, axis)
    # [.125,1.125,.125,1.125] is one concrete joint solution.
    concrete = oriented([.125, 1.125, .125, 1.125], axis)
    assert np.all((low <= concrete)&(concrete <= high))
    assert np.max(abs(np.diff(concrete,axis=axis))) == 1.
    report = local._joint_chain_intervals(low, high, witness, axis, 1.)
    assert report['feasible'] is True
    assert report['checked_pair_count'] == 3
    assert report['connected_chain_count'] == 1
    assert report['all_witness_edges_checked'] is True
    assert report['comparison'] == 'strict_no_epsilon'


@pytest.mark.parametrize('axis', [0, 1])
def test_only_an_actual_witness_gap_resets_a_chain(axis):
    low, high, witness = intervals([0., 1.25, 2.25, 3.5], .125, axis)
    if axis == 0:
        witness[2, 0] = False
    else:
        witness[0, 2] = False
    report = local._joint_chain_intervals(low, high, witness, axis, 1.)
    assert report['checked_pair_count'] == 1
    assert report['connected_chain_count'] == 1


@pytest.mark.parametrize('axis', [0, 1])
def test_no_epsilon_at_an_exact_binary_interval_boundary(axis):
    low = high = oriented([0., 1.], axis)
    witness = np.ones(low.shape, bool)
    assert local._joint_chain_intervals(low, high, witness, axis, 1.) is not None
    assert local._joint_chain_intervals(low, high, witness, axis,
                                        math.nextafter(1., -math.inf)) is None


def image_chain(values, axis):
    source = np.full((len(values), 24, 4), 100, np.uint8)
    source[..., 3] = 255
    candidate = source.copy()
    candidate[:, 0, :3] = np.asarray(values, np.uint8)[:, None]
    witness = np.ones(source.shape[:2], bool)
    if axis == 1:
        source, candidate, witness = (np.swapaxes(a, 0, 1).copy()
                                      for a in (source, candidate, witness))
    return source, candidate, witness


@pytest.mark.parametrize('axis', [0, 1])
def test_real_8bit_staircase_is_not_a_joint_quantization_explanation(axis):
    source, candidate, witness = image_chain([100, 104, 108], axis)
    source_L = lightness(source[..., :3])
    low = lightness(np.clip(candidate[..., :3].astype(float)-.5,0,255))-source_L
    high = lightness(np.clip(candidate[..., :3].astype(float)+.5,0,255))-source_L
    low_chain = low[:, 0] if axis == 0 else low[0]
    high_chain = high[:, 0] if axis == 0 else high[0]
    # This is the old per-edge necessary condition; every adjacent pair passes.
    assert np.all(low_chain[1:]-high_chain[:-1] <= 1.25)
    # Independent endpoint lower bound proves no joint three-pixel solution.
    assert low_chain[-1]-high_chain[0] > 2*1.25
    assert local.quantization_pixel_review(source,candidate,witness,axis,1.25) is None


@pytest.mark.parametrize('axis', [0, 1])
def test_real_joint_feasible_candidate_remains_review_only_and_unchanged(axis):
    source, candidate, witness = image_chain([100, 104, 100], axis)
    before = [a.copy() for a in (source, candidate, witness)]
    result = local.quantization_pixel_review(source,candidate,witness,axis,1.25)
    assert result is not None
    assert result['joint_chain_feasibility']['checked_pair_count'] == 48
    assert result['joint_chain_feasibility']['connected_chain_count'] == 24
    assert result['original_threshold_passed'] is False
    assert result['rounding_cause_proven'] is False
    assert result['quality_checks_complete'] is False
    assert result['source_pixels_held_fixed'] is True
    assert result['candidate_RGB_uncertainty_per_channel'] == .5
    for old, actual in zip(before,(source,candidate,witness)):
        np.testing.assert_array_equal(old,actual)


def test_one_infeasible_chain_rejects_among_independent_feasible_chains():
    low = np.array([[0.,0.],[1.25,1.25],[2.5,0.]])-.125
    high = low+.25
    assert local._joint_chain_intervals(low,high,np.ones(low.shape,bool),0,1.) is None


def test_no_original_overlimit_edges_does_not_create_a_warning():
    source,candidate,witness = image_chain([100,101,102],0)
    assert local.quantization_pixel_review(source,candidate,witness,0,1.25) is None
