"""Tests for per-track NO_MATCH dummies in one-to-one assignment."""
import numpy as np

from locatemot.evaluation.assignment import (
    assign_tracks_to_candidates,
    build_assignment_cost,
)


def test_three_tracks_all_no_match():
    match = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    no_match = np.array([2.0, 2.0, 2.0])
    out = assign_tracks_to_candidates(match, no_match)
    assert all(tag == "NO_MATCH" for _, tag in out)


def test_two_no_match_one_real_match():
    match = np.array([
        [5.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    no_match = np.array([0.0, 3.0, 3.0])
    out = assign_tracks_to_candidates(match, no_match)
    by_track = dict(out)
    assert by_track[0].startswith("candidate:0")
    assert by_track[1] == "NO_MATCH"
    assert by_track[2] == "NO_MATCH"


def test_candidate_not_shared():
    match = np.array([
        [5.0, 0.1],
        [0.2, 0.1],
    ])
    no_match = np.array([0.0, 0.5])
    out = assign_tracks_to_candidates(match, no_match)
    cands = [tag for _, tag in out if tag.startswith("candidate")]
    assert len(cands) == 1
    by_track = dict(out)
    assert by_track[0].startswith("candidate:0")
    assert by_track[1] == "NO_MATCH"


def test_each_track_uses_own_dummy_only():
    match = np.zeros((3, 2))
    no_match = np.array([1.0, 0.0, 0.0])
    cost = build_assignment_cost(match, no_match)
    # dummy columns are 2,3,4; only column 2 has low cost for track 0
    assert cost[0, 2] == -1.0
    assert cost[0, 3] == 1e6
    assert cost[1, 3] == 0.0
    assert cost[1, 2] == 1e6
    assert cost[2, 4] == 0.0
    out = assign_tracks_to_candidates(match, no_match)
    by_track = dict(out)
    assert by_track[0] == "NO_MATCH"
    assert by_track[1].startswith("candidate:")
    assert by_track[2].startswith("candidate:")
