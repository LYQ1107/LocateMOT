"""One-to-one track-candidate assignment with per-track NO_MATCH dummies.

Matrix shape: [M, N+M].
- Columns 0..N-1: real current candidates.
- Column N+i: NO_MATCH dummy owned by track i (usable only by track i).
This allows any number of tracks to be simultaneously NO_MATCH while keeping
each real candidate matched to at most one track.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def build_assignment_cost(
    match_logits: np.ndarray,
    no_match_logits: np.ndarray,
    large: float = 1e6,
) -> np.ndarray:
    match_logits = np.asarray(match_logits, dtype=np.float64)
    no_match_logits = np.asarray(no_match_logits, dtype=np.float64).reshape(-1)
    m, n = match_logits.shape
    assert m == no_match_logits.shape[0]
    cost = -match_logits  # higher logit = lower cost
    dummies = np.full((m, m), large, dtype=np.float64)
    for i in range(m):
        dummies[i, i] = -float(no_match_logits[i])
    return np.concatenate([cost, dummies], axis=1)


def assign_tracks_to_candidates(
    match_logits: np.ndarray,
    no_match_logits: np.ndarray,
) -> List[Tuple[int, str]]:
    """Returns [(track_i, 'candidate:<j>' | 'NO_MATCH')] for each track."""
    cost = build_assignment_cost(match_logits, no_match_logits)
    m, total = cost.shape
    n = total - m
    rows, cols = linear_sum_assignment(cost)
    assignments: List[Tuple[int, str]] = []
    for r, c in zip(rows, cols):
        if c < n:
            assignments.append((int(r), f"candidate:{int(c)}"))
        else:
            assignments.append((int(r), "NO_MATCH"))
    assignments.sort(key=lambda x: x[0])
    return assignments
