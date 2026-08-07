"""Shared one-to-one assignment helpers for T0-T6."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(refs, curs):
    """refs [M,4], curs [N,4] xyxy -> [M,N] IoU."""
    r = refs[:, None, :]
    c = curs[None, :, :]
    ix1 = np.maximum(r[:, :, 0], c[:, :, 0])
    iy1 = np.maximum(r[:, :, 1], c[:, :, 1])
    ix2 = np.minimum(r[:, :, 2], c[:, :, 2])
    iy2 = np.minimum(r[:, :, 3], c[:, :, 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    ar = np.maximum(0.0, r[:, :, 2] - r[:, :, 0]) * np.maximum(0.0, r[:, :, 3] - r[:, :, 1])
    ac = np.maximum(0.0, c[:, :, 2] - c[:, :, 0]) * np.maximum(0.0, c[:, :, 3] - c[:, :, 1])
    union = ar + ac - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def center_dist_matrix(refs, curs):
    rc = (refs[:, :2] + refs[:, 2:]) / 2
    cc = (curs[:, :2] + curs[:, 2:]) / 2
    d = np.sqrt(((rc[:, None, :] - cc[None, :, :]) ** 2).sum(-1))
    return d


def hungarian_with_no_match(match: np.ndarray, no_match: np.ndarray):
    """Returns list of (track_idx, 'candidate:j' | 'NO_MATCH')."""
    from locatemot.evaluation.assignment import assign_tracks_to_candidates
    return assign_tracks_to_candidates(np.asarray(match), np.asarray(no_match))


def hungarian_max(cost: np.ndarray, threshold: float):
    """Maximization assignment; entries below threshold are rejected."""
    m, n = cost.shape
    if m == 0 or n == 0:
        return []
    rows, cols = linear_sum_assignment(-cost)
    out = []
    for r, c in zip(rows, cols):
        if cost[r, c] >= threshold:
            out.append((int(r), int(c)))
    return out
