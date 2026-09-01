"""Causal RMOT-only fragment repair utilities for Stage L18."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cosine(a, b):
    a = np.asarray(a, np.float32).reshape(-1)
    b = np.asarray(b, np.float32).reshape(-1)
    return float(np.dot(a, b) / max(1e-6, np.linalg.norm(a) * np.linalg.norm(b)))


def box_iou(a, b):
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(1e-6, aa + bb - inter)


@dataclass
class Fragment:
    identity: int
    last_frame: int
    box: np.ndarray
    appearance: np.ndarray
    track_embedding: np.ndarray
    source: int


def repair_fragment(active: list[Fragment], frame: int, membership: float,
                    appearance: np.ndarray,
                    track_embedding: np.ndarray, box: np.ndarray,
                    source: int, max_gap: int = 12,
                    threshold: float = 0.62) -> tuple[int, str, float]:
    """Return existing/new identity and a causal decision label.

    This helper is deliberately conservative: it may return ``new`` and never
    forces a merge.  The evaluator keeps repair IDs in an RMOT-only namespace.
    """
    active[:] = [item for item in active
                 if 0 <= int(frame) - int(item.last_frame) <= max_gap]
    candidates = []
    for item in active:
        gap = int(frame) - int(item.last_frame)
        # A track identity can be used at most once per frame.  This keeps
        # repair from collapsing two simultaneous detections into one ID.
        if gap <= 0 or gap > max_gap:
            continue
        appearance_score = cosine(appearance, item.appearance)
        track_score = cosine(track_embedding, item.track_embedding)
        geometry_score = box_iou(box, item.box)
        source_penalty = 0.05 if int(source) != int(item.source) else 0.0
        score = (0.45 * float(membership) + 0.25 * appearance_score +
                 0.20 * track_score + 0.15 * geometry_score - source_penalty)
        candidates.append((score, item))
    if candidates:
        score, best = max(candidates, key=lambda value: value[0])
        if score >= threshold:
            best.last_frame = int(frame)
            best.box = np.asarray(box, np.float32)
            best.appearance = np.asarray(appearance, np.float32)
            best.track_embedding = np.asarray(track_embedding, np.float32)
            return best.identity, "merge", float(score)
    next_id = max([item.identity for item in active], default=0) + 1
    active.append(Fragment(next_id, int(frame), np.asarray(box, np.float32),
                           np.asarray(appearance, np.float32),
                           np.asarray(track_embedding, np.float32), int(source)))
    return next_id, "new", float(max([x[0] for x in candidates], default=0.0))
