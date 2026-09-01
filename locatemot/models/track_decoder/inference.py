"""Inference-time assignment with per-track NO_MATCH dummies."""
from __future__ import annotations

import numpy as np

from locatemot.evaluation.assignment import assign_tracks_to_candidates


def assign_batch(match_logits, no_match_logits, ref_mask=None):
    """match_logits [B,M,N], no_match_logits [B,M]; returns list of per-sample
    [(track_i, 'candidate:j' | 'NO_MATCH')]."""
    out = []
    for b in range(match_logits.shape[0]):
        m = match_logits[b].detach().cpu().numpy()
        nm = no_match_logits[b].detach().cpu().numpy()
        out.append(assign_tracks_to_candidates(m, nm))
    return out
