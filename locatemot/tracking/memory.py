"""TrackMemory helpers: anchor + EMA in raw feature space (T5)."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def init_memory(features: Dict[str, np.ndarray], conf: float):
    """Copy features into anchor + EMA slots."""
    anchor = {k: np.asarray(v, dtype=np.float32).copy() for k, v in features.items()}
    ema = {k: np.asarray(v, dtype=np.float32).copy() for k, v in features.items()}
    return anchor, ema


def update_ema(ema: Dict[str, np.ndarray], features: Dict[str, np.ndarray], alpha: float = 0.5):
    out = {}
    for k in ("pbd", "region"):
        if k in ema and k in features:
            out[k] = (1.0 - alpha) * ema[k] + alpha * np.asarray(features[k], dtype=np.float32)
    return {**ema, **out}


def merge_feature_dicts(*dicts, geom=None, gen=None):
    """Weighted raw-space combination (identity weights) for fallback."""
    out = {}
    for k in ("pbd", "region"):
        parts = [d.get(k) for d in dicts if d and d.get(k) is not None]
        if parts:
            out[k] = np.mean(parts, axis=0).astype(np.float32)
    if geom is not None:
        out["geom"] = np.asarray(geom, dtype=np.float32)
    if gen is not None:
        out["gen"] = float(gen)
    return out
