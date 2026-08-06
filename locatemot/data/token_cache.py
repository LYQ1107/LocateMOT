"""Binary ObjectToken cache with atomic writes and resume markers."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Dict, Optional

import numpy as np
import torch
from safetensors.torch import load_file, save_file


def cache_key(dataset: str, video_id: str, frame: int, protocol: str) -> str:
    return f"{dataset}/{video_id}/{int(frame):05d}/{protocol}"


def _paths(root: str, key: str):
    base = os.path.join(root, key)
    return base + ".safetensors", base + ".meta.json", base + ".complete"


def exists(root: str, key: str) -> bool:
    _, _, complete = _paths(root, key)
    return os.path.exists(complete)


def write_frame_cache(
    root: str,
    key: str,
    features: Dict[str, np.ndarray],
    meta: dict,
) -> None:
    feat_path, meta_path, complete_path = _paths(root, key)
    os.makedirs(os.path.dirname(feat_path), exist_ok=True)
    tensors = {
        name: torch.from_numpy(arr.astype(np.float16))
        for name, arr in features.items()
    }
    tmp_feat = feat_path + ".tmp"
    tmp_meta = meta_path + ".tmp"
    save_file(tensors, tmp_feat)
    with open(tmp_meta, "w") as f:
        json.dump(meta, f, ensure_ascii=False)
    os.replace(tmp_feat, feat_path)
    os.replace(tmp_meta, meta_path)
    with open(complete_path, "w") as f:
        f.write("ok\n")


def read_frame_cache(root: str, key: str) -> Optional[Dict]:
    return _read_frame_cache_cached(root, key)


@lru_cache(maxsize=16384)
def _read_frame_cache_cached(root: str, key: str) -> Optional[Dict]:
    feat_path, meta_path, _ = _paths(root, key)
    if not os.path.exists(feat_path) or not os.path.exists(meta_path):
        return None
    feats = load_file(feat_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return {"features": feats, "meta": meta}
