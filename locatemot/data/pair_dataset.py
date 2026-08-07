"""PyTorch dataset for two-frame pair records backed by the binary token cache."""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from locatemot.data.token_cache import read_frame_cache
from locatemot.models.track_decoder.features import category_hash_embedding

MAX_REF = 8
MAX_CUR = 32
EMBED_DIM = 32


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class PairDataset(Dataset):
    def __init__(self, records: List[dict], cache_root: str, seed: int = 20260806,
                 cache_items: bool = True):
        self.records = records
        self.cache_root = cache_root
        self.rng = np.random.RandomState(seed)
        self._cache = {} if cache_items else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]
        rec = self.records[idx]
        ref = read_frame_cache(self.cache_root, rec["reference_token_id"])
        cur = read_frame_cache(self.cache_root, rec["current_token_id"])
        if ref is None or cur is None:
            return self._empty(rec)
        ref_f, ref_m = ref["features"], ref["meta"]
        cur_f, cur_m = cur["features"], cur["meta"]
        n_cur = min(cur_m.get("candidate_count", 0), MAX_CUR)
        if n_cur == 0:
            return self._empty(rec)

        # reference tokens
        ref_entries = []
        for t in rec["reference_targets"][:MAX_REF]:
            tid = t["track_id"]
            c_idx = t.get("reference_candidate_index")
            if c_idx is not None and c_idx < _count(ref_f):
                ref_entries.append(_token_feature(ref_f, int(c_idx)))
            else:
                crop = _crop_feature(ref_f, ref_m, tid)
                ref_entries.append(crop)
        if not ref_entries:
            return self._empty(rec)

        cur_entries = [_token_feature(cur_f, i) for i in range(n_cur)]
        proto = rec["protocol"]
        cat_embed = category_hash_embedding(proto, EMBED_DIM)

        M = len(ref_entries)
        N = len(cur_entries)
        match_targets = -np.ones(M, dtype=np.int64)
        no_match_targets = np.zeros(M, dtype=np.float32)
        candidate_missing = np.zeros(M, dtype=bool)
        visible = np.ones(M, dtype=bool)
        labels = np.zeros((M, N), dtype=np.float32)
        gt_iou = np.zeros(M, dtype=np.float32)

        ref_ids = [t["track_id"] for t in rec["reference_targets"][:MAX_REF]]
        for t in rec["assignment_targets"]:
            if t["track_id"] not in ref_ids:
                continue
            mi = ref_ids.index(t["track_id"])
            ci = t["candidate_index"]
            if ci < N:
                match_targets[mi] = ci
                labels[mi, ci] = 1.0
                gt_iou[mi] = _iou(
                    rec["reference_boxes"][mi],
                    cur_entries[ci]["box"],
                )
        for tid in rec["no_match_targets"]:
            if tid in ref_ids:
                no_match_targets[ref_ids.index(tid)] = 1.0
        for tid in rec["candidate_missing_targets"]:
            if tid in ref_ids:
                candidate_missing[ref_ids.index(tid)] = True
                visible[ref_ids.index(tid)] = False

        out = {
            "ref_pbd": torch.stack([e["pbd"] for e in ref_entries]),
            "ref_pbd_be": torch.stack([e["pbd_be"] for e in ref_entries]),
            "ref_region": torch.stack([e["region"] for e in ref_entries]),
            "ref_geom": torch.stack([e["geom"] for e in ref_entries]),
            "ref_gen": torch.tensor([e["gen"] for e in ref_entries], dtype=torch.float32),
            "ref_cat": cat_embed.unsqueeze(0).expand(M, EMBED_DIM).clone(),
            "ref_mask": torch.ones(M, dtype=torch.bool),
            "cur_pbd": torch.stack([e["pbd"] for e in cur_entries]),
            "cur_pbd_be": torch.stack([e["pbd_be"] for e in cur_entries]),
            "cur_region": torch.stack([e["region"] for e in cur_entries]),
            "cur_geom": torch.stack([e["geom"] for e in cur_entries]),
            "cur_gen": torch.tensor([e["gen"] for e in cur_entries], dtype=torch.float32),
            "cur_cat": cat_embed.unsqueeze(0).expand(N, EMBED_DIM).clone(),
            "cur_mask": torch.ones(N, dtype=torch.bool),
            "match_targets": torch.from_numpy(match_targets),
            "no_match_targets": torch.from_numpy(no_match_targets),
            "candidate_missing": torch.from_numpy(candidate_missing),
            "visible": torch.from_numpy(visible),
            "labels": torch.from_numpy(labels),
            "gt_iou": torch.from_numpy(gt_iou),
            "ref_boxes": torch.tensor([e["box"] for e in ref_entries], dtype=torch.float32),
            "cur_boxes": torch.tensor([e["box"] for e in cur_entries], dtype=torch.float32),
            "gap": torch.tensor([rec["temporal_gap"]], dtype=torch.float32),
            "dataset": rec["dataset"],
            "video_id": rec["video_id"],
            "protocol": rec["protocol"],
            "split": rec["split"],
        }
        if self._cache is not None:
            self._cache[idx] = out
        return out

    def _empty(self, rec):
        return {
            "ref_pbd": torch.zeros((1, 2048)), "ref_pbd_be": torch.zeros((1, 2048)),
            "ref_region": torch.zeros((1, 4608)),
            "ref_geom": torch.zeros((1, 5)), "ref_gen": torch.zeros(1),
            "ref_cat": torch.zeros((1, EMBED_DIM)), "ref_mask": torch.zeros(1, dtype=torch.bool),
            "cur_pbd": torch.zeros((1, 2048)), "cur_pbd_be": torch.zeros((1, 2048)),
            "cur_region": torch.zeros((1, 4608)),
            "cur_geom": torch.zeros((1, 5)), "cur_gen": torch.zeros(1),
            "cur_cat": torch.zeros((1, EMBED_DIM)), "cur_mask": torch.zeros(1, dtype=torch.bool),
            "match_targets": torch.full((1,), -1, dtype=torch.long),
            "no_match_targets": torch.zeros(1), "candidate_missing": torch.ones(1, dtype=torch.bool),
            "visible": torch.zeros(1, dtype=torch.bool),
            "labels": torch.zeros((1, 1)), "gt_iou": torch.zeros(1),
            "ref_boxes": torch.zeros((1, 4)), "cur_boxes": torch.zeros((1, 4)),
            "gap": torch.tensor([rec["temporal_gap"]], dtype=torch.float32),
            "dataset": rec["dataset"], "video_id": rec["video_id"],
            "protocol": rec["protocol"], "split": rec["split"],
        }


class PrecomputedPairSet:
    """All pair samples padded to fixed (8, 32) and stacked once; training then
    only slices precomputed tensors, removing per-sample feature overhead."""

    TENSOR_KEYS = [
        "ref_pbd", "ref_pbd_be", "ref_region", "ref_geom", "ref_gen", "ref_cat", "ref_mask", "ref_boxes",
        "cur_pbd", "cur_pbd_be", "cur_region", "cur_geom", "cur_gen", "cur_cat", "cur_mask", "cur_boxes",
        "match_targets", "no_match_targets", "candidate_missing", "visible", "labels",
        "gt_iou", "gap",
    ]

    def __init__(
        self,
        ds: PairDataset,
        max_ref: int = MAX_REF,
        max_cur: int = MAX_CUR,
        cache_path: str = None,
    ):
        self.max_ref = max_ref
        self.max_cur = max_cur
        self.metas = [] if ds is None else [ds.records[i]["split"] for i in range(len(ds))]
        if cache_path and os.path.exists(cache_path):
            self._tensors = torch.load(cache_path, map_location="cpu", weights_only=False)
        else:
            self._tensors = {k: [] for k in self.TENSOR_KEYS}
            for i in range(len(ds)):
                item = ds[i]
                for k in self.TENSOR_KEYS:
                    ref_side = k.startswith("ref") or k in (
                        "match_targets", "no_match_targets", "candidate_missing",
                        "visible", "gt_iou", "labels",
                    )
                    n = max_ref if ref_side else max_cur
                    self._tensors[k].append(_pad_item(item[k], n, max_cur, k))
            self._tensors = {k: torch.stack(v) for k, v in self._tensors.items()}
            if cache_path:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                torch.save(self._tensors, cache_path)
        # ensure every sample has at least one unmasked key so attention does
        # not produce NaN; loss still excludes these via candidate_missing/ref_mask
        for b in range(self._tensors["ref_mask"].shape[0]):
            if not self._tensors["ref_mask"][b].any():
                self._tensors["ref_mask"][b, 0] = True
            if not self._tensors["cur_mask"][b].any():
                self._tensors["cur_mask"][b, 0] = True

    def __len__(self):
        return self._tensors["ref_mask"].shape[0]

    def batch(self, indices):
        idx = torch.as_tensor(indices, dtype=torch.long)
        return {k: t[idx] for k, t in self._tensors.items()}


def _pad_item(t, n, max_cur, key):
    if key == "labels":
        m, n2 = t.shape
        out = torch.zeros((n, max_cur), dtype=t.dtype)
        out[: min(n, m), : min(max_cur, n2)] = t[:n, :max_cur]
        return out
    if t.dim() == 1:
        out = torch.zeros(n, dtype=t.dtype)
        out[: min(n, t.shape[0])] = t[:n]
        return out
    if t.dim() == 2:
        m, d = t.shape
        out = torch.zeros((n, d), dtype=t.dtype)
        out[: min(n, m)] = t[:n]
        return out
    if t.dim() == 3:
        m, n2, d = t.shape
        out = torch.zeros((n, max_cur, d), dtype=t.dtype)
        out[: min(n, m), : min(max_cur, n2)] = t[:n, :max_cur]
        return out
    raise ValueError(f"unsupported dim {t.dim()}")


def _count(features) -> int:
    if "pbd_coord_mean_last" in features:
        return int(features["pbd_coord_mean_last"].shape[0])
    return 0


def _token_feature(f, i) -> dict:
    def arr(name, default_dim):
        if name in f and int(f[name].shape[0]) > i:
            return f[name][i].float()
        return torch.zeros(default_dim)
    return {
        "pbd": arr("pbd_coord_mean_last", 2048),
        "pbd_be": arr("pbd_box_end_last", 2048),
        "region": arr("region", 4608),
        "geom": arr("geometry", 5) if "geometry" in f and f["geometry"].shape[0] > i else torch.zeros(5),
        "gen": float(f["gen_score"][i]) if "gen_score" in f and f["gen_score"].shape[0] > i else 0.0,
        "box": f["boxes"][i].tolist() if "boxes" in f and f["boxes"].shape[0] > i else [0, 0, 0, 0],
    }


def _crop_feature(f, meta, tid) -> dict:
    ids = meta.get("crop_object_ids", [])
    if "crop_region" in f and str(tid) in [str(x) for x in ids]:
        idx = [str(x) for x in ids].index(str(tid))
        region = f["crop_region"][idx].float()
    else:
        region = torch.zeros(4608)
    return {
        "pbd": torch.zeros(2048),
        "pbd_be": torch.zeros(2048),
        "region": region,
        "geom": torch.zeros(5),
        "gen": 0.0,
        "box": meta.get("gt_boxes", {}).get(str(tid), [0, 0, 0, 0]),
    }
