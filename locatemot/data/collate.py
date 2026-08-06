"""Collate PairDataset samples into padded batches."""
from __future__ import annotations

import torch


def collate_track_batch(samples):
    B = len(samples)
    M = max(int(s["ref_mask"].sum()) for s in samples)
    N = max(int(s["cur_mask"].sum()) for s in samples)
    batch = {}
    for key in ["ref_pbd", "ref_region", "ref_geom"]:
        batch[key] = torch.stack([_pad(s[key], M, 0) for s in samples])
    batch["ref_gen"] = torch.stack([_pad1(s["ref_gen"], M, 0.0) for s in samples])
    batch["ref_cat"] = torch.stack([_pad(s["ref_cat"], M) for s in samples])
    batch["ref_mask"] = torch.stack([_pad_bool(s["ref_mask"], M) for s in samples])
    for key in ["cur_pbd", "cur_region", "cur_geom"]:
        batch[key] = torch.stack([_pad(s[key], N, 0) for s in samples])
    batch["cur_gen"] = torch.stack([_pad1(s["cur_gen"], N, 0.0) for s in samples])
    batch["cur_cat"] = torch.stack([_pad(s["cur_cat"], N) for s in samples])
    batch["cur_mask"] = torch.stack([_pad_bool(s["cur_mask"], N) for s in samples])
    batch["match_targets"] = torch.stack([_pad_targets(s["match_targets"], M) for s in samples])
    batch["no_match_targets"] = torch.stack([_pad1(s["no_match_targets"], M, 0.0) for s in samples])
    batch["candidate_missing"] = torch.stack([_pad_bool(s["candidate_missing"], M) for s in samples])
    batch["visible"] = torch.stack([_pad_bool(s["visible"], M) for s in samples])
    batch["labels"] = torch.stack([
        _pad2(s["labels"], M, N) for s in samples
    ])
    batch["gt_iou"] = torch.stack([_pad1(s["gt_iou"], M, 0.0) for s in samples])
    batch["ref_boxes"] = torch.stack([_pad(s["ref_boxes"], M) for s in samples])
    batch["cur_boxes"] = torch.stack([_pad(s["cur_boxes"], N) for s in samples])
    batch["gap"] = torch.stack([s["gap"] for s in samples])
    batch["meta"] = [{k: s[k] for k in ("dataset", "video_id", "protocol", "split")} for s in samples]
    return batch


def _pad(t, n, dim_idx=0):
    if t.shape[dim_idx] >= n:
        return t[:n]
    pad = [0, 0] * (t.dim() - dim_idx - 1) + [0, n - t.shape[dim_idx]]
    return torch.nn.functional.pad(t, pad)


def _pad1(t, n, value):
    if t.shape[0] >= n:
        return t[:n]
    return torch.cat([t, torch.full((n - t.shape[0],), value, dtype=t.dtype)])


def _pad_bool(t, n):
    if t.shape[0] >= n:
        return t[:n]
    return torch.cat([t, torch.zeros(n - t.shape[0], dtype=torch.bool)])


def _pad_targets(t, n):
    if t.shape[0] >= n:
        return t[:n]
    return torch.cat([t, torch.full((n - t.shape[0],), -1, dtype=t.dtype)])


def _pad2(t, m, n):
    if t.shape[0] == m and t.shape[1] == n:
        return t
    out = torch.zeros((m, n), dtype=t.dtype)
    out[: t.shape[0], : t.shape[1]] = t
    return out
