"""Shared frozen-bank and target helpers for Stage L20 RMOT-only tools."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from tools.train_l18_carr import (
    BankStore, TextStore, FEATURE_NAMES, expression_text, frame_features,
    load_items,
)
from tools.train_l19 import l19_track_membership_index


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")


def _box_iou_np(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return intersection / max(1e-6, area_a + area_b - intersection)


def strict_observation_groups(bank: dict, frame_index: int,
                              iou_threshold: float = 0.80,
                              appearance_threshold: float = 0.82) -> np.ndarray:
    """Build GT-free one-to-one main/reserve groups for one frame.

    Candidate edges are restricted to the existing broad observation-group
    relation, then require a stricter IoU/appearance pair.  Each main and
    reserve row chooses its best eligible opposite-source row; only mutual
    nearest pairs are merged.  Unmatched rows remain singleton groups.  The
    output is local to the frame and never acts as a temporal track key.
    """
    tensors = bank["tensors"]
    begin, end = map(int, tensors["frame_ptr"][frame_index:frame_index + 2])
    count = end - begin
    source_tensor = tensors.get("pool_id")
    source = (source_tensor[begin:end].numpy().astype(np.int64)
              if source_tensor is not None else np.zeros(count, np.int64))
    boxes = tensors["box"][begin:end].numpy().astype(np.float32)
    clips = tensors["clip"][begin:end].float().numpy().astype(np.float32)
    norms = np.linalg.norm(clips, axis=1, keepdims=True).clip(min=1e-6)
    clips = clips / norms
    broad = tensors.get("observation_group_id")
    broad = (broad[begin:end].numpy().astype(np.int64)
             if broad is not None else np.arange(count, dtype=np.int64))
    main = np.flatnonzero(source == 0)
    reserve = np.flatnonzero(source == 1)
    candidates = {}
    for mi in main.tolist():
        for ri in reserve.tolist():
            if broad[mi] != broad[ri]:
                continue
            overlap = _box_iou_np(boxes[mi], boxes[ri])
            appearance = float(np.dot(clips[mi], clips[ri]))
            if overlap < float(iou_threshold) or appearance < float(appearance_threshold):
                continue
            # IoU is the primary matching evidence; appearance resolves ties.
            candidates[(mi, ri)] = overlap + 0.10 * appearance
    main_best = {}
    reserve_best = {}
    for (mi, ri), value in candidates.items():
        if mi not in main_best or value > main_best[mi][1]:
            main_best[mi] = (ri, value)
        if ri not in reserve_best or value > reserve_best[ri][1]:
            reserve_best[ri] = (mi, value)
    matched = {}
    for mi, (ri, _value) in main_best.items():
        if reserve_best.get(ri, (None, None))[0] == mi:
            matched[mi] = ri
            matched[ri] = mi
    groups = np.arange(count, dtype=np.int64)
    next_group = count
    for mi in main.tolist():
        if mi not in matched:
            continue
        ri = matched[mi]
        if mi < ri:
            groups[mi] = next_group
            groups[ri] = next_group
            next_group += 1
    # Stable local IDs make the rule auditable but are not reused as state keys.
    remap = {}
    output = np.empty(count, dtype=np.int64)
    next_id = 0
    for value in groups.tolist():
        if value not in remap:
            remap[value] = next_id
            next_id += 1
    for index, value in enumerate(groups.tolist()):
        output[index] = remap[value]
    return output


def cached_strict_observation_groups(bank: dict, frame_index: int,
                                     iou_threshold: float = 0.80,
                                     appearance_threshold: float = 0.82) -> np.ndarray:
    """Cache the deterministic GT-free grouping inside a loaded bank."""
    cache = bank.setdefault("_l20_strict_group_cache", {})
    key = (float(iou_threshold), float(appearance_threshold), int(frame_index))
    if key not in cache:
        cache[key] = strict_observation_groups(
            bank, frame_index, iou_threshold, appearance_threshold)
    return cache[key]


def l20_group_ids(bank: dict, frame_index: int,
                  training_conflict_singletons: bool = False,
                  iou_threshold: float = 0.80,
                  appearance_threshold: float = 0.82) -> np.ndarray:
    """Return strict IDs, optionally splitting known train GT conflicts.

    The default path is GT-free and is used by evaluation/inference.  The
    training-only option uses the frozen train sidecar labels solely to avoid
    presenting a mixed-identity group as one supervised target; each row in a
    conflicting group becomes a singleton.  It is never enabled by evaluator
    callers.
    """
    groups = cached_strict_observation_groups(
        bank, frame_index, iou_threshold, appearance_threshold).copy()
    if not training_conflict_singletons:
        return groups
    tensors = bank["tensors"]
    begin, end = map(int, tensors["frame_ptr"][frame_index:frame_index + 2])
    labels = bank.get("candidate_gt", [None] * len(tensors["track_id"]))[begin:end]
    next_group = int(groups.max()) + 1 if len(groups) else 0
    for group_id in np.unique(groups).tolist():
        indices = np.flatnonzero(groups == group_id)
        identities = {str(labels[index]) for index in indices
                      if labels[index] is not None}
        if len(identities) <= 1:
            continue
        for index in indices.tolist():
            groups[index] = next_group
            next_group += 1
    return groups


def l20_frame_features(bank: dict, frame_index: int,
                       device: torch.device,
                       training_conflict_singletons: bool = False):
    features, track_ids, begin, end = frame_features(bank, frame_index, device)
    groups = bank["tensors"].get("observation_group_id")
    if groups is None:
        groups = torch.arange(begin, end, dtype=torch.long)
    else:
        groups = groups[begin:end]
    features["observation_group_id"] = groups.to(
        device, non_blocking=True).long()
    strict_groups = l20_group_ids(
        bank, frame_index,
        training_conflict_singletons=training_conflict_singletons)
    features["l20_group_id"] = torch.as_tensor(
        strict_groups, device=device, dtype=torch.long)
    return features, track_ids, begin, end


def _lookup_labels(entry: dict, frame_id: int) -> set[str]:
    labels = entry.get("label", {})
    return {str(value) for value in labels.get(
        str(frame_id), labels.get(frame_id, []))}


def l20_frame_targets(bank: dict, begin: int, end: int, entry: dict,
                      frame_id: int,
                      track_membership: dict[int, set[str]] | None = None,
                      training_conflict_singletons: bool = False) -> dict:
    """Collapse row labels to observation groups without GT leakage at eval.

    ``membership`` is historical track identity, while ``observation`` and
    ``group_target`` are current-frame expression matches.  NULL is positive
    exactly for ABSENT and PRESENT_UNCOVERED frames; a covered frame may have
    multiple positive groups.
    """
    if track_membership is None:
        track_membership = l19_track_membership_index(bank)
    tensors = bank["tensors"]
    candidate_gt = bank.get("candidate_gt", [None] * len(
        tensors["track_id"]))[begin:end]
    track_ids = tensors["track_id"][begin:end].tolist()
    source_tensor = tensors.get("pool_id")
    source = (source_tensor[begin:end].numpy().astype(np.int64)
              if source_tensor is not None else np.zeros(end - begin, np.int64))
    frame_index = bank.get("frame_to_index", {}).get(int(frame_id))
    if frame_index is not None:
        groups = l20_group_ids(
            bank, int(frame_index),
            training_conflict_singletons=training_conflict_singletons)
    else:
        groups_tensor = tensors.get("observation_group_id")
        groups = (groups_tensor[begin:end].numpy().astype(np.int64)
                  if groups_tensor is not None else
                  np.arange(end - begin, dtype=np.int64))
    target_ids = _lookup_labels(entry, int(frame_id))
    row_membership = np.asarray([
        float(bool(target_ids.intersection(track_membership.get(
            int(track_id), set())))) for track_id in track_ids], np.float32)
    row_presence = np.asarray([float(value is not None)
                               for value in candidate_gt], np.float32)
    row_match = np.asarray([
        float(value is not None and str(value) in target_ids)
        for value in candidate_gt], np.float32)
    main_covered = bool(np.any((row_match > 0.5) & (source == 0)))
    reserve_covered = bool(np.any((row_match > 0.5) & (source == 1)))
    if not target_ids:
        state = 0  # ABSENT
    elif main_covered:
        state = 1  # MAIN_COVERED
    elif reserve_covered:
        state = 2  # RESERVE_COVERED
    else:
        state = 3  # PRESENT_UNCOVERED

    unique = np.unique(groups)
    result = {
        "group_ids": unique.astype(np.int64), "membership": [],
        "observation": [], "presence": [], "group_target": [],
        "source": [], "sizes": [], "row_membership": row_membership,
        "row_presence": row_presence, "row_match": row_match,
        "row_group": groups, "state": int(state),
        "null_target": float(state in (0, 3)), "active": bool(target_ids),
        "target_ids": sorted(target_ids), "main_covered": main_covered,
        "reserve_covered": reserve_covered,
    }
    for group_id in unique.tolist():
        indices = np.flatnonzero(groups == group_id)
        sources = set(int(value) for value in source[indices].tolist())
        result["membership"].append(float(row_membership[indices].max(initial=0.0)))
        result["observation"].append(float(row_match[indices].max(initial=0.0)))
        result["presence"].append(float(row_presence[indices].max(initial=0.0)))
        result["group_target"].append(float(row_match[indices].max(initial=0.0)))
        result["source"].append(next(iter(sources)) if len(sources) == 1 else 2)
        result["sizes"].append(int(len(indices)))
    for key in ("membership", "observation", "presence", "group_target",
                "source", "sizes"):
        dtype = np.int64 if key in {"source", "sizes"} else np.float32
        result[key] = np.asarray(result[key], dtype=dtype)
    return result


def query_identity_set(entry: dict) -> set[str]:
    result = set()
    for values in entry.get("label", {}).values():
        result.update(str(value) for value in values)
    return result


def item_category_l20(item: dict, bank: dict) -> tuple[str, dict]:
    """Metadata-only bucket assignment for balanced L20 episodes."""
    tensors = bank["tensors"]
    frame_to_index = bank["frame_to_index"]
    target_frames = main = reserve = uncovered = hard = 0
    labels = bank.get("candidate_gt", [])
    pool = tensors.get("pool_id")
    pool_values = pool.numpy() if pool is not None else np.zeros(len(labels), np.int64)
    for frame, ids in item["entry"].get("label", {}).items():
        if not ids or int(frame) not in frame_to_index:
            continue
        target_frames += 1
        fi = frame_to_index[int(frame)]
        begin, end = map(int, tensors["frame_ptr"][fi:fi + 2])
        target = {str(value) for value in ids}
        positive = np.asarray([
            value is not None and str(value) in target
            for value in labels[begin:end]], bool)
        main_here = bool(np.any(positive & (pool_values[begin:end] == 0)))
        reserve_here = bool(np.any(positive & (pool_values[begin:end] == 1)))
        main |= main_here
        reserve |= reserve_here
        uncovered |= not (main_here or reserve_here)
        hard |= len(target) > 1 or (positive.sum() and len(positive) - positive.sum() >= 2)
    if reserve and not main:
        primary = "reserve_positive"
    elif uncovered and not main and not reserve:
        primary = "present_uncovered"
    elif main:
        primary = "main_positive"
    else:
        primary = "ordinary_negative"
    return primary, {
        "primary": primary, "has_target_frames": bool(target_frames),
        "hard_negative": bool(hard), "identity_count": len(query_identity_set(item["entry"])),
        "has_main_covered": bool(main), "has_reserve_covered": bool(reserve),
        "has_present_uncovered": bool(uncovered),
    }


def build_l20_buckets(items_by_domain: dict, store: BankStore) -> tuple[dict, dict]:
    buckets = defaultdict(list)
    metadata = {}
    grouped = defaultdict(list)
    for domain, items in items_by_domain.items():
        for item in items:
            grouped[(item["bank_dataset"], item["video"])].append((domain, item))
    for key, values in sorted(grouped.items()):
        bank = store.get(*key)
        for domain, item in values:
            primary, row = item_category_l20(item, bank)
            token = (domain, item["video"], expression_text(item["entry"]))
            metadata[token] = {**row, "domain": domain, "video": item["video"]}
            buckets[primary].append(item)
            if row["has_reserve_covered"] and primary != "reserve_positive":
                buckets["reserve_positive"].append(item)
            if row["has_present_uncovered"] and primary != "present_uncovered":
                buckets["present_uncovered"].append(item)
            if row["hard_negative"]:
                buckets["hard_negative"].append(item)
    return dict(buckets), metadata
