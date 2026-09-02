#!/usr/bin/env python3
"""L82 Phase-D frozen-representation rank probe.

Three representations are rebuilt in one process from the immutable L69 bank:
the L81 candidate evidence, a fresh L59-style fused-memory seed, and the L82
fixed-reference decoder hidden state.  They all use one identical small probe,
loss, fit-video split, seed and schedule.  Feature tensors are process-local
RAM only; no dense representation cache is serialized.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from locatemot.models.l82_rank_probe import L82FactorizedRankProbe, L82RankProbeConfig  # noqa: E402
from locatemot.rmot.l82_losses import l82_rank_loss  # noqa: E402
from locatemot.rmot.l82_grounding_runtime import (  # noqa: E402
    GroundingCandidateReferenceRuntime,
    install_clip_torchvision_compat,
)


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
DATASETS = ("refer_kitti_v1", "refer_kitti_v2")
CATEGORIES = ("positive", "multi_positive", "inactive", "present_uncovered")
REPRESENTATIONS = ("l81_candidate_evidence", "l59_fused_roi", "l82_candidate_reference")
TRAIN_UNITS = ROOT / "outputs/l49/data/train_units.jsonl"
SPLIT_PATH = ROOT / "outputs/l82/protocol/fit_video_train_dev_split.json"
L81_CHECKPOINT = ROOT / "outputs/l81/train/probe500_retry1/checkpoint_l81_step100.pt"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
L69_ROOT = ROOT / "outputs/l69/attempt9/budget40_features/kitti"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path), "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "mtime_ns": path.stat().st_mtime_ns if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value.float()).all()):
        raise FloatingPointError(f"nonfinite {name}")


def digest_keys(keys: list[tuple[Any, ...]]) -> str:
    return hashlib.sha256(json.dumps([list(key) for key in keys], sort_keys=False).encode()).hexdigest()


def key_only_row(raw: dict[str, Any]) -> dict[str, Any]:
    """Extract only address/text fields; supervision is not exposed here."""
    sentence = str(raw.get("sentence") or raw.get("expression") or "")
    result = {
        "unit_key": str(raw["unit_key"]),
        "dataset": str(raw["dataset"]),
        "video": str(raw["video"]),
        "query_id": int(raw["query_id"]),
        "frame_id": int(raw["frame_id"]),
        "sentence": sentence,
        "expression": sentence,
    }
    if not sentence:
        raise AssertionError(f"empty expression: {result['unit_key']}")
    forbidden = {"target_ids", "positive_indices", "positive_count", "category", "labels", "target_present"}
    if forbidden.intersection(result):
        raise AssertionError("label field leaked into key-only row")
    return result


def load_key_only_fit_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with TRAIN_UNITS.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("split") != "fit" or raw.get("dataset") not in DATASETS:
                continue
            rows.append(key_only_row(raw))
    if len(rows) != 5314 or len({row["unit_key"] for row in rows}) != len(rows):
        raise AssertionError(f"fit key-only row drift: {len(rows)}")
    return rows


def load_fit_label_rows() -> dict[str, dict[str, Any]]:
    """Read only fit labels, lazily after a complete feature group exists."""
    result: dict[str, dict[str, Any]] = {}
    with TRAIN_UNITS.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("split") == "fit" and raw.get("dataset") in DATASETS:
                key = str(raw["unit_key"])
                if key in result:
                    raise AssertionError(f"duplicate fit label unit: {key}")
                result[key] = raw
    if len(result) != 5314:
        raise AssertionError(f"fit label row drift: {len(result)}")
    return result


def group_key(row: dict[str, Any]) -> str:
    return f"{row['dataset']}|{row['video']}|{int(row['frame_id'])}"


def load_groups() -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    split = json.loads(SPLIT_PATH.read_text())
    train_keys = [str(key) for key in split["train_group_keys"]]
    dev_keys = [str(key) for key in split["dev_group_keys"]]
    if len(train_keys) != 524 or len(dev_keys) != 138:
        raise AssertionError("L82 video-disjoint split count drift")
    rows_by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_key_only_fit_rows():
        rows_by_group[group_key(row)].append(row)
    groups: dict[str, dict[str, Any]] = {}
    for key in train_keys + dev_keys:
        if key in groups:
            raise AssertionError(f"duplicate split group: {key}")
        rows = sorted(rows_by_group.get(key, []), key=lambda item: (int(item["query_id"]), item["unit_key"]))
        if not rows:
            raise AssertionError(f"split group has no fit queries: {key}")
        dataset, video, frame = key.split("|")
        if dataset not in DATASETS:
            raise AssertionError(f"unexpected dataset: {key}")
        groups[key] = {
            "group_key": key, "dataset": dataset, "video": video, "frame_id": int(frame),
            "queries": rows,
        }
    return groups, train_keys, dev_keys


@dataclass
class GroupData:
    group_key: str
    dataset: str
    video: str
    frame_id: int
    query_unit_keys: list[str]
    query_ids: list[int]
    query_lengths: list[int]
    features: dict[str, torch.Tensor]
    labels: torch.Tensor
    membership_mask: torch.Tensor
    categories: list[str]
    target_ids: list[list[str]]
    candidate_gt: list[list[str | None]]
    candidate_count: int
    row_offsets: list[int]
    row_keys_digest: list[str]
    candidate_indices: list[int]
    pool_ids: list[int]
    track_ids: list[int]


def load_l81_model(device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    from locatemot.models.l81_hierarchical_early_fusion import L81Config, L81HierarchicalEarlyFusion

    package = torch.load(L81_CHECKPOINT, map_location="cpu")
    config = L81Config(**package["model_config"])
    model = L81HierarchicalEarlyFusion(config).to(device=device, dtype=torch.float32)
    result = model.load_state_dict(package["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"L81 strict load failed: {result}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        if parameter.grad is not None:
            raise AssertionError("L81 control has a pre-existing gradient")
    return model, {
        "checkpoint": file_meta(L81_CHECKPOINT),
        "step": int(package.get("step", 100)),
        "model_config": config.__dict__,
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "candidate_output": "candidate_evidence from frozen L81 forward",
    }


def build_group_features(
    group: dict[str, Any],
    store: Any,
    clip_model: Any,
    clip_cache: Any,
    l81_model: torch.nn.Module,
    dino_runtime: GroundingCandidateReferenceRuntime,
    device: torch.device,
) -> tuple[list[Any], dict[str, torch.Tensor], dict[str, Any]]:
    """Construct every query representation before any fit label is attached."""
    from locatemot.rmot.l81_runtime import raw_inputs_for_l81

    batches = [store.build_unit(row) for row in group["queries"]]
    if not batches:
        raise AssertionError(f"empty group: {group['group_key']}")
    first = batches[0]
    n = first.candidate_count
    if n <= 0 or len(first.row_offsets) != n:
        raise AssertionError(f"empty/drifting candidate set: {group['group_key']}")
    l81_values: list[torch.Tensor] = []
    for batch in batches:
        if batch.candidate_count != n or batch.row_offsets != first.row_offsets:
            raise AssertionError(f"same-frame candidate row drift: {batch.unit_key}")
        if batch.row_keys == first.row_keys:
            # Query id is part of the immutable key, so equal lists are only
            # possible for a duplicated query and are rejected below.
            pass
        if int((batch.history_frame_ids > int(batch.frame_id)).sum()) != 0:
            raise AssertionError(f"future history: {batch.unit_key}")
        raw = raw_inputs_for_l81(clip_model, batch, device, clip_cache)
        with torch.inference_mode():
            output = l81_model(
                raw["visual_pyramid"], raw["local_tokens"], raw["text_tokens"], raw["text_mask"],
                batch.history_observations.to(device).clone(), batch.history_mask.to(device).clone(),
                batch.history_frame_ids.to(device).clone(), int(batch.frame_id), raw["boxes_norm"],
                return_audit=True,
            )
        value = output["candidate_evidence"].float().detach().cpu().contiguous()
        finite(value, "L81 candidate evidence")
        if value.shape != (n, 256):
            raise AssertionError(f"L81 candidate evidence shape drift: {tuple(value.shape)}")
        l81_values.append(value)
        del raw, output

    dino_result = dino_runtime.extract_group(batches)
    if len(dino_result["outputs"]) != len(batches):
        raise AssertionError("GroundingDINO query output count drift")
    l59_values = [entry["visual_seed"].float().cpu().contiguous() for entry in dino_result["outputs"]]
    l82_values = [entry["candidate_reference"].float().cpu().contiguous() for entry in dino_result["outputs"]]
    for name, values in (("L59 visual seed", l59_values), ("L82 candidate reference", l82_values)):
        for value in values:
            finite(value, name)
            if value.shape != (n, 256):
                raise AssertionError(f"{name} shape drift: {tuple(value.shape)}")

    # Only now may fit-only expression labels be opened.  Validation rows are
    # never read by this function.
    return batches, {
        "l81_candidate_evidence": torch.stack(l81_values),
        "l59_fused_roi": torch.stack(l59_values),
        "l82_candidate_reference": torch.stack(l82_values),
    }, {
        "native_image_shape": dino_result["native_image_shape"],
        "native_scale_factor": dino_result["native_scale_factor"],
        "native_seconds": dino_result["native_seconds"],
        "replay_seconds": dino_result["replay_seconds"],
        "candidate_reference_events": [entry["candidate_reference_event"] for entry in dino_result["outputs"]],
    }


def attach_fit_labels(batches: list[Any], feature_values: dict[str, torch.Tensor], label_rows: dict[str, dict[str, Any]], store: Any, group: dict[str, Any], audit: dict[str, Any]) -> GroupData:
    first = batches[0]
    n = first.candidate_count
    labels: list[torch.Tensor] = []
    masks: list[bool] = []
    categories: list[str] = []
    targets: list[list[str]] = []
    candidate_gt: list[list[str | None]] = []
    query_keys: list[str] = []
    query_ids: list[int] = []
    query_lengths: list[int] = []
    row_digests: list[str] = []
    for batch in batches:
        if batch.unit_key not in label_rows:
            raise KeyError(f"fit label missing after feature construction: {batch.unit_key}")
        attached = store.attach_labels(batch, label_rows[batch.unit_key])
        label = attached["labels"].bool().cpu()
        if label.shape != (n,):
            raise AssertionError(f"label length drift: {batch.unit_key}")
        labels.append(label)
        masks.append(bool(attached["coverage_mask"]))
        categories.append(str(attached["category"]))
        targets.append([str(x) for x in attached["target_ids"]])
        candidate_gt.append([None if x is None else str(x) for x in attached["sidecar_candidate_gt"]])
        query_keys.append(str(batch.unit_key))
        query_ids.append(int(batch.query_id))
        query_lengths.append(int(batch.text_mask.sum()))
        row_digests.append(digest_keys(batch.row_keys))
        if batch.row_offsets != first.row_offsets:
            raise AssertionError(f"row order changed during label attach: {batch.unit_key}")
    for name, value in feature_values.items():
        if value.shape != (len(batches), n, 256) or not bool(torch.isfinite(value).all()):
            raise AssertionError(f"feature shape/finite drift: {name}/{group['group_key']}")
    if len(set(query_keys)) != len(query_keys):
        raise AssertionError(f"duplicate query unit in group: {group['group_key']}")
    return GroupData(
        group_key=str(group["group_key"]), dataset=str(group["dataset"]), video=str(group["video"]),
        frame_id=int(group["frame_id"]), query_unit_keys=query_keys, query_ids=query_ids,
        query_lengths=query_lengths, features={key: value.clone() for key, value in feature_values.items()},
        labels=torch.stack(labels), membership_mask=torch.tensor(masks, dtype=torch.bool),
        categories=categories, target_ids=targets, candidate_gt=candidate_gt, candidate_count=n,
        row_offsets=[int(x) for x in first.row_offsets], row_keys_digest=row_digests,
        candidate_indices=[int(x) for x in first.candidate_indices],
        pool_ids=[int(x) for x in first.pool_ids], track_ids=[int(x) for x in first.track_ids],
    )


def build_local_groups(
    group_keys: list[str], groups: dict[str, dict[str, Any]], device: torch.device,
) -> tuple[list[GroupData], dict[str, Any]]:
    # The compatibility shim is deliberately process-local and installed
    # before either CLIP or MMDetection is imported.
    clip_compat = install_clip_torchvision_compat()
    from locatemot.rmot.l80_data import L80BankStore
    from locatemot.rmot.l81_runtime import FrameFeatureCache, load_clip

    clip_model = load_clip(device)
    l81_model, l81_info = load_l81_model(device)
    dino_runtime = GroundingCandidateReferenceRuntime(device)
    store = L80BankStore(max_history=8)
    clip_cache = FrameFeatureCache(max_items=1)
    label_rows: dict[str, dict[str, Any]] | None = None
    built: list[GroupData] = []
    feature_started = time.perf_counter()
    total_native = total_replay = 0.0
    max_candidate = 0
    for key in group_keys:
        group = groups[key]
        batches, feature_values, audit = build_group_features(
            group, store, clip_model, clip_cache, l81_model, dino_runtime, device)
        # The first complete feature group is the explicit label boundary.
        if label_rows is None:
            label_rows = load_fit_label_rows()
        data = attach_fit_labels(batches, feature_values, label_rows, store, group, audit)
        built.append(data)
        total_native += float(audit["native_seconds"])
        total_replay += float(audit["replay_seconds"])
        max_candidate = max(max_candidate, data.candidate_count)
        del batches, feature_values, audit
        if device.type == "cuda":
            torch.cuda.empty_cache()

    dino_runtime.close()
    del dino_runtime, l81_model, clip_model, store, clip_cache, label_rows
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if not built:
        raise AssertionError("no local L82 groups built")
    counts = Counter(category for data in built for category in data.categories)
    domain_counts = Counter(data.dataset for data in built)
    return built, {
        "group_count": len(built), "category_counts": dict(counts),
        "domain_counts": dict(domain_counts), "max_candidate_count": max_candidate,
        "native_replay_wall_seconds_sum": total_native + total_replay,
        "native_seconds_sum": total_native, "replay_seconds_sum": total_replay,
        "feature_construction_wall_seconds": time.perf_counter() - feature_started,
        "clip_compatibility": clip_compat, "l81": l81_info,
        "features_persistent": False, "features_in_memory_only": True,
        "validation_labels_read": False,
    }


def average_precision(scores: list[float], labels: list[bool]) -> float | None:
    if not scores or not any(labels):
        return None
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    positives = 0
    total = 0.0
    for rank, index in enumerate(order, 1):
        if labels[index]:
            positives += 1
            total += positives / rank
    return total / positives


def group_metrics(data: GroupData, scores: torch.Tensor, repeat_noise: float = 0.0) -> tuple[dict[str, Any], dict[str, Any]]:
    if scores.shape != data.labels.shape or not bool(torch.isfinite(scores).all()):
        raise AssertionError(f"score shape/finite drift: {data.group_key}")
    score_np = scores.detach().float().cpu().numpy()
    labels_np = data.labels.numpy().astype(bool)
    hard_total = hard_bad = target_hard_total = target_hard_bad = 0
    swap_total = swap_correct = 0
    swap_pairs: list[tuple[float, float]] = []
    r1_total = r1_hit = r5_hit = 0
    multi_total = multi_hit = 0
    row_aps: list[float] = []
    bag_aps: list[float] = []
    margins: list[float] = []
    inactive_false = 0
    inactive_total = 0
    empty = 0
    for q in range(scores.shape[0]):
        if data.categories[q] == "inactive":
            inactive_total += 1
            inactive_false += int(float(score_np[q].max()) >= 0.0)
            empty += int(not bool((score_np[q] >= 0.0).any()))
            continue
        if data.categories[q] == "present_uncovered" or not bool(data.membership_mask[q]):
            continue
        positive = labels_np[q]
        if not positive.any():
            continue
        negative = ~positive
        if negative.any():
            min_positive = float(score_np[q][positive].min())
            max_negative = float(score_np[q][negative].max())
            margins.append(min_positive - max_negative)
            hard_total += 1
            hard_bad += int(max_negative >= min_positive)
        row_ap = average_precision(score_np[q].tolist(), positive.tolist())
        if row_ap is not None:
            row_aps.append(row_ap)
        target_set = set(data.target_ids[q])
        candidate_targets = data.candidate_gt[q]
        if target_set:
            top_order = np.argsort(-score_np[q], kind="stable")
            r1_total += 1
            r1_hit += int(candidate_targets[int(top_order[0])] in target_set)
            r5_hit += int(any(candidate_targets[int(index)] in target_set for index in top_order[:5]))
            if data.categories[q] == "multi_positive":
                multi_total += 1
                positive_count = int(positive.sum())
                top_indices = top_order[:max(positive_count, len(target_set))]
                multi_hit += int(all(target in {candidate_targets[int(index)] for index in top_indices} for target in target_set))
            unique_targets = sorted({value for value in candidate_targets if value is not None})
            bag_scores = []
            bag_labels = []
            for target in unique_targets + ["__none__"]:
                values = [float(score_np[q][i]) for i, value in enumerate(candidate_targets) if (value if value is not None else "__none__") == target]
                if values:
                    bag_scores.append(max(values)); bag_labels.append(target in target_set)
            bag_ap = average_precision(bag_scores, bag_labels)
            if bag_ap is not None:
                bag_aps.append(bag_ap)
            target_rows = np.asarray([value in target_set for value in candidate_targets], dtype=bool)
            other_rows = ~target_rows
            if target_rows.any() and other_rows.any():
                target_hard_total += 1
                target_hard_bad += int(float(score_np[q][other_rows].max()) >= float(score_np[q][target_rows].min()))

    # Same-frame query-swap evidence: one candidate row, two expressions.
    for left in range(scores.shape[0]):
        if data.categories[left] in {"inactive", "present_uncovered"} or not bool(data.membership_mask[left]):
            continue
        for right in range(left + 1, scores.shape[0]):
            if data.categories[right] in {"inactive", "present_uncovered"} or not bool(data.membership_mask[right]):
                continue
            flips = labels_np[left] != labels_np[right]
            for i in np.flatnonzero(flips):
                if labels_np[left, i]:
                    positive_score, negative_score = float(score_np[left, i]), float(score_np[right, i])
                else:
                    positive_score, negative_score = float(score_np[right, i]), float(score_np[left, i])
                swap_pairs.append((positive_score, negative_score))
                swap_total += 1
                swap_correct += int(positive_score > negative_score)

    candidate_effect = score_np.mean(axis=0)
    query_effect = score_np.mean(axis=1)
    interaction = score_np - candidate_effect[None, :] - query_effect[:, None] + float(score_np.mean())
    per_group = {
        "format": "locatemot-l82-rank-group-metrics-v1", "group_key": data.group_key,
        "dataset": data.dataset, "video": data.video, "frame_id": data.frame_id,
        "query_count": len(data.query_unit_keys), "candidate_count": data.candidate_count,
        "hard_violation": (hard_bad / hard_total) if hard_total else None,
        "hard_total": hard_total, "hard_bad": hard_bad,
        "target_bag_hard_violation": (target_hard_bad / target_hard_total) if target_hard_total else None,
        "target_bag_hard_total": target_hard_total, "target_bag_hard_bad": target_hard_bad,
        "query_swap_accuracy": (swap_correct / swap_total) if swap_total else None,
        "query_swap_total": swap_total, "query_swap_correct": swap_correct,
        "target_bag_recall_at1": (r1_hit / r1_total) if r1_total else None,
        "target_bag_recall_at5": (r5_hit / r1_total) if r1_total else None,
        "target_bag_total": r1_total, "target_bag_hit_at1": r1_hit, "target_bag_hit_at5": r5_hit,
        "multi_positive_target_coverage": (multi_hit / multi_total) if multi_total else None,
        "multi_positive_total": multi_total, "multi_positive_hit": multi_hit,
        "row_ap": float(np.mean(row_aps)) if row_aps else None,
        "target_bag_ap": float(np.mean(bag_aps)) if bag_aps else None,
        "strict_margin_mean": float(np.mean(margins)) if margins else None,
        "strict_margin_min": float(np.min(margins)) if margins else None,
        "score_mean": float(score_np.mean()), "score_std": float(score_np.std()),
        "interaction_variance": float(interaction.var()),
        "candidate_main_variance": float(candidate_effect.var()),
        "query_main_variance": float(query_effect.var()),
        "repeat_run_noise": float(repeat_noise),
        "inactive_false_acceptance": (inactive_false / inactive_total) if inactive_total else None,
        "inactive_total": inactive_total, "inactive_false": inactive_false, "empty_count": empty,
        "empty_rate": (empty / inactive_total) if inactive_total else None,
        "candidate_deletion": False, "candidate_truncation": False,
        "candidate_count_complete": True, "finite": True,
        "row_keys_digest": list(data.row_keys_digest), "row_offsets": list(data.row_offsets),
        "query_ids": list(data.query_ids), "query_lengths": list(data.query_lengths),
        "categories": list(data.categories), "positive_counts": [int(x.sum()) for x in data.labels],
        "duplicate_candidate_index_count": len(data.candidate_indices) - len(set(data.candidate_indices)),
    }
    auxiliary = {"swap_pairs": swap_pairs, "margins": margins}
    return per_group, auxiliary


def aggregate_group_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    def ratio(num: str, den: str) -> float | None:
        denominator = sum(int(row.get(den, 0)) for row in records)
        return (sum(int(row.get(num, 0)) for row in records) / denominator) if denominator else None

    def mean(field: str) -> float | None:
        values = [float(row[field]) for row in records if row.get(field) is not None and math.isfinite(float(row[field]))]
        return float(np.mean(values)) if values else None

    swap_pairs = []
    for row in records:
        swap_pairs.extend(row.pop("_swap_pairs", []))
    if swap_pairs:
        auc = float(np.mean([1.0 if positive > negative else 0.5 if positive == negative else 0.0 for positive, negative in swap_pairs]))
    else:
        auc = None
    result = {
        "group_count": len(records),
        "hard_violation": ratio("hard_bad", "hard_total"),
        "target_bag_hard_violation": ratio("target_bag_hard_bad", "target_bag_hard_total"),
        "query_swap_accuracy": ratio("query_swap_correct", "query_swap_total"),
        "query_swap_auc": auc,
        "target_bag_recall_at1": ratio("target_bag_hit_at1", "target_bag_total"),
        "target_bag_recall_at5": ratio("target_bag_hit_at5", "target_bag_total"),
        "multi_positive_target_coverage": ratio("multi_positive_hit", "multi_positive_total"),
        "row_ap_macro": mean("row_ap"), "target_bag_ap_macro": mean("target_bag_ap"),
        "strict_margin_mean": mean("strict_margin_mean"), "score_mean": mean("score_mean"),
        "score_std": mean("score_std"), "interaction_variance": mean("interaction_variance"),
        "candidate_main_variance": mean("candidate_main_variance"),
        "query_main_variance": mean("query_main_variance"), "repeat_run_noise": mean("repeat_run_noise"),
        "inactive_false_acceptance": ratio("inactive_false", "inactive_total"),
        "empty_rate": ratio("empty_count", "inactive_total"),
        "hard_total": sum(int(row.get("hard_total", 0)) for row in records),
        "query_swap_total": sum(int(row.get("query_swap_total", 0)) for row in records),
        "target_bag_total": sum(int(row.get("target_bag_total", 0)) for row in records),
        "multi_positive_total": sum(int(row.get("multi_positive_total", 0)) for row in records),
        "inactive_total": sum(int(row.get("inactive_total", 0)) for row in records),
        "empty_count": sum(int(row.get("empty_count", 0)) for row in records),
        # These fields describe whether a violation occurred.  Group records
        # carry ``False`` when every candidate row was retained; do not invert
        # that contract while aggregating the audit.
        "candidate_deletion": any(bool(row.get("candidate_deletion", True)) for row in records),
        "candidate_truncation": any(bool(row.get("candidate_truncation", True)) for row in records),
        "finite": all(bool(row.get("finite", False)) for row in records),
    }
    empty_denominator = sum(int(row.get("inactive_total", 0)) for row in records)
    empty_numerator = sum(int(row.get("empty_count", 0)) for row in records)
    result["empty_rate"] = (empty_numerator / empty_denominator) if empty_denominator else None
    return result


def add_breakdowns(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field, values in (("dataset", DATASETS), ("video", sorted({str(row["video"]) for row in records}))):
        result[field] = {}
        for value in values:
            subset = [row.copy() for row in records if str(row[field]) == str(value)]
            for row in subset:
                row["_swap_pairs"] = []
            result[field][str(value)] = aggregate_group_metrics(subset) if subset else {"group_count": 0}
    category_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        for category in row.get("categories", []):
            category_groups[category].append(row)
    result["category_group_presence"] = {key: len(value) for key, value in category_groups.items()}
    result["query_length_distribution"] = {
        "count": sum(len(row.get("query_lengths", [])) for row in records),
        "values": [int(value) for row in records for value in row.get("query_lengths", [])],
    }
    result["candidate_count_distribution"] = {
        "count": len(records), "values": [int(row["candidate_count"]) for row in records],
    }
    return result


def paired_bootstrap(l81: list[dict[str, Any]], other: list[dict[str, Any]], seed: int = SEED) -> dict[str, Any]:
    by_l81 = {str(row["group_key"]): row for row in l81}
    by_other = {str(row["group_key"]): row for row in other}
    keys = [key for key in by_l81 if key in by_other and by_l81[key].get("hard_violation") is not None and by_other[key].get("hard_violation") is not None]
    if not keys:
        return {"status": "CI_INCONCLUSIVE", "group_count": 0}
    differences = np.asarray([float(by_l81[key]["hard_violation"]) - float(by_other[key]["hard_violation"]) for key in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    samples = np.asarray([rng.choice(differences, size=len(differences), replace=True).mean() for _ in range(1000)], dtype=np.float64)
    result = {
        "status": "complete", "group_count": len(keys), "resamples": 1000,
        "point_estimate": float(differences.mean()), "ci95_low": float(np.quantile(samples, .025)),
        "ci95_high": float(np.quantile(samples, .975)), "seed": seed,
        "ci_lower_positive": bool(float(np.quantile(samples, .025)) > 0.0),
    }
    if len(keys) < 30:
        result["status"] = "CI_INCONCLUSIVE"
    return result


def train_one_representation(
    representation: str, train_data: list[GroupData], dev_data: list[GroupData],
    device: torch.device, out: Path, rank: int, world_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    config = L82RankProbeConfig()
    base = L82FactorizedRankProbe(config).to(device=device, dtype=torch.float32)
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel
        model: torch.nn.Module = DistributedDataParallel(base, device_ids=[device.index], output_device=device.index)
    else:
        model = base
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    updates_per_epoch = max(1, len(train_data))
    total_updates = updates_per_epoch * 10
    warmup = max(1, int(round(total_updates * 0.05)))

    def schedule(step: int) -> float:
        if step <= warmup:
            return float(step) / float(warmup)
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total_updates - warmup)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule(int(step)))
    trace: list[dict[str, Any]] = []
    step = 0
    max_grad = 0.0
    for epoch in range(1, 11):
        model.train()
        for data in train_data:
            step += 1
            optimizer.zero_grad(set_to_none=True)
            value = data.features[representation].to(device=device, dtype=torch.float32).clone()
            output = model(value)
            loss, parts = l82_rank_loss(output["interaction"], data.labels.to(device), data.membership_mask.to(device), data.categories)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite rank loss at {representation} step {step}")
            loss.backward()
            gradients = [parameter.grad for parameter in base.parameters() if parameter.grad is not None]
            if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
                raise FloatingPointError(f"invalid rank gradient at {representation} step {step}")
            grad_norm = float(torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0))
            if not math.isfinite(grad_norm) or grad_norm <= 0.0:
                raise FloatingPointError(f"zero/nonfinite rank gradient at {representation} step {step}")
            max_grad = max(max_grad, grad_norm)
            optimizer.step(); scheduler.step()
            trace.append({
                "representation": representation, "epoch": epoch, "step": step,
                "loss": float(loss.detach().cpu()), "grad_norm": grad_norm,
                "lr": float(optimizer.param_groups[0]["lr"]),
                **{key: (float(value.detach().cpu()) if torch.is_tensor(value) else value) for key, value in parts.items()},
                "finite": True, "nonzero_gradient": True,
            })
            del value, output, loss
    if step != total_updates:
        raise AssertionError("L82 update count drift")
    if world_size > 1:
        dist.barrier()
    model.eval()
    dev_records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for data in dev_data:
            value = data.features[representation].to(device=device, dtype=torch.float32).clone()
            out1 = model(value)
            out2 = model(value)
            scores = out1["interaction"].float().cpu()
            repeat_noise = float((out1["interaction"] - out2["interaction"]).abs().max().cpu())
            record, aux = group_metrics(data, scores, repeat_noise)
            record["_swap_pairs"] = aux["swap_pairs"]
            dev_records.append(record)
            del value, out1, out2
    state_model = base
    package = {
        "format": "locatemot-l82-frozen-rank-probe-checkpoint-v1",
        "stage": "phase_d_frozen_representation_rank_probe", "representation": representation,
        "epoch": 10, "step": step, "seed": SEED,
        "model_config": config.__dict__, "model_state_dict": state_model.state_dict(),
        "model_parameter_count": int(sum(p.numel() for p in state_model.parameters())),
        "schedule": {"optimizer": "AdamW", "lr": 2e-4, "weight_decay": 1e-4, "warmup_fraction": .05, "scheduler": "cosine", "gradient_clip": 1.0, "epochs": 10},
        "primary_score": "interaction", "nuisance_controls": ["candidate_main", "query_main"],
        "candidate_deletion": False, "candidate_truncation": False,
    }
    path = out / "checkpoints" / f"{representation}_checkpoint_epoch10.pt"
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(package, path)
    if world_size > 1:
        dist.barrier()
    info = {"representation": representation, "checkpoint": file_meta(path) if rank == 0 else {"path": str(path.resolve())},
            "trace_steps": len(trace), "max_grad_norm": max_grad,
            "parameter_report": state_model.parameter_report()}
    del model, base, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return info, dev_records, {"loss_trace": trace}


def contract_only(args: argparse.Namespace, out: Path, device: torch.device) -> dict[str, Any]:
    groups, train_keys, dev_keys = load_groups()
    selected = (train_keys + dev_keys)[:max(1, int(args.max_groups))]
    built, build_info = build_local_groups(selected, groups, device)
    checks = []
    for data in built:
        for name in REPRESENTATIONS:
            probe = L82FactorizedRankProbe().to(device)
            value = data.features[name].to(device).clone().requires_grad_(False)
            output = probe(value)
            loss, parts = l82_rank_loss(output["interaction"], data.labels.to(device), data.membership_mask.to(device), data.categories)
            loss.backward()
            grad_values = [p.grad for p in probe.parameters() if p.grad is not None]
            if not grad_values or not all(bool(torch.isfinite(g).all()) for g in grad_values):
                raise AssertionError(f"contract gradient failed: {data.group_key}/{name}")
            checks.append({"group_key": data.group_key, "representation": name, "candidate_count": data.candidate_count,
                           "feature_shape": list(value.shape), "loss": float(loss.detach()), "finite": True,
                           "nonzero_gradient": any(bool(g.abs().sum() > 0) for g in grad_values),
                           "row_keys_complete": len(data.row_keys_digest) == value.shape[0],
                           "candidate_deletion": False, "candidate_truncation": False})
            del probe, value, output, loss
    payload = {
        "format": "locatemot-l82-rank-probe-contract-v1", "status": "complete",
        "stage": "phase_d_contract_only", "command": " ".join([sys.executable] + sys.argv),
        "cwd": str(ROOT), "luna_thread": THREAD, "seed": SEED,
        "selected_group_keys": selected, "train_keys_available": len(train_keys), "dev_keys_available": len(dev_keys),
        "checks": checks, "feature_build": build_info,
        "labels_scope": "fit-only labels attached after each group's complete label-free representation construction",
        "same_class_hard_negative_metadata": "unavailable",
        "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
        "training_run": False, "hota_trackeval_run": False, "candidate_deletion": False, "candidate_truncation": False,
        "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED", "gpu_world_size": 1,
    }
    write_json(out / "contract.json", payload)
    write_json(out / "provenance.json", {"format": "locatemot-l82-rank-probe-contract-provenance-v1", "status": "complete", "inputs": {"manifest": file_meta(MANIFEST), "split": file_meta(SPLIT_PATH), "train_units": file_meta(TRAIN_UNITS), "l69_root": str(L69_ROOT), "l81_checkpoint": file_meta(L81_CHECKPOINT)}, "labels_scope": "fit-only; no calibration/validation/screening/official-test labels", "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False})
    write_json(out / "status.json", {"format": "locatemot-l82-rank-probe-status-v1", "status": "complete", "failure_root_cause": None, "next_action": "run the preregistered 10-epoch four-GPU rank probe", "command": " ".join([sys.executable] + sys.argv)})
    return payload


def run_training(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 4:
        raise RuntimeError(f"L82 world size exceeds four: {world_size}")
    if not torch.cuda.is_available():
        raise RuntimeError("L82 Phase D requires CUDA")
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    np.random.seed(SEED + rank)
    groups, train_keys, dev_keys = load_groups()
    all_keys = train_keys + dev_keys
    local_keys = all_keys[rank::world_size]
    local_train = [key for key in train_keys if key in set(local_keys)]
    local_dev = [key for key in dev_keys if key in set(local_keys)]
    if not local_train or not local_dev:
        raise AssertionError(f"rank {rank} lacks train/dev groups: {len(local_train)}/{len(local_dev)}")
    data, build_info = build_local_groups(local_keys, groups, device)
    data_by_key = {value.group_key: value for value in data}
    train_data = [data_by_key[key] for key in local_train]
    dev_data = [data_by_key[key] for key in local_dev]
    local_summaries: dict[str, Any] = {}
    all_dev_records: dict[str, list[dict[str, Any]]] = {}
    all_traces: dict[str, list[dict[str, Any]]] = {}
    start = time.perf_counter()
    for representation in REPRESENTATIONS:
        info, dev_records, trace_data = train_one_representation(representation, train_data, dev_data, device, out, rank, world_size)
        local_summaries[representation] = info
        all_dev_records[representation] = dev_records
        all_traces[representation] = trace_data["loss_trace"]
    gathered_records: dict[str, list[list[dict[str, Any]]]] = {}
    # ``gather_object`` requires a destination-side list with one slot per
    # rank.  The per-representation gathers above already follow this
    # contract; keep the aggregate gathers identical so the 4-rank run does
    # not fail after completing feature extraction/training.
    gathered_summaries: list[dict[str, Any] | None] = [None] * world_size if rank == 0 else []
    gathered_build_info: list[dict[str, Any] | None] = [None] * world_size if rank == 0 else []
    gathered_traces: dict[str, list[dict[str, Any]]] = {}
    if world_size > 1:
        for representation in REPRESENTATIONS:
            holder: list[list[dict[str, Any]]] | None = [None] * world_size if rank == 0 else None
            dist.gather_object(all_dev_records[representation], holder, dst=0)
            if rank == 0:
                gathered_records[representation] = holder or []
        dist.gather_object(local_summaries, gathered_summaries if rank == 0 else None, dst=0)
        dist.gather_object(build_info, gathered_build_info if rank == 0 else None, dst=0)
        for representation in REPRESENTATIONS:
            holder_trace: list[list[dict[str, Any]]] | None = [None] * world_size if rank == 0 else None
            dist.gather_object(all_traces[representation], holder_trace, dst=0)
            if rank == 0:
                gathered_traces[representation] = [row for shard in (holder_trace or []) for row in shard]
    else:
        gathered_records = {key: [value] for key, value in all_dev_records.items()}
        gathered_summaries = [local_summaries]
        gathered_build_info = [build_info]
        gathered_traces = all_traces
    if rank == 0:
        flat_records: dict[str, list[dict[str, Any]]] = {}
        for representation in REPRESENTATIONS:
            flat_records[representation] = [record for shard in gathered_records[representation] for record in shard]
            # Preserve auxiliary pair lists for bootstrap/AUC but never write
            # them to the compact group JSONL twice.
        metrics: dict[str, Any] = {}
        for representation in REPRESENTATIONS:
            records = flat_records[representation]
            metrics[representation] = {
                "aggregate": aggregate_group_metrics([record.copy() for record in records]),
                "breakdowns": add_breakdowns(records),
                "group_records": len(records),
                "all_dev_groups_present": len({row["group_key"] for row in records}) == len(dev_keys),
            }
        for representation in REPRESENTATIONS:
            baseline = flat_records["l81_candidate_evidence"]
            metrics[representation]["paired_bootstrap_vs_l81"] = paired_bootstrap(baseline, flat_records[representation])
        l82_metrics = metrics["l82_candidate_reference"]["aggregate"]
        l81_metrics = metrics["l81_candidate_evidence"]["aggregate"]
        l59_metrics = metrics["l59_fused_roi"]["aggregate"]
        v1_swap = metrics["l82_candidate_reference"]["breakdowns"]["dataset"].get("refer_kitti_v1", {}).get("query_swap_accuracy")
        v2_swap = metrics["l82_candidate_reference"]["breakdowns"]["dataset"].get("refer_kitti_v2", {}).get("query_swap_accuracy")
        bootstrap = metrics["l82_candidate_reference"]["paired_bootstrap_vs_l81"]
        gate_reasons = []
        checks = {
            "dev_hard_violation": l82_metrics.get("hard_violation") is not None and l82_metrics["hard_violation"] <= .8666667,
            "hard_improvement_vs_l81": l81_metrics.get("hard_violation") is not None and l82_metrics.get("hard_violation") is not None and l81_metrics["hard_violation"] - l82_metrics["hard_violation"] >= .05,
            "query_swap_accuracy": l82_metrics.get("query_swap_accuracy") is not None and l82_metrics["query_swap_accuracy"] >= .70,
            "query_swap_auc": l82_metrics.get("query_swap_auc") is not None and l82_metrics["query_swap_auc"] >= .75,
            "target_bag_recall_at1": l82_metrics.get("target_bag_recall_at1") is not None and l82_metrics["target_bag_recall_at1"] >= .7894444,
            "multi_positive_target_coverage": l82_metrics.get("multi_positive_target_coverage") is not None and l82_metrics["multi_positive_target_coverage"] >= .7894444,
            "v1_query_swap": v1_swap is not None and v1_swap >= .65,
            "v2_query_swap": v2_swap is not None and v2_swap >= .65,
            "interaction_above_repeat_noise": l82_metrics.get("interaction_variance") is not None and l82_metrics.get("repeat_run_noise") is not None and l82_metrics["interaction_variance"] > l82_metrics["repeat_run_noise"],
            "candidate_rows_complete": all(value.get("all_dev_groups_present") and value["aggregate"].get("candidate_deletion") is False and value["aggregate"].get("candidate_truncation") is False and value["aggregate"].get("finite") for value in metrics.values()),
            "l82_not_worse_than_l59": l59_metrics.get("hard_violation") is None or l82_metrics.get("hard_violation") <= l59_metrics["hard_violation"],
            "bootstrap_positive_or_inconclusive": bootstrap.get("ci_lower_positive") is True or bootstrap.get("status") == "CI_INCONCLUSIVE",
        }
        gate_reasons.extend(key for key, value in checks.items() if not value)
        gate = {
            "format": "locatemot-l82-rank-gate-v1", "status": "rank_representation_gate_pass" if not gate_reasons else "rank_representation_gate_fail",
            "stage": "phase_d_frozen_representation_rank_probe", "checks": checks,
            "failed_checks": gate_reasons, "thresholds": {"dev_hard_violation_max": .8666667, "hard_improvement_min": .05, "query_swap_accuracy_min": .70, "query_swap_auc_min": .75, "target_bag_recall_at1_min": .7894444, "multi_positive_target_coverage_min": .7894444, "domain_query_swap_min": .65},
            "selection": "none; all three probes use epoch10 and no dev-based continuation",
            "l59_best_control": {"hard_violation": l59_metrics.get("hard_violation")},
            "paired_bootstrap": bootstrap,
            "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "training_run": True, "hota_trackeval_run": False, "candidate_deletion": False, "candidate_truncation": False,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED", "gpu_world_size": world_size,
        }
        write_json(out / "representation_metrics.json", metrics)
        compact_records = []
        for representation in REPRESENTATIONS:
            for record in flat_records[representation]:
                compact = {key: value for key, value in record.items() if not key.startswith("_")}
                compact["representation"] = representation
                compact_records.append(compact)
        append_jsonl(out / "dev_group_metrics.jsonl", compact_records)
        for representation in REPRESENTATIONS:
            write_json(out / f"loss_trace_{representation}.json", gathered_traces.get(representation, []))
        write_json(out / "rank_gate.json", gate)
        write_json(out / "summary.json", {
            "format": "locatemot-l82-rank-probe-summary-v1", "status": gate["status"],
            "train_group_count": len(train_keys), "dev_group_count": len(dev_keys),
            "local_build_info": gathered_build_info, "representation_metrics": metrics,
            "checkpoint_summaries": gathered_summaries[0] if gathered_summaries else {},
            "elapsed_seconds": time.perf_counter() - start,
            "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "training_run": True, "hota_trackeval_run": False, "candidate_deletion": False, "candidate_truncation": False,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED", "gpu_world_size": world_size,
        })
        write_json(out / "provenance.json", {
            "format": "locatemot-l82-rank-probe-provenance-v1", "status": "complete", "command": " ".join([sys.executable] + sys.argv), "cwd": str(ROOT), "luna_thread": THREAD, "seed": SEED,
            "inputs": {"manifest": file_meta(MANIFEST), "split": file_meta(SPLIT_PATH), "train_units": file_meta(TRAIN_UNITS), "l69_root": str(L69_ROOT), "l81_checkpoint": file_meta(L81_CHECKPOINT)},
            "fit_scope": {"train_group_count": len(train_keys), "dev_group_count": len(dev_keys), "video_disjoint": True, "validation_labels_read": False, "screening_labels_read": False, "official_test_labels_read": False},
            "representations": {"l81_candidate_evidence": "frozen L81 step100 candidate_evidence", "l59_fused_roi": "fresh frozen GroundingDINO fused post-encoder visual seed", "l82_candidate_reference": "fresh frozen GroundingDINO fixed-reference decoder final hidden"},
            "probe": L82FactorizedRankProbe().parameter_report(), "loss": {"same_class_hard_negative_metadata": "unavailable", "fallback": "all current-frame negatives and exact same-frame query flips"},
            "resources": {"features_persistent": False, "raw_dense_cache_written": False, "clip_weight_copied": False, "max_gpu_count": world_size, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
            "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "training_run": True, "hota_trackeval_run": False, "candidate_deletion": False, "candidate_truncation": False, "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED", "l81_modified": False, "uidm_shared_checkpoint_modified": False,
        })
        write_json(out / "status.json", {"format": "locatemot-l82-rank-probe-status-v1", "status": gate["status"], "failure_root_cause": gate_reasons[0] if gate_reasons else None, "next_action": "stop Phase D and write report" if gate_reasons else "request separate authorization for Phase E", "command": " ".join([sys.executable] + sys.argv)})
        if world_size > 1:
            dist.barrier()
        return {"gate": gate, "metrics": metrics}
    if world_size > 1:
        dist.barrier()
    return {"rank": rank, "status": "worker_complete"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/l82/train/frozen_rank_probe")
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--max-groups", type=int, default=4)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA drift")
    if not out.exists():
        out.mkdir(parents=True)
    elif any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L82 output: {out}")
    if args.contract_only:
        if not torch.cuda.is_available():
            raise RuntimeError("contract audit requires CUDA")
        torch.cuda.set_device(0)
        torch.manual_seed(SEED); np.random.seed(SEED)
        contract_only(args, out, torch.device("cuda:0"))
        return 0
    try:
        result = run_training(args, out)
        if int(os.environ.get("RANK", "0")) == 0:
            print(json.dumps({"status": result.get("gate", {}).get("status", "complete"), "out": str(out)}, sort_keys=True))
        return 0
    except Exception as exc:
        rank = int(os.environ.get("RANK", "0"))
        payload = "# INCOMPLETE L82 rank probe\n\n"
        payload += f"rank={rank}\nfirst_error={type(exc).__name__}: {exc}\n\n```text\n{traceback.format_exc()}\n```\n"
        try:
            (out / f"INCOMPLETE.rank{rank}.md").write_text(payload)
            if rank == 0:
                (out / "INCOMPLETE.md").write_text(payload)
        except Exception:
            pass
        raise
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
