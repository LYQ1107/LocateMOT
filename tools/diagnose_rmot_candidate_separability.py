"""Threshold-free separability audit for frozen RMOT candidate banks.

This is the first successor-stage diagnostic after L20. It uses only the
frozen fast manifest, frozen bank features, and the existing expression/GT
labels. It does not load a checkpoint, train a model, choose a threshold, or
run TrackEval. Every cue is kept separate so a later scorer cannot hide a
weak feature behind a fused score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))


FEATURE_NAMES = (
    "clip_text_similarity",
    "appearance_similarity",
    "geometry_similarity",
    "objectness",
    "tracklet_history_similarity",
    "source_reserve_indicator",
    "observation_history_similarity",
    "motion_similarity",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def finite_vector(value, length: int) -> np.ndarray:
    result = np.asarray(value, np.float32).reshape(-1)
    if len(result) != length:
        raise ValueError(
            f"feature length mismatch: expected {length}, got {len(result)}")
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def unit_rows(value: np.ndarray) -> np.ndarray:
    value = np.nan_to_num(np.asarray(value, np.float32), copy=False)
    norm = np.linalg.norm(value, axis=1, keepdims=True)
    return value / np.maximum(norm, 1e-6)


def scalar_stats(values) -> dict:
    values = np.asarray(values, np.float64).reshape(-1)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.90)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, np.float64).reshape(-1)
    labels = np.asarray(labels, bool).reshape(-1)
    valid = np.isfinite(scores)
    scores, labels = scores[valid], labels[valid]
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if not positive or not negative:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return float((ranks[labels].sum() - positive * (positive + 1) / 2.0) /
                 (positive * negative))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, np.float64).reshape(-1)
    labels = np.asarray(labels, bool).reshape(-1)
    valid = np.isfinite(scores)
    scores, labels = scores[valid], labels[valid]
    positive = int(labels.sum())
    if not positive:
        return None
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order].astype(np.float64)
    cumulative = np.cumsum(ordered)
    positions = np.flatnonzero(ordered)
    precision = cumulative[positions] / (positions + 1.0)
    return float(precision.mean())


def rank_stats(values) -> dict:
    return scalar_stats(values)


def combine_stats(stats_list: list[dict]) -> dict:
    """Combine scalar moments without retaining all per-candidate ranks."""
    valid = [value for value in stats_list if value.get("count", 0)]
    if not valid:
        return {"count": 0}
    count = sum(int(value["count"]) for value in valid)
    mean = sum(int(value["count"]) * float(value["mean"])
               for value in valid) / count
    second = sum(int(value["count"]) *
                 (float(value["std"]) ** 2 + float(value["mean"]) ** 2)
                 for value in valid) / count
    return {
        "count": count,
        "mean": float(mean),
        "std": float(max(0.0, second - mean * mean) ** 0.5),
        "median": None,
        "q90": None,
        "min": min(float(value["min"]) for value in valid),
        "max": max(float(value["max"]) for value in valid),
    }


def percentile01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, np.float32)
    if not len(values):
        return values.copy()
    low, high = np.quantile(values, [0.05, 0.95])
    if float(high) <= float(low) + 1e-6:
        return np.full(len(values), 0.5, np.float32)
    return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", str(text).lower()))


def geometry_score(text: str, geometry: np.ndarray) -> np.ndarray:
    """Convert explicit spatial/size words into a transparent geometry cue.

    The bank has normalized center/size geometry but no learned query geometry
    head. A query without one of these explicit words receives a constant 0.5
    score, so it cannot accidentally look separable.
    """
    geometry = np.asarray(geometry, np.float32)
    if geometry.ndim != 2 or geometry.shape[1] < 7:
        raise ValueError(f"expected [N,7] geometry, got {geometry.shape}")
    token = words(text)
    cx, cy = geometry[:, 0], geometry[:, 1]
    width, height, area, bottom = (geometry[:, 2], geometry[:, 3],
                                   geometry[:, 4], geometry[:, 6])
    candidates = []
    if "left" in token:
        candidates.append(1.0 - cx)
    if "right" in token:
        candidates.append(cx)
    if token.intersection({"middle", "center"}):
        candidates.append(1.0 - np.abs(2.0 * cx - 1.0))
    if "top" in token:
        candidates.append(1.0 - cy)
    if "bottom" in token:
        candidates.append(bottom)
    size = percentile01(area + 0.25 * height + 0.10 * width)
    if token.intersection({"large", "big", "tall", "near", "front", "closest"}):
        candidates.append(size)
    if token.intersection({"small", "short", "far", "back", "farthest"}):
        candidates.append(1.0 - size)
    if not candidates:
        return np.full(len(geometry), 0.5, np.float32)
    return np.mean(np.stack(candidates, axis=0), axis=0).astype(np.float32)


def motion_score(text: str, motion: np.ndarray) -> np.ndarray:
    """Transparent motion/direction cue from the frozen 8-D motion feature."""
    motion = np.asarray(motion, np.float32)
    if motion.ndim != 2 or motion.shape[1] < 5:
        raise ValueError(f"expected [N,8] motion, got {motion.shape}")
    token = words(text)
    candidates = []
    speed = percentile01(np.abs(motion[:, 4]))
    if token.intersection({"moving", "running", "walking", "dancing", "driving",
                           "turning", "riding", "going", "entering", "leaving"}):
        candidates.append(speed)
    if token.intersection({"stationary", "standing", "parked", "parking",
                           "stopping"}):
        candidates.append(1.0 - speed)
    if "left" in token:
        candidates.append(1.0 - percentile01(motion[:, 0]))
    if "right" in token:
        candidates.append(percentile01(motion[:, 0]))
    if "front" in token:
        candidates.append(percentile01(motion[:, 1]))
    if "back" in token:
        candidates.append(1.0 - percentile01(motion[:, 1]))
    if not candidates:
        return np.full(len(motion), 0.5, np.float32)
    return np.mean(np.stack(candidates, axis=0), axis=0).astype(np.float32)


def load_metadata() -> dict[tuple[str, str], dict]:
    result = {}
    for path in (
            ROOT / "outputs/l11/data/rmot_kitti/expressions.json",
            ROOT / "outputs/l16/data/kitti_missing/records/expressions.json"):
        for video, entries in json.loads(path.read_text()).items():
            for entry in entries:
                expression = str(entry.get("expression",
                                          entry.get("sentence", "")))
                result[(video, expression)] = entry
    return result


def load_bank(bank_path: Path) -> dict:
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    tensors = bank["tensors"]
    required = {
        "frame", "frame_ptr", "frame_ids", "track_id", "box", "clip",
        "history_clip", "geometry", "motion", "objectness", "pool_id",
    }
    missing = sorted(required - set(tensors))
    if missing:
        raise ValueError(f"{bank_path} missing candidate fields: {missing}")
    n = len(tensors["track_id"])
    labels_path = bank_path.with_suffix(".labels.json")
    if not labels_path.exists():
        raise FileNotFoundError(
            f"candidate GT sidecar is required for diagnostic: {labels_path}")
    labels = json.loads(labels_path.read_text())["candidate_gt"]
    if len(labels) != n:
        raise ValueError(f"candidate sidecar length mismatch in {labels_path}")
    arrays = {
        "frame": tensors["frame"].numpy().astype(np.int32),
        "frame_ids": tensors["frame_ids"].numpy().astype(np.int32),
        "frame_ptr": tensors["frame_ptr"].numpy().astype(np.int64),
        "clip": tensors["clip"].float().numpy().astype(np.float32),
        "history_clip": tensors["history_clip"].float().numpy().astype(np.float32),
        "geometry": tensors["geometry"].float().numpy().astype(np.float32),
        "motion": tensors["motion"].float().numpy().astype(np.float32),
        "objectness": tensors["objectness"].float().numpy().astype(np.float32),
        "source": tensors["pool_id"].numpy().astype(np.int8),
    }
    for name, value in arrays.items():
        if name not in {"frame_ids", "frame_ptr"} and len(value) != n:
            raise ValueError(
                f"{bank_path} field {name} has length {len(value)} != {n}")
    for index, frame_id in enumerate(arrays["frame_ids"].tolist()):
        begin, end = arrays["frame_ptr"][index:index + 2]
        if not np.all(arrays["frame"][begin:end] == frame_id):
            raise ValueError(f"frame_ptr/frame mismatch in {bank_path} at {frame_id}")
    arrays["labels"] = np.asarray(labels, dtype=object)
    arrays["metadata"] = bank.get("metadata", {})
    arrays["bank_sha256"] = sha256_file(bank_path)
    arrays["labels_sha256"] = sha256_file(labels_path)
    del bank
    return arrays


def query_targets(data: dict, entry: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = data["labels"]
    frame = data["frame"]
    source = data["source"]
    target_by_frame = {
        int(key): {str(value) for value in values}
        for key, values in entry.get("label", {}).items()
    }
    positive = np.zeros(len(frame), bool)
    null = np.zeros(len(frame), bool)
    state = np.zeros(len(frame), np.int8)
    for frame_id in np.unique(frame).tolist():
        indices = np.flatnonzero(frame == int(frame_id))
        target_ids = target_by_frame.get(int(frame_id), set())
        positive[indices] = [
            value is not None and str(value) in target_ids
            for value in labels[indices]
        ]
        main_covered = bool(np.any(positive[indices] & (source[indices] == 0)))
        reserve_covered = bool(np.any(positive[indices] & (source[indices] == 1)))
        if not target_ids:
            state[indices] = 0  # ABSENT
        elif main_covered:
            state[indices] = 1  # MAIN_COVERED
        elif reserve_covered:
            state[indices] = 2  # RESERVE_COVERED
        else:
            state[indices] = 3  # PRESENT_UNCOVERED
        null[indices] = (state[indices] == 0) | (state[indices] == 3)
    return positive, null, state


def feature_scores(text: str, spec: list[float], data: dict) -> dict[str, np.ndarray]:
    query = finite_vector(spec, 512).reshape(1, -1)
    query = unit_rows(query)[0]
    clip = unit_rows(data["clip"])
    history = unit_rows(data["history_clip"])
    appearance = np.sum(clip * history, axis=1).astype(np.float32)
    return {
        "clip_text_similarity": (clip @ query).astype(np.float32),
        "appearance_similarity": appearance,
        "geometry_similarity": geometry_score(text, data["geometry"]),
        "objectness": finite_vector(data["objectness"], len(data["frame"])),
        "tracklet_history_similarity": (history @ query).astype(np.float32),
        "source_reserve_indicator": (data["source"] == 1).astype(np.float32),
        # The bank exposes one history EMA, so this is an explicit alias
        # rather than a hidden second appearance feature.
        "observation_history_similarity": appearance.copy(),
        "motion_similarity": motion_score(text, data["motion"]),
    }


def summarize(scores: dict[str, np.ndarray], labels: np.ndarray,
              frame: np.ndarray, source: np.ndarray,
              null: np.ndarray,
              unit: np.ndarray | None = None) -> dict:
    # A raw frame number is not globally unique: every query reuses the same
    # video bank, and different videos can have the same frame numbers.  The
    # pooled report therefore groups by (query unit, frame), while the
    # per-query call below uses a zero unit.
    if unit is None:
        unit = np.zeros(len(frame), np.int64)
    unit = np.asarray(unit, np.int64).reshape(-1)
    if len(unit) != len(frame):
        raise ValueError("unit/frame length mismatch")
    frame_key = unit * 1_000_000 + np.asarray(frame, np.int64)
    frame_counts = []
    positive_frame_count = 0
    top1_hits = defaultdict(int)
    top5_hits = defaultdict(int)
    positive_ranks = defaultdict(list)
    source_top1 = defaultdict(lambda: {0: [0, 0], 1: [0, 0]})
    source_top5 = defaultdict(lambda: {0: [0, 0], 1: [0, 0]})
    null_max = defaultdict(list)
    for frame_id in np.unique(frame_key).tolist():
        indices = np.flatnonzero(frame_key == int(frame_id))
        frame_counts.append(len(indices))
        positive_indices = indices[labels[indices]]
        is_null = bool(null[indices][0]) if len(indices) else False
        for name, values in scores.items():
            if is_null:
                null_max[name].append(float(np.max(values[indices])))
            order = indices[np.argsort(-values[indices], kind="stable")]
            if len(positive_indices):
                top1_hits[name] += int(
                    bool(np.intersect1d(order[:1], positive_indices).size))
                top5_hits[name] += int(
                    bool(np.intersect1d(order[:5], positive_indices).size))
                rank = {int(row): position
                        for position, row in enumerate(order, 1)}
                positive_ranks[name].extend(
                    rank[int(row)] for row in positive_indices)
            for source_id in (0, 1):
                source_indices = indices[source[indices] == source_id]
                if not len(source_indices):
                    continue
                source_order = source_indices[np.argsort(
                    -values[source_indices], kind="stable")]
                selected = source_order[:1]
                source_top1[name][source_id][0] += int(
                    labels[selected].sum())
                source_top1[name][source_id][1] += len(selected)
                selected = source_order[:5]
                source_top5[name][source_id][0] += int(
                    labels[selected].sum())
                source_top5[name][source_id][1] += len(selected)
        if len(positive_indices):
            positive_frame_count += 1
    feature_report = {}
    for name, values in scores.items():
        source_stats = {}
        for source_id, source_name in ((0, "main"), (1, "reserve")):
            source_mask = source == source_id
            source_stats[source_name] = {
                "candidate_count": int(source_mask.sum()),
                "positive_count": int((source_mask & labels).sum()),
                "positive_rate": float((source_mask & labels).sum() /
                                       max(1, source_mask.sum())),
                "top1_precision": float(
                    source_top1[name][source_id][0] /
                    max(1, source_top1[name][source_id][1])),
                "top1_correct": int(source_top1[name][source_id][0]),
                "top1_selected": int(source_top1[name][source_id][1]),
                "top5_precision": float(
                    source_top5[name][source_id][0] /
                    max(1, source_top5[name][source_id][1])),
                "top5_correct": int(source_top5[name][source_id][0]),
                "top5_selected": int(source_top5[name][source_id][1]),
            }
        feature_report[name] = {
            "roc_auc": roc_auc(values, labels),
            "pr_auc": average_precision(values, labels),
            "positive": scalar_stats(values[labels]),
            "negative": scalar_stats(values[~labels]),
            "top1_frame_recall": float(top1_hits[name] /
                                       max(1, positive_frame_count)),
            "top5_frame_recall": float(top5_hits[name] /
                                       max(1, positive_frame_count)),
            "top1_hit_count": int(top1_hits[name]),
            "top5_hit_count": int(top5_hits[name]),
            "positive_rank": rank_stats(positive_ranks[name]),
            "null_frame_highest_score": scalar_stats(null_max[name]),
            "source_internal": source_stats,
        }
    null_frame_count = sum(
        bool(null[np.flatnonzero(frame_key == value)][0])
        for value in np.unique(frame_key).tolist())
    return {
        "candidate_count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "negative_count": int((~labels).sum()),
        "positive_frame_count": int(positive_frame_count),
        "frame_count": int(len(np.unique(frame))),
        "candidate_per_frame": scalar_stats(frame_counts),
        "null_frame_count": int(null_frame_count),
        "features": feature_report,
        "_positive_frame_count": int(positive_frame_count),
        "_null_max_values": {
            name: [float(value) for value in values]
            for name, values in null_max.items()
        },
    }


def merge_arrays(chunks: list[dict]) -> dict:
    result = {}
    for name in ("labels", "frame", "source", "null",
                 "unit_query", "unit_video"):
        result[name] = np.concatenate([chunk[name] for chunk in chunks])
    for feature in FEATURE_NAMES:
        result[feature] = np.concatenate([
            chunk["scores"][feature] for chunk in chunks])
    return result


def summarize_pooled(scores: dict[str, np.ndarray], labels: np.ndarray,
                     frame: np.ndarray, source: np.ndarray,
                     null: np.ndarray, unit_query: np.ndarray,
                     unit_video: np.ndarray,
                     chunks: list[dict]) -> dict:
    """Summarize pooled candidates without repeating per-frame sorting.

    Per-query frame ranking is already computed once while constructing each
    chunk. Reusing those counters avoids treating equal raw frame IDs from
    different queries as one frame and avoids a second expensive sort pass.
    """
    # Keep all three tuple components in the key even though query_index is
    # currently globally unique in this manifest. This prevents an accidental
    # cross-video merge if a future manifest reuses query indices.
    frame_key = (np.asarray(unit_query, np.int64) * 1_000_000_000_000 +
                 np.asarray(unit_video, np.int64) * 1_000_000_000 +
                 np.asarray(frame, np.int64))
    frame_counts = np.unique(frame_key, return_counts=True)[1]
    local_summaries = [chunk["local_summary"] for chunk in chunks]
    positive_frame_count = sum(
        int(summary["_positive_frame_count"]) for summary in local_summaries)
    null_frame_count = sum(
        int(summary["null_frame_count"]) for summary in local_summaries)
    feature_report = {}
    for name, values in scores.items():
        local_features = [summary["features"][name]
                          for summary in local_summaries]
        source_stats = {}
        for source_id, source_name in ((0, "main"), (1, "reserve")):
            candidate_count = sum(
                int(value["source_internal"][source_name]["candidate_count"])
                for value in local_features)
            positive_count = sum(
                int(value["source_internal"][source_name]["positive_count"])
                for value in local_features)
            top1_correct = sum(
                int(value["source_internal"][source_name]["top1_correct"])
                for value in local_features)
            top1_selected = sum(
                int(value["source_internal"][source_name]["top1_selected"])
                for value in local_features)
            top5_correct = sum(
                int(value["source_internal"][source_name]["top5_correct"])
                for value in local_features)
            top5_selected = sum(
                int(value["source_internal"][source_name]["top5_selected"])
                for value in local_features)
            source_stats[source_name] = {
                "candidate_count": candidate_count,
                "positive_count": positive_count,
                "positive_rate": float(positive_count /
                                         max(1, candidate_count)),
                "top1_precision": float(top1_correct /
                                         max(1, top1_selected)),
                "top1_correct": top1_correct,
                "top1_selected": top1_selected,
                "top5_precision": float(top5_correct /
                                         max(1, top5_selected)),
                "top5_correct": top5_correct,
                "top5_selected": top5_selected,
            }
        null_values = np.concatenate([
            np.asarray(summary["_null_max_values"].get(name, []), np.float32)
            for summary in local_summaries
        ]) if local_summaries else np.zeros(0, np.float32)
        feature_report[name] = {
            "roc_auc": roc_auc(values, labels),
            "pr_auc": average_precision(values, labels),
            "positive": scalar_stats(values[labels]),
            "negative": scalar_stats(values[~labels]),
            "top1_frame_recall": float(sum(
                int(value["top1_hit_count"]) for value in local_features) /
                max(1, positive_frame_count)),
            "top5_frame_recall": float(sum(
                int(value["top5_hit_count"]) for value in local_features) /
                max(1, positive_frame_count)),
            "top1_hit_count": sum(int(value["top1_hit_count"])
                                  for value in local_features),
            "top5_hit_count": sum(int(value["top5_hit_count"])
                                  for value in local_features),
            "positive_rank": combine_stats([
                value["positive_rank"] for value in local_features]),
            "null_frame_highest_score": scalar_stats(null_values),
            "source_internal": source_stats,
        }
    return {
        "candidate_count": int(len(labels)),
        "positive_count": int(labels.sum()),
        "negative_count": int((~labels).sum()),
        "positive_frame_count": int(positive_frame_count),
        "frame_count": int(len(np.unique(frame_key))),
        "candidate_per_frame": scalar_stats(frame_counts),
        "null_frame_count": int(null_frame_count),
        "features": feature_report,
    }


def macro_auc(query_reports: list[dict], feature: str, metric: str):
    values = [report["features"][feature][metric]
              for report in query_reports
              if report["features"][feature][metric] is not None]
    return float(np.mean(values)) if values else None


def render_markdown(report: dict) -> str:
    lines = [
        "# RMOT frozen candidate separability diagnostic",
        "",
        "- No checkpoint, training, threshold selection, or TrackEval was used.",
        f"- Manifest: {report['provenance']['manifest']}",
        f"- Manifest SHA256: {report['provenance']['manifest_sha256']}",
        f"- Queries: {report['provenance']['query_count']} "
        f"(calibration {report['provenance']['calibration_query_count']}, "
        f"screening {report['provenance']['screening_query_count']})",
        "- Scores are threshold-free diagnostics; calibration/screening are reported separately.",
        "",
    ]
    for split in ("all", "calibration", "screening"):
        section = report["splits"][split]
        lines.extend([
            f"## {split}",
            "",
            f"Candidates: {section['candidate_count']}, positives: "
            f"{section['positive_count']}, frames: {section['frame_count']}, "
            f"NULL frames: {section['null_frame_count']}.",
            "",
            "| feature | ROC-AUC | PR-AUC | top-1 frame recall | top-5 frame recall | mean positive rank | NULL max mean |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for name in FEATURE_NAMES:
            value = section["features"][name]
            positive_rank = value["positive_rank"].get("mean", float("nan"))
            null_max = value["null_frame_highest_score"].get("mean", float("nan"))
            lines.append(
                f"| {name} | {value['roc_auc'] if value['roc_auc'] is not None else 'NA'} "
                f"| {value['pr_auc'] if value['pr_auc'] is not None else 'NA'} "
                f"| {value['top1_frame_recall']:.4f} | {value['top5_frame_recall']:.4f} "
                f"| {positive_rank:.2f} | {null_max:.4f} |"
            )
        lines.extend([
            "",
            "| feature | main top-1 P | reserve top-1 P | main top-5 P | reserve top-5 P |",
            "|---|---:|---:|---:|---:|",
        ])
        for name in FEATURE_NAMES:
            source = section["features"][name]["source_internal"]
            lines.append(
                f"| {name} | {source['main']['top1_precision']:.4f} "
                f"| {source['reserve']['top1_precision']:.4f} "
                f"| {source['main']['top5_precision']:.4f} "
                f"| {source['reserve']['top5_precision']:.4f} |"
            )
        lines.append("")
    lines.extend([
        "## Decision",
        "",
        f"{report['decision']['status']}: {report['decision']['reason']}",
        "",
        "appearance_similarity and observation_history_similarity are an explicit "
        "alias because the frozen bank contains one history-EMA appearance view; "
        "they must not be double-counted in a later scorer.",
        "",
        "The JSON file contains per-query diagnostics, pooled split statistics, "
        "bank hashes, and the exact feature definitions.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest",
                        default="outputs/l19/protocol/kitti_fast_eval_manifest.json")
    parser.add_argument("--bank-root", default="outputs/l19/dual_banks_features")
    parser.add_argument("--out",
                        default="outputs/l20/eval/candidate_separability_baseline")
    args = parser.parse_args()

    manifest_path = resolve_path(args.manifest)
    bank_root = resolve_path(args.bank_root)
    out_root = resolve_path(args.out)
    if out_root.exists():
        raise FileExistsError(
            f"refusing to overwrite existing diagnostic: {out_root}")
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = sha256_file(manifest_path)
    if manifest.get("selection_uses_model_scores", True):
        raise ValueError(
            "candidate diagnostic requires a score-independent manifest")
    rows = sorted(manifest["queries"],
                  key=lambda value: int(value["query_index"]))
    if len(rows) != 160:
        raise ValueError(
            f"expected the fixed 160-query manifest, got {len(rows)}")
    metadata = load_metadata()
    videos = sorted({row["video"] for row in rows})
    video_codes = {video: index + 1 for index, video in enumerate(videos)}
    banks = {}
    for video in videos:
        banks[video] = load_bank(bank_root / "kitti" / f"{video}.pt")

    split_chunks = defaultdict(list)
    query_reports = []
    for row in rows:
        key = (row["video"], row["expression"])
        if key not in metadata:
            raise KeyError(f"manifest expression missing from metadata: {key}")
        entry = metadata[key]
        bank = banks[row["video"]]
        labels, null, state = query_targets(bank, entry)
        text = str(entry.get("sentence", entry.get("expression", "")))
        scores = feature_scores(text, entry["spec"], bank)
        local_summary = summarize(
            scores, labels, bank["frame"], bank["source"], null)
        chunk = {
            "labels": labels, "null": null, "state": state,
            "frame": bank["frame"], "source": bank["source"],
            "unit_query": np.full(
                len(labels), int(row["query_index"]), np.int64),
            "unit_video": np.full(
                len(labels), int(video_codes[row["video"]]), np.int64),
            "scores": scores, "local_summary": local_summary,
        }
        split = str(row["split"])
        if split not in {"calibration", "screening"}:
            raise ValueError(f"invalid fast manifest split: {split}")
        split_chunks[split].append(chunk)
        query_summary = {
            "query_index": int(row["query_index"]),
            "video": row["video"], "expression": row["expression"],
            "split": split, "sentence": text,
            "positive_count": int(labels.sum()),
            "candidate_count": int(len(labels)),
            "null_frame_count": int(np.count_nonzero([
                bool(null[np.flatnonzero(bank["frame"] == value)][0])
                for value in np.unique(bank["frame"]).tolist()
            ])),
            "features": local_summary["features"],
        }
        query_reports.append(query_summary)

    all_chunks = split_chunks["calibration"] + split_chunks["screening"]
    split_reports = {}
    for split, chunks in (
            ("all", all_chunks), ("calibration", split_chunks["calibration"]),
            ("screening", split_chunks["screening"])):
        pooled = merge_arrays(chunks)
        pooled_scores = {name: pooled[name] for name in FEATURE_NAMES}
        summary = summarize_pooled(
            pooled_scores, pooled["labels"], pooled["frame"],
            pooled["source"], pooled["null"], pooled["unit_query"],
            pooled["unit_video"], chunks)
        query_values = [
            value for value in query_reports
            if split == "all" or value["split"] == split
        ]
        summary["query_count"] = len(query_values)
        summary["macro_query_auc"] = {
            name: macro_auc(query_values, name, "roc_auc")
            for name in FEATURE_NAMES
        }
        summary["macro_query_pr_auc"] = {
            name: macro_auc(query_values, name, "pr_auc")
            for name in FEATURE_NAMES
        }
        split_reports[split] = summary

    usable = [
        value["roc_auc"]
        for name, value in split_reports["all"]["features"].items()
        if name != "source_reserve_indicator" and value["roc_auc"] is not None
    ]
    near_chance = bool(usable) and all(
        0.45 <= value <= 0.55 for value in usable)
    if near_chance:
        status = "STOP_AND_REAUDIT"
        reason = (
            "all non-source cues have pooled ROC-AUC in [0.45, 0.55]; "
            "do not design a scorer before rechecking expression/GT, frame, "
            "candidate coverage, track IDs, and source alignment"
        )
    else:
        status = "BASIC_SIGNAL_EXISTS"
        reason = (
            "at least one non-source frozen cue separates positives from negatives; "
            "a minimal scorer may be considered, but only with independent "
            "static/motion/correspondence outputs and no grouping/membership/NULL head"
        )

    report = {
        "format": "locatemot-rmot-candidate-separability-v1",
        "provenance": {
            "project_root": str(ROOT), "manifest": str(manifest_path),
            "manifest_sha256": manifest_sha, "query_count": len(rows),
            "calibration_query_count": sum(
                row["split"] == "calibration" for row in rows),
            "screening_query_count": sum(
                row["split"] == "screening" for row in rows),
            "selection_uses_model_scores": manifest["selection_uses_model_scores"],
            "bank_root": str(bank_root),
            "pooled_frame_unit": "(query_index, video_id, frame_id)",
            "video_code_map": video_codes,
            "bank_files": {
                video: {
                    "path": str(bank_root / "kitti" / f"{video}.pt"),
                    "sha256": banks[video]["bank_sha256"],
                    "labels_sha256": banks[video]["labels_sha256"],
                    "metadata": banks[video]["metadata"],
                }
                for video in videos
            },
            "gt_used_for": "threshold-free separability diagnostics only",
            "checkpoint_used": False, "training_used": False,
            "trackeval_used": False, "official_eval_used": False,
        },
        "feature_contract": {
            "clip_text_similarity": "cosine(query spec, current clip)",
            "appearance_similarity": "cosine(current clip, history clip EMA)",
            "geometry_similarity": "explicit left/right/center/top/bottom/size word cue over geometry",
            "objectness": "frozen bank objectness scalar",
            "tracklet_history_similarity": "cosine(query spec, history clip EMA)",
            "source_reserve_indicator": "1 for reserve, 0 for main; diagnostic only, never a scorer input",
            "observation_history_similarity": "same frozen current/history EMA cosine as appearance_similarity",
            "motion_similarity": "explicit motion/direction word cue over frozen motion feature",
            "label": "current-frame candidate_gt in expression target IDs; no historical membership",
            "null": "ABSENT or PRESENT_UNCOVERED frame from current-frame coverage only",
        },
        "splits": split_reports,
        "queries": query_reports,
        "decision": {"status": status, "reason": reason},
    }
    out_root.mkdir(parents=True, exist_ok=False)
    (out_root / "diagnostic.json").write_text(
        json.dumps(report, indent=2) + "\n")
    (out_root / "diagnostic.md").write_text(render_markdown(report))
    print(json.dumps({
        "status": status, "output": str(out_root),
        "manifest_sha256": manifest_sha, "queries": len(rows),
        "all_auc": {
            name: value["roc_auc"]
            for name, value in split_reports["all"]["features"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()
