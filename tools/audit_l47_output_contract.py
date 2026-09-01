#!/usr/bin/env python3
"""Train/held-out-video audit for the L47 teacher output contract.

The audit reads only the 15 Refer-KITTI train videos and their expression-level
labels.  The fixed fast manifest is hashed for provenance, but its screening
labels are never loaded.  Full candidate-set counts are collected per video;
L29 replay statistics are deliberately computed on a small deterministic,
cross-video audit sample so that this contract check does not become another
large evaluation job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l29_frame_membership_set_decoder import (  # noqa: E402
    L29FrameMembershipSetDecoder,
)
from tools.audit_l28_identity_bank import load_labels  # noqa: E402
from tools.train_l26_crossmodal_adapter import (  # noqa: E402
    FAST,
    SPLIT,
    V5,
    load_expressions,
)
from tools.train_l28_track_set_decoder import state_at  # noqa: E402

L19 = ROOT / "outputs/l19/dual_banks_features/kitti"
L28 = ROOT / "outputs/l28/track_sequence_bank_final"
L29 = ROOT / "outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"
OUT = ROOT / "outputs/l47/audit"
REPORT = ROOT / "reports/l47_output_contract_audit.md"

ALL_TRAIN_VIDEOS = (
    "0000", "0001", "0002", "0003", "0006", "0007", "0008", "0009",
    "0010", "0012", "0014", "0015", "0016", "0017", "0020",
)
# Three complete train videos are withheld from fitting.  In particular, the
# smoke cannot silently degenerate to the video-0000-only L46 protocol.
FIT_VIDEOS = ALL_TRAIN_VIDEOS[:12]
VAL_VIDEOS = ALL_TRAIN_VIDEOS[12:]
FROZEN_L29_THRESHOLD = -1.1392689042308812


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def finite(value) -> bool:
    return bool(torch.isfinite(torch.as_tensor(value).float()).all())


def load_bank(video: str):
    path = L19 / f"{video}.pt"
    blob = torch.load(path, map_location="cpu", weights_only=False)
    tensors = blob["tensors"]
    labels, label_path = load_labels(path, int(tensors["track_id"].numel()), tensors=tensors)
    return {
        "path": path,
        "metadata": blob.get("metadata", {}),
        "box": tensors["box"].float(),
        "frame": tensors["frame"].long(),
        "candidate_index": tensors["candidate_index"].long(),
        "track": tensors["track_id"].long(),
        "pool": tensors["pool_id"].long(),
        "objectness": tensors["objectness"].float(),
        "clip": tensors["clip"].float(),
        "history_clip": tensors["history_clip"].float(),
        "geometry": tensors["geometry"].float(),
        "motion": tensors["motion"].float(),
        "context": tensors["context"].float(),
        "lifecycle": tensors["lifecycle"].float(),
        "frame_ids": tensors["frame_ids"].long(),
        "ptr": tensors["frame_ptr"].long(),
        "labels": labels,
        "label_path": str(label_path),
    }


def load_queries():
    train = {str(x) for x in json.loads(SPLIT.read_text())["kitti_v2"]["train"]}
    text_manifest = json.loads((V5 / "text_manifest.json").read_text())["expressions"]
    text_index = {(str(x["video"]), str(x["expression"])): int(x["query_index"])
                  for x in text_manifest}
    rows = []
    for row in load_expressions():
        video = str(row["video"])
        key = (video, str(row["expression"]))
        if video not in train or key not in text_index:
            continue
        rows.append({
            "video": video,
            "expression": str(row["expression"]),
            "sentence": str(row.get("sentence", row["expression"])),
            "query_index": int(text_index[key]),
            "target": {int(k): {str(x) for x in values}
                       for k, values in row.get("label", {}).items()},
        })
    if len(rows) != 7757:
        raise AssertionError(f"expected 7757 train expressions, found {len(rows)}")
    return rows


def valid_track_indices(cache, cutoff: int):
    ptr = cache["track_ptr"].numpy()
    frames = cache["obs_frame"].numpy()
    return [i for i in range(len(ptr) - 1)
            if np.any(frames[int(ptr[i]):int(ptr[i + 1])] <= int(cutoff))]


def unit_labels(bank, query, frame_index):
    begin, end = int(bank["ptr"][frame_index]), int(bank["ptr"][frame_index + 1])
    frame = int(bank["frame_ids"][frame_index])
    targets = query["target"].get(frame, set())
    y = np.asarray([
        bank["labels"][row] is not None and str(bank["labels"][row]) in targets
        for row in range(begin, end)
    ], dtype=bool)
    return frame, begin, end, targets, y


def sample_units_for_video(bank, queries, limit_per_video=2):
    """Pick deterministic category-diverse audit units without screening data."""
    buckets = {"multi_positive": [], "positive": [], "inactive": [], "other": []}
    for query in queries:
        for fi in range(len(bank["frame_ids"])):
            frame, begin, end, targets, y = unit_labels(bank, query, fi)
            category = (
                "multi_positive" if int(y.sum()) > 1 else
                "positive" if bool(y.any()) else
                "inactive" if not targets else "other"
            )
            if len(buckets[category]) < limit_per_video:
                buckets[category].append((query, fi, y))
            if all(len(values) >= limit_per_video for values in buckets.values()):
                return [x for name in buckets for x in buckets[name]]
    return [x for name in buckets for x in buckets[name]]


def hard_indices(y, objectness, score, prelimit=96, hard_limit=24):
    neg = np.flatnonzero(~np.asarray(y, dtype=bool))
    if not len(neg):
        return np.empty(0, dtype=np.int64)
    pre = neg[np.argsort(-np.asarray(objectness)[neg], kind="stable")[:prelimit]]
    return pre[np.argsort(-np.asarray(score)[pre], kind="stable")[:hard_limit]]


def distribution(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "q10": None, "q90": None}
    x = np.asarray(values, dtype=np.float64)
    return {
        "count": int(x.size), "mean": float(x.mean()),
        "median": float(np.median(x)), "q10": float(np.quantile(x, .10)),
        "q90": float(np.quantile(x, .90)),
    }


def replay_teacher(teacher, cache, bank, query, frame_index, text_hidden, text_mask):
    frame, begin, end, targets, y = unit_labels(bank, query, frame_index)
    obs, obs_mask, obs_time, _, _ = state_at(cache, frame, history=8)
    with torch.inference_mode():
        encoded = teacher.encode_observations(obs, obs_mask, obs_time)
        out = teacher.forward_encoded(
            encoded, encoded[1], text_hidden[query["query_index"]],
            text_mask[query["query_index"]],
        )
    valid = valid_track_indices(cache, frame)
    track_to_score = {
        int(track): float(score)
        for track, score in zip(cache["track_ids"][valid].tolist(),
                                out["current_membership_logits"].float().tolist())
    }
    track_rows = bank["track"][begin:end].tolist()
    teacher_score = np.asarray(
        [track_to_score.get(int(track), np.nan) for track in track_rows],
        dtype=np.float32,
    )
    if not np.isfinite(teacher_score).all():
        raise RuntimeError(f"missing/nonfinite L29 mapping {query['video']}/{frame}")
    hard = hard_indices(y, bank["objectness"][begin:end].numpy(), teacher_score)
    pos = np.flatnonzero(y)
    if len(pos) and len(hard):
        strict = float(teacher_score[pos].min() - teacher_score[hard].max())
        correct = int((teacher_score[pos, None] > teacher_score[hard][None, :]).sum())
        pairs = int(len(pos) * len(hard))
    else:
        strict, correct, pairs = None, 0, 0
    order = np.argsort(-teacher_score, kind="stable")
    return {
        "video": query["video"], "expression": query["expression"],
        "query_index": int(query["query_index"]), "frame": frame,
        "candidate_count": int(end - begin), "positive_count": int(y.sum()),
        "negative_count": int((~y).sum()), "target_ids": sorted(targets),
        "teacher_top1": float(bool(y[order[:1]].any())) if len(pos) else None,
        "teacher_top5": float(bool(y[order[:5]].any())) if len(pos) else None,
        "teacher_recall_at_frozen_threshold": float(
            ((teacher_score >= FROZEN_L29_THRESHOLD) & y).sum() / max(1, int(y.sum()))
        ) if len(pos) else None,
        "teacher_positive_min_margin": strict,
        "teacher_hard_violation": None if strict is None else float(strict < 0),
        "teacher_hard_pairs": pairs,
        "teacher_hard_pair_correct_fraction": correct / max(1, pairs),
        "teacher_score_stats": distribution(teacher_score.tolist()),
        "hard_candidate_local_indices": [int(x) for x in hard.tolist()],
        "hard_candidate_track_ids": [int(bank["track"][begin + x]) for x in hard.tolist()],
        "semantic_shortcuts_used": [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-sample-per-video", type=int, default=2)
    parser.add_argument("--out", default=str(OUT / "output_contract.json"))
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    queries = load_queries()
    by_video = defaultdict(list)
    for query in queries:
        by_video[query["video"]].append(query)
    text = torch.load(V5 / "text_tokens.pt", map_location="cpu", weights_only=False)
    text_hidden = text["token_hidden"].float()
    text_mask = text["attention_mask"].bool()

    teacher = L29FrameMembershipSetDecoder()
    teacher_state = torch.load(L29, map_location="cpu", weights_only=False)
    teacher.load_state_dict(teacher_state["model"], strict=True)
    teacher.eval()

    required = (
        "box", "frame", "candidate_index", "track", "pool", "clip",
        "history_clip", "geometry", "motion", "context", "lifecycle",
        "frame_ids", "ptr", "labels",
    )
    nonfinite = Counter()
    missing_fields = {}
    duplicate_keys = 0
    missing_cache = []
    cache_alignment_errors = []
    population = {"fit": Counter(), "validation": Counter()}
    candidate_sizes = {"fit": [], "validation": []}
    per_video = {}
    sample_replays = {"fit": [], "validation": []}
    sample_margins = {"fit": [], "validation": []}
    sample_hard_violations = {"fit": [], "validation": []}
    category_counts = {"fit": Counter(), "validation": Counter()}

    for video in ALL_TRAIN_VIDEOS:
        split_name = "fit" if video in FIT_VIDEOS else "validation"
        bank = load_bank(video)
        cache_path = L28 / f"{video}.pt"
        if not cache_path.exists():
            missing_cache.append(str(cache_path))
            continue
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        missing = [key for key in required if key not in bank]
        if missing:
            missing_fields[video] = missing
        for field in ("box", "clip", "history_clip", "geometry", "motion",
                      "context", "lifecycle"):
            if not finite(bank[field]):
                nonfinite[f"{video}:{field}"] += 1

        # Stable row identity contains the observation row, while pool and
        # candidate_index are retained as provenance and never as semantics.
        seen = set()
        for row, track in enumerate(bank["track"].tolist()):
            key = (video, int(bank["frame"][row]), int(track), int(row))
            if key in seen:
                duplicate_keys += 1
            seen.add(key)

        rows_by_track = defaultdict(list)
        for row, track in enumerate(bank["track"].tolist()):
            rows_by_track[int(track)].append(row)
        for values in rows_by_track.values():
            values.sort(key=lambda row: (int(bank["frame"][row]), row))
        cptr = cache["track_ptr"].numpy()
        cframes = cache["obs_frame"].numpy()
        for ti, track in enumerate(cache["track_ids"].tolist()):
            cb, ce = int(cptr[ti]), int(cptr[ti + 1])
            cached_frames = [int(x) for x in cframes[cb:ce].tolist()]
            bank_frames = [int(bank["frame"][row]) for row in rows_by_track.get(int(track), [])]
            if cached_frames != bank_frames:
                cache_alignment_errors.append({
                    "video": video, "track_id": int(track),
                    "cache_frames": cached_frames, "bank_frames": bank_frames,
                })

        counts = population[split_name]
        q_video = by_video[video]
        sample_units = sample_units_for_video(bank, q_video, args.teacher_sample_per_video)
        sample_lookup = {(id(query), int(fi)): (query, y)
                         for query, fi, y in sample_units}
        for query in q_video:
            for fi in range(len(bank["frame_ids"])):
                frame, begin, end, targets, y = unit_labels(bank, query, fi)
                n = end - begin
                counts["frame_units"] += 1
                counts["candidate_rows"] += n
                counts["positive_rows"] += int(y.sum())
                counts["negative_rows"] += int((~y).sum())
                counts["positive_negative_pairs"] += int(y.sum()) * int((~y).sum())
                counts["positive_frame_units"] += int(y.any())
                counts["multi_positive_frame_units"] += int(y.sum() > 1)
                counts["target_frame_units"] += int(bool(targets))
                counts["inactive_or_null_frame_units"] += int(not targets)
                counts["missing_target_frame_units"] += int(bool(targets) and not y.any())
                candidate_sizes[split_name].append(n)
                category = (
                    "multi_positive" if int(y.sum()) > 1 else
                    "positive" if bool(y.any()) else
                    "inactive" if not targets else "other"
                )
                category_counts[split_name][category] += 1
                item = sample_lookup.get((id(query), fi))
                if item is not None and len(sample_replays[split_name]) < 64:
                    replay = replay_teacher(teacher, cache, bank, query, fi,
                                            text_hidden, text_mask)
                    sample_replays[split_name].append(replay)
                    if replay["teacher_positive_min_margin"] is not None:
                        sample_margins[split_name].append(replay["teacher_positive_min_margin"])
                        sample_hard_violations[split_name].append(replay["teacher_hard_violation"])

        per_video[video] = {
            "split": split_name,
            "expressions": len(q_video),
            "candidate_rows": int(bank["track"].numel()),
            "frame_count": int(len(bank["frame_ids"])),
            "track_count": int(len(cache["track_ids"])),
            "candidate_count_distribution": distribution(
                (bank["ptr"][1:] - bank["ptr"][:-1]).tolist()
            ),
            "bank_path": str(bank["path"].resolve()),
            "bank_sha256": sha256(bank["path"]),
            "label_path": bank["label_path"],
            "cache_path": str(cache_path.resolve()),
            "cache_sha256": sha256(cache_path),
            "cache_feature_shape": list(cache["obs_features"].shape),
        }
        del cache, bank

    text_finite = finite(text_hidden)
    fast_manifest = json.loads(FAST.read_text())
    audit_failed = bool(
        missing_fields or missing_cache or cache_alignment_errors or duplicate_keys
        or nonfinite or not text_finite
    )
    payload = {
        "format": "locatemot-l47-output-contract-audit-v1",
        "stage": "L47-A",
        "project_root": str(ROOT),
        "started_at": started,
        "completed_at": time.time(),
        "split_policy": {
            "fit_videos": list(FIT_VIDEOS),
            "validation_videos": list(VAL_VIDEOS),
            "fit_video_count": len(FIT_VIDEOS),
            "validation_video_count": len(VAL_VIDEOS),
            "video_disjoint": not bool(set(FIT_VIDEOS) & set(VAL_VIDEOS)),
            "fit_expressions": int(sum(len(by_video[v]) for v in FIT_VIDEOS)),
            "validation_expressions": int(sum(len(by_video[v]) for v in VAL_VIDEOS)),
            "smoke_minimum_distinct_videos": 8,
        },
        "population_counts": {name: dict(value) for name, value in population.items()},
        "category_counts": {name: dict(value) for name, value in category_counts.items()},
        "candidate_count_distribution": {
            name: distribution(values) for name, values in candidate_sizes.items()
        },
        "per_video": per_video,
        "row_contract": {
            "emitted_row_key": ["video", "query_index", "frame", "track_id", "observation_row"],
            "observation_row_unique_within_frame": duplicate_keys == 0,
            "candidate_set_is_complete_per_frame": True,
            "labels_are_expression_level_gt_derived": True,
            "same_frame_hard_negative_contract": "all y=0 candidates; efficient teacher-hard is objectness top-96 then teacher score top-24",
            "multi_positive_retained": True,
            "inactive_null_retained": True,
        },
        "teacher": {
            "checkpoint": str(L29.resolve()),
            "sha256": sha256(L29),
            "logit": "L29 current_membership_logits mapped by current track_id",
            "frozen_threshold_for_diagnostic_only": FROZEN_L29_THRESHOLD,
            "sample_unit_policy": "deterministic category-diverse units per train video; no screening labels",
            "fit_sample": {
                "units": len(sample_replays["fit"]),
                "positive_min_margin": distribution(sample_margins["fit"]),
                "hard_violation_rate": float(np.mean(sample_hard_violations["fit"])) if sample_hard_violations["fit"] else None,
                "examples": sample_replays["fit"],
            },
            "validation_sample": {
                "units": len(sample_replays["validation"]),
                "positive_min_margin": distribution(sample_margins["validation"]),
                "hard_violation_rate": float(np.mean(sample_hard_violations["validation"])) if sample_hard_violations["validation"] else None,
                "examples": sample_replays["validation"],
            },
        },
        "feature_contract": {
            "frozen_inputs": ["clip", "history_clip", "geometry", "motion", "context", "lifecycle", "word_level_text_tokens", "L29_current_membership_logit"],
            "semantic_inputs_excluded": ["source_id", "pool_id", "group_id", "state_key", "query_index_as_feature"],
            "source_pool_group_state": "provenance/statistics only",
            "token_span_region_verified": False,
            "static_motion_language_mask_verified": False,
            "motion_language_decomposition": "not claimed",
        },
        "cache_contract": {
            "l19_bank_root": str(L19.resolve()),
            "l28_cache_root": str(L28.resolve()),
            "missing_cache_files": missing_cache,
            "alignment_error_count": len(cache_alignment_errors),
            "alignment_error_examples": cache_alignment_errors[:8],
        },
        "fixed_fast_manifest": {
            "path": str(FAST.resolve()),
            "sha256": sha256(FAST),
            "expected_sha256": "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa",
            "query_count": len(fast_manifest.get("queries", [])),
            "calibration_queries": 64,
            "screening_queries": 96,
            "labels_loaded": False,
            "used_for_training_or_validation": False,
        },
        "finite_checks": {
            "nonfinite_fields": dict(nonfinite),
            "text_hidden_finite": text_finite,
            "duplicate_row_keys": duplicate_keys,
        },
        "screening_gt_used": False,
        "decision": "enter_b0_smoke" if not audit_failed else "incomplete",
        "elapsed_sec": time.time() - started,
    }
    output = Path(args.out)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (output.parent / "README.md").write_text(
        "# L47 output-contract audit\n\n"
        "Train/held-out-video audit only. The fixed fast manifest is hashed for provenance; "
        "screening labels are not loaded. See output_contract.json.\n"
    )
    if audit_failed:
        (output.parent / "INCOMPLETE.md").write_text(
            "# INCOMPLETE\n\n"
            "The L47 cross-video output contract did not pass. No training was started. "
            "See `output_contract.json` for the first actionable audit evidence.\n"
        )
    report_lines = [
        "# Stage L47 — output contract audit", "",
        f"- Decision: **{payload['decision']}**",
        f"- Fit videos: {len(FIT_VIDEOS)}; validation videos: {len(VAL_VIDEOS)}",
        f"- Fit/validation expressions: {payload['split_policy']['fit_expressions']}/{payload['split_policy']['validation_expressions']}",
        f"- Fit frame units/rows: {population['fit']['frame_units']:,}/{population['fit']['candidate_rows']:,}",
        f"- Validation frame units/rows: {population['validation']['frame_units']:,}/{population['validation']['candidate_rows']:,}",
        f"- Fit multi-positive/inactive: {category_counts['fit']['multi_positive']:,}/{category_counts['fit']['inactive']:,}",
        f"- Validation multi-positive/inactive: {category_counts['validation']['multi_positive']:,}/{category_counts['validation']['inactive']:,}",
        f"- Duplicate row keys: {duplicate_keys}; cache alignment errors: {len(cache_alignment_errors)}; missing caches: {len(missing_cache)}",
        f"- L29 fit sample margin: {payload['teacher']['fit_sample']['positive_min_margin']}",
        f"- L29 validation sample margin: {payload['teacher']['validation_sample']['positive_min_margin']}",
        f"- Fast manifest SHA256: `{payload['fixed_fast_manifest']['sha256']}`; screening labels loaded: **False**",
        "",
        "The audit uses expression-level frame-to-GT supervision. Token/span-to-region and "
        "static/motion language masks remain UNALIGNED and are not claimed.",
    ]
    REPORT.write_text("\n".join(report_lines) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
