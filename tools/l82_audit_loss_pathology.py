#!/usr/bin/env python3
"""Fit-only duplicate-positive and loss-pathology audit for L82.

The audit uses the already materialized L82 fit matrix and immutable L69
sidecars.  It does not instantiate a model or open calibration/validation
labels.  Prior L80/L81 validation numbers are imported only as aggregate,
already-frozen context; they are not recomputed or used for selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
MATRIX = ROOT / "outputs/l82/data/frame_query_groups.jsonl"
L69_ROOT = ROOT / "outputs/l69/attempt9/budget40_features/kitti"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
L80_TRACE = ROOT / "outputs/l80/train/r2_loss_probe500/loss_trace.json"
L81_GATE = ROOT / "outputs/l81/eval/semantic_16cal24val/gate_decision.json"
L81_SEMANTIC = ROOT / "outputs/l81/eval/semantic_16cal24val/semantic.json"
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
DATASETS = ("refer_kitti_v1", "refer_kitti_v2")
FIT_CATEGORIES = ("positive", "multi_positive", "inactive", "present_uncovered")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def meta(path: Path, include_hash: bool = True) -> dict[str, Any]:
    path = path.resolve()
    result: dict[str, Any] = {
        "path": str(path), "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "mtime_ns": path.stat().st_mtime_ns if path.is_file() else None,
    }
    if include_hash and path.is_file():
        result["sha256"] = sha256_file(path)
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def finite_number(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise AssertionError(f"nonfinite numeric field: {value!r}")
    return result


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None,
                "p05": None, "p25": None, "p50": None, "p75": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size), "min": float(array.min()), "max": float(array.max()),
        "mean": float(array.mean()), "std": float(array.std()),
        "p05": float(np.quantile(array, .05)), "p25": float(np.quantile(array, .25)),
        "p50": float(np.quantile(array, .50)), "p75": float(np.quantile(array, .75)),
        "p95": float(np.quantile(array, .95)),
    }


def box_iou(left: list[float], right: list[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in left]
    bx1, by1, bx2, by2 = [float(x) for x in right]
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-12, area_a + area_b - inter)


def pair_count(values: list[Any], predicate) -> int:
    return sum(1 for i in range(len(values)) for j in range(i + 1, len(values)) if predicate(i, j))


def summarize_pairs(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"pair_count": 0, "mean": None, "std": None, "p05": None,
                "p50": None, "p95": None}
    values = [float(row["value"]) for row in records]
    result = distribution(values)
    result["pair_count"] = len(values)
    return result


def load_sidecar(video: str) -> list[Any]:
    path = (L69_ROOT / f"{video}.labels.json").resolve()
    payload = json.loads(path.read_text())
    values = payload.get("candidate_gt")
    if not isinstance(values, list):
        raise AssertionError(f"missing candidate_gt list: {path}")
    return values


def loss_trace_summary() -> dict[str, Any]:
    traces = json.loads(L80_TRACE.read_text())
    if not isinstance(traces, list) or len(traces) != 500:
        raise AssertionError("L80 R2 loss trace is not the immutable 500-step trace")
    by_count: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    by_category: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    finite = 0
    for row in traces:
        for key in ("total", "membership_bce", "pairwise", "listwise", "minimum_positive",
                    "positive_floor", "brier"):
            if key in row:
                value = finite_number(row[key])
                by_count[str(int(row.get("positive_count", 0)))][key].append(value)
                by_category[str(row.get("category", "unknown"))][key].append(value)
        if row.get("loss_finite") is True or row.get("gradient_finite") is True:
            finite += 1

    def aggregate(source: dict[str, dict[str, list[float]]]) -> dict[str, Any]:
        return {
            outer: {inner: distribution(values) for inner, values in inner_map.items()}
            for outer, inner_map in source.items()
        }

    return {
        "source": meta(L80_TRACE), "steps": len(traces),
        "rows_with_finite_markers": finite,
        "by_positive_count": aggregate(by_count),
        "by_category": aggregate(by_category),
        "interpretation": {
            "minimum_positive_direct_gradient_scope": "one lowest-scoring positive row per covered unit; listwise/pairwise terms still include every positive",
            "positive_floor_scope": "every positive row in the L80-R2 mean positive-floor term",
            "not_a_model_selection_signal": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L82 pathology output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable] + sys.argv)
    started = time.perf_counter()
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        matrix = read_jsonl(MATRIX)
        if len(matrix) != 2931:
            raise AssertionError(f"expected 2931 L82 frame groups, got {len(matrix)}")
        sidecars: dict[str, list[Any]] = {}
        records: list[dict[str, Any]] = []
        category_stats: dict[tuple[str, str], dict[str, list[float] | int]] = {}
        domain_stats: dict[str, Counter] = {dataset: Counter() for dataset in DATASETS}
        duplicate_records: list[dict[str, Any]] = []
        pair_iou_records: list[dict[str, Any]] = []
        target_records: list[dict[str, Any]] = []
        hard_records: list[dict[str, Any]] = []
        unit_count = 0
        total_candidate_rows = 0
        total_positive_rows = 0
        for group in matrix:
            dataset = str(group["dataset"])
            video = str(group["video"])
            if dataset not in DATASETS:
                raise AssertionError(f"unexpected dataset: {dataset}")
            if video not in sidecars:
                sidecars[video] = load_sidecar(video)
            candidate_gt = sidecars[video]
            offsets = [int(x) for x in group["row_offsets"]]
            boxes = [[finite_number(x) for x in box] for box in group["boxes"]]
            objectness = [finite_number(x) for x in group["objectness"]]
            candidate_indices = [int(x) for x in group["candidate_index"]]
            pools = [int(x) for x in group["pool_id"]]
            tracks = [int(x) for x in group["track_id"]]
            if len(offsets) != int(group["candidate_count"]) or len(boxes) != len(offsets):
                raise AssertionError(f"candidate row count drift: {group['group_key']}")
            if len(set(offsets)) != len(offsets) or offsets != list(range(int(group["begin"]), int(group["end"]))):
                raise AssertionError(f"native row order drift: {group['group_key']}")
            if max(offsets, default=-1) >= len(candidate_gt):
                raise AssertionError(f"sidecar too short for {group['group_key']}")
            total_candidate_rows += len(offsets)
            all_index_dups = len(candidate_indices) - len(set(candidate_indices))
            all_box_groups: dict[bytes, list[int]] = defaultdict(list)
            for index, box in enumerate(boxes):
                all_box_groups[np.asarray(box, dtype=np.float32).tobytes()].append(index)
            for query in group["queries"]:
                unit_count += 1
                labels = [bool(int(x)) for x in query["label_vector"]]
                positive = [i for i, value in enumerate(labels) if value]
                negative = [i for i, value in enumerate(labels) if not value]
                category = str(query["category"])
                if category not in FIT_CATEGORIES:
                    raise AssertionError(f"unknown recomputed category: {category}")
                domain_stats[dataset][category] += 1
                key = (dataset, category)
                if key not in category_stats:
                    category_stats[key] = {
                        "unit_count": 0, "candidate_count": [], "positive_count": [],
                        "negative_count": [], "duplicate_positive_index_count": [],
                        "duplicate_positive_box_count": [], "positive_iou_pairs_50": [],
                        "positive_iou_pairs_70": [], "same_target_iou_pairs_50": [],
                        "same_target_iou_pairs_70": [], "objectness_positive": [],
                        "objectness_negative": [], "hard_negative_objectness": [],
                    }
                stat = category_stats[key]
                stat["unit_count"] += 1
                for field, value in (("candidate_count", len(offsets)), ("positive_count", len(positive)),
                                     ("negative_count", len(negative))):
                    stat[field].append(float(value))
                total_positive_rows += len(positive)
                positive_indices_dup = len(positive) - len({candidate_indices[i] for i in positive})
                positive_box_dup = len(positive) - len({np.asarray(boxes[i], dtype=np.float32).tobytes() for i in positive})
                stat["duplicate_positive_index_count"].append(float(positive_indices_dup))
                stat["duplicate_positive_box_count"].append(float(positive_box_dup))
                stat["objectness_positive"].extend(objectness[i] for i in positive)
                stat["objectness_negative"].extend(objectness[i] for i in negative)
                hard_values = sorted((objectness[i] for i in negative), reverse=True)[:min(16, len(negative))]
                stat["hard_negative_objectness"].extend(hard_values)
                target_set = {str(x) for x in query.get("target_ids", [])}
                target_to_rows: dict[str, list[int]] = defaultdict(list)
                for local_index in positive:
                    value = candidate_gt[offsets[local_index]]
                    if value is not None and str(value) in target_set:
                        target_to_rows[str(value)].append(local_index)
                for target_id, target_rows in target_to_rows.items():
                    target_records.append({
                        "format": "locatemot-l82-fit-positive-target-record-v1",
                        "unit_key": str(query["unit_key"]), "dataset": dataset, "video": video,
                        "frame_id": int(group["frame_id"]), "target_id": target_id,
                        "positive_row_count": len(target_rows),
                        "candidate_indices": [candidate_indices[i] for i in target_rows],
                        "pool_ids": [pools[i] for i in target_rows],
                        "track_ids": [tracks[i] for i in target_rows],
                        "objectness": [objectness[i] for i in target_rows],
                        "evidence_class": "FIT_ONLY_LABEL_DIAGNOSTIC",
                    })
                pos_iou_50 = pos_iou_70 = same_iou_50 = same_iou_70 = 0
                for left in range(len(positive)):
                    for right in range(left + 1, len(positive)):
                        i, j = positive[left], positive[right]
                        value = box_iou(boxes[i], boxes[j])
                        if value >= .50:
                            pos_iou_50 += 1
                        if value >= .70:
                            pos_iou_70 += 1
                        left_ids = {str(candidate_gt[offsets[i]])} if candidate_gt[offsets[i]] is not None else set()
                        right_ids = {str(candidate_gt[offsets[j]])} if candidate_gt[offsets[j]] is not None else set()
                        if left_ids & right_ids:
                            if value >= .50:
                                same_iou_50 += 1
                            if value >= .70:
                                same_iou_70 += 1
                        pair_iou_records.append({
                            "format": "locatemot-l82-fit-positive-pair-v1",
                            "unit_key": str(query["unit_key"]), "frame_id": int(group["frame_id"]),
                            "left_offset": offsets[i], "right_offset": offsets[j], "iou": value,
                            "same_candidate_index": candidate_indices[i] == candidate_indices[j],
                            "same_target_id": bool(left_ids & right_ids),
                            "evidence_class": "FIT_ONLY_LABEL_DIAGNOSTIC",
                        })
                stat["positive_iou_pairs_50"].append(float(pos_iou_50))
                stat["positive_iou_pairs_70"].append(float(pos_iou_70))
                stat["same_target_iou_pairs_50"].append(float(same_iou_50))
                stat["same_target_iou_pairs_70"].append(float(same_iou_70))
                same_index_positive_negative = sum(
                    1 for i in positive for j in negative if candidate_indices[i] == candidate_indices[j])
                high_iou_positive_negative = sum(
                    1 for i in positive for j in negative if box_iou(boxes[i], boxes[j]) >= .50)
                top_hard = sorted(negative, key=lambda i: (-objectness[i], i))[:min(16, len(negative))]
                hard_records.append({
                    "format": "locatemot-l82-fit-hard-negative-record-v1",
                    "unit_key": str(query["unit_key"]), "dataset": dataset, "video": video,
                    "frame_id": int(group["frame_id"]), "category": category,
                    "positive_count": len(positive), "negative_count": len(negative),
                    "hard_negative_count": len(top_hard),
                    "same_candidate_index_positive_negative_pairs": same_index_positive_negative,
                    "positive_negative_iou50_pairs": high_iou_positive_negative,
                    "hard_negative_candidate_indices": [candidate_indices[i] for i in top_hard],
                    "hard_negative_pool_ids": [pools[i] for i in top_hard],
                    "evidence_class": "FIT_ONLY_LABEL_DIAGNOSTIC",
                })
                duplicate_records.append({
                    "format": "locatemot-l82-fit-duplicate-record-v1",
                    "unit_key": str(query["unit_key"]), "dataset": dataset, "video": video,
                    "frame_id": int(group["frame_id"]), "category": category,
                    "candidate_count": len(offsets), "all_duplicate_candidate_index_count": all_index_dups,
                    "positive_duplicate_candidate_index_count": positive_indices_dup,
                    "positive_duplicate_exact_box_count": positive_box_dup,
                    "positive_count": len(positive), "negative_count": len(negative),
                    "pool_counts": dict(Counter(str(pools[i]) for i in range(len(pools)))),
                    "positive_pool_counts": dict(Counter(str(pools[i]) for i in positive)),
                    "same_target_positive_row_counts": {key: len(value) for key, value in target_to_rows.items()},
                    "evidence_class": "FIT_ONLY_LABEL_DIAGNOSTIC",
                })
                records.append({
                    "format": "locatemot-l82-fit-pathology-unit-v1",
                    "unit_key": str(query["unit_key"]), "dataset": dataset, "video": video,
                    "frame_id": int(group["frame_id"]), "category": category,
                    "candidate_count": len(offsets), "positive_count": len(positive),
                    "negative_count": len(negative), "target_count": len(target_set),
                    "candidate_present": bool(positive), "present_uncovered": bool(query["present_uncovered"]),
                    "duplicate_candidate_index_count": all_index_dups,
                    "positive_duplicate_candidate_index_count": positive_indices_dup,
                    "positive_duplicate_exact_box_count": positive_box_dup,
                    "positive_iou_pairs_ge_0_50": pos_iou_50,
                    "positive_iou_pairs_ge_0_70": pos_iou_70,
                    "same_target_iou_pairs_ge_0_50": same_iou_50,
                    "same_target_iou_pairs_ge_0_70": same_iou_70,
                    "same_candidate_index_positive_negative_pairs": same_index_positive_negative,
                    "positive_negative_iou50_pairs": high_iou_positive_negative,
                    "evidence_class": "FIT_ONLY_LABEL_DIAGNOSTIC",
                })

        def summarize_category(stat: dict[str, Any]) -> dict[str, Any]:
            result: dict[str, Any] = {"unit_count": int(stat["unit_count"])}
            for field, values in stat.items():
                if field == "unit_count":
                    continue
                result[field] = distribution([float(x) for x in values])
            return result

        category_summary = {
            f"{dataset}|{category}": summarize_category(stat)
            for (dataset, category), stat in sorted(category_stats.items())
        }
        target_count_values = [int(row["positive_row_count"]) for row in target_records]
        duplicate_positive_units = sum(1 for row in records if row["positive_duplicate_candidate_index_count"] > 0)
        exact_box_duplicate_units = sum(1 for row in records if row["positive_duplicate_exact_box_count"] > 0)
        positive_pair_values = [float(row["iou"]) for row in pair_iou_records]
        target_pair_values = [float(row["iou"]) for row in pair_iou_records if row["same_target_id"]]
        l80_summary = loss_trace_summary()
        l81_context = {
            "gate": json.loads(L81_GATE.read_text()),
            "semantic_summary": {
                key: json.loads(L81_SEMANTIC.read_text()).get(key)
                for key in ("decision", "seed", "evidence_type", "status")
            },
            "use": "immutable aggregate context only; no L81 validation labels were opened by this fit pathology computation",
        }
        summary = {
            "format": "locatemot-l82-duplicate-loss-pathology-v1",
            "status": "complete", "stage": "phase_b_fit_only_pathology",
            "command": command, "cwd": str(ROOT),
            "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745",
            "inputs": {"matrix": meta(MATRIX), "manifest": meta(MANIFEST),
                       "l80_r2_loss_trace": meta(L80_TRACE), "l81_gate_context": meta(L81_GATE),
                       "l81_semantic_context": meta(L81_SEMANTIC),
                       "l69_sidecar_videos": sorted(sidecars)},
            "fit_unit_count": unit_count, "frame_group_count": len(matrix),
            "candidate_row_count": total_candidate_rows, "positive_row_count": total_positive_rows,
            "category_summary": category_summary,
            "target_positive_row_distribution": distribution([float(x) for x in target_count_values]),
            "positive_pair_iou_distribution": distribution(positive_pair_values),
            "same_target_pair_iou_distribution": distribution(target_pair_values),
            "duplicate_summary": {
                "units_with_positive_candidate_index_duplicates": duplicate_positive_units,
                "units_with_positive_exact_box_duplicates": exact_box_duplicate_units,
                "all_unit_duplicate_candidate_index_distribution": distribution([
                    float(row["duplicate_candidate_index_count"]) for row in records]),
                "positive_duplicate_candidate_index_distribution": distribution([
                    float(row["positive_duplicate_candidate_index_count"]) for row in records]),
                "positive_duplicate_exact_box_distribution": distribution([
                    float(row["positive_duplicate_exact_box_count"]) for row in records]),
                "candidate_deletion": False, "candidate_truncation": False,
                "duplicate_rows_retained": True,
            },
            "hard_negative_descriptive_summary": {
                "record_count": len(hard_records),
                "same_candidate_index_positive_negative_pairs": sum(
                    int(row["same_candidate_index_positive_negative_pairs"]) for row in hard_records),
                "positive_negative_iou50_pairs": sum(int(row["positive_negative_iou50_pairs"]) for row in hard_records),
                "metadata": "same-class metadata unavailable; all-negative/query-independent objectness fallback is descriptive only",
            },
            "l80_minimum_positive_pathology": {
                "loss_source": "L80-R2 immutable fit trace",
                "trace_summary": l80_summary,
                "analytical_risk": "a min() term directly selects one positive, while positive_floor/listwise terms are required to keep every positive in the gradient path; repeated target rows make row-level weighting differ from target-bag weighting",
                "target_bag_rows_retained": True,
            },
            "l81_broad_acceptance_context": l81_context,
            "label_boundary": {
                "scope": "fit-only L49 expression-level target_ids plus L69 candidate_gt after native row reconstruction",
                "calibration_labels_opened": False, "validation_labels_opened": False,
                "screening_labels_opened": False, "official_test_labels_opened": False,
            },
            "same_class_hard_negative_metadata": "unavailable; no class inferred from expression",
            "evidence_class": "FIT_ONLY_LABEL_DIAGNOSTIC; best-positive/oracle-like summaries are not model metrics",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "training_run": False,
            "hota_trackeval_run": False, "candidate_deletion": False,
            "candidate_truncation": False, "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED", "failure_root_cause": None,
            "next_action": "run Phase C GroundingDINO candidate-reference interface contract only",
            "elapsed_sec": time.perf_counter() - started,
        }
        write_json(out / "l80_l81_loss_pathology.json", summary)
        (out / "pathology_records.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in records))
        (out / "duplicate_records.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in duplicate_records))
        (out / "target_positive_records.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in target_records))
        (out / "hard_negative_records.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in hard_records))
        write_json(out / "provenance.json", {
            "format": "locatemot-l82-pathology-provenance-v1", "status": "complete",
            "command": command, "inputs": summary["inputs"],
            "outputs": [str(path) for path in (out / "l80_l81_loss_pathology.json",
                         out / "pathology_records.jsonl", out / "duplicate_records.jsonl",
                         out / "target_positive_records.jsonl", out / "hard_negative_records.jsonl")],
            "labels": summary["label_boundary"], "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "training_run": False, "hota_trackeval_run": False,
            "candidate_deletion": False, "candidate_truncation": False,
            "failure_root_cause": None, "next_action": summary["next_action"],
        })
        write_json(out / "status.json", {
            "format": "locatemot-l82-status-v1", "status": "complete",
            "stage": "phase_b", "command": command,
            "outputs": [str(out / "l80_l81_loss_pathology.json")], "failure_root_cause": None,
            "next_action": summary["next_action"], "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "training_run": False, "hota_trackeval_run": False,
        })
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
        return 0
    except Exception:
        (out / "INCOMPLETE.md").write_text(
            "# L82 pathology audit — INCOMPLETE\n\n" + traceback.format_exc() +
            "\nNo model, training, screening, official-test, TrackEval/HOTA, MOT or OVMOT action was run.\n")
        write_json(out / "status.json", {
            "format": "locatemot-l82-status-v1", "status": "incomplete",
            "stage": "phase_b", "command": command,
            "failure_root_cause": "first actionable exception preserved in INCOMPLETE.md",
            "next_action": "fix only the first pathology-contract error and rerun in a new attempt",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "training_run": False,
            "hota_trackeval_run": False,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
