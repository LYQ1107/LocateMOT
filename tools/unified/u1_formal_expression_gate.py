#!/usr/bin/env python3
"""Formal expression-grounding measurement on a frozen query/frame sample.

Every legal calibration/validation query contributes exactly one deterministic
representative labeled frame (the middle frame in sorted non-empty label
frames).  The unit remains query/frame/target, and missing source target IDs
are counted in the denominator instead of being silently converted to
negatives or removed.  This keeps the full query universe finite and makes
the contract reproducible before any adaptation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locatemot.unified.data.legal_scope import (  # noqa: E402
    LEGAL_SPLITS,
    assert_legal_path,
    assert_legal_video,
)
from locatemot.unified.runtime.grounding_inference import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    load_grounding_runtime,
    prediction_arrays,
    run_grounding,
    runtime_metadata,
)

THRESHOLDS = (0.25, 0.50, 0.75)
TOP_KS = (1, 5, 10, 100, 300)
SPLITS = ("calibration", "validation")
STRATA = ("overall", "refer_kitti_v1", "refer_kitti_v2")
PRECISION_TOP_K = 10
SCORE_THRESHOLD = 0.05


def query_records() -> list[dict[str, Any]]:
    query_rows: list[dict[str, Any]] = []
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        root = PROJECT_ROOT / "data" / dataset
        for split in SPLITS:
            for raw_video in LEGAL_SPLITS[dataset][split]:
                video = assert_legal_video(raw_video, dataset=dataset)
                expression_root = root / "expression" / video
                for path in sorted(expression_root.glob("*.json")):
                    item = json.loads(path.read_text(encoding="utf-8"))
                    label_map = item.get("label", {})
                    frame_rows = []
                    for raw_frame, raw_targets in label_map.items():
                        targets = raw_targets if isinstance(raw_targets, list) else [raw_targets]
                        targets = [str(target) for target in targets if target is not None]
                        if targets:
                            frame_rows.append((int(raw_frame), sorted(set(targets))))
                    frame_rows.sort()
                    positive = None if not frame_rows else frame_rows[len(frame_rows) // 2]
                    image_root = PROJECT_ROOT / "data/kitti_tracking/training/image_02" / video
                    image_frames = sorted(int(path.stem) for path in image_root.glob("*.png"))
                    labelled_frames = {frame for frame, _ in frame_rows}
                    empty_frames = [frame for frame in image_frames if frame not in labelled_frames]
                    query_rows.append({
                        "dataset": dataset,
                        "split": split,
                        "video": video,
                        "expression": path.stem,
                        "sentence": str(item.get("sentence", path.stem)),
                        "query_id": 0,
                        "positive": positive,
                        "empty_frame": empty_frames[0] if empty_frames else None,
                        "expression_path": path,
                    })
    query_rows.sort(key=lambda row: (row["dataset"], row["video"], row["expression"], row["sentence"]))
    rows: list[dict[str, Any]] = []
    for query_id, query in enumerate(query_rows):
        if query["positive"] is not None:
            frame, targets = query["positive"]
            rows.append({
                **query,
                "query_id": query_id,
                "frame": frame,
                "target_ids": targets,
                "image": assert_legal_path(
                    PROJECT_ROOT / "data/kitti_tracking/training/image_02"
                    / query["video"] / f"{frame:06d}.png"
                ),
                "row_kind": "positive",
            })
        if query["empty_frame"] is not None:
            frame = query["empty_frame"]
            rows.append({
                **query,
                "query_id": query_id,
                "frame": frame,
                "target_ids": [],
                "image": assert_legal_path(
                    PROJECT_ROOT / "data/kitti_tracking/training/image_02"
                    / query["video"] / f"{frame:06d}.png"
                ),
                "row_kind": "empty",
            })
    return rows


def read_gt(path: Path, width: int, height: int) -> dict[str, list[float]]:
    boxes: dict[str, list[float]] = {}
    if not path.is_file():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        track_id = str(fields[1])
        cx, cy, box_w, box_h = (float(value) for value in fields[2:6])
        boxes[track_id] = [
            (cx - box_w / 2.0) * width,
            (cy - box_h / 2.0) * height,
            (cx + box_w / 2.0) * width,
            (cy + box_h / 2.0) * height,
        ]
    return boxes


def iou(left: list[float], right: list[float]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - inter
    return inter / union if union > 0.0 else 0.0


def new_stats() -> dict[str, dict[str, int]]:
    return {
        str(k): {str(threshold): 0 for threshold in THRESHOLDS}
        for k in TOP_KS
    }


def update_stats(stats: dict[str, dict[str, int]], boxes: list[float], target: list[float]) -> None:
    for k in TOP_KS:
        best = max((iou(candidate, target) for candidate in boxes[:k]), default=0.0)
        for threshold in THRESHOLDS:
            stats[str(k)][str(threshold)] += int(best >= threshold)


def empty_metrics() -> dict[str, float]:
    return {
        "precision_hits": 0.0,
        "precision_predictions": 0.0,
        "ap50_sum": 0.0,
        "positive_queries": 0.0,
        "single_queries": 0.0,
        "single_hits": 0.0,
        "multi_queries": 0.0,
        "multi_target_total": 0.0,
        "multi_target_hits": 0.0,
        "multi_exact_hits": 0.0,
        "empty_queries": 0.0,
        "empty_fp": 0.0,
        "score_threshold_predictions": 0.0,
        "score_threshold_queries": 0.0,
    }


def average_precision(predictions: list[list[float]], scores: list[float], targets: list[list[float]]) -> float:
    """Query-level AP50 with one-to-one GT matching in score order."""
    if not targets:
        return 0.0
    matched: set[int] = set()
    tp: list[int] = []
    for box in predictions:
        best = max(((iou(box, target), index) for index, target in enumerate(targets) if index not in matched), default=(0.0, -1))
        if best[0] >= 0.50:
            matched.add(best[1])
            tp.append(1)
        else:
            tp.append(0)
    hit = 0
    previous_recall = 0.0
    ap = 0.0
    for index, value in enumerate(tp, start=1):
        hit += value
        recall = hit / len(targets)
        precision = hit / index
        ap += precision * max(0.0, recall - previous_recall)
        previous_recall = recall
    return ap


def run_worker(args: argparse.Namespace) -> int:
    all_rows = query_records()
    rows = all_rows[args.worker_index :: args.worker_count]
    runtime = load_grounding_runtime(
        config_path=DEFAULT_CONFIG,
        checkpoint=DEFAULT_CHECKPOINT,
        max_per_img=max(TOP_KS),
        device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu",
        training_contract=False,
    )
    stats = {stratum: new_stats() for stratum in STRATA}
    query_metrics = {stratum: empty_metrics() for stratum in STRATA}
    query_frame_count = 0
    target_reference_count = 0
    valid_target_count = 0
    missing_target_count = 0
    valid_by_dataset = {dataset: 0 for dataset in ("refer_kitti_v1", "refer_kitti_v2")}
    missing_by_dataset = {dataset: 0 for dataset in ("refer_kitti_v1", "refer_kitti_v2")}
    target_units: set[tuple[Any, ...]] = set()
    started = time.perf_counter()
    for index, row in enumerate(rows):
        result = run_grounding(runtime, row["image"], row["sentence"])
        prediction_boxes, prediction_scores = prediction_arrays(result)
        order = torch.argsort(prediction_scores, descending=True).tolist()
        boxes = prediction_boxes[order].tolist()
        scores = prediction_scores[order].tolist()
        if not bool(torch.isfinite(prediction_boxes).all() and torch.isfinite(prediction_scores).all()):
            raise RuntimeError(f"non-finite expression prediction: {row['image']}")
        # The query-conditioned forward is complete before opening the label.
        from PIL import Image
        with Image.open(row["image"]) as image:
            width, height = image.size
        label_path = (
            PROJECT_ROOT / "data" / row["dataset"] / "labels_with_ids" / "image_02"
            / row["video"] / f"{row['frame']:06d}.txt"
        )
        ground_truth = read_gt(label_path, width, height)
        query_frame_count += 1
        for stratum in ("overall", row["dataset"]):
            qm = query_metrics[stratum]
            qm["score_threshold_queries"] += 1
            qm["score_threshold_predictions"] += sum(score >= SCORE_THRESHOLD for score in scores)
            if row["row_kind"] == "empty":
                qm["empty_queries"] += 1
                qm["empty_fp"] += sum(score >= SCORE_THRESHOLD for score in scores)
        for target_id in row["target_ids"]:
            unit = (row["dataset"], row["video"], row["query_id"], row["frame"], target_id)
            if unit in target_units:
                continue
            target_units.add(unit)
            target_reference_count += 1
            target = ground_truth.get(str(target_id))
            if target is None:
                missing_target_count += 1
                missing_by_dataset[row["dataset"]] += 1
                continue
            valid_target_count += 1
            valid_by_dataset[row["dataset"]] += 1
            for stratum in ("overall", row["dataset"]):
                update_stats(stats[stratum], boxes, target)
        if row["row_kind"] == "positive":
            targets = [ground_truth[target_id] for target_id in row["target_ids"] if target_id in ground_truth]
            if targets:
                for stratum in ("overall", row["dataset"]):
                    qm = query_metrics[stratum]
                    qm["positive_queries"] += 1
                    if len(targets) == 1:
                        qm["single_queries"] += 1
                        qm["single_hits"] += int(max((iou(box, targets[0]) for box in boxes[:PRECISION_TOP_K]), default=0.0) >= 0.50)
                    else:
                        qm["multi_queries"] += 1
                        hits = sum(
                            int(max((iou(box, target) for box in boxes[:PRECISION_TOP_K]), default=0.0) >= 0.50)
                            for target in targets
                        )
                        qm["multi_target_total"] += len(targets)
                        qm["multi_target_hits"] += hits
                        qm["multi_exact_hits"] += int(hits == len(targets))
                    qm["ap50_sum"] += average_precision(boxes[:TOP_KS[-1]], scores[:TOP_KS[-1]], targets)
                    selected = boxes[:PRECISION_TOP_K]
                    matched = 0
                    used: set[int] = set()
                    for box in selected:
                        best = max(((iou(box, target), index) for index, target in enumerate(targets) if index not in used), default=(0.0, -1))
                        if best[0] >= 0.50:
                            used.add(best[1])
                            matched += 1
                    qm["precision_hits"] += matched
                    qm["precision_predictions"] += len(selected)
        if args.progress and (index + 1) % args.progress == 0:
            print(
                f"worker={args.worker_index} {index + 1}/{len(rows)} "
                f"queries targets={target_reference_count}",
                flush=True,
            )
    payload = {
        "format": "locatemot-u-u1-formal-expression-shard-v1",
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "gpu": args.gpu,
        "top_k": list(TOP_KS),
        "iou_thresholds": list(THRESHOLDS),
        "query_frame_count": query_frame_count,
        "target_reference_count": target_reference_count,
        "valid_target_count": valid_target_count,
        "missing_source_target_count": missing_target_count,
        "valid_target_count_by_dataset": valid_by_dataset,
        "missing_source_target_count_by_dataset": missing_by_dataset,
        "stats": stats,
        "query_metrics": query_metrics,
        "runtime": runtime_metadata(runtime),
        "elapsed_seconds": time.perf_counter() - started,
        "official_test_labels_read": False,
        "screening_gt_used": False,
        "training_launched": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"worker": args.worker_index, "queries": query_frame_count, "targets": target_reference_count}, sort_keys=True))
    return 0


def merge_stats(target: dict[str, dict[str, int]], source: dict[str, Any]) -> None:
    for k in TOP_KS:
        for threshold in THRESHOLDS:
            target[str(k)][str(threshold)] += int(source[str(k)][str(threshold)])


def merge_query_metrics(target: dict[str, float], source: dict[str, Any]) -> None:
    for key in target:
        target[key] += float(source[key])


def aggregate(args: argparse.Namespace) -> int:
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.shards)]
    if not shards:
        raise RuntimeError("no expression shards")
    merged = {stratum: new_stats() for stratum in STRATA}
    merged_query_metrics = {stratum: empty_metrics() for stratum in STRATA}
    query_frame_count = target_reference_count = valid_target_count = missing_target_count = 0
    valid_by_dataset = {dataset: 0 for dataset in ("refer_kitti_v1", "refer_kitti_v2")}
    missing_by_dataset = {dataset: 0 for dataset in ("refer_kitti_v1", "refer_kitti_v2")}
    for shard in shards:
        for stratum in STRATA:
            merge_stats(merged[stratum], shard["stats"][stratum])
            merge_query_metrics(merged_query_metrics[stratum], shard["query_metrics"][stratum])
        query_frame_count += int(shard["query_frame_count"])
        target_reference_count += int(shard["target_reference_count"])
        valid_target_count += int(shard["valid_target_count"])
        missing_target_count += int(shard["missing_source_target_count"])
        for dataset in valid_by_dataset:
            valid_by_dataset[dataset] += int(shard["valid_target_count_by_dataset"][dataset])
            missing_by_dataset[dataset] += int(shard["missing_source_target_count_by_dataset"][dataset])
    metrics: dict[str, Any] = {}
    for stratum in STRATA:
        metrics[stratum] = {
            str(k): {
                str(threshold): {
                    "hit": merged[stratum][str(k)][str(threshold)],
                    "total": valid_target_count if stratum == "overall" else None,
                }
                for threshold in THRESHOLDS
            }
            for k in TOP_KS
        }
    # Dataset denominators use the corresponding physical target units rather
    # than the combined V1/V2 count.
    denominator_by_stratum = {stratum: {str(k): 0 for k in TOP_KS} for stratum in STRATA}
    for k in TOP_KS:
        denominator_by_stratum["overall"][str(k)] = valid_target_count
        denominator_by_stratum["refer_kitti_v1"][str(k)] = valid_by_dataset["refer_kitti_v1"]
        denominator_by_stratum["refer_kitti_v2"][str(k)] = valid_by_dataset["refer_kitti_v2"]
    for stratum in STRATA:
        for k in TOP_KS:
            for threshold in THRESHOLDS:
                cell = metrics[stratum][str(k)][str(threshold)]
                denom = denominator_by_stratum[stratum][str(k)]
                cell["total"] = denom
                cell["recall"] = cell["hit"] / denom if denom else 0.0
    main = metrics["refer_kitti_v1"], metrics["refer_kitti_v2"]
    gate_pass = all(
        metrics[dataset]["1"]["0.5"]["recall"] >= 0.75
        for dataset in ("refer_kitti_v1", "refer_kitti_v2")
    )
    payload = {
        "format": "locatemot-u-u1-formal-expression-gate-v1",
        "status": "FORMAL_COMPLETE_REPRESENTATIVE_FRAME_PER_QUERY",
        "gate_pass": bool(gate_pass),
        "gate_goal": "Top1 expression grounding recall at IoU 0.50 >= 0.75 for both V1 and V2",
        "protocol": {
            "unit": "dataset/video/query_id/representative_frame/target_id",
            "query_frame_sampling": "one middle non-empty labeled frame per legal calibration/validation query",
            "split_parts": list(SPLITS),
            "top_k_diagnostic": [1, 5, 10, 100, 300],
            "iou_thresholds": list(THRESHOLDS),
            "missing_source_target_policy": "count in source denominator, exclude only from valid-positive metric denominator",
            "precision_top_k": PRECISION_TOP_K,
            "score_threshold": SCORE_THRESHOLD,
        },
        "query_frame_count": query_frame_count,
        "target_reference_count": target_reference_count,
        "valid_target_count": valid_target_count,
        "missing_source_target_count": missing_target_count,
        "valid_target_count_by_dataset": valid_by_dataset,
        "missing_source_target_count_by_dataset": missing_by_dataset,
        "metrics": metrics,
        "query_metrics": {
            stratum: {
                "precision_at_top10_iou050": values["precision_hits"] / values["precision_predictions"] if values["precision_predictions"] else 0.0,
                "ap50_mean": values["ap50_sum"] / values["positive_queries"] if values["positive_queries"] else 0.0,
                "empty_frame_fp_per_query": values["empty_fp"] / values["empty_queries"] if values["empty_queries"] else 0.0,
                "predictions_ge_score005_per_query": values["score_threshold_predictions"] / values["score_threshold_queries"] if values["score_threshold_queries"] else 0.0,
                "single_target_recall_top10_iou050": values["single_hits"] / values["single_queries"] if values["single_queries"] else 0.0,
                "multi_target_recall_top10_iou050": values["multi_target_hits"] / values["multi_target_total"] if values["multi_target_total"] else 0.0,
                "multi_target_exact_set_coverage_top10_iou050": values["multi_exact_hits"] / values["multi_queries"] if values["multi_queries"] else 0.0,
                "positive_queries": int(values["positive_queries"]),
                "empty_queries": int(values["empty_queries"]),
            }
            for stratum, values in merged_query_metrics.items()
        },
        "main_top1_by_dataset": {dataset: metrics[dataset]["1"] for dataset in ("refer_kitti_v1", "refer_kitti_v2")},
        "runtime": shards[0]["runtime"],
        "official_test_labels_read": False,
        "screening_gt_used": False,
        "training_launched": False,
        "shards": [str(path) for path in args.shards],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "gate_pass": payload["gate_pass"], "query_frames": query_frame_count, "valid_targets": valid_target_count, "missing_targets": missing_target_count, "main_top1_by_dataset": payload["main_top1_by_dataset"]}, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--progress", type=int, default=100)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--shards", nargs="*", type=Path, default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.aggregate:
        return aggregate(args)
    if args.worker_index is None or args.worker_index < 0 or args.worker_index >= args.worker_count:
        raise ValueError("worker-index must be in [0, worker-count)")
    return run_worker(args)


if __name__ == "__main__":
    raise SystemExit(main())
