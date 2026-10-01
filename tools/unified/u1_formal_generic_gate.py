#!/usr/bin/env python3
"""Formal query-independent generic proposal gate for LocateMOT-U.

The model sees one fixed vocabulary for every image.  Labels are opened only
after the official MMDetection inference call has completed.  Workers emit
compact aggregates rather than raw predictions; the aggregate is keyed by
physical dataset/video/frame/track units and reports the frozen split and size
strata explicitly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image

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

VOCABULARY = (
    "car . van . truck . bus . tram . pedestrian . person . cyclist . "
    "bicycle . motorcycle ."
)
THRESHOLDS = (0.25, 0.50, 0.75)
TOP_KS = (100, 150, 300, 900)
SPLITS = ("calibration", "validation")
STRATA = ("overall", "refer_kitti_v1", "refer_kitti_v2", "small", "medium", "large")


def frame_records() -> list[dict[str, Any]]:
    """Return one record per unique legal image in calibration/validation."""
    records: dict[tuple[str, int], dict[str, Any]] = {}
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        for split in SPLITS:
            for raw_video in LEGAL_SPLITS[dataset][split]:
                video = assert_legal_video(raw_video, dataset=dataset)
                root = PROJECT_ROOT / "data/kitti_tracking/training/image_02" / video
                for path in sorted(root.glob("*.png")):
                    key = (video, int(path.stem))
                    row = records.setdefault(key, {
                        "video": video,
                        "frame": int(path.stem),
                        "image": assert_legal_path(path),
                        "datasets": [],
                    })
                    row["datasets"].append({"dataset": dataset, "split": split})
    return [records[key] for key in sorted(records)]


def read_gt(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        track_id = str(fields[1])
        cx, cy, box_w, box_h = (float(value) for value in fields[2:6])
        x0 = (cx - box_w / 2.0) * width
        y0 = (cy - box_h / 2.0) * height
        x1 = (cx + box_w / 2.0) * width
        y1 = (cy + box_h / 2.0) * height
        boxes.append({
            "track_id": track_id,
            "box": [x0, y0, x1, y1],
            "height": max(0.0, y1 - y0),
        })
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


def new_stats() -> dict[str, dict[str, dict[str, int]]]:
    return {
        str(k): {
            str(t): {"total": 0, "hit": 0}
            for t in THRESHOLDS
        }
        for k in TOP_KS
    }


def update_stats(
    stats: dict[str, dict[str, dict[str, int]]],
    boxes: list[float],
    target: list[float],
) -> None:
    for k in TOP_KS:
        values = [iou(candidate, target) for candidate in boxes[:k]]
        best = max(values, default=0.0)
        for threshold in THRESHOLDS:
            cell = stats[str(k)][str(threshold)]
            cell["total"] += 1
            cell["hit"] += int(best >= threshold)


def run_worker(args: argparse.Namespace) -> int:
    all_records = frame_records()
    records = all_records[args.worker_index :: args.worker_count]
    runtime = load_grounding_runtime(
        config_path=DEFAULT_CONFIG,
        checkpoint=DEFAULT_CHECKPOINT,
        max_per_img=max(TOP_KS),
        device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu",
        training_contract=False,
    )
    stats = {stratum: new_stats() for stratum in STRATA}
    unique_objects: set[tuple[str, str, str]] = set()
    image_count = 0
    target_count = 0
    started = time.perf_counter()
    for index, record in enumerate(records):
        result = run_grounding(runtime, record["image"], VOCABULARY)
        prediction_boxes, prediction_scores = prediction_arrays(result)
        order = torch.argsort(prediction_scores, descending=True).tolist()
        boxes = prediction_boxes[order].tolist()
        if not all(torch.isfinite(prediction_boxes).flatten().tolist()) or not all(
            torch.isfinite(prediction_scores).flatten().tolist()
        ):
            raise RuntimeError(f"non-finite generic prediction: {record['image']}")
        # The official inference call is complete before any label path is
        # opened.  The same prediction serves both dataset label versions when
        # a video/frame is shared by V1 and V2.
        with Image.open(record["image"]) as image:
            width, height = image.size
        for scope in record["datasets"]:
            dataset = scope["dataset"]
            split = scope["split"]
            label_path = (
                PROJECT_ROOT / "data" / dataset / "labels_with_ids" / "image_02"
                / record["video"] / f"{record['frame']:06d}.txt"
            )
            # The image pool contains a few frames without a materialized
            # tracking label.  They contribute no physical GT unit and are
            # retained in the frame universe only for deterministic traversal.
            if not label_path.is_file():
                continue
            labels = read_gt(label_path, width, height)
            for label in labels:
                unique_objects.add((dataset, record["video"], label["track_id"]))
                target_count += 1
                height_px = label["height"]
                size = "small" if height_px < 32 else "medium" if height_px <= 96 else "large"
                for stratum in ("overall", dataset, size):
                    update_stats(stats[stratum], boxes, label["box"])
        image_count += 1
        if args.progress and (index + 1) % args.progress == 0:
            print(
                f"worker={args.worker_index} {index + 1}/{len(records)} "
                f"images targets={target_count}",
                flush=True,
            )
    payload = {
        "format": "locatemot-u-u1-formal-generic-shard-v1",
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "gpu": args.gpu,
        "vocabulary": VOCABULARY,
        "top_k": list(TOP_KS),
        "iou_thresholds": list(THRESHOLDS),
        "image_count": image_count,
        "target_count": target_count,
        "unique_physical_objects": len(unique_objects),
        "stats": stats,
        "runtime": runtime_metadata(runtime),
        "elapsed_seconds": time.perf_counter() - started,
        "official_test_labels_read": False,
        "screening_gt_used": False,
        "training_launched": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"worker": args.worker_index, "images": image_count, "targets": target_count}, sort_keys=True))
    return 0


def merge_stats(target: dict[str, dict[str, dict[str, int]]], source: dict[str, Any]) -> None:
    for stratum in STRATA:
        for k in TOP_KS:
            for threshold in THRESHOLDS:
                for key in ("total", "hit"):
                    target[stratum][str(k)][str(threshold)][key] += int(
                        source[stratum][str(k)][str(threshold)][key]
                    )


def rate(cell: dict[str, int]) -> float:
    return cell["hit"] / cell["total"] if cell["total"] else 0.0


def aggregate(args: argparse.Namespace) -> int:
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.shards)]
    if not shards:
        raise RuntimeError("no generic shards")
    merged = {stratum: new_stats() for stratum in STRATA}
    image_count = target_count = unique_objects = 0
    for shard in shards:
        merge_stats(merged, shard["stats"])
        image_count += int(shard["image_count"])
        target_count += int(shard["target_count"])
        unique_objects += int(shard["unique_physical_objects"])
    metrics: dict[str, Any] = {}
    for stratum in STRATA:
        metrics[stratum] = {}
        for k in TOP_KS:
            metrics[stratum][str(k)] = {
                str(threshold): {
                    "hit": merged[stratum][str(k)][str(threshold)]["hit"],
                    "total": merged[stratum][str(k)][str(threshold)]["total"],
                    "recall": rate(merged[stratum][str(k)][str(threshold)]),
                }
                for threshold in THRESHOLDS
            }
    main = metrics["overall"]["150"]
    goal = all(
        metrics[dataset]["150"][str(threshold)]["recall"] >= 0.90
        for dataset in ("refer_kitti_v1", "refer_kitti_v2")
        for threshold in THRESHOLDS
    )
    payload = {
        "format": "locatemot-u-u1-formal-generic-gate-v1",
        "status": "FORMAL_COMPLETE",
        "gate_pass": bool(goal and unique_objects >= 500),
        "gate_goal": "Top150 recall >= 0.90 at IoU .25/.50/.75 for both V1 and V2",
        "fixed_vocabulary": VOCABULARY,
        "protocol": {
            "query_independent": True,
            "physical_gt_unit": "dataset/video/frame/track_id",
            "deduplication": "one label row per dataset/video/frame/track_id",
            "split_parts": list(SPLITS),
            "top_k_diagnostic": [100, 300, 900],
            "top_k_main": 150,
            "iou_thresholds": list(THRESHOLDS),
            "size_bins_px": {"small": "height<32", "medium": "32<=height<=96", "large": "height>96"},
        },
        "image_count": image_count,
        "target_count": target_count,
        "unique_physical_objects": unique_objects,
        "metrics": metrics,
        "main_top150": main,
        "runtime": shards[0]["runtime"],
        "official_test_labels_read": False,
        "screening_gt_used": False,
        "training_launched": False,
        "shards": [str(path) for path in args.shards],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "gate_pass": payload["gate_pass"], "unique_physical_objects": unique_objects, "main_top150": main}, indent=2, sort_keys=True))
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
