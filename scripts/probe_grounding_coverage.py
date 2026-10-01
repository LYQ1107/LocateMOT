#!/usr/bin/env python3
"""Measure frozen GroundingDINO candidate coverage on a legal public slice.

This is a candidate-input probe for the post-migration reconstruction.  It
does not train, alter the tracker, or use official-evaluation labels.  Each
prediction is completed before the corresponding public label file is opened.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from locatemot.paths import IMAGE_ROOT, RESTORE_ROOT, V1_ROOT, V2_ROOT
from locatemot.rmot.l49_data import L49_SPLITS, load_l49_queries
from locatemot.rmot.l82_grounding_runtime import GroundingCandidateReferenceRuntime


OFFICIAL_EVAL = {"0005", "0011", "0013", "0019"}


def iou_xyxy(left: list[float], right: list[float]) -> float:
    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_left = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    area_right = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = area_left + area_right - inter
    return inter / union if union > 0.0 else 0.0


def gt_boxes(label_path: Path, width: int, height: int) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        track_id = str(fields[1])
        cx, cy, box_w, box_h = (float(value) for value in fields[2:6])
        result[track_id] = [
            (cx - box_w / 2.0) * width,
            (cy - box_h / 2.0) * height,
            (cx + box_w / 2.0) * width,
            (cy + box_h / 2.0) * height,
        ]
    return result


def choose_rows(per_split: dict[str, int]) -> tuple[list[dict[str, Any]], int]:
    selected: list[dict[str, Any]] = []
    skipped_anomaly = 0
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        root = V1_ROOT if dataset == "refer_kitti_v1" else V2_ROOT
        rows = load_l49_queries(dataset)
        for split in ("calibration", "validation"):
            wanted = int(per_split.get(f"{dataset}:{split}", 0))
            count = 0
            for row in rows:
                if row["split"] != split or str(row["video"]) in OFFICIAL_EVAL:
                    continue
                valid_frames: list[tuple[int, list[str]]] = []
                for frame, targets in sorted(row["target"].items()):
                    if not targets:
                        continue
                    label = root / "labels_with_ids" / "image_02" / str(row["video"]) / f"{int(frame):06d}.txt"
                    if not label.is_file():
                        continue
                    available = {line.split()[1] for line in label.read_text(encoding="utf-8").splitlines() if len(line.split()) >= 2}
                    if any(str(target) not in available for target in targets):
                        skipped_anomaly += 1
                        continue
                    valid_frames.append((int(frame), sorted(str(target) for target in targets)))
                if valid_frames:
                    # Use the temporal midpoint of the query's valid span so
                    # the probe is not dominated by frame zero for expressions
                    # that persist through an entire sequence.
                    frame, targets = valid_frames[len(valid_frames) // 2]
                    item = dict(row)
                    item["probe_frame"] = frame
                    item["probe_targets"] = targets
                    selected.append(item)
                    count += 1
                if count >= wanted:
                    break
    return selected, skipped_anomaly


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESTORE_ROOT / "grounding_coverage_probe.json")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    per_split = {
        "refer_kitti_v1:calibration": 8,
        "refer_kitti_v1:validation": 8,
        "refer_kitti_v2:calibration": 8,
        "refer_kitti_v2:validation": 16,
    }
    rows, skipped_anomaly = choose_rows(per_split)
    if not rows:
        raise RuntimeError("no legal probe rows")
    runtime = GroundingCandidateReferenceRuntime(torch.device(args.device))
    measurements: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        image = IMAGE_ROOT / str(row["video"]) / f"{int(row['probe_frame']):06d}.png"
        with torch.inference_mode():
            result = runtime.inference_detector(runtime.model, str(image), text_prompt=str(row["sentence"]), custom_entities=True)
        # Labels are opened only after the model prediction is complete.
        root = V1_ROOT if row["dataset"] == "refer_kitti_v1" else V2_ROOT
        label = root / "labels_with_ids" / "image_02" / str(row["video"]) / f"{int(row['probe_frame']):06d}.txt"
        width, height = Image.open(image).size
        targets = gt_boxes(label, width, height)
        prediction = result.pred_instances
        boxes = prediction.bboxes.detach().float().cpu().tolist()
        scores = prediction.scores.detach().float().cpu().tolist()
        best_by_target: dict[str, dict[str, float]] = {}
        for target_id in row["probe_targets"]:
            box = targets.get(str(target_id))
            if box is None:
                continue
            best_iou = 0.0
            best_score = 0.0
            for candidate, score in zip(boxes, scores):
                value = iou_xyxy(candidate, box)
                if value > best_iou:
                    best_iou = value
                    best_score = float(score)
            best_by_target[str(target_id)] = {"best_iou": best_iou, "best_score": best_score}
        measurements.append({
            "dataset": row["dataset"], "split": row["split"], "video": row["video"],
            "query_id": int(row["query_id"]), "frame_id": int(row["probe_frame"]),
            "sentence": row["sentence"], "target_count": len(best_by_target),
            "candidate_count": len(boxes), "score_min": min(scores) if scores else None,
            "score_max": max(scores) if scores else None, "target_metrics": best_by_target,
            "finite_boxes": all(torch.isfinite(torch.tensor(value)).all().item() for value in boxes),
            "finite_scores": all(torch.isfinite(torch.tensor(value)).item() for value in scores),
        })
        print(f"[{index + 1}/{len(rows)}] {row['dataset']} {row['video']} frame={row['probe_frame']}", flush=True)
    target_values = [metric for row in measurements for metric in row["target_metrics"].values()]
    result = {
        "format": "locatemot-grounding-coverage-probe-v1",
        "status": "complete",
        "device": str(args.device),
        "selection": per_split,
        "rows": measurements,
        "aggregate": {
            "rows": len(measurements),
            "target_boxes": len(target_values),
            "iou_at_025": sum(value["best_iou"] >= 0.25 for value in target_values) / max(1, len(target_values)),
            "iou_at_050": sum(value["best_iou"] >= 0.50 for value in target_values) / max(1, len(target_values)),
            "mean_best_iou": sum(value["best_iou"] for value in target_values) / max(1, len(target_values)),
            "mean_best_score": sum(value["best_score"] for value in target_values) / max(1, len(target_values)),
            "median_best_score": sorted(value["best_score"] for value in target_values)[len(target_values) // 2] if target_values else None,
            "candidate_count_distribution": dict(Counter(str(row["candidate_count"]) for row in measurements)),
            "skipped_anomaly_frame_references": skipped_anomaly,
            "elapsed_seconds": time.perf_counter() - started,
        },
        "prediction_before_label": True,
        "official_test_labels_read": False,
        "screening_gt_used": False,
        "ordinary_mot_ovmot_touched": False,
        "candidate_deletion": False,
        "candidate_truncation": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
