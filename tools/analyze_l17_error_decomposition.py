"""Decompose L17 RMOT errors from completed TrackEval prediction trees.

This is a diagnostic, not a replacement for TrackEval.  It uses IoU>=0.5
one-to-one matching only to localize inactive-frame false positives,
candidate-level wrong-ID matches, duplicate selected IDs, and fragmentation.
The official HOTA/DetA/AssA/DetRe/DetPr/IDF1 values are read from the
TrackEval detailed CSV produced beside the prediction tree.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l16_track_selector import FAMILY_NAMES, expression_family_vector


def box_iou(left, right) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(1e-9, area_left + area_right - inter)


def read_rows(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        value = [float(part) for part in line.split(",")]
        rows.append({
            "frame": int(value[0]), "track_id": int(value[1]),
            "box": (value[2], value[3], value[2] + value[4], value[3] + value[5]),
        })
    return rows


def match_frame(predictions: list[dict], ground_truth: list[dict]) -> list[tuple[int, int, float]]:
    pairs = sorted(
        (box_iou(prediction["box"], truth["box"]), pi, gi)
        for pi, prediction in enumerate(predictions)
        for gi, truth in enumerate(ground_truth)
    )
    used_predictions, used_truth = set(), set()
    matches = []
    for overlap, pi, gi in reversed(pairs):
        if overlap < 0.5 or pi in used_predictions or gi in used_truth:
            continue
        used_predictions.add(pi)
        used_truth.add(gi)
        matches.append((pi, gi, float(overlap)))
    return matches


def load_bank_labels(bank_path: Path) -> dict[tuple[int, int], set[str]] | None:
    labels_path = bank_path.with_suffix(".labels.json")
    if not labels_path.exists():
        # Official-eval banks intentionally omit supervision labels.
        return None
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    labels = json.loads(labels_path.read_text())["candidate_gt"]
    tensors = bank["tensors"]
    lookup = defaultdict(set)
    frame_ids = tensors["frame_ids"].tolist()
    frame_ptr = tensors["frame_ptr"].tolist()
    track_ids = tensors["track_id"].tolist()
    for frame_index, frame_id in enumerate(frame_ids):
        start, end = int(frame_ptr[frame_index]), int(frame_ptr[frame_index + 1])
        for index in range(start, end):
            label = labels[index]
            if label is not None:
                lookup[(int(frame_id), int(track_ids[index]))].add(str(label))
    return lookup


def query_diagnostic(prediction_path: Path, ground_truth_path: Path,
                     bank_labels: dict[tuple[int, int], set[str]] | None) -> dict:
    predictions = read_rows(prediction_path)
    ground_truth = read_rows(ground_truth_path)
    pred_by_frame = defaultdict(list)
    gt_by_frame = defaultdict(list)
    for row in predictions:
        pred_by_frame[row["frame"]].append(row)
    for row in ground_truth:
        gt_by_frame[row["frame"]].append(row)

    inactive = 0
    matched_predictions = 0
    matched_ground_truth = 0
    wrong_id = 0 if bank_labels is not None else None
    duplicate_events = 0
    duplicate_extra_rows = 0
    matched_by_gt = defaultdict(list)
    for frame, frame_predictions in pred_by_frame.items():
        frame_ground_truth = gt_by_frame.get(frame, [])
        if not frame_ground_truth:
            inactive += len(frame_predictions)
        matches = match_frame(frame_predictions, frame_ground_truth)
        matched_predictions += len(matches)
        matched_ground_truth += len(matches)
        for pi, gi, _overlap in matches:
            prediction = frame_predictions[pi]
            truth = frame_ground_truth[gi]
            truth_id = str(truth["track_id"])
            matched_by_gt[truth_id].append((frame, prediction["track_id"]))
            if bank_labels is not None and truth_id not in bank_labels.get(
                    (frame, prediction["track_id"]), set()):
                wrong_id += 1
        per_truth = defaultdict(set)
        for pi, gi, _overlap in matches:
            per_truth[str(frame_ground_truth[gi]["track_id"])].add(
                frame_predictions[pi]["track_id"])
        for track_ids in per_truth.values():
            if len(track_ids) > 1:
                duplicate_events += 1
                duplicate_extra_rows += len(track_ids) - 1

    fragments = 0
    fragmented_ground_truth = 0
    switch_events = 0
    for values in matched_by_gt.values():
        ordered = [track_id for _frame, track_id in sorted(values)]
        distinct = len(set(ordered))
        fragments += max(0, distinct - 1)
        fragmented_ground_truth += int(distinct > 1)
        switch_events += sum(left != right for left, right in zip(ordered, ordered[1:]))

    return {
        "prediction_rows": len(predictions),
        "ground_truth_rows": len(ground_truth),
        "prediction_gt_row_ratio": len(predictions) / max(1, len(ground_truth)),
        "inactive_frame_prediction_rows": inactive,
        "inactive_frame_fp_fraction": inactive / max(1, len(predictions)),
        "iou50_matched_prediction_rows": matched_predictions,
        "iou50_matched_ground_truth_rows": matched_ground_truth,
        "iou50_candidate_wrong_id_rows": wrong_id,
        "candidate_wrong_id_available": bank_labels is not None,
        "duplicate_id_frame_events": duplicate_events,
        "duplicate_id_extra_rows": duplicate_extra_rows,
        "fragmented_ground_truth_ids": fragmented_ground_truth,
        "fragment_events": fragments,
        "sequential_id_switch_events": switch_events,
    }


def trackeval_rows(eval_root: Path) -> dict[str, dict]:
    csv_path = eval_root / "uidm17" / "pedestrian_detailed.csv"
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    metrics = ("HOTA___AUC", "DetA___AUC", "AssA___AUC",
               "DetRe___AUC", "DetPr___AUC", "IDF1")
    return {
        row["seq"]: {metric: float(row[metric]) * 100.0 for metric in metrics}
        for row in rows
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", required=True)
    parser.add_argument("--dataset", choices=["kitti_v1", "kitti_v2", "dance"], required=True)
    parser.add_argument("--bank-root", default="outputs/l16/track_banks_dedup")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    eval_root = Path(args.eval_root)
    bank_dataset = "dance_eval" if args.dataset == "dance" else "kitti"
    trackeval = trackeval_rows(eval_root)
    query_stats = {}
    for prediction_path in sorted((eval_root / "uidm17").glob("*/*/predict.txt")):
        video = prediction_path.parts[-3]
        expression = prediction_path.parts[-2]
        query = f"{video}+{expression}"
        bank_path = ROOT / args.bank_root / bank_dataset / f"{video}.pt"
        query_stats[query] = query_diagnostic(
            prediction_path,
            prediction_path.with_name("gt.txt"),
            load_bank_labels(bank_path),
        )

    aggregate = {}
    for key in (
            "prediction_rows", "ground_truth_rows",
            "inactive_frame_prediction_rows", "iou50_matched_prediction_rows",
            "iou50_matched_ground_truth_rows",
            "duplicate_id_frame_events", "duplicate_id_extra_rows",
            "fragmented_ground_truth_ids", "fragment_events",
            "sequential_id_switch_events"):
        aggregate[key] = int(sum(row[key] for row in query_stats.values()))
    aggregate["prediction_gt_row_ratio"] = (
        aggregate["prediction_rows"] / max(1, aggregate["ground_truth_rows"]))
    aggregate["inactive_frame_fp_fraction"] = (
        aggregate["inactive_frame_prediction_rows"] /
        max(1, aggregate["prediction_rows"]))
    aggregate["candidate_wrong_id_available"] = all(
        row["candidate_wrong_id_available"] for row in query_stats.values())
    if aggregate["candidate_wrong_id_available"]:
        aggregate["iou50_candidate_wrong_id_rows"] = int(sum(
            row["iou50_candidate_wrong_id_rows"] for row in query_stats.values()))

    family_summary = {}
    for family_index, family in enumerate(FAMILY_NAMES[1:], start=1):
        selected = []
        for query, stats in query_stats.items():
            expression = query.split("+", 1)[1]
            flags = expression_family_vector(expression).tolist()
            if flags[family_index] > 0.5:
                selected.append((query, stats))
        if not selected:
            continue
        family_metrics = [trackeval.get(query, {}) for query, _stats in selected]
        family_summary[family] = {
            "queries": len(selected),
            "mean_hota": float(np.mean([row["HOTA___AUC"] for row in family_metrics])),
            "mean_deta": float(np.mean([row["DetA___AUC"] for row in family_metrics])),
            "mean_assa": float(np.mean([row["AssA___AUC"] for row in family_metrics])),
            "mean_detpr": float(np.mean([row["DetPr___AUC"] for row in family_metrics])),
            "mean_det_re": float(np.mean([row["DetRe___AUC"] for row in family_metrics])),
            "mean_inactive_fp_fraction": float(np.mean([
                stats["inactive_frame_fp_fraction"] for _query, stats in selected])),
            "mean_prediction_gt_row_ratio": float(np.mean([
                stats["prediction_gt_row_ratio"] for _query, stats in selected])),
        }

    payload = {
        "dataset": args.dataset,
        "eval_root": str(eval_root),
        "matching_rule": "greedy one-to-one IoU >= 0.5; diagnostic only",
        "trackeval_combined_percent": trackeval.get("COMBINED", {}),
        "aggregate_diagnostic": aggregate,
        "query_diagnostic": query_stats,
        "overlapping_expression_family_summary": family_summary,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "dataset": args.dataset,
        "trackeval": payload["trackeval_combined_percent"],
        "aggregate": aggregate,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
