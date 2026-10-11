#!/usr/bin/env python3
"""Paired language counterfactual audit for the frozen U1 expression model.

The audit uses exactly 500 legal validation query/frame pairs.  For each pair,
the same image is evaluated with its true sentence and with a sentence from a
different target set in the same video.  Predictions are completed before the
source frame label is opened, preserving the measurement ordering used by the
formal expression gate.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locatemot.unified.data.legal_scope import assert_legal_path, assert_legal_video  # noqa: E402
from locatemot.unified.runtime.grounding_inference import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    load_grounding_runtime,
    prediction_arrays,
    run_grounding,
    runtime_metadata,
)
from tools.unified.u1_formal_expression_gate import (  # noqa: E402
    average_precision,
    iou,
    query_records,
    read_gt,
)

PAIR_COUNT = 500
TOP_K = 10
MAX_PREDICTIONS = 300


def make_pairs() -> list[dict[str, Any]]:
    """Make a deterministic fixed-size validation pair list."""
    rows = [
        row
        for row in query_records()
        if row["split"] == "validation"
        and row["row_kind"] == "positive"
        and row["target_ids"]
    ]
    rows.sort(key=lambda row: (row["dataset"], row["video"], row["expression"], row["frame"]))
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        assert_legal_video(row["video"], dataset=row["dataset"])
        groups[(row["dataset"], row["video"])].append(row)

    pairs: list[dict[str, Any]] = []
    for row in rows:
        target_set = tuple(sorted(str(target) for target in row["target_ids"]))
        donors = [
            donor
            for donor in groups[(row["dataset"], row["video"])]
            if tuple(sorted(str(target) for target in donor["target_ids"])) != target_set
            and donor["sentence"] != row["sentence"]
        ]
        if not donors:
            continue
        donor = donors[0]
        pairs.append(
            {
                "pair_index": len(pairs),
                "dataset": row["dataset"],
                "video": row["video"],
                "frame": int(row["frame"]),
                "image": str(assert_legal_path(Path(row["image"]))),
                "true_sentence": row["sentence"],
                "shuffled_sentence": donor["sentence"],
                "true_expression": row["expression"],
                "shuffled_expression": donor["expression"],
                "target_ids": target_set,
                "shuffled_target_ids": tuple(sorted(str(target) for target in donor["target_ids"])),
            }
        )
        if len(pairs) == PAIR_COUNT:
            break
    if len(pairs) != PAIR_COUNT:
        raise RuntimeError(f"only {len(pairs)} eligible legal validation pairs; need {PAIR_COUNT}")
    return pairs


def sorted_predictions(runtime: Any, image: str, sentence: str) -> tuple[list[list[float]], list[float]]:
    result = run_grounding(runtime, image, sentence)
    boxes, scores = prediction_arrays(result)
    if not bool(torch.isfinite(boxes).all() and torch.isfinite(scores).all()):
        raise RuntimeError(f"non-finite expression prediction: {image}")
    order = torch.argsort(scores, descending=True)
    return boxes[order].tolist(), scores[order].tolist()


def pair_metrics(pair: dict[str, Any], true_boxes: list[list[float]], shuffled_boxes: list[list[float]]) -> dict[str, Any]:
    """Open GT only after both query-conditioned forwards have completed."""
    from PIL import Image

    image_path = Path(pair["image"])
    with Image.open(image_path) as image:
        width, height = image.size
    label_path = (
        PROJECT_ROOT / "data" / pair["dataset"] / "labels_with_ids" / "image_02"
        / pair["video"] / f"{int(pair['frame']):06d}.txt"
    )
    ground_truth = read_gt(label_path, width, height)
    targets = [ground_truth[target_id] for target_id in pair["target_ids"] if target_id in ground_truth]
    if not targets:
        raise RuntimeError(f"no valid target boxes for {pair['dataset']} {pair['video']} {pair['frame']}")

    def top10_recall(boxes: list[list[float]]) -> float:
        return sum(
            int(max((iou(box, target) for box in boxes[:TOP_K]), default=0.0) >= 0.50)
            for target in targets
        ) / len(targets)

    return {
        "pair_index": pair["pair_index"],
        "dataset": pair["dataset"],
        "video": pair["video"],
        "frame": pair["frame"],
        "true_expression": pair["true_expression"],
        "shuffled_expression": pair["shuffled_expression"],
        "true_target_ids": list(pair["target_ids"]),
        "shuffled_target_ids": list(pair["shuffled_target_ids"]),
        "target_count": len(targets),
        "true_top10_recall_iou050": top10_recall(true_boxes),
        "shuffled_top10_recall_iou050": top10_recall(shuffled_boxes),
        "true_ap50_top300": average_precision(true_boxes[:MAX_PREDICTIONS], [], targets),
        "shuffled_ap50_top300": average_precision(shuffled_boxes[:MAX_PREDICTIONS], [], targets),
    }


def run_worker(args: argparse.Namespace) -> int:
    pairs = make_pairs()
    selected = pairs[args.worker_index :: args.worker_count]
    runtime = load_grounding_runtime(
        config_path=Path(args.config),
        checkpoint=Path(args.checkpoint),
        max_per_img=MAX_PREDICTIONS,
        device=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu",
        training_contract=False,
    )
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, pair in enumerate(selected):
        true_boxes, _ = sorted_predictions(runtime, pair["image"], pair["true_sentence"])
        shuffled_boxes, _ = sorted_predictions(runtime, pair["image"], pair["shuffled_sentence"])
        rows.append(pair_metrics(pair, true_boxes, shuffled_boxes))
        if args.progress and (index + 1) % args.progress == 0:
            print(f"worker={args.worker_index} {index + 1}/{len(selected)}", flush=True)
    payload = {
        "format": "locatemot-u-u1-expression-counterfactual-shard-v1",
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "pair_count": len(rows),
        "pairs": rows,
        "runtime": runtime_metadata(runtime, config_path=Path(args.config)),
        "official_test_labels_read": False,
        "screening_gt_used": False,
        "training_launched": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def aggregate(args: argparse.Namespace) -> int:
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.shards)]
    pairs = [pair for shard in shards for pair in shard["pairs"]]
    pairs.sort(key=lambda row: row["pair_index"])
    if len(pairs) != PAIR_COUNT or [row["pair_index"] for row in pairs] != list(range(PAIR_COUNT)):
        raise RuntimeError(f"expected pair indices 0..{PAIR_COUNT - 1}, got {len(pairs)} rows")
    if any(set(row["true_target_ids"]) == set(row["shuffled_target_ids"]) for row in pairs):
        raise RuntimeError("counterfactual donor target set unexpectedly matches true target set")

    def mean(key: str) -> float:
        return sum(float(row[key]) for row in pairs) / len(pairs)

    true_recall = mean("true_top10_recall_iou050")
    shuffled_recall = mean("shuffled_top10_recall_iou050")
    true_ap = mean("true_ap50_top300")
    shuffled_ap = mean("shuffled_ap50_top300")
    payload = {
        "format": "locatemot-u-u1-expression-counterfactual-audit-v1",
        "status": "FORMAL_COMPLETE_FIXED_500_LEGAL_VALIDATION_PAIRS",
        "pair_count": len(pairs),
        "protocol": {
            "split": "legal validation only",
            "unit": "same image with true sentence versus same-video sentence from a different target set",
            "target_iou_threshold": 0.50,
            "top_k_recall": TOP_K,
            "ap50_prediction_limit": MAX_PREDICTIONS,
            "label_read_order": "both query-conditioned forwards before source label",
        },
        "true_top10_recall_iou050_mean": true_recall,
        "shuffled_top10_recall_iou050_mean": shuffled_recall,
        "true_ap50_top300_mean": true_ap,
        "shuffled_ap50_top300_mean": shuffled_ap,
        "top10_recall_margin": true_recall - shuffled_recall,
        "ap50_margin": true_ap - shuffled_ap,
        "true_better_top10_recall": bool(true_recall > shuffled_recall),
        "true_better_ap50": bool(true_ap > shuffled_ap),
        "gate_pass": bool(true_recall > shuffled_recall and true_ap > shuffled_ap),
        "pairs": pairs,
        "runtime": shards[0]["runtime"],
        "official_test_labels_read": False,
        "screening_gt_used": False,
        "training_launched": False,
        "shards": [str(path) for path in args.shards],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "status", "pair_count", "true_top10_recall_iou050_mean", "shuffled_top10_recall_iou050_mean",
        "true_ap50_top300_mean", "shuffled_ap50_top300_mean", "gate_pass",
    )}, indent=2, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--progress", type=int, default=25)
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
