"""Materialize L19 validation score caches and run the formal ablations.

The cache is produced by :mod:`tools.diagnose_l19`.  This keeps ablation
comparisons on exactly the same checkpoint, bank, query order and TrackEval
protocol; only the final score transformation and optional cross-pool
deduplication change.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from tools.eval_l18_carr import (  # noqa: E402
    run_trackeval, trainval_queries, write_trainval_gt,
)


MODES = (
    "raw",
    "no_gate",
    "coverage_aux_only",
    "no_reserve_bias",
    "source_norm",
    "query_norm",
    "cross_pool_dedup",
)


def safe_expression(text: str) -> str:
    return text.replace("/", "_")


def cache_path(cache_root: Path, dataset: str, video: str,
               expression: str) -> Path:
    return cache_root / dataset / video / f"{safe_expression(expression)}.npz"


def load_caches(cache_root: Path, dataset: str, queries):
    values = []
    for video, expression, _spec in queries:
        path = cache_path(cache_root, dataset, video, expression)
        if not path.exists():
            raise FileNotFoundError(path)
        values.append((video, expression, path, np.load(path, allow_pickle=False)))
    return values


def score_array(data, mode: str, norm_stats: dict,
                query_stat: dict | None = None) -> np.ndarray:
    raw = np.asarray(data["raw"], dtype=np.float32)
    no_gate = np.asarray(data["no_gate"], dtype=np.float32)
    bias = np.asarray(data["bias"], dtype=np.float32)
    if mode == "raw":
        return raw
    if mode in {"no_gate", "coverage_aux_only"}:
        return no_gate
    if mode == "no_reserve_bias":
        return raw - bias
    if mode == "source_norm":
        source = np.asarray(data["source"], dtype=np.int8)
        result = np.empty_like(raw)
        for pool in (0, 1):
            mask = source == pool
            mean = float(norm_stats[str(pool)]["mean"])
            std = max(1e-6, float(norm_stats[str(pool)]["std"]))
            result[mask] = (raw[mask] - mean) / std
        return result
    if mode == "query_norm":
        stat = query_stat or {"mean": 0.0, "std": 1.0}
        return (raw - float(stat["mean"])) / max(
            1e-6, float(stat["std"]))
    if mode == "cross_pool_dedup":
        return raw
    raise ValueError(mode)


def derive_norm_stats(caches) -> dict:
    stats = {}
    for pool in (0, 1):
        pieces = []
        for _video, _expression, _path, data in caches:
            source = np.asarray(data["source"], dtype=np.int8)
            raw = np.asarray(data["raw"], dtype=np.float32)
            values = raw[source == pool]
            if len(values):
                pieces.append(values)
        values = np.concatenate(pieces) if pieces else np.zeros(0, np.float32)
        stats[str(pool)] = {
            "count": int(len(values)),
            "mean": float(values.mean()) if len(values) else 0.0,
            "std": float(values.std()) if len(values) else 1.0,
        }
    return stats


def derive_query_stats(caches) -> dict[str, dict]:
    result = {}
    for _video, expression, path, data in caches:
        values = np.asarray(data["raw"], dtype=np.float32)
        result[str(path)] = {
            "video": _video, "expression": expression,
            "count": int(len(values)),
            "mean": float(values.mean()) if len(values) else 0.0,
            "std": float(values.std()) if len(values) else 1.0,
        }
    return result


def calibration_labels(data) -> np.ndarray:
    # gt_iou is computed against the current query's annotated boxes and the
    # candidate bank uses the same IoU>=.50 audit label.  It is validation-only
    # and never enters official prediction construction.
    return np.asarray(data["gt_iou"], dtype=np.float32) >= 0.50


def threshold_grid(caches, modes, norm_stats, query_stats) -> dict[str, float]:
    result = {}
    labels = []
    for _video, _expression, _path, data in caches:
        labels.append(calibration_labels(data))
    labels = np.concatenate(labels) if labels else np.zeros(0, bool)
    for mode in modes:
        scores = []
        for _video, _expression, path, data in caches:
            scores.append(score_array(data, mode, norm_stats,
                                      query_stats[str(path)]))
        scores = np.concatenate(scores) if scores else np.zeros(0, np.float32)
        if not len(scores) or not labels.any():
            result[mode] = 0.0
            continue
        # Pooled quantiles make the calibration scale-independent while
        # retaining a single threshold for every validation video/query.
        candidates = np.unique(np.quantile(
            scores, np.linspace(0.01, 0.995, 80))).tolist()
        candidates.extend([float(scores.min()), float(scores.max())])
        best = None
        for threshold in candidates:
            selected = scores >= threshold
            tp = int(np.count_nonzero(selected & labels))
            fp = int(np.count_nonzero(selected & ~labels))
            fn = int(np.count_nonzero(~selected & labels))
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
            key = (f1, recall, -abs(float(threshold)))
            if best is None or key > best[0]:
                best = (key, float(threshold), tp, fp, fn, precision, recall)
        result[mode] = best[1]
    return result


def deduplicate(data, scores: np.ndarray) -> np.ndarray:
    frame = np.asarray(data["frame"], dtype=np.int64)
    group = np.asarray(data["group"], dtype=np.int64)
    keep = np.ones(len(scores), dtype=bool)
    grouped = defaultdict(list)
    for index, (frame_id, group_id) in enumerate(zip(frame.tolist(), group.tolist())):
        grouped[(int(frame_id), int(group_id))].append(index)
    for indices in grouped.values():
        if len(indices) <= 1:
            continue
        winner = max(indices, key=lambda index: (float(scores[index]), -index))
        for index in indices:
            keep[index] = index == winner
    return keep


def write_prediction(path: Path, data, scores: np.ndarray, mode: str,
                     threshold: float, frame_offset: int = 0) -> dict:
    frame = np.asarray(data["frame"], dtype=np.int64)
    track_id = np.asarray(data["track_id"], dtype=np.int64)
    boxes = np.asarray(data["box"], dtype=np.float32)
    keep = scores >= float(threshold)
    before_dedup = int(keep.sum())
    if mode == "cross_pool_dedup":
        dedup_keep = deduplicate(data, scores)
        keep &= dedup_keep
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for index in np.flatnonzero(keep):
            x1, y1, x2, y2 = [float(value) for value in boxes[index]]
            handle.write(
                f"{int(frame[index]) + int(frame_offset)},{int(track_id[index])},{x1:.3f},{y1:.3f},"
                f"{x2-x1:.3f},{y2-y1:.3f},{1.0/(1.0+math.exp(-float(np.clip(scores[index], -40, 40)))):.6f},-1,-1,-1\n"
            )
    return {"selected_before_dedup": before_dedup,
            "selected": int(keep.sum()),
            "dedup_removed": before_dedup - int(keep.sum())}


def prepare_gt(out_root: Path, dataset: str, queries, gt_root: Path) -> None:
    result_root = out_root / "uidm18"
    for video, expression, _spec in queries:
        source = gt_root / video / expression / "gt.txt"
        destination = result_root / video / expression / "gt.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            continue
        destination.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("trainval_kitti", "trainval_dance"),
                        required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--modes", nargs="*", default=list(MODES))
    parser.add_argument("--threshold", type=float, default=None,
                        help="Use one supplied global threshold for all modes")
    parser.add_argument("--norm-stats", default="",
                        help="Optional pooled source-normalization JSON")
    parser.add_argument("--skip-trackeval", action="store_true")
    args = parser.parse_args()
    unknown = set(args.modes) - set(MODES)
    if unknown:
        raise ValueError(f"unknown modes: {sorted(unknown)}")
    queries, gt_root, seqmap, sequences, _protocol = trainval_queries(args.dataset)
    write_trainval_gt(args.dataset, queries, gt_root)
    caches = load_caches((ROOT / args.cache_root).resolve(), args.dataset, queries)
    norm_stats = derive_norm_stats(caches)
    if args.norm_stats:
        norm_stats = json.loads((ROOT / args.norm_stats).resolve().read_text())
    query_stats = derive_query_stats(caches)
    thresholds = threshold_grid(caches, args.modes, norm_stats, query_stats)
    if args.threshold is not None:
        thresholds = {mode: float(args.threshold) for mode in args.modes}
    out_root = (ROOT / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "source_norm_stats.json").write_text(
        json.dumps(norm_stats, indent=2) + "\n")
    payload = {
        "dataset": args.dataset,
        "checkpoint": str((ROOT / args.checkpoint).resolve()),
        "cache_root": str((ROOT / args.cache_root).resolve()),
        "queries": len(queries), "candidate_rows": 0,
        "thresholds": thresholds, "source_norm_stats": norm_stats,
        "query_norm_stats": query_stats,
        "modes": {},
    }
    payload["candidate_rows"] = int(sum(len(data["raw"])
                                       for _v, _e, _p, data in caches))
    for mode in args.modes:
        mode_root = out_root / mode
        mode_root.mkdir(parents=True, exist_ok=True)
        prepare_gt(mode_root, args.dataset, queries, gt_root)
        per_query = []
        for video, expression, path, data in caches:
            scores = score_array(data, mode, norm_stats,
                                 query_stats[str(path)])
            prediction = mode_root / "uidm18" / video / expression / "predict.txt"
            diag = write_prediction(prediction, data, scores, mode,
                                    thresholds[mode],
                                    0 if args.dataset == "trainval_dance" else 1)
            per_query.append({"video": video, "expression": expression, **diag})
        metrics = {}
        log = None
        if not args.skip_trackeval:
            metrics, log = run_trackeval(args.dataset, mode_root, seqmap,
                                         sequences,
                                         {(video, expression) for video, expression, _ in queries})
        payload["modes"][mode] = {
            "threshold": thresholds[mode], "queries": per_query,
            "selected": int(sum(row["selected"] for row in per_query)),
            "dedup_removed": int(sum(row["dedup_removed"] for row in per_query)),
            "trackeval": metrics, "trackeval_log": str(log) if log else None,
        }
        print(json.dumps({"mode": mode, "threshold": thresholds[mode],
                          "selected": payload["modes"][mode]["selected"],
                          "trackeval": metrics}, indent=2), flush=True)
    output = out_root / "ablation_summary.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"output={output}")


if __name__ == "__main__":
    main()
