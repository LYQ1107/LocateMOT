"""Validate and merge Stage L19 fast-eval query shards.

The merger is deliberately strict: a run with a missing query, duplicate
query index, mismatched checkpoint/config/manifest SHA, temporary file, or
missing completion marker is rejected rather than summarized as complete.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.eval_l18_carr import (  # noqa: E402
    run_trackeval, trainval_queries, write_trainval_gt,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def safe_expression(text: str) -> str:
    return str(text).replace("/", "_")


def auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, np.float64).reshape(-1)
    labels = np.asarray(labels, bool).reshape(-1)
    valid = np.isfinite(scores)
    scores, labels = scores[valid], labels[valid]
    positive = int(labels.sum())
    negative = int((~labels).sum())
    if not positive or not negative:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), np.float64)
    sorted_scores = scores[order]
    begin = 0
    while begin < len(order):
        end = begin + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[begin]:
            end += 1
        ranks[order[begin:end]] = (begin + end - 1) / 2.0 + 1.0
        begin = end
    return float((ranks[labels].sum() - positive * (positive + 1) / 2.0) /
                 (positive * negative))


def threshold_summary(scores: np.ndarray, labels: np.ndarray,
                      source: np.ndarray, threshold: float) -> dict:
    selected = scores >= float(threshold)
    tp = int(np.count_nonzero(selected & labels))
    fp = int(np.count_nonzero(selected & ~labels))
    fn = int(np.count_nonzero(~selected & labels))
    result = {
        "threshold": float(threshold), "selected": int(selected.sum()),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
    }
    for source_id, name in ((0, "main"), (1, "reserve")):
        pool = source == source_id
        pool_selected = pool & selected
        pool_positive = pool & labels
        pool_tp = int(np.count_nonzero(pool_selected & labels))
        result[f"{name}_selected"] = int(pool_selected.sum())
        result[f"{name}_positive"] = int(pool_positive.sum())
        result[f"{name}_selected_recall"] = float(
            pool_tp / max(1, int(pool_positive.sum())))
        result[f"{name}_selected_precision"] = float(
            pool_tp / max(1, int(pool_selected.sum())))
    return result


def threshold_grid(scores: np.ndarray, labels: np.ndarray) -> float:
    if not len(scores) or not labels.any():
        return 0.0
    candidates = np.unique(np.quantile(
        scores, np.linspace(0.01, 0.995, 96))).tolist()
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
            best = (key, float(threshold))
    return best[1]


def ranking_summary(arrays: list[dict[str, np.ndarray]]) -> dict:
    mrr, aps, ranks, reserve_ranks = [], [], [], []
    all_scores, all_labels = [], []
    reserve_scores, main_negative_scores = [], []
    for data in arrays:
        frame = data["frame"]
        score = data["score"]
        label = data["current_match"].astype(bool)
        source = data["source"]
        all_scores.append(score)
        all_labels.append(label)
        for frame_id in np.unique(frame):
            indices = np.flatnonzero(frame == frame_id)
            order = indices[np.argsort(-score[indices], kind="stable")]
            ordered = label[order]
            positive_positions = np.flatnonzero(ordered)
            if not len(positive_positions):
                continue
            mrr.append(1.0 / float(positive_positions[0] + 1))
            precisions = [float(ordered[:pos + 1].mean())
                          for pos in positive_positions]
            aps.append(float(np.mean(precisions)))
            rank_by_index = {int(index): rank
                             for rank, index in enumerate(order, 1)}
            positive_indices = indices[label[indices]]
            positive_values = [rank_by_index[int(index)]
                               for index in positive_indices]
            ranks.extend(positive_values)
            reserve_indices = indices[(source[indices] == 1) & label[indices]]
            reserve_ranks.extend([rank_by_index[int(index)]
                                  for index in reserve_indices])
            reserve_scores.extend(score[reserve_indices].tolist())
            main_negative_scores.extend(
                score[indices[(source[indices] == 0) & ~label[indices]]].tolist())
    scores = np.concatenate(all_scores) if all_scores else np.zeros(0, np.float32)
    labels = np.concatenate(all_labels) if all_labels else np.zeros(0, bool)
    def stats(values: list[float]) -> dict:
        if not values:
            return {"count": 0}
        values = np.asarray(values, np.float64)
        return {"count": int(len(values)), "mean": float(values.mean()),
                "median": float(np.median(values)),
                "q90": float(np.quantile(values, 0.90))}
    return {
        "mrr": float(np.mean(mrr)) if mrr else None,
        "frame_ap": float(np.mean(aps)) if aps else None,
        "positive_frame_rank": stats(ranks),
        "reserve_positive_rank": stats(reserve_ranks),
        "reserve_vs_main_negative_auc": auc_from_scores(
            np.asarray(reserve_scores + main_negative_scores),
            np.asarray([True] * len(reserve_scores) +
                       [False] * len(main_negative_scores))),
        "positive_vs_negative_auc": auc_from_scores(scores, labels),
        "reserve_positive_score": stats(reserve_scores),
        "main_negative_score": stats(main_negative_scores),
    }


def aggregate(records: list[tuple[dict, dict[str, np.ndarray]]],
              threshold: float, split: str) -> dict:
    chosen = [(meta, data) for meta, data in records
              if split == "all" or meta["split"] == split]
    if not chosen:
        return {"queries": 0, "threshold": threshold}
    scores = np.concatenate([data["score"] for _meta, data in chosen])
    labels = np.concatenate([data["current_match"].astype(bool)
                             for _meta, data in chosen])
    source = np.concatenate([data["source"] for _meta, data in chosen])
    result = {
        "queries": len(chosen),
        "candidate_rows": int(len(scores)),
        "threshold_metrics": threshold_summary(scores, labels, source, threshold),
        "ranking_metrics": ranking_summary([data for _meta, data in chosen]),
    }
    return result


def prepare_trackeval(out_root: Path, queries, gt_root: Path,
                      records: list[tuple[dict, dict[str, np.ndarray]]],
                      threshold: float) -> tuple[set[tuple[str, str]], Path]:
    result_root = out_root / "uidm18"
    allowed = set()
    for meta, data in records:
        if meta["split"] != "screening":
            continue
        video, expression = meta["video"], meta["expression"]
        allowed.add((video, expression))
        gt = gt_root / video / expression / "gt.txt"
        destination = result_root / video / expression / "gt.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.symlink_to(gt.resolve())
        prediction = result_root / video / expression / "predict.txt"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        keep = data["score"] >= float(threshold)
        with prediction.open("w") as handle:
            for index in np.flatnonzero(keep):
                x1, y1, x2, y2 = [float(value) for value in data["box"][index]]
                handle.write(
                    f"{int(data['frame'][index]) + 1},{int(data['track_id'][index])},"
                    f"{x1:.3f},{y1:.3f},{x2-x1:.3f},{y2-y1:.3f},"
                    f"{1.0/(1.0+math.exp(-float(np.clip(data['score'][index], -40, 40)))):.6f},-1,-1,-1\n"
                )
    return allowed, gt_root / "seqmap.txt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-manifest", default=
                        "outputs/l19/protocol/kitti_fast_eval_manifest.json")
    parser.add_argument("--shards-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--split", choices=("all", "calibration", "screening"),
                        default="all")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--run-trackeval", action="store_true")
    args = parser.parse_args()
    manifest_path = (ROOT / args.query_manifest).resolve()
    manifest_sha = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    expected_rows = [row for row in manifest["queries"]
                     if args.split == "all" or row["split"] == args.split]
    expected = {int(row["query_index"]): row for row in expected_rows}
    if len(expected) != len(expected_rows):
        raise ValueError("duplicate query index in manifest")

    shards_root = (ROOT / args.shards_root).resolve()
    records = {}
    checkpoint_sha = cfg_sha = None
    for shard_index in range(args.num_shards):
        shard = (shards_root if args.num_shards == 1 else
                 shards_root / f"shard_{shard_index}")
        run_path = shard / "run_manifest.json"
        if not run_path.exists():
            raise FileNotFoundError(run_path)
        run = json.loads(run_path.read_text())
        if not run.get("complete"):
            raise ValueError(f"shard is incomplete: {run_path}")
        if run.get("manifest_sha256") != manifest_sha:
            raise ValueError(f"manifest SHA mismatch in shard {shard_index}")
        if int(run.get("num_shards", -1)) != args.num_shards or \
                int(run.get("shard_index", -1)) != shard_index:
            raise ValueError(f"shard topology mismatch: {run_path}")
        if checkpoint_sha is None:
            checkpoint_sha = run.get("checkpoint_sha256")
            cfg_sha = run.get("cfg_sha256")
        elif (checkpoint_sha != run.get("checkpoint_sha256") or
              cfg_sha != run.get("cfg_sha256")):
            raise ValueError(f"checkpoint/config SHA mismatch in shard {shard_index}")
        for index in run.get("completed_query_indices", []):
            index = int(index)
            if index not in expected:
                raise ValueError(f"unexpected query {index} in shard {shard_index}")
            if index in records:
                raise ValueError(f"duplicate query {index}")
            result_path = shard / "queries" / f"q{index:05d}.json"
            cache_path = shard / "scores" / f"q{index:05d}.npz"
            marker = shard / "complete" / f"q{index:05d}.complete"
            if not result_path.exists() or not cache_path.exists() or not marker.exists():
                raise ValueError(f"query {index} lacks atomic completion files")
            result = json.loads(result_path.read_text())
            if not result.get("complete") or int(result.get("query_index", -1)) != index:
                raise ValueError(f"invalid query result {result_path}")
            if result.get("checkpoint_sha256") != checkpoint_sha or \
                    result.get("manifest_sha256") != manifest_sha or \
                    result.get("cfg_sha256") != cfg_sha:
                raise ValueError(f"query provenance mismatch: {result_path}")
            with np.load(cache_path, allow_pickle=False) as loaded:
                data = {key: np.asarray(loaded[key]) for key in loaded.files}
            required = {"frame", "track_id", "box", "score", "source",
                        "current_match", "gt_iou"}
            if set(data) != required:
                raise ValueError(f"minimal cache fields mismatch: {cache_path}")
            if len(data["score"]) != int(result["rows"]):
                raise ValueError(f"result/cache row mismatch: {result_path}")
            records[index] = (result, data)
    missing = sorted(set(expected) - set(records))
    if missing:
        raise ValueError(f"missing query indices: {missing[:20]} ({len(missing)} total)")

    ordered = [(records[index][0], records[index][1])
               for index in sorted(records)]
    calibration_arrays = [data for meta, data in ordered
                          if meta["split"] == "calibration"]
    cal_scores = np.concatenate([data["score"] for data in calibration_arrays]) \
        if calibration_arrays else np.zeros(0, np.float32)
    cal_labels = np.concatenate([data["current_match"].astype(bool)
                                 for data in calibration_arrays]) \
        if calibration_arrays else np.zeros(0, bool)
    threshold = float(args.threshold) if args.threshold is not None \
        else threshold_grid(cal_scores, cal_labels)
    out_root = (ROOT / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    queries, gt_root, seqmap, sequences, _protocol = trainval_queries(
        "trainval_kitti")
    write_trainval_gt("trainval_kitti", queries, gt_root)
    payload = {
        "complete": True, "dataset": "trainval_kitti",
        "manifest": str(manifest_path), "manifest_sha256": manifest_sha,
        "shards_root": str(shards_root), "num_shards": args.num_shards,
        "checkpoint_sha256": checkpoint_sha, "cfg_sha256": cfg_sha,
        "query_count": len(ordered), "expected_query_count": len(expected),
        "threshold": threshold,
        "calibration": aggregate(ordered, threshold, "calibration"),
        "screening": aggregate(ordered, threshold, "screening"),
        "all": aggregate(ordered, threshold, "all"),
        "trackeval": None,
    }
    per_query_path = out_root / "per_query.jsonl"
    with per_query_path.open("w") as handle:
        for meta, data in ordered:
            item = dict(meta)
            item["threshold_metrics"] = threshold_summary(
                data["score"], data["current_match"].astype(bool),
                data["source"], threshold)
            item["ranking_metrics"] = ranking_summary([data])
            handle.write(json.dumps(item) + "\n")
    summary_path = out_root / "summary.json"
    if args.run_trackeval:
        allowed, seqmap = prepare_trackeval(
            out_root, queries, gt_root, ordered, threshold)
        metrics, log = run_trackeval(
            "trainval_kitti", out_root, seqmap, sequences, allowed)
        payload["trackeval"] = {"metrics": metrics, "log": str(log),
                                "split": "screening"}
    atomic_json(summary_path, payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
