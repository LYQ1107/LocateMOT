"""Strict merge and optional TrackEval for Stage L20 grouped shards."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from tools.eval_l18_carr import (  # noqa: E402
    run_trackeval, trainval_queries, write_trainval_gt,
)
from tools.eval_l20_fast import (  # noqa: E402
    auc, load_manifest, ranking_summary, threshold_grid, threshold_summary,
)


REQUIRED = {
    "frame", "track_id", "box", "score", "raw_logit", "null_logit",
    "source", "group_id", "group_size", "cross_pool", "current_match",
    "membership", "observation", "gt_iou", "null_target", "state",
}


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def aggregate(records, threshold: float, split: str) -> dict:
    chosen = [(meta, data) for meta, data in records
              if split == "all" or meta["split"] == split]
    if not chosen:
        return {"queries": 0, "candidate_groups": 0}
    data = {
        key: np.concatenate([item[key] for _meta, item in chosen])
        for key in next(iter(chosen))[1]
    }
    result = {
        "queries": len(chosen),
        "candidate_groups": int(len(data["score"])),
        "threshold_metrics": threshold_summary(data, threshold),
        "ranking_metrics": ranking_summary(data),
    }
    return result


def prepare_trackeval(out_root: Path, records, threshold: float):
    result_root = out_root / "uidm18"
    allowed = set()
    for meta, data in records:
        if meta["split"] != "screening":
            continue
        video, expression = meta["video"], meta["expression"]
        allowed.add((video, expression))
        gt = ROOT / "outputs/l18/data/trainval_gt/kitti" / video / expression / "gt.txt"
        destination = result_root / video / expression / "gt.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.symlink_to(gt.resolve())
        prediction = result_root / video / expression / "predict.txt"
        prediction.parent.mkdir(parents=True, exist_ok=True)
        keep = data["score"] >= float(threshold)
        with prediction.open("w") as handle:
            for index in np.flatnonzero(keep):
                x1, y1, x2, y2 = [float(v) for v in data["box"][index]]
                score = 1.0 / (1.0 + np.exp(-np.clip(float(data["score"][index]), -40, 40)))
                handle.write(
                    f"{int(data['frame'][index]) + 1},{int(data['track_id'][index])},"
                    f"{x1:.3f},{y1:.3f},{x2-x1:.3f},{y2-y1:.3f},"
                    f"{score:.6f},-1,-1,-1\n"
                )
    return allowed


def main():
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
    manifest, manifest_sha = load_manifest(manifest_path)
    expected_rows = [row for row in manifest["queries"]
                     if args.split == "all" or row["split"] == args.split]
    expected = {int(row["query_index"]): row for row in expected_rows}
    if len(expected) != len(expected_rows):
        raise ValueError("duplicate query index in manifest")
    shards_root = (ROOT / args.shards_root).resolve()
    records = {}
    checkpoint_sha = cfg_sha = None
    for shard_index in range(args.num_shards):
        shard = shards_root if args.num_shards == 1 else \
            shards_root / f"shard_{shard_index}"
        run_path = shard / "run_manifest.json"
        if not run_path.exists():
            raise FileNotFoundError(run_path)
        run = json.loads(run_path.read_text())
        if not run.get("complete"):
            raise ValueError(f"incomplete shard: {run_path}")
        if run.get("manifest_sha256") != manifest_sha:
            raise ValueError(f"manifest SHA mismatch: shard {shard_index}")
        if int(run.get("num_shards", -1)) != args.num_shards or \
                int(run.get("shard_index", -1)) != shard_index:
            raise ValueError(f"shard topology mismatch: {run_path}")
        if checkpoint_sha is None:
            checkpoint_sha, cfg_sha = run.get("checkpoint_sha256"), run.get("cfg_sha256")
        elif checkpoint_sha != run.get("checkpoint_sha256") or \
                cfg_sha != run.get("cfg_sha256"):
            raise ValueError(f"checkpoint/config mismatch: shard {shard_index}")
        for index in run.get("completed_query_indices", []):
            index = int(index)
            if index not in expected:
                raise ValueError(f"unexpected query {index}")
            if index in records:
                raise ValueError(f"duplicate query {index}")
            result_path = shard / "queries" / f"q{index:05d}.json"
            cache_path = shard / "scores" / f"q{index:05d}.npz"
            marker = shard / "complete" / f"q{index:05d}.complete"
            if not result_path.exists() or not cache_path.exists() or not marker.exists():
                raise ValueError(f"query {index} lacks complete atomic outputs")
            meta = json.loads(result_path.read_text())
            if not meta.get("complete") or int(meta.get("query_index", -1)) != index:
                raise ValueError(f"invalid query result: {result_path}")
            if (meta.get("manifest_sha256") != manifest_sha or
                    meta.get("checkpoint_sha256") != checkpoint_sha or
                    meta.get("cfg_sha256") != cfg_sha):
                raise ValueError(f"query provenance mismatch: {result_path}")
            with np.load(cache_path, allow_pickle=False) as loaded:
                data = {key: np.asarray(loaded[key]) for key in loaded.files}
            if set(data) != REQUIRED:
                raise ValueError(f"minimal L20 fields mismatch: {cache_path}")
            if len(data["score"]) != int(meta["rows"]):
                raise ValueError(f"row count mismatch: {result_path}")
            records[index] = (meta, data)
    missing = sorted(set(expected) - set(records))
    if missing:
        raise ValueError(f"missing query indices ({len(missing)}): {missing[:20]}")
    ordered = [records[index] for index in sorted(records)]
    calibration = [data for meta, data in ordered if meta["split"] == "calibration"]
    if calibration:
        cal = {key: np.concatenate([data[key] for data in calibration])
               for key in calibration[0]}
    else:
        cal = {key: np.zeros(0, dtype=data.dtype)
               for key, data in ordered[0][1].items()}
    threshold = float(args.threshold) if args.threshold is not None else threshold_grid(cal)
    out_root = (ROOT / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    queries, gt_root, seqmap, _sequences, _protocol = trainval_queries("trainval_kitti")
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
        "all": aggregate(ordered, threshold, "all"), "trackeval": None,
    }
    with (out_root / "per_query.jsonl").open("w") as handle:
        for meta, data in ordered:
            item = dict(meta)
            item["threshold_metrics"] = threshold_summary(data, threshold)
            item["ranking_metrics"] = ranking_summary(data)
            handle.write(json.dumps(item) + "\n")
    if args.run_trackeval:
        allowed = prepare_trackeval(out_root, ordered, threshold)
        metrics, log = run_trackeval(
            "trainval_kitti", out_root, seqmap, _sequences, allowed)
        payload["trackeval"] = {
            "metrics": metrics, "log": str(log), "split": "screening",
        }
    atomic_json(out_root / "summary.json", payload)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
