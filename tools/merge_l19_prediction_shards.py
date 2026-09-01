"""Merge strict, independent formal RMOT prediction shards.

This handles the prediction-only shards produced by ``eval_l18_carr.py``.
It copies only disjoint per-query ``predict.txt`` files into a new TrackEval
root and rejects missing/duplicate query pairs or provenance mismatches.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.eval_l18_carr import trainval_queries, write_trainval_gt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("num-shards must be positive")
    shards_root = (ROOT / args.shards_root).resolve()
    out_root = (ROOT / args.out_root).resolve()
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"refusing to merge into non-empty path: {out_root}")
    out_root.mkdir(parents=True, exist_ok=True)
    queries, gt_root, _seqmap, _sequences, _protocol = trainval_queries(
        "trainval_kitti")
    expected = {(video, expression) for video, expression, _ in queries}
    seen = {}
    checkpoint_sha = None
    source_manifests = []
    for shard_index in range(args.num_shards):
        shard_root = shards_root / f"shard_{shard_index}"
        manifest_candidates = sorted(
            shard_root.glob(f"prediction_manifest_shard{shard_index}of*.json"))
        if len(manifest_candidates) != 1:
            raise FileNotFoundError(
                f"expected one manifest for shard {shard_index}: {shard_root}")
        manifest_path = manifest_candidates[0]
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("dataset") != "trainval_kitti":
            raise ValueError(f"dataset mismatch: {manifest_path}")
        current_sha = manifest.get("checkpoint_sha256")
        if checkpoint_sha is None:
            checkpoint_sha = current_sha
        elif current_sha != checkpoint_sha:
            raise ValueError(f"checkpoint SHA mismatch: {manifest_path}")
        source_manifests.append(str(manifest_path))
        source_root = shard_root / "uidm18"
        for item in manifest.get("queries", []):
            pair = (item.get("video"), item.get("expression"))
            if pair not in expected:
                raise ValueError(f"unexpected query pair in {manifest_path}: {pair}")
            if pair in seen:
                raise ValueError(f"duplicate query pair {pair}: shards "
                                 f"{seen[pair]} and {shard_index}")
            source = source_root / pair[0] / pair[1] / "predict.txt"
            if not source.exists():
                raise FileNotFoundError(source)
            seen[pair] = shard_index
            destination = out_root / "uidm18" / pair[0] / pair[1] / "predict.txt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
    missing = sorted(expected - set(seen))
    if missing:
        raise ValueError(f"missing query pairs: {missing[:10]} ({len(missing)} total)")
    # Recreate the protocol GT links in the merged root; source shard GT links
    # are not treated as evidence and are never silently omitted.
    write_trainval_gt("trainval_kitti", queries, gt_root)
    for video, expression, _spec in queries:
        source = gt_root / video / expression / "gt.txt"
        destination = out_root / "uidm18" / video / expression / "gt.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            destination.symlink_to(source.resolve())
    payload = {
        "complete": True, "dataset": "trainval_kitti",
        "num_shards": args.num_shards, "query_count": len(seen),
        "expected_query_count": len(expected),
        "checkpoint_sha256": checkpoint_sha,
        "source_manifests": source_manifests,
    }
    (out_root / "merge_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
