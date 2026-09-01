"""Mine L20 hard negatives from a *validated training-only* score cache.

The miner deliberately performs a provenance pass over every source shard and
query before opening any score array.  A non-training source invalidates the
whole requested output; it is never repaired by filtering a few rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from tools.train_l18_carr import load_items  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def invalid_output(path: Path, reason: str) -> None:
    """Mark an output as invalid without deleting or partially rewriting it."""
    marker = path.with_suffix(path.suffix + ".INVALID.md")
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        "# INVALID L20 hard-negative cache\n\n"
        "This cache is not eligible for training.\n\n"
        f"Reason: {reason}\n"
    )


def train_inventory() -> tuple[dict, dict, dict]:
    items, protocol = load_items()
    split_sets = {}
    for domain, key in (("kitti", "kitti_v2"), ("dance", "refer_dance")):
        split_sets[domain] = {
            split: set(protocol[key][split])
            for split in ("train", "train_val", "official_eval")
        }
        train = split_sets[domain]["train"]
        if train & split_sets[domain]["train_val"]:
            raise ValueError(f"train/train_val overlap for {domain}")
        if train & split_sets[domain]["official_eval"]:
            raise ValueError(f"train/official overlap for {domain}")
        if split_sets[domain]["train_val"] & split_sets[domain]["official_eval"]:
            raise ValueError(f"train_val/official overlap for {domain}")
    by_key = {}
    for domain in ("kitti", "dance"):
        for item in items["train"][domain]:
            video = str(item["video"])
            if video not in split_sets[domain]["train"]:
                raise ValueError(f"load_items train item outside train videos: {item}")
            expression = str(item["entry"].get(
                "sentence", item["entry"].get("expression", "")))
            by_key[(domain, video, expression)] = item
    return items, protocol, {"split_sets": split_sets, "by_key": by_key}


def load_source_manifests(shards_root: Path, num_shards: int,
                          output: Path) -> list[tuple[Path, dict]]:
    roots = [shards_root] if num_shards == 1 else [
        shards_root / f"shard_{index}" for index in range(num_shards)]
    result = []
    try:
        for shard in roots:
            run_path = shard / "run_manifest.json"
            if not run_path.exists():
                raise ValueError(f"missing run manifest: {run_path}")
            run = json.loads(run_path.read_text())
            if not run.get("complete"):
                raise ValueError(f"incomplete source shard: {shard}")
            if run.get("data_split") != "train":
                raise ValueError(
                    "source shard is not train-only: "
                    f"{shard} has data_split={run.get('data_split')!r}")
            if run.get("split") not in ("train", "train_only"):
                raise ValueError(f"source shard split is not train: {shard}")
            if run.get("query_source") != "load_items()[train]":
                raise ValueError(f"source query source is not load_items()[train]: {shard}")
            if run.get("gt_source") != "train_sidecar_labels":
                raise ValueError(f"source GT is not train sidecar labels: {shard}")
            if run.get("official_gt_used") is not False:
                raise ValueError(f"source does not prove official_gt_used=false: {shard}")
            if not run.get("checkpoint_sha256"):
                raise ValueError(f"source lacks checkpoint SHA: {shard}")
            result.append((shard, run))
    except Exception as error:
        invalid_output(output, str(error))
        raise
    return result


def query_manifest_rows(args, source_runs: list[tuple[Path, dict]],
                        inventory: dict, output: Path) -> dict[int, dict]:
    by_index = {}
    try:
        for shard, run in source_runs:
            manifest_path = run.get("query_manifest")
            if manifest_path:
                path = Path(manifest_path)
                if not path.is_absolute():
                    path = (ROOT / path).resolve()
                if not path.exists():
                    raise ValueError(f"missing train query manifest: {path}")
                manifest = json.loads(path.read_text())
                if manifest.get("split") != "train" or \
                        manifest.get("query_source") != "load_items()[train]":
                    raise ValueError(f"query manifest is not train-only: {path}")
                rows = manifest.get("queries", [])
            else:
                rows = None
            completed = [int(index) for index in
                         run.get("completed_query_indices", [])]
            for index in completed:
                result_path = shard / "queries" / f"q{index:05d}.json"
                if not result_path.exists():
                    raise ValueError(f"missing query provenance: {result_path}")
                row = json.loads(result_path.read_text())
                if rows is not None:
                    row_lookup = {int(value["query_index"]): value
                                  for value in rows}
                    if index not in row_lookup:
                        raise ValueError(f"query {index} missing from manifest {path}")
                    row = {**row_lookup[index], **row}
                if row.get("data_split") != "train" or \
                        row.get("split") not in ("train", "train_only") or \
                        row.get("official_gt_used") is not False:
                    raise ValueError(f"query {index} is not proven train-only: {result_path}")
                domain = str(row.get("domain", "kitti"))
                key = (domain, str(row["video"]), str(row["expression"]))
                if key not in inventory["by_key"]:
                    raise ValueError(f"query {index} is outside load_items()[train]: {key}")
                if index in by_index:
                    raise ValueError(f"duplicate query index across shards: {index}")
                by_index[index] = row
    except Exception as error:
        invalid_output(output, str(error))
        raise
    return by_index


def mine(args):
    output = (ROOT / args.out).resolve()
    _items, protocol, inventory = train_inventory()
    shards_root = (ROOT / args.shards_root).resolve()
    source_runs = load_source_manifests(shards_root, args.num_shards, output)
    by_index = query_manifest_rows(args, source_runs, inventory, output)
    records = []
    counts = Counter()
    try:
        for shard, run in source_runs:
            for query_index in run["completed_query_indices"]:
                query_index = int(query_index)
                row = by_index[query_index]
                path = shard / "scores" / f"q{query_index:05d}.npz"
                if not path.exists():
                    raise ValueError(f"missing score cache: {path}")
                with np.load(path, allow_pickle=False) as loaded:
                    data = {key: np.asarray(loaded[key]) for key in loaded.files}
                required = {"frame", "track_id", "group_id", "source", "score",
                            "raw_logit", "null_logit", "current_match",
                            "null_target"}
                if set(data) < required:
                    raise ValueError(f"score cache lacks hard-negative fields: {path}")
                selected = []
                for frame_id in np.unique(data["frame"]):
                    frame = np.flatnonzero(data["frame"] == frame_id)
                    for source_id, name in ((0, "main_hard_negative"),
                                            (1, "reserve_hard_negative"),
                                            (2, "cross_pool_hard_negative")):
                        pool = frame[(data["source"][frame] == source_id) &
                                     (data["current_match"][frame] == 0)]
                        if len(pool):
                            index = int(pool[np.argmax(data["score"][pool])])
                            selected.append((name, index))
                    if data["null_target"][frame][0]:
                        index = int(frame[np.argmax(data["score"][frame])])
                        selected.append(("absent_or_uncovered_hard_negative", index))
                if len(selected) > args.max_per_query:
                    selected = sorted(
                        selected, key=lambda pair: float(data["score"][pair[1]]),
                        reverse=True)[:args.max_per_query]
                for reason, index in selected:
                    record = {
                        "query_index": query_index, "domain": row.get("domain", "kitti"),
                        "video": row["video"], "expression": row["expression"],
                        "split": "train", "frame": int(data["frame"][index]),
                        "track_id": int(data["track_id"][index]),
                        "group_id": int(data["group_id"][index]),
                        "source": int(data["source"][index]),
                        "score": float(data["score"][index]),
                        "raw_logit": float(data["raw_logit"][index]),
                        "null_logit": float(data["null_logit"][index]),
                        "reason": reason,
                        "label": int(data["current_match"][index]),
                        "provenance": "train-sidecar L20 checkpoint output; official_gt_used=false",
                    }
                    records.append(record)
                    counts[reason] += 1
    except Exception as error:
        invalid_output(output, str(error))
        raise

    records.sort(key=lambda value: (value["query_index"], value["frame"],
                                    value["reason"], value["group_id"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        invalid_output(output, "refusing to overwrite an existing cache")
        raise FileExistsError(output)
    output.write_text("\n".join(json.dumps(row, sort_keys=True) for row in records) +
                      ("\n" if records else ""))
    checkpoint_shas = sorted({run["checkpoint_sha256"] for _path, run in source_runs})
    source_sha256 = {
        "split_manifest": sha256_file(ROOT / "outputs/l16/data/protocol/split_manifest.json"),
        "source_run_manifests": {
            str(path): sha256_file(path / "run_manifest.json")
            for path, _run in source_runs
        },
    }
    manifest = {
        "format": "locatemot-l20-hard-negatives-v2",
        "valid": True, "split": "train", "query_source": "load_items()[train]",
        "gt_source": "train_sidecar_labels", "official_gt_used": False,
        "videos": sorted({(row.get("domain", "kitti"), row["video"])
                           for row in by_index.values()}),
        "queries": [by_index[index] for index in sorted(by_index)],
        "source_sha256": source_sha256,
        "checkpoint_sha256": checkpoint_shas[0] if len(checkpoint_shas) == 1 else checkpoint_shas,
        "query_count": len(by_index), "record_count": len(records),
        "counts": dict(counts), "max_per_query": args.max_per_query,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    summary = {key: manifest[key] for key in
               ("format", "valid", "split", "official_gt_used", "query_count",
                "record_count", "counts", "max_per_query")}
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-root", required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-per-query", type=int, default=128)
    parser.add_argument("--out", default="outputs/l20/protocol/hard_negatives.jsonl")
    # Kept as an explicit option for compatibility, but it is never trusted as
    # the train inventory; query provenance must also be embedded in the run.
    parser.add_argument("--query-manifest", default="")
    args = parser.parse_args()
    mine(args)


if __name__ == "__main__":
    main()
