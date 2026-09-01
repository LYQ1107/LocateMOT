"""Build train-only fragment-pair supervision for the L20 graph.

Only cached train annotations and frozen bank provenance are used.  The output
contains pair metadata, not copied visual features, so it is compact and can
be regenerated from the immutable bank.  No official V1/V2/Dance GT enters
the file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from tools.l20_common import BankStore, build_l20_buckets, query_identity_set  # noqa: E402
from tools.train_l18_carr import load_items  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bank_path(root: Path, dataset: str, video: str) -> Path:
    candidates = [root / dataset / f"{video}.pt"]
    candidates.extend([
        ROOT / "outputs/l16/track_banks_dedup" / dataset / f"{video}.pt",
        ROOT / "outputs/l16/track_banks" / dataset / f"{video}.pt",
    ])
    if dataset == "dance_train":
        candidates.extend([
            ROOT / "outputs/l16/track_banks_dedup/dance_train" / f"{video}.pt",
            ROOT / "outputs/l16/track_banks/dance_train" / f"{video}.pt",
        ])
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(candidates[0])


def validate_train_inventory(items: dict, protocol: dict) -> list[dict]:
    all_items = []
    for domain, protocol_key in (("kitti", "kitti_v2"),
                                 ("dance", "refer_dance")):
        split_map = protocol[protocol_key]
        train = set(split_map["train"])
        train_val = set(split_map["train_val"])
        official = set(split_map["official_eval"])
        if train & (train_val | official) or train_val & official:
            raise ValueError(f"split overlap in {domain}")
        for item in items["train"][domain]:
            if item["video"] not in train:
                raise ValueError(f"non-train item in pair input: {item}")
            all_items.append(item)
    return all_items


def expression_text(entry: dict) -> str:
    return str(entry.get("sentence", entry.get("expression", "")))


def rows_by_frame(bank: dict) -> dict[int, list[dict]]:
    tensors = bank["tensors"]
    labels = bank.get("candidate_gt", [])
    groups = tensors.get("observation_group_id")
    result = defaultdict(list)
    for frame_index, frame_id in enumerate(tensors["frame_ids"].tolist()):
        begin, end = map(int, tensors["frame_ptr"][frame_index:frame_index + 2])
        for row in range(begin, end):
            result[int(frame_id)].append({
                "frame": int(frame_id), "row": int(row),
                "track_id": int(tensors["track_id"][row]),
                "source": int(tensors.get("pool_id", torch.zeros(
                    len(tensors["track_id"]), dtype=torch.long))[row]),
                "group_id": int(groups[row]) if groups is not None else int(row),
                "gt_id": None if labels[row] is None else str(labels[row]),
            })
    return dict(result)


def make_pair(base: dict, other: dict, label: int, reason: str,
              query_index: int, video: str, expression: str,
              target_ids: list[str]) -> dict:
    gap = abs(int(other["frame"]) - int(base["frame"]))
    return {
        "query_index": int(query_index), "video": video,
        "expression": expression, "target_ids": target_ids,
        "label": int(label), "reason": reason,
        "frame_a": int(base["frame"]), "frame_b": int(other["frame"]),
        "track_id_a": int(base["track_id"]), "track_id_b": int(other["track_id"]),
        "group_id_a": int(base["group_id"]), "group_id_b": int(other["group_id"]),
        "source_a": int(base["source"]), "source_b": int(other["source"]),
        "gt_id_a": base["gt_id"], "gt_id_b": other["gt_id"],
        "time_gap": int(gap),
    }


def build(args):
    items, protocol = load_items()
    store = BankStore((ROOT / args.bank_root).resolve(), cache_size=1)
    all_items = validate_train_inventory(items, protocol)
    all_items.sort(key=lambda item: (
        item["domain"], item["video"], expression_text(item["entry"])))
    output = (ROOT / args.out).resolve()
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest_path.exists():
        marker = output.with_suffix(output.suffix + ".INVALID.md")
        marker.write_text(
            "# INVALID L20 pair-supervision cache\n\n"
            "Existing output was not overwritten; inspect its manifest before reuse.\n"
        )
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    counts = Counter()
    query_count = 0
    pair_count = 0
    seen_banks = {}
    with output.open("w") as handle:
        for query_index, item in enumerate(all_items):
            if args.max_queries and query_count >= args.max_queries:
                break
            target_ids = sorted(query_identity_set(item["entry"]))
            if not target_ids:
                continue
            key = (item["bank_dataset"], item["video"])
            if key not in seen_banks:
                seen_banks[key] = rows_by_frame(store.get(*key))
            by_frame = seen_banks[key]
            occurrences = defaultdict(list)
            for rows in by_frame.values():
                for row in rows:
                    if row["gt_id"] in target_ids:
                        occurrences[row["gt_id"]].append(row)
            candidates = []
            # Same-identity pairs: consecutive temporal views, endpoints, and
            # explicit main/reserve pairs are all retained before sampling.
            for gt_id in target_ids:
                values = sorted(occurrences.get(gt_id, []),
                                key=lambda row: (row["frame"], row["source"], row["row"]))
                for left, right in zip(values, values[1:]):
                    if left["frame"] != right["frame"]:
                        candidates.append(make_pair(
                            left, right, 1, "same_identity_temporal",
                            query_index, item["video"], expression_text(item["entry"]),
                            target_ids))
                for left in values:
                    cross = [right for right in values
                             if right["frame"] == left["frame"] and
                             right["source"] != left["source"]]
                    if cross:
                        candidates.append(make_pair(
                            left, cross[0], 1, "main_reserve_same_identity",
                            query_index, item["video"], expression_text(item["entry"]),
                            target_ids))
            positives = [pair for pair in candidates if pair["label"]]
            # Negative pairs are drawn from the same frame first, which makes
            # them identity hard negatives rather than easy random examples.
            negative = []
            for rows in by_frame.values():
                relevant = [row for row in rows
                            if row["gt_id"] is not None]
                for left in relevant:
                    others = [row for row in relevant
                              if row["gt_id"] != left["gt_id"]]
                    for right in others[:args.max_negative_per_frame]:
                        negative.append(make_pair(
                            left, right, 0, "same_frame_different_identity",
                            query_index, item["video"], expression_text(item["entry"]),
                            target_ids))
                    missing = [row for row in rows if row["gt_id"] is None]
                    if missing:
                        negative.append(make_pair(
                            left, missing[0], 0, "no_match_fragment",
                            query_index, item["video"], expression_text(item["entry"]),
                            target_ids))
            if len(positives) > args.max_positive:
                positives = rng.sample(positives, args.max_positive)
            if len(negative) > args.max_negative:
                negative = rng.sample(negative, args.max_negative)
            for pair in positives + negative:
                handle.write(json.dumps(pair, sort_keys=True) + "\n")
                counts[pair["reason"]] += 1
                pair_count += 1
            query_count += 1
    summary = {
        "format": "locatemot-l20-pair-supervision-v1",
        "source": "train annotations and frozen dual-bank provenance only",
        "seed": args.seed, "query_count": query_count,
        "pair_count": pair_count, "counts": dict(counts),
        "bank_root": str((ROOT / args.bank_root).resolve()),
        "max_positive": args.max_positive, "max_negative": args.max_negative,
        "split": "train", "query_source": "load_items()[train]",
        "gt_source": "train_sidecar_labels", "official_gt_used": False,
    }
    used_keys = sorted({(item["bank_dataset"], item["video"])
                        for item in all_items[:query_count]})
    source_sha256 = {
        "split_manifest": sha256_file(
            ROOT / "outputs/l16/data/protocol/split_manifest.json"),
        "metadata": {
            str(path): sha256_file(path) for path in (
                ROOT / "outputs/l11/data/rmot_kitti/expressions.json",
                ROOT / "outputs/l16/data/kitti_missing/records/expressions.json",
                ROOT / "outputs/l16/data/protocol/refer_dance_expressions.json",
            ) if path.exists()
        },
        "banks": {},
    }
    for dataset, video in used_keys:
        path = bank_path((ROOT / args.bank_root).resolve(), dataset, video)
        source_sha256["banks"][str(path)] = sha256_file(path)
        labels = path.with_suffix(".labels.json")
        if not labels.exists():
            raise ValueError(f"missing train sidecar labels: {labels}")
        source_sha256["banks"][str(labels)] = sha256_file(labels)
    manifest = {
        "format": "locatemot-l20-pair-supervision-v2",
        "valid": True, "split": "train",
        "query_source": "load_items()[train]",
        "gt_source": "train_sidecar_labels", "official_gt_used": False,
        "videos": sorted({(item["domain"], item["video"])
                           for item in all_items[:query_count]}),
        "queries": [{"query_index": index, "domain": item["domain"],
                     "video": item["video"],
                     "expression": expression_text(item["entry"])}
                    for index, item in enumerate(all_items[:query_count])],
        "source_sha256": source_sha256,
        "checkpoint_sha256": (sha256_file((ROOT / args.checkpoint).resolve())
                              if args.checkpoint else None),
        "checkpoint_role": "not_used_for_label_construction",
        **summary,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    output.with_suffix(output.suffix + ".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-root", default="outputs/l19/dual_banks_features")
    parser.add_argument("--out", default="outputs/l20/protocol/pair_supervision.jsonl")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--max-positive", type=int, default=128)
    parser.add_argument("--max-negative", type=int, default=256)
    parser.add_argument("--max-negative-per-frame", type=int, default=2)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
