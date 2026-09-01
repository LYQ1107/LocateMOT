"""Build and freeze the deterministic Stage L19 KITTI fast-eval manifest.

Selection uses only the train-val annotations and frozen candidate-bank
coverage metadata.  It never reads a checkpoint or a score cache.  The
manifest is intentionally split into calibration and screening queries so a
threshold selected on one part is not reported on the same part.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from locatemot.models.l16_track_selector import FAMILY_NAMES, expression_family_vector
from tools.eval_l18_carr import metadata, trainval_queries
from tools.train_l18_carr import BankStore, l19_frame_targets, l19_track_membership_index

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def classify_query(bank: dict, entry: dict) -> dict:
    """Summarize coverage states without consulting model scores."""
    tensors = bank["tensors"]
    if "l19_track_membership" not in bank:
        bank["l19_track_membership"] = l19_track_membership_index(bank)
    frame_ptr = tensors["frame_ptr"].tolist()
    states = Counter()
    target_frames = 0
    main_positive = 0
    reserve_positive = 0
    present_uncovered = 0
    for frame_index, frame_id in enumerate(tensors["frame_ids"].tolist()):
        begin, end = int(frame_ptr[frame_index]), int(frame_ptr[frame_index + 1])
        target = l19_frame_targets(
            bank, begin, end, entry, int(frame_id), bank["l19_track_membership"])
        state = int(target["state"])
        states[state] += 1
        if target["active"]:
            target_frames += 1
            main_positive += int(target["main_covered"])
            reserve_positive += int(target["reserve_covered"])
            present_uncovered += int(
                not target["main_covered"] and not target["reserve_covered"])
    # A primary label is used only for stratification; all state counts remain
    # in the manifest for auditing.
    # State 2 means the target is covered only by the reserve pool.  A reserve
    # duplicate can also be present when state 1 is MAIN_COVERED; counting all
    # such duplicates as reserve-positive would collapse the stratification.
    reserve_only = int(states.get(2, 0))
    main_only = int(states.get(1, 0))
    uncovered = int(states.get(3, 0))
    dominant_state = max(states, key=lambda key: (states[key], -key)) 
    if dominant_state == 0:
        primary = "ABSENT"
    elif dominant_state == 3 or uncovered:
        primary = "PRESENT_UNCOVERED"
    elif dominant_state == 2 or reserve_only:
        primary = "RESERVE_COVERED"
    elif main_only:
        primary = "MAIN_COVERED"
    else:
        primary = "ABSENT"
    text = str(entry.get("sentence", entry.get("expression", "")))
    family = expression_family_vector(text).tolist()
    family_flags = [name for name, value in zip(FAMILY_NAMES, family)
                    if float(value) > 0.5]
    return {
        "coverage_primary": primary,
        "coverage_states": {str(key): int(value)
                             for key, value in sorted(states.items())},
        "target_frames": int(target_frames),
        "video_frames": int(len(tensors["frame_ids"])),
        "main_positive_frames": int(main_positive),
        "reserve_positive_frames": int(reserve_positive),
        "reserve_only_frames": reserve_only,
        "present_uncovered_frames": int(present_uncovered),
        "reserve_positive": bool(reserve_only),
        "family_flags": family_flags,
    }


def choose_stratified(records: list[dict], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups = defaultdict(list)
    for record in records:
        family = tuple(record["family_flags"])
        groups[(record["coverage_primary"], family)].append(record)
    for values in groups.values():
        values.sort(key=lambda row: row["query_index"])
        rng.shuffle(values)

    # Coverage quotas deliberately reserve space for the DINO-covered cases.
    quotas = {
        "RESERVE_COVERED": max(1, int(round(count * 0.35))),
        "MAIN_COVERED": max(1, int(round(count * 0.30))),
        "PRESENT_UNCOVERED": max(1, int(round(count * 0.20))),
        "ABSENT": max(1, int(round(count * 0.15))),
    }
    selected = []
    selected_indices = set()
    for primary in ("RESERVE_COVERED", "MAIN_COVERED",
                    "PRESENT_UNCOVERED", "ABSENT"):
        candidates = [record for record in records
                      if record["coverage_primary"] == primary]
        candidates.sort(key=lambda row: (tuple(row["family_flags"]),
                                         row["query_index"]))
        rng.shuffle(candidates)
        for record in candidates[:quotas[primary]]:
            if len(selected) >= count:
                break
            selected.append(record)
            selected_indices.add(record["query_index"])

    # Round-robin over fine-grained coverage/family buckets fills the rest and
    # prevents a single long sequence or expression family from dominating.
    keys = sorted(groups)
    cursor = 0
    while len(selected) < count and keys:
        key = keys[cursor % len(keys)]
        cursor += 1
        values = groups[key]
        while values and values[0]["query_index"] in selected_indices:
            values.pop(0)
        if values:
            record = values.pop(0)
            selected.append(record)
            selected_indices.add(record["query_index"])
        if cursor >= len(keys) and all(
                not values or values[0]["query_index"] in selected_indices
                for values in groups.values()):
            break
    selected.sort(key=lambda row: row["query_index"])
    return selected


def split_records(records: list[dict], calibration_count: int,
                  seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed + 17)
    groups = defaultdict(list)
    for record in records:
        groups[record["coverage_primary"]].append(record)
    calibration, screening = [], []
    # Allocate within each coverage class so both splits retain the same
    # reserve/uncovered coverage regimes.
    for key in sorted(groups):
        values = list(groups[key])
        rng.shuffle(values)
        take = int(round(calibration_count * len(values) / max(1, len(records))))
        calibration.extend(values[:take])
        screening.extend(values[take:])
    # Correct rounding while preserving deterministic order.
    pool = [record for record in screening]
    while len(calibration) < calibration_count and pool:
        calibration.append(pool.pop(0))
    while len(calibration) > calibration_count:
        screening.append(calibration.pop())
    calibration.sort(key=lambda row: row["query_index"])
    screening = [record for record in records
                 if record["query_index"] not in
                 {row["query_index"] for row in calibration}]
    screening.sort(key=lambda row: row["query_index"])
    return calibration, screening


def distribution(records: list[dict]) -> dict:
    coverage = Counter(row["coverage_primary"] for row in records)
    states = Counter()
    families = Counter()
    videos = Counter(row["video"] for row in records)
    for row in records:
        states.update({int(key): int(value)
                       for key, value in row["coverage_states"].items()})
        families.update(row["family_flags"])
    return {
        "queries": len(records), "videos": dict(sorted(videos.items())),
        "coverage_primary": dict(sorted(coverage.items())),
        "coverage_states": {str(key): int(value)
                             for key, value in sorted(states.items())},
        "family_flags": dict(sorted(families.items())),
        "reserve_positive_queries": int(sum(
            row["reserve_positive"] for row in records)),
        "present_uncovered_queries": int(sum(
            row["present_uncovered_frames"] > 0 for row in records)),
        "absent_frame_queries": int(sum(
            int(row["coverage_states"].get("0", 0)) > 0 for row in records)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=
                        "outputs/l19/protocol/kitti_fast_eval_manifest.json")
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--count", type=int, default=160)
    parser.add_argument("--calibration-count", type=int, default=64)
    parser.add_argument("--bank-root", default="outputs/l19/dual_banks_features")
    args = parser.parse_args()
    if args.count < 16 or not 0 < args.calibration_count < args.count:
        raise ValueError("count/calibration-count are inconsistent")

    queries, _gt_root, _seqmap, _sequences, _protocol = trainval_queries(
        "trainval_kitti")
    lookup = metadata("kitti_v2")
    store = BankStore((ROOT / args.bank_root).resolve(), cache_size=1)
    records = []
    for query_index, (video, expression, spec) in enumerate(queries):
        entry = dict(lookup[(video, expression)])
        entry["spec"] = np.asarray(spec, np.float32).tolist()
        bank = store.get("kitti", video)
        record = {
            "query_index": int(query_index), "video": video,
            "expression": expression,
        }
        record.update(classify_query(bank, entry))
        records.append(record)
    selected = choose_stratified(records, min(args.count, len(records)), args.seed)
    calibration, screening = split_records(
        selected, min(args.calibration_count, len(selected) - 1), args.seed)
    calibration_ids = {row["query_index"] for row in calibration}
    screening_ids = {row["query_index"] for row in screening}
    for row in selected:
        row["split"] = "calibration" if row["query_index"] in calibration_ids \
            else "screening"
    selected.sort(key=lambda row: row["query_index"])
    query_payload = [{
        "query_index": row["query_index"], "video": row["video"],
        "expression": row["expression"], "split": row["split"],
    } for row in selected]
    canonical_queries = json.dumps(query_payload, sort_keys=True,
                                   separators=(",", ":")).encode()
    split_manifest = ROOT / "outputs/l16/data/protocol/split_manifest.json"
    payload = {
        "manifest_version": 1,
        "dataset": "trainval_kitti",
        "seed": int(args.seed), "selection_uses_model_scores": False,
        "selection_rule": {
            "count": len(selected),
            "calibration_count": len(calibration),
            "screening_count": len(screening),
            "coverage_quotas": {
                "RESERVE_COVERED": 0.35, "MAIN_COVERED": 0.30,
                "PRESENT_UNCOVERED": 0.20, "ABSENT": 0.15,
            },
            "stratify_by": ["coverage_primary", "family_flags"],
        },
        "source": {
            "metadata": [str(ROOT / "outputs/l11/data/rmot_kitti/expressions.json"),
                         str(ROOT / "outputs/l16/data/kitti_missing/records/expressions.json")],
            "split_manifest": str(split_manifest),
            "bank_root": str((ROOT / args.bank_root).resolve()),
            "split_manifest_sha256": sha256_bytes(split_manifest.read_bytes()),
            "generator": str(Path(__file__).resolve()),
            "generator_sha256": sha256_bytes(Path(__file__).read_bytes()),
        },
        "query_sha256": sha256_bytes(canonical_queries),
        "summary": {
            "all": distribution(selected),
            "calibration": distribution(calibration),
            "screening": distribution(screening),
        },
        "queries": selected,
    }
    destination = (ROOT / args.out).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    digest = sha256_bytes(destination.read_bytes())
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{digest}  {destination.name}\n")
    print(json.dumps({
        "manifest": str(destination), "sha256": digest,
        "summary": payload["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
