#!/usr/bin/env python3
"""Pre-register a fit-video-disjoint development split for L82 Phase D."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
ELIGIBLE = ROOT / "outputs/l82/data/eligible_group_keys.json"
MATRIX = ROOT / "outputs/l82/data/frame_query_groups.jsonl"
OUT_DEFAULT = ROOT / "outputs/l82/protocol/fit_video_train_dev_split.json"
SEED = 20260829


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists():
        raise FileExistsError(f"refusing existing L82 split: {out}")
    eligible = json.loads(ELIGIBLE.read_text())
    keys = [str(key) for key in eligible["keys"]]
    if len(keys) != int(eligible["count"]) or len(set(keys)) != len(keys):
        raise AssertionError("eligible group key contract drift")
    groups_by_video: dict[tuple[str, str], list[str]] = defaultdict(list)
    for key in keys:
        parts = key.split("|")
        if len(parts) != 3:
            raise AssertionError(f"invalid group key {key}")
        groups_by_video[(parts[0], parts[1])].append(key)
    assignments: dict[tuple[str, str], str] = {}
    ranking: dict[str, list[dict[str, object]]] = {}
    for dataset in sorted({dataset for dataset, _ in groups_by_video}):
        videos = sorted({video for item_dataset, video in groups_by_video if item_dataset == dataset})
        ordered = sorted(videos, key=lambda video: hashlib.sha256(
            f"{SEED}|{dataset}|{video}".encode()).hexdigest())
        dev_count = max(1, int((len(ordered) + 4) // 5))
        dev_videos = set(ordered[-dev_count:])
        if dev_videos == set(ordered):
            raise AssertionError("video-disjoint split would have empty train")
        ranking[dataset] = [{"video": video, "sha256_sort_key": hashlib.sha256(
            f"{SEED}|{dataset}|{video}".encode()).hexdigest(),
            "partition": "dev" if video in dev_videos else "train",
            "eligible_group_count": len(groups_by_video[(dataset, video)])}
            for video in ordered]
        for video in videos:
            assignments[(dataset, video)] = "dev" if video in dev_videos else "train"
    train_keys = [key for key in keys if assignments[tuple(key.split("|")[:2])] == "train"]
    dev_keys = [key for key in keys if assignments[tuple(key.split("|")[:2])] == "dev"]
    if not train_keys or not dev_keys:
        raise AssertionError("empty train/dev group partition")
    if {key.split("|")[0] for key in train_keys} != {"refer_kitti_v1", "refer_kitti_v2"}:
        raise AssertionError("train partition lacks a domain")
    if {key.split("|")[0] for key in dev_keys} != {"refer_kitti_v1", "refer_kitti_v2"}:
        raise AssertionError("dev partition lacks a domain")
    payload = {
        "format": "locatemot-l82-video-disjoint-fit-dev-split-v1",
        "status": "complete", "stage": "phase_d_preregistered_before_scores",
        "command": " ".join([sys.executable] + sys.argv), "cwd": str(ROOT),
        "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745", "seed": SEED,
        "selection": "sha256 sort of (seed,dataset,video), last ceil(20%) per dataset as dev",
        "group_source": {"path": str(ELIGIBLE.resolve()), "sha256": sha256_file(ELIGIBLE)},
        "matrix_source": {"path": str(MATRIX.resolve()), "sha256": sha256_file(MATRIX)},
        "video_assignment": ranking,
        "train_videos": sorted([f"{dataset}|{video}" for (dataset, video), part in assignments.items() if part == "train"]),
        "dev_videos": sorted([f"{dataset}|{video}" for (dataset, video), part in assignments.items() if part == "dev"]),
        "train_group_count": len(train_keys), "dev_group_count": len(dev_keys),
        "train_group_keys": train_keys, "dev_group_keys": dev_keys,
        "same_video_partition_rule": "all eligible groups for each (dataset,video) share one partition",
        "labels_used_for_split": False, "validation_labels_used": False,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "training_run": False,
        "hota_trackeval_run": False, "candidate_deletion": False,
        "candidate_truncation": False, "token_span_region_alignment": "UNALIGNED",
        "static_motion_alignment": "UNALIGNED", "gpu_world_size": 0,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "complete", "train_groups": len(train_keys),
                      "dev_groups": len(dev_keys), "train_videos": payload["train_videos"],
                      "dev_videos": payload["dev_videos"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
