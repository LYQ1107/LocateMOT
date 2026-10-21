#!/usr/bin/env python3
"""Register a small, deterministic TAO TRAIN native-export scope.

The complete TAO TRAIN manifest remains immutable.  This tool selects one
annotated frame from evenly spaced videos so a real native OVMOT stream can be
run for joint-training development without pretending that a partial cache is
the full TRAIN set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


DEFAULT_PARENT = Path(
    "/data2/user/LocateMOT_GroundJEV/outputs/groundjev/training_data/"
    "tao_native_ovmot_train_scope_v1/manifest.json"
)
DEFAULT_ANNOTATION = Path("/data1/user/TAO-Amodal/annotations/train.json")
DEFAULT_OUT = Path(
    "/data2/user/LocateMOT_GroundJEV/outputs/groundjev/training_data/"
    "tao_native_ovmot_train_subset64_v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_key(path: str) -> str:
    value = Path(path).as_posix()
    marker = "/frames/train/"
    if marker in value:
        return "train/" + value.split(marker, 1)[1]
    if value.startswith("train/"):
        return value
    raise ValueError(f"not a TAO TRAIN frame path: {path}")


def evenly_spaced(values: list[int], count: int) -> list[int]:
    if count <= 0 or len(values) < count:
        raise ValueError(f"requested {count} videos but only {len(values)} eligible videos exist")
    if count == 1:
        return [values[len(values) // 2]]
    return [values[round(index * (len(values) - 1) / (count - 1))] for index in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--videos", type=int, default=64)
    parser.add_argument("--shards", type=int, default=4)
    args = parser.parse_args()
    if args.videos < 1 or args.shards < 1 or args.videos % args.shards:
        parser.error("--videos must be positive and divisible by --shards")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite {args.out}")
    parent = json.loads(args.parent.read_text())
    annotation = json.loads(args.annotation.read_text())
    if parent.get("dataset") != "tao_amodal" or parent.get("task") != "ovmot":
        raise ValueError("unexpected TAO parent manifest")
    if parent.get("training_on_this_scope") is not True:
        raise ValueError("parent manifest is not explicitly registered TRAIN scope")
    ann_by_key = {str(item["file_name"]): item for item in annotation["images"]}
    category_by_id = {int(item["id"]): item for item in annotation["categories"]}
    eligible_categories = {
        category_id for category_id, item in category_by_id.items()
        if item.get("frequency") in {"c", "f"}
    }
    eligible_image_ids = defaultdict(list)
    annotations_by_image = defaultdict(list)
    for item in annotation["annotations"]:
        image_id = int(item["image_id"])
        if int(item["category_id"]) in eligible_categories:
            annotations_by_image[image_id].append(item)
    for item in annotation["images"]:
        if annotations_by_image.get(int(item["id"])):
            eligible_image_ids[int(item["video_id"])].append(item)
    parent_images = parent["images"]
    parent_frames = parent.get("frame_numbers", list(range(len(parent_images))))
    parent_by_key = {train_key(path): index for index, path in enumerate(parent_images)}
    videos = []
    for video_id, items in sorted(eligible_image_ids.items()):
        candidates = [item for item in sorted(items, key=lambda x: (int(x.get("frame_index", 0)), int(x["id"])))
                      if str(item["file_name"]) in parent_by_key]
        if candidates:
            videos.append((video_id, candidates[0]))
    selected_video_ids = evenly_spaced([video_id for video_id, _ in videos], args.videos)
    selected_by_video = {video_id: item for video_id, item in videos}
    selected = []
    for video_id in selected_video_ids:
        image = selected_by_video[video_id]
        index = parent_by_key[str(image["file_name"])]
        selected.append({
            "video_id": video_id,
            "image_id": int(image["id"]),
            "frame_index": int(image.get("frame_index", 0)),
            "parent_index": index,
            "path": parent_images[index],
            "frame_number": parent_frames[index],
            "eligible_annotation_count": len(annotations_by_image[int(image["id"])]),
        })
    selected.sort(key=lambda item: item["parent_index"])
    args.out.mkdir(parents=True)
    subset = dict(parent)
    subset.update({
        "scope": "tao_train_registered_native_subset64",
        "scope_contract": {
            "registered_train_subset": True,
            "full_train_image_scope": False,
            "selection": "one eligible annotated frame from 64 evenly spaced TAO TRAIN videos",
            "official_test_labels_read": False,
            "runtime_GT_fields": False,
        },
        "parent_manifest": str(args.parent),
        "parent_manifest_sha256": sha256(args.parent),
        "sample_strategy": "evenly_spaced_videos_one_frame_each_v1",
        "selected_videos": [item["video_id"] for item in selected],
        "selected_frames": selected,
        "image_ids": [item["image_id"] for item in selected],
        "images": [item["path"] for item in selected],
        "frame_numbers": [item["frame_number"] for item in selected],
        "training_on_this_scope": True,
        "official_test_labels_read": False,
        "runtime_GT_fields": False,
    })
    subset_path = args.out / "subset_manifest.json"
    subset_path.write_text(json.dumps(subset, indent=2, ensure_ascii=False) + "\n")
    subset_sha = sha256(subset_path)
    per_shard = args.videos // args.shards
    for shard_index in range(args.shards):
        start = shard_index * per_shard
        end = (shard_index + 1) * per_shard
        shard_dir = args.out / f"shard_{shard_index:02d}_of_{args.shards:02d}"
        shard_dir.mkdir()
        shard = dict(subset)
        shard.update({
            "scope": "tao_train_registered_native_subset64_shard",
            "images": subset["images"][start:end],
            "image_ids": subset["image_ids"][start:end],
            "frame_numbers": subset["frame_numbers"][start:end],
            "selected_videos": subset["selected_videos"][start:end],
            "selected_frames": subset["selected_frames"][start:end],
            "shard": {
                "index": shard_index,
                "count": args.shards,
                "start_index": start,
                "end_index_exclusive": end,
                "parent_manifest_sha256": subset_sha,
            },
        })
        manifest_path = shard_dir / "manifest.json"
        manifest_path.write_text(json.dumps(shard, indent=2, ensure_ascii=False) + "\n")
        (shard_dir / "scope_receipt.json").write_text(json.dumps({
            "status": "REGISTERED_NATIVE_TAO_TRAIN_SUBSET_SCOPE",
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "parent_manifest": str(subset_path),
            "parent_manifest_sha256": subset_sha,
            "shard_index": shard_index,
            "frames": end - start,
            "videos": end - start,
            "native_candidate_export_started": False,
            "native_ovmot_training_rows": False,
        }, indent=2, ensure_ascii=False) + "\n")
    receipt = {
        "status": "REGISTERED_NATIVE_TAO_TRAIN_SUBSET_SCOPE",
        "parent_manifest": str(args.parent),
        "parent_manifest_sha256": sha256(args.parent),
        "subset_manifest": str(subset_path),
        "subset_manifest_sha256": subset_sha,
        "num_shards": args.shards,
        "parent_frames": len(parent_images),
        "subset_frames": len(selected),
        "parent_videos": len(eligible_image_ids),
        "subset_videos": len(selected),
        "registered_train_subset": True,
        "full_train_image_scope": False,
        "official_test_labels_read": False,
        "native_candidate_export_started": False,
        "native_ovmot_training_rows": False,
    }
    (args.out / "shards_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
