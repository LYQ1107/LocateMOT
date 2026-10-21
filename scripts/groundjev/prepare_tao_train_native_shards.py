#!/usr/bin/env python3
"""Create resumable, label-free shards of the registered TAO TRAIN scope.

The shards are runtime manifests for the native GroundAnything exporter.  They
contain only image/query inputs; TAO annotations remain referenced for the
offline join and are never copied into a model-input manifest.  Each shard is
independent because native perception has no temporal state.  The shard
receipts make it possible to merge all completed streams and prove exact
coverage before the native OVMOT training gate is opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slice(values: list, start: int, end: int) -> list:
    return values[start:end]


def build(args: argparse.Namespace) -> dict:
    source = args.manifest.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    if args.shards < 1:
        raise ValueError("--shards must be positive")
    payload = json.loads(source.read_text())
    if payload.get("dataset") != "tao_amodal" or payload.get("task") != "ovmot":
        raise ValueError("source manifest is not the registered TAO OVMOT scope")
    if payload.get("training_on_this_scope") is not True:
        raise ValueError("source manifest is not explicitly registered as TRAIN scope")
    if payload.get("official_test_labels_read") is not False or payload.get("runtime_GT_fields") is not False:
        raise ValueError("source manifest does not prove a GT-free runtime input")
    images = list(payload.get("images", []))
    if not images:
        raise ValueError("source manifest has no images")
    fields = ("image_ids", "video_ids", "frame_numbers")
    if any(len(payload.get(field, [])) != len(images) for field in fields):
        raise ValueError("source manifest image metadata lengths do not agree")
    if len(set(images)) != len(images):
        raise ValueError("source manifest contains duplicate image paths")
    parent_sha = sha256(source)
    out.mkdir(parents=True)
    shard_receipts = []
    for shard_index in range(args.shards):
        start = (shard_index * len(images)) // args.shards
        end = ((shard_index + 1) * len(images)) // args.shards
        if start == end:
            raise ValueError(f"--shards={args.shards} creates an empty shard {shard_index}")
        shard_dir = out / f"shard_{shard_index:02d}_of_{args.shards:02d}"
        shard_dir.mkdir()
        shard = {
            "dataset": payload["dataset"],
            "task": payload["task"],
            "scope": f"{payload.get('scope', 'tao_train_native_export')}/shard_{shard_index:02d}_of_{args.shards:02d}",
            "images": _slice(images, start, end),
            "image_ids": _slice(payload["image_ids"], start, end),
            "video_ids": _slice(payload["video_ids"], start, end),
            "frame_numbers": _slice(payload["frame_numbers"], start, end),
            "vocabulary": list(payload["vocabulary"]),
            "vocabulary_path": payload.get("vocabulary_path"),
            "vocabulary_policy": payload.get("vocabulary_policy"),
            "training_on_this_scope": True,
            "official_test_labels_read": False,
            "runtime_GT_fields": False,
            "model_may_read_training_labels": False,
            "annotation_path_for_offline_join_only": payload.get("annotation_path_for_offline_join_only"),
            "annotation_sha256_for_offline_join_only": payload.get("annotation_sha256_for_offline_join_only"),
            "image_root_policy": payload.get("image_root_policy", []),
            "physical_video_fps_verified": payload.get("physical_video_fps_verified", False),
            "native_per_box_score": payload.get("native_per_box_score", "absent"),
            "native_ovmot_training_rows": False,
            "optimizer_updates": 0,
            "gpu_launched": False,
            "shard": {
                "parent_manifest": str(source),
                "parent_manifest_sha256": parent_sha,
                "num_shards": args.shards,
                "shard_index": shard_index,
                "start_index": start,
                "end_index_exclusive": end,
                "coverage_contract": "contiguous disjoint slices of the parent manifest in sorted image order",
            },
        }
        manifest_path = shard_dir / "manifest.json"
        manifest_path.write_text(json.dumps(shard, indent=2, ensure_ascii=False) + "\n")
        receipt = {
            "status": "TAO_TRAIN_NATIVE_SHARD_SCOPE_REGISTERED_NOT_STREAMED",
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "parent_manifest": str(source),
            "parent_manifest_sha256": parent_sha,
            "shard_index": shard_index,
            "num_shards": args.shards,
            "start_index": start,
            "end_index_exclusive": end,
            "frames": end - start,
            "videos": len(set(shard["video_ids"])),
            "native_candidate_export_started": False,
            "native_ovmot_training_rows": False,
            "required_stream_command": "tools/groundjev/stream_tao_native_candidates.py --allow-training-scope",
        }
        (shard_dir / "scope_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        shard_receipts.append(receipt)
    summary = {
        "status": "TAO_TRAIN_NATIVE_SHARDS_REGISTERED_NOT_STREAMED",
        "parent_manifest": str(source),
        "parent_manifest_sha256": parent_sha,
        "num_shards": args.shards,
        "parent_frames": len(images),
        "parent_videos": len(set(payload["video_ids"])),
        "shards": shard_receipts,
        "coverage": {
            "disjoint_contiguous_slices": True,
            "covered_frames": sum(item["frames"] for item in shard_receipts),
            "missing_frames": 0,
            "duplicate_frames": 0,
        },
        "native_candidate_export_started": False,
        "native_ovmot_training_rows": False,
        "next_step": "Stream every shard, verify each receipt, concatenate in shard order, then run prepare_native_ovmot_train_join.py on the complete stream.",
    }
    (out / "shards_receipt.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=8)
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
