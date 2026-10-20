#!/usr/bin/env python3
"""Register a TAO validation video that is outside the G2 Clean train rows.

The runtime evaluator lives in the GroundJEV checkout, while this repository
stores the reproducible scope-selection rule and the audit.  The generated
manifest contains only image paths and fixed vocabulary; evaluation labels are
read by the official scorer after inference.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


RUNTIME_ROOT = Path("/data2/user/LocateMOT_GroundJEV")
ANNOTATION = Path("/data2/user/TAO-Amodal/annotations/validation.json")
TRAIN_MANIFEST = Path(
    "/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/"
    "g2_clean_train_manifest.json"
)
VOCABULARY = Path("/data2/user/LocateMOT/outputs/unified/u2/protocol/lvis_v1_vocabulary.json")
OUT_DIR = RUNTIME_ROOT / "outputs/groundjev/multitask_pilot_data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", type=int, default=105)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "ovmot_tao_independent105_full.json")
    parser.add_argument(
        "--audit-out",
        type=Path,
        default=OUT_DIR / "ovmot_tao_independent105_audit.json",
    )
    args = parser.parse_args()

    annotation = json.loads(ANNOTATION.read_text())
    videos = {int(video["id"]): video for video in annotation["videos"]}
    if args.video_id not in videos:
        raise ValueError(f"unknown TAO validation video id: {args.video_id}")
    video = videos[args.video_id]
    images = sorted(
        (image for image in annotation["images"] if image["video_id"] == args.video_id),
        key=lambda image: image["frame_index"],
    )
    if not images:
        raise ValueError(f"TAO validation video {args.video_id} has no images")

    category_by_id = {int(category["id"]): category for category in annotation["categories"]}
    category_counts = collections.Counter(
        int(item["category_id"])
        for item in annotation["annotations"]
        if item["video_id"] == args.video_id
    )
    if not category_counts:
        raise ValueError(f"TAO validation video {args.video_id} has no annotations")
    if not any(category_by_id[cid].get("frequency") == "r" for cid in category_counts):
        raise ValueError(
            f"TAO validation video {args.video_id} has no rare (Novel) category; "
            "choose a video with both Base and Novel TETA support"
        )

    vocabulary_payload = json.loads(VOCABULARY.read_text())
    vocabulary = [str(category["name"]).replace("_", " ") for category in vocabulary_payload["categories"]]
    if len(vocabulary) != 1203:
        raise ValueError(f"expected 1203 fixed LVIS names, got {len(vocabulary)}")

    image_paths = [Path("/data2/user/LocateMOT/data/TAO-Amodal/frames") / image["file_name"] for image in images]
    missing = [str(path) for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])

    train_manifest = json.loads(TRAIN_MANIFEST.read_text())
    # G2 Clean uses TAO train images.  Check the exact validation relative path
    # against the registered train coverage and retain an explicit audit even
    # when a future corpus changes its coverage representation.
    train_rows = Path(train_manifest["rows_path"])
    validation_rel = {image["file_name"] for image in images}
    exact_overlap: set[str] = set()
    if train_rows.is_file():
        for line in train_rows.open():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            path = str(row.get("image", row.get("path", "")))
            for prefix in ("/data2/user/TAO-Amodal/frames/", "/data2/user/LocateMOT/data/TAO-Amodal/frames/"):
                if path.startswith(prefix):
                    relative = path[len(prefix) :]
                    if relative in validation_rel:
                        exact_overlap.add(relative)
                    break
    if exact_overlap:
        raise RuntimeError(f"validation scope overlaps G2 Clean TRAIN images: {sorted(exact_overlap)[:3]}")

    manifest = {
        "dataset": "tao_amodal",
        "sequence": video["name"],
        "video_id": args.video_id,
        "task": "ovmot",
        "annotation_path_for_evaluation_only": str(ANNOTATION),
        "vocabulary_path": str(VOCABULARY),
        "vocabulary": vocabulary,
        "vocabulary_policy": "all 1203 fixed LVIS-v1 names; no GT-driven class selection",
        "images": [str(path) for path in image_paths],
        "image_ids": [int(image["id"]) for image in images],
        "frame_numbers": [int(image["frame_index"]) + 1 for image in images],
        "timestamps": list(range(len(images))),
        "fps": 1,
        "timestamp_policy": "annotation-step units; physical source FPS unavailable",
        "physical_video_fps_verified": False,
        "scope": "independent_full_validation_video_annotation_frames",
        "GT_policy": "all original modal xywh boxes; no clipping, visibility, or category exclusions",
        "base_novel_policy": "official TETA frequency c/f=Base, r=Novel; retain all original categories",
        "novel_evaluation_GT_used_for_training": False,
        "training_on_this_scope": False,
        "official_test_labels_read": False,
        "runtime_GT_fields": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    audit = {
        "status": "INDEPENDENT_TAO_VALIDATION_SCOPE_REGISTERED",
        "video_id": args.video_id,
        "sequence": video["name"],
        "frames": len(images),
        "image_ids": [int(image["id"]) for image in images],
        "annotation_sha256": sha256(ANNOTATION),
        "g2_train_manifest_sha256": sha256(TRAIN_MANIFEST),
        "g2_train_rows_path": str(train_rows),
        "exact_image_overlap_count": len(exact_overlap),
        "exact_image_overlap": sorted(exact_overlap),
        "category_counts": {
            str(cid): {
                "name": category_by_id[cid]["name"],
                "frequency": category_by_id[cid]["frequency"],
                "annotations": count,
            }
            for cid, count in sorted(category_counts.items())
        },
        "base_novel_support": {
            "base_categories": sorted(
                category_by_id[cid]["name"]
                for cid in category_counts
                if category_by_id[cid].get("frequency") in {"c", "f"}
            ),
            "novel_categories": sorted(
                category_by_id[cid]["name"]
                for cid in category_counts
                if category_by_id[cid].get("frequency") == "r"
            ),
        },
        "manifest": str(args.out),
    }
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(audit, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
