#!/usr/bin/env python3
"""Register all TAO TRAIN images for a native OVMOT candidate export.

This manifest contains image paths and the fixed 1,203-category query
vocabulary only.  It does not put annotation boxes or category labels in the
runtime input.  The labels remain available to the separate offline join
step, which is required before a manifest can pass the native OVMOT training
gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RUNTIME_ROOT = Path("/data2/user/LocateMOT_GroundJEV")
ANNOTATION = Path("/data1/user/TAO-Amodal/annotations/train.json")
VOCABULARY = Path("/data2/user/LocateMOT/outputs/unified/u2/protocol/lvis_v1_vocabulary.json")
FRAME_ROOTS = (
    Path("/data2/user/LocateMOT/data/TAO-Amodal/frames"),
    Path("/data1/user/TAO-Amodal/frames"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_image(file_name: str) -> Path:
    if not file_name.startswith("train/"):
        raise ValueError(f"TAO TRAIN manifest contains a non-train path: {file_name}")
    candidates = [root / file_name for root in FRAME_ROOTS]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(str(candidates[0]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, default=ANNOTATION)
    parser.add_argument("--vocabulary", type=Path, default=VOCABULARY)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    annotation = json.loads(args.annotation.read_text())
    vocabulary_payload = json.loads(args.vocabulary.read_text())
    vocabulary = [str(item["name"]).replace("_", " ") for item in vocabulary_payload["categories"]]
    if len(vocabulary) != 1203:
        raise ValueError(f"expected the registered 1203-class vocabulary, got {len(vocabulary)}")
    images = sorted(annotation["images"], key=lambda item: (int(item["video_id"]), int(item["frame_index"]), int(item["id"])))
    if len(images) != 18274:
        raise ValueError(f"unexpected TAO TRAIN image count: {len(images)}")
    resolved = [resolve_image(str(item["file_name"])) for item in images]
    if len({str(path) for path in resolved}) != len(resolved):
        raise ValueError("TAO TRAIN image paths are not unique")
    out.mkdir(parents=True)
    manifest = {
        "dataset": "tao_amodal",
        "task": "ovmot",
        "scope": "tao_train_all_images_native_candidate_export",
        "images": [str(path) for path in resolved],
        "image_ids": [int(item["id"]) for item in images],
        "video_ids": [int(item["video_id"]) for item in images],
        "frame_numbers": [int(item["frame_index"]) + 1 for item in images],
        "vocabulary": vocabulary,
        "vocabulary_path": str(args.vocabulary.resolve()),
        "vocabulary_policy": "all 1203 fixed LVIS-v1 names on every image; no annotation-driven query selection",
        "training_on_this_scope": True,
        "official_test_labels_read": False,
        "runtime_GT_fields": False,
        "model_may_read_training_labels": False,
        "annotation_path_for_offline_join_only": str(args.annotation.resolve()),
        "annotation_sha256_for_offline_join_only": sha256(args.annotation),
        "image_root_policy": [str(root) for root in FRAME_ROOTS],
        "physical_video_fps_verified": False,
        "native_per_box_score": "not required; absent scores remain explicit",
        "native_ovmot_training_rows": False,
        "optimizer_updates": 0,
        "gpu_launched": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    receipt = {
        "status": "TAO_TRAIN_NATIVE_OVMOT_SCOPE_REGISTERED_NOT_STREAMED",
        "manifest": str(out / "manifest.json"),
        "manifest_sha256": sha256(out / "manifest.json"),
        "frames": len(images),
        "videos": len({int(item["video_id"]) for item in images}),
        "resolved_images": len(resolved),
        "annotation_labels_emitted_to_runtime": False,
        "official_test_labels_read": False,
        "native_candidate_export_started": False,
        "native_ovmot_training_rows": False,
        "stream_entrypoint": "tools/groundjev/stream_tao_native_candidates.py --allow-training-scope",
        "next_offline_step": "tools/groundjev/prepare_native_ovmot_train_join.py",
    }
    (out / "scope_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
