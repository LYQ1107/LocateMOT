#!/usr/bin/env python3
"""Freeze ordinary KITTI tracking using legal V2 videos and original IDs."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from locatemot.unified.data.legal_scope import LEGAL_SPLITS, assert_legal_path, assert_legal_video
from tools.unified.u1_formal_generic_gate import VOCABULARY

LABEL_ROOT = Path("/data2/user/reference_envs/kitti_tracking_labels_20261001/training/label_02")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    from PIL import Image

    sequences = []
    for split in ("fit", "calibration", "validation"):
        for raw_video in LEGAL_SPLITS["refer_kitti_v2"][split]:
            video = assert_legal_video(raw_video, dataset="refer_kitti_v2")
            image_dir = assert_legal_path(ROOT / "data/kitti_tracking/training/image_02" / video)
            images = sorted(image_dir.glob("*.png"))
            if not images or [int(path.stem) for path in images] != list(range(len(images))):
                raise RuntimeError(f"Incomplete original KITTI frame coverage: {video}")
            gt_path = LABEL_ROOT / f"{video}.txt"
            # Check the video before opening its original label; filenames
            # ending .txt are not matched by the generic path-component guard.
            assert_legal_video(video, dataset="refer_kitti_v2")
            rows = [line.split() for line in gt_path.read_text().splitlines()]
            if any(not 0 <= int(row[0]) < len(images) for row in rows):
                raise RuntimeError(f"KITTI original GT references a missing frame: {video}")
            with Image.open(images[0]) as image:
                width, height = image.size
            frames = [path.name for path in images]
            sequences.append({"sequence": video, "split": split, "image_dir": str(image_dir),
                              "gt_path": str(gt_path), "gt_sha256": digest(gt_path),
                              "gt_format": "original_kitti_tracking", "fps": 10,
                              "width": width, "height": height, "length": len(frames),
                              "train_frames": frames if split == "fit" else [],
                              "calibration_frames": frames if split == "calibration" else [],
                              "validation_frames": frames if split == "validation" else [],
                              "development_frames": frames if split != "fit" else [],
                              "source_annotation_rows": len(rows),
                              "source_category_counts": dict(sorted(Counter(row[2] for row in rows).items()))})
    protocol = {"format": "locatemot-u2-kitti-legal-tracking-v1", "dataset": "kitti_legal_development",
                "source": "https://www.cvlibs.net/datasets/kitti/eval_tracking.php",
                "original_label_release": "data_tracking_label_2.zip (previously acquired for U1R2)",
                "license": "KITTI dataset terms", "legal_scope_version": "v2",
                "split_source": "locatemot/unified/data/legal_scope.py",
                "split_source_sha256": digest(ROOT / "locatemot/unified/data/legal_scope.py"),
                "training_split": "V2 fit only; do not union V1 fit with V2 development videos",
                "development_split": "V2 calibration and validation, reported separately and combined",
                "vocabulary": VOCABULARY, "candidate_top_k": 150, "max_per_img": 900,
                "classes_to_evaluate": ["car", "pedestrian"],
                "gt_policy": "Original KITTI tracking IDs/xyxy and unchanged TrackEval Kitti2DBox preprocessing; retain distractor/DontCare/occlusion/truncation metadata",
                "sequences": sequences,
                "train_images": sum(len(seq["train_frames"]) for seq in sequences),
                "calibration_images": sum(len(seq["calibration_frames"]) for seq in sequences),
                "validation_images": sum(len(seq["validation_frames"]) for seq in sequences),
                "development_images": sum(len(seq["development_frames"]) for seq in sequences),
                "official_test_labels_read": False,
                "refer_kitti_official_eval_labels_read": False}
    path = ROOT / "outputs/unified/u2/protocol/kitti_legal_tracking.json"
    text = json.dumps(protocol, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != text:
        raise RuntimeError("An existing frozen ordinary KITTI protocol differs")
    path.write_text(text)
    # The U1 generic references remain immutable. Ordinary tracking needs a
    # separate full original-GT audit, rather than inheriting Refer GT coverage.
    audit_sequences = []
    for seq in sequences:
        if not seq["development_frames"]:
            continue
        rows = [line.split() for line in Path(seq["gt_path"]).read_text().splitlines()]
        count = sum(row[2].lower() in ("car", "pedestrian")
                    and int(float(row[4])) <= 2 and int(float(row[3])) <= 0 for row in rows)
        audit_sequences.append({**seq, "validation_frames": seq["development_frames"],
                                "positive_gt_boxes": count})
    audit = {"format": "locatemot-u2-kitti-original-gt-portability-v1", "dataset": protocol["dataset"],
             "source": protocol["source"], "version": "original KITTI labels; legal V2 scope",
             "tracking_protocol": str(path), "tracking_protocol_sha256": digest(path),
             "vocabulary": VOCABULARY, "max_per_img": 900, "candidate_top_k": 150,
             "measured_top_k": [150, 300], "iou_threshold": .5,
             "gt_policy": {"positive": "Car/Pedestrian; the unchanged official reader's int-cast occlusion <= 2 and truncation <= 0",
                           "boxes": "original xyxy; no shift or clipping", "frame_index": "zero-based source frame",
                           "source": "TrackEval Kitti2DBox 12c8791b303e0a0b50f753af204249e622d0281a",
                           "gt_height_cutoff": None},
             "sequences": audit_sequences, "validation_images": protocol["development_images"],
             "positive_gt_boxes": sum(seq["positive_gt_boxes"] for seq in audit_sequences),
             "official_test_labels_read": False, "refer_kitti_official_eval_labels_read": False}
    audit_path = path.parent / "kitti_ordinary_portability.json"
    audit_text = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    if audit_path.exists() and audit_path.read_text() != audit_text:
        raise RuntimeError("The frozen original-KITTI portability scope differs")
    audit_path.write_text(audit_text)
    print(json.dumps({"protocol": str(path), "sha256": digest(path), "videos": len(sequences),
                      **{key: protocol[key] for key in ("train_images", "development_images", "calibration_images", "validation_images")},
                      "ordinary_portability_protocol": str(audit_path), "ordinary_portability_gt_units": audit["positive_gt_boxes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
