#!/usr/bin/env python3
"""Register every original BDD MOT train/val frame and official GT semantics."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from locatemot.unified.data.bdd_tracking import CLASSES, NAME_MAPPING, IGNORED_MAPPING, frame_objects

DIRECTORY = ROOT / "outputs/unified/u2/protocol"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    acquisition = json.loads((ROOT / "outputs/unified/u2/acquisition/bdd_labels_extraction.json").read_text())
    if acquisition["state"] != "COMPLETE" or not acquisition["crc_checked_during_extraction"]:
        raise RuntimeError("Original tracking annotations are not CRC-complete")
    view, sequences = ROOT / "data/bdd100k", []
    categories, ignores = {}, {}
    for split, expected in (("train", 1400), ("val", 200)):
        files = sorted((view / "labels/box_track_20" / split).glob("*.json"))
        if len(files) != expected:
            raise RuntimeError(f"Incomplete original BDD {split} videos")
        counts, ignore_counts = Counter(), Counter()
        for path in files:
            video = json.loads(path.read_text())
            frames = [frame["name"] for frame in sorted(video, key=lambda frame: frame["frameIndex"])]
            if len(set(frames)) != len(frames):
                raise RuntimeError(f"Repeated frame in source annotation: {path}")
            images = view / "images/track" / split / path.stem
            if split == "val" and sorted(image.name for image in images.glob("*.jpg")) != sorted(frames):
                raise RuntimeError(f"Official BDD validation image/annotation scope differs: {path.stem}")
            positives = 0
            for frame in video:
                objects, ignored = frame_objects(frame)
                positives += len(objects)
                counts.update(obj["category"] for obj in objects)
                ignore_counts.update(obj["reason"] for obj in ignored)
            sequences.append({"sequence": path.stem, "split": split, "image_dir": str(images),
                              "gt_path": str(path), "gt_sha256": digest(path), "gt_format": "scalabel_json",
                              "width": 1280, "height": 720, "fps": 5, "length": len(frames),
                              "train_frames": frames if split == "train" else [],
                              "validation_frames": frames if split == "val" else [],
                              "positive_gt_boxes": positives})
        categories[split], ignores[split] = dict(sorted(counts.items())), dict(sorted(ignore_counts.items()))
    sources = {}
    for name, url in {
        "bdd_mot_config.txt": "https://github.com/bdd100k/bdd100k/blob/master/bdd100k/configs/box_track.toml",
        "bdd_to_scalabel.py": "https://github.com/bdd100k/bdd100k/blob/master/bdd100k/label/to_scalabel.py",
        "scalabel_mot.txt": "https://github.com/scalabel/scalabel/blob/master/scalabel/eval/mot.py",
        "scalabel_transforms.py": "https://github.com/scalabel/scalabel/blob/master/scalabel/label/transforms.py",
    }.items():
        path = DIRECTORY / "sources" / name
        sources[name] = {"url": url, "path": str(path), "sha256": digest(path)}
    payload = {"format": "locatemot-u2-bdd-tracking-v1", "dataset": "bdd100k", "root": str(view),
               "version": "MOT 2020 raw Scalabel labels", "source": acquisition["source"],
               "mirror_revision": acquisition["revision"], "source_archive_sha256": acquisition["source_archive_sha256"],
               "license": "BDD100K original dataset terms; repackaged mirror", "split": "official train/val",
               "protocol_sources": sources, "vocabulary": " . ".join(CLASSES) + " .",
               "custom_entities": True, "max_per_img": 900, "measured_top_k": [150, 300],
               "candidate_top_k": 150, "iou_threshold": .5,
               "gt_policy": {"positive": "all eight original MOT categories excluding toolkit ignored/crowd objects",
                             "name_mapping": NAME_MAPPING, "ignored_mapping": IGNORED_MAPPING,
                             "visibility_cutoff": None,
                             "boxes": "official inclusive box2d to xywh (+1 width/height), then continuous xyxy; no clipping",
                             "recall_unit": "one positive physical GT object per frame"},
               "positive_category_counts": categories, "ignored_source_object_counts": ignores,
               "sequences": sequences, "train_images": sum(len(s["train_frames"]) for s in sequences),
               "validation_images": sum(len(s["validation_frames"]) for s in sequences),
               "train_image_coverage_required_by_data_manifest": True, "official_test_labels_read": False}
    path = DIRECTORY / "bdd100k.json"
    contents = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != contents:
        raise RuntimeError(f"An existing frozen BDD protocol differs: {path}")
    path.write_text(contents)
    return {"protocol": str(path), "sha256": digest(path), "train_images": payload["train_images"],
            "validation_images": payload["validation_images"], "positive_category_counts": categories,
            "ignored_source_object_counts": ignores}


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2))
