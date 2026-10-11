#!/usr/bin/env python3
"""Freeze official DanceTrack frames while preserving the project video split."""
from __future__ import annotations

import configparser
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "outputs/unified/u2/protocol"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze() -> dict:
    acquisition = ROOT / "outputs/unified/u2/acquisition/dance_extraction.json"
    extracted = json.loads(acquisition.read_text())
    if extracted["state"] != "COMPLETE" or not all(
        entry["crc_checked_during_extraction"] for entry in extracted["archives"]
    ):
        raise RuntimeError("DanceTrack release extraction is incomplete")
    split_sources, split_videos, sequences, differences = {}, {}, [], []
    for split in ("train", "calibration", "val"):
        source = ROOT / f"configs/data/l1_a_dancetrack_{split}.json"
        registered = json.loads(source.read_text())
        split_sources[split] = {"path": str(source), "sha256": digest(source), "seed": registered["seed"]}
        split_videos[split] = sorted(video["video_id"] for video in registered["videos"])
        for video in sorted(registered["videos"], key=lambda item: item["video_id"]):
            directory = ROOT / "data/DanceTrack" / ("val" if split == "val" else "train") / video["video_id"]
            info = configparser.ConfigParser()
            info.read(directory / "seqinfo.ini")
            metadata = info["Sequence"]
            length = int(metadata["seqLength"])
            frames = [f"{frame:08d}{metadata['imExt']}" for frame in range(1, length + 1)]
            image_dir = directory / metadata["imDir"]
            if sorted(path.name for path in image_dir.glob(f"*{metadata['imExt']}")) != frames:
                raise RuntimeError(f"Incomplete official frame coverage: {directory}")
            gt = directory / "gt/gt.txt"
            fields = [[float(value) for value in line.split(",")] for line in gt.read_text().splitlines()]
            if any(not 1 <= row[0] <= length for row in fields):
                raise RuntimeError(f"GT frame outside official sequence: {gt}")
            if video["frames"] != length:
                differences.append({"sequence": directory.name, "split": split,
                                    "old_metadata_frames": video["frames"], "official_frames": length})
            sequences.append({"sequence": directory.name, "split": split,
                              "image_dir": str(image_dir), "gt_path": str(gt), "gt_sha256": digest(gt),
                              "seqinfo_sha256": digest(directory / "seqinfo.ini"),
                              "width": int(metadata["imWidth"]), "height": int(metadata["imHeight"]),
                              "fps": int(metadata["frameRate"]), "length": length,
                              "train_frames": frames if split == "train" else [],
                              "calibration_frames": frames if split == "calibration" else [],
                              "validation_frames": frames if split == "val" else [],
                              "positive_gt_boxes": sum(row[6] == 1 and row[7] == 1 for row in fields)})
    groups = [set(videos) for videos in split_videos.values()]
    if any(groups[i] & groups[j] for i in range(len(groups)) for j in range(i)):
        raise RuntimeError("Project DanceTrack video splits overlap")
    if set(path.name for path in (ROOT / "data/DanceTrack/train").iterdir()) != groups[0] | groups[1]:
        raise RuntimeError("Project train/calibration videos differ from the official train release")
    if set(path.name for path in (ROOT / "data/DanceTrack/val").iterdir()) != groups[2]:
        raise RuntimeError("Project validation videos differ from the official validation release")
    payload = {"format": "locatemot-u2-dancetrack-v1", "dataset": "dancetrack",
               "root": str(ROOT / "data/DanceTrack"), "version": "a0ba42ac690c41e9850a20e76a0b9450a6fb6a47",
               "source": "https://huggingface.co/datasets/noahcao/dancetrack",
               "license": "CC-BY-4.0 annotations; non-commercial research video terms",
               "source_archives": extracted["archives"], "project_split_sources": split_sources,
               "split_videos": split_videos, "metadata_frame_differences": differences,
               "vocabulary": "person .", "custom_entities": True, "max_per_img": 900,
               "measured_top_k": [150, 300], "candidate_top_k": 150, "iou_threshold": .5,
               "gt_policy": {"positive": "mark == 1 and category == 1", "visibility_cutoff": None,
                             "boxes": "source xywh converted to xyxy without clipping or coordinate shift",
                             "recall_unit": "one positive physical GT object per frame"},
               "sequences": sequences,
               "train_images": sum(len(s["train_frames"]) for s in sequences),
               "calibration_images": sum(len(s["calibration_frames"]) for s in sequences),
               "validation_images": sum(len(s["validation_frames"]) for s in sequences),
               "official_test_labels_read": False}
    path = PROTOCOL / "dancetrack.json"
    contents = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text() != contents:
        raise RuntimeError(f"Existing frozen protocol differs: {path}")
    path.write_text(contents)
    return {"protocol": str(path), "sha256": digest(path),
            **{key: payload[key] for key in ("train_images", "calibration_images", "validation_images", "metadata_frame_differences")}}


if __name__ == "__main__":
    print(json.dumps(freeze(), indent=2))
