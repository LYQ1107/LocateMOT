#!/usr/bin/env python3
"""Compile dense C-TAO plus modal TAO TRAIN labels into separate base-only GT."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.unified.build_u2_candidate_cache import DIRECTORY, PLAN, digest, write_json

OUT = ROOT / "outputs/unified/u2/ovmot_supervision"


def physical_frame_index(name: str, camera_source_index: int | None = None) -> int:
    match = re.fullmatch(r"(?:frame)?(\d+)\.jpg", Path(name).name)
    if match is not None:
        return int(match[1]) - 1
    if "/ArgoVerse/" in name and camera_source_index is not None:
        # Camera filenames contain nanosecond timestamps, not frame numbers.
        # Original TAO supplies absolute indices; dense camera timestamps are
        # aligned between those anchors, preserving missing-image gaps.
        return int(camera_source_index)
    raise ValueError(f"Original image timeline has no registered time mapping: {name}")


def compile_supervision(out: Path = OUT) -> dict:
    vocabulary_path = ROOT / "outputs/unified/u2/protocol/lvis_v1_vocabulary.json"
    vocabulary = json.loads(vocabulary_path.read_text())["categories"]
    by_synset = {category["synset"]: category for category in vocabulary}
    base_ids = {category["id"] - 1 for category in vocabulary if category["frequency"] in {"f", "c"}}
    groups = {group["video"]: group for group in (json.loads(line) for line in PLAN.read_text().splitlines())
              if group["dataset"] == "ovmot" and group["role"] == "train"}
    registered = json.loads((DIRECTORY / "supervision_source_inventory.json").read_text())["ovmot"]
    frame_records: dict[str, dict] = {}
    camera_anchors: dict[str, list[tuple[int, int]]] = defaultdict(list)
    identity_classes: dict[tuple[str, str], int] = {}
    identity_source_ids: dict[tuple[str, str], dict[str, int]] = defaultdict(dict)
    video_negatives: dict[str, list[set[int]]] = defaultdict(list)
    counts, source_records = Counter(), {}
    for stream, filename in (("tao_train", "data/TAO-Amodal/annotations/train.json"),
                             ("ctao", "data/C-TAO/ctao_base.json")):
        path = ROOT / filename
        expected = registered[stream]["sha256"]
        if digest(path) != expected:
            raise ValueError(f"Frozen {stream} original annotations changed")
        data = json.loads(path.read_text())
        classes = {category["id"]: by_synset.get(category["synset"]) for category in data["categories"]}

        def map_negative(source_ids):
            return {classes[category]["id"] - 1 for category in source_ids
                    if classes.get(category) is not None and classes[category]["frequency"] in {"f", "c"}}

        videos = {video["id"]: video for video in data["videos"]}
        for video in videos.values():
            if video["name"] not in groups or not video["name"].startswith("train/"):
                raise ValueError("OVMOT supervision cannot read validation/test videos")
            video_negatives[video["name"]].append(map_negative(video.get("neg_category_ids", [])))
        images = {image["id"]: image for image in data["images"]}
        if stream == "tao_train":
            for image in images.values():
                name = image["file_name"]
                if "/ArgoVerse/" in name:
                    timestamp = int(Path(name).stem.rsplit("_", 1)[1])
                    camera_anchors[str(Path(name).parent)].append((timestamp, image["frame_index"]))
            for anchors in camera_anchors.values():
                anchors.sort()
        for image in images.values():
            name = image["file_name"]
            video = str(Path(name).parent)
            frame = Path(name).name
            if video not in groups or frame not in groups[video]["frames"]:
                raise ValueError("Original OVMOT source image is absent from frozen candidate scope")
            camera_index = image["frame_index"] if stream == "tao_train" else None
            if stream == "ctao" and "/ArgoVerse/" in name:
                if name in frame_records:
                    camera_index = frame_records[name]["source_frame_index"]
                    counts["ctao_camera_original_time_anchors"] += 1
                else:
                    anchors = camera_anchors[video]
                    timestamp = int(Path(name).stem.rsplit("_", 1)[1])
                    right = bisect_right(anchors, (timestamp, float("inf")))
                    if not 0 < right < len(anchors):
                        raise ValueError("Dense camera image falls outside original TAO time anchors")
                    left_time, left_index = anchors[right - 1]
                    right_time, right_index = anchors[right]
                    span = right_time - left_time
                    offset = (timestamp - left_time) * (right_index - left_index)
                    camera_index = left_index + (2 * offset + span) // (2 * span)
                    counts["ctao_camera_timestamp_interpolated_source_indices"] += 1
            physical = physical_frame_index(name, camera_index)
            if stream == "ctao" and image.get("frame_index") != physical:
                counts["ctao_source_frame_index_differs_from_physical_filename"] += 1
            record = frame_records.setdefault(name, {"frame": frame, "video": video, "source_frame_index": physical,
                                                     "annotations": {}, "ignore_boxes": [], "source_image_ids": {}})
            record["source_image_ids"][stream] = image["id"]
            counts[f"{stream}_images"] += 1
        for annotation in data["annotations"]:
            counts[f"{stream}_raw_gt_units"] += 1
            category = classes.get(annotation["category_id"])
            if category is None:
                counts[f"{stream}_unmapped_gt_excluded_from_training"] += 1
                continue
            if category["frequency"] not in {"f", "c"}:
                counts[f"{stream}_novel_gt_excluded_from_training"] += 1
                continue
            # Do not inspect a novel/unmapped identity or box to form a label.
            image = images[annotation["image_id"]]
            name, video = image["file_name"], str(Path(image["file_name"]).parent)
            uuid = annotation["_scale_uuid"]
            key = video, uuid
            category_index = category["id"] - 1
            if key in identity_classes and identity_classes[key] != category_index:
                raise ValueError("Original annotation UUID changes base category across sources")
            identity_classes[key] = category_index
            source_id = int(annotation["track_id"])
            if stream in identity_source_ids[key] and identity_source_ids[key][stream] != source_id:
                raise ValueError("Original UUID has multiple source track IDs inside one stream/video")
            identity_source_ids[key][stream] = source_id
            x, y, width, height = map(float, annotation["bbox"])
            box = [x, y, x + width, y + height]
            if width <= 0 or height <= 0:
                # The official dense release contains nine zero-area units.
                # Preserve them in source coverage; ordinary IoU gives zero
                # overlap, so no candidate acquires an invented GT match.
                counts[f"{stream}_nonpositive_extent_gt_units_preserved"] += 1
            record = frame_records[name]
            if annotation.get("iscrowd", False):
                if box not in record["ignore_boxes"]:
                    record["ignore_boxes"].append(box)
                counts[f"{stream}_base_crowd_units"] += 1
                continue
            value = {"uuid": uuid, "category_id": category_index, "box": box}
            if uuid in record["annotations"]:
                if record["annotations"][uuid] != value:
                    raise ValueError("Duplicate original base UUID/frame has different source geometry/category")
                counts["duplicate_base_gt_units_shared_by_ctao_and_tao"] += 1
            else:
                record["annotations"][uuid] = value
                counts["unique_base_gt_units"] += 1
            counts[f"{stream}_base_positive_gt_units"] += 1
        source_records[stream] = {"path": str(path), "sha256": expected}
        del data, images, videos
    if counts["ctao_novel_gt_excluded_from_training"] or counts["ctao_unmapped_gt_excluded_from_training"]:
        raise ValueError("Registered C-TAO base release contains a novel/unmapped positive class")
    manifest_groups = []
    for video, group in sorted(groups.items()):
        uuids = sorted(uuid for identity_video, uuid in identity_classes if identity_video == video)
        canonical = {uuid: index for index, uuid in enumerate(uuids)}
        # Only categories certified absent by every available original stream
        # are used as negative class labels. Other unmatched proposals remain
        # in the model but are not automatically labeled background/NEW.
        negatives = set.intersection(*video_negatives[video]) if video_negatives[video] else set()
        records = []
        for frame in group["frames"]:
            name = f"{video}/{frame}"
            record = frame_records[name]
            annotations = [record["annotations"][uuid] for uuid in sorted(record["annotations"])]
            records.append({"frame": frame, "source_frame_index": record["source_frame_index"],
                            "source_image_ids": record["source_image_ids"],
                            "boxes": [annotation["box"] for annotation in annotations],
                            "identity_ids": [canonical[annotation["uuid"]] for annotation in annotations],
                            "category_ids": [annotation["category_id"] for annotation in annotations],
                            "ignore_boxes": record["ignore_boxes"]})
        if any(left["source_frame_index"] >= right["source_frame_index"] for left, right in zip(records, records[1:])):
            raise ValueError("Frozen OVMOT physical filename sequence is not causal")
        tracks = [{"identity_id": canonical[uuid], "annotation_uuid": uuid,
                   "category_id": identity_classes[(video, uuid)], "source_track_ids": identity_source_ids[(video, uuid)]}
                  for uuid in uuids]
        for track in tracks:
            source_ids = track["source_track_ids"]
            if len(source_ids) == 2 and source_ids["ctao"] != source_ids["tao_train"]:
                counts["shared_uuid_with_different_numeric_source_track_ids"] += 1
        payload = {"format": "locatemot-u2-ovmot-base-supervision-video-v1", "group_key": group["key"],
                   "video": video, "frames": records, "tracks": tracks, "negative_category_ids": sorted(negatives),
                   "class_id_convention": "zero-based frozen LVIS-v1 vocabulary index",
                   "novel_class_or_identity_supervision": False}
        path = out / "videos" / video / "supervision.json"
        write_json(path, payload)
        manifest_groups.append({"key": group["key"], "video": video, "path": str(path.resolve()),
                                "sha256": digest(path), "frames": len(records), "tracks": len(tracks),
                                "positive_gt_units": sum(len(record["boxes"]) for record in records)})
    if (sum(group["frames"] for group in manifest_groups) != len(frame_records)
            or sum(group["positive_gt_units"] for group in manifest_groups) != counts["unique_base_gt_units"]):
        raise ValueError("Compiled base supervision omits an original registered image or GT unit")
    report = {"format": "locatemot-u2-ovmot-base-supervision-manifest-v1", "status": "COMPLETE",
              "inventory_sha256": digest(PLAN), "vocabulary_sha256": digest(vocabulary_path),
              "source_annotations": source_records, "counts": dict(counts), "videos": len(manifest_groups),
              "images": len(frame_records), "base_class_ids": sorted(base_ids), "groups": manifest_groups,
              "physical_time": "frameNNNN.jpg and LaSOT NNNN.jpg -> NNNN-1; ArgoVerse original TAO indices and integer-rounded timestamp interpolation between original anchors, preserving missing-frame gaps; dense frame_id/reset frame_index never used",
              "identity_namespace": "per-video original annotation UUID, with source numeric IDs retained separately",
              "unknown_proposals": "retained as model input; no automatic NEW/background/identity label",
              "novel_training_gt_units": 0, "validation_or_test_labels_read": False,
              "formal_joint_optimization_launched": False}
    write_json(out / "manifest.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    report = compile_supervision(args.out)
    print(json.dumps({key: report[key] for key in ("status", "videos", "images", "counts", "novel_training_gt_units")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
