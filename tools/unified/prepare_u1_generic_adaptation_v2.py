#!/usr/bin/env python3
"""Build the class-aware, physical-frame U1R2 generic stream.

Refer-KITTI ``labels_with_ids`` stores normalized top-left ``x, y, w, h``.
The Refer track-id namespace is not the KITTI tracking namespace, so class
recovery uses a deterministic same-frame box match to the official KITTI
training label.  The audit records the match IoU and any low-confidence or
unmatched rows; no calibration, validation, or reserved video is opened.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locatemot.unified.data.legal_scope import (  # noqa: E402
    LEGAL_SPLITS,
    assert_legal_path,
    assert_legal_video,
)

OFFICIAL_TYPE_TO_NAME = {
    "Car": "car",
    "Van": "van",
    "Truck": "truck",
    "Bus": "bus",
    "Pedestrian": "pedestrian",
    "Person_sitting": "person",
    "Cyclist": "cyclist",
    "Tram": "tram",
    # Misc is intentionally ignored under the fixed U1R2 policy.
    "Misc": None,
    "DontCare": None,
}
CATEGORY_NAMES = (
    "car", "van", "truck", "bus", "tram", "pedestrian", "person",
    "cyclist", "bicycle", "motorcycle",
)
CATEGORY_TO_ID = {name: index for index, name in enumerate(CATEGORY_NAMES)}


def box_iou(left: list[float], right: list[float]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_l = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_r = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_l + area_r - inter
    return inter / union if union > 0.0 else 0.0


def center_distance(left: list[float], right: list[float], width: int, height: int) -> float:
    lx = (left[0] + left[2]) / 2.0 / width
    ly = (left[1] + left[3]) / 2.0 / height
    rx = (right[0] + right[2]) / 2.0 / width
    ry = (right[1] + right[3]) / 2.0 / height
    return math.hypot(lx - rx, ly - ry)


def read_refer(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        x, y, box_w, box_h = (float(value) for value in fields[2:6])
        rows.append({
            "refer_track_id": str(fields[1]),
            "box": [x * width, y * height, (x + box_w) * width, (y + box_h) * height],
        })
    return rows


def read_official(path: Path) -> dict[int, list[dict[str, Any]]]:
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 10:
            continue
        frame = int(fields[0])
        object_type = fields[2]
        if object_type == "DontCare":
            continue
        by_frame[frame].append({
            "official_track_id": str(fields[1]),
            "type": object_type,
            "name": OFFICIAL_TYPE_TO_NAME.get(object_type),
            "truncation": float(fields[3]),
            "occlusion": int(float(fields[4])),
            "box": [float(value) for value in fields[6:10]],
        })
    return by_frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-label-root", type=Path, default=Path("/data2/user/reference_envs/kitti_tracking_labels_20261001/training/label_02"))
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "outputs/unified/u1r2")
    parser.add_argument("--match-iou", type=float, default=0.0,
                        help="minimum positive same-frame overlap; zero keeps every geometrically overlapping source box and reports low-confidence matches")
    parser.add_argument("--low-confidence-iou", type=float, default=0.50)
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    # (video, frame, refer track id) -> source boxes.  This is intentionally
    # keyed by physical video/frame, never dataset/video/frame.
    source: dict[tuple[str, int, str], dict[str, Any]] = {}
    source_frame_scopes: defaultdict[tuple[str, int], set[str]] = defaultdict(set)
    source_counts = Counter()
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        for raw_video in LEGAL_SPLITS[dataset]["fit"]:
            video = assert_legal_video(raw_video, dataset=dataset)
            label_root = assert_legal_path(PROJECT_ROOT / "data" / dataset / "labels_with_ids" / "image_02" / video)
            image_root = assert_legal_path(PROJECT_ROOT / "data/kitti_tracking/training/image_02" / video)
            official_path = assert_legal_path(args.official_label_root / f"{video}.txt")
            if not official_path.is_file():
                raise FileNotFoundError(f"missing official legal-fit label: {official_path}")
            for label_path in sorted(label_root.glob("*.txt")):
                frame = int(label_path.stem)
                image_path = assert_legal_path(image_root / f"{label_path.stem}.png")
                if not image_path.is_file():
                    continue
                with Image.open(image_path) as image:
                    width, height = image.size
                for row in read_refer(label_path, width, height):
                    key = (video, frame, row["refer_track_id"])
                    item = source.setdefault(key, {
                        "video": video,
                        "frame": frame,
                        "image": str(image_path.resolve()),
                        "width": width,
                        "height": height,
                        "sources": {},
                    })
                    item["sources"].setdefault(dataset, row["box"])
                    source_frame_scopes[(video, frame)].add(dataset)
                    source_counts[dataset] += 1

    # Preload only legal-fit official labels.  The explicit scope check above
    # is repeated here so a future split edit cannot silently read reserved GT.
    official_by_video = {}
    for video, _frame in sorted(source_frame_scopes):
        assert_legal_video(video)
        p = assert_legal_path(args.official_label_root / f"{video}.txt")
        official_by_video[video] = read_official(p)

    stats = Counter()
    class_counts = Counter()
    match_iou_values: list[float] = []
    conflict_deltas: list[float] = []
    chosen_by_frame: defaultdict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for item in source.values():
        video, frame = item["video"], item["frame"]
        boxes = item["sources"]
        if "refer_kitti_v1" in boxes and "refer_kitti_v2" in boxes:
            stats["v1_v2_overlap_rows"] += 1
            v1_box, v2_box = boxes["refer_kitti_v1"], boxes["refer_kitti_v2"]
            conflict_deltas.append(max(abs(a - b) for a, b in zip(v1_box, v2_box)))
            if box_iou(v1_box, v2_box) < 0.95:
                stats["v1_v2_conflicting_box_count"] += 1
            stats["v1_v2_rows_removed_by_dedup"] += 1
        # V2 is the corrected source when both source versions contain the
        # same Refer track id; otherwise retain the sole available box.
        box = boxes["refer_kitti_v2"] if "refer_kitti_v2" in boxes else boxes["refer_kitti_v1"]
        official = official_by_video[video].get(frame, [])
        candidates = [row for row in official if row["name"] is not None]
        match = max(
            ((box_iou(box, row["box"]), row) for row in candidates),
            key=lambda value: (value[0], -center_distance(box, value[1]["box"], item["width"], item["height"])),
            default=(0.0, None),
        )
        best_iou, best = match
        match_iou_values.append(best_iou)
        if best is None or best_iou <= args.match_iou:
            stats["missing_class_count"] += 1
            continue
        if best_iou < args.low_confidence_iou:
            stats["low_confidence_class_match_count"] += 1
        # Avoid a duplicate physical official object when V1/V2 use distinct
        # local ids for the same box in one frame.
        official_key = best["official_track_id"]
        prior = chosen_by_frame[(video, frame)].get(official_key)
        if prior is not None:
            stats["duplicate_official_object_rows_removed"] += 1
            if best_iou > prior["match_iou"]:
                chosen_by_frame[(video, frame)][official_key] = {
                    "item": item, "box": box, "match": best, "match_iou": best_iou,
                }
            continue
        chosen_by_frame[(video, frame)][official_key] = {
            "item": item, "box": box, "match": best, "match_iou": best_iou,
        }

    rows: list[dict[str, Any]] = []
    official_type_counts = Counter()
    for key in sorted(chosen_by_frame):
        entries = chosen_by_frame[key]
        instances = []
        for entry in sorted(entries.values(), key=lambda value: value["match"]["official_track_id"]):
            item, box, match = entry["item"], entry["box"], entry["match"]
            name = match["name"]
            if name not in CATEGORY_TO_ID:
                stats["ignored_misc_or_unknown_count"] += 1
                continue
            class_counts[name] += 1
            official_type_counts[match["type"]] += 1
            instances.append({
                "bbox": [float(value) for value in box],
                "label": CATEGORY_TO_ID[name],
                "category": name,
                "source_refer_track_id": next(iter(item["sources"].keys()), ""),
                "official_track_id": match["official_track_id"],
                "official_type": match["type"],
                "match_iou": entry["match_iou"],
                "truncation": match["truncation"],
                "occlusion": match["occlusion"],
            })
        if instances:
            video, frame = key
            # Use the first item only for dimensions/path; all entries share it.
            any_item = next(iter(entries.values()))["item"]
            rows.append({
                "filename": any_item["image"],
                "height": any_item["height"],
                "width": any_item["width"],
                "detection": {"instances": instances},
                "dataset": "refer_kitti_physical_fit",
                "video": video,
                "frame": frame,
                "source_datasets": sorted(source_frame_scopes[(video, frame)]),
            })
    rows.sort(key=lambda row: (row["video"], row["frame"]))
    manifest = args.out_root / "generic_train_manifest.json"
    manifest.write_text(json.dumps({
        "format": "locatemot-u-u1r2-generic-physical-frame-manifest-v1",
        "policy": {
            "unit": "one physical KITTI video/frame row",
            "source_scope": "Refer-KITTI V1/V2 legal fit only",
            "refer_box_contract": "normalized top-left xywh",
            "class_recovery": "same-frame maximum-IoU match to official KITTI tracking training labels; Refer and official track-id namespaces are not assumed equal",
            "official_type_mapping": OFFICIAL_TYPE_TO_NAME,
            "misc_policy": "ignore",
            "v1_v2_dedup": "same physical frame and official matched object, prefer V2 box when both local ids agree",
            "match_iou_threshold": args.match_iou,
        },
        "rows": rows,
        "stats": {
            "unique_images": len(rows),
            "unique_boxes": sum(len(row["detection"]["instances"]) for row in rows),
            "source_rows_v1": source_counts["refer_kitti_v1"],
            "source_rows_v2": source_counts["refer_kitti_v2"],
            "v1_v2_overlap_removed": stats["v1_v2_rows_removed_by_dedup"],
            "v1_v2_conflicting_box_count": stats["v1_v2_conflicting_box_count"],
            "duplicate_official_object_rows_removed": stats["duplicate_official_object_rows_removed"],
            "missing_class_count": stats["missing_class_count"],
            "low_confidence_class_match_count": stats["low_confidence_class_match_count"],
            "ignored_misc_or_unknown_count": stats["ignored_misc_or_unknown_count"],
            "class_counts": dict(sorted(class_counts.items())),
            "official_type_counts": dict(sorted(official_type_counts.items())),
            "match_iou": {
                "count": len(match_iou_values),
                "mean": sum(match_iou_values) / len(match_iou_values) if match_iou_values else 0.0,
                "below_0_5": sum(value < 0.5 for value in match_iou_values),
                "below_0_3": sum(value < 0.3 for value in match_iou_values),
                "min": min(match_iou_values, default=0.0),
                "p50": sorted(match_iou_values)[len(match_iou_values) // 2] if match_iou_values else 0.0,
            },
            "v1_v2_coordinate_delta_max_mean": sum(conflict_deltas) / len(conflict_deltas) if conflict_deltas else 0.0,
        },
        "official_test_labels_read": False,
        "screening_gt_used": False,
        "training_started": False,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    jsonl = args.out_root / "generic_train_manifest.jsonl"
    with jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    mapping = args.out_root / "generic_category_mapping.json"
    mapping.write_text(json.dumps({
        "format": "locatemot-u-u1r2-generic-category-mapping-v1",
        "category_to_id": CATEGORY_TO_ID,
        "id_to_category": {str(value): key for key, value in CATEGORY_TO_ID.items()},
        "official_type_to_category": OFFICIAL_TYPE_TO_NAME,
        "misc_policy": "ignore",
        "source_track_id_policy": "Refer track ids are local annotation ids; class is recovered by legal same-frame geometry match to official KITTI labels",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    label_map = args.out_root / "generic_label_map.json"
    label_map.write_text(json.dumps({str(value): key for key, value in CATEGORY_TO_ID.items()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest), "jsonl": str(jsonl), "rows": len(rows), "stats": json.loads(manifest.read_text())["stats"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
