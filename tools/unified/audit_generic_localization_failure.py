#!/usr/bin/env python3
"""Audit the U1 generic GT coordinate and large-object contract.

Only legal calibration/validation Refer labels are read.  The official KITTI
labels are used for a same-frame geometry comparison (the two sources use
different track-id namespaces); reserved videos are rejected before access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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

DEFAULT_CHECKPOINT = Path("/data2/user/reference_envs/u1_generic_adaptation_20261001/valid_work_bf16_batch2_frozen01_noacc/epoch_4.pth")
OFFICIAL_TYPE = {
    "Car": "car", "Van": "van", "Truck": "truck", "Bus": "bus",
    "Pedestrian": "pedestrian", "Person_sitting": "person",
    "Cyclist": "cyclist", "Tram": "tram",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iou(left: list[float], right: list[float]) -> float:
    x0 = max(left[0], right[0]); y0 = max(left[1], right[1])
    x1 = min(left[2], right[2]); y1 = min(left[3], right[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_l = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_r = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_l + area_r - inter
    return inter / union if union > 0 else 0.0


def read_official(path: Path) -> dict[int, list[dict[str, Any]]]:
    output: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 10 or fields[2] == "DontCare":
            continue
        output[int(fields[0])].append({
            "track_id": fields[1],
            "type": fields[2],
            "box": [float(value) for value in fields[6:10]],
            "truncation": float(fields[3]),
            "occlusion": int(float(fields[4])),
        })
    return dict(output)


def read_refer(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    output = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        x, y, w, h = (float(value) for value in fields[2:6])
        output.append({
            "track_id": fields[1],
            "box": [x * width, y * height, (x + w) * width, (y + h) * height],
        })
    return output


def new_stats() -> dict[str, Any]:
    return {
        "total_gt_boxes": 0,
        "out_of_image": {"x0_lt_0": 0, "y0_lt_0": 0, "x1_gt_width": 0, "y1_gt_height": 0, "any": 0},
        "size": {name: {"total": 0, "out_of_image": 0} for name in ("small", "medium", "large")},
        "refer_official_match": {"count": 0, "iou_sum": 0.0, "iou_ge_0_5": 0, "iou_ge_0_9": 0, "track_id_equal": 0, "coordinate_delta_sum": 0.0, "truncation": Counter()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--official-label-root", type=Path, default=Path("/data2/user/reference_envs/kitti_tracking_labels_20261001/training/label_02"))
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs/unified/u1r2/generic_localization_failure_audit.json")
    parser.add_argument("--formal-result", type=Path, default=None)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    by_dataset = {dataset: new_stats() for dataset in ("refer_kitti_v1", "refer_kitti_v2")}
    union = new_stats()
    official_cache: dict[str, dict[int, list[dict[str, Any]]]] = {}
    samples = []
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        for split in ("calibration", "validation"):
            for raw_video in LEGAL_SPLITS[dataset][split]:
                video = assert_legal_video(raw_video, dataset=dataset)
                label_root = assert_legal_path(PROJECT_ROOT / "data" / dataset / "labels_with_ids" / "image_02" / video)
                image_root = assert_legal_path(PROJECT_ROOT / "data/kitti_tracking/training/image_02" / video)
                official_path = assert_legal_path(args.official_label_root / f"{video}.txt")
                if video not in official_cache:
                    official_cache[video] = read_official(official_path)
                for label_path in sorted(label_root.glob("*.txt")):
                    image_path = assert_legal_path(image_root / f"{label_path.stem}.png")
                    if not image_path.is_file():
                        continue
                    with Image.open(image_path) as image:
                        width, height = image.size
                    labels = read_refer(label_path, width, height)
                    stats = by_dataset[dataset]
                    for label in labels:
                        box = label["box"]
                        height_px = max(0.0, box[3] - box[1])
                        size = "small" if height_px < 32 else "medium" if height_px <= 96 else "large"
                        any_out = box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height
                        for target in (stats, union):
                            target["total_gt_boxes"] += 1
                            target["out_of_image"]["x0_lt_0"] += int(box[0] < 0)
                            target["out_of_image"]["y0_lt_0"] += int(box[1] < 0)
                            target["out_of_image"]["x1_gt_width"] += int(box[2] > width)
                            target["out_of_image"]["y1_gt_height"] += int(box[3] > height)
                            target["out_of_image"]["any"] += int(any_out)
                            target["size"][size]["total"] += 1
                            target["size"][size]["out_of_image"] += int(any_out)
                        official = official_cache[video].get(int(label_path.stem), [])
                        candidate = max(((iou(box, row["box"]), row) for row in official), key=lambda value: value[0], default=(0.0, None))
                        best_iou, match = candidate
                        if match is not None:
                            delta = max(abs(a - b) for a, b in zip(box, match["box"]))
                            for target in (stats, union):
                                metric = target["refer_official_match"]
                                metric["count"] += 1
                                metric["iou_sum"] += best_iou
                                metric["iou_ge_0_5"] += int(best_iou >= 0.5)
                                metric["iou_ge_0_9"] += int(best_iou >= 0.9)
                                metric["track_id_equal"] += int(label["track_id"] == match["track_id"])
                                metric["coordinate_delta_sum"] += delta
                                metric["truncation"][str(match["truncation"])] += 1
                            if len(samples) < 12:
                                samples.append({"dataset": dataset, "video": video, "frame": int(label_path.stem), "refer_track_id": label["track_id"], "official_track_id": match["track_id"], "official_type": match["type"], "iou": best_iou, "coordinate_delta_max": delta, "truncation": match["truncation"], "occlusion": match["occlusion"]})
    def clean(value: Any) -> Any:
        if isinstance(value, Counter):
            return dict(value)
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        return value
    for value in [*by_dataset.values(), union]:
        metric = value["refer_official_match"]
        count = metric["count"]
        metric["mean_iou"] = metric["iou_sum"] / count if count else 0.0
        metric["mean_coordinate_delta_max"] = metric["coordinate_delta_sum"] / count if count else 0.0
        metric["iou_sum"] = None
        metric["coordinate_delta_sum"] = None
    payload = {
        "format": "locatemot-u-u1r2-generic-localization-failure-audit-v1",
        "status": "COMPLETE",
        "checkpoint": {"path": str(args.checkpoint.resolve()), "sha256": sha256(args.checkpoint)},
        "coordinate_contract": {"refer_label_format": "normalized top-left xywh", "evaluation_policy": "source extent retained; no GT clipping", "official_matching": "same-frame maximum-IoU geometry; track IDs are separate namespaces"},
        "by_dataset": clean(by_dataset),
        "union": clean(union),
        "samples": samples,
        "formal_result": json.loads(args.formal_result.read_text(encoding="utf-8")) if args.formal_result and args.formal_result.is_file() else None,
        "official_test_labels_read": False,
        "screening_gt_used": False,
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "union": payload["union"], "checkpoint_sha256": payload["checkpoint"]["sha256"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
