#!/usr/bin/env python3
"""Prepare the frozen legal-fit ODVG stream for controlled generic adaptation.

The generated JSONL and label map live in the isolated reference environment,
not in the repository.  Only the deterministic category mapping and metadata
are committed.  Calibration/validation/official videos are never opened.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locatemot.unified.data.legal_scope import LEGAL_SPLITS, assert_legal_path, assert_legal_video  # noqa: E402

CATEGORY_MAPPING = {"0": "car"}


def read_boxes(path: Path, width: int, height: int) -> list[list[float]]:
    boxes: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        _, _, cx, cy, box_w, box_h = fields[:6]
        cx, cy, box_w, box_h = map(float, (cx, cy, box_w, box_h))
        x0 = max(0.0, (cx - box_w / 2.0) * width)
        y0 = max(0.0, (cy - box_h / 2.0) * height)
        x1 = min(float(width), (cx + box_w / 2.0) * width)
        y1 = min(float(height), (cy + box_h / 2.0) * height)
        if x1 > x0 and y1 > y0:
            boxes.append([x0, y0, x1, y1])
    return boxes


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Keep one row per dataset/video/frame so V1/V2 labels are never merged
    # through an implicit ID conversion.  Both sources remain legal fit data.
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        for raw_video in LEGAL_SPLITS[dataset]["fit"]:
            video = assert_legal_video(raw_video, dataset=dataset)
            label_root = PROJECT_ROOT / "data" / dataset / "labels_with_ids" / "image_02" / video
            image_root = PROJECT_ROOT / "data/kitti_tracking/training/image_02" / video
            for label_path in sorted(label_root.glob("*.txt")):
                image_path = assert_legal_path(image_root / f"{label_path.stem}.png")
                if not image_path.is_file():
                    continue
                from PIL import Image
                with Image.open(image_path) as image:
                    width, height = image.size
                boxes = read_boxes(label_path, width, height)
                if not boxes:
                    continue
                rows.append({
                    "filename": str(image_path),
                    "height": height,
                    "width": width,
                    "detection": {
                        "instances": [
                            {"bbox": box, "label": 0}
                            for box in boxes
                        ]
                    },
                    "dataset": dataset,
                    "video": video,
                    "frame": int(label_path.stem),
                })
    rows.sort(key=lambda row: (row["dataset"], row["video"], row["frame"]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    ann = args.out_root / "legal_fit_odvg.jsonl"
    with ann.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    label_map = args.out_root / "legal_category_map.json"
    label_map.write_text(json.dumps(CATEGORY_MAPPING, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "format": "locatemot-u-u1-generic-adaptation-data-v1",
        "dataset": "Refer-KITTI legal fit only",
        "rows": len(rows),
        "category_mapping": CATEGORY_MAPPING,
        "ann_file": str(ann.resolve()),
        "label_map_file": str(label_map.resolve()),
        "official_test_labels_read": False,
        "screening_gt_used": False,
        "training_started": False,
    }
    (args.out_root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    mapping_out = PROJECT_ROOT / "outputs/unified/u1/category_mapping.json"
    mapping_out.parent.mkdir(parents=True, exist_ok=True)
    mapping_out.write_text(json.dumps({
        "format": "locatemot-u-u1-category-mapping-v1",
        "generic_vocabulary": ["car", "van", "truck", "bus", "tram", "pedestrian", "person", "cyclist", "bicycle", "motorcycle"],
        "fit_label_mapping": CATEGORY_MAPPING,
        "policy": "fixed before adaptation; no dev-result vocabulary edits",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
