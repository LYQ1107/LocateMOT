#!/usr/bin/env python3
"""Audit the recovered Refer-KITTI V1/V2 train-pool assets.

Only fit/calibration/validation videos are opened.  The reserved official
evaluation videos are recorded by name but their labels and expressions are
not read.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from locatemot.paths import IMAGE_ROOT, V1_ROOT, V2_ROOT

SPLITS = {
    "refer_kitti_v1": {
        "fit": ("0001", "0002", "0003", "0006", "0007", "0008", "0009", "0010", "0012", "0014", "0015", "0020"),
        "calibration": ("0016",),
        "validation": ("0004", "0018"),
        "official_eval_reserved": ("0005", "0011", "0013"),
    },
    "refer_kitti_v2": {
        "fit": ("0000", "0001", "0002", "0003", "0006", "0007", "0008", "0009", "0010", "0012", "0014"),
        "calibration": ("0015",),
        "validation": ("0016", "0017", "0020"),
        "official_eval_reserved": ("0005", "0011", "0013", "0019"),
    },
}
ROOTS = {"refer_kitti_v1": V1_ROOT, "refer_kitti_v2": V2_ROOT}
OUT = Path("outputs/restore_20261001/refer_kitti_audit.json")


def label_ids(path: Path) -> set[str]:
    values: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            values.add(str(fields[1]))
    return values


def audit_dataset(dataset: str, root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"dataset": dataset, "splits": {}, "official_test_labels_read": False}
    for split, videos in SPLITS[dataset].items():
        if split == "official_eval_reserved":
            result[split] = {"videos": list(videos), "read": False}
            continue
        expr_paths = [p for video in videos for p in sorted((root / "expression" / video).glob("*.json"))]
        label_paths = {p.stem: p for video in videos for p in sorted((root / "labels_with_ids" / "image_02" / video).glob("*.txt"))}
        image_paths = {p.stem: p for video in videos for p in sorted((IMAGE_ROOT / video).glob("*.png"))}
        frame_refs = 0
        target_refs = 0
        missing_label_frames: list[str] = []
        missing_target_ids: list[str] = []
        query_records = 0
        sentences_empty = 0
        for expr in expr_paths:
            payload = json.loads(expr.read_text(encoding="utf-8"))
            sentence = str(payload.get("sentence", "")).strip()
            sentences_empty += int(not sentence)
            query_records += 1
            video = expr.parent.name
            for frame, ids in payload.get("label", {}).items():
                frame_refs += 1
                frame_name = f"{int(frame):06d}"
                label_path = root / "labels_with_ids" / "image_02" / video / f"{frame_name}.txt"
                if not label_path.is_file():
                    missing_label_frames.append(f"{video}/{frame_name}")
                    continue
                available = label_ids(label_path)
                for target in ids or []:
                    target_refs += 1
                    if str(target) not in available:
                        missing_target_ids.append(f"{video}/{frame_name}/{target}")
        result["splits"][split] = {
            "videos": list(videos),
            "expression_files": len(expr_paths),
            "label_files": len(label_paths),
            "image_files": len(image_paths),
            "frame_references": frame_refs,
            "target_references": target_refs,
            "empty_sentences": sentences_empty,
            "missing_label_frames": missing_label_frames[:20],
            "missing_target_ids": missing_target_ids[:20],
            "complete": not missing_label_frames and not missing_target_ids and sentences_empty == 0,
        }
    return result


def main() -> None:
    payload = {
        "format": "locatemot-refer-kitti-restore-audit-v1",
        "datasets": {name: audit_dataset(name, root) for name, root in ROOTS.items()},
        "official_test_labels_read": False,
        "source": {
            "v1": "https://github.com/wudongming97/RMOT/releases/download/v1.0/",
            "v2": "https://drive.google.com/drive/folders/1eaxuRK-ewl0cpGshOxylSFZ5PPu3_WUT",
            "images": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for name, audit in payload["datasets"].items():
        print(name)
        for split, stats in audit["splits"].items():
            if isinstance(stats, dict) and "complete" in stats:
                print(split, stats["expression_files"], stats["label_files"], stats["image_files"], stats["frame_references"], stats["target_references"], stats["complete"])


if __name__ == "__main__":
    main()
