#!/usr/bin/env python3
"""Build a legal, deterministic manifest for the latest public RMOT inputs.

The migrated checkout has the expression JSONs and corrected labels, but not
the historical L49/L69 generated indexes.  This manifest is the explicit
replacement input contract for a new reconstruction.  It reads only the
fit, calibration, and validation videos; official-evaluation videos are
listed as reserved metadata and never opened.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from locatemot.paths import IMAGE_ROOT, RESTORE_ROOT, V1_ROOT, V2_ROOT
from locatemot.rmot.l49_data import L49_SPLITS, load_l49_queries


OFFICIAL_EVAL = {
    "refer_kitti_v1": ["0005", "0011", "0013"],
    "refer_kitti_v2": ["0005", "0011", "0013", "0019"],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def label_path(root: Path, video: str, frame: int) -> Path:
    return root / "labels_with_ids" / "image_02" / video / f"{int(frame):06d}.txt"


def image_path(video: str, frame: int) -> Path:
    return IMAGE_ROOT / video / f"{int(frame):06d}.png"


def label_ids(path: Path, cache: dict[Path, set[str]]) -> set[str]:
    if path in cache:
        return cache[path]
    ids: set[str] = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 2:
                ids.add(str(fields[1]))
    cache[path] = ids
    return ids


def build_dataset(dataset: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = V1_ROOT if dataset == "refer_kitti_v1" else V2_ROOT
    rows = load_l49_queries(dataset)
    by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in ("fit", "calibration", "validation")}
    id_cache: dict[Path, set[str]] = {}
    missing_images: list[dict[str, Any]] = []
    missing_labels: list[dict[str, Any]] = []
    anomaly_counts: Counter[tuple[str, str, int, str]] = Counter()
    frame_refs = 0
    target_refs = 0
    for row in rows:
        split = str(row["split"])
        targets = row["target"]
        query_targets: dict[str, list[str]] = {}
        for frame_value, target_ids in sorted(targets.items(), key=lambda item: int(item[0])):
            frame = int(frame_value)
            values = sorted(str(value) for value in target_ids)
            query_targets[str(frame)] = values
            frame_refs += 1
            target_refs += len(values)
            image = image_path(str(row["video"]), frame)
            label = label_path(root, str(row["video"]), frame)
            if not image.is_file():
                missing_images.append({"video": str(row["video"]), "frame": frame, "path": str(image)})
            if not label.is_file():
                missing_labels.append({"video": str(row["video"]), "frame": frame, "path": str(label)})
            present = label_ids(label, id_cache)
            for target_id in values:
                if target_id not in present:
                    anomaly_counts[(str(row["video"]), str(row["query_id"]), frame, target_id)] += 1
        materialized = {
            "dataset": dataset,
            "split": split,
            "video": str(row["video"]),
            "query_id": int(row["query_id"]),
            "expression": str(row["expression"]),
            "sentence": str(row["sentence"]),
            "source": str(row["label_source"]),
            "targets_by_frame": query_targets,
        }
        by_split[split].append(materialized)

    split_summary: dict[str, Any] = {}
    query_rows: list[dict[str, Any]] = []
    for split in ("fit", "calibration", "validation"):
        values = by_split[split]
        split_summary[split] = {
            "videos": list(L49_SPLITS[dataset][split]),
            "query_count": len(values),
            "frame_references": sum(len(value["targets_by_frame"]) for value in values),
            "target_references": sum(sum(len(ids) for ids in value["targets_by_frame"].values()) for value in values),
        }
        query_rows.extend(values)

    anomalies = [
        {"video": video, "query_id": int(query_id), "frame": frame, "target_id": target_id, "occurrences": count}
        for (video, query_id, frame, target_id), count in sorted(anomaly_counts.items())
    ]
    summary = {
        "expression_root": str((root / "expression").resolve()),
        "label_root": str((root / "labels_with_ids").resolve()),
        "splits": split_summary,
        "label_files_opened": len(id_cache),
        "missing_images": missing_images,
        "missing_labels": missing_labels,
        "missing_target_id_references": anomalies,
        "missing_target_id_reference_count": sum(anomaly_counts.values()),
        "complete_frames_and_labels": not missing_images and not missing_labels,
    }
    return summary, query_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESTORE_ROOT / "latest_public_manifest.json")
    parser.add_argument("--queries", type=Path, default=RESTORE_ROOT / "latest_public_queries.jsonl")
    args = parser.parse_args()
    datasets: dict[str, Any] = {}
    all_queries: list[dict[str, Any]] = []
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        summary, queries = build_dataset(dataset)
        datasets[dataset] = summary
        all_queries.extend(queries)
    payload = {
        "format": "locatemot-latest-public-manifest-v1",
        "status": "complete_with_known_annotation_anomalies",
        "project_root": str(Path(__file__).resolve().parents[1]),
        "image_root": str(IMAGE_ROOT.resolve()),
        "official_eval_reserved": OFFICIAL_EVAL,
        "official_test_labels_read": False,
        "datasets": datasets,
        "query_manifest": str(args.queries.resolve()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.queries.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with args.queries.open("w", encoding="utf-8") as handle:
        for row in all_queries:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "manifest": str(args.out.resolve()),
        "queries": str(args.queries.resolve()),
        "query_count": len(all_queries),
        "v1_missing_target_refs": datasets["refer_kitti_v1"]["missing_target_id_reference_count"],
        "v2_missing_target_refs": datasets["refer_kitti_v2"]["missing_target_id_reference_count"],
        "complete_frames_and_labels": all(value["complete_frames_and_labels"] for value in datasets.values()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
