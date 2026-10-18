#!/usr/bin/env python3
"""Audit the existing G2 rows before creating a clean coverage manifest.

This audit is deliberately read-only.  It records the known held-out MOT20-05
intersection, MOT validity fields, sparse TAO category structure, Refer-KITTI
temporal coverage, RMOT26 query text, and path-level overlap evidence.  The
old manifest and checkpoints are never rewritten.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path
from typing import Any
import sys


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_mot(root: Path, excluded: set[str]) -> dict[str, Any]:
    sequences: dict[str, dict[str, int]] = {}
    class_counts: collections.Counter[str] = collections.Counter()
    validity = collections.Counter()
    if not root.is_dir():
        return {"root": str(root), "missing": True}
    for gt in sorted(root.glob("**/gt/gt.txt")):
        sequence = gt.parent.parent.name
        counts = collections.Counter()
        for line in gt.read_text(errors="replace").splitlines():
            fields = line.split(",")
            if len(fields) < 6:
                continue
            counts["rows"] += 1
            try:
                mark = int(float(fields[6])) if len(fields) > 6 else 1
                cls = int(float(fields[7])) if len(fields) > 7 else 1
                width = float(fields[4])
                height = float(fields[5])
            except ValueError:
                counts["malformed"] += 1
                continue
            class_counts[str(cls)] += 1
            validity[f"mark_{mark}"] += 1
            if mark != 1:
                counts["mark_invalid"] += 1
            if cls != 1:
                counts["non_person_class"] += 1
            if mark == 1 and cls == 1 and width > 0 and height > 0:
                counts["valid_person"] += 1
            else:
                counts["excluded_from_clean_person_rows"] += 1
        counts["heldout_excluded"] = int(sequence in excluded)
        sequences[sequence] = dict(counts)
    return {
        "root": str(root),
        "sequence_count": len(sequences),
        "sequences": sequences,
        "class_counts": dict(sorted(class_counts.items())),
        "validity_counts": dict(sorted(validity.items())),
    }


def audit_tao(annotation: Path) -> dict[str, Any]:
    if not annotation.is_file():
        return {"annotation": str(annotation), "missing": True}
    data = json.loads(annotation.read_text())
    categories = {int(item["id"]): item for item in data.get("categories", [])}
    annotations = data.get("annotations", [])
    by_image: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for item in annotations:
        by_image[int(item["image_id"])].append(item)
    image_categories = {
        image_id: {int(item["category_id"]) for item in items}
        for image_id, items in by_image.items()
    }
    category_frequency = collections.Counter(str(item.get("frequency", "unknown")) for item in categories.values())
    category_ann_frequency = collections.Counter(
        str(categories.get(int(item["category_id"]), {}).get("frequency", "unknown"))
        for item in annotations
    )
    multi = collections.Counter(len(values) for values in image_categories.values())
    novel = {int(k) for k, v in categories.items() if v.get("frequency") == "r"}
    base = set(categories) - novel
    return {
        "annotation": str(annotation),
        "image_count": len(data.get("images", [])),
        "annotation_count": len(annotations),
        "category_count": len(categories),
        "category_frequency": dict(sorted(category_frequency.items())),
        "annotation_frequency": dict(sorted(category_ann_frequency.items())),
        "base_categories_frequency_ne_r": len(base),
        "novel_categories_frequency_r": len(novel),
        "images_with_annotations": len(image_categories),
        "image_category_cardinality": dict(sorted((str(k), v) for k, v in multi.items())),
        "images_with_multiple_categories": sum(v for k, v in multi.items() if k > 1),
        "sparse_unannotated_images": len(data.get("images", [])) - len(image_categories),
        "policy": "clean corpus emits annotated Base (frequency!=r) categories only; no Novel GT or sparse negative is fabricated",
    }


def audit_refer(expression_root: Path, fit_videos: set[str]) -> dict[str, Any]:
    expression_count = 0
    frame_entries = 0
    multi_frame = 0
    per_video = collections.Counter()
    if expression_root.is_dir():
        for video in sorted(fit_videos):
            for path in sorted((expression_root / video).glob("*.json")):
                record = json.loads(path.read_text())
                if record.get("ignore"):
                    continue
                labels = record.get("label", {})
                count = sum(1 for value in labels.values() if value)
                if count:
                    expression_count += 1
                    frame_entries += count
                    multi_frame += int(count > 1)
                    per_video[video] += count
    return {
        "fit_videos": sorted(fit_videos),
        "expressions_with_labels": expression_count,
        "labeled_expression_frames": frame_entries,
        "expressions_with_multiple_labeled_frames": multi_frame,
        "per_video_labeled_frames": dict(sorted(per_video.items())),
        "old_builder_policy": "first valid labeled frame only",
        "clean_builder_policy": "one row per valid labeled frame, preserving exact expression text",
    }


def audit_rmot26(annotation: Path) -> dict[str, Any]:
    if not annotation.is_file():
        return {"annotation": str(annotation), "missing": True}
    rows = json.loads(annotation.read_text())
    question_lengths = [len(re.sub(r"\\s+", " ", str(row.get("question", "")).replace("Question:", "", 1)).strip()) for row in rows]
    return {
        "annotation": str(annotation),
        "records": len(rows),
        "records_with_question": sum(bool(row.get("question")) for row in rows),
        "records_with_objects": sum(bool(row.get("objects")) for row in rows),
        "question_length_min_max": [min(question_lengths), max(question_lengths)] if question_lengths else None,
        "clean_builder_policy": "use normalized exact record.question; never replace it with the selected target",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows_path = Path(manifest["rows_path"])
    row_count = 0
    source_counts: collections.Counter[str] = collections.Counter()
    dataset_paths: dict[str, set[str]] = collections.defaultdict(set)
    mot20_05_rows: list[str] = []
    rmot_placeholder_rows: list[str] = []
    row_ids: list[str] = []
    for line in rows_path.open():
        row = json.loads(line)
        row_count += 1
        row_ids.append(str(row["id"]))
        source = str(row.get("source", ""))
        dataset = source.split(":", 1)[0]
        source_counts[source] += 1
        for image in row.get("images", []):
            path = str(image.get("path", ""))
            dataset_paths[dataset].add(str(Path(path).resolve()))
            if "MOT20-05" in path or "MOT20-05" in str(row["id"]):
                mot20_05_rows.append(str(row["id"]))
        text = " ".join(str(message.get("content", "")) for message in row.get("messages", []))
        if "the selected target" in text.lower():
            rmot_placeholder_rows.append(str(row["id"]))
    overlaps: dict[str, list[str]] = {}
    datasets = sorted(dataset_paths)
    for index, left in enumerate(datasets):
        for right in datasets[index + 1 :]:
            shared = sorted(dataset_paths[left] & dataset_paths[right])
            if shared:
                overlaps[f"{left}__{right}"] = shared[:100]
    from locatemot.unified.data.legal_scope import LEGAL_SPLITS

    result: dict[str, Any] = {
        "status": "CONTAMINATION_CONFIRMED_MOT20_05" if mot20_05_rows else "NO_MOT20_05_ROWS",
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "rows_path": str(rows_path.resolve()),
        "rows_sha256": sha256(rows_path),
        "row_count": row_count,
        "unique_row_ids": len(set(row_ids)),
        "source_counts": dict(sorted(source_counts.items())),
        "mot20_05_rows": {"count": len(mot20_05_rows), "first_ids": mot20_05_rows[:20], "heldout_scope": "MOT20-05 G0 long export"},
        "rmot26_placeholder_rows": {"count": len(rmot_placeholder_rows), "first_ids": rmot_placeholder_rows[:20]},
        "path_overlaps": overlaps,
        "mot17": audit_mot(Path("/data2/user/LocateMOT/data/MOT17"), set()),
        "mot20": audit_mot(Path("/data2/user/LocateMOT/data/MOT20"), {"MOT20-05"}),
        "tao": audit_tao(Path("/data2/user/LocateMOT/data/TAO-Amodal/annotations/train.json")),
        "refer_kitti": audit_refer(Path("/data2/user/LocateMOT/data/refer_kitti_v2/expression"), set(LEGAL_SPLITS["refer_kitti_v2"]["fit"])),
        "rmot26": audit_rmot26(Path("/data3/user/LocateMOT_FullDataV2/downloads/RMOT26/train/annotations.json")),
        "heldout_boundaries": {
            "mot20": ["MOT20-05"],
            "refer_kitti_v2": list(LEGAL_SPLITS["refer_kitti_v2"]["validation"]),
            "burst": "excluded until legal TRAIN image root and TAO overlap audit are available",
        },
        "clean_manifest_required": bool(mot20_05_rows),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
