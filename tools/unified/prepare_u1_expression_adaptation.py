#!/usr/bin/env python3
"""Prepare the legal-fit multi-target U1-B expression stream.

The source release has expression-level frame maps but no word spans.  Each
positive sample therefore uses the complete sentence as one positive phrase
and includes every visible referred box in that frame.  V2 frame references
with missing source target IDs are skipped and never converted into negatives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
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


def read_label(path: Path, width: int, height: int) -> dict[str, list[float]]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        x, y, w, h = (float(value) for value in fields[2:6])
        result[str(fields[1])] = [x * width, y * height, (x + w) * width, (y + h) * height]
    return result


def deterministic_pick(items: list[int], key: str) -> int | None:
    if not items:
        return None
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return items[int.from_bytes(digest[:8], "big") % len(items)]


def build_expression_rows(dataset: str, empty_fraction: float) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    stats = Counter()
    for raw_video in LEGAL_SPLITS[dataset]["fit"]:
        video = assert_legal_video(raw_video, dataset=dataset)
        expression_root = assert_legal_path(PROJECT_ROOT / "data" / dataset / "expression" / video)
        label_root = assert_legal_path(PROJECT_ROOT / "data" / dataset / "labels_with_ids" / "image_02" / video)
        image_root = assert_legal_path(PROJECT_ROOT / "data/kitti_tracking/training/image_02" / video)
        image_paths = sorted(image_root.glob("*.png"))
        all_frames = [int(path.stem) for path in image_paths]
        size_cache: dict[int, tuple[int, int]] = {}
        for expression_path in sorted(expression_root.glob("*.json")):
            payload = json.loads(expression_path.read_text(encoding="utf-8"))
            sentence = str(payload.get("sentence") or expression_path.stem).strip()
            target_map = {int(frame): [str(value) for value in (ids or [])] for frame, ids in payload.get("label", {}).items()}
            valid_positive: list[tuple[int, list[list[float]]]] = []
            for frame, target_ids in sorted(target_map.items()):
                label_path = assert_legal_path(label_root / f"{frame:06d}.txt")
                image_path = assert_legal_path(image_root / f"{frame:06d}.png")
                if not label_path.is_file() or not image_path.is_file():
                    stats["missing_label_frame"] += 1
                    continue
                if frame not in size_cache:
                    with Image.open(image_path) as image:
                        size_cache[frame] = image.size
                width, height = size_cache[frame]
                labels = read_label(label_path, width, height)
                missing = [target for target in target_ids if target not in labels]
                if missing:
                    stats["missing_source_target_reference"] += len(missing)
                    stats["skipped_frame_with_missing_target"] += 1
                    continue
                boxes = [labels[target] for target in target_ids]
                valid_positive.append((frame, boxes))
            if not valid_positive:
                stats["query_without_valid_positive"] += 1
                continue
            frame, boxes = valid_positive[len(valid_positive) // 2]
            image_path = assert_legal_path(image_root / f"{frame:06d}.png")
            width, height = size_cache[frame]
            rows.append({
                "filename": str(image_path.resolve()),
                "height": height,
                "width": width,
                "grounding": {
                    "caption": sentence,
                    "regions": [{"bbox": boxes, "phrase": sentence, "tokens_positive": [[0, len(sentence)]]}],
                },
                "dataset": dataset,
                "video": video,
                "expression": expression_path.stem,
                "row_kind": "positive",
                "target_count": len(boxes),
            })
            stats["positive_query_rows"] += 1
            stats["positive_target_boxes"] += len(boxes)
            stats["multi_target_positive_rows"] += int(len(boxes) > 1)

            # The original JSON lists only positive frames.  Select a legal
            # image frame not in the expression's positive map for roughly
            # one empty/absence sample per five query samples.
            empty_candidates = [candidate for candidate in all_frames if candidate not in target_map]
            digest = hashlib.sha256(f"{dataset}|{video}|{expression_path.stem}".encode()).digest()
            include_empty = (int.from_bytes(digest[:8], "big") % 10000) < int(empty_fraction * 10000)
            empty_frame = deterministic_pick(empty_candidates, f"empty|{dataset}|{video}|{expression_path.stem}") if include_empty else None
            if empty_frame is not None:
                empty_path = assert_legal_path(image_root / f"{empty_frame:06d}.png")
                if empty_frame not in size_cache:
                    with Image.open(empty_path) as image:
                        size_cache[empty_frame] = image.size
                empty_width, empty_height = size_cache[empty_frame]
                rows.append({
                    "filename": str(empty_path.resolve()),
                    "height": empty_height,
                    "width": empty_width,
                    "grounding": {"caption": sentence, "regions": []},
                    "dataset": dataset,
                    "video": video,
                    "expression": expression_path.stem,
                    "row_kind": "empty",
                    "target_count": 0,
                })
                stats["empty_rows"] += 1
    rows.sort(key=lambda row: (row["video"], row["expression"], row["row_kind"], row["filename"]))
    return rows, dict(stats)


def weighted_rows(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    if not rows or count <= 0:
        return []
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    return [rows[order[index % len(order)]] for index in range(count)]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "outputs/unified/u1r2")
    parser.add_argument("--mix-total", type=int, default=5000)
    parser.add_argument("--empty-fraction", type=float, default=0.20)
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)
    v1_rows, v1_stats = build_expression_rows("refer_kitti_v1", args.empty_fraction)
    v2_rows, v2_stats = build_expression_rows("refer_kitti_v2", args.empty_fraction)
    generic_payload = json.loads((args.out_root / "generic_train_manifest.json").read_text(encoding="utf-8"))
    generic_rows = generic_payload["rows"]
    n_generic = round(args.mix_total * 0.20)
    n_v1 = round(args.mix_total * 0.30)
    n_v2 = args.mix_total - n_generic - n_v1
    mix_generic = weighted_rows(generic_rows, n_generic, 701)
    mix_v1 = weighted_rows(v1_rows, n_v1, 702)
    mix_v2 = weighted_rows(v2_rows, n_v2, 703)
    write_jsonl(args.out_root / "expression_mix_generic.jsonl", mix_generic)
    write_jsonl(args.out_root / "expression_mix_v1.jsonl", mix_v1)
    write_jsonl(args.out_root / "expression_mix_v2.jsonl", mix_v2)
    manifest = {
        "format": "locatemot-u-u1r2-expression-adaptation-manifest-v1",
        "policy": {
            "source_scope": "legal fit only",
            "positive_sample": "one deterministic middle valid frame per expression, all visible referred boxes",
            "token_span": "whole expression [0, len(sentence)); no fabricated word-level spans",
            "empty_target_sampling": args.empty_fraction,
            "v2_missing_target_policy": "skip frame and do not create positive or false-negative loss",
            "mix": {"generic": 0.20, "refer_kitti_v1": 0.30, "refer_kitti_v2": 0.50},
            "mix_total_rows": args.mix_total,
        },
        "source_rows": {"generic": len(generic_rows), "refer_kitti_v1": len(v1_rows), "refer_kitti_v2": len(v2_rows)},
        "source_stats": {"refer_kitti_v1": v1_stats, "refer_kitti_v2": v2_stats},
        "mix_rows": {"generic": len(mix_generic), "refer_kitti_v1": len(mix_v1), "refer_kitti_v2": len(mix_v2)},
        "files": {"generic": str((args.out_root / "expression_mix_generic.jsonl").resolve()), "refer_kitti_v1": str((args.out_root / "expression_mix_v1.jsonl").resolve()), "refer_kitti_v2": str((args.out_root / "expression_mix_v2.jsonl").resolve())},
        "official_test_labels_read": False,
        "screening_gt_used": False,
        "training_started": False,
    }
    path = args.out_root / "expression_train_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
