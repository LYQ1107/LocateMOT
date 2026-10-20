#!/usr/bin/env python3
"""Audit joint GroundJEV rows against the protected G2 TRAIN manifest.

This is a path/scope audit only.  It does not read evaluation labels and does
not decide that an overlapping TRAIN row is independent evidence.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping


def _norm(path: str) -> str:
    return os.path.realpath(path) if os.path.isabs(path) else os.path.normpath(path)


def _jsonl_rows(path: Path):
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_number}: row must be an object")
                yield row


def _g2_media(row: Mapping[str, Any]) -> list[str]:
    media = row.get("media") or row.get("images") or []
    return [str(item["path"]) for item in media
            if isinstance(item, Mapping) and item.get("path")]


def audit(joint_rows: Path, g2_rows: Path, *, g2_manifest: Path | None = None) -> dict[str, Any]:
    g2_paths: dict[str, str] = {}
    for row in _jsonl_rows(g2_rows):
        source = str(row.get("source", "unknown"))
        for media_path in _g2_media(row):
            g2_paths.setdefault(_norm(media_path), source)

    by_task: Counter[str] = Counter()
    by_g2_source: Counter[str] = Counter()
    overlap_examples: list[dict[str, Any]] = []
    joint_rows_count = 0
    joint_images: set[str] = set()
    overlap_rows = 0
    for row in _jsonl_rows(joint_rows):
        joint_rows_count += 1
        image = row.get("current_image")
        if not image:
            continue
        normalized = _norm(str(image))
        joint_images.add(normalized)
        source = g2_paths.get(normalized)
        if source is None:
            continue
        overlap_rows += 1
        task = str(row.get("task", "unknown"))
        by_task[task] += 1
        by_g2_source[source] += 1
        if len(overlap_examples) < 12:
            overlap_examples.append({
                "id": row.get("id"), "task": task,
                "image": str(image), "g2_source": source,
            })

    manifest = {}
    if g2_manifest is not None:
        manifest = json.loads(g2_manifest.read_text())
    return {
        "joint_rows": joint_rows_count,
        "joint_unique_images": len(joint_images),
        "g2_unique_media_paths": len(g2_paths),
        "exact_image_overlap_rows": overlap_rows,
        "exact_image_overlap_unique_images": len(joint_images & set(g2_paths)),
        "overlap_by_joint_task": dict(sorted(by_task.items())),
        "overlap_by_g2_source": dict(sorted(by_g2_source.items())),
        "overlap_examples": overlap_examples,
        "g2_manifest": str(g2_manifest.resolve()) if g2_manifest else None,
        "g2_manifest_sha256": manifest.get("rows_sha256"),
        "official_test_labels_read": False,
        "independent_generalization_claim_allowed": overlap_rows == 0,
        "scope_statement": (
            "Joint rows with exact image overlap are TRAIN-scope controls; they cannot "
            "support an independent G2 generalization claim."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-rows", required=True)
    parser.add_argument("--g2-rows", required=True)
    parser.add_argument("--g2-manifest")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = audit(Path(args.joint_rows), Path(args.g2_rows),
                    g2_manifest=Path(args.g2_manifest) if args.g2_manifest else None)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
