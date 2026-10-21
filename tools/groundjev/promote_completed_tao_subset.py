#!/usr/bin/env python3
"""Promote only complete rows from an interrupted registered TAO stream.

The native TAO stream writes and flushes one JSONL record per completed frame.
When a long stream is intentionally stopped, this tool preserves the source
attempt and creates a new, explicitly registered subset from complete records
only.  It never fills a missing frame, changes a candidate box, reads labels,
or presents the promoted subset as full TAO coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_record(record: dict[str, Any], expected_path: str, source: Path) -> None:
    if record.get("image_path") != expected_path:
        raise ValueError(
            f"{source}: row path does not match registered manifest: "
            f"{record.get('image_path')!r} != {expected_path!r}"
        )
    perception = record.get("perception", {})
    if perception.get("GT_read") is not False:
        raise ValueError(f"{source}: candidate row does not prove GT-free inference")
    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"{source}: candidate list is missing")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"{source}: candidate {index} is not an object")
        if "native_region_feature" not in candidate:
            raise ValueError(f"{source}: candidate {index} has no native region feature")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    selected: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    for shard_manifest_path in sorted(source_root.glob("shard_*_of_*/manifest.json")):
        shard_dir = shard_manifest_path.parent
        manifest = read_json(shard_manifest_path)
        candidate_path = shard_dir / "native_candidates" / "frame_candidates.jsonl"
        if not candidate_path.is_file():
            continue
        lines = [line for line in candidate_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            continue
        expected_images = list(manifest.get("images", []))
        if len(lines) > len(expected_images):
            raise ValueError(f"{candidate_path}: more rows than its registered shard")
        shard_rows = []
        for row_index, line in enumerate(lines):
            record = json.loads(line)
            validate_record(record, expected_images[row_index], candidate_path)
            shard_rows.append(record)
        source_rows.append({
            "shard": shard_dir.name,
            "candidate_archive": str(candidate_path),
            "candidate_archive_sha256": sha256(candidate_path),
            "complete_rows": len(shard_rows),
            "registered_rows": len(expected_images),
        })
        selected.extend(shard_rows)
    if not selected:
        raise ValueError("no complete native candidate rows were found")

    out.mkdir(parents=True)
    archive = out / "frame_candidates.jsonl"
    with archive.open("w", encoding="utf-8") as stream:
        for record in selected:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    images = [record["image_path"] for record in selected]
    # Every source shard was built from one frame per video.  The row-level
    # records are the authoritative promoted scope; no annotation is opened.
    subset_manifest = {
        "status": "REGISTERED_NATIVE_TAO_PARTIAL_SUBSET_SCOPE",
        "dataset": "tao_amodal",
        "task": "ovmot",
        "scope": "train",
        "registered_train_subset": True,
        "full_train_image_scope": False,
        "training_on_this_scope": True,
        "official_test_labels_read": False,
        "runtime_GT_fields": False,
        "vocabulary_size": 1203,
        "all1203classes_requested_on_every_frame": True,
        "frames": len(selected),
        "videos": len(selected),
        "images": images,
        "source_attempt": str(source_root),
        "source_shards": source_rows,
        "candidate_archive": str(archive),
        "candidate_archive_sha256": sha256(archive),
        "partial_stream_policy": "promote complete flushed rows only; retain source attempt; no missing-row fill",
    }
    (out / "subset_manifest.json").write_text(
        json.dumps(subset_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    receipt = {
        "status": "COMPLETE_REGISTERED_NATIVE_TAO_PARTIAL_SUBSET_STREAM",
        "scope_root": str(out),
        "candidate_archive": str(archive),
        "candidate_archive_sha256": sha256(archive),
        "frames": len(selected),
        "videos": len(selected),
        "registered_train_subset": True,
        "full_train_image_scope": False,
        "source_attempt_preserved": str(source_root),
        "source_shards": source_rows,
        "GT_or_Novel_labels_read_by_model": False,
        "end_to_end_tracking_FPS_claim": False,
        "promoted_complete_rows_only": True,
    }
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
