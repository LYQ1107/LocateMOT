#!/usr/bin/env python3
"""Verify and merge completed TAO TRAIN native candidate shard streams."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TERMINAL_STREAM_STATUS = "COMPLETE_LEGAL_NATIVE_VOCABULARY_AND_ROI_STREAM"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return json.loads(path.read_text())


def merge(args: argparse.Namespace) -> dict:
    parent = args.parent_manifest.resolve()
    shard_root = args.shard_root.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    parent_payload = _load(parent)
    parent_sha = sha256(parent)
    if parent_payload.get("dataset") != "tao_amodal" or parent_payload.get("task") != "ovmot":
        raise ValueError("parent manifest is not TAO OVMOT")
    if parent_payload.get("training_on_this_scope") is not True:
        raise ValueError("parent manifest is not explicitly registered as TRAIN scope")
    images = list(parent_payload.get("images", []))
    if len(images) != len(set(images)):
        raise ValueError("parent manifest contains duplicate images")
    summary_path = shard_root / "shards_receipt.json"
    summary = _load(summary_path)
    shard_count = int(args.shards or summary.get("num_shards", 0))
    if shard_count < 1 or shard_count != int(summary.get("num_shards", shard_count)):
        raise ValueError("shard count does not match registration receipt")
    if summary.get("parent_manifest_sha256") != parent_sha:
        raise ValueError("shard registration parent SHA does not match the requested manifest")

    out.mkdir(parents=True)
    combined = out / "frame_candidates.jsonl"
    receipts = []
    seen = set()
    total_lines = 0
    try:
        with combined.open("w", encoding="utf-8") as destination:
            for shard_index in range(shard_count):
                shard_dir = shard_root / f"shard_{shard_index:02d}_of_{shard_count:02d}"
                manifest = _load(shard_dir / "manifest.json")
                registered = _load(shard_dir / "scope_receipt.json")
                receipt = _load(shard_dir / "receipt.json")
                if receipt.get("status") != TERMINAL_STREAM_STATUS:
                    raise ValueError(f"shard {shard_index} is not terminal: {receipt.get('status')}")
                if receipt.get("training_on_this_scope") is not True:
                    raise ValueError(f"shard {shard_index} is not marked TRAIN scope")
                shard_meta = manifest.get("shard", {})
                if shard_meta.get("parent_manifest_sha256") != parent_sha:
                    raise ValueError(f"shard {shard_index} parent SHA mismatch")
                start = int(shard_meta.get("start_index", -1))
                end = int(shard_meta.get("end_index_exclusive", -1))
                if not (0 <= start < end <= len(images)):
                    raise ValueError(f"shard {shard_index} has invalid range")
                expected_images = images[start:end]
                if manifest.get("images") != expected_images:
                    raise ValueError(f"shard {shard_index} image slice differs from parent")
                if registered.get("manifest_sha256") != sha256(shard_dir / "manifest.json"):
                    raise ValueError(f"shard {shard_index} manifest receipt SHA mismatch")
                candidate_path = shard_dir / "frame_candidates.jsonl"
                line_count = 0
                with candidate_path.open(encoding="utf-8") as source:
                    for line in source:
                        if not line.strip():
                            continue
                        record = json.loads(line)
                        path = str(record.get("image_path", ""))
                        expected = expected_images[line_count] if line_count < len(expected_images) else None
                        if path != expected:
                            raise ValueError(f"shard {shard_index} frame order/path mismatch at line {line_count + 1}")
                        perception = record.get("perception", {})
                        if perception.get("GT_read") is not False or perception.get("training_on_this_scope") is not True:
                            raise ValueError(f"shard {shard_index} contains non-GT-free runtime metadata")
                        if path in seen:
                            raise ValueError(f"duplicate image path across native shards: {path}")
                        seen.add(path)
                        destination.write(line if line.endswith("\n") else line + "\n")
                        line_count += 1
                if line_count != len(expected_images) or receipt.get("frames") != line_count:
                    raise ValueError(f"shard {shard_index} frame count mismatch: {line_count} vs {len(expected_images)}")
                total_lines += line_count
                receipts.append({
                    "shard_index": shard_index,
                    "frames": line_count,
                    "candidate_archive": str(candidate_path),
                    "candidate_archive_sha256": sha256(candidate_path),
                    "receipt_status": receipt["status"],
                })
    except Exception:
        # Preserve the failed output for diagnosis; callers can choose a fresh
        # output directory on retry rather than silently replacing evidence.
        raise

    if total_lines != len(images) or len(seen) != len(images):
        raise ValueError(f"merged coverage is incomplete: {total_lines}/{len(images)}")
    result = {
        "status": "COMPLETE_FULL_NATIVE_TAO_TRAIN_CANDIDATE_STREAM",
        "parent_manifest": str(parent),
        "parent_manifest_sha256": parent_sha,
        "candidate_archive": str(combined),
        "candidate_archive_sha256": sha256(combined),
        "shards": receipts,
        "frames": total_lines,
        "videos": len(set(parent_payload.get("video_ids", []))),
        "all1203classes_requested_on_every_frame": True,
        "GT_or_Novel_labels_read_by_model": False,
        "native_ovmot_training_rows": False,
        "optimizer_updates": 0,
        "next_step": "Run prepare_native_ovmot_train_join.py on this complete TRAIN archive; only its full-scope receipt can open the native OVMOT gate.",
    }
    (out / "receipt.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(merge(args), indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
