#!/usr/bin/env python3
"""Finalize and contract-check a label-free L89D Z1 supplement."""
from __future__ import annotations

import argparse
import gc
import json
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from l89d_fullvideo_common import (
    ASSET_ROOT,
    BASE_Z1_CACHE,
    L80BankStore,
    MANIFEST_SHA,
    THREAD,
    WORK_ROOT,
    Z1CacheIndex,
    command_line,
    file_meta,
    load_video_scopes,
    manifest_assertion,
    native_groups,
    sha256_file,
    standard_flags,
    validate_native_batch,
    write_json,
)


FORMAT = "locatemot-l89d-dense-z1-supplement-v1"
FORBIDDEN = {
    "target_ids", "positive_indices", "positive_count", "category", "labels",
    "candidate_gt", "candidate_scores", "coverage_mask", "declared_category",
}


def _audit_missing(audit: Path, scope: str) -> tuple[dict[str, Any], set[str]]:
    coverage = json.loads((audit.resolve() / "coverage.json").read_text(encoding="utf-8"))
    if coverage.get("format") != "locatemot-l89d-dense-coverage-v1" or coverage.get("scope") != scope:
        raise AssertionError("coverage audit format/scope mismatch")
    if coverage.get("labels_read") or coverage.get("screening_gt_used") or coverage.get("official_test_labels_read"):
        raise AssertionError("coverage audit violated label boundary")
    missing = coverage.get("missing_groups", [])
    keys = {str(item["group_key"]) for item in missing}
    if len(keys) != len(missing):
        raise AssertionError("duplicate audited missing group")
    return coverage, keys


def _validate_item(item: dict[str, Any], group: dict[str, Any], batch: Any) -> dict[str, Any]:
    key = str(group["group_key"])
    if str(item.get("group_key")) != key:
        raise AssertionError(f"group key mismatch: {key}")
    if FORBIDDEN.intersection(item) or item.get("labels_in_cache"):
        raise AssertionError(f"labels in supplement item: {key}")
    if item.get("candidate_deletion") or item.get("candidate_truncation"):
        raise AssertionError(f"row-retention flag drift: {key}")
    expected_qids = [int(row["query_id"]) for row in group["queries"]]
    if [int(x) for x in item.get("query_ids", [])] != expected_qids:
        raise AssertionError(f"query order mismatch: {key}")
    if [str(x) for x in item.get("sentences", [])] != [str(row["sentence"]) for row in group["queries"]]:
        raise AssertionError(f"sentence mismatch: {key}")
    if int(item.get("candidate_count", -1)) != int(batch.candidate_count):
        raise AssertionError(f"candidate count mismatch: {key}")
    offsets = [int(x) for x in item.get("row_offsets", [])]
    if offsets != [int(x) for x in batch.row_offsets]:
        raise AssertionError(f"row offset mismatch: {key}")
    z1, text, frame = item.get("z1"), item.get("text_global"), item.get("frame_global")
    if not (torch.is_tensor(z1) and torch.is_tensor(text) and torch.is_tensor(frame)):
        raise AssertionError(f"missing supplement tensor: {key}")
    expected = (len(expected_qids), int(batch.candidate_count), 256)
    if tuple(z1.shape) != expected or tuple(text.shape) != (len(expected_qids), 256) or tuple(frame.shape) != (len(expected_qids), 256):
        raise AssertionError(f"supplement shape drift: {key}")
    if not all(bool(torch.isfinite(value.float()).all()) for value in (z1, text, frame)):
        raise FloatingPointError(f"nonfinite supplement tensor: {key}")
    return {"group_key": key, "query_count": len(expected_qids), "candidate_count": int(batch.candidate_count),
            "row_offsets": offsets, "z1_shape": list(z1.shape), "path": None}


def run(args: argparse.Namespace) -> int:
    root = args.cache_root.resolve()
    command = command_line()
    started = time.perf_counter()
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong L89D cwd: {Path.cwd()}")
        manifest = manifest_assertion()
        coverage, missing_keys = _audit_missing(args.audit, args.scope)
        if not missing_keys:
            raise AssertionError("finalizer was called although audit has no missing groups")
        if not root.is_dir():
            raise FileNotFoundError(root)
        existing_summary = root / "summary.json"
        if existing_summary.is_file():
            prior = json.loads(existing_summary.read_text(encoding="utf-8"))
            if prior.get("status") == "complete":
                raise FileExistsError(f"refusing to overwrite finalized supplement: {root}")
        index = Z1CacheIndex(root, require_complete_summary=False)
        if index.invalid:
            raise AssertionError(f"invalid supplement files: {index.invalid}")
        actual_keys = set(index.paths)
        extras = sorted(actual_keys - missing_keys)
        absent = sorted(missing_keys - actual_keys)
        if extras:
            raise AssertionError(f"supplement contains groups not in audit missing set: {extras[:5]}")
        if absent:
            raise AssertionError(f"supplement still missing audited groups: {absent[:5]}")

        store = L80BankStore(max_history=8)
        groups_by_key: dict[str, dict[str, Any]] = {}
        for video_scope in load_video_scopes(args.scope):
            groups = native_groups(video_scope, store)
            for group in groups:
                groups_by_key[str(group["group_key"])] = group
        if missing_keys - set(groups_by_key):
            raise AssertionError("audited missing group is outside the native timeline")
        rows: list[dict[str, Any]] = []
        total_bytes = 0
        for key in sorted(missing_keys):
            group = groups_by_key[key]
            first = store.build_unit(dict(group["queries"][0]))
            native = validate_native_batch(first)
            item, path = index.read(key)
            checked = _validate_item(item, group, first)
            checked.update({
                "dataset": str(group["dataset"]), "video": str(group["video"]),
                "frame_id": int(group["frame_id"]), "path": str(path.resolve()),
                "bytes": int(path.stat().st_size), "row_key_digest": native["row_key_digest"],
            })
            rows.append(checked); total_bytes += int(path.stat().st_size)
            del item, first, native
            gc.collect()
        rows.sort(key=lambda row: str(row["group_key"]))
        with (root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
        summary = {
            "format": FORMAT, "status": "complete", "scope": args.scope,
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "output_root": str(root), "item_root": str((root / "items").resolve()),
            "audit": str(args.audit.resolve()), "audit_sha256": sha256_file(args.audit.resolve() / "coverage.json"),
            "audit_missing_group_count": len(missing_keys), "finalized_group_count": len(rows),
            "extra_group_count": len(extras), "absent_group_count": len(absent),
            "bytes": total_bytes, "manifest_jsonl": str((root / "manifest.jsonl").resolve()),
            "items": rows, "cache_labels_included": False, "raw_pixels_in_cache": False,
            "dense_detector_maps_in_cache": False, "query_independent": True,
            "candidate_deletion": False, "candidate_truncation": False,
            "inputs": {"manifest": manifest, "base_z1_cache": file_meta(args.base_z1_cache / "summary.json"),
                       "audit_coverage": file_meta(args.audit.resolve() / "coverage.json")},
            **standard_flags(hota_trackeval_run=False), "failure_root_cause": None,
            "next_action": "rerun L89D dense coverage audits with this supplement",
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(root / "summary.json", summary)
        write_json(root / "provenance.json", summary)
        write_json(root / "status.json", {"format": FORMAT, "status": "complete", "scope": args.scope,
                                           "finalized_group_count": len(rows), "bytes": total_bytes,
                                           "screening_gt_used": False, "official_test_labels_read": False,
                                           "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                                           "zero_training": True, "new_checkpoint_created": False})
        store._store._bank = None; store._store._text_cache = None
        del store
        return 0
    except Exception as exc:
        root.mkdir(parents=True, exist_ok=True)
        (root / "INCOMPLETE.md").write_text("# L89D dense Z1 finalization — INCOMPLETE\n\n" + traceback.format_exc(), encoding="utf-8")
        write_json(root / "status.json", {"format": FORMAT, "status": "incomplete", "scope": args.scope,
                                           "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                           "failure_root_cause": f"{type(exc).__name__}: {exc}",
                                           "next_action": "repair first actionable finalization error and use a new attempt",
                                           **standard_flags(hota_trackeval_run=False)})
        raise
    finally:
        gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("dev", "internal"), required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--base-z1-cache", type=Path, default=BASE_Z1_CACHE)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
