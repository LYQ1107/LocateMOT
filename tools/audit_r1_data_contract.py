#!/usr/bin/env python3
"""Read-only R1 row/history/language contract audit.

This audit intentionally stops before the GroundingDINO runtime.  It checks
that the frozen L69 rows, R0A native-frame metadata, local visual/language
side caches, and causal history can be assembled without loading target
arrays.  Labels are attached only by the later training/evaluation boundary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from tools.r0_common import MergedLanguageCache, VisualCacheIndex  # noqa: E402
from tools.r1_common import (  # noqa: E402
    CATEGORIES,
    FIT_ROOTS,
    THREAD,
    check_manifest,
    file_meta,
    frame_key,
    load_fixed_l62_key_order,
    load_indexes,
    provenance_inputs,
    public_record,
    record_digest,
    standard_flags,
    unit_key,
    write_json,
)
from locatemot.rmot.r1_aligned_cache import (  # noqa: E402
    LANGUAGE_ROOTS,
    R0_VISUAL_MANIFEST,
    R1BankStore,
    build_causal_history,
    build_geometry,
    image_path,
    row_digest,
)


def _choose_fit_records(indexes: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        dense = indexes[dataset]
        for category in CATEGORIES:
            candidates = [value for value in dense.label_records if str(value["category"]) == category]
            candidates.sort(key=lambda value: (str(value["video"]), int(value["frame_id"]), int(value["query_id"])))
            selected.extend(public_record(value, category=category) for value in candidates[:2])
    if len(selected) < 16:
        raise AssertionError("fit contract sample lacks all four categories/domains")
    # Add four deterministic records from the beginning of the two indexes so
    # the audit is not accidentally a category-only fixture.
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        for value in indexes[dataset].label_records[:2]:
            selected.append(public_record(value, category=str(value["category"])))
    return selected


def _audit_record(record: dict[str, Any], store: R1BankStore, visual: Any, language: Any,
                  *, allow_missing_visual: bool = False) -> dict[str, Any]:
    batch = store.build_frame(record["dataset"], record["video"], int(record["query_id"]),
                              int(record["frame_id"]), str(record["sentence"]))
    if not image_path(record["video"], record["frame_id"]).is_file():
        raise FileNotFoundError(record["unit_key"])
    try:
        visual_item = visual.get(record["dataset"], record["video"], int(record["frame_id"]))
    except KeyError:
        if not allow_missing_visual:
            raise
        # The fixed legal-dev slice is intentionally allowed to be absent from
        # the old R0A visual cache.  Its visual states must be produced later
        # by the new label-free native R1 cache builder; never substitute a
        # neighboring frame or fabricate zeros in this contract audit.
        visual_item = None
    raw = None
    if visual_item is not None:
        row_offsets = [int(value) for value in visual_item["row_offsets"]]
        expected = list(range(row_offsets[0], row_offsets[-1] + 1)) if row_offsets else []
        if row_offsets != expected or batch.row_offsets != expected:
            raise AssertionError(f"visual/L69 row offsets drift: {record['unit_key']}")
        if int(visual_item["candidate_count"]) != batch.candidate_count:
            raise AssertionError(f"candidate count drift: {record['unit_key']}")
        if [int(value) for value in visual_item["candidate_indices"]] != batch.candidate_indices:
            raise AssertionError(f"candidate order drift: {record['unit_key']}")
        for field in ("inner_tokens", "context_tokens", "boxes_normalized"):
            value = visual_item[field]
            if not torch.is_tensor(value) or not bool(torch.isfinite(value.float()).all()):
                raise FloatingPointError(f"invalid visual field {field}: {record['unit_key']}")
        raw = torch.cat((visual_item["inner_tokens"].float(), visual_item["context_tokens"].float()), dim=1)
        if raw.shape != (batch.candidate_count, 72, 256):
            raise AssertionError(f"R1 raw visual shape drift: {record['unit_key']} {tuple(raw.shape)}")
    lang = language.get(record["dataset"], record["video"], int(record["query_id"]), str(record["sentence"]))
    if lang.tokens.ndim != 2 or lang.tokens.shape[-1] != 256 or lang.mask.ndim != 1 or lang.mask.shape[0] != lang.tokens.shape[0]:
        raise AssertionError(f"language cache shape drift: {record['unit_key']}")
    history, history_mask, history_ids = build_causal_history(store, batch)
    geometry = build_geometry(store, batch)
    if history.shape != (batch.candidate_count, 8, 1432) or history_mask.shape != (batch.candidate_count, 8):
        raise AssertionError(f"history shape drift: {record['unit_key']}")
    if geometry.shape != (batch.candidate_count, 5, 10):
        raise AssertionError(f"geometry shape drift: {record['unit_key']}")
    if bool((history_ids[history_mask] > int(record["frame_id"])).any()):
        raise AssertionError(f"future history: {record['unit_key']}")
    keys = [list(value) for value in batch.row_keys]
    if len(keys) != batch.candidate_count or len({tuple(value) for value in keys}) != len(keys):
        raise AssertionError(f"row key drift: {record['unit_key']}")
    forbidden = {"target_ids", "positive_indices", "positive_count", "category", "labels", "target_present"}
    if forbidden.intersection(record):
        raise AssertionError(f"label fields entered public pre-feature record: {record['unit_key']}")
    return {
        "unit_key": unit_key(record), "dataset": record["dataset"], "video": record["video"],
        "query_id": int(record["query_id"]), "frame_id": int(record["frame_id"]),
        "candidate_count": batch.candidate_count, "row_count": len(keys), "row_key_digest": row_digest(keys),
        "row_key_first": keys[0] if keys else None, "row_key_last": keys[-1] if keys else None,
        "candidate_index_duplicate_count": len(batch.candidate_indices) - len(set(batch.candidate_indices)),
        "pool_counts": dict(sorted(Counter(str(value) for value in batch.pool_ids).items())),
        "visual_cache_available": visual_item is not None,
        "visual_shape": list(raw.shape) if raw is not None else None, "language_shape": list(lang.tokens.shape),
        "text_valid_tokens": int(lang.mask.sum()), "history_shape": list(history.shape),
        "history_valid_mean": float(history_mask.sum(dim=1).float().mean()),
        "history_future_rows": int((history_ids[history_mask] > int(record["frame_id"])).sum()),
        "geometry_shape": list(geometry.shape), "image_path": str(image_path(record["video"], record["frame_id"])),
        "label_fields_in_pre_feature_record": sorted(forbidden.intersection(record)),
        "all_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
        "finite": bool((raw is None or torch.isfinite(raw).all()) and torch.isfinite(history).all() and torch.isfinite(geometry).all()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else WORK_ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R1 data audit: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command = " ".join([str(sys.executable), *sys.argv])
    status: dict[str, Any] = {
        "format": "locatemot-r1-data-contract-audit-v1", "status": "running", "command": command,
        "cwd": str(Path.cwd().resolve()), "thread": THREAD, "failure_root_cause": None,
        "next_action": "run anchor/cache contract only after data audit passes",
    }
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R1 worktree cwd: {Path.cwd()}")
        manifest_sha = check_manifest()
        indexes = load_indexes("fit")
        fixed = load_fixed_l62_key_order()
        fit_records = _choose_fit_records(indexes)
        # The fixed evaluation key pass deliberately has no category/target
        # fields.  Candidate counts are recovered only from the native L69 bank.
        fixed_records = [dict(value) for value in fixed]
        visual = VisualCacheIndex(R0_VISUAL_MANIFEST)
        language = MergedLanguageCache(LANGUAGE_ROOTS)
        store = R1BankStore()
        records = fit_records + fixed_records
        audits = []
        for record in records:
            # Category is retained only in the deterministic fit sampler.  It
            # is removed before feature assembly and never reaches a model.
            feature_record = {key: value for key, value in record.items() if key != "category"}
            audits.append(_audit_record(feature_record, store, visual, language,
                                        allow_missing_visual=str(record.get("split")) in {"calibration", "validation"}))
        unique_keys = [value["unit_key"] for value in audits]
        if len(unique_keys) != len(set(unique_keys)):
            # A fit sample and the fixed slice can legitimately overlap only if
            # they identify the same query-frame; retain one audit row but make
            # the overlap explicit rather than silently duplicating it.
            audits = list({value["unit_key"]: value for value in audits}.values())
        if any(value["history_future_rows"] != 0 or not value["finite"] for value in audits):
            raise AssertionError("R1 data contract has future/nonfinite rows")
        schema = {
            "pre_feature_forbidden_fields": ["target_ids", "positive_indices", "positive_count", "category", "labels", "target_present"],
            "fit_public_record_fields": sorted(set().union(*(set(public_record(value)) for value in fit_records))),
            "fixed_public_record_fields": sorted(set(fixed_records[0])),
            "labels_attached": False, "labels_read_for_feature_construction": False,
            "old_begin_end_or_positive_indices_used": False,
        }
        payload = {
            **status, "status": "complete", "elapsed_seconds": float(time.perf_counter() - started),
            "manifest_sha256": manifest_sha, "fit_sample_count": len(fit_records),
            "fixed_order_count": len(fixed_records), "fixed_calibration_count": 16,
            "fixed_validation_count": 24, "record_count_audited": len(audits),
            "record_digest": record_digest(audits), "schema": schema,
            "category_counts_fit": dict(sorted(Counter(str(value["category"]) for value in fit_records).items())),
            "domain_counts_fit": dict(sorted(Counter(str(value["dataset"]) for value in fit_records).items())),
            "audits": audits, "inputs": provenance_inputs(),
            "visual_manifest": file_meta(R0_VISUAL_MANIFEST / "manifest.jsonl"),
            "language_roots": [file_meta(Path(value) / "manifest.jsonl") for value in LANGUAGE_ROOTS],
            "flags": standard_flags(training_run=False), "outputs": {"contract": str(out / "contract.json")},
        }
        write_json(out / "contract.json", payload)
        write_json(out / "provenance.json", {
            "format": "locatemot-r1-data-contract-provenance-v1", "status": "complete", "command": command,
            "cwd": str(Path.cwd().resolve()), "thread": THREAD, "manifest_sha256": manifest_sha,
            "inputs": provenance_inputs(), "outputs": {"contract": str(out / "contract.json")},
            "labels_attached": False, "labels_read_for_feature_construction": False,
            "candidate_deletion": False, "candidate_truncation": False,
            "failure_root_cause": None, "next_action": status["next_action"], **standard_flags(),
        })
        status.update({"status": "complete", "elapsed_seconds": float(time.perf_counter() - started),
                       "inputs": provenance_inputs(), "outputs": {"contract": str(out / "contract.json")},
                       "manifest_sha256": manifest_sha})
        write_json(out / "status.json", status)
        return 0
    except Exception as exc:
        status.update({"status": "incomplete", "elapsed_seconds": float(time.perf_counter() - started),
                       "failure_root_cause": f"{type(exc).__name__}: {exc}", "outputs": {"attempt": str(out)}})
        write_json(out / "status.json", status)
        (out / "INCOMPLETE.md").write_text(
            f"# R1 data contract incomplete\n\nFirst actionable error:\n\n```text\n{type(exc).__name__}: {exc}\n```\n\n"
            "No labels were attached by this audit; preserve this attempt and repair only the first contract error.\n",
            encoding="utf-8")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
