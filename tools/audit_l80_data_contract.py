#!/usr/bin/env python3
"""P0 L80 data/freeze/causal-index audit; no model or training is run."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
sys.path.insert(0, str(ROOT))
from locatemot.rmot.l80_data import (  # noqa: E402
    ALL_L69_VIDEOS, CATEGORIES, DATASETS, EXPECTED_MANIFEST_SHA, FIT_VIDEOS,
    FORBIDDEN_LABEL_FIELDS, L80BankStore, MANIFEST, load_fit_units,
    load_fixed_key_units, load_full_unit_for_labels, sha256_file,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def key_only(row: dict) -> dict:
    allowed = ("unit_key", "dataset", "video", "query_id", "frame_id", "sentence", "expression", "evaluation_partition")
    result = {key: row[key] for key in allowed if key in row}
    if FORBIDDEN_LABEL_FIELDS.intersection(result):
        raise AssertionError(f"label field in pre-feature row {row.get('unit_key')}")
    return result


def summarize(batch, labels: dict | None, source: str) -> dict:
    candidate_indices = [int(x) for x in batch.candidate_indices]
    duplicate_indices = len(candidate_indices) - len(set(candidate_indices))
    pools = Counter(int(x) for x in batch.pool_ids)
    record = {
        "format": "locatemot-l80-data-contract-unit-v1", "status": "complete",
        "source": source, "unit_key": batch.unit_key, "dataset": batch.dataset,
        "video": batch.video, "query_id": int(batch.query_id), "frame_id": int(batch.frame_id),
        "bank_path": str(batch.bank_path), "image_path": str(batch.image_path),
        "candidate_count": int(batch.candidate_count), "row_offsets": [int(x) for x in batch.row_offsets],
        "row_keys": [list(x) for x in batch.row_keys], "candidate_index_provenance": candidate_indices,
        "track_id_provenance": [int(x) for x in batch.track_ids],
        "pool_id_provenance": [int(x) for x in batch.pool_ids],
        "duplicate_candidate_index_count": int(duplicate_indices), "pool_counts": dict(pools),
        "history_shape": list(batch.history_observations.shape),
        "history_valid_rows": int(batch.history_mask.sum()),
        "history_future_rows": int((batch.history_frame_ids > int(batch.frame_id)).sum()),
        "text_shape": list(batch.text_tokens.shape), "text_valid_tokens": int(batch.text_mask.sum()),
        "image_size": list(batch.image_size), "image_exists": bool(Path(batch.image_path).is_file()),
        "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
        "old_l49_ranges_used": False, "old_l49_positive_indices_used": False,
        "ids_are_provenance_only": True, "preselection_label_fields_absent": True,
    }
    if labels is not None:
        record.update({"category": str(labels["category"]), "positive_count": int(labels["positive_count"]),
                       "target_present": bool(labels["target_present"]), "candidate_present": bool(labels["candidate_present"]),
                       "coverage_mask": bool(labels["coverage_mask"]), "labels_attached_after_feature_construction": True})
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty output {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command = " ".join([sys.executable] + sys.argv)
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd {Path.cwd()}")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA changed")
    fit = load_fit_units()
    sample = []
    for dataset in DATASETS:
        for category in CATEGORIES:
            values = sorted((row for row in fit if row["dataset"] == dataset and row["category"] == category),
                            key=lambda row: (str(row["video"]), int(row["query_id"]), int(row["frame_id"]), str(row["unit_key"])))
            sample.extend(values[:3])
    if len(sample) != 24:
        raise AssertionError(f"stratified fit sample drift {len(sample)}")
    fixed = load_fixed_key_units()
    raw_rows = []
    label_rows = []
    store = L80BankStore(max_history=8)
    try:
        for source, metadata_rows in (("fit_contract_sample", sample), ("fixed_40_contract", fixed)):
            for metadata in metadata_rows:
                clean = key_only(metadata)
                batch = store.build_unit(clean)
                raw = summarize(batch, None, source)
                raw_rows.append(raw)
                if source == "fixed_40_contract":
                    # Explicit post-construction boundary: only now are the
                    # expression target and sidecar candidate labels loaded.
                    labels = store.attach_labels(batch, load_full_unit_for_labels(batch.unit_key))
                    label_rows.append(summarize(batch, labels, source))
                del batch
    finally:
        store._store._bank = None
        store._store._text_cache = None
    if len(label_rows) != 40:
        raise AssertionError("fixed label audit row count drift")
    if any(row["history_future_rows"] != 0 for row in raw_rows):
        raise AssertionError("future history detected")
    if any(row["candidate_count"] != len(row["row_offsets"]) for row in raw_rows):
        raise AssertionError("candidate row count drift")
    (out / "unit_records.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in raw_rows))
    (out / "fixed_labeled_records.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in label_rows))
    contract = {
        "format": "locatemot-l80-data-contract-v1", "status": "complete", "command": command,
        "inputs": {"manifest": str(MANIFEST), "manifest_sha256": sha256_file(MANIFEST),
                    "l69_feature_root": str((ROOT / "outputs/l69/attempt9/budget40_features/kitti").resolve()),
                    "train_units": str((ROOT / "outputs/l49/data/train_units.jsonl").resolve()),
                    "fixed_score_records": str((ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl").resolve())},
        "outputs": [str(out / "unit_records.jsonl"), str(out / "fixed_labeled_records.jsonl")],
        "fit_sample_units": len(sample), "fixed_units": len(fixed), "calibration_units": 16,
        "validation_units": 24, "fit_unit_total": len(fit), "fit_videos": list(FIT_VIDEOS),
        "l69_videos_allowed": list(ALL_L69_VIDEOS), "obs_dim": 1432, "history_length": 8,
        "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
        "duplicate_candidate_indices_retained": True, "old_l49_ranges_used": False,
        "old_l49_positive_indices_used": False, "history_future_rows": 0,
        "labels_loaded_after_complete_feature_record": True,
        "preselection_forbidden_label_fields_absent": True,
        "forbidden_label_fields": sorted(FORBIDDEN_LABEL_FIELDS),
        "same_class_hard_negative_metadata": "unavailable; all-negative query-independent fallback",
        "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "training_run": False, "hota_trackeval_run": False,
        "failure_root_cause": None, "next_action": "run L80 forward/loss contract audit",
        "elapsed_sec": time.perf_counter() - started,
    }
    write_json(out / "contract.json", contract)
    write_json(out / "provenance.json", {"format": "locatemot-l80-data-provenance-v1", "status": "complete",
        "command": command, "cwd": str(Path.cwd().resolve()), "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745",
        "manifest_sha256": sha256_file(MANIFEST), "fixed_order": [row["unit_key"] for row in fixed],
        "source_rows": {"fit_sample": len(sample), "fixed": len(fixed)},
        "labels_attached_only_after_build": True, "screening_gt_used": False,
        "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
        "training_run": False, "hota_trackeval_run": False})
    write_json(out / "status.json", {"format": "locatemot-l80-status-v1", "status": "complete", "command": command,
        "inputs": [str(MANIFEST), str(ROOT / "outputs/l49/data/train_units.jsonl")],
        "outputs": [str(out / "contract.json"), str(out / "unit_records.jsonl"), str(out / "fixed_labeled_records.jsonl")],
        "failure_root_cause": None, "next_action": "run L80 forward/loss contract audit",
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "training_run": False, "hota_trackeval_run": False})
    print(json.dumps(contract, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
