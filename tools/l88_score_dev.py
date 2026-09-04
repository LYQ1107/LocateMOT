#!/usr/bin/env python3
"""Score all registered even L88 checkpoints on the internal fit/dev groups.

This is deliberately separate from checkpoint selection: it writes complete
candidate rows and target labels after feature construction, but does not fit
thresholds or read the fixed semantic validation slice.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from l88_eval_common import (
    ASSET_ROOT, L88_CACHE, L85_CACHE, MANIFEST, MANIFEST_SHA, SEED, THREAD,
    EncoderCacheReader, L88ClipStore, build_label_free_group, checkpoint_info,
    load_checkpoint_into, make_runtime, release_group, score_label_free_group,
    sha256, write_json,
)
from l88_eval_metrics import fit_rule_set, metric


WORK_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=L88_CACHE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-tile", type=int, default=4)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L88 dev output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    runtime = store = reader = sidecar = None
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256(MANIFEST) != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        if int(args.query_tile) < 1:
            raise ValueError("query tile must be positive")
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        checkpoint_paths = sorted(args.checkpoint_dir.resolve().glob("checkpoint_l88_epoch*.pt"),
                                  key=lambda path: int(path.stem.split("epoch")[-1]))
        expected_epochs = list(range(2, 41, 2))
        actual_epochs = [int(path.stem.split("epoch")[-1]) for path in checkpoint_paths]
        if actual_epochs != expected_epochs:
            raise AssertionError(f"L88 even checkpoint contract drift: {actual_epochs}")
        reader = EncoderCacheReader(args.cache)
        store = L88ClipStore(L85_CACHE, load_cache_into_ram=False)
        dev_keys = [str(value) for value in store.dev_keys]
        if len(dev_keys) != 138:
            raise AssertionError(f"L88 dev group count drift: {len(dev_keys)}")
        records_path = out / "score_records.jsonl"
        summary_rows: list[dict[str, Any]] = []
        all_records_by_checkpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
        runtime, injector, base_digest = make_runtime(device)
        with records_path.open("w", encoding="utf-8") as handle:
            for checkpoint_path in checkpoint_paths:
                sidecar, package_info = load_checkpoint_into(runtime, injector, checkpoint_path, device)
                checkpoint_key = str(checkpoint_path.resolve())
                checkpoint_records: list[dict[str, Any]] = []
                for index, group_key in enumerate(dev_keys):
                    group = build_label_free_group(store, group_key, temporal_enabled=True)
                    try:
                        values = score_label_free_group(
                            group, runtime, reader, store,
                            # load_checkpoint_into returns a fresh frozen sidecar;
                            # keep it in the local variable for the group loop.
                            sidecar, package_info, device,
                            query_tile=int(args.query_tile), attach_group_labels=True,
                        )
                        for row in values:
                            if not row.get("labels_attached_after_feature_construction"):
                                raise AssertionError(f"L88 dev label timing drift: {row['unit_key']}")
                            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            checkpoint_records.append(row)
                    finally:
                        release_group(store, group)
                    if (index + 1) % 25 == 0:
                        print(f"[l88-dev] epoch={package_info['epoch']} group={index + 1}/{len(dev_keys)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
                if len(checkpoint_records) == 0 or len({str(row["unit_key"]) for row in checkpoint_records}) != len(checkpoint_records):
                    raise AssertionError(f"L88 dev record count/key drift epoch={package_info['epoch']}")
                all_records_by_checkpoint[checkpoint_key].extend(checkpoint_records)
                measured = metric(checkpoint_records, 0.0, -1.0, 0.0)
                summary_rows.append({"checkpoint": package_info, "record_count": len(checkpoint_records),
                                     "group_count": len(dev_keys), "unfitted_reference_metrics": measured,
                                     "candidate_rows_retained": True, "candidate_deletion": False,
                                     "candidate_truncation": False, "labels_scope": "fit/dev only"})
                del sidecar
                sidecar = None
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        cheap = {
            "format": "locatemot-l88-cheap-dev-score-v1", "status": "complete",
            "stage": "internal fit/dev checkpoint shortlist; thresholds not selected here",
            "checkpoint_count": len(summary_rows), "dev_group_count": len(dev_keys),
            "dev_record_count_per_checkpoint": [row["record_count"] for row in summary_rows],
            "checkpoint_summaries": summary_rows,
            "score_records": str(records_path.resolve()), "score_records_sha256": sha256(records_path),
            "base_detector_digest": base_digest, "adapter_target_manifest": injector.manifest(),
            "query_tile": int(args.query_tile), "cache_summary_sha256": reader.summary_sha256,
            "fit_dev_labels_only": True, "fixed_calibration_read": False, "fixed_validation_read": False,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "no_hota_or_trackeval": True, "groundingdino_lora_used": True,
            "groundingdino_backbone_trainable": False, "candidate_deletion": False,
            "candidate_truncation": False, "z1_representation_changed": True,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "wall_seconds": time.perf_counter() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "next_action": "fit registered Rule B/R/P on dev target-level metrics and shortlist at most five checkpoints",
        }
        write_json(out / "cheap_dev_scores.json", cheap)
        write_json(out / "provenance.json", {
            "format": "locatemot-l88-cheap-dev-provenance-v1", "status": "complete",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD, "seed": SEED,
            "inputs": {"checkpoint_dir": str(args.checkpoint_dir.resolve()), "l88_cache": str(args.cache.resolve()),
                       "l85_cache": str(L85_CACHE.resolve()), "manifest": str(MANIFEST),
                       "manifest_sha256": MANIFEST_SHA, "cache_summary_sha256": reader.summary_sha256},
            "outputs": [str(records_path.resolve()), str((out / "cheap_dev_scores.json").resolve())],
            "label_boundary": "complete label-free feature construction and score first; fit/dev labels then attached per row",
            "all_candidate_rows_scored": True, "candidate_deletion": False, "candidate_truncation": False,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "no_hota_or_trackeval": True, "z1_representation_changed": True,
        })
        write_json(out / "status.json", {"format": "locatemot-l88-cheap-dev-status-v1", "status": "complete",
                                         "checkpoint_count": len(summary_rows), "dev_group_count": len(dev_keys),
                                         "record_count": sum(row["record_count"] for row in summary_rows),
                                         "candidate_deletion": False, "candidate_truncation": False,
                                         "screening_gt_used": False, "official_test_labels_read": False,
                                         "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        return 0
    except Exception:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L88 cheap dev scoring — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {"format": "locatemot-l88-cheap-dev-status-v1", "status": "incomplete",
                                         "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                         "failure_root_cause": "first traceback in INCOMPLETE.md",
                                         "screening_gt_used": False, "official_test_labels_read": False,
                                         "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        raise
    finally:
        if runtime is not None:
            runtime.close()
        if store is not None:
            store.release_loaded_cache_items(); store.close()
        if sidecar is not None:
            del sidecar
        del reader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
