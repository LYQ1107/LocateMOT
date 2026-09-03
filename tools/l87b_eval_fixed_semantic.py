#!/usr/bin/env python3
"""Fixed 16-calibration/24-validation evaluation for corrected L87-B policy."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(os.environ.get(
    "LOCATEMOT_ASSET_ROOT", "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT"
)).resolve()
if str(WORK_ROOT) not in sys.path: sys.path.insert(0, str(WORK_ROOT))
if str(ASSET_ROOT) not in sys.path: sys.path.append(str(ASSET_ROOT))
sys.path.insert(0, str(WORK_ROOT / "locatemot" / "rmot"))

# The isolated worktree intentionally contains the L87 modules, while the
# historical L80/L29 compatibility model remains read-only in the asset root.
# Extend only the existing package search path; do not copy or modify it.
import locatemot.models as _locatemot_models  # noqa: E402
_asset_models = str(ASSET_ROOT / "locatemot" / "models")
if _asset_models not in _locatemot_models.__path__:
    _locatemot_models.__path__.append(_asset_models)

from locatemot.rmot.l80_data import L80BankStore, load_fixed_key_units  # noqa: E402
from l87_eval_policy import contract_summary, metric  # noqa: E402
from tools.l86_eval_fixed_semantic import (  # noqa: E402
    attach_labels, load_model, score_fixed_rows,
)
from tools.eval_l80_v12 import (  # noqa: E402
    L29_VALIDATION_CONTROL, immutable_control_thresholds, make_control_records,
    metric as l29_metric,
)

THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
L62_ROWS = ASSET_ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
DEFAULT_CACHE = ASSET_ROOT / "outputs/l85/features/fit_dev_eval_full_attempt2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L87-B semantic output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv]); started = time.perf_counter()
    try:
        if Path.cwd().resolve() != WORK_ROOT: raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256(MANIFEST) != MANIFEST_SHA: raise AssertionError("fixed manifest SHA drift")
        selection_path = args.selection.resolve(); selection = json.loads(selection_path.read_text())
        selected = selection["selected"]; checkpoint_info = selected["checkpoint_info"]
        checkpoint = Path(checkpoint_info["path"]).resolve()
        if sha256(checkpoint) != str(checkpoint_info["sha256"]): raise AssertionError("checkpoint SHA drift")
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
            torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        model, loaded = load_model(checkpoint, device)
        metadata = load_fixed_key_units()
        immutable = [json.loads(line) for line in L62_ROWS.read_text().splitlines() if line.strip()]
        if len(metadata) != 40 or len(immutable) != 40:
            raise AssertionError("fixed 40-unit count drift")
        if [str(row["unit_key"]) for row in metadata] != [str(row["unit_key"]) for row in immutable]:
            raise AssertionError("fixed L62 unit order drift")
        store = L80BankStore(max_history=8)
        preselection = score_fixed_rows(metadata, args.cache.resolve(), model, store, device)
        if len(preselection) != 40 or [int(row["fixed_eval_order"]) for row in preselection] != list(range(40)):
            raise AssertionError("L87-B fixed preselection order drift")
        forbidden = {name for row in preselection for name in (
            "target_ids", "positive_indices", "positive_count", "category", "labels", "target_present", "candidate_gt"
        ) if name in row}
        if forbidden: raise AssertionError(f"preselection label leakage: {sorted(forbidden)}")
        if not all(len(row["score"]) == int(row["candidate_count"]) == len(row["row_keys"]) for row in preselection):
            raise AssertionError("preselection candidate row drift")
        preselection_audit = {
            "format": "locatemot-l87b-preselection-label-isolation-v1", "status": "complete",
            "fixed_records": 40, "calibration_records": 16, "validation_records": 24,
            "schema": sorted(preselection[0].keys()), "forbidden_fields_absent": True,
            "native_order": [str(row["unit_key"]) for row in preselection],
            "candidate_counts": [int(row["candidate_count"]) for row in preselection],
            "score_finite": all(np.isfinite(np.asarray(row["score"], dtype=np.float64)).all() for row in preselection),
            "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "labels_loaded": False, "selection_frozen_before_labels": True,
            "validation_labels_not_loaded": True, "corrected_candidate_vs_null_rule": True,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        }
        write_json(out / "preselection_label_isolation.json", preselection_audit)
        rule = selected["rule_fit"]
        candidate = [attach_labels(row, store) for row in preselection]
        cal = candidate[:16]; val = candidate[16:]
        c = float(rule["candidate_threshold"]); p = float(rule["presence_threshold"]); margin = float(rule["null_margin"])
        cal_candidate = metric(cal, c, -float("inf"), -float("inf"))
        cal_final = metric(cal, c, p, margin)
        val_candidate = metric(val, c, -float("inf"), -float("inf"))
        val_final = metric(val, c, p, margin)
        control_records = make_control_records(); thresholds = immutable_control_thresholds()
        controls = {}
        for name, field in (("l29_teacher", "l29"), ("l53_m0", "m0"), ("l54_continuous", "m54")):
            controls[name] = {"source": str(L62_ROWS), "threshold": thresholds[name],
                              "calibration": l29_metric(control_records[:16], field, thresholds[name]),
                              "validation": l29_metric(control_records[16:], field, thresholds[name])}
        checks = {
            "target_bag_hard_violation": val_final["target_bag_hard_violation"] <= .8666667,
            "legacy_recall_floor": val_final["candidate_recall"] >= .7233333,
            "legacy_precision_floor": val_final["candidate_precision"] >= .0830188679,
            "legacy_fp_per_frame_floor": val_final["fp_per_frame"] <= 11.125,
            "legacy_predictions_per_positive_floor": val_final["predictions_per_positive"] <= 4.069,
            "multi_target_recall_floor": (val_final["multi_positive_recall"] is not None and val_final["multi_positive_recall"] >= .7894444),
            "inactive_false_acceptance_lt_1": val_final["inactive_false_acceptance"] < 1.0,
            "complete_finite_keys": len(val) == 24 and all(
                len(row["score"]) == int(row["candidate_count"]) == len(row["row_keys"]) and row["finite_scores"] for row in val),
            "candidate_deletion_false": all(not row["candidate_deletion"] and not row["candidate_truncation"] for row in preselection),
        }
        decision = "corrected_semantic_gate_pass_pending_supervisor" if all(checks.values()) else "corrected_semantic_gate_fail"
        semantic = {
            "format": "locatemot-l87b-fixed-semantic-v1", "status": "complete", "decision": decision,
            "evidence_type": "fixed 16-calibration/24-validation corrected target-bag semantic diagnostic",
            "command": command, "work_root": str(WORK_ROOT), "asset_root": str(ASSET_ROOT),
            "cwd": str(WORK_ROOT), "luna_thread": THREAD, "selected_checkpoint": loaded,
            "selection_source": str(selection_path), "selected_rule": rule,
            "calibration": {"candidate_only": cal_candidate, "corrected_rule": cal_final, "labels_used_for_selection": False},
            "validation": {"candidate_only": val_candidate, "corrected_rule": val_final, "labels_used_for_selection": False},
            "immutable_controls": controls, "l29_validation_control": L29_VALIDATION_CONTROL,
            "gate": {"checks": checks, "target_bag_rule": "unique referred target bags; background singleton negatives",
                      "emission_rule": "presence >= threshold AND candidate >= threshold AND candidate-null >= margin"},
            "candidate_rows": {"calibration": int(sum(row["candidate_count"] for row in cal)),
                               "validation": int(sum(row["candidate_count"] for row in val)), "all_rows_retained": True,
                               "candidate_deletion": False, "candidate_truncation": False},
            "label_events": {"preselection_complete_before_labels": True, "selection_frozen_before_labels": True,
                              "calibration_attached_after_selection": True, "validation_attached_after_calibration": True},
            "manifest_sha256": MANIFEST_SHA, "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False, "no_hota_or_trackeval": True,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
        }
        gate = {"format": "locatemot-l87b-fixed-gate-v1", "status": decision, "decision": decision,
                "checks": checks, "selected_checkpoint": loaded, "selected_rule": rule,
                "calibration_units": 16, "validation_units": 24, "screening_gt_used": False,
                "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                "hota_trackeval_run": False, "no_hota_or_trackeval": True}
        write_json(out / "semantic.json", semantic); write_json(out / "gate_decision.json", gate)
        with (out / "score_records.jsonl").open("w") as handle:
            for row in candidate: handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        provenance = {"format": "locatemot-l87b-fixed-semantic-provenance-v1", "status": "complete",
                      "command": command, "work_root": str(WORK_ROOT), "asset_root": str(ASSET_ROOT),
                      "cwd": str(WORK_ROOT), "luna_thread": THREAD, "seed": 20260829,
                      "inputs": {"cache": str(args.cache.resolve()), "cache_summary_sha256": sha256(args.cache / "summary.json"),
                                 "selection": str(selection_path), "selection_sha256": sha256(selection_path),
                                 "immutable_l62_rows": str(L62_ROWS), "immutable_l62_rows_sha256": sha256(L62_ROWS),
                                 "manifest": str(MANIFEST), "manifest_sha256": MANIFEST_SHA},
                      "selected_checkpoint": loaded, "selection_rule": rule,
                      "preselection_label_isolation": str(out / "preselection_label_isolation.json"),
                      "row_contract": contract_summary(preselection), "candidate_rows_retained": True,
                      "candidate_deletion": False, "candidate_truncation": False,
                      "screening_gt_used": False, "official_test_labels_read": False,
                      "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                      "no_hota_or_trackeval": True, "token_span_region_alignment": "UNALIGNED",
                      "static_motion_alignment": "UNALIGNED", "wall_seconds": time.perf_counter() - started}
        write_json(out / "semantic.json", semantic); write_json(out / "gate_decision.json", gate)
        write_json(out / "provenance.json", provenance); write_json(out / "status.json", {"format": "locatemot-l87b-fixed-semantic-status-v1",
            "status": decision, "selected_checkpoint": loaded, "selected_rule": rule,
            "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False, "no_hota_or_trackeval": True})
        print(json.dumps({"status": decision, "validation": val_final, "checks": checks}, indent=2), flush=True)
        return 0
    except Exception:
        trace = traceback.format_exc(); (out / "INCOMPLETE.md").write_text("# L87-B fixed semantic — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {"format": "locatemot-l87b-fixed-semantic-status-v1", "status": "incomplete",
                                         "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                         "failure_root_cause": "first traceback in INCOMPLETE.md", "screening_gt_used": False,
                                         "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                                         "hota_trackeval_run": False})
        raise


if __name__ == "__main__": raise SystemExit(main())
