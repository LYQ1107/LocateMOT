#!/usr/bin/env python3
"""Evaluate the frozen L86 dev-selected rule on the immutable 16/24 slice.

The fixed units are loaded key-only and scored before any calibration or
validation labels are attached.  The selected checkpoint/rule was frozen on
the separate internal dev split; this tool never changes it and never reads
screening or official-test labels.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
L62_ROWS = ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
DEFAULT_CACHE = ROOT / "outputs/l85/features/fit_dev_eval_full_attempt2"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from locatemot.models.l86_full_rmot import L86Config, L86FullRMOT  # noqa: E402
from locatemot.rmot.l80_data import (L80BankStore, load_fixed_key_units,
                                     load_full_unit_for_labels)  # noqa: E402
from tools.l86_select_checkpoint import metric as l86_metric  # noqa: E402
from tools.eval_l80_v12 import (  # noqa: E402
    L29_VALIDATION_CONTROL, immutable_control_thresholds, make_control_records, metric as l29_metric,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def load_model(path: Path, device: torch.device) -> tuple[L86FullRMOT, dict[str, Any]]:
    package = torch.load(path, map_location="cpu", weights_only=False)
    model = L86FullRMOT(L86Config(**package["model_config"]))
    result = model.load_state_dict(package["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"strict L86 reload failure: {result}")
    model.to(device=device, dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {"path": str(path.resolve()), "sha256": sha256(path),
                   "epoch": int(package.get("epoch", 0)), "step": int(package.get("step", 0)),
                   "model_config": package["model_config"], "strict_reload": True}


def load_fixed_cache_manifest(cache_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    with (cache_root / "manifest.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row["group_key"])
            path = Path(row["path"]).resolve()
            if key in result and result[key] != path:
                raise AssertionError(f"cache group duplicate: {key}")
            result[key] = path
    return result


def score_fixed_rows(metadata: list[dict[str, Any]], cache_root: Path, model: L86FullRMOT,
                     store: L80BankStore, device: torch.device) -> list[dict[str, Any]]:
    cache_by_group = load_fixed_cache_manifest(cache_root)
    records: list[dict[str, Any]] = []
    item_cache: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        for metadata_row in metadata:
            group_key = f"{metadata_row['dataset']}|{metadata_row['video']}|{int(metadata_row['frame_id'])}"
            path = cache_by_group.get(group_key)
            if path is None:
                raise KeyError(f"fixed cache group missing: {group_key}")
            if group_key not in item_cache:
                item_cache[group_key] = torch.load(path, map_location="cpu", weights_only=False)
            item = item_cache[group_key]
            qids = [int(value) for value in item["query_ids"]]
            qid = int(metadata_row["query_id"])
            if qid not in qids:
                raise KeyError(f"fixed query missing from cache item: {metadata_row['unit_key']}")
            q = qids.index(qid)
            batch = store.build_unit(metadata_row)
            if int(item["candidate_count"]) != batch.candidate_count or [int(x) for x in item["row_offsets"]] != batch.row_offsets:
                raise AssertionError(f"fixed cache/bank candidate contract drift: {metadata_row['unit_key']}")
            z1 = item["z1"][q:q + 1].float().clone().to(device)
            text = item["text_global"][q:q + 1].float().clone().to(device)
            frame_global = item["frame_global"][q:q + 1].float().clone().to(device)
            history = batch.history_observations.float().clone().to(device)
            history_mask = batch.history_mask.clone().to(device)
            history_frames = batch.history_frame_ids.clone().to(device)
            current = batch.observations.float().clone().to(device)
            if int((history_frames > int(batch.frame_id)).sum()) != 0:
                raise AssertionError(f"future history in fixed unit: {batch.unit_key}")
            output = model(z1, text, frame_global, current, history, history_mask, history_frames,
                           batch.frame_id, temporal_enabled=True)
            scores = output["candidate_energy"][0].float().cpu().numpy()
            r_total = output["r_total"][0].float().cpu().numpy()
            prior = output["candidate_prior"].float().cpu().numpy()
            presence = float(output["presence_logit"][0].float().cpu())
            null = float(output["null_logit"][0].float().cpu())
            if not all(np.isfinite(value).all() for value in (scores, r_total, prior)) or not np.isfinite([presence, null]).all():
                raise FloatingPointError(f"nonfinite fixed L86 output: {batch.unit_key}")
            row_keys = [list(value) for value in batch.row_keys]
            if len(scores) != batch.candidate_count or len(row_keys) != batch.candidate_count:
                raise AssertionError(f"fixed score row count drift: {batch.unit_key}")
            records.append({
                "format": "locatemot-l86-fixed-preselection-v1", "fixed_eval_order": int(metadata_row["fixed_eval_order"]),
                "fixed_eval_split": str(metadata_row["evaluation_partition"]), "unit_key": str(batch.unit_key),
                "dataset": str(batch.dataset), "video": str(batch.video), "query_id": int(batch.query_id),
                "frame_id": int(batch.frame_id), "group_key": group_key, "candidate_count": int(batch.candidate_count),
                "row_offsets": [int(x) for x in batch.row_offsets], "row_keys": row_keys,
                "candidate_indices": [int(x) for x in batch.candidate_indices], "track_ids": [int(x) for x in batch.track_ids],
                "score": scores.astype(np.float64).tolist(), "r_total": r_total.astype(np.float64).tolist(),
                "candidate_prior": prior.astype(np.float64).tolist(), "presence_logit": presence,
                "null_logit": null, "future_history_count": int((history_frames > batch.frame_id).sum()),
                "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
                "labels_attached": False, "finite_scores": True,
            })
            del output, batch, z1, text, frame_global, history, history_mask, history_frames, current
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return sorted(records, key=lambda row: int(row["fixed_eval_order"]))


def attach_labels(record: dict[str, Any], store: L80BankStore) -> dict[str, Any]:
    full = load_full_unit_for_labels(str(record["unit_key"]))
    key_only = {
        "unit_key": str(record["unit_key"]), "dataset": str(record["dataset"]), "video": str(record["video"]),
        "query_id": int(record["query_id"]), "frame_id": int(record["frame_id"]), "sentence": str(full.get("sentence") or full.get("expression")),
    }
    batch = store.build_unit(key_only)
    labels = store.attach_labels(batch, full)
    result = dict(record)
    for key in ("labels", "sidecar_candidate_gt"):
        value = labels[key]
        output_key = "candidate_gt" if key == "sidecar_candidate_gt" else key
        result[output_key] = [bool(x) for x in value.tolist()] if torch.is_tensor(value) else list(value)
    result.update({
        "positive_indices": [int(x) for x in labels["positive_indices"]], "positive_count": int(labels["positive_count"]),
        "target_ids": [str(x) for x in labels["target_ids"]], "target_present": bool(labels["target_present"]),
        "candidate_present": bool(labels["candidate_present"]), "coverage_mask": bool(labels["coverage_mask"]),
        "category": str(labels["category"]), "declared_category": str(full.get("category", "unknown")),
        "label_source": str(labels["label_source"]), "labels_attached_after_frozen_selection": True,
    })
    if len(result["labels"]) != int(record["candidate_count"]):
        raise AssertionError(f"fixed label length drift: {record['unit_key']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L86 fixed output: {out}")
    out.mkdir(parents=True, exist_ok=True); command = " ".join([sys.executable, *sys.argv]); started = time.perf_counter()
    try:
        if Path.cwd().resolve() != ROOT: raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256(MANIFEST) != MANIFEST_SHA: raise AssertionError("fixed manifest SHA drift")
        selection_path = args.selection.resolve(); selection = json.loads(selection_path.read_text())
        selected = selection["selected"]; checkpoint_info = selected["checkpoint_info"]
        checkpoint = Path(checkpoint_info["path"]).resolve()
        if sha256(checkpoint) != str(checkpoint_info["sha256"]): raise AssertionError("selected checkpoint SHA drift")
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
            torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        model, loaded = load_model(checkpoint, device)
        metadata = load_fixed_key_units()
        if len(metadata) != 40 or [int(x["fixed_eval_order"]) for x in metadata] != list(range(40)):
            raise AssertionError("immutable fixed order drift")
        immutable = [json.loads(line) for line in L62_ROWS.read_text().splitlines() if line.strip()]
        if len(immutable) != 40 or [str(x["unit_key"]) for x in immutable] != [str(x["unit_key"]) for x in metadata]:
            raise AssertionError("L62 immutable key order drift")
        store = L80BankStore(max_history=8)
        preselection = score_fixed_rows(metadata, args.cache.resolve(), model, store, device)
        if len(preselection) != 40 or [int(x["fixed_eval_order"]) for x in preselection] != list(range(40)):
            raise AssertionError("fixed L86 preselection order drift")
        forbidden = {field for row in preselection for field in (
            "target_ids", "positive_indices", "positive_count", "category", "labels", "target_present", "candidate_gt") if field in row}
        if forbidden: raise AssertionError(f"labels exposed before selection: {sorted(forbidden)}")
        preselection_audit = {
            "format": "locatemot-l86-preselection-label-isolation-v1", "status": "complete", "fixed_records": 40,
            "calibration_records": 16, "validation_records": 24, "schema": sorted(preselection[0].keys()),
            "forbidden_fields": sorted(["target_ids", "positive_indices", "positive_count", "category", "labels", "target_present", "candidate_gt"]),
            "forbidden_fields_absent": True, "native_order": [str(x["unit_key"]) for x in preselection],
            "candidate_rows_and_scores_complete": all(len(x["score"]) == int(x["candidate_count"]) for x in preselection),
            "candidate_deletion": False, "candidate_truncation": False, "labels_loaded": False,
            "selection_frozen_before_calibration_labels": True, "validation_labels_not_loaded": True,
            "event": "dev-selected checkpoint and rule frozen before fixed labels were attached",
            "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False,
        }
        write_json(out / "preselection_label_isolation.json", preselection_audit)
        # The checkpoint/rule is frozen on internal dev.  Calibration labels
        # are attached only now for a descriptive 16-unit table.
        calibration = [attach_labels(row, store) for row in preselection[:16]]
        rule = dict(selected["rule_fit"])
        c = float(rule["candidate_threshold"]); p = float(rule["presence_threshold"]); m = float(rule["null_margin"])
        cal_candidate = l86_metric(calibration, c, -float("inf"), -float("inf"))
        cal_final = l86_metric(calibration, c, p, m)
        # This is the first point at which validation labels are read.
        validation = [attach_labels(row, store) for row in preselection[16:]]
        val_candidate = l86_metric(validation, c, -float("inf"), -float("inf"))
        val_final = l86_metric(validation, c, p, m)
        control_records = make_control_records(); thresholds = immutable_control_thresholds()
        controls = {name: {"source": str(L62_ROWS), "threshold": thresholds[name],
                           "calibration": l29_metric(control_records[:16], field, thresholds[name]),
                           "validation": l29_metric(control_records[16:], field, thresholds[name])}
                    for name, field in (("l29_teacher", "l29"), ("l53_m0", "m0"), ("l54_continuous", "m54"))}
        checks = {
            "hard_violation_max": val_final["hard_violation"] <= .8666667,
            "recall_min": val_final["candidate_recall"] >= .7233333,
            "precision_min": val_final["candidate_precision"] >= .0830188679,
            "fp_per_frame_max": val_final["fp_per_frame"] <= 11.125,
            "predictions_per_positive_max": val_final["predictions_per_positive"] <= 4.069,
            "multi_positive_min": (val_final["multi_positive_recall"] is not None and val_final["multi_positive_recall"] >= .7894444),
            "inactive_false_acceptance_lt_1": val_final["inactive_false_acceptance"] < 1.0,
            "complete_finite_keys": len(validation) == 24 and all(
                len(x["score"]) == int(x["candidate_count"]) == len(x["row_keys"]) == len(x["labels"]) and x["finite_scores"] for x in validation),
            "candidate_deletion_false": all(not x["candidate_deletion"] and not x["candidate_truncation"] for x in preselection),
        }
        decision = "semantic_gate_pass_pending_supervisor" if all(checks.values()) else "semantic_gate_fail"
        semantic = {
            "format": "locatemot-l86-fixed-semantic-v1", "status": "complete", "decision": decision,
            "evidence_type": "fixed 16-calibration/24-validation semantic diagnostic; not screening/test",
            "command": command, "cwd": str(ROOT), "luna_thread": THREAD, "seed": 20260829,
            "selected_checkpoint": loaded, "selection_source": str(selection_path), "selected_rule": rule,
            "calibration": {"candidate_only": cal_candidate, "final_frozen_rule": cal_final, "labels_used_for_selection": False},
            "validation": {"candidate_only": val_candidate, "final_frozen_rule": val_final, "labels_used_for_selection": False},
            "immutable_controls": controls, "gate": {"checks": checks, "l29_validation_control": L29_VALIDATION_CONTROL},
            "candidate_rows": {"calibration": int(sum(x["candidate_count"] for x in calibration)),
                               "validation": int(sum(x["candidate_count"] for x in validation)), "all_rows_retained": True,
                               "candidate_deletion": False, "candidate_truncation": False},
            "label_events": {"preselection_complete_before_labels": True, "calibration_attached_after_frozen_selection": True,
                              "validation_attached_after_frozen_selection": True},
            "manifest_sha256": MANIFEST_SHA, "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False, "no_hota_or_trackeval": True,
            "z1_representation_changed": False, "groundingdino_lora_used": False,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
        }
        gate = {"format": "locatemot-l86-fixed-gate-v1", "status": decision, "decision": decision, "checks": checks,
                "selected_checkpoint": loaded, "selected_rule": rule, "calibration_units": 16, "validation_units": 24,
                "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                "hota_trackeval_run": False, "no_hota_or_trackeval": True}
        write_json(out / "semantic.json", semantic); write_json(out / "gate_decision.json", gate)
        with (out / "score_records.jsonl").open("w") as handle:
            for row in calibration + validation:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        provenance = {"format": "locatemot-l86-fixed-semantic-provenance-v1", "status": "complete",
                      "command": command, "cwd": str(ROOT), "luna_thread": THREAD, "seed": 20260829,
                      "inputs": {"cache": str(args.cache.resolve()), "cache_summary_sha256": sha256(args.cache.resolve() / "summary.json"),
                                 "selection": str(selection_path), "selection_sha256": sha256(selection_path),
                                 "immutable_l62_rows": str(L62_ROWS), "immutable_l62_rows_sha256": sha256(L62_ROWS),
                                 "manifest": str(MANIFEST), "manifest_sha256": MANIFEST_SHA},
                      "selected_checkpoint": loaded, "preselection_label_isolation": str(out / "preselection_label_isolation.json"),
                      "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
                      "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                      "hota_trackeval_run": False, "no_hota_or_trackeval": True, "token_span_region_alignment": "UNALIGNED",
                      "static_motion_alignment": "UNALIGNED", "wall_seconds": time.perf_counter() - started}
        write_json(out / "provenance.json", provenance)
        write_json(out / "status.json", {"format": "locatemot-l86-fixed-semantic-status-v1", "status": decision,
                                         "selected_rule": rule, "selected_checkpoint": loaded,
                                         "screening_gt_used": False, "official_test_labels_read": False,
                                         "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        print(json.dumps({"status": decision, "validation": val_final, "gate": checks}, indent=2), flush=True)
        return 0
    except Exception:
        trace = traceback.format_exc(); (out / "INCOMPLETE.md").write_text("# L86 fixed semantic — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {"format": "locatemot-l86-fixed-semantic-status-v1", "status": "incomplete",
                                         "command": command, "cwd": str(ROOT), "luna_thread": THREAD,
                                         "failure_root_cause": "first traceback in INCOMPLETE.md", "screening_gt_used": False,
                                         "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                                         "hota_trackeval_run": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
