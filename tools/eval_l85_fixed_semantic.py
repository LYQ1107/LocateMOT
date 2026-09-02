#!/usr/bin/env python3
"""Frozen L85 fixed-slice semantic report.

The checkpoint and global emission rule are frozen by the internal-dev
selection artifact before this tool attaches any fixed calibration or
validation labels.  The fixed 16/24 slice is a semantic diagnostic; it is not
the full-video TrackEval result produced by the later inference tool.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from locatemot.models.l85_full_rmot import L85Config, L85FullRMOT  # noqa: E402
from locatemot.rmot.l80_data import (  # noqa: E402
    L80BankStore,
    key_only,
    load_fixed_key_units,
    load_full_unit_for_labels,
)
from locatemot.rmot.l85_fullvideo_bank import EXPECTED_MANIFEST_SHA, MANIFEST, sha256_file  # noqa: E402
from locatemot.rmot.l85_runtime import build_groups, load_internal_eval_groups  # noqa: E402
from tools.l85_calibrate_dev import history_for_final_stage, score_group  # noqa: E402
from tools.eval_l80_v12 import (  # noqa: E402
    L29_VALIDATION_CONTROL,
    immutable_control_thresholds,
    make_control_records,
    metric as immutable_metric,
)


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
CACHE_FORMAT = "locatemot-l85-z1-semantic-cache-v1"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def attach_label(record: dict[str, Any], full: dict[str, Any], store: L80BankStore) -> dict[str, Any]:
    batch = store.build_unit(key_only(full))
    sidecar_path = Path(batch.bank_path).with_suffix(".labels.json")
    sidecar = json.loads(sidecar_path.read_text())
    candidate_gt = sidecar.get("candidate_gt")
    if not isinstance(candidate_gt, list):
        raise AssertionError(f"missing candidate_gt sidecar: {sidecar_path}")
    offsets = [int(x) for x in record["row_offsets"]]
    if max(offsets, default=-1) >= len(candidate_gt):
        raise AssertionError(f"sidecar shorter than row offsets: {record['unit_key']}")
    targets = {str(x) for x in full.get("target_ids", [])}
    labels = [bool(candidate_gt[offset] is not None and str(candidate_gt[offset]) in targets) for offset in offsets]
    target_present = bool(targets)
    candidate_present = bool(any(labels))
    category = ("inactive" if not target_present else
                "present_uncovered" if not candidate_present else
                "multi_positive" if sum(labels) > 1 else "positive")
    result = dict(record)
    result.update({
        "labels": labels, "positive_indices": [i for i, x in enumerate(labels) if x],
        "positive_count": int(sum(labels)), "target_ids": sorted(targets),
        "target_present": target_present, "candidate_present": candidate_present,
        "coverage_mask": not (target_present and not candidate_present), "category": category,
        "declared_category": str(full.get("category", "unknown")),
        "sidecar_candidate_gt": [None if candidate_gt[offset] is None else str(candidate_gt[offset]) for offset in offsets],
        "label_source": str(sidecar_path.resolve()), "sidecar_labels_loaded": True,
        "labels_attached_after_frozen_selection": True,
    })
    if len(labels) != int(record["candidate_count"]):
        raise AssertionError(f"label length drift: {record['unit_key']}")
    return result


def l85_metric(records: list[dict[str, Any]], candidate_threshold: float,
               presence_threshold: float, null_margin: float, *, use_presence: bool,
               _stratify: bool = True) -> dict[str, Any]:
    """Metric with optional final energy/presence emission rule."""
    tp = fp = fn = selected_rows = positive_rows = 0
    top1 = top5 = top_units = empty = 0
    hard: list[bool] = []; strict: list[float] = []; best: list[float] = []; average: list[float] = []
    multi_recall: list[float] = []; minimum_coverage: list[float] = []
    inactive = inactive_accept = inactive_fp = 0; present = present_uncovered = 0
    all_scores: list[float] = []
    by_category: dict[str, list[dict[str, Any]]] = {}; by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        score = np.asarray(row["score"], dtype=np.float64)
        labels = np.asarray(row["labels"], dtype=bool)
        if score.shape != labels.shape or not np.isfinite(score).all():
            raise AssertionError(f"score/label drift {row['unit_key']}")
        score_selected = score >= float(candidate_threshold)
        gate = (float(row["presence"]) >= float(presence_threshold) and
                float(row["presence"]) - float(row["null_logit"]) >= float(null_margin))
        selected = score_selected & bool(gate) if use_presence else score_selected
        tp += int((selected & labels).sum()); fp += int((selected & ~labels).sum()); fn += int((~selected & labels).sum())
        selected_rows += int(selected.sum()); positive_rows += int(labels.sum()); all_scores.extend(score.tolist())
        category = str(row.get("category", "unknown"))
        by_category.setdefault(category, []).append(row); by_dataset.setdefault(str(row["dataset"]), []).append(row)
        if category == "inactive":
            inactive += 1; inactive_accept += int(bool(selected.any())); inactive_fp += int((selected & ~labels).sum())
        elif category == "present_uncovered":
            present_uncovered += 1
        else:
            present += 1
        if labels.any():
            order = np.argsort(-score, kind="stable")
            top1 += int(bool(labels[order[:1]].any())); top5 += int(bool(labels[order[:5]].any())); top_units += 1
        empty += int(not selected.any())
        pos = np.flatnonzero(labels); neg = np.flatnonzero(~labels)
        if len(pos) and len(neg):
            strict_value = float(score[pos].min() - score[neg].max()); strict.append(strict_value)
            best.append(float(score[pos].max() - score[neg].max())); average.append(float(score[pos].mean() - score[neg].max()))
            hard.append(strict_value < 0.0)
        if len(pos) > 1:
            multi_recall.append(float(selected[pos].sum() / len(pos))); minimum_coverage.append(float(selected[pos].all()))
    def stats(values: list[float]) -> dict[str, Any]:
        if not values: return {"count": 0, "mean": None, "p50": None, "p90": None}
        a = np.asarray(values, dtype=np.float64)
        return {"count": int(a.size), "mean": float(a.mean()), "p50": float(np.quantile(a, .5)), "p90": float(np.quantile(a, .9))}
    result = {
        "candidate_threshold": float(candidate_threshold), "presence_threshold": float(presence_threshold),
        "null_margin": float(null_margin), "use_presence_and_null_rule": bool(use_presence), "units": len(records),
        "candidate_rows": int(sum(len(x["labels"]) for x in records)), "positive_rows": positive_rows,
        "selected_rows": selected_rows, "true_positive_rows": tp, "false_positive_rows": fp, "false_negative_rows": fn,
        "candidate_precision": tp / max(1, selected_rows), "candidate_recall": tp / max(1, tp + fn),
        "fp_per_frame": fp / max(1, len(records)), "predictions_per_positive": selected_rows / max(1, positive_rows),
        "top1": top1 / max(1, top_units), "top5": top5 / max(1, top_units),
        "hard_violation": float(np.mean(hard)) if hard else None,
        "strict_margin": stats(strict), "best_margin": stats(best), "average_margin": stats(average),
        "multi_positive_recall": float(np.mean(multi_recall)) if multi_recall else None,
        "minimum_positive_coverage": float(np.mean(minimum_coverage)) if minimum_coverage else None,
        "empty_rate": empty / max(1, len(records)), "inactive_units": inactive,
        "inactive_false_acceptance": inactive_accept / max(1, inactive), "inactive_false_positive_rows": inactive_fp,
        "present_units": present, "present_uncovered_units": present_uncovered,
        "score_distribution": stats(all_scores), "candidate_rows_retained": True,
        "candidate_deletion": False, "candidate_truncation": False,
        "per_category": {}, "per_dataset": {},
    }
    if _stratify:
        for key, values in by_category.items():
            result["per_category"][key] = l85_metric(
                values, candidate_threshold, presence_threshold, null_margin,
                use_presence=use_presence, _stratify=False)
        for key, values in by_dataset.items():
            result["per_dataset"][key] = l85_metric(
                values, candidate_threshold, presence_threshold, null_margin,
                use_presence=use_presence, _stratify=False)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty fixed semantic output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable] + sys.argv)
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA drift")
    cache_root = (args.cache if args.cache.is_absolute() else ROOT / args.cache).resolve()
    cache_summary = json.loads((cache_root / "summary.json").read_text())
    if cache_summary.get("format") != CACHE_FORMAT or cache_summary.get("status") != "complete" or cache_summary.get("labels_in_cache"):
        raise AssertionError("invalid label-free L85 cache")
    selection_path = (args.selection if args.selection.is_absolute() else ROOT / args.selection).resolve()
    selection = json.loads(selection_path.read_text())
    selected = selection["selected"]
    checkpoint_info = selected["checkpoint_info"]
    checkpoint = Path(checkpoint_info["path"]).resolve()
    package = torch.load(checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    model = L85FullRMOT(L85Config(**package["model_config"])).to(device=device, dtype=torch.float32)
    loaded = model.load_state_dict(package["model_state_dict"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise AssertionError(f"strict L85 checkpoint load failed: {loaded}")
    model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    if device.type == "cuda":
        if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
        torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        model = model.to(device)
    metadata = load_fixed_key_units()
    # The canonical order is checked directly against the immutable records
    # without importing any L62 scores into the L85 score fields.
    immutable_rows = [json.loads(line) for line in (ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl").read_text().splitlines() if line.strip()]
    if len(immutable_rows) != 40:
        raise AssertionError("immutable L62 row count drift")
    expected_order = [str(row["unit_key"]) for row in immutable_rows]
    if [str(row["unit_key"]) for row in metadata] != expected_order:
        raise AssertionError("fixed L62 order drift")
    fixed_by_key = {str(row["unit_key"]): row for row in metadata}
    eval_groups, _, _ = load_internal_eval_groups()
    cache_manifest = [json.loads(line) for line in (cache_root / "manifest.jsonl").read_text().splitlines() if line.strip()]
    cache_by_group = {str(row["group_key"]): row for row in cache_manifest}
    wanted_groups = {f"{row['dataset']}|{row['video']}|{int(row['frame_id'])}" for row in metadata}
    preselection_by_key: dict[str, dict[str, Any]] = {}
    store = L80BankStore(max_history=8)
    try:
        for group_key in sorted(wanted_groups):
            if group_key not in eval_groups or group_key not in cache_by_group:
                raise KeyError(f"fixed group missing: {group_key}")
            item = torch.load(cache_by_group[group_key]["path"], map_location="cpu", weights_only=False)
            current = score_group(item, eval_groups[group_key], store, model, device)
            for record in current:
                if record["unit_key"] in fixed_by_key:
                    preselection_by_key[record["unit_key"]] = record
            del item, current
        preselection = [preselection_by_key[key] for key in expected_order]
        if len(preselection) != 40:
            raise AssertionError(f"fixed score count drift: {len(preselection)}")
        for order, record in enumerate(preselection):
            if record["unit_key"] != expected_order[order] or len(record["score"]) != record["candidate_count"]:
                raise AssertionError(f"fixed order/candidate drift: {record['unit_key']}")
            if not np.isfinite(np.asarray(record["score"], dtype=np.float64)).all():
                raise FloatingPointError(f"nonfinite fixed score: {record['unit_key']}")
        forbidden = sorted({field for record in preselection for field in (
            "target_ids", "positive_indices", "positive_count", "category", "labels", "target_present") if field in record})
        if forbidden:
            raise AssertionError(f"labels in L85 preselection: {forbidden}")
        write_json(out / "preselection_label_isolation.json", {
            "format": "locatemot-l85-preselection-label-isolation-v1", "status": "complete",
            "fixed_records": 40, "calibration_records": 16, "validation_records": 24,
            "schema": sorted(preselection[0]), "forbidden_fields": [
                "target_ids", "positive_indices", "positive_count", "category", "labels", "target_present"],
            "forbidden_fields_absent": True, "native_order": expected_order,
            "candidate_rows_and_scores_complete": True, "candidate_deletion": False, "candidate_truncation": False,
            "sidecar_labels_loaded": False, "selection_frozen_before_fixed_labels": True,
            "event": "selected dev checkpoint/rule frozen before fixed calibration/validation label attachment",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        })
        # Fixed calibration labels are descriptive only; no threshold or
        # checkpoint is reselected from this slice.
        calibration = [attach_label(preselection[index], load_full_unit_for_labels(expected_order[index]), store) for index in range(16)]
        rule = selected["rule_fit"]
        cal_candidate = l85_metric(calibration, rule["candidate_threshold"], rule["presence_threshold"], rule["null_margin"], use_presence=False)
        cal_final = l85_metric(calibration, rule["candidate_threshold"], rule["presence_threshold"], rule["null_margin"], use_presence=True)
        # Selection is now fully frozen.  Only this line begins the validation
        # label attach event.
        validation = [attach_label(preselection[index], load_full_unit_for_labels(expected_order[index]), store) for index in range(16, 40)]
        val_candidate = l85_metric(validation, rule["candidate_threshold"], rule["presence_threshold"], rule["null_margin"], use_presence=False)
        val_final = l85_metric(validation, rule["candidate_threshold"], rule["presence_threshold"], rule["null_margin"], use_presence=True)
        controls = make_control_records(); control_thresholds = immutable_control_thresholds()
        control_results = {}
        for name, field in (("l29_teacher", "l29"), ("l53_m0", "m0"), ("l54_continuous", "m54")):
            control_results[name] = {"source": "immutable L62 score_records.jsonl", "fixed_threshold": control_thresholds[name],
                                    "calibration": immutable_metric(controls[:16], field, control_thresholds[name]),
                                    "validation": immutable_metric(controls[16:], field, control_thresholds[name])}
        gate_checks = {
            "hard_negative_improvement_ge_0_05": val_final["hard_violation"] is not None and val_final["hard_violation"] <= L29_VALIDATION_CONTROL["hard_violation"] - .05,
            "recall_floor": val_final["candidate_recall"] >= .7233333,
            "precision_floor": val_final["candidate_precision"] >= .0830188679,
            "fp_per_frame_ceiling": val_final["fp_per_frame"] <= 11.125,
            "predictions_per_positive_ceiling": val_final["predictions_per_positive"] <= 4.069,
            "multi_positive_floor": val_final["multi_positive_recall"] is not None and val_final["multi_positive_recall"] >= .7894444,
            "inactive_false_acceptance_lt_1": val_final["inactive_false_acceptance"] < 1.0,
            "complete_finite_keys": len(validation) == 24 and all(
                row["candidate_count"] == len(row["row_offsets"]) == len(row["row_keys"]) == len(row["score"]) == len(row["labels"]) and row["finite_scores"]
                for row in validation),
            "candidate_deletion_false": all(not row["candidate_deletion"] and not row["candidate_truncation"] for row in preselection),
        }
        decision = "semantic_gate_pass_pending_supervisor" if all(gate_checks.values()) else "semantic_gate_fail"
        semantic = {
            "format": "locatemot-l85-fixed-semantic-v1", "status": "complete", "decision": decision,
            "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()), "luna_thread": THREAD, "seed": SEED,
            "selected_checkpoint": checkpoint_info, "selected_rule": rule,
            "calibration": {"candidate_only": cal_candidate, "final_energy_presence_rule": cal_final, "labels_used_for_selection": False},
            "validation": {"candidate_only": val_candidate, "final_energy_presence_rule": val_final, "labels_used_for_selection": False},
            "immutable_controls": control_results, "gate": {"checks": gate_checks, "l29_validation_control": L29_VALIDATION_CONTROL,
                "thresholds": {"hard_violation_max": .8666667, "recall_min": .7233333, "precision_min": .0830188679,
                               "fp_per_frame_max": 11.125, "predictions_per_positive_max": 4.069, "multi_positive_min": .7894444}},
            "candidate_rows": {"calibration": int(sum(x["candidate_count"] for x in calibration)),
                               "validation": int(sum(x["candidate_count"] for x in validation)), "all_rows_retained": True,
                               "candidate_deletion": False, "candidate_truncation": False},
            "label_events": {"preselection_scores_complete": True, "calibration_labels_attached_after_frozen_dev_selection": True,
                              "validation_labels_attached_after_frozen_dev_selection": True},
            "evidence_type": "fixed 16-calibration/24-validation semantic diagnostic after internal-dev selection; not screening, official test, HOTA or TrackEval",
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        }
        gate = {"format": "locatemot-l85-fixed-gate-v1", "status": decision, "decision": decision,
                "selected_checkpoint": checkpoint_info, "selected_rule": rule, "checks": gate_checks,
                "calibration_units": 16, "validation_units": 24, "dev_selection_source": str(selection_path),
                "candidate_set": "complete L69 rows; no top-k/NMS/deletion", "screening_gt_used": False,
                "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True}
        write_json(out / "semantic.json", semantic); write_json(out / "gate_decision.json", gate)
        with (out / "score_records.jsonl").open("w") as handle:
            for order, row in enumerate(preselection):
                labeled = calibration[order] if order < 16 else validation[order - 16]
                handle.write(json.dumps({**row, **{key: labeled[key] for key in (
                    "labels", "positive_indices", "positive_count", "target_ids", "target_present", "candidate_present",
                    "coverage_mask", "category", "declared_category", "sidecar_candidate_gt", "label_source", "sidecar_labels_loaded")}},
                                        ensure_ascii=False, default=str) + "\n")
        write_json(out / "provenance.json", {"format": "locatemot-l85-fixed-semantic-provenance-v1", "status": "complete",
                  "command": command, "cwd": str(ROOT), "luna_thread": THREAD, "seed": SEED,
                  "inputs": {"cache": str(cache_root), "cache_summary_sha256": sha256_file(cache_root / "summary.json"),
                             "selection": str(selection_path), "selection_sha256": sha256_file(selection_path),
                             "manifest": str(MANIFEST), "manifest_sha256": sha256_file(MANIFEST),
                             "immutable_l62_rows": str(ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"),
                             "immutable_l62_rows_sha256": sha256_file(ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl")},
                  "selected_checkpoint": checkpoint_info, "preselection_label_isolation": str(out / "preselection_label_isolation.json"),
                  "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
                  "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                  "hota_trackeval_run": False})
        write_json(out / "status.json", {"format": "locatemot-l85-fixed-semantic-status-v1", "status": decision,
                  "selected_checkpoint": checkpoint_info, "next_action": "run frozen full-video internal dev/validation inference and TrackEval diagnostics",
                  "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        return {"status": decision, "selected_checkpoint": checkpoint_info, "validation": val_final, "gate": gate_checks, "output": str(out)}
    finally:
        store._store._bank = None; store._store._text_cache = None
        del model, package
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
