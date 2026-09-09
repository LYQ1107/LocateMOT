#!/usr/bin/env python3
"""Fixed 16-calibration/24-validation replay with phase-correct history."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
WORK_ROOT = Path(__file__).resolve().parents[1]
for value in (WORK_ROOT, WORK_ROOT / "tools", ROOT, ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from l89d_fullvideo_common import (  # noqa: E402
    MANIFEST_SHA,
    SEED,
    THREAD,
    WORK_ROOT as COMMON_WORK_ROOT,
    command_line,
    manifest_assertion,
    sha256_file,
    standard_flags,
    write_json,
)
from l89_eval_fixed_semantic import (  # noqa: E402
    FORBIDDEN,
    attach,
    fixed_order,
    index_cache_items,
    label_free_record,
)
from l88c_eval_metrics import metric  # noqa: E402
from locatemot.models.l89_full_rmot import L89Config, L89FullRMOT  # noqa: E402
from locatemot.rmot.l80_data import L80BankStore  # noqa: E402
from locatemot.rmot.l89_language_cache import L89LanguageTokenCache  # noqa: E402
from l89e_phase_policy import history_for_batch, phase_policy_for_epoch  # noqa: E402


L29 = {
    "recall": 0.7333333333333333,
    "precision": 0.0830188679245283,
    "fp_per_frame": 10.125,
    "predictions_per_positive": 8.833333333333334,
    "hard_violation": 0.9166666666666666,
    "multi_positive_recall": 0.8194444444444443,
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def _load_selected_model(selection: dict[str, Any], device: torch.device) -> tuple[L89FullRMOT, dict[str, Any], Any]:
    checkpoint = dict(selection["final_selection"]["checkpoint_info"])
    path = Path(str(checkpoint["path"])).resolve()
    if sha256_file(path) != str(checkpoint["sha256"]):
        raise AssertionError(f"selected checkpoint SHA drift: {path}")
    package = torch.load(path, map_location="cpu", weights_only=False)
    if package.get("format") != "locatemot-l89-checkpoint-v1" or int(package.get("seed", -1)) != SEED:
        raise AssertionError("invalid selected L89 package")
    if str(package.get("manifest_sha256")) != MANIFEST_SHA:
        raise AssertionError("selected checkpoint manifest drift")
    policy = phase_policy_for_epoch(int(package["epoch"]), str(package["phase"]))
    model = L89FullRMOT(L89Config(**package["model_config"])).to(device=device, dtype=torch.float32)
    result = model.load_state_dict(package["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"strict L89 reload failed: {result}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    loaded = dict(checkpoint)
    loaded.update({
        "path": str(path), "sha256": sha256_file(path), "epoch": int(package["epoch"]),
        "phase": str(package["phase"]), "optimizer_step": int(package["optimizer_step"]),
        "model_config": package["model_config"], "strict_package": True,
        "phase_policy": policy.to_dict(), "phase_consistent_temporal": True,
    })
    return model, loaded, policy


def _record_phase(record: dict[str, Any], policy: Any) -> None:
    record.update({
        "phase_consistent_temporal": True,
        "checkpoint_phase": policy.phase,
        "temporal_enabled": policy.temporal_enabled,
        "history_mode": policy.history_mode,
        "history_length": policy.history_length,
        "history_contract": policy.history_mode,
        "phase_policy": policy.to_dict(),
    })


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L89E fixed semantic output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = command_line()
    store: L80BankStore | None = None
    try:
        if Path.cwd().resolve() != COMMON_WORK_ROOT:
            raise RuntimeError(f"wrong L89E worktree cwd: {Path.cwd()}")
        manifest_assertion()
        selection = _load(args.selection)
        if selection.get("format") != "locatemot-l89e-checkpoint-selection-v1" or selection.get("status") != "complete":
            raise AssertionError("L89E selection is incomplete or wrong format")
        if not bool(selection.get("selection_frozen_before_fixed_validation")) or not bool(selection.get("phase_consistent_temporal")):
            raise AssertionError("L89E selection is not frozen with phase proof")
        final = selection["final_selection"]
        rule_name = str(final["rule"])
        thresholds = {
            name: float(final["rule_object"][name])
            for name in ("candidate_threshold", "presence_threshold", "null_margin")
        }
        if not all(np.isfinite(value) for value in thresholds.values()):
            raise AssertionError("nonfinite frozen L89E thresholds")
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        model, checkpoint, policy = _load_selected_model(selection, device)
        language = L89LanguageTokenCache(args.language_cache.resolve())
        store = L80BankStore(max_history=8)
        rows = fixed_order()
        group_cache = index_cache_items(args.z1_cache.resolve())
        records: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            batch = store.build_unit(row)
            group_key = f"{batch.dataset}|{batch.video}|{batch.frame_id}"
            item_path = group_cache.get(group_key)
            if item_path is None:
                raise KeyError(f"L89E fixed group cache missing: {group_key}")
            item = torch.load(item_path, map_location="cpu", weights_only=False)
            qids = [int(value) for value in item["query_ids"]]
            qindex = qids.index(int(batch.query_id))
            history, history_mask, history_frames = history_for_batch(batch, policy)
            tokens, mask = language.get_batch(
                str(row["dataset"]), str(row["video"]), [int(row["query_id"])], [str(row["sentence"])], device,
            )
            with torch.inference_mode():
                output = model(
                    item["z1"][qindex:qindex + 1].float().to(device),
                    tokens,
                    mask,
                    item["text_global"][qindex:qindex + 1].float().to(device),
                    item["frame_global"][qindex:qindex + 1].float().to(device),
                    batch.observations.float().to(device),
                    history.float().to(device),
                    history_mask.bool().to(device),
                    history_frames.long().to(device),
                    batch.frame_id,
                    temporal_enabled=policy.temporal_enabled,
                )
            record = label_free_record(batch, row, output, checkpoint)
            _record_phase(record, policy)
            record["history_valid_count"] = int(history_mask.sum().item())
            record["history_valid_count_per_candidate"] = [int(value) for value in history_mask.sum(dim=1).tolist()]
            if policy.phase == "S" and int(history_mask.sum()) != 0:
                raise AssertionError(f"Stage-S history is not empty: {record['unit_key']}")
            if int(record["candidate_count"]) != len(record["score"]) or not np.isfinite(np.asarray(record["score"], dtype=np.float64)).all():
                raise AssertionError(f"L89E fixed score shape/finite drift: {record['unit_key']}")
            records.append(record)
            del item, tokens, mask, output, batch, history, history_mask, history_frames
            gc.collect()
        if len(records) != 40 or [int(row["fixed_eval_order"]) for row in records] != list(range(40)):
            raise AssertionError("L89E fixed label-free order drift")
        preselection = {
            "format": "locatemot-l89e-fixed-preselection-v1",
            "status": "complete",
            "record_count": 40,
            "forbidden_label_fields": sorted(FORBIDDEN),
            "forbidden_fields_absent": [not FORBIDDEN.intersection(row) for row in records],
            "all_forbidden_fields_absent": all(not FORBIDDEN.intersection(row) for row in records),
            "candidate_lengths_complete": all(len(row["score"]) == int(row["candidate_count"]) for row in records),
            "candidate_rows_retained": True,
            "candidate_deletion": False,
            "candidate_truncation": False,
            "phase_consistent_temporal": True,
            "selection_frozen_before_fixed_labels": True,
            "selection": str(args.selection.resolve()),
            "screening_gt_used": False,
            "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False,
            "no_hota_or_trackeval": True,
        }
        write_json(out / "preselection_label_isolation.json", preselection)
        if not preselection["all_forbidden_fields_absent"]:
            raise AssertionError("L89E fixed preselection label leak")
        # Calibration labels are attached after the label-free score records and
        # selection/threshold object are frozen; validation labels come later.
        for index in range(16):
            batch = store.build_unit(rows[index])
            attach(records[index], batch, store)
        calibration = records[:16]
        calibration_metrics = metric(calibration, **thresholds)
        for index in range(16, 40):
            batch = store.build_unit(rows[index])
            attach(records[index], batch, store)
        validation = records[16:]
        validation_metrics = metric(validation, **thresholds)
        checks = {
            "hard_improvement": float(validation_metrics["legacy_row_hard_violation"]) <= L29["hard_violation"] - 0.05,
            "recall_floor": float(validation_metrics["legacy_candidate_recall"]) >= 0.7233333,
            "precision_floor": float(validation_metrics["legacy_candidate_precision"]) >= 0.0830188679,
            "fp_per_frame_ceiling": float(validation_metrics["legacy_fp_per_frame"]) <= 11.125,
            "predictions_per_positive_ceiling": float(validation_metrics["legacy_predictions_per_positive"]) <= 4.069,
            "multi_positive_floor": validation_metrics["legacy_row_multi_positive_recall"] is not None and float(validation_metrics["legacy_row_multi_positive_recall"]) >= 0.7894444,
            "inactive_nonuniversal": float(validation_metrics["inactive_false_acceptance"]) < 1.0,
            "complete_finite_rows": bool(validation_metrics["candidate_rows_retained"] and validation_metrics["finite_scores"] and not validation_metrics["candidate_deletion"] and not validation_metrics["candidate_truncation"]),
        }
        decision = "semantic_gate_pass" if all(checks.values()) else "semantic_gate_fail"
        semantic = {
            "format": "locatemot-l89e-fixed-semantic-v1",
            "status": "complete",
            "evidence_type": "fixed 16-calibration/24-validation diagnostic after L89E fit/dev selection",
            "corrected_candidate_vs_null": True,
            "emission_contract": "candidate>=candidate_threshold & candidate-null>=null_margin & presence>=presence_threshold",
            "zero_training": True,
            "checkpoint": checkpoint,
            "selection": selection,
            "selection_frozen_before_fixed_validation": True,
            "rule": rule_name,
            "thresholds": thresholds,
            "phase_policy": policy.to_dict(),
            "phase_consistent_temporal": True,
            "calibration": calibration_metrics,
            "validation": validation_metrics,
            "gate_checks": checks,
            "l29_teacher": {"evidence_type": "immutable accepted historical control", "validation": L29},
            "l87a_historic": {"V1_HOTA": 28.5752, "V2_HOTA": 22.1300},
            "candidate_rows_retained": True,
            "candidate_deletion": False,
            "candidate_truncation": False,
            "record_count": 40,
            "calibration_count": 16,
            "validation_count": 24,
            "screening_gt_used": False,
            "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False,
            "no_hota_or_trackeval": True,
            "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED",
        }
        gate = {
            "format": "locatemot-l89e-fixed-gate-v1",
            "status": "complete",
            "decision": decision,
            "corrected_candidate_vs_null": True,
            "emission_contract": semantic["emission_contract"],
            "zero_training": True,
            "checks": checks,
            "thresholds": thresholds,
            "rule": rule_name,
            "phase_policy": policy.to_dict(),
            "phase_consistent_temporal": True,
            "baseline": L29,
            "screening_gt_used": False,
            "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False,
            "no_hota_or_trackeval": True,
        }
        with (out / "score_records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_json(out / "semantic.json", semantic)
        write_json(out / "gate_decision.json", gate)
        provenance = {
            "format": "locatemot-l89e-fixed-semantic-provenance-v1",
            "status": "complete",
            "command": command,
            "cwd": str(COMMON_WORK_ROOT),
            "luna_thread": THREAD,
            "seed": SEED,
            "inputs": {
                "selection": str(args.selection.resolve()),
                "selection_sha256": sha256_file(args.selection.resolve()),
                "z1_cache": str(args.z1_cache.resolve()),
                "language_cache": str(args.language_cache.resolve()),
                "l62_fixed_rows": str(ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"),
                "l62_fixed_rows_sha256": sha256_file(ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"),
                "manifest_sha256": MANIFEST_SHA,
            },
            "outputs": {
                "semantic": str((out / "semantic.json").resolve()),
                "gate": str((out / "gate_decision.json").resolve()),
                "score_records": str((out / "score_records.jsonl").resolve()),
                "preselection_label_isolation": str((out / "preselection_label_isolation.json").resolve()),
            },
            "label_boundary": "all 40 score rows were phase-consistently built and scored before labels; calibration labels attached before frozen semantic report; validation labels attached afterward",
            "phase_consistency_contract": "S: temporal off + zero history; T/J: temporal on + last4 causal history",
            "corrected_candidate_vs_null": True,
            "zero_training": True,
            "candidate_rows_retained": True,
            "candidate_deletion": False,
            "candidate_truncation": False,
            "phase_consistent_temporal": True,
            "screening_gt_used": False,
            "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False,
            "no_hota_or_trackeval": True,
        }
        write_json(out / "provenance.json", provenance)
        write_json(out / "status.json", {
            "format": "locatemot-l89e-fixed-semantic-v1",
            "status": "complete",
            "decision": decision,
            "record_count": 40,
            "calibration_count": 16,
            "validation_count": 24,
            "phase_consistent_temporal": True,
            "screening_gt_used": False,
            "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False,
            "next_action": "run L89E true-full-video internal TrackEval after phase-consistent dev replay" if decision == "semantic_gate_fail" else "stop for supervisor review",
        })
        return 0
    except Exception as exc:
        (out / "INCOMPLETE.md").write_text("# L89E fixed semantic — INCOMPLETE\n\n" + traceback.format_exc(), encoding="utf-8")
        write_json(out / "status.json", {
            "format": "locatemot-l89e-fixed-semantic-v1",
            "status": "incomplete",
            "command": command,
            "cwd": str(COMMON_WORK_ROOT),
            "luna_thread": THREAD,
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "next_action": "repair the first phase-consistent fixed-evaluation error and use a new output",
            **standard_flags(hota_trackeval_run=False),
        })
        raise
    finally:
        if store is not None:
            store._store._bank = None
            store._store._text_cache = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--z1-cache", type=Path, required=True)
    parser.add_argument("--language-cache", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
