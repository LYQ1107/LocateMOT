#!/usr/bin/env python3
"""Immutable 16-calibration/24-validation L89 semantic replay.

Checkpoint and Rule B/R/P are frozen by internal fit/dev selection before
this script attaches fixed-slice labels.  The fixed slice is diagnostic and
does not select a checkpoint or rescue a threshold.
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
WORK_ROOT = Path(__file__).resolve().parents[1]
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
Z1_CACHE = ROOT / "outputs/l85/features/fit_dev_eval_full_attempt2"
LANG_CACHE = WORK_ROOT / "outputs/l89/cache/language_tokens_retry1"
L62_ROWS = ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
L29 = {
    "recall": 0.7333333333333333, "precision": 0.0830188679245283,
    "fp_per_frame": 10.125, "predictions_per_positive": 8.833333333333334,
    "hard_violation": 0.9166666666666666, "multi_positive_recall": 0.8194444444444443,
}
FORBIDDEN = {"target_ids", "positive_indices", "positive_count", "category", "labels",
             "target_present", "candidate_gt", "coverage_mask", "declared_category"}

for path in (WORK_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import locatemot.models as _models_package  # noqa: E402
import locatemot.rmot as _rmot_package  # noqa: E402
for package, name in ((_models_package, "models"), (_rmot_package, "rmot")):
    value = str(WORK_ROOT / "locatemot" / name)
    if value not in [str(x) for x in package.__path__]:
        package.__path__.append(value)

from locatemot.models.l89_full_rmot import L89Config, L89FullRMOT  # noqa: E402
from locatemot.rmot.l80_data import L80BankStore, load_fixed_key_units, load_full_unit_for_labels  # noqa: E402
from locatemot.rmot.l89_language_cache import L89LanguageTokenCache  # noqa: E402
from l88_eval_metrics import fit_rule_set, metric  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def fixed_order() -> list[dict[str, Any]]:
    records = [json.loads(line) for line in L62_ROWS.read_text().splitlines() if line.strip()]
    if len(records) != 40 or [str(x["unit_key"]) for x in records] != list(dict.fromkeys(str(x["unit_key"]) for x in records)):
        raise AssertionError("L62 fixed order is not 40 unique units")
    units = load_fixed_key_units()
    by_key = {str(x["unit_key"]): x for x in units}
    result = []
    for index, record in enumerate(records):
        key = str(record["unit_key"])
        if key not in by_key:
            raise AssertionError(f"fixed key missing from key-only L49 rows: {key}")
        row = dict(by_key[key]); row["fixed_eval_order"] = index
        row["evaluation_partition"] = "calibration" if index < 16 else "validation"
        result.append(row)
    return result


def language_for(cache: L89LanguageTokenCache, row: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return cache.get_batch(str(row["dataset"]), str(row["video"]), [int(row["query_id"])], [str(row["sentence"])], device)


def index_cache_items(cache_root: Path) -> dict[str, Path]:
    """Index compact L85 items without relying on an optional summary list."""
    result: dict[str, Path] = {}
    for path in sorted(cache_root.rglob("*.pt")):
        item = torch.load(path, map_location="cpu", weights_only=False)
        key = str(item.get("group_key", ""))
        if not key:
            raise AssertionError(f"L85 cache item has no group_key: {path}")
        if key in result:
            raise AssertionError(f"duplicate L85 cache group: {key}")
        if item.get("labels_in_cache") or item.get("candidate_deletion") or item.get("candidate_truncation"):
            raise AssertionError(f"invalid label-free L85 cache item: {key}")
        result[key] = path
        del item
    if not result:
        raise AssertionError(f"empty L85 cache: {cache_root}")
    return result


def label_free_record(batch: Any, row: dict[str, Any], output: dict[str, torch.Tensor], checkpoint: dict[str, Any]) -> dict[str, Any]:
    n = len(batch.row_offsets)
    arrays = {name: output[name][0].detach().float().cpu().numpy().astype(np.float64).tolist()
              for name in ("candidate_energy", "r_static", "r_total")}
    prior = output["candidate_prior"].detach().float().cpu().numpy()
    if prior.shape == (n,):
        arrays["candidate_prior"] = prior.astype(np.float64).tolist()
    elif prior.shape == (1, n):
        arrays["candidate_prior"] = prior[0].astype(np.float64).tolist()
    else:
        raise AssertionError(f"L89 fixed candidate_prior shape drift: {batch.unit_key}: {prior.shape}")
    if any(len(value) != n or not np.isfinite(np.asarray(value, dtype=np.float64)).all() for value in arrays.values()):
        raise AssertionError(f"L89 fixed score length/finite drift: {batch.unit_key}")
    return {
        "format": "locatemot-l89-fixed-score-record-v1", "fixed_eval_order": int(row["fixed_eval_order"]),
        "evaluation_partition": str(row["evaluation_partition"]), "checkpoint": checkpoint,
        "unit_key": str(batch.unit_key), "dataset": str(batch.dataset), "video": str(batch.video),
        "query_id": int(batch.query_id), "frame_id": int(batch.frame_id), "candidate_count": n,
        "row_offsets": [int(x) for x in batch.row_offsets], "row_keys": [list(x) for x in batch.row_keys],
        "candidate_indices": [int(x) for x in batch.candidate_indices], "track_ids": [int(x) for x in batch.track_ids],
        "pool_ids": [int(x) for x in batch.pool_ids], "score": arrays["candidate_energy"],
        "candidate_energy": arrays["candidate_energy"], "r_static": arrays["r_static"],
        "r_total": arrays["r_total"], "candidate_prior": arrays["candidate_prior"],
        "presence_logit": float(output["presence_logit"][0].detach().float().cpu()),
        "null_logit": float(output["null_logit"][0].detach().float().cpu()),
        "future_history_count": int((batch.history_frame_ids > int(batch.frame_id)).sum()),
        "labels_attached": False, "candidate_rows_retained": True, "candidate_deletion": False,
        "candidate_truncation": False, "finite_scores": True,
    }


def attach(record: dict[str, Any], batch: Any, bank: L80BankStore) -> None:
    full = load_full_unit_for_labels(str(batch.unit_key))
    labels = bank.attach_labels(batch, full)
    record.update({
        "labels": [bool(x) for x in labels["labels"].tolist()],
        "target_ids": [str(x) for x in labels["target_ids"]],
        "candidate_gt": [None if x is None else str(x) for x in labels["sidecar_candidate_gt"]],
        "positive_indices": [int(x) for x in labels["positive_indices"]], "positive_count": int(labels["positive_count"]),
        "target_present": bool(labels["target_present"]), "candidate_present": bool(labels["candidate_present"]),
        "coverage_mask": bool(labels["coverage_mask"]), "category": str(labels["category"]),
        "declared_category": str(labels.get("declared_category", "unknown")), "label_source": str(labels["label_source"]),
        "labels_attached": True, "labels_attached_after_feature_construction": True,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--z1-cache", type=Path, default=Z1_CACHE)
    parser.add_argument("--language-cache", type=Path, default=LANG_CACHE)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L89 fixed semantic output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv]); started = time.perf_counter()
    store = None
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong L89 cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        selection = json.loads(args.selection.resolve().read_text())
        if selection.get("status") != "complete" or not selection.get("selection_frozen_before_fixed_validation"):
            raise AssertionError("L89 checkpoint/rule selection not frozen")
        final = selection["final_selection"]
        checkpoint_path = Path(str(final["checkpoint_info"]["path"])).resolve()
        if sha256_file(checkpoint_path) != str(final["checkpoint_info"]["sha256"]):
            raise AssertionError("selected checkpoint SHA drift")
        package = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = L89FullRMOT(L89Config(**package["model_config"])).to(device=args.device, dtype=torch.float32)
        loaded = model.load_state_dict(package["model_state_dict"], strict=True)
        if loaded.missing_keys or loaded.unexpected_keys:
            raise AssertionError(f"L89 strict reload failed: {loaded}")
        model.eval(); device = torch.device(args.device)
        if device.type == "cuda":
            torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
        language = L89LanguageTokenCache(args.language_cache.resolve())
        store = L80BankStore(max_history=8)
        rows = fixed_order()
        group_cache = index_cache_items(args.z1_cache.resolve())
        records: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            batch = store.build_unit(row)
            group_key = f"{batch.dataset}|{batch.video}|{batch.frame_id}"
            if group_key not in group_cache:
                raise KeyError(f"L85 fixed group cache missing: {group_key}")
            item = torch.load(group_cache[group_key], map_location="cpu", weights_only=False)
            qids = [int(x) for x in item["query_ids"]]
            qindex = qids.index(int(batch.query_id))
            tokens, mask = language_for(language, row, device)
            with torch.inference_mode():
                output = model(
                    item["z1"][qindex:qindex + 1].float().to(device), tokens, mask,
                    item["text_global"][qindex:qindex + 1].float().to(device), item["frame_global"][qindex:qindex + 1].float().to(device),
                    batch.observations.float().to(device), batch.history_observations.float().to(device), batch.history_mask.to(device),
                    batch.history_frame_ids.to(device), batch.frame_id, temporal_enabled=True,
                )
            record = label_free_record(batch, row, output, final["checkpoint_info"])
            records.append(record)
            del item, tokens, mask, output, batch
            gc.collect()
        if len(records) != 40 or [int(x["fixed_eval_order"]) for x in records] != list(range(40)):
            raise AssertionError("L89 fixed label-free order drift")
        preselection = {
            "format": "locatemot-l89-fixed-preselection-v1", "status": "complete", "record_count": 40,
            "forbidden_label_fields": sorted(FORBIDDEN),
            "forbidden_fields_absent": [not FORBIDDEN.intersection(row) for row in records],
            "all_forbidden_fields_absent": all(not FORBIDDEN.intersection(row) for row in records),
            "candidate_lengths_complete": all(len(row["score"]) == int(row["candidate_count"]) for row in records),
            "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "selection_frozen_before_fixed_labels": True, "selection": str(args.selection.resolve()),
            "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "no_hota_or_trackeval": True,
        }
        write_json(out / "preselection_label_isolation.json", preselection)
        if not preselection["all_forbidden_fields_absent"]:
            raise AssertionError("fixed preselection label leak")
        for index in range(16):
            batch = store.build_unit(rows[index]); attach(records[index], batch, store)
        calibration = records[:16]
        rule_name = str(final["rule"])
        selected_rule = final["rule_object"]
        thresholds = {name: float(selected_rule[name]) for name in ("candidate_threshold", "presence_threshold", "null_margin")}
        calibration_metrics = metric(calibration, **thresholds)
        for index in range(16, 40):
            batch = store.build_unit(rows[index]); attach(records[index], batch, store)
        validation = records[16:]
        validation_metrics = metric(validation, **thresholds)
        gate_metrics = validation_metrics
        checks = {
            "hard_improvement": float(gate_metrics["legacy_row_hard_violation"]) <= L29["hard_violation"] - 0.05,
            "recall_floor": float(gate_metrics["legacy_candidate_recall"]) >= 0.7233333,
            "precision_floor": float(gate_metrics["legacy_candidate_precision"]) >= 0.0830188679,
            "fp_per_frame_ceiling": float(gate_metrics["legacy_fp_per_frame"]) <= 11.125,
            "predictions_per_positive_ceiling": float(gate_metrics["legacy_predictions_per_positive"]) <= 4.069,
            "multi_positive_floor": gate_metrics["legacy_row_multi_positive_recall"] is not None and float(gate_metrics["legacy_row_multi_positive_recall"]) >= 0.7894444,
            "inactive_nonuniversal": float(gate_metrics["inactive_false_acceptance"]) < 1.0,
            "complete_finite_rows": bool(gate_metrics["candidate_rows_retained"] and gate_metrics["finite_scores"] and not gate_metrics["candidate_deletion"] and not gate_metrics["candidate_truncation"]),
        }
        semantic = {
            "format": "locatemot-l89-fixed-semantic-v1", "status": "complete",
            "evidence_type": "fixed 16 calibration / 24 validation diagnostic after internal fit/dev selection",
            "checkpoint": final["checkpoint_info"], "selection": selection,
            "selection_frozen_before_fixed_validation": True, "rule": rule_name, "thresholds": thresholds,
            "calibration": calibration_metrics, "validation": validation_metrics,
            "l29_teacher": {"evidence_type": "immutable accepted historical control", "validation": L29},
            "l87a_historic": {"V1_HOTA": 28.5752, "V2_HOTA": 22.1300},
            "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "record_count": 40, "calibration_count": 16, "validation_count": 24,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "no_hota_or_trackeval": True, "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
        }
        gate = {"format": "locatemot-l89-fixed-gate-v1", "status": "complete",
                "decision": "semantic_gate_pass" if all(checks.values()) else "semantic_gate_fail",
                "checks": checks, "thresholds": thresholds, "rule": rule_name,
                "baseline": L29, "screening_gt_used": False, "official_test_labels_read": False,
                "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False, "no_hota_or_trackeval": True}
        with (out / "score_records.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_json(out / "semantic.json", semantic)
        write_json(out / "gate_decision.json", gate)
        provenance = {
            "format": "locatemot-l89-fixed-semantic-provenance-v1", "status": "complete",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD, "seed": SEED,
            "inputs": {"selection": str(args.selection.resolve()), "selection_sha256": sha256_file(args.selection.resolve()),
                       "z1_cache": str(args.z1_cache.resolve()), "language_cache": str(args.language_cache.resolve()),
                       "l62_fixed_rows": str(L62_ROWS), "l62_fixed_rows_sha256": sha256_file(L62_ROWS),
                       "manifest_sha256": MANIFEST_SHA},
            "outputs": {"semantic": str((out / "semantic.json").resolve()), "gate": str((out / "gate_decision.json").resolve()),
                        "score_records": str((out / "score_records.jsonl").resolve())},
            "label_boundary": "all 40 score rows were built and scored before labels; calibration labels attached before frozen-rule report, validation labels attached afterward",
            "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False, "no_hota_or_trackeval": True,
        }
        write_json(out / "provenance.json", provenance)
        write_json(out / "status.json", {"format": "locatemot-l89-fixed-semantic-v1", "status": "complete",
                                          "decision": gate["decision"], "record_count": 40, "calibration_count": 16, "validation_count": 24,
                                          "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                                          "hota_trackeval_run": False, "next_action": "run legal internal dev/full-video TrackEval matrix"})
        return 0
    except Exception:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L89 fixed semantic — INCOMPLETE\n\n" + trace, encoding="utf-8")
        write_json(out / "status.json", {"format": "locatemot-l89-fixed-semantic-v1", "status": "incomplete", "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                          "failure_root_cause": "first traceback in INCOMPLETE.md", "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        raise
    finally:
        if store is not None:
            store._store._bank = None; store._store._text_cache = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
