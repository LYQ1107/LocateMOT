#!/usr/bin/env python3
"""Compact final L80 ceiling summary after the bounded repair sequence.

This audit reads only completed development artifacts.  It does not forward a
model, open screening/test labels, train, or create any feature cache.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
L76 = ROOT / "outputs/l76/audit/v2_candidate_coverage/coverage.json"
L63 = ROOT / "outputs/l63/audit/oracle_ceiling_final_retry1/oracle_ceiling.json"
R0_ROWS = ROOT / "outputs/l80/eval/semantic_16cal24val_r0_retry1/score_records.jsonl"
SEMANTIC = {
    "r0": ROOT / "outputs/l80/eval/semantic_16cal24val_r0_retry1/semantic.json",
    "r1": ROOT / "outputs/l80/eval/semantic_16cal24val_r1/semantic.json",
    "r2": ROOT / "outputs/l80/eval/semantic_16cal24val_r2/semantic.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def metric_fields(value: dict[str, Any]) -> dict[str, Any]:
    fields = ("candidate_recall", "candidate_precision", "fp_per_frame",
              "predictions_per_positive", "hard_violation", "multi_positive_recall",
              "empty_rate", "inactive_false_acceptance", "selected_rows",
              "true_positive_rows", "false_positive_rows", "false_negative_rows")
    return {key: value.get(key) for key in fields}


def fixed_oracle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dataset: dict[str, dict[str, Any]] = defaultdict(lambda: {"units": 0, "target_present": 0,
        "candidate_present": 0, "present_uncovered": 0, "candidate_rows": 0, "positive_rows": 0})
    by_category = Counter(); duplicate_indices = 0; duplicate_tracks = 0
    records = []
    for row in rows:
        labels = [int(x) for x in row["labels"]]
        if len(labels) != int(row["candidate_count"]):
            raise AssertionError(f"fixed oracle label length drift: {row['unit_key']}")
        if row.get("candidate_deletion") or row.get("candidate_truncation"):
            raise AssertionError(f"candidate deletion in fixed oracle: {row['unit_key']}")
        category = str(row.get("category", "unknown")); dataset = str(row["dataset"])
        values = by_dataset[dataset]
        values["units"] += 1; values["target_present"] += int(category != "inactive")
        values["candidate_present"] += int(bool(any(labels)))
        values["present_uncovered"] += int(category == "present_uncovered")
        values["candidate_rows"] += int(row["candidate_count"]); values["positive_rows"] += sum(labels)
        by_category[category] += 1
        indices = row.get("candidate_index_provenance", [])
        duplicate_indices += len(indices) - len(set(indices))
        tracks = row.get("track_id_provenance", [])
        duplicate_tracks += len(tracks) - len(set(tracks))
        records.append({"unit_key": row["unit_key"], "dataset": dataset, "video": row["video"],
                        "fixed_eval_order": int(row["fixed_eval_order"]), "category": category,
                        "candidate_count": int(row["candidate_count"]), "positive_count": int(sum(labels)),
                        "target_present": category != "inactive", "candidate_present": bool(any(labels)),
                        "oracle_gt_overlap_only_precision": 1.0 if sum(labels) else None,
                        "oracle_coverage_hit": bool(any(labels)) if category != "inactive" else None})
    target_present = sum(value["target_present"] for value in by_dataset.values())
    candidate_present = sum(value["candidate_present"] for value in by_dataset.values())
    candidate_rows = sum(value["candidate_rows"] for value in by_dataset.values())
    positive_rows = sum(value["positive_rows"] for value in by_dataset.values())
    return {
        "label": "GT_PRIVILEGED_ORACLE; fixed L80 development slice only",
        "units": len(rows), "target_present_units": target_present,
        "covered_target_present_units": candidate_present,
        "present_uncovered_units": target_present - candidate_present,
        "unit_coverage_ceiling": candidate_present / max(1, target_present),
        "candidate_rows": candidate_rows, "positive_rows": positive_rows,
        "gt_overlap_only_membership_precision": 1.0 if positive_rows else None,
        "gt_overlap_only_candidate_recall": 1.0 if positive_rows else 0.0,
        "category_counts": dict(by_category), "by_dataset": dict(by_dataset),
        "duplicate_candidate_index_rows": duplicate_indices,
        "duplicate_track_rows_descriptive": duplicate_tracks,
        "records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty final oracle output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable] + sys.argv)
    try:
        if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA changed")
        l76 = json.loads(L76.read_text()); prior_l63 = json.loads(L63.read_text())
        rows = read_jsonl(R0_ROWS)
        if len(rows) != 40 or [int(row["fixed_eval_order"]) for row in rows] != list(range(40)):
            raise AssertionError("R0 fixed score rows are not the immutable 40 order")
        fixed = fixed_oracle(rows)
        semantic_summary = {}
        for name, path in SEMANTIC.items():
            data = json.loads(path.read_text())
            selected = data["checkpoint_selection"]
            selected_method = str(selected["method"])
            selected_metrics = data["methods"][selected_method]
            semantic_summary[name] = {
                "decision": data["decision"], "selected_method": selected_method,
                "selected_step": int(selected["step"]),
                "validation_candidate_only": metric_fields(selected_metrics["validation_candidate_only"]),
                "validation_candidate_plus_null": metric_fields(selected_metrics["validation_candidate_plus_null"]),
                "all_validation_checkpoints": {
                    method: metric_fields(values["validation_candidate_plus_null"])
                    for method, values in data["methods"].items()
                },
            }
        v2 = l76["decision"]
        coverage = {
            "label": "GT_PRIVILEGED_ORACLE; immutable L76 full V2 validation coverage",
            "units": 768, "target_present_units": 576, "covered_units": 462,
            "unit_coverage": float(v2["full_unit_coverage"]), "target_level_micro_coverage": float(v2["full_target_level_micro_coverage"]),
            "present_uncovered_units": 114, "inactive_units": 192,
            "baseline_reproduced": bool(l76["baseline_reproduction"]["exact_match"]),
            "source": str(L76),
        }
        prior_identity = {
            "label": "prior immutable L63 oracle context; not a new L80 model result",
            "source": str(L63),
            "clip_same_gt_vs_same_frame_auc": prior_l63["identity_ceiling"]["clip"]["all_same_gt_vs_same_frame_different_gt"]["roc_auc"],
            "history_clip_same_gt_vs_same_frame_auc": prior_l63["identity_ceiling"]["history_clip"]["all_same_gt_vs_same_frame_different_gt"]["roc_auc"],
            "uidm_h_same_gt_vs_same_frame_auc": prior_l63["identity_ceiling"]["uidm_h"]["all_same_gt_vs_same_frame_different_gt"]["roc_auc"],
            "clip_pair_order_violation": prior_l63["identity_ceiling"]["clip"]["all_same_gt_vs_same_frame_different_gt"]["pair_order_violation"],
            "history_clip_pair_order_violation": prior_l63["identity_ceiling"]["history_clip"]["all_same_gt_vs_same_frame_different_gt"]["pair_order_violation"],
            "scope_note": "L63's audited fixed-slice identity ceiling is retained as context; no new dense feature audit was run here.",
        }
        all_semantic_fail = all(value["decision"] == "semantic_gate_fail" for value in semantic_summary.values())
        v2_adequate = coverage["unit_coverage"] >= 0.7233333 and coverage["target_level_micro_coverage"] >= 0.80
        r2_validation = semantic_summary["r2"]["validation_candidate_only"]
        automatic_decision = {
            "candidate_coverage_primary_blocker": not v2_adequate,
            "frozen_visual_identity_has_some_but_not_clean_separation": prior_identity["clip_same_gt_vs_same_frame_auc"] >= 0.75 and prior_identity["clip_pair_order_violation"] > 0.20,
            "all_l80_versions_failed_semantic_gate": all_semantic_fail,
            "r2_hard_negative_still_failed": r2_validation["hard_violation"] is None or r2_validation["hard_violation"] > 0.8666667,
            "null_only_blocker": False,
            "label": "l80_correspondence_ceiling_insufficient_after_r1_r2",
            "rule": "coverage adequate and repeated R0/R1/R2 held-out hard-negative gate failure; no NULL-only explanation",
            "next_action": "supervisor-approved new end-to-end hierarchical early-fusion/query-region representation audit; freeze L80 loss/threshold variants",
        }
        payload = {
            "format": "locatemot-l80-final-oracle-ceiling-v1", "status": "complete",
            "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
            "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745",
            "scope": "post-R0/R1/R2 development ceiling summary; no new model forward",
            "fixed_manifest_sha256": sha256_file(MANIFEST), "coverage_ceiling": coverage,
            "fixed_slice_gt_oracle": {key: value for key, value in fixed.items() if key != "records"},
            "prior_identity_ceiling_context": prior_identity,
            "l80_semantic_evidence": semantic_summary,
            "automatic_decision": automatic_decision,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "training_run": False,
            "hota_trackeval_run": False, "all_candidate_rows_retained": True,
            "persistent_dense_or_raw_cache_written": False,
            "oracle_not_model_or_hota": True,
        }
        write_json(out / "oracle_ceiling.json", payload)
        with (out / "records.jsonl").open("w") as handle:
            for record in fixed["records"]:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        with (out / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["version", "decision", "selected_method", "selected_step", "recall", "precision", "fp_per_frame", "predictions_per_positive", "hard_violation", "multi_positive_recall", "inactive_false_acceptance"])
            writer.writeheader()
            for version, value in semantic_summary.items():
                metric = value["validation_candidate_plus_null"]
                writer.writerow({"version": version, "decision": value["decision"], "selected_method": value["selected_method"], "selected_step": value["selected_step"], "recall": metric["candidate_recall"], "precision": metric["candidate_precision"], "fp_per_frame": metric["fp_per_frame"], "predictions_per_positive": metric["predictions_per_positive"], "hard_violation": metric["hard_violation"], "multi_positive_recall": metric["multi_positive_recall"], "inactive_false_acceptance": metric["inactive_false_acceptance"]})
        input_paths = [MANIFEST, L76, L63, R0_ROWS, *SEMANTIC.values()]
        write_json(out / "provenance.json", {"format": "locatemot-l80-final-oracle-provenance-v1", "status": "complete",
            "command": command, "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
            "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745",
            "inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in input_paths],
            "gt_use": "post-hoc development artifacts only; no new labels loaded",
            "coverage_source": str(L76), "identity_source": str(L63), "semantic_sources": {key: str(value) for key, value in SEMANTIC.items()},
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "training_run": False, "hota_trackeval_run": False,
            "raw_dense_cache_written": False, "candidate_deletion": False, "candidate_truncation": False})
        write_json(out / "status.json", {"format": "locatemot-l80-final-oracle-status-v1", "status": "complete",
            "command": command, "failure_root_cause": automatic_decision["label"],
            "next_action": automatic_decision["next_action"], "outputs": [str(out / name) for name in ("oracle_ceiling.json", "records.jsonl", "summary.csv")],
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "training_run": False, "hota_trackeval_run": False})
        return 0
    except Exception:
        (out / "INCOMPLETE.md").write_text("# L80 final oracle ceiling — INCOMPLETE\n\n" + __import__("traceback").format_exc() + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
