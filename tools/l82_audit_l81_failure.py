#!/usr/bin/env python3
"""Machine-readable audit of the frozen L81 implementation and failure.

No model is instantiated and no new labels are opened.  The audit combines
static source facts with immutable L81 JSON evidence so a later L82 report can
separate implementation validity from semantic failure.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
L81_MODEL = ROOT / "locatemot/models/l81_hierarchical_early_fusion.py"
L80_LOSS = ROOT / "locatemot/rmot/l80_losses.py"
L81_EVAL = ROOT / "outputs/l81/eval/semantic_16cal24val/semantic.json"
L81_GATE = ROOT / "outputs/l81/eval/semantic_16cal24val/gate_decision.json"
L81_CONTRACT = ROOT / "outputs/l81/audit/representation_contract_retry7/contract.json"
L81_GRAD = ROOT / "outputs/l81/audit/representation_contract_retry7/gradient_report.json"
L81_SENS = ROOT / "outputs/l81/audit/representation_contract_retry7/sensitivity_report.json"
L81_PROBE = ROOT / "outputs/l81/train/probe500_retry1/metrics_l81_fit.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def meta(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "mtime_ns": path.stat().st_mtime_ns if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None}


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def source_facts(path: Path) -> dict[str, Any]:
    text = path.read_text()
    tree = ast.parse(text, filename=str(path))
    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    methods = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    return {
        "file": meta(path), "classes": classes, "methods": methods,
        "contains_set_competition": "_set_competition" in text or "RelationAwareSetLayer" in text,
        "contains_history": "history" in text.lower(),
        "contains_null_head": "null_head" in text,
        "contains_cardinality_head": "cardinality_head" in text,
        "contains_multiple_canonical_heads": all(name in text for name in (
            "track_head", "continuation_head", "quality_head", "membership_head")),
        "forbidden_id_strings_present_only_as_contract": all(name in text for name in (
            "source_id", "pool_id", "query_id", "track_id", "candidate_index")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L82 L81 audit output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command = " ".join([sys.executable] + sys.argv)
    contract = read(L81_CONTRACT)
    gate = read(L81_GATE)
    semantic = read(L81_EVAL)
    probe = read(L81_PROBE)
    grad = read(L81_GRAD)
    sens = read(L81_SENS)
    l29 = semantic.get("gate", {}).get("l29_validation_control", {})
    selected = semantic.get("selected_metrics", {})
    # Older L81 JSON stores the selected metric under a nested method object;
    # retain both raw evidence and a normalized lookup without editing it.
    if not selected:
        selected = semantic.get("methods", {}).get("step100", {}).get("validation", {})
    report = {
        "format": "locatemot-l82-l81-code-failure-audit-v1",
        "status": "complete", "stage": "phase_pre_audit",
        "command": command, "cwd": str(ROOT),
        "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745",
        "inputs": {
            "manifest": meta(MANIFEST), "l81_model": meta(L81_MODEL), "l80_loss": meta(L80_LOSS),
            "l81_contract": meta(L81_CONTRACT), "l81_gradient": meta(L81_GRAD),
            "l81_sensitivity": meta(L81_SENS), "l81_probe": meta(L81_PROBE),
            "l81_semantic": meta(L81_EVAL), "l81_gate": meta(L81_GATE),
        },
        "manifest_sha_matches": sha256_file(MANIFEST) == EXPECTED_MANIFEST_SHA,
        "static_model_facts": source_facts(L81_MODEL),
        "static_loss_facts": {
            "file": meta(L80_LOSS),
            "contains_balanced_bce": "balanced_bce" in L80_LOSS.read_text(),
            "contains_pairwise": "pair" in L80_LOSS.read_text(),
            "contains_listwise": "listwise" in L80_LOSS.read_text(),
            "contains_minimum_positive": "minimum" in L80_LOSS.read_text(),
            "contains_inactive_null": "inactive" in L80_LOSS.read_text() and "null" in L80_LOSS.read_text(),
            "present_uncovered_masked": "present_uncovered" in L80_LOSS.read_text(),
        },
        "immutable_contract_facts": {
            "contract_status": contract.get("status"),
            "candidate_rows_retained": contract.get("candidate_rows_retained"),
            "candidate_deletion": contract.get("candidate_deletion"),
            "candidate_truncation": contract.get("candidate_truncation"),
            "history_future_rows": contract.get("history_future_rows"),
            "strict_reload": contract.get("strict_reload"),
            "visual_forward_count": contract.get("visual_forward_count"),
            "expression_candidate_delta": sens.get("expression_candidate_logit_delta_max", sens.get("max_candidate_logit_delta")),
            "marker_candidate_delta": sens.get("marker_candidate_logit_delta_max", sens.get("max_candidate_logit_delta_marker")),
            "candidate_logit_std": sens.get("candidate_logit_std"),
            "gradient_report_status": grad.get("status"),
            "probe_steps": probe.get("steps"),
            "probe_finite_steps": probe.get("finite_steps"),
            "probe_nonzero_gradient_steps": probe.get("nonzero_gradient_steps"),
        },
        "semantic_failure_facts": {
            "gate_decision": gate.get("decision", gate.get("status")),
            "l29_validation_control": l29,
            "raw_gate_checks": gate.get("checks", {}),
            "semantic_evidence_path": str(L81_EVAL),
            "selected_step": gate.get("selected_step", semantic.get("selected_step")),
            "selected_method": gate.get("selected_method", semantic.get("selected_method")),
            "selected_metric_lookup": selected,
            "probe_domains": probe.get("domains_seen", probe.get("domain_counts")),
            "probe_categories": probe.get("categories_seen", probe.get("category_counts")),
        },
        "root_cause": {
            "status": "query_candidate_correspondence_ceiling_insufficient",
            "implementation_contract_failure": False,
            "candidate_deletion_or_truncation": False,
            "first_actionable": "held-out query-candidate correspondence/generalization and output-volume calibration remain insufficient; L81 sensitivity is not an intrinsic frozen-representation ceiling",
            "not_supported_as_root_cause": [
                "missing fit-only expression supervision",
                "candidate deletion/truncation",
                "validation threshold rescue opportunity",
                "coverage-only explanation (prior L69/L76 oracle context is adequate at unit/target level)",
            ],
        },
        "evidence_class": "implementation contract + fit probe + fixed calibration/validation semantic failure; not oracle/HOTA/TrackEval/screening",
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "training_run": False,
        "hota_trackeval_run": False, "candidate_deletion": False,
        "candidate_truncation": False, "token_span_region_alignment": "UNALIGNED",
        "static_motion_alignment": "UNALIGNED", "failure_root_cause": None,
        "next_action": "complete L82 Phase A exact fit-only expression matrix before any architecture code",
        "elapsed_sec": time.perf_counter() - started,
    }
    out_file = out / "l81_code_failure_audit.json"
    out_file.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    (out / "status.json").write_text(json.dumps({
        "format": "locatemot-l82-status-v1", "status": "complete", "stage": "phase_pre_audit",
        "command": command, "outputs": [str(out_file)], "failure_root_cause": None,
        "next_action": report["next_action"], "screening_gt_used": False,
        "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
        "training_run": False, "hota_trackeval_run": False,
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
