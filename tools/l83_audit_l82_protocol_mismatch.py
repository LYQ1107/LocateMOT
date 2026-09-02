#!/usr/bin/env python3
"""Machine-audited record of the L82 protocol mismatches preserved by L83."""
from __future__ import annotations

import ast
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
DEFAULT_OUT = ROOT / "outputs/l83/audit/l82_protocol_mismatch"
LOSSES = ROOT / "locatemot/rmot/l82_losses.py"
PROBE = ROOT / "locatemot/models/l82_rank_probe.py"
TRAIN = ROOT / "tools/l82_train_frozen_rank_probe.py"
GROUNDING = ROOT / "tools/l82_audit_grounding_interface.py"
NATIVE_DINO = Path("/data1/LWR/vranlee/LLM/mmdetection-3.3.0/mmdet/models/detectors/dino.py").resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def line_numbers(path: Path, needles: list[str]) -> dict[str, list[int]]:
    result = {needle: [] for needle in needles}
    for number, line in enumerate(lines(path), 1):
        for needle in needles:
            if needle in line:
                result[needle].append(number)
    return result


def ast_functions(path: Path) -> dict[str, ast.AST]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return {node.name: node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L83 output: {out}")
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    loss_source = LOSSES.read_text()
    probe_source = PROBE.read_text()
    train_source = TRAIN.read_text()
    grounding_source = GROUNDING.read_text()
    native_source = NATIVE_DINO.read_text()
    loss_fn = ast_functions(LOSSES).get("l82_rank_loss")
    aggregate_fn = ast_functions(TRAIN).get("aggregate_group_metrics")
    group_fn = ast_functions(TRAIN).get("group_metrics")
    checks: dict[str, dict[str, Any]] = {}

    required_loss = [
        "pos = interaction[q][labels[q]]",
        "neg = interaction[q][~labels[q]]",
        "_smooth_min(pos)",
        "floor_terms.append(F.softplus(-pos).mean())",
    ]
    checks["A_row_level_positive_negative_and_floor"] = {
        "passed": all(token in loss_source for token in required_loss),
        "function": "l82_rank_loss", "ast_lines": [getattr(loss_fn, "lineno", None), getattr(loss_fn, "end_lineno", None)],
        "evidence": line_numbers(LOSSES, required_loss),
    }
    checks["B_no_candidate_gt_in_primary_loss"] = {
        "passed": "candidate_gt" not in loss_source and "target_to_rows" not in loss_source,
        "function": "l82_rank_loss", "evidence": "candidate_gt/target_to_rows absent from l82_losses.py",
    }
    fake_auc = "1.0 if positive > negative else 0.5 if positive == negative else 0.0"
    checks["C_reported_auc_is_pairwise_concordance"] = {
        "passed": fake_auc in train_source and "roc_auc_score" not in train_source,
        "function": "aggregate_group_metrics", "ast_lines": [getattr(aggregate_fn, "lineno", None), getattr(aggregate_fn, "end_lineno", None)],
        "evidence": line_numbers(TRAIN, ["query_swap_auc", fake_auc]),
        "corrected_in_l83": "independent rank-based ROC-AUC",
    }
    checks["D_target_bag_r5_is_row_top5"] = {
        "passed": "top_order[:5]" in train_source,
        "function": "group_metrics", "ast_lines": [getattr(group_fn, "lineno", None), getattr(group_fn, "end_lineno", None)],
        "evidence": line_numbers(TRAIN, ["top_order =", "top_order[:5]", "candidate_targets"]),
        "corrected_in_l83": "unique target bags; rows are diagnostics only",
    }
    score_calls = probe_source.count("self._score(")
    checks["E_three_scores_share_self_score"] = {
        "passed": "self.score = nn.Sequential" in probe_source and score_calls >= 3,
        "function": "L82FactorizedRankProbe.forward", "evidence": {"self_score_calls": score_calls, "lines": line_numbers(PROBE, ["self.score =", "candidate_main =", "query_main ="])},
        "corrected_in_l83": "independent target-bag/representation probe; factorized heads remain conditional",
    }
    checks["F_candidate_seed_injects_reference_position"] = {
        "passed": "visual_seed, roi_audit = pool_memory_by_box" in grounding_source and "candidate_seed_with_reference" in grounding_source and "self.query_embedding.weight" in native_source,
        "function": "capture_candidate_state plus native pre_decoder", "evidence": {"wrapper": line_numbers(GROUNDING, ["visual_seed, roi_audit", "candidate_seed_with_reference"]), "native_dino": line_numbers(NATIVE_DINO, ["self.query_embedding.weight"])},
    }
    checks["G_fixed_reference_disables_refinement_and_self_attention_mask"] = {
        "passed": "passes no regression branches" in grounding_source and "self_attn_mask=None" in grounding_source and "reg_branches" in native_source,
        "function": "fixed_reference_decoder", "evidence": line_numbers(GROUNDING, ["passes no regression branches", "self_attn_mask=None", "reg_branches"]),
    }
    all_passed = all(bool(value["passed"]) for value in checks.values())
    payload = {
        "format": "locatemot-l83-l82-protocol-mismatch-audit-v1",
        "status": "complete" if all_passed else "l82_source_mismatch",
        "checks": checks,
        "inputs": {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): {"sha256": sha256(path), "path": str(path)} for path in (LOSSES, PROBE, TRAIN, GROUNDING, NATIVE_DINO)},
        "source_contract": {"l82_immutable": True, "l82_modified": False, "l83_is_new_namespace": True},
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        "candidate_deletion": False, "candidate_truncation": False,
        "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
        "command": "python tools/l83_audit_l82_protocol_mismatch.py",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "l82_protocol_mismatch.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary = [
        "# L82 protocol mismatch audit",
        "",
        f"Status: `{payload['status']}`.  L82 source files are immutable evidence; this audit does not edit them.",
        "",
        "| Check | Result | Evidence / L83 correction |",
        "|---|---|---|",
    ]
    labels = {
        "A_row_level_positive_negative_and_floor": "row-level pos/neg, smooth-min and positive floor",
        "B_no_candidate_gt_in_primary_loss": "no target-bag grouping in old primary loss",
        "C_reported_auc_is_pairwise_concordance": "reported AUC is pairwise concordance; L83 uses real ROC-AUC",
        "D_target_bag_r5_is_row_top5": "old R@5 is row top-5; L83 ranks unique bags",
        "E_three_scores_share_self_score": "candidate/query diagnostics share self.score",
        "F_candidate_seed_injects_reference_position": "candidate visual seed plus reference position vs native query embedding",
        "G_fixed_reference_disables_refinement_and_self_attention_mask": "fixed reference has no reg branches and self_attn_mask=None",
    }
    for key, value in checks.items():
        summary.append(f"| {labels[key]} | `{bool(value['passed'])}` | `{value.get('function')}` |")
    summary += ["", "The mismatch facts are the reason L83 is a new stage; no L82 result is overwritten.", ""]
    (ROOT / "reports/l83/L83_L82_PROTOCOL_MISMATCH_AUDIT.md").write_text("\n".join(summary))
    print(json.dumps({"status": payload["status"], "out": str(out / "l82_protocol_mismatch.json")}))
    return 0 if all_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
