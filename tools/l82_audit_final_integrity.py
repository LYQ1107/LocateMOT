#!/usr/bin/env python3
"""Finalize compact, read-only integrity evidence for the L82-A probe."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
OUT = ROOT / "outputs/l82/audit/final_integrity.json"
RETRY = ROOT / "outputs/l82/train/frozen_rank_probe_retry3"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
UIDM = ROOT / "outputs/l11/checkpoints/uidm_l11_main/step11000.pt"
L81 = ROOT / "outputs/l81/train/probe500_retry1/checkpoint_l81_step100.pt"
SOURCE_TRUTH = ROOT / "outputs/l81/preregister/source_of_truth_check.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def meta(path: Path, digest: bool = True) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "mtime_ns": path.stat().st_mtime_ns if path.is_file() else None,
        "sha256": sha256(path) if digest and path.is_file() else None,
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (int, str, bool)) or value is None:
        return True
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    return False


def main() -> int:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    required = [
        "representation_metrics.json", "rank_gate.json", "summary.json",
        "provenance.json", "status.json", "dev_group_metrics.jsonl",
    ]
    payloads = {name: load_json(RETRY / name) for name in required[:-1]}
    lines = [json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
             for line in (RETRY / required[-1]).read_text().splitlines() if line.strip()]
    representation_names = ["l59_fused_roi", "l81_candidate_evidence", "l82_candidate_reference"]
    rows_by_rep: dict[str, list[dict[str, Any]]] = {name: [] for name in representation_names}
    for row in lines:
        rows_by_rep.setdefault(str(row.get("representation")), []).append(row)
    row_counts = {name: len(rows_by_rep.get(name, [])) for name in representation_names}
    group_sets = {
        name: {str(row.get("group_key")) for row in rows_by_rep.get(name, [])}
        for name in representation_names
    }
    candidate_checks = []
    for row in lines:
        candidate_checks.append(
            int(row.get("candidate_count", -1)) == len(row.get("row_offsets", []))
            and len(row.get("row_keys_digest", [])) == int(row.get("query_count", -2))
            and len(row.get("row_offsets", [])) == len(set(row.get("row_offsets", [])))
            and row.get("candidate_deletion") is False
            and row.get("candidate_truncation") is False
        )
    traces = {}
    for path in sorted(RETRY.glob("loss_trace_*.json")):
        trace = load_json(path)
        traces[path.name] = {
            "steps": len(trace),
            "all_finite": all(bool(row.get("finite")) and finite_tree(row) for row in trace),
            "all_nonzero_gradient": all(bool(row.get("nonzero_gradient")) for row in trace),
        }
    gate = payloads["rank_gate.json"]
    summary = payloads["summary.json"]
    source_truth = load_json(SOURCE_TRUTH)
    expected_l69 = {
        row["name"]: row for row in source_truth["l69_feature_bank_manifest"]["files"]
    }
    current_l69 = {}
    l69_root = ROOT / "outputs/l69/attempt9/budget40_features/kitti"
    for name in sorted(expected_l69):
        current_l69[name] = meta(l69_root / name)
    l69_match = all(
        current_l69[name]["sha256"] == expected_l69[name]["sha256"]
        and current_l69[name]["bytes"] == expected_l69[name]["bytes"]
        for name in expected_l69
    )
    checks = {
        "required_files": all((RETRY / name).is_file() for name in required),
        "json_finite": all(finite_tree(value) for value in payloads.values()) and all(finite_tree(row) for row in lines),
        "three_representations_present": all(row_counts[name] == 138 for name in representation_names),
        "same_138_dev_groups": bool(group_sets) and all(
            group_sets[name] == group_sets[representation_names[0]] for name in representation_names
        ) and len(group_sets[representation_names[0]]) == 138,
        "candidate_rows_complete": bool(candidate_checks) and all(candidate_checks),
        "loss_traces_finite_and_nonzero": all(
            value["steps"] == 5240 and value["all_finite"] and value["all_nonzero_gradient"]
            for value in traces.values()
        ) and len(traces) == 3,
        "rank_gate_is_scientific_fail": gate.get("status") == "rank_representation_gate_fail",
        "world_size_at_most_four": int(summary.get("gpu_world_size", 99)) <= 4,
        "manifest_sha_unchanged": sha256(MANIFEST) == "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa",
        "uidm_sha_unchanged": sha256(UIDM) == "f6529743630e335b947118bf860cc2215a95315a4c551351e6866ea9e1076343",
        "l81_sha_unchanged": sha256(L81) == "2b6131584f4fe0fe018ee4494d61f481ac8eacb5f7ed7abe1125bc4a37c46915",
        "l69_manifest_unchanged": l69_match,
        "no_screening_or_official_test": payloads["provenance.json"].get("screening_gt_used") is False and payloads["provenance.json"].get("official_test_labels_read") is False,
        "ordinary_mot_ovmot_untouched": payloads["provenance.json"].get("ordinary_mot_ovmot_touched") is False,
    }
    result = {
        "format": "locatemot-l82-final-integrity-v1",
        "status": "complete" if all(checks.values()) else "invalid",
        "stage": "phase_d_frozen_representation_rank_probe",
        "cwd": str(ROOT),
        "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745",
        "command": " ".join([os.environ.get("PYTHON", "python"), "tools/l82_audit_final_integrity.py"]),
        "checks": checks,
        "row_counts": row_counts,
        "dev_group_count": len(next(iter(group_sets.values()))) if group_sets else 0,
        "trace_checks": traces,
        "rank_gate_status": gate.get("status"),
        "rank_gate_failed_checks": gate.get("failed_checks", []),
        "frozen_inputs": {"manifest": meta(MANIFEST), "uidm": meta(UIDM), "l81": meta(L81), "l69": current_l69},
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "hota_trackeval_run": False,
        "candidate_deletion": False,
        "candidate_truncation": False,
        "token_span_region_alignment": "UNALIGNED",
        "static_motion_alignment": "UNALIGNED",
        "next_action": "STOPPED_PENDING_SUPERVISOR_REVIEW; do not run Phase E within L82",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "path": str(OUT), "checks": checks}, sort_keys=True))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
