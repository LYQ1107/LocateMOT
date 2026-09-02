#!/usr/bin/env python3
"""Read-only source-of-truth check for the registered L81 experiment.

The check intentionally runs before any L81 model or training code is created.
It records immutable input hashes and verifies the numerical facts that L81 is
allowed to inherit from L80/L69/L27.  It never changes an existing artifact.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
EXPECTED_CLIP = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
EXPECTED_UIDM = "f6529743630e335b947118bf860cc2215a95315a4c551351e6866ea9e1076343"
L29 = {
    "recall": 0.7333333333333333,
    "precision": 0.0830188679245283,
    "fp_per_frame": 10.125,
    "predictions_per_positive": 8.833333333333334,
    "hard_violation": 0.9166666666666666,
    "multi_positive_recall": 0.8194444444444443,
}
TOL = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def file_meta(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        return {"path": str(path), "exists": False}
    return {
        "path": str(path), "exists": True, "bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns, "sha256": sha256_file(path),
    }


def assert_close(actual: Any, expected: float, label: str, errors: list[str]) -> None:
    try:
        value = float(actual)
    except (TypeError, ValueError):
        errors.append(f"{label}: nonnumeric {actual!r}")
        return
    if abs(value - expected) > TOL:
        errors.append(f"{label}: expected {expected!r}, got {value!r}")


def l69_manifest() -> dict[str, Any]:
    root = ROOT / "outputs/l69/attempt9/budget40_features/kitti"
    files = []
    for path in sorted(root.glob("*.pt")):
        files.append({"name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "root": str(root), "file_count": len(files), "files": files,
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def run() -> int:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd().resolve()}")
    out_json = ROOT / "outputs/l81/preregister/source_of_truth_check.json"
    out_report = ROOT / "reports/l81/L81_SOURCE_OF_TRUTH_CHECK.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    manifest = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
    clip = Path("/home/lwr/.cache/clip/ViT-B-16.pt")
    uidm = ROOT / "outputs/l11/checkpoints/uidm_l11_main/step11000.pt"
    l62_rows = ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
    required_paths = [
        ROOT / "locatemot/models/l80_raw_region_correspondence.py",
        ROOT / "locatemot/rmot/l80_data.py",
        ROOT / "locatemot/rmot/l80_runtime.py",
        ROOT / "locatemot/rmot/l80_losses.py",
        ROOT / "locatemot/rmot/l80_r2_losses.py",
        ROOT / "tools/train_l80_v12_joint.py",
        ROOT / "tools/eval_l80_v12.py",
        ROOT / "locatemot/models/l79_hierarchical_correspondence.py",
        ROOT / "locatemot/models/l59_fused_roi_scorer.py",
        ROOT / "locatemot/models/l75_candidate_marked_vlm.py",
        ROOT / "locatemot/models/l77_region_cross_attention.py",
        manifest, l62_rows, clip, uidm,
        ROOT / "outputs/l69/audit/budget40_bank_contract_attempt13_full/provenance.json",
        ROOT / "outputs/l69/audit/budget40_bank_contract_attempt13_full/contract.json",
        ROOT / "outputs/l80/train/r0_full_fit60/metrics_l80_fit.json",
        ROOT / "outputs/l80/train/r0_full_fit60/config.json",
        ROOT / "outputs/l80/eval/semantic_16cal24val_r0_retry1/gate_decision.json",
        ROOT / "outputs/l80/eval/semantic_16cal24val_r1/gate_decision.json",
        ROOT / "outputs/l80/eval/semantic_16cal24val_r2/gate_decision.json",
        ROOT / "outputs/l80/audit/final_oracle_ceiling/oracle_ceiling.json",
    ]
    metas = {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): file_meta(path)
             for path in required_paths}
    for label, meta in metas.items():
        if not meta.get("exists"):
            errors.append(f"missing required source-of-truth input: {label}")
    if metas[str(manifest.relative_to(ROOT))].get("sha256") != EXPECTED_MANIFEST:
        errors.append("fixed manifest SHA mismatch")
    if metas[str(clip)].get("sha256") != EXPECTED_CLIP:
        errors.append("CLIP SHA mismatch")
    if metas[str(uidm.relative_to(ROOT))].get("sha256") != EXPECTED_UIDM:
        errors.append("UIDM step11000 SHA mismatch")

    r0_metrics = read_json(ROOT / "outputs/l80/train/r0_full_fit60/metrics_l80_fit.json")
    if r0_metrics.get("status") != "complete":
        errors.append(f"L80 R0 fit status drift: {r0_metrics.get('status')!r}")
    for field, expected in {
        "steps": 318840, "epochs": 60, "fit_units_available": 5314,
        "finite_steps": 318840, "nonzero_gradient_steps": 318840,
        "candidate_key_drift": 0,
    }.items():
        if r0_metrics.get(field) != expected:
            errors.append(f"L80 R0 {field}: expected {expected!r}, got {r0_metrics.get(field)!r}")
    if r0_metrics.get("candidate_deletion") is not False or r0_metrics.get("candidate_truncation") is not False:
        errors.append("L80 R0 candidate deletion/truncation flag drift")
    if r0_metrics.get("model", {}).get("trainable_parameter_count") != 3664694:
        errors.append("L80 R0 trainable parameter count drift")
    if r0_metrics.get("seed") != 20260829 or r0_metrics.get("learning_rate") != 0.0002:
        errors.append("L80 R0 seed or learning rate drift")

    gate_paths = {
        "r0": ROOT / "outputs/l80/eval/semantic_16cal24val_r0_retry1/gate_decision.json",
        "r1": ROOT / "outputs/l80/eval/semantic_16cal24val_r1/gate_decision.json",
        "r2": ROOT / "outputs/l80/eval/semantic_16cal24val_r2/gate_decision.json",
    }
    gates = {name: read_json(path) for name, path in gate_paths.items()}
    for name, gate in gates.items():
        if gate.get("status") != "semantic_gate_fail":
            errors.append(f"L80 {name} gate status drift: {gate.get('status')!r}")
        control = gate.get("l29_validation_control", {})
        for field, expected in L29.items():
            assert_close(control.get(field), expected, f"L29 control {name}.{field}", errors)
    r0_oracle = read_json(ROOT / "outputs/l80/audit/final_oracle_ceiling/oracle_ceiling.json")
    coverage = r0_oracle.get("coverage_ceiling", {})
    for field, expected in {
        "units": 768, "target_present_units": 576, "covered_units": 462,
        "present_uncovered_units": 114, "inactive_units": 192,
    }.items():
        if coverage.get(field) != expected:
            errors.append(f"L69/L76 coverage {field}: expected {expected!r}, got {coverage.get(field)!r}")
    for field, expected in {"unit_coverage": 0.8020833333333334, "target_level_micro_coverage": 0.8610846812559467}.items():
        assert_close(coverage.get(field), expected, f"coverage {field}", errors)
    if r0_oracle.get("automatic_decision", {}).get("candidate_coverage_primary_blocker") is not False:
        errors.append("oracle decision no longer says coverage is not the primary blocker")
    if r0_oracle.get("automatic_decision", {}).get("all_l80_versions_failed_semantic_gate") is not True:
        errors.append("oracle decision does not record all L80 variants as failed")

    git_status = subprocess.run(["git", "status", "--short"], cwd=ROOT, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout
    status_lines = git_status.splitlines()
    bank_manifest = l69_manifest()
    result = {
        "format": "locatemot-l81-source-of-truth-check-v1",
        "status": "complete" if not errors else "source_of_truth_mismatch",
        "command": " ".join(sys.argv), "cwd": str(ROOT),
        "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745",
        "fixed_manifest": {"path": str(manifest), "expected_sha256": EXPECTED_MANIFEST,
                           "actual_sha256": metas[str(manifest.relative_to(ROOT))].get("sha256")},
        "frozen_inputs": metas,
        "l69_feature_bank_manifest": bank_manifest,
        "source_facts": {
            "l29_validation_control": L29,
            "l80_r0_fit": {"steps": r0_metrics.get("steps"), "epochs": r0_metrics.get("epochs"),
                           "finite_steps": r0_metrics.get("finite_steps"),
                           "nonzero_gradient_steps": r0_metrics.get("nonzero_gradient_steps"),
                           "trainable_parameters": r0_metrics.get("model", {}).get("trainable_parameter_count")},
            "l80_gate_status": {name: gate.get("status") for name, gate in gates.items()},
            "l69_v2_coverage": coverage,
            "l80_final_decision": r0_oracle.get("automatic_decision", {}),
        },
        "git_status_short": {
            "line_count": len(status_lines),
            "sha256": hashlib.sha256(git_status.encode()).hexdigest(),
            "preview": status_lines[:40],
        },
        "errors": errors,
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "hota_trackeval_run": False,
        "training_run": False,
        "next_action": "write L81 preregistration and implementation" if not errors else "stop before L81 implementation",
    }
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")

    report = [
        "# L81 source-of-truth check",
        "",
        f"- Status: `{result['status']}`",
        f"- Project root: `{ROOT}`",
        "- Luna thread: `01a02014-fce8-7f51-8414-e7ed6ab44745`",
        f"- Command: `{' '.join(sys.argv)}`",
        "",
        "## Frozen numerical facts",
        "",
        "- Fixed manifest SHA matches the registered value.",
        "- L80 R0 fit: 5,314 fit units, 60 epochs, 318,840 finite/nonzero-gradient updates, 3,664,694 trainable parameters, candidate deletion/truncation false.",
        "- L80 R0/R1/R2 fixed semantic status: all `semantic_gate_fail`.",
        "- L69/L76 V2 coverage: 768 units, 576 target-present, 462 covered, unit coverage 0.8020833, target-micro 0.8610847, 114 present-uncovered, 192 inactive.",
        "- Immutable L29 validation control: recall 0.7333333, precision 0.0830189, FP/frame 10.125, predictions/positive 8.8333, hard violation 0.9166667, multi-positive recall 0.8194444.",
        "",
        "## Hashes and status",
        "",
        f"- Manifest: `{metas[str(manifest.relative_to(ROOT))].get('sha256')}`",
        f"- CLIP ViT-B/16: `{metas[str(clip)].get('sha256')}`",
        f"- UIDM step11000: `{metas[str(uidm.relative_to(ROOT))].get('sha256')}`",
        f"- L69 feature-bank file count: `{bank_manifest['file_count']}`; manifest digest: `{bank_manifest['manifest_sha256']}`",
        f"- Existing worktree status lines: `{len(status_lines)}`; no reset/clean/checkout/commit/push was performed.",
        "",
        "## Result",
        "",
    ]
    if errors:
        report.extend(["The registered facts do not match:", "", *[f"- {item}" for item in errors], "", "L81 implementation and training are not authorized."])
    else:
        report.append("All registered source-of-truth checks passed. L81 may proceed to its pre-registration and label-free contract, using the frozen inputs recorded in `outputs/l81/preregister/source_of_truth_check.json`.")
    report.extend([
        "", "## Evidence boundaries", "",
        "- This check is implementation/provenance only; no L81 model, training, screening, official-test, HOTA or TrackEval run occurred.",
        "- `screening_gt_used=false`", "- `official_test_labels_read=false`",
        "- `ordinary_mot_ovmot_touched=false`", "- `hota_trackeval_run=false`",
    ])
    out_report.write_text("\n".join(report) + "\n")
    print(json.dumps({"status": result["status"], "errors": errors, "output": str(out_json)}, ensure_ascii=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(run())
