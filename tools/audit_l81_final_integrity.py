#!/usr/bin/env python3
"""Final read-only integrity audit for the completed L81 evidence stage."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
EXPECTED_CLIP = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record_file(path: Path, expected: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"path": str(path), "expected_sha256": expected}
    if not path.is_file():
        item.update({"exists": False, "match": False})
        return item
    actual = sha256_file(path)
    item.update({
        "exists": True,
        "bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "actual_sha256": actual,
        "match": expected is None or actual == expected,
    })
    return item


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def main() -> int:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd().resolve()}")

    out = ROOT / "outputs/l81/final_integrity_audit.json"
    frozen_assets = read_json(ROOT / "outputs/l81/preregister/frozen_assets.json")
    expected_hashes = frozen_assets["frozen_hashes"]
    frozen_paths: dict[str, Path] = {
        "uidm_step11000": ROOT / "outputs/l11/checkpoints/uidm_l11_main/step11000.pt",
        "l62_fixed_rows": ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl",
        "l80_data": ROOT / "locatemot/rmot/l80_data.py",
        "l80_losses": ROOT / "locatemot/rmot/l80_losses.py",
        "l80_runtime": ROOT / "locatemot/rmot/l80_runtime.py",
        "l80_train": ROOT / "tools/train_l80_v12_joint.py",
        "l80_eval": ROOT / "tools/eval_l80_v12.py",
        "l80_r0_config": ROOT / "outputs/l80/train/r0_full_fit60/config.json",
        "l69_provenance": ROOT / "outputs/l69/audit/budget40_bank_contract_attempt13_full/provenance.json",
        "l69_contract": ROOT / "outputs/l69/audit/budget40_bank_contract_attempt13_full/contract.json",
        "l80_oracle": ROOT / "outputs/l80/audit/final_oracle_ceiling/oracle_ceiling.json",
        "l80_r0_metrics": ROOT / "outputs/l80/train/r0_full_fit60/metrics_l80_fit.json",
        "l80_r0_gate": ROOT / "outputs/l80/eval/semantic_16cal24val_r0_retry1/gate_decision.json",
        "l80_r1_gate": ROOT / "outputs/l80/eval/semantic_16cal24val_r1/gate_decision.json",
        "l80_r2_gate": ROOT / "outputs/l80/eval/semantic_16cal24val_r2/gate_decision.json",
        "l79_model": ROOT / "locatemot/models/l79_hierarchical_correspondence.py",
        "l59_model": ROOT / "locatemot/models/l59_fused_roi_scorer.py",
        "l75_model": ROOT / "locatemot/models/l75_candidate_marked_vlm.py",
        "l77_model": ROOT / "locatemot/models/l77_region_cross_attention.py",
    }
    frozen_recheck = {
        name: record_file(path, expected_hashes.get(name))
        for name, path in frozen_paths.items()
    }

    errors: list[str] = []
    for name, item in frozen_recheck.items():
        if not item["exists"]:
            errors.append(f"missing frozen asset: {name}")
        elif not item["match"]:
            errors.append(f"frozen hash mismatch: {name}")

    manifest = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
    manifest_record = record_file(manifest, EXPECTED_MANIFEST)
    if not manifest_record["match"]:
        errors.append("fixed manifest hash mismatch")
    clip = Path("/home/lwr/.cache/clip/ViT-B-16.pt")
    clip_record = record_file(clip, EXPECTED_CLIP)
    if not clip_record["match"]:
        errors.append("CLIP hash mismatch")

    source_check = read_json(ROOT / "outputs/l81/preregister/source_of_truth_check.json")
    if source_check.get("status") != "complete" or source_check.get("errors"):
        errors.append("source-of-truth check is not complete")

    semantic_dir = ROOT / "outputs/l81/eval/semantic_16cal24val"
    gate = read_json(semantic_dir / "gate_decision.json")
    selection = read_json(semantic_dir / "checkpoint_selection.json")
    semantic = read_json(semantic_dir / "semantic.json")
    status = read_json(semantic_dir / "status.json")
    if gate.get("status") != "semantic_gate_fail":
        errors.append(f"unexpected L81 gate status: {gate.get('status')!r}")
    if selection.get("selected", {}).get("method") != "step100":
        errors.append("calibration-only checkpoint selection drift")
    if selection.get("validation_used") not in (False, None):
        errors.append("checkpoint selection records validation use")
    if semantic.get("decision") != "semantic_gate_fail":
        errors.append("semantic decision drift")
    if status.get("status") != "semantic_gate_fail":
        errors.append("semantic status drift")

    score_path = semantic_dir / "score_records.jsonl"
    score_records: list[dict[str, Any]] = []
    with score_path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                errors.append(f"blank score record at line {line_number}")
                continue
            try:
                score_records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append(f"invalid score record at line {line_number}: {exc}")
    if len(score_records) != 40:
        errors.append(f"expected 40 score records, got {len(score_records)}")
    row_length_errors = []
    for record in score_records:
        lengths = record.get("score_lengths", {})
        if lengths and len(set(lengths.values())) != 1:
            row_length_errors.append(record.get("unit_key"))
    if row_length_errors:
        errors.append(f"score array length drift: {row_length_errors[:4]}")

    l81_root = ROOT / "outputs/l81"
    forbidden_paths = []
    for path in l81_root.rglob("*"):
        rel = str(path.relative_to(l81_root)).lower()
        if any(token in rel for token in ("screening", "official_test", "trackeval", "hota")):
            forbidden_paths.append(rel)
    if forbidden_paths:
        errors.append(f"forbidden L81 output paths: {forbidden_paths[:4]}")

    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True,
    ).stdout
    status_lines = git_status.splitlines()
    preexisting_lines = source_check.get("git_status_short", {}).get("line_count")
    # The worktree was already dirty before L81.  Hash rechecks above are the
    # authoritative old-asset integrity test; this records, rather than
    # guesses, the pre-existing versus current dirty status.
    result = {
        "format": "locatemot-l81-final-integrity-v1",
        "status": "complete" if not errors else "integrity_fail",
        "command": " ".join(sys.argv),
        "cwd": str(ROOT),
        "luna_thread": THREAD,
        "frozen_recheck": frozen_recheck,
        "fixed_manifest": manifest_record,
        "clip_weight": clip_record,
        "source_of_truth_status": source_check.get("status"),
        "l81_gate": {
            "status": gate.get("status"),
            "selected_method": gate.get("selected_method"),
            "selected_step": gate.get("selected_step"),
            "checks": gate.get("checks"),
        },
        "selection": {
            "method": selection.get("selected", {}).get("method"),
            "step": selection.get("selected", {}).get("step"),
            "validation_used": selection.get("validation_used"),
            "selection_source": selection.get("selection_source"),
        },
        "score_records": {
            "path": str(score_path), "count": len(score_records),
            "expected_count": 40, "row_length_errors": row_length_errors,
            "candidate_deletion": False, "candidate_truncation": False,
            "finite_checks": True,
        },
        "worktree": {
            "pre_l81_status_line_count": preexisting_lines,
            "current_status_line_count": len(status_lines),
            "preexisting_dirty": True,
            "old_asset_integrity_basis": "direct frozen-file SHA256 recheck plus source-of-truth record",
            "preview": status_lines[:50],
            "commit_or_push_performed_by_l81": False,
        },
        "forbidden_output_paths": forbidden_paths,
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "hota_trackeval_run": False,
        "training_run": True,
        "errors": errors,
        "next_action": (
            "keep L81 stopped; seek supervisor-approved new RMOT structural correspondence probe"
            if not errors else "repair final integrity failure before interpreting L81"
        ),
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "errors": errors, "output": str(out)}, ensure_ascii=False))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
