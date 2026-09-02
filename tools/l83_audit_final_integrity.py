#!/usr/bin/env python3
"""Compact, read-only integrity audit for the completed L83 evidence chain."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def main() -> int:
    out = ROOT / "outputs/l83/audit/final_integrity"
    out.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
    source = read_json(ROOT / "outputs/l83/preregister/source_of_truth.json")
    faithful = read_json(ROOT / "outputs/l83/train/faithful_bag_attempt1/faithful_gate.json")
    reload_audit = read_json(ROOT / "outputs/l83/audit/faithful_reload_attempt1/reload_audit.json")
    decoder_dir = ROOT / "outputs/l83/audit/decoder_sharpness_attempt9"
    decoder = read_json(decoder_dir / "decoder_sharpness.json")
    decoder_status = read_json(decoder_dir / "status.json")
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    checks["cwd"] = str(ROOT) == "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT"
    checks["thread"] = THREAD in json.dumps([source, faithful, reload_audit, decoder])
    checks["manifest_sha"] = manifest.exists() and sha256_file(manifest) == EXPECTED_MANIFEST
    checks["source_snapshot_complete"] = source.get("branch") == "codex/l83-faithful-targetbag-20260902"
    checks["faithful_gate_failed"] = faithful.get("status") == "faithful_target_bag_training_gate_fail"
    checks["faithful_gate_rows_complete"] = all(
        bool(item.get("checks", {}).get("G6_complete_finite_no_deletion"))
        for item in faithful.get("checks_by_representation", [])
    ) and len(faithful.get("checks_by_representation", [])) == 3
    checks["reload_complete"] = reload_audit.get("status") == "complete" and all(
        bool(item.get("strict_reload_pass")) and float(item.get("max_reload_output_difference", 1.0)) == 0.0
        for item in reload_audit.get("representation_results", [])
    )
    checks["decoder_complete"] = decoder.get("status") == "complete" and decoder_status.get("status") == "complete"
    checks["decoder_split"] = decoder.get("train_group_count") == 524 and decoder.get("dev_group_count") == 138
    checks["decoder_stages"] = decoder.get("stage_names") == ["Z0", "Zp", "Z1", "Z2", "Z3", "Z4", "Z5", "Z6"]
    checks["decoder_flags"] = all(
        decoder.get(name) is False
        for name in ("candidate_deletion", "candidate_truncation", "screening_gt_used", "official_test_labels_read", "ordinary_mot_ovmot_touched", "hota_trackeval_run", "features_persistent")
    )
    stage_metrics = decoder.get("stage_metrics", {})
    checks["stage_metric_contract"] = all(
        value.get("all_dev_groups_present") is True
        and value.get("loss_trace_steps") == 5240
        and value.get("aggregate", {}).get("finite") is True
        and value.get("aggregate", {}).get("candidate_deletion") is False
        and value.get("aggregate", {}).get("candidate_truncation") is False
        for value in stage_metrics.values()
    ) and len(stage_metrics) == 8
    record_path = decoder_dir / "dev_group_metrics.jsonl"
    record_counts: dict[str, int] = {}
    record_keys: set[tuple[str, str]] = set()
    finite_records = True
    if record_path.exists():
        with record_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (str(row["stage"]), str(row["group_key"]))
                record_keys.add(key)
                record_counts[key[0]] = record_counts.get(key[0], 0) + 1
                finite_records = finite_records and bool(row.get("finite", False))
    checks["decoder_record_rows"] = len(record_keys) == 8 * 138 and all(count == 138 for count in record_counts.values())
    checks["decoder_records_finite"] = finite_records
    checkpoint_hashes: dict[str, str] = {}
    checkpoint_ok = True
    for stage in decoder.get("stage_names", []):
        checkpoint = decoder_dir / "checkpoints" / f"{stage}_checkpoint_epoch10.pt"
        checkpoint_ok = checkpoint_ok and checkpoint.exists()
        if checkpoint.exists():
            checkpoint_hashes[stage] = sha256_file(checkpoint)
            checkpoint_ok = checkpoint_ok and checkpoint_hashes[stage] == stage_metrics[stage]["checkpoint"]["sha256"]
    checks["decoder_checkpoint_hashes"] = checkpoint_ok
    # Compare the small, immutable source anchors recorded before the L83 run.
    source_anchor_checks: dict[str, bool] = {}
    for name in ("fixed_manifest", "l69_feature_manifest", "l48_text_cache", "l59_control", "l81_control", "l81_step100", "l82_candidate_reference", "uidm_step11000"):
        item = source.get("inputs", {}).get(name, {})
        path = Path(item["path"]) if item.get("path") else None
        expected = item.get("sha256")
        source_anchor_checks[name] = bool(path and expected and path.exists() and sha256_file(path) == expected)
    checks["source_anchors_unchanged"] = all(source_anchor_checks.values())
    checks["no_forbidden_stage"] = all(
        not any(token in str(path).lower() for token in ("screening", "official_test", "trackeval"))
        for path in decoder_dir.glob("*")
    )
    checks["all_checks_pass"] = all(checks.values())
    details["checkpoint_hashes"] = checkpoint_hashes
    details["record_counts"] = record_counts
    details["source_anchor_checks"] = source_anchor_checks
    details["decoder_conclusion"] = decoder.get("conclusion")
    payload = {
        "format": "locatemot-l83-final-integrity-v1",
        "status": "complete" if checks["all_checks_pass"] else "invalid",
        "command": "python tools/l83_audit_final_integrity.py",
        "cwd": str(ROOT),
        "luna_thread": THREAD,
        "checks": checks,
        "details": details,
        "inputs": {
            "source_of_truth": str(ROOT / "outputs/l83/preregister/source_of_truth.json"),
            "faithful_gate": str(ROOT / "outputs/l83/train/faithful_bag_attempt1/faithful_gate.json"),
            "reload_audit": str(ROOT / "outputs/l83/audit/faithful_reload_attempt1/reload_audit.json"),
            "decoder_audit": str(decoder_dir / "decoder_sharpness.json"),
        },
        "outputs": {"integrity": str(out / "final_integrity.json")},
        "failure_root_cause": None if checks["all_checks_pass"] else "one or more immutable evidence checks failed",
        "next_action": "STOPPED_PENDING_SUPERVISOR_REVIEW" if checks["all_checks_pass"] else "inspect failed integrity check before reporting",
        "screening_gt_used": False,
        "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False,
        "hota_trackeval_run": False,
        "candidate_deletion": False,
        "candidate_truncation": False,
        "token_span_region_alignment": "UNALIGNED",
        "static_motion_alignment": "UNALIGNED",
    }
    (out / "final_integrity.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if checks["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
