#!/usr/bin/env python3
"""Zero-training L87-B re-selection of immutable L86 dev score records."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

WORK_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(os.environ.get(
    "LOCATEMOT_ASSET_ROOT", "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT"
)).resolve()
MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
BASE_SHA = "97bff208929474d4c4b0d659c80e7eba2f3f5d0a"
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
DEV_RECORDS = ASSET_ROOT / "outputs/l86/eval/dev_cheap_attempt2/score_records.jsonl"
ORIGINAL_SELECTION = ASSET_ROOT / "outputs/l86/eval/dev_selection_attempt1/checkpoint_selection.json"
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))
sys.path.insert(0, str(WORK_ROOT / "locatemot" / "rmot"))

from l87_eval_policy import (  # noqa: E402
    checkpoint_selection_key,
    contract_summary,
    fit_rules,
    metric,
    target_bags,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def load_records() -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with DEV_RECORDS.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                grouped[str(row["checkpoint"]["path"])].append(row)
    if not grouped:
        raise AssertionError("immutable L86 dev records are empty")
    return dict(grouped)


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/l87b/selection"))
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L87-B selection output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong L87-B cwd: {Path.cwd()}")
        if sha256(MANIFEST) != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        if not DEV_RECORDS.is_file():
            raise FileNotFoundError(DEV_RECORDS)
        grouped = load_records()
        checkpoint_evidence: list[dict[str, Any]] = []
        for checkpoint_path, values in sorted(grouped.items(), key=lambda item: int(item[1][0]["checkpoint"]["epoch"])):
            keys = [str(row["unit_key"]) for row in values]
            if len(keys) != len(set(keys)):
                raise AssertionError(f"duplicate dev unit rows for {checkpoint_path}")
            summary = contract_summary(values)
            if summary["candidate_deletion"] or summary["candidate_truncation"] or not summary["finite_scores"]:
                raise AssertionError(f"invalid immutable dev records: {checkpoint_path}")
            # Function-level contract assertion on one real row, before any
            # selection operation.
            first = values[0]
            bags, referred = target_bags(first["score"], first["candidate_gt"], first["target_ids"])
            if len(bags) < len(referred) or int(first["candidate_count"]) != len(first["score"]):
                raise AssertionError("target-bag shape/count contract failed")
            rules = fit_rules(values)
            selected_rule = rules["B"]
            evidence = {
                "checkpoint_info": values[0]["checkpoint"], "record_count": len(values),
                "group_keys": sorted({str(row["group_key"]) for row in values}),
                "contract": summary, "rule_fits": rules,
                "selection_key": checkpoint_selection_key(selected_rule["metrics"], int(values[0]["checkpoint"]["epoch"])),
                "target_bag_metrics": metric(values, selected_rule["candidate_threshold"],
                                              selected_rule["presence_threshold"], selected_rule["null_margin"]),
            }
            checkpoint_evidence.append(evidence)
        chosen = min(checkpoint_evidence, key=lambda row: tuple(row["selection_key"]))
        selected_path = str(chosen["checkpoint_info"]["path"])
        selected_records = grouped[selected_path]
        selected_rules = fit_rules(selected_records)
        original = None
        if ORIGINAL_SELECTION.is_file():
            original_payload = json.loads(ORIGINAL_SELECTION.read_text())
            original = original_payload.get("selected", original_payload)
        result = {
            "format": "locatemot-l87b-corrected-selection-v1", "status": "complete",
            "stage": "L87-B zero-train corrected reselection and deployment",
            "evidence_type": "internal fit/dev re-selection; no forward, optimizer, backward or new checkpoint",
            "command": command, "work_root": str(WORK_ROOT), "asset_root": str(ASSET_ROOT),
            "cwd": str(WORK_ROOT), "luna_thread": THREAD, "base_l86_commit": BASE_SHA,
            "common_eval_policy_sha": sha256(WORK_ROOT / "locatemot/rmot/l87_eval_policy.py"),
            "input_scores": str(DEV_RECORDS), "input_scores_sha256": sha256(DEV_RECORDS),
            "immutable_checkpoint_directory": str((ASSET_ROOT / "outputs/l86/train/joint40").resolve()),
            "original_l86_selection": original,
            "original_epoch": 14, "original_rule": "L86 Rule B with presence-vs-NULL deployment",
            "checkpoint_evidence": checkpoint_evidence,
            "selected": {
                "checkpoint_info": chosen["checkpoint_info"], "rule_fit": selected_rules["B"],
                "all_rule_fits": selected_rules, "selection_tuple": chosen["selection_key"],
                "selection_objective": "lower target-bag hard; higher target-bag hit@1; higher multi-target exact; lower inactive FA; higher distinct recall; earlier epoch",
            },
            "same_checkpoint_as_l86": int(chosen["checkpoint_info"]["epoch"]) == 14,
            "same_rule_as_l86": False,
            "zero_training": True, "optimizer_used": False, "backward_used": False,
            "model_parameter_changes": False, "new_checkpoint": False,
            "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "no_hota_or_trackeval": True, "z1_representation_changed": False,
            "groundingdino_lora_used": False, "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED", "failure_root_cause": None,
            "next_action": "run corrected fixed semantic and internal full-video inference",
        }
        for name in ("corrected_l86_selection.json", "checkpoint_selection.json", "provenance.json", "status.json"):
            write_json(out / name, result)
        print(json.dumps({"status": result["status"], "selected": result["selected"],
                          "same_checkpoint_as_l86": result["same_checkpoint_as_l86"]}, indent=2), flush=True)
        return 0
    except Exception:
        import traceback
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L87-B selection — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {"format": "locatemot-l87b-selection-v1", "status": "incomplete",
                                         "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                         "failure_root_cause": "first traceback in INCOMPLETE.md",
                                         "screening_gt_used": False, "official_test_labels_read": False,
                                         "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
