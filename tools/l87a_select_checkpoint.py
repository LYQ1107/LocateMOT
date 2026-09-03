#!/usr/bin/env python3
"""L87-A corrected target-bag checkpoint/rule selection on internal dev."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

WORK_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(os.environ.get(
    "LOCATEMOT_ASSET_ROOT", "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT"
)).resolve()
if str(WORK_ROOT) not in sys.path: sys.path.insert(0, str(WORK_ROOT))
sys.path.insert(0, str(WORK_ROOT / "locatemot" / "rmot"))

from l87_eval_policy import checkpoint_selection_key, contract_summary, fit_rules, metric  # noqa: E402

BASE_SHA = "97bff208929474d4c4b0d659c80e7eba2f3f5d0a"
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); out = args.out.resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"refusing nonempty L87-A selection output: {out}")
    out.mkdir(parents=True, exist_ok=True); command = " ".join([sys.executable, *sys.argv])
    try:
        if Path.cwd().resolve() != WORK_ROOT: raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        records_path = args.scores.resolve() / "score_records.jsonl"
        with records_path.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line); grouped[str(row["checkpoint"]["path"])].append(row)
        evidence = []
        for path, values in sorted(grouped.items(), key=lambda item: int(item[1][0]["checkpoint"]["epoch"])):
            if len({str(row["unit_key"]) for row in values}) != len(values): raise AssertionError(f"duplicate dev key {path}")
            summary = contract_summary(values)
            if summary["candidate_deletion"] or summary["candidate_truncation"] or not summary["finite_scores"]:
                raise AssertionError(f"invalid A dev records {path}")
            rules = fit_rules(values); selected_rule = rules["B"]
            metrics = selected_rule["metrics"]
            evidence.append({"checkpoint_info": values[0]["checkpoint"], "record_count": len(values),
                             "contract": summary, "rule_fits": rules,
                             "selection_key": checkpoint_selection_key(metrics, int(values[0]["checkpoint"]["epoch"])),
                             "selected_rule_metrics": metric(values, selected_rule["candidate_threshold"],
                                                               selected_rule["presence_threshold"], selected_rule["null_margin"])})
        if not evidence: raise AssertionError("no A dev records")
        chosen = min(evidence, key=lambda row: tuple(row["selection_key"]))
        chosen_path = str(chosen["checkpoint_info"]["path"])
        chosen_records = grouped[chosen_path]
        rules = fit_rules(chosen_records)
        result = {
            "format": "locatemot-l87a-corrected-selection-v1", "status": "complete",
            "stage": "L87-A corrected temporal retraining dev selection",
            "evidence_type": "internal fit/dev selection; no fixed validation labels",
            "command": command, "work_root": str(WORK_ROOT), "asset_root": str(ASSET_ROOT),
            "cwd": str(WORK_ROOT), "luna_thread": THREAD, "base_l86_commit": BASE_SHA,
            "common_eval_policy_sha": sha256(WORK_ROOT / "locatemot/rmot/l87_eval_policy.py"),
            "input_scores": str(records_path), "input_scores_sha256": sha256(records_path),
            "checkpoint_evidence": evidence,
            "selected": {"checkpoint_info": chosen["checkpoint_info"], "rule_fit": rules["B"],
                          "all_rule_fits": rules, "selection_tuple": chosen["selection_key"],
                          "selection_objective": "lower target-bag hard; higher target-bag hit@1; higher multi-target exact; lower inactive FA; higher distinct recall; earlier epoch"},
            "dev_full_video_hota_available": False, "validation_not_read": True,
            "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False, "no_hota_or_trackeval": True, "z1_representation_changed": False,
            "groundingdino_lora_used": False, "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "failure_root_cause": None, "next_action": "freeze corrected rule and evaluate fixed semantic units",
        }
        for name in ("checkpoint_selection.json", "provenance.json", "status.json"):
            write_json(out / name, result)
        print(json.dumps({"status": "complete", "selected": result["selected"]}, indent=2), flush=True)
        return 0
    except Exception:
        trace = traceback.format_exc(); (out / "INCOMPLETE.md").write_text("# L87-A selection — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {"format": "locatemot-l87a-selection-v1", "status": "incomplete",
                                         "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                         "failure_root_cause": "first traceback in INCOMPLETE.md", "screening_gt_used": False,
                                         "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                                         "hota_trackeval_run": False})
        raise


if __name__ == "__main__": raise SystemExit(main())
