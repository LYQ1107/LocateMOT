#!/usr/bin/env python3
"""Correct the L84 stage-selection tie-break and rerun only the authorized
no-refPE test for the correctly selected representation.

The original paired run is immutable evidence.  This script does not rerun
the seven representation stages; it reads their completed metrics, applies
the registered lexicographic order, and runs the no-refPE structural test only
for the corrected selected stage.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.l84_train_paired_middecoder import (  # noqa: E402
    INIT_ROOT,
    MANIFEST,
    SEEDS,
    THREAD,
    build_group_states,
    file_meta,
    make_schedules,
    train_one_stage,
    write_json,
)

PAIRED = ROOT / "outputs/l84/train/paired_middecoder"
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"


def state_max_diff(left: Path, right: Path) -> float:
    left_package = torch.load(left, map_location="cpu", weights_only=False)
    right_package = torch.load(right, map_location="cpu", weights_only=False)
    left_state = left_package["model_state_dict"]
    right_state = right_package["model_state_dict"]
    if list(left_state) != list(right_state):
        raise AssertionError("tied stage checkpoint keys differ")
    return max(float((left_state[key] - right_state[key]).abs().max()) for key in left_state)


def select_stage(stable: dict[str, Any]) -> tuple[str, list[str]]:
    qualifying = list(stable.get("qualifying_stages", []))
    if not qualifying:
        return "Z0", []

    def key(stage: str) -> tuple[float, float, float, float, float, int]:
        means = stable["checks"][stage]["means"]
        return (
            -float(means["aggregate_hard_improvement"]),
            -float(means["aggregate_hit_at1_improvement"]),
            -float(means["v2_hard_improvement"]),
            -float(means["v2_hit_at1_improvement"]),
            -float(means["multi_exact_improvement"]),
            int(("Z0", "Z1", "Z4", "Z6", "R1", "R4", "R6").index(stage)),
        )

    return sorted(qualifying, key=key)[0], qualifying


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty selection-correction output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    device = torch.device("cuda:0")
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if not torch.cuda.is_available():
            raise RuntimeError("L84 selection correction requires CUDA")
        if not MANIFEST.is_file():
            raise FileNotFoundError(MANIFEST)
        if __import__("hashlib").sha256(MANIFEST.read_bytes()).hexdigest() != EXPECTED_MANIFEST:
            raise AssertionError("fixed manifest SHA drift")
        stable = json.loads((PAIRED / "stable_gate.json").read_text())
        paired = json.loads((PAIRED / "paired_stage_metrics.json").read_text())
        selected_original, qualifying = select_stage(stable)
        if selected_original != "Z1":
            raise AssertionError(f"unexpected corrected L84 selection: {selected_original}")
        tied_checkpoint_diffs = {
            str(seed): state_max_diff(
                PAIRED / f"seed{seed}/Z1/checkpoint.pt",
                PAIRED / f"seed{seed}/R1/checkpoint.pt",
            )
            for seed in SEEDS
        }
        tied_metric_hashes = {
            str(seed): {
                "Z1": hashlib.sha256((PAIRED / f"seed{seed}/Z1/dev_group_metrics.jsonl").read_bytes()).hexdigest(),
                "R1": hashlib.sha256((PAIRED / f"seed{seed}/R1/dev_group_metrics.jsonl").read_bytes()).hexdigest(),
            }
            for seed in SEEDS
        }
        write_json(out / "selection_correction.json", {
            "format": "locatemot-l84-selection-correction-v1",
            "status": "selection_corrected",
            "historical_output": str(PAIRED),
            "historical_selection": json.loads((PAIRED / "final_selection.json").read_text()),
            "qualifying_stages": qualifying,
            "corrected_selection": selected_original,
            "registered_tuple": [
                "lower_mean_hard", "higher_mean_hit_at1", "lower_mean_v2_hard",
                "higher_mean_v2_hit_at1", "higher_mean_multi_exact", "earliest_simple_stage",
            ],
            "tie_evidence": {
                "z1_r1_checkpoint_max_abs_diff_by_seed": tied_checkpoint_diffs,
                "z1_r1_dev_record_sha256_by_seed": tied_metric_hashes,
            },
            "checkpoint_selection_only": True,
            "no_refpe_required_for_corrected_selection": True,
            "screening_gt_used": False, "official_test_labels_read": False,
            "hota_trackeval_run": False, "ordinary_mot_ovmot_touched": False,
        })

        from tools.l82_train_frozen_rank_probe import load_groups

        groups, train_keys, dev_keys = load_groups()
        # Preserve the registered four-rank schedule metadata.  The
        # correction itself runs on one GPU, but its global order must be the
        # same order used by the completed paired run.
        schedules = make_schedules(train_keys, 4, ROOT / "outputs/l84")
        all_keys = train_keys + dev_keys
        data, model_info = build_group_states(
            all_keys, groups, device, no_refpe_in_content=True,
            selected_name=selected_original,
        )
        by_key = {item.group_key: item for item in data}
        local_order = [key for key in schedules[SEEDS[0]]["global_group_order"] if key in by_key]
        local_dev_order = list(dev_keys)
        summaries: dict[str, Any] = {}
        for seed in SEEDS:
            summary, _records, _aux, _trace = train_one_stage(
                f"{selected_original}_no_refpe", seed, by_key, local_order,
                local_dev_order, len(dev_keys), device, out, 0, 1,
            )
            if summary is None:
                raise AssertionError(f"missing no-refPE summary for seed {seed}")
            summaries[str(seed)] = summary
        original_rows = [paired["stage_metrics"][selected_original][str(seed)] for seed in SEEDS]
        no_rows = [summaries[str(seed)] for seed in SEEDS]
        original_hard = sum(float(row["aggregate"]["target_bag_hard_violation"]) for row in original_rows) / len(original_rows)
        no_hard = sum(float(row["aggregate"]["target_bag_hard_violation"]) for row in no_rows) / len(no_rows)
        original_hit = sum(float(row["aggregate"]["target_bag_hit_at1"]) for row in original_rows) / len(original_rows)
        no_hit = sum(float(row["aggregate"]["target_bag_hit_at1"]) for row in no_rows) / len(no_rows)
        original_v2_hard = sum(float(row["breakdowns"]["dataset"]["refer_kitti_v2"]["target_bag_hard_violation"]) for row in original_rows) / len(original_rows)
        no_v2_hard = sum(float(row["breakdowns"]["dataset"]["refer_kitti_v2"]["target_bag_hard_violation"]) for row in no_rows) / len(no_rows)
        no_ref_pass = bool(no_hard <= original_hard - 0.02 and no_hit >= original_hit + 0.02 and no_v2_hard <= original_v2_hard)
        final_selected = f"{selected_original}_no_refpe" if no_ref_pass else selected_original
        final_status = "no_refpe_selected" if no_ref_pass else "original_content_seed_selected"
        write_json(out / "no_refpe_metrics.json", {
            "format": "locatemot-l84-no-refpe-corrected-metrics-v1",
            "status": "complete", "selected_original": selected_original,
            "no_refpe_stage": f"{selected_original}_no_refpe",
            "seed_metrics": summaries,
            "means": {
                "original_hard": original_hard, "no_refpe_hard": no_hard,
                "original_hit_at1": original_hit, "no_refpe_hit_at1": no_hit,
                "original_v2_hard": original_v2_hard, "no_refpe_v2_hard": no_v2_hard,
            },
            "pass": no_ref_pass,
            "source_of_original_metrics": str(PAIRED / "paired_stage_metrics.json"),
        })
        write_json(out / "final_selection.json", {
            "format": "locatemot-l84-final-selection-corrected-v1",
            "status": final_status, "selected_original": selected_original,
            "selected_representation": final_selected,
            "no_refpe_tested": True, "selection_correction": True,
            "next_stage": "L85_FULL_RMOT", "candidate_deletion": False,
            "candidate_truncation": False, "screening_gt_used": False,
            "official_test_labels_read": False, "hota_trackeval_run": False,
            "ordinary_mot_ovmot_touched": False,
        })
        write_json(out / "config.json", {
            "format": "locatemot-l84-selection-correction-config-v1",
            "command": command, "thread": THREAD,
            "corrected_selected_original": selected_original,
            "no_refpe_test": True, "seeds": list(SEEDS),
            "train_groups": len(train_keys), "dev_groups": len(dev_keys),
            "representation": "L84 selected Z1 with content=visual_seed, references retained",
            "loss": "l83_target_bag_loss_unchanged", "epochs": 10,
            "candidate_deletion": False, "candidate_truncation": False,
        })
        write_json(out / "provenance.json", {
            "format": "locatemot-l84-selection-correction-provenance-v1",
            "command": command, "thread": THREAD, "project_root": str(ROOT),
            "historical_paired_output": file_meta(PAIRED / "stable_gate.json"),
            "manifest": file_meta(MANIFEST),
            "canonical_initializations": [file_meta(INIT_ROOT / f"probe_init_seed{seed}.pt") for seed in SEEDS],
            "input_features": "rebuilt process-local from frozen L69/runtime; no dense/raw cache",
            "labels": "L49 fit units only, attached after complete state construction",
            "selection": "corrected registered stable tuple; no validation labels",
            "selected_representation": final_selected, "final_status": final_status,
            "candidate_deletion": False, "candidate_truncation": False,
            "screening_gt_used": False, "official_test_labels_read": False,
            "hota_trackeval_run": False, "ordinary_mot_ovmot_touched": False,
            "next_stage": "L85_FULL_RMOT",
        })
        write_json(out / "status.json", {
            "format": "locatemot-l84-selection-correction-status-v1",
            "status": final_status, "selected_representation": final_selected,
            "next_stage": "L85_FULL_RMOT", "candidate_deletion": False,
            "candidate_truncation": False, "screening_gt_used": False,
            "official_test_labels_read": False, "hota_trackeval_run": False,
            "ordinary_mot_ovmot_touched": False,
        })
        del data, by_key, groups
        gc.collect()
        torch.cuda.empty_cache()
        return 0
    except Exception as exc:
        write_json(out / "status.json", {
            "format": "locatemot-l84-selection-correction-status-v1",
            "status": "incomplete", "command": command,
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "next_action": "preserve this attempt and inspect the first traceback",
            "screening_gt_used": False, "official_test_labels_read": False,
            "hota_trackeval_run": False, "ordinary_mot_ovmot_touched": False,
        })
        (out / "traceback.txt").write_text(traceback.format_exc())
        (out / "INCOMPLETE.md").write_text(
            f"L84 selection correction incomplete. First error: {type(exc).__name__}: {exc}\n"
            f"Command: {command}\n"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
