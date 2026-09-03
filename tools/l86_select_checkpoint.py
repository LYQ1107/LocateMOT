#!/usr/bin/env python3
"""Select the frozen L86 checkpoint and emission rule using internal dev only."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
GRID = [-1.0, -.75, -.5, -.25, 0.0, .25, .5, .75, 1.0]
NULL_GRID = [0.0, .25, .5, .75]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def bag_scores(row: dict[str, Any]) -> tuple[list[tuple[str, str | int]], np.ndarray, np.ndarray]:
    scores = np.asarray(row["score"], dtype=np.float64)
    candidate_gt = [None if x is None else str(x) for x in row["candidate_gt"]]
    targets = {str(x) for x in row["target_ids"]}
    if scores.ndim != 1 or len(candidate_gt) != scores.size:
        raise AssertionError(f"bag row shape drift {row['unit_key']}")
    groups: dict[str, list[int]] = {}; background: list[int] = []
    for index, value in enumerate(candidate_gt):
        if value is None: background.append(index)
        else: groups.setdefault(value, []).append(index)
    keys: list[tuple[str, str | int]] = []; values: list[float] = []; positives: list[bool] = []
    for target in sorted(groups):
        keys.append(("target", target)); values.append(float(scores[groups[target]].max())); positives.append(target in targets)
    for index in background:
        keys.append(("background", index)); values.append(float(scores[index])); positives.append(False)
    return keys, np.asarray(values), np.asarray(positives, dtype=bool)


def metric(records: list[dict[str, Any]], candidate_threshold: float, presence_threshold: float,
           null_margin: float) -> dict[str, Any]:
    tp = fp = fn = selected_rows = positive_rows = 0
    top1 = top5 = top_units = empty = 0
    hard: list[bool] = []; strict: list[float] = []; best: list[float] = []; average: list[float] = []
    multi_recall: list[float] = []; multi_exact: list[float] = []
    inactive = inactive_accept = inactive_fp = 0; present_uncovered = 0
    bag_hit1 = bag_top5 = bag_units = 0; candidate_scores: list[float] = []
    for row in records:
        scores = np.asarray(row["score"], dtype=np.float64)
        labels = np.asarray(row["labels"], dtype=bool)
        if scores.shape != labels.shape or len(row["row_keys"]) != scores.size or not np.isfinite(scores).all():
            raise AssertionError(f"candidate/score/key drift {row['unit_key']}")
        candidate_scores.extend(scores.tolist())
        unit_gate = float(row["presence_logit"]) >= float(presence_threshold) and \
            float(row["presence_logit"]) - float(row["null_logit"]) >= float(null_margin)
        selected = (scores >= float(candidate_threshold)) if unit_gate else np.zeros_like(labels)
        tp += int((selected & labels).sum()); fp += int((selected & ~labels).sum()); fn += int((~selected & labels).sum())
        selected_rows += int(selected.sum()); positive_rows += int(labels.sum()); empty += int(not selected.any())
        if labels.any():
            order = np.argsort(-scores, kind="stable"); top_units += 1
            top1 += int(bool(labels[order[:1]].any())); top5 += int(bool(labels[order[:5]].any()))
        pos = np.flatnonzero(labels); neg = np.flatnonzero(~labels)
        if len(pos) and len(neg):
            strict_value = float(scores[pos].min() - scores[neg].max())
            strict.append(strict_value); best.append(float(scores[pos].max() - scores[neg].max()))
            average.append(float(scores[pos].mean() - scores[neg].max())); hard.append(strict_value < 0.0)
        if len(pos) > 1:
            value = float(selected[pos].sum() / len(pos)); multi_recall.append(value); multi_exact.append(float(selected[pos].all()))
        category = str(row.get("category", "unknown"))
        if category == "inactive":
            inactive += 1; inactive_accept += int(bool(selected.any())); inactive_fp += int((selected & ~labels).sum())
        if category == "present_uncovered": present_uncovered += 1
        keys, bags, bag_positive = bag_scores(row)
        if bags.size:
            positive_bags = bags[bag_positive]
            negative_bags = bags[~bag_positive]
            if positive_bags.size:
                bag_order = np.argsort(-bags, kind="stable"); bag_units += 1
                bag_hit1 += int(bool(bag_positive[bag_order[:1]].any())); bag_top5 += int(bool(bag_positive[bag_order[:5]].any()))
    def dist(values: list[float]) -> dict[str, Any]:
        if not values: return {"count": 0, "mean": None, "p50": None, "p90": None}
        arr = np.asarray(values, dtype=np.float64)
        return {"count": int(arr.size), "mean": float(arr.mean()), "p50": float(np.quantile(arr, .5)), "p90": float(np.quantile(arr, .9))}
    f1 = 2.0 * tp / max(1.0, 2.0 * tp + fp + fn)
    return {
        "units": len(records), "candidate_rows": int(sum(len(x["score"]) for x in records)),
        "positive_rows": positive_rows, "selected_rows": selected_rows, "true_positive_rows": tp,
        "false_positive_rows": fp, "false_negative_rows": fn, "f1": float(f1),
        "candidate_precision": tp / max(1, selected_rows), "candidate_recall": tp / max(1, tp + fn),
        "fp_per_frame": fp / max(1, len(records)), "predictions_per_positive": selected_rows / max(1, positive_rows),
        "top1": top1 / max(1, top_units), "top5": top5 / max(1, top_units),
        "hard_violation": float(np.mean(hard)) if hard else 1.0,
        "strict_margin": dist(strict), "best_margin": dist(best), "average_margin": dist(average),
        "multi_positive_recall": float(np.mean(multi_recall)) if multi_recall else None,
        "multi_target_exact": float(np.mean(multi_exact)) if multi_exact else None,
        "minimum_positive_coverage": float(np.mean(multi_recall)) if multi_recall else None,
        "empty_rate": empty / max(1, len(records)), "inactive_units": inactive,
        "inactive_false_acceptance": inactive_accept / max(1, inactive), "inactive_false_positive_rows": inactive_fp,
        "present_uncovered_units": present_uncovered, "target_bag_hit1": bag_hit1 / max(1, bag_units),
        "target_bag_top5": bag_top5 / max(1, bag_units), "target_bag_units": bag_units,
        "score_distribution": dist(candidate_scores), "candidate_threshold": float(candidate_threshold),
        "presence_threshold": float(presence_threshold), "null_margin": float(null_margin),
        "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
    }


def rules_for(records: list[dict[str, Any]]) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for p in GRID:
        for n in NULL_GRID:
            for c in GRID:
                candidates.append(metric(records, c, p, n))
    def b_key(value: dict[str, Any]) -> tuple[float, float, float]:
        return (float(value["f1"]), -float(value["inactive_false_acceptance"]), float(value["multi_target_exact"] or 0.0))
    def r_key(value: dict[str, Any]) -> tuple[int, float, float]:
        return (int(float(value["candidate_precision"]) >= .08), float(value["candidate_recall"]), -float(value["inactive_false_acceptance"]))
    def p_key(value: dict[str, Any]) -> tuple[int, float, float]:
        return (int(float(value["candidate_recall"]) >= .60), float(value["candidate_precision"]), float(value["multi_target_exact"] or 0.0))
    selected = {"B": max(candidates, key=b_key), "R": max(candidates, key=r_key), "P": max(candidates, key=p_key)}
    return {name: {"rule": name, "metrics": value, "candidate_threshold": value["candidate_threshold"],
                   "presence_threshold": value["presence_threshold"], "null_margin": value["null_margin"]}
            for name, value in selected.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); out = args.out.resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"refusing nonempty selection output: {out}")
    out.mkdir(parents=True, exist_ok=True); command = " ".join([sys.executable, *sys.argv])
    try:
        if Path.cwd().resolve() != ROOT: raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256(MANIFEST) != MANIFEST_SHA: raise AssertionError("fixed manifest SHA drift")
        payload = json.loads((args.scores.resolve() / "cheap_dev_scores.json").read_text())
        records: dict[str, list[dict[str, Any]]] = defaultdict(list)
        with (args.scores.resolve() / "score_records.jsonl").open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line); records[str(row["checkpoint"]["path"])].append(row)
        checkpoint_evidence = []
        for path, values in sorted(records.items(), key=lambda item: int(item[1][0]["checkpoint"]["epoch"])):
            if len(values) != len(set(str(x["unit_key"]) for x in values)):
                raise AssertionError(f"duplicate dev unit records for {path}")
            rule = rules_for(values)["B"]; m = rule["metrics"]
            checkpoint_evidence.append({"checkpoint_info": values[0]["checkpoint"], "record_count": len(values),
                                        "rule_b": rule, "selection_key": [float(m["hard_violation"]), -float(m["target_bag_hit1"]),
                                        -float(m["multi_target_exact"] or 0.0), float(m["inactive_false_acceptance"]),
                                        -float(m["candidate_recall"]), int(values[0]["checkpoint"]["epoch"])],
                                        "candidate_rows_retained": True})
        if not checkpoint_evidence: raise AssertionError("no dev checkpoints")
        chosen = min(checkpoint_evidence, key=lambda x: tuple(x["selection_key"]))
        chosen_path = str(chosen["checkpoint_info"]["path"]); chosen_records = records[chosen_path]
        all_rules = rules_for(chosen_records)
        selected = {"checkpoint_info": chosen["checkpoint_info"], "rule_fit": all_rules["B"],
                    "all_rule_fits": all_rules, "selection_tuple": chosen["selection_key"],
                    "selection_objective": "cheap internal-dev target-bag tuple; dev full-video HOTA unavailable",
                    "validation_not_read": True}
        result = {"format": "locatemot-l86-checkpoint-selection-v1", "status": "complete",
                  "evidence_type": "internal fit/dev selection", "command": command, "cwd": str(ROOT), "luna_thread": THREAD,
                  "input_scores": str(args.scores.resolve()), "input_scores_sha256": sha256(args.scores.resolve() / "cheap_dev_scores.json"),
                  "checkpoint_evidence": checkpoint_evidence, "selected": selected,
                  "dev_full_video_hota_available": False, "dev_full_video_hota_run": False,
                  "threshold_grid": {"candidate": GRID, "presence": GRID, "null_margin": NULL_GRID},
                  "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                  "hota_trackeval_run": False, "no_hota_or_trackeval": True, "candidate_deletion": False,
                  "candidate_truncation": False, "z1_representation_changed": False, "groundingdino_lora_used": False,
                  "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
                  "failure_root_cause": None, "next_action": "freeze selected checkpoint/rule and evaluate fixed semantic units"}
        write_json(out / "checkpoint_selection.json", result); write_json(out / "dev_metrics.json", result); write_json(out / "provenance.json", result); write_json(out / "status.json", result)
        return 0
    except Exception:
        trace = traceback.format_exc(); (out / "INCOMPLETE.md").write_text("# L86 selection — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {"format": "locatemot-l86-checkpoint-selection-v1", "status": "incomplete",
                                         "command": command, "cwd": str(ROOT), "luna_thread": THREAD,
                                         "failure_root_cause": "first traceback in INCOMPLETE.md", "screening_gt_used": False,
                                         "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
