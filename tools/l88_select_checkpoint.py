#!/usr/bin/env python3
"""Fit the registered L88 dev rules and build the frozen shortlist.

This stage consumes only the 138-group fit/dev records produced by
``l88_score_dev.py``.  It does not read the fixed calibration/validation
slice and it does not select a final checkpoint; full-video internal dev
TrackEval does that later.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

from l88_eval_metrics import fit_rule_set


WORK_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
EXPECTED_EPOCHS = list(range(2, 41, 2))
EXPECTED_RECORDS = 498


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True,
                               ensure_ascii=False, default=str) + "\n")


def load_records(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("format") != "locatemot-l88-score-record-v1":
            raise AssertionError(f"unexpected L88 score record format at line {line_number}")
        checkpoint = row.get("checkpoint") or {}
        path_key = str(Path(str(checkpoint["path"])).resolve())
        if int(checkpoint.get("epoch", -1)) not in EXPECTED_EPOCHS:
            raise AssertionError(f"unexpected L88 checkpoint epoch at line {line_number}")
        scores = row.get("score")
        if not isinstance(scores, list) or len(scores) != int(row["candidate_count"]):
            raise AssertionError(f"candidate score length drift: {row.get('unit_key')}")
        if len(row.get("row_keys", [])) != len(scores):
            raise AssertionError(f"candidate row-key length drift: {row.get('unit_key')}")
        if bool(row.get("candidate_deletion")) or bool(row.get("candidate_truncation")):
            raise AssertionError(f"candidate deletion/truncation in {row.get('unit_key')}")
        if not bool(row.get("finite_scores")):
            raise AssertionError(f"nonfinite score flag in {row.get('unit_key')}")
        grouped[path_key].append(row)
    return grouped


def checkpoint_info(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise AssertionError("empty checkpoint record group")
    values = [records[0]["checkpoint"], *[row["checkpoint"] for row in records[1:]]]
    epochs = {int(value["epoch"]) for value in values}
    paths = {str(Path(str(value["path"])).resolve()) for value in values}
    shas = {str(value["sha256"]) for value in values}
    if len(epochs) != 1 or len(paths) != 1 or len(shas) != 1:
        raise AssertionError("checkpoint metadata drift within record group")
    return dict(records[0]["checkpoint"])


def candidate_rank_features(rule_fits: dict[str, Any]) -> dict[str, Any]:
    """Return the registered metrics used for shortlist eligibility.

    The fixed epoch candidates and the best-F1 candidate use Rule B.  The
    best-recall candidate uses Rule R, whose precision floor is explicit.
    Keeping these namespaces separate prevents an unfiltered score from
    accidentally satisfying the shortlist floor.
    """
    b = rule_fits["B"]["metrics"]
    r = rule_fits["R"]["metrics"]
    return {
        "rule_b_target_bag_precision": float(b["target_bag_precision"]),
        "rule_b_target_bag_f1": float(b["target_bag_f1"]),
        "rule_b_distinct_target_recall": float(b["distinct_target_recall"]),
        "rule_r_target_bag_precision": float(r["target_bag_precision"]),
        "rule_r_distinct_target_recall": float(r["distinct_target_recall"]),
        "rule_b_precision_floor": bool(float(b["target_bag_precision"]) >= 0.08),
        "rule_r_precision_floor": bool(float(r["target_bag_precision"]) >= 0.08),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L88 shortlist output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != MANIFEST_SHA:
            raise AssertionError("fixed L19 manifest SHA drift")
        source = args.scores_dir.resolve()
        status = json.loads((source / "status.json").read_text())
        if status.get("status") != "complete":
            raise AssertionError("source cheap dev scoring is not complete")
        if status.get("fixed_calibration_read") or status.get("fixed_validation_read"):
            raise AssertionError("cheap dev source has fixed-slice label leakage")
        cheap = json.loads((source / "cheap_dev_scores.json").read_text())
        records_path = Path(str(cheap["score_records"])).resolve()
        if not records_path.is_file():
            records_path = (source / "score_records.jsonl").resolve()
        if not records_path.is_file():
            raise FileNotFoundError(records_path)
        grouped = load_records(records_path)
        epochs = sorted(int(checkpoint_info(rows)["epoch"]) for rows in grouped.values())
        if epochs != EXPECTED_EPOCHS:
            raise AssertionError(f"L88 dev checkpoint set drift: {epochs}")
        summaries: list[dict[str, Any]] = []
        for path_key, rows in sorted(grouped.items(), key=lambda item: int(item[1][0]["checkpoint"]["epoch"])):
            if len(rows) != EXPECTED_RECORDS:
                raise AssertionError(f"L88 dev record count drift epoch={rows[0]['checkpoint']['epoch']}: {len(rows)}")
            units = [str(row["unit_key"]) for row in rows]
            if len(set(units)) != len(units):
                raise AssertionError(f"duplicate L88 dev unit keys epoch={rows[0]['checkpoint']['epoch']}")
            info = checkpoint_info(rows)
            rules = fit_rule_set(rows)
            summary = {
                "checkpoint_info": info,
                "record_count": len(rows),
                "unit_count": len(units),
                "rule_fits": rules,
                "shortlist_rank_features": candidate_rank_features(rules),
                "labels_scope": "internal fit/dev only",
            }
            summaries.append(summary)

        by_epoch = {int(item["checkpoint_info"]["epoch"]): item for item in summaries}
        eligible_b = [item for item in summaries
                      if item["shortlist_rank_features"]["rule_b_precision_floor"]]
        eligible_r = [item for item in summaries
                      if item["shortlist_rank_features"]["rule_r_precision_floor"]]
        requested: list[tuple[str, dict[str, Any]]] = []
        for epoch in (8, 20, 40):
            item = by_epoch[epoch]
            if item["shortlist_rank_features"]["rule_b_precision_floor"]:
                requested.append((f"fixed_epoch_{epoch:02d}", item))
        if eligible_b:
            best_f1 = max(eligible_b, key=lambda item: (
                item["shortlist_rank_features"]["rule_b_target_bag_f1"],
                item["shortlist_rank_features"]["rule_b_distinct_target_recall"],
                -float(item["checkpoint_info"]["epoch"])))
            requested.append(("best_rule_b_target_bag_f1", best_f1))
        else:
            best_f1 = max(summaries, key=lambda item: (
                item["shortlist_rank_features"]["rule_b_target_bag_f1"],
                item["shortlist_rank_features"]["rule_b_distinct_target_recall"],
                -float(item["checkpoint_info"]["epoch"])))
            requested.append(("best_rule_b_target_bag_f1_precision_floor_unmet", best_f1))

        recall_fallback = False
        if eligible_r:
            best_recall = max(eligible_r, key=lambda item: (
                item["shortlist_rank_features"]["rule_r_distinct_target_recall"],
                item["shortlist_rank_features"]["rule_r_target_bag_precision"],
                -float(item["checkpoint_info"]["epoch"])))
        else:
            recall_fallback = True
            best_recall = max(summaries, key=lambda item: (
                item["shortlist_rank_features"]["rule_r_distinct_target_recall"],
                item["shortlist_rank_features"]["rule_r_target_bag_precision"],
                -float(item["checkpoint_info"]["epoch"])))
        requested.append(("best_rule_r_distinct_target_recall_precision_floor_fallback" if recall_fallback
                          else "best_rule_r_distinct_target_recall", best_recall))

        shortlist: list[dict[str, Any]] = []
        seen: set[str] = set()
        for reason, item in requested:
            path_key = str(Path(str(item["checkpoint_info"]["path"])).resolve())
            if path_key in seen:
                continue
            seen.add(path_key)
            shortlist.append({
                "shortlist_index": len(shortlist), "reason": reason,
                "checkpoint_info": item["checkpoint_info"],
                "rule_fits": item["rule_fits"],
                "shortlist_rank_features": item["shortlist_rank_features"],
            })
        if len(shortlist) > 5:
            raise AssertionError(f"registered shortlist exceeds five: {len(shortlist)}")
        payload = {
            "format": "locatemot-l88-dev-shortlist-v1", "status": "complete",
            "stage": "internal fit/dev rule fitting; final selection pending full-video TrackEval",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD, "seed": SEED,
            "source_scores_dir": str(source), "source_scores_sha256": sha256_file(records_path),
            "source_record_count": sum(len(rows) for rows in grouped.values()),
            "checkpoint_count": len(summaries), "expected_checkpoint_epochs": EXPECTED_EPOCHS,
            "checkpoint_summaries": summaries, "shortlist": shortlist,
            "shortlist_count": len(shortlist), "recall_precision_floor_fallback": recall_fallback,
            "shortlist_protocol": {
                "fixed_epochs": [8, 20, 40], "precision_floor": 0.08,
                "fixed_and_f1_rule": "Rule B target-bag precision",
                "recall_rule": "Rule R distinct-target recall among target-bag precision >= 0.08",
                "fallback": "highest Rule R distinct-target recall if no checkpoint meets precision floor",
                "deduplicate_by_checkpoint_path": True, "maximum": 5,
            },
            "fit_dev_labels_only": True, "fixed_calibration_read": False,
            "fixed_validation_read": False, "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False, "no_hota_or_trackeval": True,
            "candidate_deletion": False, "candidate_truncation": False,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "selection_pending_trackeval": True,
            "next_action": "run full-video internal dev TrackEval for Rules B/R/P on this deduplicated shortlist",
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(out / "shortlist.json", payload)
        write_json(out / "dev_rule_fits.json", {"format": "locatemot-l88-dev-rule-fits-v1",
                                                  "status": "complete", "summaries": summaries,
                                                  "screening_gt_used": False, "official_test_labels_read": False,
                                                  "ordinary_mot_ovmot_touched": False,
                                                  "hota_trackeval_run": False, "no_hota_or_trackeval": True})
        provenance = {
            "format": "locatemot-l88-dev-shortlist-provenance-v1", "status": "complete",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD, "seed": SEED,
            "inputs": {"score_records": str(records_path), "score_records_sha256": sha256_file(records_path),
                       "source_cheap_dev": str((source / "cheap_dev_scores.json").resolve()),
                       "manifest": str(MANIFEST), "manifest_sha256": MANIFEST_SHA},
            "outputs": [str((out / "shortlist.json").resolve()), str((out / "dev_rule_fits.json").resolve())],
            "label_boundary": "only fit/dev labels were present; fixed calibration/validation were not read",
            "shortlist_frozen_for_fullvideo_dev": True, "candidate_deletion": False,
            "candidate_truncation": False, "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False, "no_hota_or_trackeval": True,
        }
        write_json(out / "provenance.json", provenance)
        write_json(out / "status.json", {"format": "locatemot-l88-dev-shortlist-status-v1", "status": "complete",
                                          "checkpoint_count": len(summaries), "shortlist_count": len(shortlist),
                                          "record_count": sum(len(rows) for rows in grouped.values()),
                                          "selection_pending_trackeval": True, "screening_gt_used": False,
                                          "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                                          "hota_trackeval_run": False, "no_hota_or_trackeval": True})
        return 0
    except Exception:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L88 dev shortlist — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {"format": "locatemot-l88-dev-shortlist-status-v1", "status": "incomplete",
                                          "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                          "failure_root_cause": "first traceback in INCOMPLETE.md",
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
