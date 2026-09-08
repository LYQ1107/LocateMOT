#!/usr/bin/env python3
"""Create the registered L89 checkpoint shortlist or finalize it from dev TrackEval.

Without ``--trackeval`` this consumes only fit/dev score records and writes a
maximum-five shortlist.  With it, the same shortlist is frozen and the final
choice is made from legal internal dev full-video metrics only; fixed
calibration/validation is never read here.
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

import numpy as np


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
WORK_ROOT = Path(__file__).resolve().parents[1]
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"

for path in (WORK_ROOT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import locatemot.rmot as _rmot_package  # noqa: E402
package_path = str(WORK_ROOT / "locatemot" / "rmot")
if package_path not in [str(value) for value in _rmot_package.__path__]:
    _rmot_package.__path__.append(package_path)
from l88c_eval_metrics import fit_rule_set as corrected_fit_rule_set  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def read_scores(path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("format") != "locatemot-l89-score-record-v1":
            raise AssertionError(f"unexpected score format at line {line_number}")
        count = int(row["candidate_count"])
        for name in ("score", "candidate_energy", "r_static", "r_total", "candidate_prior", "labels", "candidate_gt", "row_keys"):
            if len(row.get(name, [])) != count:
                raise AssertionError(f"L89 score row length drift {name}: {row.get('unit_key')}")
        if not bool(row.get("finite_scores")) or row.get("candidate_deletion") or row.get("candidate_truncation"):
            raise AssertionError(f"invalid L89 score flags: {row.get('unit_key')}")
        if not np.isfinite(np.asarray(row["score"], dtype=np.float64)).all():
            raise AssertionError(f"nonfinite L89 score: {row.get('unit_key')}")
        checkpoint = row.get("checkpoint") or {}
        key = str(Path(str(checkpoint["path"])).resolve())
        grouped[key].append(row)
    return grouped


def info(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise AssertionError("empty L89 checkpoint record group")
    values = [row["checkpoint"] for row in rows]
    if len({int(value["epoch"]) for value in values}) != 1 or len({str(value["sha256"]) for value in values}) != 1:
        raise AssertionError("checkpoint metadata drift in L89 score records")
    return dict(values[0])


def fit_rules_fast(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = corrected_fit_rule_set(records)
    for name in ("B", "R", "P"):
        if name not in result:
            raise AssertionError(f"corrected L89C rule missing: {name}")
        metrics = result[name]["metrics"]
        if str(metrics.get("rule")) != "grid_candidate_energy_null":
            raise AssertionError(
                f"wrong L89C emission rule for {name}: {metrics.get('rule')}"
            )
    return result


def build_shortlist(grouped: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for _path, rows in sorted(grouped.items(), key=lambda value: int(value[1][0]["checkpoint"]["epoch"])):
        rules = fit_rules_fast(rows)
        b = rules["B"]["metrics"]
        r = rules["R"]["metrics"]
        summaries.append({"checkpoint_info": info(rows), "record_count": len(rows), "rule_fits": rules,
                          "rule_b_target_bag_f1": float(b["target_bag_f1"]),
                          "rule_b_distinct_target_recall": float(b["distinct_target_recall"]),
                          "rule_b_target_bag_precision": float(b["target_bag_precision"]),
                          "rule_r_distinct_target_recall": float(r["distinct_target_recall"]),
                          "rule_r_target_bag_precision": float(r["target_bag_precision"])})
    by_epoch = {int(x["checkpoint_info"]["epoch"]): x for x in summaries}
    if not set((8, 20, 40)).issubset(by_epoch):
        raise AssertionError("registered fixed shortlist epochs missing")
    requested: list[tuple[str, dict[str, Any]]] = [(f"fixed_epoch_{epoch:02d}", by_epoch[epoch]) for epoch in (8, 20, 40)]
    best_f1 = max(summaries, key=lambda x: (x["rule_b_target_bag_f1"], x["rule_b_distinct_target_recall"], -int(x["checkpoint_info"]["epoch"])))
    requested.append(("best_rule_b_target_bag_f1", best_f1))
    eligible = [x for x in summaries if x["rule_r_target_bag_precision"] >= 0.08]
    best_recall = max(eligible or summaries, key=lambda x: (x["rule_r_distinct_target_recall"], x["rule_r_target_bag_precision"], -int(x["checkpoint_info"]["epoch"])))
    requested.append(("best_rule_r_distinct_target_recall_precision_floor" if eligible else "best_rule_r_distinct_target_recall_precision_fallback", best_recall))
    shortlist: list[dict[str, Any]] = []
    seen: set[str] = set()
    for reason, value in requested:
        path = str(Path(str(value["checkpoint_info"]["path"])).resolve())
        if path in seen:
            continue
        seen.add(path)
        shortlist.append({"shortlist_index": len(shortlist), "reason": reason,
                          "checkpoint_info": value["checkpoint_info"], "rule_fits": value["rule_fits"],
                          "fit_dev_summary": value})
    if len(shortlist) > 5:
        raise AssertionError("L89 shortlist exceeds five")
    return shortlist


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--trackeval", type=Path, default=None)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L89 selection output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv]); started = time.perf_counter()
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong L89 cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        grouped = read_scores(args.scores.resolve())
        shortlist = build_shortlist(grouped)
        if args.trackeval is None:
            payload = {
                "format": "locatemot-l89-dev-shortlist-v1", "status": "complete",
                "stage": "fit/dev score shortlist; full-video dev TrackEval selection pending",
                "evaluation_contract": "candidate_energy_vs_null",
                "protocol_repair_stage": "L89C", "zero_training": True,
                "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD, "seed": SEED,
                "source_scores": str(args.scores.resolve()), "source_scores_sha256": sha256_file(args.scores.resolve()),
                "shortlist": shortlist, "shortlist_count": len(shortlist),
                "shortlist_rule": {"fixed_epochs": [8, 20, 40], "best": ["target_bag_f1", "distinct_target_recall_with_precision_0.08"], "max": 5},
                "fixed_calibration_read": False, "fixed_validation_read": False,
                "screening_gt_used": False, "official_test_labels_read": False,
                "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                "no_hota_or_trackeval": True, "candidate_deletion": False, "candidate_truncation": False,
                "failure_root_cause": None, "next_action": "run full-video internal dev B/R/P matrix",
                "wall_seconds": time.perf_counter() - started,
            }
            write_json(out / "shortlist.json", payload)
            write_json(out / "status.json", {"format": "locatemot-l89-dev-shortlist-v1", "status": "complete",
                                               "evaluation_contract": "candidate_energy_vs_null", "protocol_repair_stage": "L89C", "zero_training": True,
                                               "shortlist_count": len(shortlist), "selection_pending_trackeval": True,
                                               "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
            write_json(out / "provenance.json", payload | {"format": "locatemot-l89-dev-shortlist-provenance-v1"})
            return 0
        trackeval = json.loads(args.trackeval.resolve().read_text())
        if trackeval.get("status") != "complete":
            raise AssertionError("dev TrackEval matrix is not complete")
        # The matrix contains one frozen-rule summary per shortlist item.  The
        # registered tie-break is HOTA, then DetA, then AssA, then earlier epoch.
        candidates = trackeval.get("results") or trackeval.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise AssertionError("dev TrackEval matrix has no candidates")
        by_path = {str(Path(str(item["checkpoint_info"]["path"])).resolve()): item for item in shortlist}
        legal = [item for item in candidates if str(Path(str(item["checkpoint_info"]["path"])).resolve()) in by_path]
        if len(legal) != len(shortlist) * 3:
            raise AssertionError("dev TrackEval shortlist/rule matrix mismatch")

        def trackeval_key(item: dict[str, Any]) -> tuple[Any, ...]:
            # Exact registered order: HOTA, DetA, AssA, distinct-target
            # recall, lower inactive acceptance, then earlier epoch.  The
            # final rule name is only a deterministic tie breaker after the
            # registered fields.
            inactive = item.get("inactive_false_acceptance")
            inactive_value = 1.0 if inactive is None else float(inactive)
            rule_order = {"B": 0, "R": 1, "P": 2}
            return (
                float(item.get("hota", item.get("HOTA", -1.0))),
                float(item.get("deta", item.get("DetA", -1.0))),
                float(item.get("assa", item.get("AssA", -1.0))),
                float(item.get("distinct_target_recall", -1.0)),
                -inactive_value,
                -int(item["checkpoint_info"]["epoch"]),
                -rule_order.get(str(item.get("rule", "")), 99),
            )

        chosen = max(legal, key=trackeval_key)
        source = by_path[str(Path(str(chosen["checkpoint_info"]["path"])).resolve())]
        rule_name = str(chosen.get("rule", "B"))
        rule = source["rule_fits"].get(rule_name)
        if rule is None:
            raise AssertionError(f"chosen dev rule missing: {rule_name}")
        selection = {
            "format": "locatemot-l89-checkpoint-selection-v1", "status": "complete",
            "stage": "internal full-video dev TrackEval selection; fixed validation pending",
            "evaluation_contract": "candidate_energy_vs_null",
            "protocol_repair_stage": "L89C", "zero_training": True,
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD, "seed": SEED,
            "shortlist": shortlist, "trackeval_source": str(args.trackeval.resolve()),
            "trackeval_source_sha256": sha256_file(args.trackeval.resolve()), "trackeval_candidates": candidates,
            "final_selection": {"checkpoint_info": chosen["checkpoint_info"], "rule": rule_name,
                                "rule_fit": rule["metrics"], "rule_object": rule, "selection_metrics": chosen},
            "selection_frozen_before_fixed_validation": True,
            "fixed_calibration_read": False, "fixed_validation_read": False,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": True,
            "no_screening_or_official_test": True, "candidate_deletion": False, "candidate_truncation": False,
            "failure_root_cause": None, "next_action": "run fixed 16-calibration/24-validation semantic replay then internal TrackEval matrix",
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(out / "checkpoint_selection.json", selection)
        write_json(out / "status.json", {"format": "locatemot-l89-checkpoint-selection-v1", "status": "complete",
                                           "evaluation_contract": "candidate_energy_vs_null", "protocol_repair_stage": "L89C", "zero_training": True,
                                           "selected_epoch": int(chosen["checkpoint_info"]["epoch"]), "rule": rule_name,
                                           "selection_frozen_before_fixed_validation": True, "screening_gt_used": False, "official_test_labels_read": False,
                                           "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": True})
        write_json(out / "provenance.json", selection | {"format": "locatemot-l89-checkpoint-selection-provenance-v1"})
        return 0
    except Exception:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L89 checkpoint selection — INCOMPLETE\n\n" + trace, encoding="utf-8")
        write_json(out / "status.json", {"format": "locatemot-l89-checkpoint-selection-v1", "status": "incomplete", "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                           "failure_root_cause": "first traceback in INCOMPLETE.md", "screening_gt_used": False, "official_test_labels_read": False,
                                           "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
