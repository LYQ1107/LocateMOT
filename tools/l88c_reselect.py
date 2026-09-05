#!/usr/bin/env python3
"""Refit the registered L88 dev rules with the corrected NULL equation.

This is a zero-training replay of the existing cheap-dev score records.  The
old L88 metric helper remains untouched; all candidate-vs-NULL measurements
in this module use :func:`l88c_eval_metrics.corrected_emission_mask`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

from l88_eval_metrics import metric as legacy_metric
from l88_eval_common import MANIFEST, MANIFEST_SHA, THREAD, write_json
from l88c_eval_metrics import fit_rule_set, metric


WORK_ROOT = Path(__file__).resolve().parents[1]
SHORTLIST_EPOCHS = (8, 20, 40)
ALL_EPOCHS = tuple(range(2, 41, 2))
RULES = ("B", "R", "P")
DATASETS = ("refer_kitti_v1", "refer_kitti_v2")
EXPECTED_RECORDS = 9960


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _checkpoint_info(records: list[dict[str, Any]], epoch: int) -> dict[str, Any]:
    values = [row.get("checkpoint") for row in records if row.get("checkpoint")]
    if not values:
        raise AssertionError(f"cheap-dev checkpoint metadata missing for epoch {epoch}")
    first = dict(values[0])
    if int(first.get("epoch", -1)) != int(epoch):
        raise AssertionError(f"cheap-dev checkpoint epoch drift: {epoch} / {first}")
    serialized = {json.dumps(value, sort_keys=True, default=str) for value in values}
    if len(serialized) != 1:
        raise AssertionError(f"checkpoint metadata drift within epoch {epoch}")
    path = Path(str(first["path"])).resolve()
    expected_sha = str(first["sha256"])
    if sha256(path) != expected_sha:
        raise AssertionError(f"checkpoint SHA drift: {path}")
    return first


def _verify_epoch(records: list[dict[str, Any]], epoch: int) -> None:
    expected = [row for row in records if int(row.get("checkpoint", {}).get("epoch", -1)) == epoch]
    if not expected:
        raise AssertionError(f"missing cheap-dev epoch {epoch}")
    keys = [str(row["unit_key"]) for row in expected]
    if len(keys) != len(set(keys)):
        raise AssertionError(f"duplicate unit keys in cheap-dev epoch {epoch}")
    for row in expected:
        count = int(row["candidate_count"])
        lengths = [len(row[name]) for name in ("score", "row_keys", "row_offsets", "candidate_indices")]
        if any(value != count for value in lengths):
            raise AssertionError(f"candidate row length drift: epoch={epoch} {row['unit_key']}")
        if bool(row.get("candidate_deletion")) or bool(row.get("candidate_truncation")):
            raise AssertionError(f"candidate deletion/truncation in cheap-dev: {row['unit_key']}")
        if not bool(row.get("finite_scores", True)):
            raise AssertionError(f"nonfinite cheap-dev row: {row['unit_key']}")
        if not all(len(key) == 6 for key in row["row_keys"]):
            raise AssertionError(f"row-key schema drift: {row['unit_key']}")


def _candidate_record(epoch: int, checkpoint: dict[str, Any], fits: dict[str, Any],
                      source_sha: str) -> dict[str, Any]:
    return {
        "candidate_name": f"corrected_epoch{epoch:03d}",
        "selection_reason": "corrected candidate-vs-NULL dev rule replay",
        "checkpoint_info": checkpoint,
        "rule_fits": fits,
        "source_scores_sha256": source_sha,
        "zero_training": True,
        "corrected_candidate_vs_null": True,
    }


def run(args: argparse.Namespace) -> int:
    if Path.cwd().resolve() != WORK_ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA drift")
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L88C reselect output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    try:
        source = args.scores.resolve()
        source_sha = sha256(source)
        records = _read_jsonl(source)
        if len(records) != EXPECTED_RECORDS:
            raise AssertionError(f"cheap-dev record count drift: {len(records)} != {EXPECTED_RECORDS}")
        by_epoch: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            epoch = int(row.get("checkpoint", {}).get("epoch", -1))
            by_epoch[epoch].append(row)
        if tuple(sorted(by_epoch)) != ALL_EPOCHS:
            raise AssertionError(f"cheap-dev epoch set drift: {sorted(by_epoch)}")

        rule_fit_by_epoch: dict[int, dict[str, Any]] = {}
        candidate_by_epoch: dict[int, dict[str, Any]] = {}
        for epoch in ALL_EPOCHS:
            epoch_records = by_epoch[epoch]
            _verify_epoch(records, epoch)
            checkpoint = _checkpoint_info(epoch_records, epoch)
            fits = fit_rule_set(epoch_records)
            rule_fit_by_epoch[epoch] = fits
            candidate_by_epoch[epoch] = _candidate_record(epoch, checkpoint, fits, source_sha)

        # High-value pure deployment-gate control: same epoch-20 scores and
        # thresholds as the original L88 final Rule B, only the equation changes.
        epoch20 = by_epoch[20]
        pure_thresholds = {"candidate_threshold": 0.75,
                           "presence_threshold": 0.0, "null_margin": 0.0}
        pure_legacy = legacy_metric(epoch20, **pure_thresholds)
        pure_corrected = metric(epoch20, **pure_thresholds)
        pure_gate = {
            "format": "locatemot-l88c-pure-gate-v1", "status": "complete",
            "name": "L88C-PURE-GATE", "epoch": 20, "rule": "B",
            "thresholds": pure_thresholds, "legacy_l88_metric": pure_legacy,
            "corrected_metric": pure_corrected,
            "only_changed": "candidate energy is compared independently with NULL; presence is independent",
            "zero_training": True, "corrected_candidate_vs_null": True,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
        }

        def b_f1(epoch: int) -> float:
            return float(rule_fit_by_epoch[epoch]["B"]["metrics"]["target_bag_f1"])

        def r_recall(epoch: int) -> float:
            return float(rule_fit_by_epoch[epoch]["R"]["metrics"]["distinct_target_recall"])

        def r_precision(epoch: int) -> float:
            return float(rule_fit_by_epoch[epoch]["R"]["metrics"]["target_bag_precision"])

        best_b = max(ALL_EPOCHS, key=lambda epoch: (b_f1(epoch), -epoch))
        eligible_r = [epoch for epoch in ALL_EPOCHS if r_precision(epoch) >= 0.08]
        best_r = max(eligible_r, key=lambda epoch: (r_recall(epoch), r_precision(epoch), -epoch)) if eligible_r else None

        selected_epochs: list[int] = list(SHORTLIST_EPOCHS)
        reasons: dict[int, list[str]] = {epoch: [f"fixed preregistered epoch {epoch}"] for epoch in selected_epochs}
        selected_epochs.append(best_b)
        reasons.setdefault(best_b, []).append("best corrected Rule B target-bag F1 on dev")
        if best_r is not None:
            selected_epochs.append(best_r)
            reasons.setdefault(best_r, []).append("best corrected Rule R distinct recall at precision >= 0.08 on dev")
        # Keep first occurrence order while deduplicating by checkpoint epoch.
        shortlist: list[dict[str, Any]] = []
        for epoch in selected_epochs:
            if any(int(row["checkpoint_info"]["epoch"]) == epoch for row in shortlist):
                continue
            item = dict(candidate_by_epoch[epoch])
            item["selection_reason"] = "; ".join(reasons[epoch])
            shortlist.append(item)
        if len(shortlist) > 5:
            raise AssertionError(f"corrected shortlist exceeds five entries: {len(shortlist)}")

        all_fit = {
            str(epoch): {rule: {
                "candidate_threshold": fit[rule]["candidate_threshold"],
                "presence_threshold": fit[rule]["presence_threshold"],
                "null_margin": fit[rule]["null_margin"],
                "metrics": fit[rule]["metrics"], "tie_rule": fit[rule]["tie_rule"],
            } for rule in RULES} for epoch, fit in rule_fit_by_epoch.items()
        }
        refit_payload = {
            "format": "locatemot-l88c-dev-rule-refit-v1", "status": "complete",
            "evidence_type": "cheap-dev fit/dev labels only; no new forward or training",
            "source": str(source), "source_sha256": source_sha,
            "record_count": len(records), "epoch_count": len(ALL_EPOCHS),
            "epochs": list(ALL_EPOCHS), "rules": list(RULES),
            "rule_fits": all_fit, "best_rule_b_epoch": best_b,
            "best_rule_r_epoch": best_r, "shortlist_epochs": [int(row["checkpoint_info"]["epoch"]) for row in shortlist],
            "zero_training": True, "corrected_candidate_vs_null": True,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
            "failure_root_cause": None,
            "next_action": "run corrected full-video dev inference for the frozen shortlist",
        }
        shortlist_payload = {
            "format": "locatemot-l88c-shortlist-v1", "status": "complete",
            "evidence_type": "corrected cheap-dev shortlist; selection still pending full-video dev TrackEval",
            "shortlist": shortlist, "shortlist_epochs": [int(row["checkpoint_info"]["epoch"]) for row in shortlist],
            "rule_refit_source": str(source), "rule_refit_sha256": source_sha,
            "best_corrected_rule_b_epoch": best_b, "best_corrected_rule_r_epoch": best_r,
            "pure_gate": pure_gate,
            "zero_training": True, "corrected_candidate_vs_null": True,
            "selection_frozen_before_fullvideo_trackeval": False,
            "selection_frozen_before_fixed_validation": False,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
            "candidate_deletion": False, "candidate_truncation": False,
        }
        provenance = {
            "format": "locatemot-l88c-reselect-provenance-v1", "status": "complete",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "base_l88_sha": "c9b44c07b9b977de9d0f839fb2ff6363abb0386e",
            "source_scores": str(source), "source_scores_sha256": source_sha,
            "source_record_count": len(records), "checkpoint_epochs": list(ALL_EPOCHS),
            "grid": [-1, -.75, -.5, -.25, 0, .25, .5, .75, 1],
            "null_margin_grid": [0, .25, .5, .75],
            "selection_data": "existing cheap-dev records only",
            "labels_read": "existing fit/dev labels embedded in frozen cheap-dev records; no validation labels",
            "zero_training": True, "corrected_candidate_vs_null": True,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
        }
        write_json(out / "corrected_rule_refit.json", refit_payload)
        write_json(out / "shortlist.json", shortlist_payload)
        write_json(out / "pure_gate.json", pure_gate)
        write_json(out / "provenance.json", provenance)
        write_json(out / "status.json", {
            "format": "locatemot-l88c-reselect-status-v1", "status": "complete",
            "shortlist_count": len(shortlist), "shortlist_epochs": [int(row["checkpoint_info"]["epoch"]) for row in shortlist],
            "record_count": len(records), "zero_training": True, "corrected_candidate_vs_null": True,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
        })
        return 0
    except Exception:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L88C reselect — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {
            "format": "locatemot-l88c-reselect-status-v1", "status": "incomplete",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "failure_root_cause": "first traceback in INCOMPLETE.md", "zero_training": True,
            "corrected_candidate_vs_null": True, "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "no_hota_or_trackeval": True,
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
