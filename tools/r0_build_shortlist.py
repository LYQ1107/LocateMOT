#!/usr/bin/env python3
"""Build the registered maximum-three legal dev shortlist for one head."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from tools.r0_common import SEED, THREAD, check_manifest, sha256_file, standard_flags, write_json  # noqa: E402


EXPECTED_EPOCHS = (2, 4, 6, 8, 10, 12)


def load_summaries(root: Path) -> list[dict[str, Any]]:
    result = []
    for epoch in EXPECTED_EPOCHS:
        path = root / f"epoch{epoch:02d}" / "metrics.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "complete":
            raise AssertionError(f"incomplete dev score summary: {path}")
        if value.get("candidate_deletion") or value.get("candidate_truncation"):
            raise AssertionError(f"candidate contract drift in dev score: {path}")
        result.append(value)
    return result


def descriptor(summary: dict[str, Any], role: str) -> dict[str, Any]:
    metrics = summary["metrics"]
    return {
        "role": role, "dataset": summary["dataset"], "epoch": int(summary["checkpoint_info"]["epoch"]),
        "global_step": int(summary["checkpoint_info"]["global_step"]),
        "checkpoint_info": summary["checkpoint_info"], "score_records": summary["score_records"],
        "score_records_sha256": summary["score_records_sha256"], "metrics": metrics,
        "rule": {"membership_threshold": 0.0, "presence_threshold": 0.0, "null_logit": 0.0,
                  "description": "registered zero-logit R0 rule; no threshold fitting"},
    }


def choose(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_epoch = {int(value["checkpoint_info"]["epoch"]): value for value in summaries}
    best_f1 = min(summaries, key=lambda value: (
        -float(value["metrics"]["target_bag_f1"]),
        float(value["metrics"]["target_bag_hard_violation"]),
        -float(value["metrics"]["distinct_target_recall"]),
        float(value["metrics"]["inactive_false_acceptance"]),
        int(value["checkpoint_info"]["epoch"]),
    ))
    eligible = [value for value in summaries if float(value["metrics"]["target_bag_precision"]) >= 0.10]
    best_distinct = min(eligible, key=lambda value: (
        -float(value["metrics"]["distinct_target_recall"]),
        -float(value["metrics"]["target_bag_precision"]),
        -float(value["metrics"]["target_bag_recall"]),
        float(value["metrics"]["inactive_false_acceptance"]),
        int(value["checkpoint_info"]["epoch"]),
    )) if eligible else None
    selected: list[dict[str, Any]] = [descriptor(best_f1, "best_target_bag_f1")]
    if best_distinct is not None:
        selected.append(descriptor(best_distinct, "best_distinct_recall_precision_ge_0.10"))
    selected.append(descriptor(by_epoch[12], "registered_epoch12"))
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for value in selected:
        key = (str(value["checkpoint_info"]["path"]), str(value["rule"]))
        if key not in seen:
            seen.add(key); unique.append(value)
    return unique[:3]


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R0 shortlist output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    base = {"format": "locatemot-r0-dev-shortlist-v1", "status": "incomplete",
            "command": command, "cwd": str(Path.cwd().resolve()), "luna_thread": THREAD,
            "seed": SEED, "source_scores_root": str(args.scores_root.resolve()),
            "manifest_sha256": check_manifest(), "registered_rule": "zero-logit membership/presence",
            "selection_uses_dev_labels_only": True, "fixed_calibration_read": False,
            "fixed_validation_read": False, "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False, "candidate_deletion": False,
            "candidate_truncation": False, "failure_root_cause": None,
            "next_action": "run legal full-video TrackEval for at most three shortlist candidates"}
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R0 shortlist cwd: {Path.cwd()}")
        summaries = load_summaries(args.scores_root.resolve())
        dataset = str(summaries[0]["dataset"])
        if any(str(value["dataset"]) != dataset for value in summaries):
            raise AssertionError("R0 shortlist mixes benchmark-specific heads")
        shortlist = choose(summaries)
        payload = {**base, "status": "complete", "dataset": dataset,
                   "score_summary_count": len(summaries), "shortlist": shortlist,
                   "shortlist_count": len(shortlist), "wall_seconds": time.perf_counter() - started}
        write_json(out / "shortlist.json", payload)
        write_json(out / "provenance.json", payload | {"format": "locatemot-r0-dev-shortlist-provenance-v1"})
        write_json(out / "status.json", payload)
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# R0 dev shortlist — INCOMPLETE\n\n" + trace, encoding="utf-8")
        payload = {**base, "failure_root_cause": f"{type(exc).__name__}: {exc}",
                   "traceback_path": str((out / "INCOMPLETE.md").resolve()),
                   "wall_seconds": time.perf_counter() - started}
        write_json(out / "provenance.json", payload); write_json(out / "status.json", payload)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
