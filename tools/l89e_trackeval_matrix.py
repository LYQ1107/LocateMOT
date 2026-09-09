#!/usr/bin/env python3
"""TrackEval wrapper guarded by L89E native-timeline and phase contracts."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
WORK_ROOT = Path(__file__).resolve().parents[1]
for value in (WORK_ROOT, WORK_ROOT / "tools", ROOT, ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from l89d_fullvideo_common import (  # noqa: E402
    MANIFEST_SHA,
    THREAD,
    WORK_ROOT as COMMON_WORK_ROOT,
    command_line,
    manifest_assertion,
    sha256_file,
    standard_flags,
    write_json,
)
from l89_trackeval_matrix import PERCENT_METRICS, np_mean, run_dataset  # noqa: E402


FORMAT = "locatemot-l89e-trackeval-matrix-v1"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def _verify_timeline(summary: dict[str, Any], scope: str) -> None:
    if summary.get("format") != "locatemot-l89e-phase-consistent-true-fullvideo-v1" or summary.get("status") != "complete":
        raise AssertionError("inference summary is not a complete L89E summary")
    if str(summary.get("scope_key")) != scope or not bool(summary.get("full_video")):
        raise AssertionError("inference summary is not formal full video")
    if not bool(summary.get("phase_consistency_passed")) or not bool(summary.get("phase_consistent_temporal")):
        raise AssertionError("inference phase contract proof is incomplete")
    if summary.get("screening_gt_used") or summary.get("official_test_labels_read"):
        raise AssertionError("inference crossed forbidden label boundary")
    timeline = summary.get("timeline_contract") or {}
    if not bool(timeline.get("all_candidates_full_video")) or not bool(timeline.get("all_candidates_pair_complete")):
        raise AssertionError("inference timeline proof is incomplete")
    if not timeline.get("native_frame_source"):
        raise AssertionError("native frame source missing")
    if int(summary.get("expected_query_frame_pairs_all_candidates", -1)) != int(summary.get("actual_query_frame_pairs_all_candidates", -2)):
        raise AssertionError("inference expected/actual pair totals differ")
    candidates = summary.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise AssertionError("empty L89E candidate matrix")
    if scope == "internal" and len(candidates) != 1:
        raise AssertionError("internal L89E TrackEval requires one candidate")
    for candidate in candidates:
        if not bool(candidate.get("full_video")) or not bool(candidate.get("phase_consistent_temporal")):
            raise AssertionError("candidate is not a phase-consistent full-video run")
        if int(candidate["expected_query_frame_pairs"]) != int(candidate["actual_query_frame_pairs"]):
            raise AssertionError("candidate expected/actual pair mismatch")
        nested_path = Path(str(candidate["summary_path"])).resolve()
        nested = _load(nested_path)
        if nested.get("format") != "locatemot-l89e-phase-consistent-true-fullvideo-v1" or nested.get("status") != "complete" or not bool(nested.get("full_video")):
            raise AssertionError(f"nested L89E summary incomplete: {nested_path}")
        if not bool(nested.get("phase_consistency_passed")):
            raise AssertionError(f"nested phase proof missing: {nested_path}")
        policy = nested.get("candidate", {}).get("phase_policy")
        if not isinstance(policy, dict) or not bool(nested.get("candidate", {}).get("phase_consistent_temporal")):
            raise AssertionError("candidate phase policy missing")
        for video in nested.get("timeline", {}).get("videos", []):
            if int(video["visited_frame_count"]) != int(video["native_frame_count"]):
                raise AssertionError(f"native frame visit mismatch: {video}")
            if int(video["actual_query_frame_pairs"]) != int(video["expected_query_frame_pairs"]):
                raise AssertionError(f"native pair mismatch: {video}")


def _descriptor_mean(candidate: dict[str, Any], rule: str, field: str, default: float) -> float:
    values = [
        float(value.get(field, default))
        for value in candidate.get("rules", {}).get(rule, {}).get("emission_descriptor", {}).values()
        if isinstance(value, dict)
    ]
    return np_mean(values) if values else float(default)


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L89E TrackEval output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = command_line()
    started = time.perf_counter()
    try:
        if Path.cwd().resolve() != COMMON_WORK_ROOT:
            raise RuntimeError(f"wrong L89E worktree cwd: {Path.cwd()}")
        manifest_assertion()
        source = args.inference_root.resolve()
        summary_path = source / "summary.json"
        summary = _load(summary_path)
        _verify_timeline(summary, args.expected_scope)
        results: list[dict[str, Any]] = []
        candidates = summary["candidates"]
        for candidate in candidates:
            nested = _load(Path(str(candidate["summary_path"])))
            epoch = int(nested["candidate"]["checkpoint"]["epoch"])
            rules = tuple(nested["candidate"]["rules"].keys())
            if args.expected_scope == "internal" and len(rules) != 1:
                raise AssertionError("internal L89E candidate has multiple rules")
            for rule in rules:
                rule_summary = nested["candidate"]["rules"][rule]
                per_dataset: list[dict[str, Any]] = []
                for dataset in sorted(nested["datasets"]):
                    source_paths = rule_summary["strategy_paths"].get(dataset)
                    if not source_paths:
                        raise AssertionError(f"missing strategy path: {dataset}/{rule}")
                    src = Path(str(source_paths["root"])).resolve()
                    dst = out / f"candidate_epoch{epoch:03d}_shortlist{int(nested['candidate'].get('shortlist_index', 0)):02d}" / rule / dataset
                    per_dataset.append(run_dataset(src, dst, dataset, rule))
                metric_keys = set(per_dataset[0]["metrics_raw"])
                metrics = {key: float(np_mean([item["metrics_raw"][key] for item in per_dataset])) for key in metric_keys}
                results.append({
                    "candidate_index": int(candidate.get("candidate_index", 0)),
                    "shortlist_index": int(nested["candidate"].get("shortlist_index", 0)),
                    "checkpoint_info": nested["candidate"]["checkpoint"],
                    "rule": rule,
                    "phase_policy": nested["candidate"]["phase_policy"],
                    "phase_consistent_temporal": True,
                    "hota": metrics.get("HOTA___AUC", -1.0),
                    "deta": metrics.get("DetA___AUC", -1.0),
                    "assa": metrics.get("AssA___AUC", -1.0),
                    "distinct_target_recall": _descriptor_mean(nested["candidate"], rule, "distinct_target_recall", 0.0),
                    "inactive_false_acceptance": _descriptor_mean(nested["candidate"], rule, "inactive_false_acceptance", 1.0),
                    "metrics_raw": metrics,
                    "metrics_percent": {key: value * 100.0 for key, value in metrics.items() if key in PERCENT_METRICS},
                    "metrics_counts": {key: value for key, value in metrics.items() if key not in PERCENT_METRICS},
                    "per_dataset": per_dataset,
                    "timeline_contract_passed": True,
                })
        expected_count = len(candidates) * (3 if args.expected_scope == "dev" else 1)
        if len(results) != expected_count:
            raise AssertionError(f"TrackEval result count drift: {len(results)} != {expected_count}")
        payload = {
            "format": FORMAT,
            "status": "complete",
            "scope": args.expected_scope,
            "evidence_type": "L89E phase-consistent true-full-video legal TrackEval; not screening or official test",
            "command": command,
            "cwd": str(COMMON_WORK_ROOT),
            "luna_thread": THREAD,
            "inference_root": str(source),
            "source_inference_summary_sha256": sha256_file(summary_path),
            "expected_scope": args.expected_scope,
            "result_count": len(results),
            "results": results,
            "timeline_contract_passed": True,
            "phase_consistency_passed": True,
            "phase_consistent_temporal": True,
            "native_timeline": True,
            "manifest_sha256": MANIFEST_SHA,
            "trackeval_root": "/data1/LWR/vranlee/SERVER_ONLY/avis/TrackEval-master",
            "trackeval_git_head": None,
            "trackeval_git_head_note": "local checkout has no verifiable HEAD",
            "candidate_deletion": False,
            "candidate_truncation": False,
            "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED",
            **standard_flags(hota_trackeval_run=True),
            "no_hota_or_trackeval": False,
            "failure_root_cause": None,
            "next_action": "run L89E selector for dev or inspect final internal metrics",
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(out / "trackeval_matrix.json", payload)
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", {
            "format": FORMAT,
            "status": "complete",
            "scope": args.expected_scope,
            "result_count": len(results),
            "timeline_contract_passed": True,
            "phase_consistency_passed": True,
            "screening_gt_used": False,
            "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": True,
            "zero_training": True,
        })
        return 0
    except Exception as exc:
        (out / "INCOMPLETE.md").write_text("# L89E TrackEval matrix — INCOMPLETE\n\n" + traceback.format_exc(), encoding="utf-8")
        write_json(out / "status.json", {
            "format": FORMAT,
            "status": "incomplete",
            "command": command,
            "cwd": str(COMMON_WORK_ROOT),
            "luna_thread": THREAD,
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "next_action": "repair the first L89E TrackEval/timeline error and use a new output",
            **standard_flags(hota_trackeval_run=False),
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--expected-scope", choices=("dev", "internal"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
