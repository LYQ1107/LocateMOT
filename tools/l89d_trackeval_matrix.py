#!/usr/bin/env python3
"""TrackEval matrix for L89D inference with a verified native timeline."""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import Any

from l89d_fullvideo_common import (
    MANIFEST_SHA,
    THREAD,
    WORK_ROOT,
    command_line,
    manifest_assertion,
    sha256_file,
    standard_flags,
    write_json,
)
from l89_trackeval_matrix import PERCENT_METRICS, np_mean, run_dataset  # noqa: E402


FORMAT = "locatemot-l89d-trackeval-matrix-v1"


def _verify_timeline(summary: dict[str, Any], scope: str) -> None:
    if summary.get("format") != "locatemot-l89d-true-fullvideo-v1" or summary.get("status") != "complete":
        raise AssertionError("inference summary is not an L89D complete summary")
    if str(summary.get("scope_key")) != scope or not bool(summary.get("full_video")):
        raise AssertionError("inference summary is not a formal full-video scope")
    if summary.get("screening_gt_used") or summary.get("official_test_labels_read"):
        raise AssertionError("inference summary crossed forbidden label boundary")
    timeline = summary.get("timeline_contract") or {}
    if not all(bool(timeline.get(name)) for name in ("all_candidates_full_video", "all_candidates_pair_complete", "native_frame_source")):
        # native_frame_source is a non-empty provenance string, not a bool.
        if not (bool(timeline.get("all_candidates_full_video")) and bool(timeline.get("all_candidates_pair_complete")) and summary.get("native_frame_source")):
            raise AssertionError("inference timeline contract proof is incomplete")
    if int(summary.get("expected_query_frame_pairs_all_candidates", -1)) != int(summary.get("actual_query_frame_pairs_all_candidates", -2)):
        raise AssertionError("inference expected/actual pair totals differ")
    for candidate in summary.get("candidates", []):
        if not bool(candidate.get("full_video")):
            raise AssertionError("candidate summary is not full video")
        if int(candidate.get("expected_query_frame_pairs", -1)) != int(candidate.get("actual_query_frame_pairs", -2)):
            raise AssertionError("candidate expected/actual pair mismatch")
        candidate_path = Path(str(candidate["summary_path"])).resolve()
        if not candidate_path.is_file():
            raise FileNotFoundError(candidate_path)
        nested = __import__("json").loads(candidate_path.read_text(encoding="utf-8"))
        if nested.get("status") != "complete" or not bool(nested.get("full_video")):
            raise AssertionError(f"nested candidate summary incomplete: {candidate_path}")
        for video in nested.get("timeline", {}).get("videos", []):
            if int(video["visited_frame_count"]) != int(video["native_frame_count"]):
                raise AssertionError(f"native frame visit mismatch: {video}")
            if int(video["actual_query_frame_pairs"]) != int(video["expected_query_frame_pairs"]):
                raise AssertionError(f"native query pair mismatch: {video}")


def _descriptor_mean(candidate: dict[str, Any], rule: str, field: str, default: float) -> float:
    descriptors = candidate.get("rules", {}).get(rule, {}).get("emission_descriptor", {})
    values = [float(value.get(field, default)) for value in descriptors.values() if isinstance(value, dict)]
    return np_mean(values) if values else float(default)


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L89D TrackEval output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = command_line()
    started = time.perf_counter()
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong L89D cwd: {Path.cwd()}")
        manifest_assertion()
        source = args.inference_root.resolve()
        summary_path = source / "summary.json"
        summary = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
        _verify_timeline(summary, args.expected_scope)
        candidates = summary.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise AssertionError("empty L89D inference candidate matrix")
        if args.expected_scope == "internal" and len(candidates) != 1:
            raise AssertionError("internal L89D TrackEval requires one selection")
        results: list[dict[str, Any]] = []
        for candidate in candidates:
            nested_path = Path(str(candidate["summary_path"])).resolve()
            nested = __import__("json").loads(nested_path.read_text(encoding="utf-8"))
            epoch = int(nested["candidate"]["checkpoint"]["epoch"])
            candidate_rules = tuple(nested["candidate"]["rules"].keys())
            if args.expected_scope == "internal" and len(candidate_rules) != 1:
                raise AssertionError("internal candidate has multiple rules")
            for rule in candidate_rules:
                per_dataset: list[dict[str, Any]] = []
                rule_summary = nested["candidate"]["rules"][rule]
                for dataset in sorted(nested["datasets"]):
                    source_paths = rule_summary["strategy_paths"].get(dataset)
                    if not source_paths:
                        raise AssertionError(f"missing strategy path for {dataset}/{rule}")
                    src = Path(str(source_paths["root"])).resolve()
                    dst = out / f"candidate_epoch{epoch:03d}_shortlist{int(nested['candidate'].get('shortlist_index', 0)):02d}" / rule / dataset
                    per_dataset.append(run_dataset(src, dst, dataset, rule))
                metric_keys = set(per_dataset[0]["metrics_raw"])
                metrics = {key: float(np_mean([item["metrics_raw"][key] for item in per_dataset])) for key in metric_keys}
                results.append({
                    "candidate_index": int(candidate.get("candidate_index", 0)),
                    "shortlist_index": int(nested["candidate"].get("shortlist_index", 0)),
                    "checkpoint_info": nested["candidate"]["checkpoint"], "rule": rule,
                    "hota": metrics.get("HOTA___AUC", -1.0), "deta": metrics.get("DetA___AUC", -1.0),
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
            "format": FORMAT, "status": "complete", "scope": args.expected_scope,
            "evidence_type": "L89D full-video legal TrackEval with native timeline; not screening or official test",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "inference_root": str(source), "source_inference_summary_sha256": sha256_file(summary_path),
            "expected_scope": args.expected_scope, "result_count": len(results), "results": results,
            "timeline_contract_passed": True, "native_timeline": True,
            "manifest_sha256": MANIFEST_SHA, "trackeval_root": "/data1/LWR/vranlee/SERVER_ONLY/avis/TrackEval-master",
            "trackeval_git_head": None, "trackeval_git_head_note": "local checkout has no verifiable HEAD",
            "candidate_deletion": False, "candidate_truncation": False,
            "groundingdino_lora_used": False, "groundingdino_trainable": False,
            "bert_trainable": False, "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            **standard_flags(hota_trackeval_run=True), "no_hota_or_trackeval": False,
            "failure_root_cause": None, "next_action": "run L89D selector for dev or inspect final internal metrics",
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(out / "trackeval_matrix.json", payload)
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", {"format": FORMAT, "status": "complete", "scope": args.expected_scope,
                                          "result_count": len(results), "timeline_contract_passed": True,
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": True,
                                          "zero_training": True})
        return 0
    except Exception as exc:
        (out / "INCOMPLETE.md").write_text("# L89D TrackEval matrix — INCOMPLETE\n\n" + traceback.format_exc(), encoding="utf-8")
        write_json(out / "status.json", {"format": FORMAT, "status": "incomplete", "command": command,
                                          "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                          "failure_root_cause": f"{type(exc).__name__}: {exc}",
                                          "next_action": "repair first TrackEval/timeline error and retry in a new output",
                                          **standard_flags(hota_trackeval_run=False)})
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--expected-scope", choices=("dev", "internal"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
