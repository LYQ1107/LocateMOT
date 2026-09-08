#!/usr/bin/env python3
"""Run the local TrackEval checkout on an L89 full-video strategy matrix.

The inference matrix has already frozen the checkpoint/rule and materialized
the legal fit/dev or internal GT.  This wrapper only evaluates those files;
it never reads screening/official-test labels and never selects a strategy.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


WORK_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
TRACK_EVAL = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TrackEval-master").resolve()
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
RULES = ("B", "R", "P")
METRIC_KEYS = (
    "HOTA___AUC", "DetA___AUC", "AssA___AUC", "LocA___AUC",
    "DetRe___AUC", "DetPr___AUC", "AssRe___AUC", "AssPr___AUC",
    "IDF1", "IDR", "IDP", "MOTA", "MOTP", "IDSW", "CLR_FP", "CLR_FN",
)
PERCENT_METRICS = {
    "HOTA___AUC", "DetA___AUC", "AssA___AUC", "LocA___AUC", "DetRe___AUC", "DetPr___AUC",
    "AssRe___AUC", "AssPr___AUC", "IDF1", "IDR", "IDP", "MOTA", "MOTP",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def parse_combined(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    combined = next((row for row in rows if row.get("seq") == "COMBINED"), None)
    if combined is None:
        raise AssertionError(f"TrackEval COMBINED row missing: {path}")
    result: dict[str, float] = {}
    for key in METRIC_KEYS:
        value = combined.get(key)
        if value in (None, ""):
            continue
        parsed = float(value)
        if not (parsed == parsed) or abs(parsed) == float("inf"):
            raise ValueError(f"nonfinite TrackEval metric {key}: {path}")
        result[key] = parsed
    if "HOTA___AUC" not in result:
        raise AssertionError(f"HOTA missing: {path}")
    return result


def run_dataset(source: Path, destination: Path, dataset: str, rule: str) -> dict[str, Any]:
    gt_folder = source / "gt"; tracker_folder = source / "trackers"; seqmap = source / "seqmap.txt"
    if not gt_folder.is_dir() or not tracker_folder.is_dir() or not seqmap.is_file():
        raise FileNotFoundError(f"incomplete L89 strategy: {source}")
    sequences = [line.strip() for line in seqmap.read_text().splitlines()[1:] if line.strip()]
    tracker_name = "l89"
    tracker_files = sorted((tracker_folder / tracker_name / "data").glob("*.txt"))
    if not sequences or len(tracker_files) != len(sequences):
        raise AssertionError(f"sequence/tracker mismatch: {source} {len(sequences)}/{len(tracker_files)}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing nonempty TrackEval destination: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    if str(TRACK_EVAL) not in sys.path:
        sys.path.insert(0, str(TRACK_EVAL))
    import trackeval  # pylint: disable=import-outside-toplevel

    evaluator_config = trackeval.Evaluator.get_default_eval_config()
    evaluator_config.update({
        "USE_PARALLEL": False, "NUM_PARALLEL_CORES": 1, "BREAK_ON_ERROR": True,
        "RETURN_ON_ERROR": False, "PRINT_RESULTS": False, "PRINT_ONLY_COMBINED": False,
        "PRINT_CONFIG": True, "TIME_PROGRESS": True, "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_SUMMARY": True, "OUTPUT_EMPTY_CLASSES": True, "OUTPUT_DETAILED": True,
        "PLOT_CURVES": False, "LOG_ON_ERROR": str((destination / "trackeval_error.log").resolve()),
    })
    dataset_config = {
        "GT_FOLDER": str(gt_folder.resolve()), "TRACKERS_FOLDER": str(tracker_folder.resolve()),
        "OUTPUT_FOLDER": str((destination / "results").resolve()), "TRACKERS_TO_EVAL": [tracker_name],
        "CLASSES_TO_EVAL": ["pedestrian"], "BENCHMARK": "MOT17", "SPLIT_TO_EVAL": "train",
        "INPUT_AS_ZIP": False, "PRINT_CONFIG": True, "DO_PREPROC": True,
        "TRACKER_SUB_FOLDER": "data", "OUTPUT_SUB_FOLDER": "", "TRACKER_DISPLAY_NAMES": None,
        "SEQMAP_FOLDER": None, "SEQMAP_FILE": None, "SEQ_INFO": {seq: None for seq in sequences},
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt.txt", "SKIP_SPLIT_FOL": True,
    }
    metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5, "PRINT_CONFIG": True}
    metrics = [trackeval.metrics.HOTA(metrics_config), trackeval.metrics.CLEAR(metrics_config),
               trackeval.metrics.Identity(metrics_config)]
    log_path = destination / "trackeval.log"
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            evaluator = trackeval.Evaluator(evaluator_config)
            dataset_object = trackeval.datasets.MotChallenge2DBox(dataset_config)
            evaluator.evaluate([dataset_object], metrics)
    detailed = destination / "results" / tracker_name / "pedestrian_detailed.csv"
    raw = parse_combined(detailed)
    return {
        "dataset": dataset, "rule": rule, "sequence_count": len(sequences),
        "tracker_file_count": len(tracker_files), "sequences": sequences,
        "source": str(source.resolve()), "detailed_csv": str(detailed.resolve()),
        "trackeval_root": str(TRACK_EVAL), "trackeval_git_head": None,
        "trackeval_git_head_note": "local checkout has no verifiable HEAD", "metrics_raw": raw,
        "metrics_percent": {key: value * 100.0 for key, value in raw.items() if key in PERCENT_METRICS},
        "metrics_counts": {key: value for key, value in raw.items() if key not in PERCENT_METRICS},
        "elapsed_seconds": time.perf_counter() - started,
    }


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L89 TrackEval matrix: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv]); started = time.perf_counter()
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong L89 cwd: {Path.cwd()}")
        if not TRACK_EVAL.is_dir() or sha256_file(MANIFEST) != MANIFEST_SHA:
            raise AssertionError("TrackEval checkout or fixed manifest invalid")
        source = args.inference_root.resolve()
        summary = json.loads((source / "summary.json").read_text())
        if summary.get("status") != "complete" or not summary.get("full_video"):
            raise AssertionError("inference source is not complete full-video output")
        if summary.get("screening_gt_used") or summary.get("official_test_labels_read"):
            raise AssertionError("forbidden labels in inference source")
        if str(summary.get("scope_key")) != str(args.expected_scope):
            raise AssertionError(
                f"L89C TrackEval scope drift: {summary.get('scope_key')} != {args.expected_scope}"
            )
        if args.expected_scope == "internal":
            if len(summary.get("candidates", [])) != 1:
                raise AssertionError("L89C final internal requires exactly one frozen checkpoint")
            internal_rules = tuple(summary["candidates"][0].get("rules", {}).keys())
            if len(internal_rules) != 1:
                raise AssertionError("L89C final internal requires exactly one frozen rule")
        results: list[dict[str, Any]] = []
        for candidate in summary.get("candidates", []):
            epoch = int(candidate["checkpoint_info"]["epoch"])
            candidate_rules = tuple(candidate.get("rules", {}).keys())
            if not candidate_rules:
                raise AssertionError("L89C inference candidate contains no rules")
            for rule in candidate_rules:
                per_dataset: list[dict[str, Any]] = []
                for dataset in sorted(summary["datasets"]):
                    src = source / f"candidate_epoch{epoch:03d}" / rule / dataset
                    dst = out / f"candidate_epoch{epoch:03d}" / rule / dataset
                    per_dataset.append(run_dataset(src, dst, dataset, rule))
                metrics = {key: float(np_mean([x["metrics_raw"][key] for x in per_dataset]))
                           for key in per_dataset[0]["metrics_raw"]}
                descriptor = candidate["rules"][rule].get("emission_descriptor", {})
                flat_descriptors = list(descriptor.values())
                distinct = float(np_mean([x.get("distinct_target_recall", 0.0) for x in flat_descriptors])) if flat_descriptors else 0.0
                inactive = float(np_mean([x.get("inactive_false_acceptance", 1.0) for x in flat_descriptors])) if flat_descriptors else 1.0
                results.append({
                    "checkpoint_info": candidate["checkpoint_info"], "rule": rule,
                    "hota": metrics.get("HOTA___AUC", -1.0), "deta": metrics.get("DetA___AUC", -1.0),
                    "assa": metrics.get("AssA___AUC", -1.0), "distinct_target_recall": distinct,
                    "inactive_false_acceptance": inactive, "metrics_raw": metrics,
                    "metrics_percent": {key: value * 100.0 for key, value in metrics.items() if key in PERCENT_METRICS},
                    "per_dataset": per_dataset,
                })
        if not results:
            raise AssertionError("empty L89 TrackEval matrix")
        payload = {
            "format": f"locatemot-l89-{summary['scope_key']}-trackeval-matrix-v1", "status": "complete",
            "scope": summary["scope"], "evidence_type": "full-video legal TrackEval; not screening or official test",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD, "inference_root": str(source),
            "expected_scope": args.expected_scope, "corrected_candidate_vs_null": True, "zero_training": True,
            "source_summary_sha256": sha256_file(source / "summary.json"), "results": results,
            "manifest_sha256": MANIFEST_SHA, "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": True, "no_hota_or_trackeval": False,
            "candidate_deletion": False, "candidate_truncation": False, "groundingdino_lora_used": False,
            "groundingdino_trainable": False, "bert_trainable": False, "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED", "wall_seconds": time.perf_counter() - started,
            "failure_root_cause": None, "next_action": "freeze final selection or run final internal matrix",
        }
        write_json(out / "trackeval_matrix.json", payload); write_json(out / "provenance.json", payload)
        write_json(out / "status.json", {"format": payload["format"], "status": "complete",
                                          "corrected_candidate_vs_null": True, "zero_training": True,
                                          "result_count": len(results),
                                          "scope": summary["scope_key"], "manifest_sha256": MANIFEST_SHA,
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": True})
        return 0
    except Exception:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L89 TrackEval matrix — INCOMPLETE\n\n" + trace, encoding="utf-8")
        write_json(out / "status.json", {"format": "locatemot-l89-trackeval-matrix-v1", "status": "incomplete",
                                          "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                          "failure_root_cause": "first traceback in INCOMPLETE.md", "screening_gt_used": False,
                                          "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                                          "hota_trackeval_run": False})
        raise


def np_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--expected-scope", choices=("dev", "internal"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
