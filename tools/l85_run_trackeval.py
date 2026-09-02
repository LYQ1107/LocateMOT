#!/usr/bin/env python3
"""Run the local TrackEval checkout on L85 internal full-video outputs.

This wrapper is deliberately separate from inference.  The inference strategy,
checkpoint and internal-validation GT are already frozen before this script is
called.  It evaluates the two legal internal Refer-KITTI domains serially and
parses only TrackEval's combined ``pedestrian_detailed.csv`` rows.  The result
is named *full-video validation HOTA*, not official KITTI test performance.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
TRACK_EVAL = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TrackEval-master").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
INFERENCE_DATASETS = ("refer_kitti_v1", "refer_kitti_v2")
METRIC_KEYS = (
    "HOTA___AUC", "DetA___AUC", "AssA___AUC", "LocA___AUC",
    "DetRe___AUC", "DetPr___AUC", "AssRe___AUC", "AssPr___AUC",
    "IDF1", "IDR", "IDP", "MOTA", "MOTP", "IDSW", "CLR_FP", "CLR_FN",
)
PERCENT_METRICS = {
    "HOTA___AUC", "DetA___AUC", "AssA___AUC", "LocA___AUC",
    "DetRe___AUC", "DetPr___AUC", "AssRe___AUC", "AssPr___AUC",
    "IDF1", "IDR", "IDP", "MOTA", "MOTP",
}


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


def parse_combined(detailed: Path) -> dict[str, float]:
    if not detailed.is_file():
        raise FileNotFoundError(f"TrackEval detailed output missing: {detailed}")
    with detailed.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next((item for item in rows if item.get("seq") == "COMBINED"), None)
    if row is None:
        raise RuntimeError(f"COMBINED row missing from {detailed}")
    result: dict[str, float] = {}
    for key in METRIC_KEYS:
        if key in row and row[key] not in ("", None):
            value = float(row[key])
            if not (value == value) or value in (float("inf"), float("-inf")):
                raise ValueError(f"non-finite TrackEval metric {key} in {detailed}")
            # TrackEval writes the HOTA/CLEAR/Identity summary values in [0,1].
            # Keep the raw value and a percent presentation in the report; this
            # avoids silently mixing conventions across historical scripts.
            result[key] = value
    if "HOTA___AUC" not in result:
        raise RuntimeError(f"HOTA___AUC missing from {detailed}")
    return result


def run_dataset(dataset: str, inference_root: Path, output_root: Path) -> dict[str, Any]:
    source_root = inference_root / dataset
    gt_folder = source_root / "gt"
    tracker_folder = source_root / "trackers"
    seqmap = source_root / "seqmap.txt"
    if not gt_folder.is_dir() or not tracker_folder.is_dir() or not seqmap.is_file():
        raise FileNotFoundError(f"inference contract incomplete for {dataset}: {source_root}")
    sequences = [line.strip() for line in seqmap.read_text().splitlines()[1:] if line.strip()]
    if not sequences:
        raise ValueError(f"empty seqmap for {dataset}")
    tracker_files = sorted((tracker_folder / "l85" / "data").glob("*.txt"))
    if len(tracker_files) != len(sequences):
        raise AssertionError(f"tracker/seqmap mismatch {dataset}: {len(tracker_files)} != {len(sequences)}")

    dataset_out = output_root / dataset
    if dataset_out.exists() and any(dataset_out.iterdir()):
        raise FileExistsError(f"refusing nonempty TrackEval output: {dataset_out}")
    dataset_out.mkdir(parents=True, exist_ok=True)
    log_path = dataset_out / "trackeval.log"
    # The checked-out CLI in this environment converts every option whose
    # default is None into a list, including OUTPUT_FOLDER and SEQMAP_FILE.
    # The dataset implementation requires those two values to be strings.  A
    # direct call through the exact same local TrackEval API lets this wrapper
    # state the types explicitly while preserving the validated sequence set.
    if str(TRACK_EVAL) not in sys.path:
        sys.path.insert(0, str(TRACK_EVAL))
    import trackeval  # pylint: disable=import-outside-toplevel

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL": False, "NUM_PARALLEL_CORES": 1,
        "BREAK_ON_ERROR": True, "RETURN_ON_ERROR": False,
        "PRINT_RESULTS": False, "PRINT_ONLY_COMBINED": False,
        "PRINT_CONFIG": True, "TIME_PROGRESS": True,
        "DISPLAY_LESS_PROGRESS": True, "OUTPUT_SUMMARY": True,
        "OUTPUT_EMPTY_CLASSES": True, "OUTPUT_DETAILED": True,
        "PLOT_CURVES": False,
        "LOG_ON_ERROR": str((dataset_out / "trackeval_error.log").resolve()),
    })
    dataset_config = {
        "GT_FOLDER": str(gt_folder.resolve()),
        "TRACKERS_FOLDER": str(tracker_folder.resolve()),
        "OUTPUT_FOLDER": str((dataset_out / "results").resolve()),
        "TRACKERS_TO_EVAL": ["l85"],
        "CLASSES_TO_EVAL": ["pedestrian"],
        "BENCHMARK": "MOT17", "SPLIT_TO_EVAL": "train",
        "INPUT_AS_ZIP": False, "PRINT_CONFIG": True,
        "DO_PREPROC": True, "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "", "TRACKER_DISPLAY_NAMES": None,
        "SEQMAP_FOLDER": None, "SEQMAP_FILE": None,
        "SEQ_INFO": {seq: None for seq in sequences},
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt.txt",
        "SKIP_SPLIT_FOL": True,
    }
    metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"],
                      "THRESHOLD": 0.5, "PRINT_CONFIG": True}
    metrics_list = []
    for metric in (trackeval.metrics.HOTA, trackeval.metrics.CLEAR,
                   trackeval.metrics.Identity):
        if metric.get_name() in metrics_config["METRICS"]:
            metrics_list.append(metric(metrics_config))
    if not metrics_list:
        raise RuntimeError("no TrackEval metrics configured")
    started = time.perf_counter()
    with log_path.open("w") as handle:
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            evaluator = trackeval.Evaluator(eval_config)
            dataset_object = trackeval.datasets.MotChallenge2DBox(dataset_config)
            evaluator.evaluate([dataset_object], metrics_list)
    detailed = dataset_out / "results" / "l85" / "pedestrian_detailed.csv"
    metrics = parse_combined(detailed)
    return {
        "dataset": dataset,
        "sequence_count": len(sequences),
        "sequences": sequences,
        "tracker_file_count": len(tracker_files),
        "trackeval_command": "direct TrackEval API; equivalent local runner config",
        "trackeval_config": {"evaluator": eval_config, "dataset": dataset_config,
                              "metrics": metrics_config},
        "trackeval_log": str(log_path.resolve()),
        "detailed_csv": str(detailed.resolve()),
        "metrics_raw": metrics,
        "metrics_percent": {key: value * 100.0 for key, value in metrics.items()
                             if key in PERCENT_METRICS},
        "metrics_counts": {key: value for key, value in metrics.items()
                            if key not in PERCENT_METRICS},
        "elapsed_seconds": time.perf_counter() - started,
    }


def run(args: argparse.Namespace) -> int:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if not TRACK_EVAL.is_dir() or not (TRACK_EVAL / "scripts/run_mot_challenge.py").is_file():
        raise FileNotFoundError(f"local TrackEval checkout unavailable: {TRACK_EVAL}")
    if not MANIFEST.is_file() or sha256_file(MANIFEST) != MANIFEST_SHA:
        raise AssertionError("fixed L19 manifest SHA drift")
    inference_root = args.inference_root.resolve()
    output_root = args.out.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing nonempty output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    command_text = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    try:
        source_summary = inference_root / "summary.json"
        summary = json.loads(source_summary.read_text())
        if summary.get("status") != "complete" or not summary.get("full_video"):
            raise AssertionError("source inference is not complete full-video output")
        if summary.get("screening_gt_used") or summary.get("official_test_labels_read"):
            raise AssertionError("source inference has forbidden label flags")
        datasets = list(args.datasets) if args.datasets else list(INFERENCE_DATASETS)
        results = [run_dataset(dataset, inference_root, output_root) for dataset in datasets]
        payload = {
            "format": "locatemot-l85-internal-fullvideo-trackeval-v1",
            "status": "complete",
            "evidence_type": "full-video validation HOTA",
            "not_official_test": True,
            "not_screening": True,
            "command": command_text,
            "cwd": str(ROOT),
            "luna_thread": THREAD,
            "python": sys.executable,
            "trackeval_root": str(TRACK_EVAL),
            "trackeval_git_head": None,
            "trackeval_git_head_note": "local checkout has no verifiable HEAD",
            "inference_root": str(inference_root),
            "inference_summary_sha256": sha256_file(source_summary),
            "datasets": results,
            "manifest": {"path": str(MANIFEST), "sha256": MANIFEST_SHA},
            "screening_gt_used": False,
            "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": True,
            "no_hota_or_trackeval": False,
            "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED",
            "wall_seconds": time.perf_counter() - started,
            "failure_root_cause": None,
            "next_action": "write the L85 final evidence report and supervisor handoff",
        }
        write_json(output_root / "trackeval_summary.json", payload)
        write_json(output_root / "provenance.json", payload)
        write_json(output_root / "status.json", payload)
        return 0
    except Exception:
        trace = traceback.format_exc()
        (output_root / "INCOMPLETE.md").write_text("# L85 TrackEval — INCOMPLETE\n\n" + trace)
        write_json(output_root / "status.json", {
            "format": "locatemot-l85-internal-fullvideo-trackeval-v1",
            "status": "incomplete", "command": command_text, "cwd": str(ROOT),
            "luna_thread": THREAD, "failure_root_cause": "first traceback in INCOMPLETE.md",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", choices=INFERENCE_DATASETS)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
