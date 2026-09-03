#!/usr/bin/env python3
"""Run the local TrackEval checkout on a frozen L86 inference/oracle output.

This is a thin L86-only adapter.  It does not create predictions, alter GT,
or select a checkpoint.  The source directory must already contain complete
internal-validation sequence files and a frozen tracker name.
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

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
TRACK_EVAL = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TrackEval-master").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
DATASETS = ("refer_kitti_v1", "refer_kitti_v2")
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
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def parse_combined(path: Path) -> dict[str, float]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next((value for value in rows if value.get("seq") == "COMBINED"), None)
    if row is None:
        raise RuntimeError(f"TrackEval COMBINED row missing: {path}")
    result: dict[str, float] = {}
    for key in METRIC_KEYS:
        value = row.get(key)
        if value in (None, ""):
            continue
        parsed = float(value)
        if not (parsed == parsed) or abs(parsed) == float("inf"):
            raise ValueError(f"nonfinite TrackEval value {key}: {parsed}")
        result[key] = parsed
    if "HOTA___AUC" not in result:
        raise RuntimeError(f"HOTA___AUC missing: {path}")
    return result


def run_dataset(dataset: str, source_root: Path, output_root: Path, tracker_name: str) -> dict[str, Any]:
    source = source_root / dataset
    gt_folder = source / "gt"
    tracker_folder = source / "trackers"
    seqmap = source / "seqmap.txt"
    if not gt_folder.is_dir() or not tracker_folder.is_dir() or not seqmap.is_file():
        raise FileNotFoundError(f"incomplete L86 source: {source}")
    sequences = [line.strip() for line in seqmap.read_text().splitlines()[1:] if line.strip()]
    tracker_files = sorted((tracker_folder / tracker_name / "data").glob("*.txt"))
    if not sequences or len(tracker_files) != len(sequences):
        raise AssertionError(f"sequence/tracker mismatch {dataset}: {len(sequences)} / {len(tracker_files)}")
    dataset_out = output_root / dataset
    if dataset_out.exists() and any(dataset_out.iterdir()):
        raise FileExistsError(f"refusing nonempty TrackEval output: {dataset_out}")
    dataset_out.mkdir(parents=True, exist_ok=True)
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
        "TRACKERS_TO_EVAL": [tracker_name], "CLASSES_TO_EVAL": ["pedestrian"],
        "BENCHMARK": "MOT17", "SPLIT_TO_EVAL": "train", "INPUT_AS_ZIP": False,
        "PRINT_CONFIG": True, "DO_PREPROC": True, "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "", "TRACKER_DISPLAY_NAMES": None,
        "SEQMAP_FOLDER": None, "SEQMAP_FILE": None,
        "SEQ_INFO": {seq: None for seq in sequences},
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt.txt", "SKIP_SPLIT_FOL": True,
    }
    metrics_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5, "PRINT_CONFIG": True}
    metrics = [trackeval.metrics.HOTA(metrics_config), trackeval.metrics.CLEAR(metrics_config),
               trackeval.metrics.Identity(metrics_config)]
    log_path = dataset_out / "trackeval.log"
    started = time.perf_counter()
    with log_path.open("w") as handle:
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            evaluator = trackeval.Evaluator(eval_config)
            dataset_object = trackeval.datasets.MotChallenge2DBox(dataset_config)
            evaluator.evaluate([dataset_object], metrics)
    detailed = dataset_out / "results" / tracker_name / "pedestrian_detailed.csv"
    raw = parse_combined(detailed)
    return {
        "dataset": dataset, "sequence_count": len(sequences), "tracker_name": tracker_name,
        "tracker_file_count": len(tracker_files), "sequences": sequences,
        "trackeval_root": str(TRACK_EVAL), "trackeval_git_head": None,
        "trackeval_git_head_note": "local checkout has no verifiable HEAD",
        "detailed_csv": str(detailed.resolve()), "trackeval_log": str(log_path.resolve()),
        "metrics_raw": raw,
        "metrics_percent": {key: value * 100.0 for key, value in raw.items() if key in PERCENT_METRICS},
        "metrics_counts": {key: value for key, value in raw.items() if key not in PERCENT_METRICS},
        "elapsed_seconds": time.perf_counter() - started,
    }


def run(args: argparse.Namespace) -> int:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if not TRACK_EVAL.is_dir():
        raise FileNotFoundError(TRACK_EVAL)
    if sha256_file(MANIFEST) != MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA drift")
    source_root = args.inference_root.resolve(); output_root = args.out.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing nonempty L86 TrackEval output: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv]); started = time.perf_counter()
    try:
        source_summary = source_root / "summary.json"
        if not source_summary.is_file():
            source_summary = source_root / "trackeval_summary.json"
        summary = json.loads(source_summary.read_text())
        if summary.get("status") != "complete" or not summary.get("full_video", summary.get("full_video_validation", False)):
            raise AssertionError("source is not complete full-video L86 output")
        if summary.get("screening_gt_used") or summary.get("official_test_labels_read"):
            raise AssertionError("source has forbidden label flags")
        tracker_name = str(args.tracker_name)
        datasets = list(args.datasets) if args.datasets else list(DATASETS)
        results = [run_dataset(dataset, source_root, output_root, tracker_name) for dataset in datasets]
        payload = {
            "format": "locatemot-l86-internal-fullvideo-trackeval-v1", "status": "complete",
            "evidence_type": "full-video internal validation HOTA", "hota_scope": "internal_full_video_validation",
            "command": command, "cwd": str(ROOT), "luna_thread": THREAD,
            "python": sys.executable, "source_root": str(source_root),
            "source_summary_sha256": sha256_file(source_summary), "tracker_name": tracker_name,
            "datasets": results, "manifest_sha256": MANIFEST_SHA,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": True,
            "no_hota_or_trackeval": False, "candidate_deletion": False,
            "candidate_truncation": False, "z1_representation_changed": False,
            "groundingdino_lora_used": False, "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED", "wall_seconds": time.perf_counter() - started,
            "failure_root_cause": None, "next_action": "stop and await supervisor review",
        }
        write_json(output_root / "trackeval_summary.json", payload)
        write_json(output_root / "provenance.json", payload); write_json(output_root / "status.json", payload)
        return 0
    except Exception:
        trace = traceback.format_exc()
        (output_root / "INCOMPLETE.md").write_text("# L86 TrackEval — INCOMPLETE\n\n" + trace)
        write_json(output_root / "status.json", {
            "format": "locatemot-l86-internal-fullvideo-trackeval-v1", "status": "incomplete",
            "command": command, "cwd": str(ROOT), "luna_thread": THREAD,
            "failure_root_cause": "first traceback in INCOMPLETE.md",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        })
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tracker-name", default="l86")
    parser.add_argument("--datasets", nargs="*", choices=DATASETS)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
