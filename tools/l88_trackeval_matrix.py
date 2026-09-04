#!/usr/bin/env python3
"""Run local TrackEval on completed L88 internal dev matrix outputs.

This adapter consumes only frozen prediction/GT directories created by
``l88_infer_fullvideo_matrix.py``.  It never creates predictions, chooses a
checkpoint, or reads screening/official-test labels.
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
ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
TRACK_EVAL = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TrackEval-master").resolve()
MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
DATASETS = ("refer_kitti_v1", "refer_kitti_v2")
RULES = ("B", "R", "P")
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
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
            raise ValueError(f"nonfinite TrackEval metric {key}: {parsed}")
        result[key] = parsed
    if "HOTA___AUC" not in result:
        raise RuntimeError(f"HOTA___AUC missing: {path}")
    return result


def run_dataset(source: Path, destination: Path, dataset: str, rule: str) -> dict[str, Any]:
    source = source.resolve()
    gt_folder = source / "gt"
    tracker_folder = source / "trackers"
    seqmap = source / "seqmap.txt"
    if not gt_folder.is_dir() or not tracker_folder.is_dir() or not seqmap.is_file():
        raise FileNotFoundError(f"incomplete L88 strategy source: {source}")
    sequences = [line.strip() for line in seqmap.read_text().splitlines()[1:] if line.strip()]
    tracker_name = "l88"
    tracker_files = sorted((tracker_folder / tracker_name / "data").glob("*.txt"))
    if not sequences or len(tracker_files) != len(sequences):
        raise AssertionError(f"sequence/tracker mismatch {source}: {len(sequences)} / {len(tracker_files)}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing nonempty TrackEval result: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
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
        "PLOT_CURVES": False, "LOG_ON_ERROR": str((destination / "trackeval_error.log").resolve()),
    })
    dataset_config = {
        "GT_FOLDER": str(gt_folder), "TRACKERS_FOLDER": str(tracker_folder),
        "OUTPUT_FOLDER": str((destination / "results").resolve()),
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
    log_path = destination / "trackeval.log"
    started = time.perf_counter()
    with log_path.open("w") as handle:
        with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
            evaluator = trackeval.Evaluator(eval_config)
            dataset_object = trackeval.datasets.MotChallenge2DBox(dataset_config)
            evaluator.evaluate([dataset_object], metrics)
    detailed = destination / "results" / tracker_name / "pedestrian_detailed.csv"
    raw = parse_combined(detailed)
    return {
        "dataset": dataset, "rule": rule, "sequence_count": len(sequences),
        "tracker_file_count": len(tracker_files), "sequences": sequences,
        "source": str(source), "detailed_csv": str(detailed.resolve()),
        "trackeval_root": str(TRACK_EVAL), "trackeval_git_head": None,
        "trackeval_git_head_note": "local checkout has no verifiable HEAD",
        "metrics_raw": raw,
        "metrics_percent": {key: value * 100.0 for key, value in raw.items() if key in PERCENT_METRICS},
        "metrics_counts": {key: value for key, value in raw.items() if key not in PERCENT_METRICS},
        "elapsed_seconds": time.perf_counter() - started,
    }


def run(args: argparse.Namespace) -> int:
    if Path.cwd().resolve() != WORK_ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if not TRACK_EVAL.is_dir():
        raise FileNotFoundError(TRACK_EVAL)
    if sha256(MANIFEST) != MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA drift")
    roots = [path.resolve() for path in args.inference_roots]
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L88 TrackEval matrix: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    try:
        for root in roots:
            summary = json.loads((root / "summary.json").read_text())
            if summary.get("status") != "complete" or not summary.get("full_video"):
                raise AssertionError(f"source is not complete full-video L88 output: {root}")
            if summary.get("screening_gt_used") or summary.get("official_test_labels_read"):
                raise AssertionError(f"forbidden labels in source: {root}")
            for candidate in summary.get("candidates", []):
                epoch = int(candidate["checkpoint"]["epoch"])
                for rule in RULES:
                    for dataset in DATASETS:
                        source = root / f"candidate_epoch{epoch:03d}" / rule / dataset
                        destination = out / root.name / f"candidate_epoch{epoch:03d}" / rule / dataset
                        results.append(run_dataset(source, destination, dataset, rule))
        payload = {
            "format": "locatemot-l88-internal-dev-trackeval-matrix-v1", "status": "complete",
            "evidence_type": "full-video internal fit/dev TrackEval for checkpoint/rule selection",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "inference_roots": [str(path) for path in roots],
            "source_summary_sha256": {str(path): sha256(path / "summary.json") for path in roots},
            "results": results, "manifest_sha256": MANIFEST_SHA,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": True,
            "no_hota_or_trackeval": False, "candidate_deletion": False,
            "candidate_truncation": False, "groundingdino_lora_used": True,
            "groundingdino_backbone_trainable": False,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "wall_seconds": time.perf_counter() - started,
            "failure_root_cause": None, "next_action": "select checkpoint/rule before fixed semantic validation",
        }
        write_json(out / "trackeval_matrix.json", payload)
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", {"format": payload["format"], "status": "complete",
                                           "result_count": len(results), "manifest_sha256": MANIFEST_SHA,
                                           "screening_gt_used": False, "official_test_labels_read": False,
                                           "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": True,
                                           "no_hota_or_trackeval": False})
        return 0
    except Exception:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L88 internal dev TrackEval matrix — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {"format": "locatemot-l88-internal-dev-trackeval-matrix-v1",
                                          "status": "incomplete", "command": command, "cwd": str(WORK_ROOT),
                                          "luna_thread": THREAD, "failure_root_cause": "first traceback in INCOMPLETE.md",
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
