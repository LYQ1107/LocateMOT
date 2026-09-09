#!/usr/bin/env python3
"""Run the local TrackEval checkout on frozen R0 dev/internal predictions."""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

WORK_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
TRACK_EVAL = ASSET_ROOT.parent / "TrackEval-master"
MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
METRIC_KEYS = ("HOTA___AUC", "DetA___AUC", "AssA___AUC", "LocA___AUC",
               "DetRe___AUC", "DetPr___AUC", "AssRe___AUC", "AssPr___AUC",
               "IDF1", "IDR", "IDP", "MOTA", "MOTP", "IDSW", "CLR_FP", "CLR_FN")
PERCENT = {"HOTA___AUC", "DetA___AUC", "AssA___AUC", "LocA___AUC", "DetRe___AUC", "DetPr___AUC",
           "AssRe___AUC", "AssPr___AUC", "IDF1", "IDR", "IDP", "MOTA", "MOTP"}


def sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n", encoding="utf-8")


def parse_combined(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row = next((value for value in rows if value.get("seq") == "COMBINED"), None)
    if row is None:
        raise AssertionError(f"TrackEval COMBINED row missing: {path}")
    result: dict[str, float] = {}
    for key in METRIC_KEYS:
        if row.get(key) in (None, ""):
            continue
        value = float(row[key])
        if not (value == value) or abs(value) == float("inf"):
            raise ValueError(f"nonfinite TrackEval metric {key}: {path}")
        result[key] = value
    if "HOTA___AUC" not in result:
        raise AssertionError(f"HOTA missing: {path}")
    return result


def run_dataset(source: Path, destination: Path, dataset: str, tracker_name: str = "r0") -> dict[str, Any]:
    gt_folder = source / "gt"; tracker_folder = source / "trackers"; seqmap = source / "seqmap.txt"
    if not gt_folder.is_dir() or not tracker_folder.is_dir() or not seqmap.is_file():
        raise FileNotFoundError(f"incomplete R0 TrackEval source: {source}")
    sequences = [line.strip() for line in seqmap.read_text(encoding="utf-8").splitlines()[1:] if line.strip()]
    tracker_files = sorted((tracker_folder / tracker_name / "data").glob("*.txt"))
    if not sequences or len(tracker_files) != len(sequences):
        raise AssertionError(f"R0 sequence/tracker mismatch: {source} {len(sequences)}/{len(tracker_files)}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"refusing nonempty R0 TrackEval destination: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    if str(TRACK_EVAL) not in sys.path:
        sys.path.insert(0, str(TRACK_EVAL))
    import trackeval  # pylint: disable=import-outside-toplevel
    evaluator_config = trackeval.Evaluator.get_default_eval_config()
    evaluator_config.update({"USE_PARALLEL": False, "NUM_PARALLEL_CORES": 1, "BREAK_ON_ERROR": True,
                             "RETURN_ON_ERROR": False, "PRINT_RESULTS": False, "PRINT_ONLY_COMBINED": False,
                             "PRINT_CONFIG": True, "TIME_PROGRESS": True, "DISPLAY_LESS_PROGRESS": True,
                             "OUTPUT_SUMMARY": True, "OUTPUT_EMPTY_CLASSES": True, "OUTPUT_DETAILED": True,
                             "PLOT_CURVES": False, "LOG_ON_ERROR": str((destination / "trackeval_error.log").resolve())})
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
    return {"dataset": dataset, "sequence_count": len(sequences), "tracker_file_count": len(tracker_files),
            "sequences": sequences, "source": str(source.resolve()), "detailed_csv": str(detailed.resolve()),
            "trackeval_root": str(TRACK_EVAL), "trackeval_git_head": None,
            "trackeval_git_head_note": "local checkout has no verifiable HEAD", "metrics_raw": raw,
            "metrics_percent": {key: value * 100.0 for key, value in raw.items() if key in PERCENT},
            "metrics_counts": {key: value for key, value in raw.items() if key not in PERCENT},
            "elapsed_seconds": time.perf_counter() - started}


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R0 TrackEval output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv]); started = time.perf_counter()
    base = {"format": "locatemot-r0-trackeval-matrix-v1", "status": "incomplete", "command": command,
            "cwd": str(Path.cwd().resolve()), "luna_thread": THREAD, "scope": args.scope,
            "inference_root": str(args.inference_root.resolve()), "manifest_sha256": sha256_file(MANIFEST),
            "expected_manifest_sha256": MANIFEST_SHA, "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": True, "no_hota_or_trackeval": False, "training_run": False,
            "candidate_deletion": False, "candidate_truncation": False, "failure_root_cause": None,
            "next_action": "select one checkpoint on the legal dev TrackEval tuple"}
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R0 TrackEval cwd: {Path.cwd()}")
        if base["manifest_sha256"] != MANIFEST_SHA or not TRACK_EVAL.is_dir():
            raise AssertionError("R0 TrackEval manifest or checkout contract failed")
        source = args.inference_root.resolve()
        summary = json.loads((source / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "complete" or not summary.get("full_video"):
            raise AssertionError("R0 inference source is not complete")
        if summary.get("scope_key") != args.scope or summary.get("screening_gt_used") or summary.get("official_test_labels_read"):
            raise AssertionError("R0 TrackEval scope/label contract failed")
        candidates = summary.get("candidates")
        if not isinstance(candidates, list) or not candidates or len(candidates) > 3:
            raise AssertionError("R0 TrackEval candidate count invalid")
        results: list[dict[str, Any]] = []
        for candidate in candidates:
            root = Path(str(candidate["root"])).resolve()
            if root.parent.parent != source and root.parent.parent.parent != source:
                # The exact nesting is checked by the dataset source below;
                # this guard prevents an arbitrary external TrackEval source.
                if source not in root.parents:
                    raise AssertionError(f"R0 TrackEval source escapes inference root: {root}")
            source_dataset = root / args.dataset
            destination = out / root.parent.name / root.name / args.dataset
            metrics = run_dataset(source_dataset, destination, args.dataset)
            results.append({"role": candidate.get("role"), "checkpoint_info": candidate["checkpoint_info"],
                            "rule": candidate.get("rule"), "dev_metrics": candidate.get("dev_metrics"),
                            "source": str(source_dataset), "trackeval": metrics,
                            "hota": metrics["metrics_raw"].get("HOTA___AUC", -1.0),
                            "deta": metrics["metrics_raw"].get("DetA___AUC", -1.0),
                            "assa": metrics["metrics_raw"].get("AssA___AUC", -1.0)})
        payload = {**base, "status": "complete", "results": results,
                   "candidate_count": len(results), "wall_seconds": time.perf_counter() - started,
                   "next_action": "select a frozen checkpoint per benchmark using the registered dev tuple"}
        write_json(out / "trackeval_matrix.json", payload); write_json(out / "provenance.json", payload)
        write_json(out / "status.json", payload)
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# R0 TrackEval matrix — INCOMPLETE\n\n" + trace, encoding="utf-8")
        payload = {**base, "failure_root_cause": f"{type(exc).__name__}: {exc}",
                   "traceback_path": str((out / "INCOMPLETE.md").resolve()),
                   "hota_trackeval_run": False, "wall_seconds": time.perf_counter() - started}
        write_json(out / "provenance.json", payload); write_json(out / "status.json", payload)
        return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--scope", choices=("dev", "internal"), required=True)
    parser.add_argument("--dataset", choices=("refer_kitti_v1", "refer_kitti_v2"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
