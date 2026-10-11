#!/usr/bin/env python3
"""Evaluate the frozen simple tracker on all five complete ordinary dev sets."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.unified.build_u2_candidate_cache import DIRECTORY, PLAN, digest, write_json
from tools.unified.select_u2_perception_adaptation import verify_selected_foundation
from locatemot.unified.runtime.u2_simple_tracker import SimpleTrackerConfig, U2SimpleTracker

REGISTRATION = ROOT / "outputs/unified/u2/protocol/simple_baseline.json"
DATASETS = ("bdd100k", "dancetrack", "mot17", "mot20", "kitti_legal_development")


def register() -> dict:
    from locatemot.unified.runtime.u2_ordinary_evaluation import verify_trackeval_source
    source = verify_trackeval_source()
    payload = {"format": "locatemot-u2-simple-baseline-registration-v1", "name": "U2_SIMPLE_BASELINE",
               "config": asdict(SimpleTrackerConfig()), "learned_parameters": 0,
               "source_sha256": digest(ROOT / "locatemot/unified/runtime/u2_simple_tracker.py"),
               "trackeval_commit": source["commit"], "datasets": list(DATASETS),
               "candidate_policy": "all frozen candidates enter association; low-score observations may recover tracks; fixed .10 birth score",
               "selection_or_parameter_sweep": False, "before_benchmark_metrics": True,
               "hota": "mean over official 19 alpha thresholds .05..95; sequence combination then class average",
               "metric_scale": "fraction, except integer IDSW", "official_test_labels_read": False}
    if REGISTRATION.exists() and json.loads(REGISTRATION.read_text()) != payload:
        raise ValueError("The fixed simple baseline registration differs; do not tune after metrics")
    write_json(REGISTRATION, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/unified/u2/simple_baseline")
    args = parser.parse_args()
    registration = register()
    if args.register_only:
        print(json.dumps(registration, indent=2))
        return 0
    if args.cache_manifest is None:
        parser.error("Formal simple tracking requires the complete selected-foundation cache manifest")
    import torch
    from locatemot.unified.runtime.u2_ordinary_evaluation import prepare_gt_view, write_predictions, evaluate_ordinary

    torch.set_num_threads(2)
    cache = json.loads(args.cache_manifest.read_text())
    spec = cache["spec"]
    if cache["status"] != "COMPLETE" or spec["phase"] != "ordinary":
        raise ValueError("Simple baseline requires the complete ordinary candidate cache")
    verify_selected_foundation(Path(spec["selection"]), spec["foundation_sha256"])
    if digest(Path(spec["checkpoint"])) != spec["foundation_sha256"] or digest(PLAN) != spec["inventory_sha256"]:
        raise ValueError("Simple baseline cache weights/frame scope differ from selection")
    inventory = [json.loads(line) for line in PLAN.read_text().splitlines()]
    dev = [group for group in inventory if group["dataset"] in DATASETS and group["role"] == "dev"]
    cached = {result["group"]["key"]: result for result in cache["groups"] if result["group"]["role"] == "dev"}
    if set(cached) != {group["key"] for group in dev}:
        raise ValueError("Simple baseline cache omits a registered development video")
    supervision = json.loads((DIRECTORY / "supervision_source_inventory.json").read_text())["ordinary"]
    reports = {}
    for dataset in DATASETS:
        groups = [group for group in dev if group["dataset"] == dataset]
        prediction_root = args.out / dataset / "predictions"
        gt_root = args.out / dataset / "gt_view"
        view = prepare_gt_view(dataset, groups, supervision, gt_root)
        write_json(args.out / dataset / "gt_view_provenance.json", view)
        for group in groups:
            tracker, frames = U2SimpleTracker(), []
            for unit in cached[group["key"]]["units"]:
                shard = torch.load(unit["path"], map_location="cpu", weights_only=False)
                if (shard["header"]["foundation_sha256"] != spec["foundation_sha256"]
                        or shard["header"]["cache_spec_sha256"] != cache["cache_spec_sha256"]
                        or shard["header"]["group_key"] != group["key"]
                        or shard["header"]["frames"] != unit["frames"]):
                    raise ValueError("Simple baseline candidate shard has different provenance")
                for item in shard["frames"]:
                    candidates = item["candidates"]
                    frame_index = int(Path(item["frame"]).stem) if dataset == "kitti_legal_development" else len(frames)
                    output = tracker.step(candidates, frame_index=frame_index)
                    indices = output["candidate_indices"]
                    frames.append({"frame": item["frame"], "track_ids": output["track_ids"].tolist(),
                                   "boxes": output["boxes"].tolist(), "scores": output["scores"].tolist(),
                                   "predicted_class": candidates["predicted_class"][indices].tolist()})
            extension = ".json" if dataset == "bdd100k" else ".txt"
            write_predictions(dataset, group, frames,
                              prediction_root / "U2_SIMPLE_BASELINE/data" / f"{group['video']}{extension}")
            print(json.dumps({"dataset": dataset, "video_complete": group["video"], "frames": len(frames)}), flush=True)
        report = evaluate_ordinary(dataset, gt_root, prediction_root, view["sequence_lengths"], "U2_SIMPLE_BASELINE")
        positive_units = [source["positive_gt_units_before_official_preprocessing"] for source in view["sources"]]
        if dataset != "kitti_legal_development" and report["gt_detections"] != sum(positive_units):
            raise ValueError("Official evaluation dropped or duplicated registered positive GT units")
        reports[dataset] = report
        write_json(args.out / dataset / "result.json", report)
    if set(reports) != set(DATASETS):
        raise ValueError("Simple baseline did not complete all five registered ordinary datasets")
    macro = {key: sum(report["metrics"][key] for report in reports.values()) / len(DATASETS)
             for key in ("HOTA", "DetA", "AssA", "IDF1")}
    result = {"format": "locatemot-u2-simple-baseline-v1", "status": "COMPLETE", "name": "U2_SIMPLE_BASELINE",
              "foundation_sha256": spec["foundation_sha256"], "candidate_cache_manifest": str(args.cache_manifest.resolve()),
              "candidate_cache_sha256": digest(args.cache_manifest), "registration_sha256": digest(REGISTRATION),
              "datasets": reports, "macro": macro, "total_idsw": sum(report["metrics"]["IDSW"] for report in reports.values()),
              "gt_tracks": sum(report["gt_tracks"] for report in reports.values()),
              "learned_core_training_launched": False, "official_test_labels_read": False}
    write_json(args.out / "result.json", result)
    print(json.dumps({"status": "COMPLETE", "macro": macro, "result": str(args.out / "result.json")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
