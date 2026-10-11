#!/usr/bin/env python3
"""Targeted causal/export/official-metric checks, not benchmark evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import torch
    from locatemot.unified.runtime.u2_simple_tracker import U2SimpleTracker
    from locatemot.unified.runtime.u2_ordinary_evaluation import prepare_gt_view, write_predictions, evaluate_ordinary
    from tools.unified.run_u2_simple_baseline import register, DATASETS
    from tools.unified.build_u2_candidate_cache import digest
    from tools.unified.select_u2_perception_adaptation import selection_payload
    from tools.unified.u1_formal_generic_gate import VOCABULARY

    torch.set_num_threads(2)
    config = register()
    candidates = {"boxes": torch.tensor([[0., 0., 20., 50.], [0., 0., 20., 50.]]),
                  "scores": torch.tensor([.9, .8]), "id_visual": torch.eye(2)}
    tracker = U2SimpleTracker()
    assert tracker.step(candidates)["track_ids"].tolist() == [0, 1]
    empty = {"boxes": torch.empty(0, 4), "scores": torch.empty(0), "id_visual": torch.empty(0, 2)}
    assert tracker.step(empty)["track_ids"].numel() == 0, "unobserved motion box was emitted"
    saved = tracker.state_dict()
    reordered = {name: value.flip(0) for name, value in candidates.items()}
    continuous = tracker.step(reordered)
    resumed = U2SimpleTracker()
    resumed.load_state_dict(saved)
    resumed_result = resumed.step(reordered)
    assert continuous["track_ids"].tolist() == [1, 0], "appearance failed to resolve coincident boxes"
    assert torch.equal(continuous["track_ids"], resumed_result["track_ids"]), "save/reload changed identities"
    # F0/incomplete portability remains ineligible after sharing the selection
    # computation with the cache builder; no GPU inference or optimizer starts.
    try:
        selection_payload(argparse.Namespace(
            foundation_sha256="4efc35ce70be5cddf2c4d9a0f91bc4edef3e9ec5277a819211ba4ac42e0d7d26", epoch=1,
            portability=ROOT / "outputs/unified/u2/perception/portability.json"))
    except ValueError:
        rejected = True
    else:
        rejected = False
    assert rejected, "Incomplete or failed F0 portability authorized cache construction"

    results = {}
    with tempfile.TemporaryDirectory(prefix="locatemot_u2_simple_contract_") as raw:
        base = Path(raw)
        for dataset in DATASETS:
            folder = base / dataset
            folder.mkdir()
            source = folder / "original_source.txt"
            video = "0000" if dataset == "kitti_legal_development" else "synthetic_sequence"
            frames = [f"{index:06d}.png" for index in range(3)] if dataset == "kitti_legal_development" else [f"{index:06d}.jpg" for index in range(6, 9)]
            group = {"key": f"{dataset}/dev/{video}", "dataset": dataset, "video": video, "frames": frames,
                     "vocabulary": VOCABULARY if dataset == "kitti_legal_development" else "person ."}
            predictions = []
            if dataset == "bdd100k":
                classes = ("pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
                group["vocabulary"] = " . ".join(classes) + " ."
                source_frames = []
                for index, frame in enumerate(frames):
                    labels, boxes = [], []
                    for category, name in enumerate(classes):
                        box = [category * 40., 0., category * 40. + 20., 50.]
                        boxes.append(box)
                        raw_name = {"pedestrian": "person", "bicycle": "bike", "motorcycle": "motor"}.get(name, name)
                        labels.append({"id": str(100 + category), "category": raw_name,
                                       "box2d": {"x1": box[0], "y1": box[1], "x2": box[2] - 1., "y2": box[3] - 1.},
                                       "attributes": {"crowd": False, "ignored": False}})
                    labels.append({"id": "500", "category": "car", "box2d": {"x1": 400., "y1": 0., "x2": 419., "y2": 49.},
                                   "attributes": {"ignored": True}})
                    source_frames.append({"name": frame, "index": index, "labels": labels})
                    predictions.append({"frame": frame, "track_ids": list(range(8)) + [500],
                                        "boxes": boxes + [[400., 0., 420., 50.]], "scores": [.9] * 9,
                                        "predicted_class": list(range(8)) + [2]})
                source.write_text(json.dumps(source_frames))
            elif dataset == "kitti_legal_development":
                lines = []
                for index, frame in enumerate(frames):
                    for identity, name, box in ((100, "Car", [0., 0., 20., 50.]),
                                                (101, "Pedestrian", [40., 0., 60., 50.]),
                                                (-1, "DontCare", [100., 0., 120., 50.])):
                        lines.append(" ".join(map(str, [index, identity, name, 0, 0, -10, *box,
                                                        -1, -1, -1, -1000, -1000, -1000, -10])))
                    predictions.append({"frame": frame, "track_ids": [0, 1, 500],
                                        "boxes": [[0., 0., 20., 50.], [40., 0., 60., 50.], [100., 0., 120., 50.]],
                                        "scores": [.9] * 3, "predicted_class": [0, 5, 0]})
                source.write_text("\n".join(lines) + "\n")
            else:
                lines = []
                for frame in frames:
                    index = int(Path(frame).stem)
                    for identity, x, category in ((100, 0, 1), (101, 40, 1), (500, 100, 8)):
                        lines.append(f"{index},{identity},{x},0,20,50,1,{category},1")
                    predictions.append({"frame": frame, "track_ids": [0, 1, 500],
                                        "boxes": [[0., 0., 20., 50.], [40., 0., 60., 50.], [100., 0., 120., 50.]],
                                        "scores": [.9] * 3, "predicted_class": [0, 0, 0]})
                source.write_text("\n".join(lines) + "\n")
            gt = folder / "gt"
            supervision = {group["key"]: {"source_path": str(source), "sha256": digest(source)}}
            view = prepare_gt_view(dataset, [group], supervision, gt)
            extension = ".json" if dataset == "bdd100k" else ".txt"
            prediction_root = folder / "predictions"
            write_predictions(dataset, group, predictions, prediction_root / "perfect/data" / (video + extension))
            perfect = evaluate_ordinary(dataset, gt, prediction_root, view["sequence_lengths"], "perfect")
            assert all(abs(perfect["metrics"][key] - 1.) < 1e-12 for key in ("HOTA", "DetA", "AssA", "IDF1")), (dataset, perfect)
            assert perfect["metrics"]["IDSW"] == 0
            assert perfect["gt_detections"] == (24 if dataset == "bdd100k" else 6)
            assert len(perfect["hota_alpha_thresholds"]) == 19
            results[dataset] = {"perfect": perfect["metrics"], "gt_detections": perfect["gt_detections"],
                                "frames_evaluated": perfect["frames_evaluated"], "official_ignore_preprocessing": "PASS"}
            if dataset == "mot17":
                changed = [{**frame, "track_ids": frame["track_ids"][:]} for frame in predictions]
                changed[-1]["track_ids"][:2] = [1, 0]
                write_predictions(dataset, group, changed, prediction_root / "switch/data" / (video + extension))
                switched = evaluate_ordinary(dataset, gt, prediction_root, view["sequence_lengths"], "switch")
                assert switched["metrics"]["DetA"] > switched["metrics"]["AssA"] and switched["metrics"]["IDSW"] == 2
                results[dataset]["identity_switch"] = switched["metrics"]
    report = {"format": "locatemot-u2-simple-pipeline-contract-v1", "status": "PASS",
              "simple_causal_identity_reordering": "PASS", "occlusion_no_ghost_output": "PASS",
              "online_save_reload": "PASS", "incomplete_f0_cache_gate": "REJECTED_AS_REQUIRED",
              "official_file_reader_and_metrics": results, "registration": config,
              "synthetic_only": True, "formal_cache_started": False, "benchmark_metrics_measured": False,
              "official_test_labels_read": False}
    path = ROOT / "outputs/unified/u2/eval/simple_pipeline_contract.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
