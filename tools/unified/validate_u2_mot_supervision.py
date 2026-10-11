#!/usr/bin/env python3
"""Compare actual original train-video supervision with upstream TrackEval."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from locatemot.unified.data.u2_mot_clips import DATASET_PROBABILITIES, CachedMOTClips, OriginalMOTSupervision
from locatemot.unified.losses.u2_association import FrameSupervision, candidate_gt_assignment
from locatemot.unified.runtime.u2_ordinary_evaluation import prepare_gt_view, write_predictions, verify_trackeval_source, trackeval
from tools.unified.build_u2_candidate_cache import DIRECTORY, PLAN


def sorted_boxes(boxes: list) -> np.ndarray:
    return np.array(sorted(tuple(box) for box in boxes), dtype=float).reshape(-1, 4)


def main() -> int:
    torch.set_num_threads(2)
    source = verify_trackeval_source()
    groups = [json.loads(line) for line in PLAN.read_text().splitlines()]
    supervision = json.loads((DIRECTORY / "supervision_source_inventory.json").read_text())["ordinary"]
    results = {}
    with tempfile.TemporaryDirectory(prefix="locatemot_u2_train_gt_") as temporary:
        base = Path(temporary)
        for dataset in DATASET_PROBABILITIES:
            group = next(group for group in groups if group["dataset"] == dataset and group["role"] == "train")
            labels = OriginalMOTSupervision(group, supervision[group["key"]])
            directory, predictions = base / dataset / "gt", base / dataset / "predictions"
            view = prepare_gt_view(dataset, [group], supervision, directory)
            empty = [{"frame": frame, "track_ids": [], "boxes": [], "scores": [], "predicted_class": []}
                     for frame in group["frames"]]
            extension = ".json" if dataset == "bdd100k" else ".txt"
            write_predictions(dataset, group, empty, predictions / "empty/data" / f"{group['video']}{extension}")
            config = {"GT_FOLDER": str(directory), "TRACKERS_FOLDER": str(predictions),
                      "TRACKERS_TO_EVAL": ["empty"], "PRINT_CONFIG": False, "TRACKER_SUB_FOLDER": "data"}
            if dataset == "bdd100k":
                adapter = trackeval.datasets.BDD100K(config)
            elif dataset == "kitti_legal_development":
                adapter = trackeval.datasets.Kitti2DBox(config)
            else:
                config.update(BENCHMARK="MOT20" if dataset == "mot20" else "MOT17", SKIP_SPLIT_FOL=True,
                              SEQ_INFO=view["sequence_lengths"], DO_PREPROC=True)
                adapter = trackeval.datasets.MotChallenge2DBox(config)
            raw = adapter.get_raw_seq_data("empty", group["video"])
            processed = [adapter.get_preprocessed_seq_data(raw, category) for category in adapter.class_list]
            official_units = sum(data["num_gt_dets"] for data in processed)
            assert official_units == labels.positive_units, (dataset, official_units, labels.positive_units)
            for position in range(len(group["frames"])):
                actual = sorted_boxes(labels[position].boxes.tolist())
                official_boxes = [box.tolist() for data in processed for box in data["gt_dets"][position]]
                if dataset in {"dancetrack", "mot17", "mot20"}:
                    # The public MOT reader retains XYWH in preprocessed
                    # arrays; BDD and KITTI arrays already contain XYXY.
                    official_boxes = [[x, y, x + width, y + height] for x, y, width, height in official_boxes]
                official = sorted_boxes(official_boxes)
                assert actual.shape == official.shape and np.allclose(actual, official, rtol=1e-6, atol=1e-4), (dataset, position)
            results[dataset] = {"video": group["video"], "role": "train", "frames": len(group["frames"]),
                                "positive_units": labels.positive_units, "all_frame_gt_boxes_match_upstream": True,
                                "source_sha256": supervision[group["key"]]["sha256"],
                                "ignored_regions": sum(len(frame.ignore_boxes) for frame in labels.frames.values()),
                                "distractor_boxes": sum(len(frame.distractor_boxes) for frame in labels.frames.values())}

    # Exact public preprocessing distinctions: distractors require an IoU
    # match; crowd/DontCare require >.5 candidate IoA; KITTI small UNMATCHED
    # proposals are ignored, whereas visible positive boxes stay supervised.
    boxes = torch.tensor([[0., 0., 20., 20.], [40., 0., 60., 20.], [80., 0., 100., 20.], [120., 0., 140., 40.]])
    gt = FrameSupervision(boxes[:1], torch.tensor([10]), boxes[3:], {10}, boxes[1:2], 25.)
    identities, eligible = candidate_gt_assignment(boxes, gt)
    assert identities.tolist() == [10, -1, -1, -1] and eligible.tolist() == [True, False, False, False]
    # Dataset choices are deterministic per step and independent of rank.
    sampler = CachedMOTClips.__new__(CachedMOTClips)
    sampler.seed = 20261012
    counts = Counter(sampler.dataset_for_step(step) for step in range(6000))
    assert set(counts) == set(DATASET_PROBABILITIES)
    report = {"format": "locatemot-u2-original-train-supervision-contract-v1", "status": "PASS",
              "scope": "one complete actual registered train video per dataset; not all train videos or cached features",
              "datasets": results, "trackeval_commit": source["commit"],
              "distractor_iou_and_region_ioa_distinguished": True, "small_unmatched_kitti_rule": True,
              "registered_sampling_probabilities": DATASET_PROBABILITIES, "seed": sampler.seed,
              "dataset_draw_counts_6000_steps": dict(counts), "gt_features_used_by_core": False,
              "formal_training_launched": False, "official_test_labels_read": False}
    path = ROOT / "outputs/unified/u2/eval/original_train_supervision_contract.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
