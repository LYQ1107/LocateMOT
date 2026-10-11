#!/usr/bin/env python3
"""Joint gate/selection/sampling fixtures; no formal benchmark optimization."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from locatemot.unified.data.u2_mot_clips import DATASET_PROBABILITIES
from locatemot.unified.data.u2_ovmot_clips import CachedOVMOTClips, joint_task_for_step
from tools.unified.build_u2_candidate_cache import digest, write_json
from tools.unified.eval_u2_joint import checkpoint_decision, select_joint
from tools.unified.train_u2_joint import EVALUATION_STEPS, initialize_from_mot, optimizer_for_core, recipe


def fixtures() -> tuple[dict, dict, dict]:
    datasets = {dataset: {"status": "COMPLETE", "frames_evaluated": 32, "sequences_evaluated": 1,
                          "gt_detections": 64, "gt_tracks": 2, "metrics": {"HOTA": .6, "AssA": .5}}
                for dataset in DATASET_PROBABILITIES}
    ordinary = {"status": "COMPLETE", "datasets": datasets, "candidate_cache_sha256": "fixture",
                "foundation_sha256": "fixture", "macro": {"AssA": .5}}
    ovmot = {"status": "COMPLETE", "metrics": {"overall": {"TETA": 34., "AssocA": 37.},
                                               "base": {"TETA": 40., "AssocA": 44.},
                                               "novel": {"TETA": 30., "AssocA": 32.}}}
    return ordinary, ovmot, copy.deepcopy(ordinary)


def main() -> int:
    torch.set_num_threads(2)
    registered = recipe()
    assert registered["maximum_optimizer_steps"] == 4000 and registered["frames_per_clip"] == 8
    ordinary, ovmot, reference = fixtures()
    assert checkpoint_decision(ordinary, ovmot, reference)["eligible"]
    boundary = copy.deepcopy(ordinary)
    boundary["datasets"]["mot20"]["metrics"] = {"HOTA": .59, "AssA": .49}
    assert checkpoint_decision(boundary, ovmot, reference)["eligible"], "Exactly one point must remain eligible"
    for metric in ("HOTA", "AssA"):
        failed = copy.deepcopy(ordinary)
        failed["datasets"]["mot20"]["metrics"][metric] -= .011
        decision = checkpoint_decision(failed, ovmot, reference)
        assert not decision["eligible"] and decision["failed_ordinary_datasets"] == ["mot20"]
    for metric, threshold in (("TETA", 34.), ("AssocA", 37.)):
        failed = copy.deepcopy(ovmot)
        failed["metrics"]["overall"][metric] = threshold - .01
        assert not checkpoint_decision(ordinary, failed, reference)["eligible"]
    near = copy.deepcopy(ovmot)
    near["metrics"]["overall"] = {"TETA": 31., "AssocA": 34.}
    assert checkpoint_decision(ordinary, near, reference)["bounded_u2_c_eligible"]
    near["metrics"]["overall"]["AssocA"] = 33.99
    assert not checkpoint_decision(ordinary, near, reference)["bounded_u2_c_eligible"]
    with tempfile.TemporaryDirectory(prefix="locatemot_u2_joint_contract_") as raw:
        directory = Path(raw)
        metadata = {key: "fixture" for key in ("foundation_sha256", "ordinary_cache_sha256", "ovmot_cache_sha256",
                                               "class_text_sha256", "mot_only_checkpoint_sha256", "tao_gt_view_sha256")}
        reference["checkpoint_sha256"] = "fixture"
        for step in EVALUATION_STEPS:
            path = directory / f"checkpoint_{step}.pth"
            path.write_bytes(f"explicitly-synthetic-joint-{step}".encode())
            report = {"status": "COMPLETE", "step": step, "checkpoint": str(path), **metadata,
                      "checkpoint_sha256": digest(path), "ordinary": ordinary, "ovmot": ovmot,
                      "decision": checkpoint_decision(ordinary, ovmot, reference)}
            write_json(directory / f"evaluation_{step:04d}/result.json", report)
        selected = select_joint(directory, reference, metadata)
        assert selected["selected"]["step"] == 1000 and not selected["ready_for_u3"]
        for step in EVALUATION_STEPS:
            path = directory / f"evaluation_{step:04d}/result.json"
            report = json.loads(path.read_text())
            report["ovmot"]["metrics"]["overall"] = {"TETA": 30., "AssocA": 33.}
            report["decision"] = checkpoint_decision(report["ordinary"], report["ovmot"], reference)
            write_json(path, report)
        assert select_joint(directory, reference, metadata)["status"] == "LOCATEMOT_U_BLOCKED_U2_OVMOT"

        # Read real-shaped label-free shard files and separate base labels.
        # Train/dev fixtures prove sampler coverage and chronological spacing.
        groups, supervision = [], []
        for role in ("train", "dev"):
            video = f"{'train' if role == 'train' else 'val'}/fixture"
            frames = [f"frame{index + 1:04d}.jpg" for index in range(32)]
            group = {"key": f"ovmot/{role}/{video}", "dataset": "ovmot", "role": role, "video": video, "frames": frames}
            candidates = {"boxes": torch.tensor([[0., 0., 20., 40.]]), "scores": torch.tensor([.9]),
                          "id_visual": torch.randn(1, 256), "semantic_visual": torch.randn(1, 256),
                          "decoder_query": torch.randn(1, 256), "predicted_class": torch.tensor([0])}
            path = directory / (role + "_features.pt")
            torch.save({"header": {"cache_spec_sha256": "fixture", "foundation_sha256": "fixture",
                                    "group_key": group["key"], "frames": frames},
                        "frames": [{"frame": frame, "candidates": candidates} for frame in frames]}, path)
            groups.append({"group": group, "units": [{"path": str(path), "frames": frames}]})
            if role == "train":
                gt_path = directory / "separate_base_labels.json"
                write_json(gt_path, {"group_key": group["key"], "novel_class_or_identity_supervision": False,
                                    "negative_category_ids": [1], "tracks": [{"identity_id": 10, "category_id": 0}],
                                    "frames": [{"frame": frame, "source_frame_index": 100 + index * 2,
                                                "boxes": [[0., 0., 20., 40.]], "identity_ids": [10], "category_ids": [0],
                                                "ignore_boxes": []} for index, frame in enumerate(frames)]})
                supervision.append({"key": group["key"], "path": str(gt_path), "sha256": digest(gt_path)})
        cache = {"status": "COMPLETE", "cache_spec_sha256": "fixture", "groups": groups,
                 "spec": {"phase": "ovmot", "foundation_sha256": "fixture", "inventory_sha256": "fixture"}}
        labels = {"status": "COMPLETE", "novel_training_gt_units": 0, "inventory_sha256": "fixture", "groups": supervision}
        sampler = CachedOVMOTClips(cache, labels)
        for step in (0, 999, 2000, 3999):
            for rank in (0, 7):
                clips = sampler.sample(step, rank, "cpu")
                again = sampler.sample(step, rank, "cpu")
                for clip, repeated in zip(clips, again):
                    assert len(clip) == 8 and [frame["frame"] for frame in clip] == [frame["frame"] for frame in repeated]
                    assert all(frame["video"].startswith("train/") for frame in clip)
                    assert all(left["frame_index"] < right["frame_index"] for left, right in zip(clip, clip[1:]))
                    assert all(frame["supervision"].category_ids.tolist() == [0] for frame in clip)
        assert len(sampler.load_candidates(groups[1]["group"], 0, "cpu")["boxes"]) == 1
    from locatemot.unified.models.u2.shared_track_core import SharedTrackCore
    torch.manual_seed(20261012)
    original = SharedTrackCore()
    expected = copy.deepcopy(original.state_dict())
    torch.manual_seed(991)
    initialized = SharedTrackCore()
    initialize_from_mot(initialized, {"format": "locatemot-u2-mot-only-checkpoint-v1", "core": expected,
                                    "metadata": {"foundation_sha256": "fixture"}}, "fixture")
    assert all(torch.equal(value, initialized.state_dict()[name]) for name, value in expected.items())
    optimizer = optimizer_for_core(initialized)
    assert [group["lr"] for group in optimizer.param_groups] == [5e-5, 1e-4]
    assert len({id(p) for group in optimizer.param_groups for p in group["params"]}) == len(list(initialized.parameters()))
    from collections import Counter
    tasks = Counter(joint_task_for_step(step) for step in range(10000))
    report = {"format": "locatemot-u2-joint-run-contract-v1", "status": "PASS", "synthetic_only": True,
              "strict_all_tensor_mot_initialization": True, "same_core_optimizer_groups": [5e-5, 1e-4],
              "eight_frame_train_only_separate_base_supervision": True, "deterministic_rank_step_sampling": True,
              "task_draws_10000": dict(tasks), "exact_one_point_boundary_accepted": True,
              "individual_hota_and_assa_regression_rejected": True, "both_teta_and_assoc_gate_required": True,
              "bounded_refinement_thresholds_checked": True, "four_boundary_selection_and_earlier_tie": True,
              "joint_selection_not_u3_readiness": True, "formal_joint_training_launched": False,
              "official_test_labels_read": False}
    write_json(ROOT / "outputs/unified/u2/eval/joint_run_contract.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
