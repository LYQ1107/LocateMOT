#!/usr/bin/env python3
"""Eight-GPU whole evaluation entry on explicitly synthetic cached sequences."""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from locatemot.unified.data.u2_mot_clips import CachedMOTClips, DATASET_PROBABILITIES
from locatemot.unified.data.u2_ovmot_clips import CachedOVMOTClips
from locatemot.unified.models.u2.shared_track_core import SharedTrackCore
from tools.unified.build_u2_candidate_cache import digest, write_json
from tools.unified.eval_u2_joint import evaluate_checkpoint


def create_fixtures(directory: Path) -> dict:
    spec = {"foundation_sha256": "synthetic", "inventory_sha256": "synthetic"}
    ordinary_groups, ordinary_labels = [], {}

    def shard(group, class_index: int):
        candidates = {"boxes": torch.tensor([[0., 0., 20., 50.]]), "scores": torch.tensor([.9]),
                      "id_visual": torch.randn(1, 256), "semantic_visual": torch.randn(1, 256),
                      "decoder_query": torch.randn(1, 256), "predicted_class": torch.tensor([class_index])}
        path = directory / f"{group['key']}/features.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"header": {"cache_spec_sha256": "synthetic", "foundation_sha256": "synthetic",
                                "group_key": group["key"], "frames": group["frames"]},
                    "frames": [{"frame": frame, "candidates": candidates} for frame in group["frames"]]}, path)
        return {"group": group, "units": [{"path": str(path), "frames": group["frames"]}]}

    for dataset in DATASET_PROBABILITIES:
        for role in ("train", "dev"):
            video = ("0000" if role == "train" else "0007") if dataset == "kitti_legal_development" else f"fixture_{role}"
            frames = [f"{index:06d}.png" for index in range(3)] if dataset == "kitti_legal_development" else [f"{index:06d}.jpg" for index in range(6, 9)]
            group = {"key": f"{dataset}/{role}/{video}", "dataset": dataset, "role": role, "video": video,
                     "frames": frames, "vocabulary": ("car ." if dataset == "kitti_legal_development" else
                                                        "pedestrian ." if dataset == "bdd100k" else "person .")}
            ordinary_groups.append(shard(group, 0))
            source = directory / f"{group['key']}/original_gt.txt"
            if dataset == "bdd100k":
                data = [{"name": frame, "index": index,
                         "labels": [{"id": "100", "category": "person", "box2d": {"x1": 0., "y1": 0., "x2": 19., "y2": 49.},
                                     "attributes": {"crowd": False, "ignored": False}}]} for index, frame in enumerate(frames)]
                source.write_text(json.dumps(data))
            elif dataset == "kitti_legal_development":
                source.write_text("\n".join(f"{index} 100 Car 0 0 -10 0 0 20 50 -1 -1 -1 -1000 -1000 -1000 -10" for index in range(3)) + "\n")
            else:
                source.write_text("\n".join(f"{int(Path(frame).stem)},100,0,0,20,50,1,1,1" for frame in frames) + "\n")
            ordinary_labels[group["key"]] = {"source_path": str(source), "sha256": digest(source)}
    ordinary = {"status": "COMPLETE", "cache_spec_sha256": "synthetic", "spec": {**spec, "phase": "ordinary"}, "groups": ordinary_groups}
    ordinary_path = directory / "ordinary_cache.json"
    write_json(ordinary_path, ordinary)
    reference = {"status": "COMPLETE", "candidate_cache_sha256": digest(ordinary_path), "foundation_sha256": "synthetic",
                 "checkpoint_sha256": "explicitly-synthetic-reference",
                 "datasets": {dataset: {"status": "COMPLETE", "frames_evaluated": 3, "sequences_evaluated": 1,
                                        "gt_detections": 3, "gt_tracks": 1, "metrics": {"HOTA": 0., "AssA": 0.}}
                              for dataset in DATASET_PROBABILITIES}}
    simple_path = directory / "simple_reference.json"
    write_json(simple_path, reference)
    vocabulary = json.loads((ROOT / "outputs/unified/u2/protocol/lvis_v1_vocabulary.json").read_text())["categories"]
    base = next(category for category in vocabulary if category["frequency"] in {"f", "c"})
    novel = next(category for category in vocabulary if category["frequency"] == "r")
    ov_groups, ov_labels, images, annotations, videos, tracks = [], [], [], [], [], []
    for role, category in (("train", base), ("dev", base), ("dev", novel)):
        video = f"{'train' if role == 'train' else 'val'}/fixture_{category['id']}"
        frames = [f"frame{index + 1:04d}.jpg" for index in range(32 if role == "train" else 3)]
        group = {"key": f"ovmot/{role}/{video}", "dataset": "ovmot", "role": role, "video": video, "frames": frames}
        ov_groups.append(shard(group, category["id"] - 1))
        if role == "train":
            path = directory / "base_train_labels.json"
            write_json(path, {"group_key": group["key"], "novel_class_or_identity_supervision": False,
                              "negative_category_ids": [], "tracks": [{"identity_id": 10, "category_id": category["id"] - 1}],
                              "frames": [{"frame": frame, "source_frame_index": index, "boxes": [[0., 0., 20., 50.]],
                                          "identity_ids": [10], "category_ids": [category["id"] - 1], "ignore_boxes": []}
                                         for index, frame in enumerate(frames)]})
            ov_labels.append({"key": group["key"], "path": str(path), "sha256": digest(path)})
        else:
            video_id = len(videos) + 1
            videos.append({"id": video_id, "name": video, "neg_category_ids": [], "not_exhaustive_category_ids": []})
            tracks.append({"id": video_id, "video_id": video_id, "category_id": category["id"]})
            for index, frame in enumerate(frames):
                image_id = len(images)
                images.append({"id": image_id, "file_name": f"{video}/{frame}", "video_id": video_id, "frame_index": index,
                               "width": 100, "height": 100})
                annotations.append({"id": image_id, "image_id": image_id, "video_id": video_id, "track_id": video_id,
                                    "category_id": category["id"], "bbox": [0., 0., 20., 50.], "area": 1000., "iscrowd": 0})
    ovmot = {"status": "COMPLETE", "cache_spec_sha256": "synthetic", "spec": {**spec, "phase": "ovmot"}, "groups": ov_groups}
    ovmot_path = directory / "ovmot_cache.json"
    write_json(ovmot_path, ovmot)
    labels = {"status": "COMPLETE", "novel_training_gt_units": 0, "inventory_sha256": "synthetic", "groups": ov_labels}
    gt_path = directory / "explicitly_synthetic_gt.json"
    write_json(gt_path, {"images": images, "videos": videos, "tracks": tracks, "annotations": annotations, "categories": vocabulary})
    gt_view_path = directory / "gt_view.json"
    write_json(gt_view_path, {"view_path": str(gt_path), "view_images": 6, "view_videos": 2, "view_gt_units": 6,
                             "synthetic_only": True})
    return {"ordinary": ordinary, "ordinary_labels": ordinary_labels, "ovmot": ovmot, "ov_labels": labels, "reference": reference,
            "args": {"cache_manifest": str(ordinary_path), "simple_baseline": str(simple_path), "ovmot_cache": str(ovmot_path),
                     "tao_gt_view": str(gt_view_path), "class_text": str(ROOT / "outputs/unified/u2/ovmot_semantics/f0_class_text.pt"),
                     "out": str(directory / "joint_eval")}}


def main() -> int:
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(hours=24))
    assert dist.get_world_size() == 8
    torch.set_num_threads(2)
    temporary = tempfile.TemporaryDirectory(prefix="locatemot_u2_joint_eval_") if rank == 0 else None
    data = [create_fixtures(Path(temporary.name)) if rank == 0 else None]
    dist.broadcast_object_list(data, 0)
    fixture = data[0]
    args = argparse.Namespace(**{key: Path(value) for key, value in fixture["args"].items()})
    mot_sampler = CachedMOTClips(fixture["ordinary"], fixture["ordinary_labels"])
    ov_sampler = CachedOVMOTClips(fixture["ovmot"], fixture["ov_labels"])
    device = torch.device("cuda", local_rank)
    torch.manual_seed(20261012)
    core = SharedTrackCore().to(device).train()
    torch.nn.init.constant_(core.association.pair[-1].bias, 3.)
    torch.nn.init.constant_(core.association.no_match[-1].bias, -3.)
    torch.nn.init.constant_(core.association.new[-1].bias, 3.)
    checkpoint = args.out.parent / "synthetic_core.pth"
    if rank == 0:
        torch.save({"core": core.state_dict(), "synthetic_only": True}, checkpoint)
    dist.barrier()
    text = torch.load(args.class_text, map_location="cpu", weights_only=False)["embeddings"].to(device)
    result = evaluate_checkpoint(core, mot_sampler, ov_sampler, fixture["ordinary"], fixture["ovmot"], fixture["ordinary_labels"],
                                 text, fixture["reference"], args, 1000, checkpoint)
    if rank == 0:
        assert result["ovmot"]["frames_evaluated"] == 6 and result["ovmot"]["videos_evaluated"] == 2
        assert set(result["ordinary"]["datasets"]) == set(DATASET_PROBABILITIES)
        assert all(item["frames_evaluated"] == 3 for item in result["ordinary"]["datasets"].values())
        (args.out / "evaluation_1000/result.json").unlink()
    dist.barrier()
    calls = [0]

    def count_forward(module, args, output):
        calls[0] += 1

    hook = core.register_forward_hook(count_forward)
    resumed = evaluate_checkpoint(core, mot_sampler, ov_sampler, fixture["ordinary"], fixture["ovmot"], fixture["ordinary_labels"],
                                  text, fixture["reference"], args, 1000, checkpoint)
    hook.remove()
    assert calls[0] == 0, "Completed causal video units were rerun on partial evaluation resume"
    if rank == 0:
        assert resumed == result
        report = {"format": "locatemot-u2-eight-gpu-joint-evaluation-contract-v1", "status": "PASS", "world_size": 8,
                  "input_source": "explicitly synthetic caches/GT/reference with actual reference-only F0 class text",
                  "ordinary": {name: item["metrics"] for name, item in result["ordinary"]["datasets"].items()},
                  "ordinary_frames_per_dataset": 3, "all_five_official_readers_executed": True,
                  "ovmot": result["ovmot"]["metrics"], "tao_fixture_frames": 6, "tao_fixture_videos": 2,
                  "official_teta_parallel_workers": 8, "base_novel_overall_reported": True,
                  "complete_video_resume_no_core_forward_calls": True, "resumed_metrics_identical": True,
                  "same_core_both_tasks": True, "benchmark_metrics_measured": False,
                  "formal_joint_training_launched": False, "official_test_labels_read": False}
        write_json(ROOT / "outputs/unified/u2/eval/ddp_eight_rank_joint_evaluation.json", report)
        print(json.dumps({"status": "PASS", "world_size": 8, "resume_complete_videos": "NO_RERUN"}), flush=True)
    dist.barrier()
    if rank == 0:
        temporary.cleanup()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
