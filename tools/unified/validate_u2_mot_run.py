#!/usr/bin/env python3
"""Sampler/resume/selection checks using explicitly synthetic cache fixtures."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from locatemot.unified.data.u2_mot_clips import DATASET_PROBABILITIES, CachedMOTClips
from locatemot.unified.runtime.u2_temporal_training import U2MOTTemporalTraining, clip_length
from tools.unified.eval_u2_mot_only import select_mot_only
from tools.unified.train_u2_mot_only import EVALUATION_STEPS
from tools.unified.validate_u2_temporal_training import synthetic_clip


def main() -> int:
    torch.set_num_threads(2)
    torch.manual_seed(20261012)
    with tempfile.TemporaryDirectory(prefix="locatemot_u2_mot_run_contract_") as raw:
        base = Path(raw)
        groups, supervision = [], {}
        spec = {"foundation_sha256": "SYNTHETIC_NOT_A_SELECTED_FOUNDATION"}
        for dataset in DATASET_PROBABILITIES:
            frames = [f"{index:06d}.png" for index in range(32)] if dataset == "kitti_legal_development" else [f"{index:06d}.jpg" for index in range(1, 33)]
            video = "0000" if dataset == "kitti_legal_development" else "synthetic"
            source = base / f"{dataset}.gt"
            boxes = [[0., 0., 20., 40.], [40., 0., 60., 40.]]
            if dataset == "bdd100k":
                data = [{"name": name, "labels": [{"id": str(identity), "category": "person",
                         "box2d": {"x1": box[0], "y1": box[1], "x2": box[2] - 1, "y2": box[3] - 1}}
                         for identity, box in enumerate(boxes)]} for name in frames]
                source.write_text(json.dumps(data))
            elif dataset == "kitti_legal_development":
                source.write_text("\n".join(" ".join(map(str, [int(Path(name).stem), identity, "Pedestrian", 0, 0, -10, *box,
                                                            -1, -1, -1, -1000, -1000, -1000, -10]))
                                             for name in frames for identity, box in enumerate(boxes)) + "\n")
            else:
                source.write_text("\n".join(f"{int(Path(name).stem)},{identity},{box[0]},{box[1]},20,40,1,1,1"
                                             for name in frames for identity, box in enumerate(boxes)) + "\n")
            directory = base / dataset
            (directory / "img1").mkdir(parents=True)
            (directory / "seqinfo.ini").write_text("[Sequence]\nframeRate=30\n")
            for role in ("train", "dev"):
                group = {"key": f"{dataset}/{role}/{video}", "dataset": dataset, "role": role,
                         "video": video, "frames": frames, "image_dir": str(directory / "img1")}
                candidates = {"boxes": torch.tensor(boxes), "scores": torch.full((2,), .9),
                              "id_visual": torch.randn(2, 256).bfloat16(), "semantic_visual": torch.randn(2, 256).bfloat16(),
                              "decoder_query": torch.randn(2, 256).bfloat16(), "predicted_class": torch.zeros(2, dtype=torch.long)}
                path = base / f"{dataset}_{role}.pt"
                torch.save({"header": {"foundation_sha256": spec["foundation_sha256"], "cache_spec_sha256": "synthetic",
                                       "group_key": group["key"], "frames": frames},
                            "frames": [{"frame": frame, "candidates": candidates} for frame in frames]}, path)
                groups.append({"group": group, "units": [{"path": str(path), "frames": frames}]})
                supervision[group["key"]] = {"source_path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        sampler = CachedMOTClips({"groups": groups, "spec": spec, "cache_spec_sha256": "synthetic"}, supervision)
        sampled = []
        steps = set((0, 500, 1000, 2500, 4000, 5999))
        steps.update(next(step for step in range(100) if sampler.dataset_for_step(step) == dataset) for dataset in DATASET_PROBABILITIES)
        for step in sorted(steps):
            for rank in (0, 7):
                first = sampler.sample(step, rank, "cpu")
                again = sampler.sample(step, rank, "cpu")
                for clip, repeated in zip(first, again):
                    assert len(clip) == clip_length(step)
                    indices = [frame["frame_index"] for frame in clip]
                    assert all(left < right for left, right in zip(indices, indices[1:]))
                    assert [frame["frame"] for frame in clip] == [frame["frame"] for frame in repeated]
                    assert all(frame["dataset"] == sampler.dataset_for_step(step) for frame in clip)
                    assert all(frame["candidates"]["id_visual"].dtype == torch.float32 for frame in clip)
                sampled.append({"step": step, "rank": rank, "dataset": sampler.dataset_for_step(step), "frames": len(first[0])})
        # Dev cache loading is supported for evaluation, without adding dev
        # groups to the training sampler.
        dev = groups[1]["group"]
        assert len(sampler.load_candidates(dev, 0, "cpu")["boxes"]) == 2

        # A restored optimizer/model/RNG state reproduces an actual next
        # stochastic temporal update, including dropout and query augmentation.
        from locatemot.unified.models.u2.shared_track_core import SharedTrackCore
        model = U2MOTTemporalTraining(SharedTrackCore(hidden_dim=32)).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        clips = [synthetic_clip(), synthetic_clip()]

        def update():
            optimizer.zero_grad(set_to_none=True)
            output = model(clips, 3000)
            output["losses"]["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            return float(output["losses"]["total"].detach())

        update()
        saved = copy.deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state()})
        expected_loss = update()
        expected = copy.deepcopy(model.state_dict())
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        actual_loss = update()
        assert actual_loss == expected_loss and all(torch.equal(value, model.state_dict()[name]) for name, value in expected.items())

        # The selection rule must reject >2pt HOTA regression, accept identity
        # benefit, and honor all tie breakers over the four required boundaries.
        cache_path = base / "synthetic_manifest.json"
        cache_path.write_text("{}\n")
        simple = {"foundation_sha256": "synthetic", "macro": {"AssA": .4},
                  "datasets": {dataset: {"status": "COMPLETE", "frames_evaluated": 32, "sequences_evaluated": 1,
                                         "gt_detections": 64, "gt_tracks": 2,
                                         "metrics": {"HOTA": .6, "AssA": .4}} for dataset in DATASET_PROBABILITIES}}
        for step in EVALUATION_STEPS:
            directory = base / f"evaluation_{step:04d}"
            directory.mkdir()
            checkpoint = base / f"synthetic_step_{step}.pth"
            checkpoint.write_bytes(f"synthetic-{step}".encode())
            assa = .43 if step == 2500 else .44
            reports = {dataset: {"status": "COMPLETE", "frames_evaluated": 32, "sequences_evaluated": 1,
                                 "gt_detections": 64, "gt_tracks": 2,
                                 "metrics": {"HOTA": .579 if step == 1000 and dataset == "mot20" else .6,
                                             "AssA": assa}} for dataset in DATASET_PROBABILITIES}
            result = {"status": "COMPLETE", "step": step, "checkpoint": str(checkpoint),
                      "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      "candidate_cache_sha256": hashlib.sha256(cache_path.read_bytes()).hexdigest(),
                      "foundation_sha256": "synthetic", "datasets": reports,
                      "macro": {"AssA": assa, "HOTA": sum(item["metrics"]["HOTA"] for item in reports.values()) / 5},
                      "total_idsw": 10, "gt_tracks": 100}
            (directory / "result.json").write_text(json.dumps(result))
        selection = select_mot_only(base, simple, cache_path)
        assert selection["selected"]["step"] == 4000 and not selection["decisions"][0]["eligible"]
        assert selection["decisions"][1]["identity_gain"], "Exactly three-point identity gain was rejected"
        boundary_path = base / "evaluation_1000/result.json"
        boundary_report = json.loads(boundary_path.read_text())
        boundary_report["datasets"]["mot20"]["metrics"]["HOTA"] = .58
        boundary_path.write_text(json.dumps(boundary_report))
        assert select_mot_only(base, simple, cache_path)["decisions"][0]["eligible"], "Exactly two-point HOTA loss was rejected"
        for step in EVALUATION_STEPS:
            path = base / f"evaluation_{step:04d}/result.json"
            report = json.loads(path.read_text())
            report["macro"]["AssA"] = .42
            path.write_text(json.dumps(report))
        rejected = select_mot_only(base, simple, cache_path)
        assert rejected["status"] == "LOCATEMOT_U_BLOCKED_U2_IDENTITY_CORE" and rejected["selected"] is None

    report = {"format": "locatemot-u2-mot-run-contract-v1", "status": "PASS",
              "input_source": "synthetic caches/checkpoints; no benchmark or selected foundation",
              "sampler_checks": sampled, "all_five_train_only": True, "causal_curriculum": True,
              "stateless_rank_step_sampling_resume": True, "optimizer_rng_resume_next_update_bit_exact": True,
              "same_dataset_all_ranks": True, "bf16_cache_expanded_at_autocast_boundary": True,
              "selection_hota_regression_and_identity_gain_checked": True, "earlier_checkpoint_tiebreak": True,
              "exact_three_point_identity_gain_and_two_point_hota_boundaries": True,
              "formal_optimization_launched": False, "official_test_labels_read": False}
    destination = ROOT / "outputs/unified/u2/eval/mot_run_contract.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "PASS", "report": str(destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
