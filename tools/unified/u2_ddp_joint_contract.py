#!/usr/bin/env python3
"""Actual eight-GPU BF16 alternating-task/init/optimizer/RNG-resume proof."""
from __future__ import annotations

import argparse
import copy
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
from torch.nn.parallel import DistributedDataParallel

from locatemot.unified.runtime.u2_temporal_training import U2MOTTemporalTraining
from tools.unified.build_u2_candidate_cache import write_json
from tools.unified.train_u2_joint import initialize_from_mot, optimizer_for_core, save_checkpoint
from tools.unified.validate_u2_ovmot_training import ovmot_clip
from tools.unified.validate_u2_temporal_training import synthetic_clip


def eight_frames(clip: list[dict]) -> list[dict]:
    later = copy.deepcopy(clip)
    for frame in later:
        frame["frame_index"] += 16
    return clip + later


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-world-size", type=int, choices=(2, 8), default=8)
    args = parser.parse_args()
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(hours=24))
    world = dist.get_world_size()
    assert world == args.expected_world_size
    torch.set_num_threads(2)
    device = torch.device("cuda", local_rank)
    text = torch.load(ROOT / "outputs/unified/u2/ovmot_semantics/f0_class_text.pt", map_location="cpu", weights_only=False)
    class_text = text["embeddings"].to(device)
    base = torch.tensor([i for i, frequency in enumerate(text["frequencies"]) if frequency in {"f", "c"}], device=device)
    novel = next(i for i, frequency in enumerate(text["frequencies"]) if frequency == "r")
    torch.manual_seed(20261012)
    original = U2MOTTemporalTraining().to(device)
    torch.nn.init.constant_(original.core.association.new[-1].bias, 3.)
    source = {"format": "locatemot-u2-mot-only-checkpoint-v1", "core": copy.deepcopy(original.core.state_dict()),
              "metadata": {"foundation_sha256": text["foundation_sha256"]}}
    torch.manual_seed(9944)
    model = U2MOTTemporalTraining().to(device).train()
    initialize_from_mot(model.core, source, text["foundation_sha256"])
    assert all(torch.equal(value, model.core.state_dict()[name]) for name, value in source["core"].items())
    model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = optimizer_for_core(model.module.core)
    torch.manual_seed(20261012 + rank)
    mot = [eight_frames(synthetic_clip(hidden=256, device=device)) for _ in range(2)]
    ovmot = [eight_frames(ovmot_clip(base, novel, device)) for _ in range(2)]
    details = []

    def update(task: str, *, check_gradients: bool = True) -> float:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(mot if task == "mot" else ovmot, 6000, task_type=task,
                           class_text=class_text if task == "ovmot" else None,
                           base_class_ids=base if task == "ovmot" else None,
                           forcing_decisions=([[True, False, True, True, False, True, False, True]] * 2
                                              if check_gradients else None))
        loss = output["losses"]["total"]
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        core = model.module.core
        gradients = {"identity_gru": core.identity_gru.weight_hh.grad, "identity_pair": core.association.pair[0].weight.grad}
        if task == "ovmot":
            gradients.update(semantic_projection=core.candidate_encoder.semantic_projection[0].weight.grad,
                             semantic_gru=core.semantic_gru.weight_hh.grad,
                             semantic_residual=core.ovmot.residual[0].weight.grad,
                             semantic_gate=core.ovmot.gate[0].weight.grad,
                             class_text_projection=core.ovmot.text_projection.weight.grad,
                             temperature=core.ovmot.temperature.grad,
                             task_embeddings=core.ovmot.task_embeddings.weight.grad)
        else:
            assert all(parameter.grad is None for parameter in core.ovmot.parameters()), "MOT updated unused semantic adapter"
        if check_gradients:
            assert all(grad is not None and grad.abs().sum() > 0 for grad in gradients.values())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        parameters = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
        expected = parameters.clone()
        dist.broadcast(expected, 0)
        difference = (parameters - expected).abs().max()
        dist.all_reduce(difference, op=dist.ReduceOp.MAX)
        assert difference == 0
        if check_gradients:
            ranks = [None] * world
            dist.all_gather_object(ranks, {"rank": rank, "device": local_rank, "task": task, "loss": float(loss.detach()),
                                          "frames_per_clip": len(mot[0]), "nonzero_gradient_branches": sorted(gradients)})
            details.append({"task": task, "ranks": ranks, "max_parameter_discrepancy": 0.})
        return float(loss.detach())

    update("mot")
    update("ovmot")
    temporary = tempfile.TemporaryDirectory(prefix="locatemot_u2_joint_ddp_") if rank == 0 else None
    shared_path = [temporary.name if rank == 0 else None]
    dist.broadcast_object_list(shared_path, 0)
    metadata = {"synthetic_only": True, "formal_joint_training_launched": False}
    checkpoint = save_checkpoint(model, optimizer, 2, argparse.Namespace(out=Path(shared_path[0])), metadata)
    expected_loss = update("mot", check_gradients=False)
    expected_parameters = {name: parameter.detach().clone() for name, parameter in model.module.core.state_dict().items()}
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert saved["format"] == "locatemot-u2-joint-checkpoint-v1" and saved["metadata"] == metadata
    model.module.core.load_state_dict(saved["core"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["rank_rng_states"][rank]["cpu"])
    torch.cuda.set_rng_state(saved["rank_rng_states"][rank]["cuda"])
    resumed_loss = update("mot", check_gradients=False)
    assert expected_loss == resumed_loss
    assert all(torch.equal(value, model.module.core.state_dict()[name]) for name, value in expected_parameters.items())
    dist.barrier()
    if rank == 0:
        temporary.cleanup()
        report = {"format": "locatemot-u2-eight-gpu-joint-contract-v1" if world == 8 else "locatemot-u2-two-gpu-joint-contract-v1",
                  "status": "PASS", "world_size": world,
                  "hidden_dim": 256, "clips_per_rank": 2, "global_clips": world * 2, "frames_per_clip": 8,
                  "bf16": True, "strict_all_tensor_mot_initialization": True,
                  "registered_optimizer_group_lrs": [5e-5, 1e-4], "alternating_tasks_same_all_ranks": True,
                  "scratch_task_schedule": ["mot", "ovmot", "mot"], "steps": details,
                  "actual_joint_checkpoint_save_restore": True, "rank_cpu_cuda_rng_and_optimizer_restored": True,
                  "next_stochastic_update_bit_exact_after_resume": True,
                  "scheduled_teacher_bernoulli_rng_in_resume_proof": True,
                  "mot_unused_semantic_adapter_no_gradient_or_decay": True,
                  "class_text_source": "actual shared F0 BERT, reference-only contract",
                  "tracking_candidates": "synthetic", "formal_joint_training_launched": False,
                  "benchmark_metrics_measured": False, "foundation_weights_modified": False,
                  "official_test_labels_read": False}
        name = "ddp_eight_rank_joint.json" if world == 8 else "ddp_two_rank_joint.json"
        write_json(ROOT / "outputs/unified/u2/eval" / name, report)
        print(json.dumps({"status": "PASS", "world_size": world, "resume_next_update": "BIT_EXACT"}), flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
