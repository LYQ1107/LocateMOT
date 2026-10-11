#!/usr/bin/env python3
"""Eight-GPU proof of the registered MOT rollout/loss, using synthetic clips."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from locatemot.unified.runtime.u2_temporal_training import U2MOTTemporalTraining
from tools.unified.validate_u2_temporal_training import synthetic_clip


def main() -> int:
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    if dist.get_world_size() != 8:
        raise ValueError("This contract requires eight actual GPU ranks")
    torch.manual_seed(20261012)
    device = torch.device("cuda", local_rank)
    model = DistributedDataParallel(U2MOTTemporalTraining().to(device).train(), device_ids=[local_rank],
                                    find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    torch.manual_seed(20261012 + rank)
    clips = [synthetic_clip(hidden=256, device=device) for _ in range(2)]
    results = []
    for step in (500, 3000):
        optimizer.zero_grad(set_to_none=True)
        decisions = [[True] * 4] * 2 if step == 500 else [[True, False, True, False]] * 2
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(clips, step, forcing_decisions=decisions)
        loss = output["losses"]["total"]
        loss.backward()
        gradients = {"candidate_projection": model.module.core.candidate_encoder.id_projection[0].weight.grad,
                     "identity_gru": model.module.core.identity_gru.weight_hh.grad,
                     "pair_head": model.module.core.association.pair[0].weight.grad,
                     "set_interaction": model.module.core.set_interaction.encoder.layers[0].linear1.weight.grad,
                     "motion": model.module.core.motion.head[0].weight.grad,
                     "alive": model.module.core.lifecycle.head[0].weight.grad,
                     "birth": model.module.core.association.new[0].weight.grad}
        finite = int(torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        nonzero = int(all(g is not None and g.abs().sum() > 0 for g in gradients.values()))
        check = torch.tensor([finite, nonzero], device=device)
        dist.all_reduce(check, op=dist.ReduceOp.MIN)
        assert check.tolist() == [1, 1], "Nonfinite or disconnected registered temporal gradients"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        parameters = torch.cat([value.detach().flatten() for value in model.parameters()])
        reference = parameters.clone()
        dist.broadcast(reference, src=0)
        difference = (parameters - reference).abs().max()
        dist.all_reduce(difference, op=dist.ReduceOp.MAX)
        assert difference == 0, "DDP parameters diverged"
        details = [None] * 8
        dist.all_gather_object(details, {"rank": rank, "cuda_device": local_rank,
                                        "teacher_frames": sum(d["teacher_forced"] for d in output["diagnostics"]),
                                        "model_frames": sum(not d["teacher_forced"] for d in output["diagnostics"]),
                                        "loss": float(loss.detach())})
        results.append({"step": step, "ranks": details, "finite_gradients": True,
                        "nonzero_gradient_branches": sorted(gradients), "max_ddp_parameter_difference": 0.})
    if rank == 0:
        report = {"format": "locatemot-u2-eight-gpu-supervision-contract-v1", "status": "PASS",
                  "world_size": 8, "hidden_dim": 256, "clips_per_rank": 2, "global_clips": 16,
                  "frames_per_clip": 4, "bf16_autocast": True, "same_task_all_ranks": "mot",
                  "registered_six_losses": True, "actual_query_augmentation": True,
                  "temporal_graph_detached": False, "steps": results,
                  "input_source": "synthetic; two scratch optimizer updates, no benchmark training",
                  "foundation_weights_modified": False, "formal_track_training_launched": False,
                  "benchmark_metrics_measured": False, "official_test_labels_read": False}
        path = ROOT / "outputs/unified/u2/eval/ddp_eight_rank_supervision.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"status": "PASS", "report": str(path)}), flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
