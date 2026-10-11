#!/usr/bin/env python3
"""Eight-GPU temporal core validation on synthetic clips, not a formal run."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from locatemot.unified.models.u2.shared_track_core import SharedTrackCore


class TemporalProbe(nn.Module):
    """Keep the full temporal graph inside one DDP forward per global step."""

    def __init__(self) -> None:
        super().__init__()
        self.core = SharedTrackCore()
        # Deterministic births/matches exercise recurrent gradients before any
        # association head has been learned. These are scratch probe weights.
        nn.init.constant_(self.core.association.new[-1].bias, 3.)
        nn.init.constant_(self.core.association.pair[-1].bias, 3.)
        nn.init.constant_(self.core.association.no_match[-1].bias, -3.)

    def forward(self, clips, task: str, text: torch.Tensor) -> torch.Tensor:
        losses = []
        for clip in clips:
            state = None
            for index, candidates in enumerate(clip):
                output = self.core(candidates, state, task_type=task, class_text=text,
                                   frame_index=index, training=True)
                state = output["state"]
                assert state.num_tracks > 0
                # Contract objective exercises every temporal identity branch
                # and the separate semantic update; it is not benchmark loss.
                loss = output["new_logits"].square().mean() + state.id_memory.square().mean()
                for field in ("pair_logits", "no_match_logits", "alive_logits", "motion_delta"):
                    if output[field].numel():
                        loss = loss + output[field].square().mean()
                if task == "ovmot":
                    loss = loss + state.semantic_memory.square().mean()
                    if output["semantic_gate"].numel():
                        loss = loss + output["semantic_gate"].square().mean()
                losses.append(loss)
        return torch.stack(losses).mean()


def main() -> int:
    rank_local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank_local)
    torch.set_num_threads(2)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise ValueError("The U2 temporal contract requires exactly eight GPU ranks")
    device = torch.device("cuda", rank_local)
    torch.manual_seed(20261002)
    model = DistributedDataParallel(TemporalProbe().to(device).train(),
                                    device_ids=[rank_local], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    generator = torch.Generator(device=device).manual_seed(20261002 + rank)
    clips = []
    for clip_index in range(2):
        visual = torch.randn(3, 256, generator=generator, device=device)
        clip = []
        for frame in range(4):
            feature = visual + .01 * frame
            boxes = torch.tensor([[0., 0., .2, .4], [.3, .1, .5, .5], [.6, .2, .8, .6]], device=device)
            clip.append({"boxes": boxes + .001 * frame,
                         "scores": torch.tensor([.9, .8, .7], device=device),
                         "id_visual": feature, "semantic_visual": feature * .8,
                         "decoder_query": feature * .4})
        clips.append(clip)
    text = torch.randn(3, 256, generator=generator, device=device)
    results = []
    for task in ("mot", "ovmot"):
        tasks = [None] * world
        dist.all_gather_object(tasks, task)
        assert set(tasks) == {task}
        optimizer.zero_grad(set_to_none=True)
        loss = model(clips, task, text)
        loss.backward()
        finite = torch.tensor([int(torch.isfinite(loss) and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()))], device=device)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        assert finite.item() == 1, "Nonfinite recurrent DDP loss/gradient"
        gradients = {"identity_gru": model.module.core.identity_gru.weight_hh.grad,
                     "set_interaction": model.module.core.set_interaction.encoder.layers[0].linear1.weight.grad,
                     "pair_head": model.module.core.association.pair[0].weight.grad}
        if task == "ovmot":
            gradients.update(semantic_gru=model.module.core.semantic_gru.weight_hh.grad,
                             semantic_text=model.module.core.ovmot.text_projection.weight.grad)
        assert all(value is not None and value.abs().sum() > 0 for value in gradients.values()), \
            "The temporal contract did not exercise every required branch"
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        parameters = torch.cat([value.detach().flatten() for value in model.parameters()])
        master = parameters.clone()
        dist.broadcast(master, src=0)
        difference = (parameters - master).abs().max()
        dist.all_reduce(difference, op=dist.ReduceOp.MAX)
        assert difference.item() == 0, "DDP optimizer parameters diverged across ranks"
        results.append({"task": task, "finite_loss_and_gradients": True,
                        "nonzero_gradient_branches": sorted(gradients),
                        "maximum_parameter_difference_across_ranks": difference.item()})
    devices = [None] * world
    dist.all_gather_object(devices, {"rank": rank, "local_rank": rank_local,
                                    "cuda_device": torch.cuda.current_device()})
    assert {item["cuda_device"] for item in devices} == set(range(8))
    if rank == 0:
        report = {"format": "locatemot-u2-eight-gpu-temporal-contract-v1", "status": "PASS",
                  "world_size": world, "rank_devices": devices, "same_task_all_ranks": True,
                  "clips_per_rank": 2, "global_clips_per_step": 16, "frames_per_clip": 4,
                  "temporal_graph_detached_between_frames": False, "tasks": results,
                  "input_source": "synthetic contract-only features",
                  "foundation_weights_modified": False, "formal_track_training_launched": False,
                  "benchmark_metrics_measured": False, "official_test_labels_read": False}
        destination = ROOT / "outputs/unified/u2/eval/ddp_eight_rank_core.json"
        destination.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
