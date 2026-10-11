#!/usr/bin/env python3
"""Partial-label/semantic contract using actual shared F0 class features."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from locatemot.unified.losses.u2_association import association_targets
from locatemot.unified.models.u2.shared_track_core import SharedTrackCore
from locatemot.unified.runtime.u2_temporal_training import U2MOTTemporalTraining
from tools.unified.validate_u2_temporal_training import synthetic_clip


def ovmot_clip(base: torch.Tensor, novel: int, device: str | torch.device) -> list[dict]:
    clip = synthetic_clip(hidden=256, device=device)
    for frame in clip:
        candidates, gt = frame["candidates"], frame["supervision"]
        # The fifth proposal is an unlabeled open-world object. No novel GT
        # geometry, identity or class annotation is supplied to the trainer.
        keep = [0, 2, 3] if len(candidates["boxes"]) == 3 else [0, 1, 2, 3]
        candidates["predicted_class"] = base[torch.tensor(keep, device=device)]
        for name in ("id_visual", "semantic_visual", "decoder_query"):
            candidates[name] = torch.cat([candidates[name], torch.randn(1, 256, device=device)])
        candidates["boxes"] = torch.cat([candidates["boxes"], torch.tensor([[180., 0., 200., 40.]], device=device)])
        candidates["scores"] = torch.cat([candidates["scores"], torch.tensor([.9], device=device)])
        candidates["predicted_class"] = torch.cat([candidates["predicted_class"], torch.tensor([novel], device=device)])
        gt.category_ids = base[:2]
        gt.negative_category_ids = {int(base[2])}
    return clip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eight-gpu", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", 0))
    if args.eight_gpu:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        assert dist.get_world_size() == 8
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    torch.set_num_threads(2)
    torch.manual_seed(20261012)
    text_path = ROOT / "outputs/unified/u2/ovmot_semantics/f0_class_text.pt"
    text = torch.load(text_path, map_location="cpu", weights_only=False)
    class_text = text["embeddings"].to(device)
    assert class_text.shape == (1203, 256) and text["reference_only"]
    base = torch.tensor([i for i, f in enumerate(text["frequencies"]) if f in {"f", "c"}], device=device)
    novel = next(i for i, f in enumerate(text["frequencies"]) if f == "r")
    core = SharedTrackCore().to(device)
    torch.nn.init.constant_(core.association.new[-1].bias, 3.)
    model = U2MOTTemporalTraining(core).train()
    if args.eight_gpu:
        model = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-2)
    torch.manual_seed(20261012 + rank)
    clips = [ovmot_clip(base, novel, device) for _ in range(2)]
    captured = []

    def capture(module, args, output):
        output["new_logits"].retain_grad()
        captured.append(output)

    hook = core.register_forward_hook(capture)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        result = model(clips, 6000, task_type="ovmot", class_text=class_text, base_class_ids=base,
                       forcing_decisions=[[True, False, True, False]] * 2, augment=False)
    result["losses"]["total"].backward()
    hook.remove()
    gradients = {"semantic_projection": core.candidate_encoder.semantic_projection[0].weight.grad,
                 "semantic_gru": core.semantic_gru.weight_hh.grad,
                 "semantic_residual": core.ovmot.residual[0].weight.grad,
                 "semantic_gate": core.ovmot.gate[0].weight.grad,
                 "shared_text_projection": core.ovmot.text_projection.weight.grad,
                 "temperature": core.ovmot.temperature.grad,
                 "task_embeddings": core.ovmot.task_embeddings.weight.grad,
                 "identity_gru": core.identity_gru.weight_hh.grad,
                 "identity_pair": core.association.pair[0].weight.grad}
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients.values())
    assert captured[0]["new_logits"].grad[-1] == 0, "Unknown proposal received a birth/negative label"
    assert captured[0]["new_logits"].grad[2] > 0, "Certified hard negative did not suppress false birth"
    assert captured[0]["new_logits"].grad[3] == 0, "Ignored region received a birth/negative label"
    # Model-in-loop births include unknown objects; their rows must not become
    # identity/alive negatives simply because base annotations omit them.
    state = captured[1]["state"]
    targets = association_targets(state, result["diagnostics"][1]["track_gt_ids"],
                                  clips[0][2]["candidates"], clips[0][2]["supervision"])
    unknown_ids = [i for i, identity in result["diagnostics"][1]["track_gt_ids"].items() if identity < 0]
    assert unknown_ids and not targets.row_supervised[
        torch.tensor([i for i, track in enumerate(state.track_ids.tolist()) if track in unknown_ids], device=device)].any()
    assert all(set(output["state"].metadata) == {"max_age"} for output in captured)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    if args.eight_gpu:
        params = torch.cat([p.detach().flatten() for p in model.parameters()])
        reference = params.clone()
        dist.broadcast(reference, 0)
        discrepancy = (params - reference).abs().max()
        dist.all_reduce(discrepancy, op=dist.ReduceOp.MAX)
        assert discrepancy == 0
    core.eval()
    with torch.no_grad():
        candidate = clips[0][0]["candidates"]
        a = core(candidate, task_type="mot", class_text=class_text, frame_index=0)
        b = core(candidate, task_type="mot", class_text=class_text.flip(0), frame_index=0)
        assert torch.equal(a["identity_pair_logits"], b["identity_pair_logits"])
        assert torch.equal(a["state"].id_memory, b["state"].id_memory)
        assert not a["semantic_gate"].any()
        logits = core.ovmot.class_logits(a["state"].semantic_memory, class_text)
        assert logits.shape[1] == 1203 and torch.isfinite(logits[:, novel]).all()
    if rank == 0:
        report = {"format": "locatemot-u2-ovmot-training-contract-v1", "status": "PASS",
                  "world_size": 8 if args.eight_gpu else 1, "hidden_dim": 256,
                  "clips_per_rank": 2, "frames_per_clip": 4, "bf16": device.type == "cuda",
                  "class_feature_source": "actual frozen shared F0 BERT; reference-only contract",
                  "foundation_sha256": text["foundation_sha256"], "classes": 1203,
                  "base_class_loss_columns": len(base), "novel_class_loss_columns": 0,
                  "losses": {k: float(v.detach()) for k, v in result["losses"].items()},
                  "nonzero_gradient_branches": sorted(gradients),
                  "unknown_candidate_retained_without_birth_negative_labels": True,
                  "unknown_track_rows_excluded_from_identity_alive_contrast_labels": True,
                  "certified_hard_negative_reject_gradient": True,
                  "mot_identity_text_isolation": True, "mot_semantic_gate_zero": True,
                  "zero_shot_novel_logits_available": True,
                  "identical_parameters_all_ranks": True if args.eight_gpu else None,
                  "gt_in_runtime_state": False, "synthetic_tracking_candidates": True,
                  "scratch_optimizer_updates": 1, "formal_joint_optimization_launched": False}
        suffix = "8gpu" if args.eight_gpu else ("single" if device.type == "cuda" else "cpu")
        path = ROOT / f"outputs/unified/u2/eval/ovmot_training_contract_{suffix}.json"
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    if args.eight_gpu:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
