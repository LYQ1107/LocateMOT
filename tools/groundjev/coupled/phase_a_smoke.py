#!/usr/bin/env python3
"""Phase-A GroundJEV gate: real G2 visual forward + typed readout backward.

This is intentionally bounded.  It uses one real MOT20 image and two legal
boxes, loads the protected G2 step-11000 projector, performs one visual forward
on the selected GPU, then backpropagates the typed readout losses.  It does not
touch the formal G2 owner or its optimizer state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image

from locatemot.groundjev.coupled.dynamic_identity_slots import build_dynamic_slots
from locatemot.groundjev.coupled.g2_feature_bridge import G2FeatureBridge
from locatemot.groundjev.coupled.groundjev_model import GroundJEVCoupledModel
from locatemot.groundjev.coupled.losses import JointLossWeights, joint_losses


ROOT = Path(__file__).resolve().parents[3]
G2_ROOT = Path("/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final")
G2_BASE = Path("/data2/user/LocateMOT_GroundJEV/models/GroundAnything-DLM")
G2_SOURCE = Path("/data2/user/LocateMOT_GroundJEV/external/GroundAnything")
IMAGE = Path("/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-03/img1/000083.jpg")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path,
                        default=Path("/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/phase_a_g2_step11000"))
    parser.add_argument("--adapter", type=Path,
                        default=G2_ROOT / "checkpoints/adapter-step011000.pt")
    parser.add_argument("--image", type=Path, default=IMAGE)
    args = parser.parse_args()
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("Phase-A real G2 visual gate requires CUDA")
    args.out.mkdir(parents=True, exist_ok=False)
    if not args.adapter.is_file() or not G2_BASE.joinpath("model.safetensors").is_file():
        raise FileNotFoundError("protected G2 step-11000 adapter or base bundle missing")
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    # Two non-overlapping annotated MOT20 boxes from frame 83.  The second is
    # a legal duplicate-free proposal used to test candidate-local NEW slots.
    boxes = [[475.0, 858.0, 501.0, 881.0], [900.0, 620.0, 970.0, 760.0]]
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    start = time.perf_counter()
    bridge = G2FeatureBridge(checkpoint=G2_BASE, device=args.device,
                             g2_adapter_checkpoint=args.adapter,
                             local_source_root=G2_SOURCE)
    state = bridge.encode(image, boxes, frame_id=83, image_key=str(args.image),
                          task_query="person", metadata={"dataset": "MOT20", "split": "TRAIN"})
    device = torch.device(args.device)
    model = GroundJEVCoupledModel(state.feature_dim, hidden_dim=256).to(device)
    model.train()
    # Empty causal history makes NEW a legal action for every candidate.
    slots = build_dynamic_slots([], max_history=31)
    slots_per_candidate = [slots, slots]
    batch = model(state, slots_per_candidate, memories={}, memory_enabled=True)
    target = []
    option_tokens = []
    for row in batch.decisions:
        target.append(next(i for i, slot in enumerate(row.slots) if slot.token == "NEW"))
        option_tokens.append([slot.token for slot in row.slots])
    losses = joint_losses(
        identity_logits=[row.identity_logits for row in batch.decisions],
        identity_targets=target,
        admit_logits=torch.stack([row.admission_logits[0] for row in batch.decisions]),
        admit_targets=torch.ones(len(batch.decisions), device=device),
        admit_valid=torch.ones(len(batch.decisions), dtype=torch.bool, device=device),
        birth_logits=torch.stack([model.readout.birth_head(model.readout._context(
            state.region_tokens, state.contextual_tokens, memory_enabled=True))[i]
            for i in range(len(batch.decisions))]).reshape(-1),
        birth_targets=torch.ones(len(batch.decisions), device=device),
        birth_valid=torch.ones(len(batch.decisions), dtype=torch.bool, device=device),
        weights=JointLossWeights(grounding=0.0, identity_choice=1.0, yesno=0.25, birth=0.5),
    )
    total = losses["total"]
    if total is None or not torch.isfinite(total):
        raise RuntimeError("Phase-A typed loss is missing or non-finite")
    total.backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
    if not torch.isfinite(torch.tensor(grad_norm)) or grad_norm <= 0:
        raise RuntimeError(f"typed readout gradient is invalid: {grad_norm}")
    elapsed = time.perf_counter() - start
    receipt = {
        "status": "PHASE_A_REAL_G2_VISUAL_TYPED_READOUT_FORWARD_BACKWARD_COMPLETE",
        "project": "LocateMOT/GroundJEV",
        "single_backbone": True,
        "second_vlm_loaded": False,
        "g2_base_bundle": str(G2_BASE),
        "g2_base_model_safetensors_sha256": sha256(G2_BASE / "model.safetensors"),
        "g2_adapter": str(args.adapter),
        "g2_adapter_sha256": sha256(args.adapter),
        "g2_adapter_step": 11000,
        "image": str(args.image),
        "image_sha256": sha256(args.image),
        "dataset_scope": "MOT20 TRAIN frame 83; engineering gate only",
        "boxes": boxes,
        "state": state.audit_dict(),
        "typed_decisions": batch.to_dict(),
        "losses": {name: (float(value.detach()) if value is not None else None)
                    for name, value in losses.items()},
        "typed_readout_trainable_parameters": sum(p.numel() for p in trainable),
        "typed_readout_gradient_norm": grad_norm,
        "visual_forward_calls": bridge.forward_calls,
        "elapsed_seconds": elapsed,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else 0,
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if torch.cuda.is_available() else 0,
        "quality_claim": False,
        "note": "Real G2 visual projector state and real MOT20 pixels; Phase-A engineering evidence, not a benchmark.",
    }
    (args.out / "phase_a_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
