#!/usr/bin/env python3
"""Real one-frame GroundJEV Phase-A smoke using the native G2 clean path.

This is deliberately bounded: one real image, two native GroundAnything
regions, one typed readout backward, and one causal runtime step.  It does not
touch the formal G2 trainer or its checkpoint files.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image

from locatemot.groundjev.coupled.dynamic_identity_slots import build_dynamic_slots
from locatemot.groundjev.coupled.g2_feature_bridge import G2FeatureBridge
from locatemot.groundjev.coupled.groundjev_model import GroundJEVModel
from locatemot.groundjev.coupled.identity_memory import IdentityMemoryState
from locatemot.groundjev.coupled.losses import structured_groundjev_losses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="immutable GroundAnything base bundle used with the G2 adapter")
    parser.add_argument("--adapter", required=True,
                        help="G2 adapter checkpoint; prevents silently using G0 visual weights")
    parser.add_argument("--runtime-root", required=True,
                        help="GroundJEV runtime checkout containing the native DLM loader")
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.environ["LOCATEMOT_GROUNDANYTHING_ROOT"] = str(Path(args.source_root).resolve())
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for the real Phase-A smoke")
    torch_device = torch.device(args.device)
    image = Image.open(args.image).convert("RGB")
    boxes = [(218, 425, 238, 503), (230, 413, 285, 675)]
    labels = ["person", "person"]
    start = time.perf_counter()
    bridge = G2FeatureBridge.from_native_clean_dlm(
        args.checkpoint, args.adapter, runtime_root=args.runtime_root,
        device=args.device, model_sha256=args.model_sha256, task_query="person",
    )
    state = bridge.encode(image, boxes, labels, frame_id=0, image_key=str(args.image),
                          task_query="person")
    model = GroundJEVModel(state.feature_dim, hidden_dim=256).to(torch_device)
    memory = IdentityMemoryState()
    slots = build_dynamic_slots(memory.ordered())
    outputs = model(state, memory, slots)
    losses = structured_groundjev_losses(
        outputs, identity_targets=[0, 1], identity_valid=[True, True],
        admit_targets=[1, 1], admit_valid=[True, True],
        yesno_targets=[1, 0], yesno_valid=[True, True],
        lifecycle_targets=[0, 1], lifecycle_valid=[True, True],
        memory_targets=[1, 0], memory_valid=[True, True],
    )
    total = losses["total"]
    if total is None or not torch.isfinite(total):
        raise RuntimeError("Phase-A loss is not finite")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(torch.as_tensor(gradient_norm)):
        raise RuntimeError("Phase-A readout gradient is not finite")
    optimizer.step()
    elapsed = time.perf_counter() - start
    peak = int(torch.cuda.max_memory_allocated(torch_device)) if torch_device.type == "cuda" else 0
    receipt = {
        "status": "PHASE_A_REAL_G2_NATIVE_DLM_FORWARD_BACKWARD_PASS",
        "project_root": "/data2/user/LocateMOT",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "runtime_root": str(Path(args.runtime_root).resolve()),
        "model_sha256": args.model_sha256,
        "image": str(Path(args.image).resolve()),
        "candidate_count": len(boxes),
        "feature_dim": state.feature_dim,
        "bridge_forward_count": bridge.forward_count,
        "native_dlm_forward_count": bridge.provider.forward_count,
        "single_backbone": True,
        "independent_vlm_instances": 0,
        "losses": {k: None if v is None else float(v.detach().cpu()) for k, v in losses.items()},
        "gradient_norm_clipped": float(torch.as_tensor(gradient_norm).detach().cpu()),
        "elapsed_seconds": elapsed,
        "peak_memory_allocated_bytes": peak,
        "memory_enabled": False,
        "scientific_claim": "engineering smoke only; no tracking metric",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
