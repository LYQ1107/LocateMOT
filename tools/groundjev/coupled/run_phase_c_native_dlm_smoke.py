#!/usr/bin/env python3
"""Run one real native G2 clean-DLM -> GroundJEV coupled forward/backward.

The official G2 DLM is the only visual/language backbone.  Its image hidden
tokens are pooled into candidate regions and its non-image clean hidden state
is fused into the typed JEV readout.  This bounded smoke does not touch the
formal G2 run directory or claim a tracking metric.
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    os.environ["LOCATEMOT_GROUNDANYTHING_ROOT"] = str(Path(args.source_root).resolve())
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native DLM coupling smoke")
    image_path = Path(args.image).resolve()
    image = Image.open(image_path).convert("RGB")
    boxes = [(218, 425, 238, 503), (230, 413, 285, 675)]
    labels = ["person", "person"]
    start = time.perf_counter()
    bridge = G2FeatureBridge.from_native_clean_dlm(
        args.checkpoint,
        args.adapter,
        runtime_root=args.runtime_root,
        device=args.device,
        model_sha256=args.model_sha256,
        task_query="person",
    )
    state = bridge.encode(
        image, boxes, labels, frame_id=0, image_key=str(image_path), task_query="person",
    )
    model = GroundJEVModel(state.feature_dim, hidden_dim=256).to(device)
    memory = IdentityMemoryState()
    slots = build_dynamic_slots(memory.ordered())
    outputs = model(state, memory, slots)
    losses = structured_groundjev_losses(
        outputs,
        identity_targets=[0, 1], identity_valid=[True, True],
        admit_targets=[1, 1], admit_valid=[True, True],
        yesno_targets=[1, 0], yesno_valid=[True, True],
        lifecycle_targets=[0, 1], lifecycle_valid=[True, True],
        memory_targets=[1, 0], memory_valid=[True, True],
    )
    total = losses["total"]
    if total is None or not torch.isfinite(total):
        raise RuntimeError("native DLM coupling loss is not finite")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if not torch.isfinite(torch.as_tensor(gradient_norm)):
        raise RuntimeError("native DLM coupling gradient is not finite")
    optimizer.step()
    elapsed = time.perf_counter() - start
    receipt = {
        "status": "PHASE_C_NATIVE_G2_DLM_COUPLED_FORWARD_BACKWARD_PASS",
        "project_root": "/data2/user/LocateMOT",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "image": str(image_path),
        "candidate_count": len(boxes),
        "feature_dim": state.feature_dim,
        "contextual_tokens_shape": None if state.contextual_tokens is None else list(state.contextual_tokens.shape),
        "contextual_tokens_used": outputs["contextual_tokens_used"],
        "bridge_forward_count": bridge.forward_count,
        "native_dlm_forward_count": bridge.provider.forward_count,
        "single_backbone": True,
        "independent_vlm_instances": 0,
        "provider_metadata": state.metadata,
        "losses": {key: None if value is None else float(value.detach().cpu()) for key, value in losses.items()},
        "gradient_norm_clipped": float(torch.as_tensor(gradient_norm).detach().cpu()),
        "elapsed_seconds": elapsed,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "scientific_claim": "native G2 visual plus clean-language coupling evidence only; no tracking metric",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
