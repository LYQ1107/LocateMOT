#!/usr/bin/env python3
"""Probe the native G2 DLM clean hidden state on one real manifest row.

This probe uses the same loader and hybrid clean-stream path as the formal G2
trainer, restores a protected adapter checkpoint in memory, and records the
image-token hidden-state shape.  It is read-only with respect to the formal
run directory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    runtime_root = Path(args.runtime_root).resolve()
    os.environ.setdefault("GAM_DLM_FLEX_CUDAGRAPHS", "0")
    os.environ.setdefault("LOCATEMOT_GROUNDANYTHING_ROOT", str(runtime_root / "external/GroundAnything"))
    sys.path.insert(0, str(runtime_root))
    sys.path.insert(0, str(runtime_root / "external/GroundAnything"))
    from tools.groundjev.g2.train_g2_dlm import load_model, _strict_load_trainable_state
    from tools.groundjev.g2.compat import patch_transformers_causal_mask
    from train.dlm.data import DLMDataCollator

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native DLM hidden-state probe")
    mask_compatibility = patch_transformers_causal_mask()
    rows_path = Path(args.rows).resolve()
    with rows_path.open() as handle:
        for index, line in enumerate(handle):
            if index == int(args.row_index):
                row = json.loads(line)
                break
        else:
            raise IndexError(args.row_index)
    start = time.perf_counter()
    model, processor, _, audit = load_model(Path(args.checkpoint).resolve(), device)
    adapter_path = Path(args.adapter).resolve()
    adapter = torch.load(adapter_path, map_location="cpu", weights_only=False)
    if int(adapter.get("step", -1)) <= 0:
        raise RuntimeError("adapter has no positive step")
    _strict_load_trainable_state(model, adapter["model_trainable_state"])
    model.set_padding_free_packing(False)
    model.gradient_checkpointing_disable()
    model.eval()
    collator = DLMDataCollator(processor, max_length=32768, max_joint_length=65536,
                               model_family="groundinganything_qwen3")
    batch = collator([row])
    tensor_batch = {key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in batch.items()}
    visual_dtype = next(model.base_model.model.visual.parameters()).dtype
    if "pixel_values" in tensor_batch:
        tensor_batch["pixel_values"] = tensor_batch["pixel_values"].to(dtype=visual_dtype)
    if "pixel_values_videos" in tensor_batch:
        tensor_batch["pixel_values_videos"] = tensor_batch["pixel_values_videos"].to(dtype=visual_dtype)
    # Match the formal trainer's BF16 autocast boundary.  The K3 vision tower
    # emits intermediate FP32 tensors in eager mode while its projector norms
    # are BF16; entering autocast at the complete native DLM boundary is
    # required for the same dtype contract used by training.
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        clean_embeds, position_ids = model._embed_clean(
            tensor_batch["input_ids"], tensor_batch["attention_mask"],
            tensor_batch.get("pixel_values"), tensor_batch.get("image_grid_thw"),
            tensor_batch.get("pixel_values_videos"), tensor_batch.get("video_grid_thw"),
            tensor_batch.get("mm_token_type_ids"), tensor_batch.get("patch_positions"),
        )
        streams = model._build_streams(
            tensor_batch["input_ids"], tensor_batch["labels"], tensor_batch["attention_mask"],
            clean_embeds, position_ids,
        )
        _, clean_hidden = model._hybrid_language_forward(streams)
    image_pad_id = int(processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    image_positions = tensor_batch["input_ids"].eq(image_pad_id)
    selected = clean_hidden[image_positions]
    receipt = {
        "status": "G2_NATIVE_CLEAN_HIDDEN_STATE_PROBE_COMPLETE",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "adapter": str(adapter_path),
        "adapter_step": int(adapter["step"]),
        "manifest_row_index": int(args.row_index),
        "row_id": row.get("id"),
        "image": row.get("images", [{}])[0].get("path"),
        "input_token_count": int(tensor_batch["input_ids"].shape[-1]),
        "image_pad_token_count": int(image_positions.sum().item()),
        "clean_embed_shape": list(clean_embeds.shape),
        "clean_hidden_shape": list(clean_hidden.shape),
        "image_hidden_shape": list(selected.shape),
        "model_trainable_parameters": audit.get("trainable_parameters"),
        "mask_compatibility": mask_compatibility,
        "elapsed_seconds": time.perf_counter() - start,
        "single_dlm_visual_forward": True,
        "independent_vlm_instances": 0,
        "scientific_claim": "native hidden-state alignment evidence only; no tracking metric",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
