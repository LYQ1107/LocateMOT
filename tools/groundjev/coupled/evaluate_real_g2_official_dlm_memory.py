#!/usr/bin/env python3
"""Inject persistent box/identity state into the official G2 DLM decoder.

This is the first coupling check on the actual GroundAnything hybrid decoder.
The same frozen G2 model receives either the normal prompt (C0) or the normal
prompt plus one projected previous-frame identity token (C2).  The memory is
inserted through ``<|vision_start|>`` marker positions, which the official
draft stream already excludes from text tokens.  A fresh projector is used in
this diagnostic; it verifies the shared-backbone interface and is not a
trained tracking result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = Path("/data2/user/LocateMOT_GroundJEV")
SOURCE_ROOT = RUNTIME_ROOT / "external/GroundAnything"
sys.path[:0] = [str(PROJECT_ROOT), str(RUNTIME_ROOT), str(SOURCE_ROOT)]

from locatemot.groundjev.coupled.memory_tokens import IdentityBoxStateTokenProjector  # noqa: E402
from locatemot.groundjev.coupled.official_dlm_memory import (  # noqa: E402
    OfficialDLMIdentityMemory,
    append_memory_markers,
)
from locatemot.groundjev.native_output_repair import repair_gam_bbox_output  # noqa: E402
from tools.groundjev.coupled.evaluate_real_g2_dense_multibox import (  # noqa: E402
    _greedy_recall,
    _read_all_boxes,
    _row,
)
from tools.groundjev.coupled.evaluate_real_g2_official_dlm_dense import (  # noqa: E402
    _prepare_prompt,
    _token_contract,
)
from tools.groundjev.coupled.train_real_g2_coupled_memory_jev import (  # noqa: E402
    _mean_token,
    _read_boxes,
)
from tools.groundjev.coupled.real_g2_memory_prefill import _prepare  # noqa: E402


def _run_official(
    engine: Any,
    config: Any,
    model: Any,
    tokenizer: Any,
    prepared: dict[str, Any],
    contract: dict[str, Any],
    *,
    max_new_tokens: int,
    memory_tokens: torch.Tensor | None = None,
    marker_token_id: int | None = None,
) -> dict[str, Any]:
    input_ids = prepared["input_ids"]
    attention_mask = prepared["attention_mask"]
    mm_types = prepared.get("mm_token_type_ids")
    context: OfficialDLMIdentityMemory | None = None
    enabled = memory_tokens is not None
    if enabled:
        if marker_token_id is None:
            raise ValueError("marker_token_id is required when memory is enabled")
        input_ids, attention_mask, mm_types = append_memory_markers(
            input_ids,
            attention_mask,
            mm_types,
            marker_token_id=marker_token_id,
            memory_count=int(memory_tokens.shape[0]),
        )
        context = OfficialDLMIdentityMemory(
            model,
            marker_token_id=marker_token_id,
            memory_tokens=memory_tokens,
        )
    generation_args = {
        **contract,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "mm_token_type_ids": mm_types,
        "pixel_values": prepared.get("pixel_values"),
        "image_grid_thw": prepared.get("image_grid_thw"),
        "pixel_values_videos": prepared.get("pixel_values_videos"),
        "video_grid_thw": prepared.get("video_grid_thw"),
        "repetition_penalty": 1.0,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": None,
        "no_repeat_ngram_size": 0,
    }
    if context is None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_ids = engine.generate(config, **generation_args)
    else:
        with context, torch.autocast("cuda", dtype=torch.bfloat16):
            output_ids = engine.generate(config, **generation_args)
    generated_ids = output_ids[0].tolist()
    text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    recovered, repair_report = repair_gam_bbox_output(text)
    return {
        "memory_enabled": enabled,
        "memory_token_count": 0 if memory_tokens is None else int(memory_tokens.shape[0]),
        "generated_text": text,
        "generated_token_count": len(generated_ids),
        "recovered_pairs": [[label, list(box)] for label, box in recovered],
        "repair_report": repair_report,
        "generation_stats": dict(getattr(model, "_last_generation_stats", {})),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--g2-adapter", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    os.environ.setdefault("GAM_DLM_FLEX_CUDAGRAPHS", "0")
    os.environ["LOCATEMOT_GROUNDANYTHING_ROOT"] = str(SOURCE_ROOT)
    from tools.groundjev.g2.compat import patch_transformers_causal_mask
    from tools.groundjev.g2.train_g2_dlm import _strict_load_trainable_state, load_model
    from train.dlm.data import DLMDataCollator
    from infer.dlm import DLMInferenceConfig, DLMInferenceEngine

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("official DLM memory evaluation requires CUDA")
    patch_transformers_causal_mask()
    start = time.perf_counter()
    model, processor, _, audit = load_model(Path(args.checkpoint).resolve(), device)
    saved = torch.load(Path(args.g2_adapter).resolve(), map_location="cpu", weights_only=False)
    _strict_load_trainable_state(model, saved["model_trainable_state"])
    model.set_padding_free_packing(False)
    model.gradient_checkpointing_disable()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    collator = DLMDataCollator(
        processor, max_length=32768, max_joint_length=65536, model_family="groundinganything_qwen3"
    )
    tokenizer = processor.tokenizer
    contract = _token_contract(tokenizer)
    from infer.dlm import DLMInferenceConfig

    config = DLMInferenceConfig(
        mode="hierarchy_dynamic",
        block_size=32,
        sub_block_size=8,
        denoise_steps=8,
        confidence_threshold=0.9,
        acceptance_policy="confidence",
        entropy_threshold=0.8,
        enforce_structured_termination=True,
        structured_max_coordinate_groups=0,
    )
    engine = DLMInferenceEngine(model)
    marker_token_id = int(tokenizer.convert_tokens_to_ids("<|vision_start|>"))
    if marker_token_id < 0:
        raise RuntimeError("tokenizer lacks the official vision marker token")
    specs = [
        (
            "MOT20-03",
            Path("/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-03"),
            Path("/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-03/gt/gt.txt"),
            82,
            27,
            1173,
            880,
        ),
        (
            "MOT20-05",
            Path("/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-05"),
            Path("/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-05/gt/gt.txt"),
            13,
            17,
            1654,
            1080,
        ),
    ]
    results: dict[str, Any] = {}
    visual_calls = 0
    for name, root, gt, previous_frame, target_id, width, height in specs:
        current_frame = previous_frame + 1
        previous_all = _read_all_boxes(gt, frames=[previous_frame], width=width, height=height)[previous_frame]
        current_all = _read_all_boxes(gt, frames=[current_frame], width=width, height=height)[current_frame]
        previous_target = _read_boxes(
            gt,
            target_id=target_id,
            frame_start=previous_frame,
            frame_count=1,
            width=width,
            height=height,
        )[previous_frame]
        previous_row = _row(root / "img1" / f"{previous_frame:06d}.jpg", previous_all)
        current_row = _row(root / "img1" / f"{current_frame:06d}.jpg", current_all)
        # One real previous-frame encode produces the causal visual summary.
        previous_features = _prepare(model, collator, previous_row, device)
        visual_calls += 1
        current_prompt = _prepare_prompt(model, collator, current_row, device)
        input_dim = int(previous_features["image_tokens"].shape[-1])
        hidden = int(current_prompt["input_ids"].shape[-1] * 0 + model.lm_head.weight.shape[-1])
        projector = IdentityBoxStateTokenProjector(input_dim, hidden, hidden_dim=512).to(device).eval()
        memory = projector(
            {target_id: _mean_token(previous_features["image_tokens"])},
            {target_id: torch.tensor([value / 999.0 for value in previous_target], device=device)},
            ordered_ids=[target_id],
        )
        # C0 and C2 are separate official requests.  The receipt records two
        # current-image encodes plus the one previous-image encode; no second
        # VLM/model instance is created.
        c0 = _run_official(
            engine, config, model, tokenizer, current_prompt, contract,
            max_new_tokens=args.max_new_tokens,
        )
        visual_calls += 1
        c2 = _run_official(
            engine, config, model, tokenizer, current_prompt, contract,
            max_new_tokens=args.max_new_tokens, memory_tokens=memory,
            marker_token_id=marker_token_id,
        )
        visual_calls += 1
        for item in (c0, c2):
            predicted = [pair[1] for pair in item["recovered_pairs"]]
            item["recall_at_iou_0_5"] = _greedy_recall(predicted, current_all)
        results[name] = {
            "previous_frame": previous_frame,
            "current_frame": current_frame,
            "target_id": target_id,
            "target_box_previous": list(previous_target),
            "target_boxes_current": [list(box) for box in current_all],
            "C0_official": c0,
            "C2_official_previous_identity_state": c2,
        }

    receipt = {
        "status": "PHASE_C_REAL_G2_OFFICIAL_DLM_MEMORY_INTERFACE_COMPLETE",
        "project_root": str(PROJECT_ROOT),
        "g2_checkpoint": str(Path(args.checkpoint).resolve()),
        "g2_adapter": str(Path(args.g2_adapter).resolve()),
        "adapter_step": int(saved["step"]),
        "base_model_sha256": args.model_sha256,
        "decoder": "official infer.dlm.DLMInferenceEngine hierarchy_dynamic",
        "memory_interface": "OfficialDLMIdentityMemory with <|vision_start|> marker suffix",
        "memory_projector": "untrained IdentityBoxStateTokenProjector(previous visual summary + normalized previous box)",
        "visual_embed_calls": visual_calls,
        "independent_vlm_instances": 0,
        "g2_frozen": True,
        "formal_g2_run_untouched": True,
        "results": results,
        "g2_trainable_parameters": audit.get("trainable_parameters"),
        "elapsed_seconds": time.perf_counter() - start,
        "scientific_claim": "official shared-backbone memory interface diagnostic only; projector is untrained and no tracking gain is claimed",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
