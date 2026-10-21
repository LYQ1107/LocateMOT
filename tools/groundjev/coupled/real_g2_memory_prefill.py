#!/usr/bin/env python3
"""Run a real G2 causal prefill comparison with reused visual embeddings.

This is a bounded Phase-C contract experiment.  The official GroundAnything
DLM encodes one previous and one current image; the current prompt embeddings
are then reused for memory-disabled, causal-memory and shuffled-memory
language prefills.  No second VLM or image encoder is instantiated.  The
memory projector is deliberately untrained and the output is not a tracking
metric.
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

from locatemot.groundjev.coupled.causal_memory import causal_prefill_with_memory
from locatemot.groundjev.coupled.memory_tokens import IdentityMemoryTokenProjector
from locatemot.groundjev.native_output_repair import repair_gam_bbox_output


def _answer(query: str, boxes: list[tuple[int, int, int, int]]) -> str:
    payload = ",".join("<" + "><".join(str(v) for v in box) + ">" for box in boxes)
    return f"<|object_ref_start|>{query}<|object_ref_end|><|box_start|>{payload}<|box_end|>"


def _row(image: Path, query: str, boxes: list[tuple[int, int, int, int]]) -> dict[str, Any]:
    return {
        "id": f"groundjev_prefill_{image.stem}",
        "messages": [
            {
                "role": "user",
                "content": f"<image>\nLocate all the instances that match the following categories: {query}.",
                "loss": None,
            },
            {"role": "assistant", "content": _answer(query, boxes), "loss": None},
        ],
        "images": [{"path": str(image), "bytes": None}],
        "source": "GroundJEV:real_memory_prefill",
    }


def _prepare(model: Any, collator: Any, row: dict[str, Any], device: torch.device) -> dict[str, Any]:
    batch = collator([row])
    tensor_batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    visual_dtype = next(model.base_model.model.visual.parameters()).dtype
    for key in ("pixel_values", "pixel_values_videos"):
        if key in tensor_batch:
            tensor_batch[key] = tensor_batch[key].to(dtype=visual_dtype)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        embeds, positions = model._embed_clean(
            tensor_batch["input_ids"], tensor_batch["attention_mask"],
            tensor_batch.get("pixel_values"), tensor_batch.get("image_grid_thw"),
            tensor_batch.get("pixel_values_videos"), tensor_batch.get("video_grid_thw"),
            tensor_batch.get("mm_token_type_ids"), tensor_batch.get("patch_positions"),
        )
    labels = tensor_batch["labels"]
    target = labels.ne(-100).nonzero(as_tuple=False)
    if target.numel() == 0:
        raise RuntimeError("collated row has no supervised answer token")
    answer_start = int(target[0, 1])
    image_pad_id = int(collator.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
    image_positions = tensor_batch["input_ids"].eq(image_pad_id)
    image_tokens = embeds[image_positions].detach().float()
    if image_tokens.numel() == 0:
        raise RuntimeError("current row has no image tokens")
    prompt_embeds = embeds[:, :answer_start]
    prompt_mask = tensor_batch["attention_mask"][:, :answer_start]
    prompt_positions = positions[..., :answer_start]
    answer_mask = tensor_batch["attention_mask"][0, answer_start:].bool()
    answer_tokens = tensor_batch["input_ids"][0, answer_start:][answer_mask].detach().clone()
    return {
        "embeds": prompt_embeds,
        "mask": prompt_mask,
        "positions": prompt_positions,
        "image_tokens": image_tokens,
        "answer_start": answer_start,
        "input_tokens": int(tensor_batch["input_ids"].shape[-1]),
        "image_tokens_count": int(image_tokens.shape[0]),
        "answer_tokens": answer_tokens,
    }


def _generate(model: Any, prepared: dict[str, Any], memory: torch.Tensor, *, enabled: bool,
              max_new_tokens: int, stop_ids: set[int], structural: dict[str, int],
              coordinate_range: tuple[int, int]) -> dict[str, Any]:
    # The formal G2 clean path enters this autocast boundary before its
    # language layers.  The restored LoRA base projections remain FP32 while
    # the multimodal embeddings are BF16; omitting the boundary causes a real
    # dtype failure before any memory comparison can run.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prefill = causal_prefill_with_memory(
            model.language_model,
            model.lm_head,
            prepared["embeds"], prepared["mask"], prepared["positions"], memory,
            native_position_ids=model._native_position_ids,
            enabled=enabled,
            inference_mode=True,
        )
    outputs = prefill["outputs"]
    cache = outputs.past_key_values
    if cache is None:
        raise RuntimeError("real G2 causal prefill did not return a KV cache")
    logits = model.lm_head(outputs.last_hidden_state[:, -1])
    generated: list[int] = []
    open_box_coordinates: int | None = None
    forced_box_end = 0
    axes = int(prefill["prepared"].position_ids.shape[0])
    batch_size = int(prefill["prepared"].inputs_embeds.shape[0])
    rope_deltas = getattr(model.multimodal_model, "rope_deltas", None)
    if rope_deltas is not None:
        rope_deltas = rope_deltas.detach().clone()
    for _ in range(max_new_tokens):
        # Apply the same single-box structural boundary used by the official
        # GroundAnything referring profile: coordinates are never invented or
        # reordered, but a malformed/incomplete language continuation cannot
        # run past the four-coordinate payload without emitting box_end.
        if open_box_coordinates is not None:
            if open_box_coordinates >= 4:
                token = torch.full(
                    (logits.shape[0], 1), structural["box_end"],
                    dtype=torch.long, device=logits.device,
                )
                forced_box_end += 1
            else:
                constrained = logits.clone()
                lo, hi = coordinate_range
                constrained[:, :lo] = -torch.inf
                constrained[:, hi + 1:] = -torch.inf
                token = constrained.argmax(dim=-1, keepdim=True)
        else:
            token = logits.argmax(dim=-1, keepdim=True)
        value = int(token[0, 0])
        generated.append(value)
        if open_box_coordinates is None and value == structural["box_start"]:
            open_box_coordinates = 0
        elif open_box_coordinates is not None and value == structural["box_end"]:
            open_box_coordinates = None
        elif open_box_coordinates is not None and coordinate_range[0] <= value <= coordinate_range[1]:
            open_box_coordinates += 1
        if value in stop_ids:
            break
        position = int(cache.get_seq_length())
        positions = model._generation_position_ids(
            torch.tensor([position], device=token.device, dtype=torch.long),
            axes, batch_size, rope_deltas,
        )
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model.language_model(
                input_ids=token,
                attention_mask=None,
                position_ids=model._native_position_ids(positions),
                past_key_values=cache,
                use_cache=True,
                cache_position=torch.tensor([position], device=token.device, dtype=torch.long),
                return_dict=True,
            )
            logits = model.lm_head(outputs.last_hidden_state[:, -1])
        cache = outputs.past_key_values
    return {"token_ids": generated, "prefill": prefill, "cache_tokens": int(cache.get_seq_length()),
            "grammar_forced_box_end": forced_box_end}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--previous-image", required=True)
    parser.add_argument("--current-image", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    runtime_root = Path(args.runtime_root).resolve()
    os.environ.setdefault("GAM_DLM_FLEX_CUDAGRAPHS", "0")
    os.environ["LOCATEMOT_GROUNDANYTHING_ROOT"] = str(Path(args.source_root).resolve())
    sys.path[:0] = [str(runtime_root), str(Path(args.source_root).resolve())]
    from tools.groundjev.g2.compat import patch_transformers_causal_mask
    from tools.groundjev.g2.train_g2_dlm import load_model, _strict_load_trainable_state
    from train.dlm.data import DLMDataCollator

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real G2 memory prefill requires CUDA")
    patch_transformers_causal_mask()
    start = time.perf_counter()
    model, processor, _, audit = load_model(Path(args.checkpoint).resolve(), device)
    saved = torch.load(Path(args.adapter).resolve(), map_location="cpu", weights_only=False)
    _strict_load_trainable_state(model, saved["model_trainable_state"])
    adapter_step = int(saved["step"])
    model.set_padding_free_packing(False)
    model.gradient_checkpointing_disable()
    model.eval()
    collator = DLMDataCollator(processor, max_length=32768, max_joint_length=65536,
                               model_family="groundinganything_qwen3")
    query = "person"
    previous_boxes = [(251, 496, 305, 617), (659, 503, 716, 632)]
    current_boxes = [(251, 494, 305, 615), (659, 501, 716, 630)]
    previous = _prepare(model, collator, _row(Path(args.previous_image), query, previous_boxes), device)
    current = _prepare(model, collator, _row(Path(args.current_image), query, current_boxes), device)
    hidden = int(current["embeds"].shape[-1])
    projector = IdentityMemoryTokenProjector(previous["image_tokens"].shape[-1], hidden, hidden_dim=512).to(device)
    # Two causal previous-frame region summaries provide a real correct/shuffled
    # pair without reading current-frame labels or running another current image
    # encoder.  The projector is untrained and is explicitly recorded as such.
    midpoint = max(1, int(previous["image_tokens"].shape[0] // 2))
    previous_memory = torch.stack((previous["image_tokens"][:midpoint].mean(0),
                                   previous["image_tokens"][midpoint:].mean(0)))
    memory_tokens = projector({0: previous_memory[0], 1: previous_memory[1]}, ordered_ids=[0, 1])
    correct = memory_tokens[:1]
    shuffled = memory_tokens[1:2]
    tokenizer = processor.tokenizer
    structural = {
        "box_start": int(tokenizer.convert_tokens_to_ids("<|box_start|>")),
        "box_end": int(tokenizer.convert_tokens_to_ids("<|box_end|>")),
    }
    coordinate_range = (
        int(tokenizer.convert_tokens_to_ids("<0>")),
        int(tokenizer.convert_tokens_to_ids("<999>")),
    )
    if structural["box_start"] < 0 or structural["box_end"] < 0 or coordinate_range[0] < 0:
        raise RuntimeError("G2 tokenizer lacks the official structured grounding token contract")
    stop_ids = {
        int(tokenizer.convert_tokens_to_ids("<|box_end|>")),
        int(tokenizer.convert_tokens_to_ids("<|im_end|>")),
    }
    conditions = {
        "C0_disabled": (torch.empty((0, hidden), device=device), False),
        "C2_correct_previous_memory": (correct, True),
        "C2_shuffled_previous_memory": (shuffled, True),
    }
    results: dict[str, Any] = {}
    for name, (memory, enabled) in conditions.items():
        generated = _generate(model, current, memory, enabled=enabled,
                              max_new_tokens=args.max_new_tokens, stop_ids=stop_ids,
                              structural=structural, coordinate_range=coordinate_range)
        text = tokenizer.decode(generated["token_ids"], skip_special_tokens=False)
        recovered, report = repair_gam_bbox_output(text)
        results[name] = {
            "memory_enabled": enabled,
            "memory_token_count": int(memory.shape[0]),
            "generated_token_count": len(generated["token_ids"]),
            "generated_text": text,
            "recovered_pairs": [[label, list(box)] for label, box in recovered],
            "repair_report": report,
            "prefill_calls": int(generated["prefill"]["prefill_calls"]),
            "cache_tokens": generated["cache_tokens"],
            "grammar_forced_box_end": generated["grammar_forced_box_end"],
            "first_token_id": int(generated["token_ids"][0]) if generated["token_ids"] else None,
        }
    receipt = {
        "status": "PHASE_C_REAL_G2_MEMORY_PREFILL_COMPARISON_COMPLETE",
        "project_root": "/data2/user/LocateMOT",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "adapter_step": adapter_step,
        "base_model_sha256": args.model_sha256,
        "previous_image": str(Path(args.previous_image).resolve()),
        "current_image": str(Path(args.current_image).resolve()),
        "conditions": results,
        "visual_embed_calls": 2,
        "current_visual_embed_reused_across_conditions": True,
        "independent_vlm_instances": 0,
        "memory_projector": "untrained IdentityMemoryTokenProjector; previous-frame image-token summaries",
        "structured_termination": "official single-box four-coordinate boundary; no coordinates synthesized",
        "g2_trainable_parameters": audit.get("trainable_parameters"),
        "elapsed_seconds": time.perf_counter() - start,
        "formal_g2_run_untouched": True,
        "scientific_claim": "real G2 causal prefill and parser-contract evidence only; no tracking or grounding gain claim",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
