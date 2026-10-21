#!/usr/bin/env python3
"""Replay a short real MOT clip through the coupled G2 causal boundary.

This is a bounded engineering diagnostic, not a benchmark.  Each frame is
encoded once by the official G2 visual path.  The next frame is then decoded
with memory disabled and with one causal summary from the immediately prior
frame.  The summary/projector is intentionally untrained, so the receipt
measures contract execution and output sensitivity rather than a tracking
gain.  Ground-truth boxes are used only to form the collator's answer suffix;
the prompt is sliced before that suffix before any language prefill runs.
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
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.groundjev.coupled.real_g2_memory_prefill import (  # noqa: E402
    _generate,
    _prepare,
    _row,
)
from locatemot.groundjev.coupled.memory_tokens import IdentityMemoryTokenProjector  # noqa: E402
from locatemot.groundjev.native_output_repair import repair_gam_bbox_output  # noqa: E402


def _read_gt(path: Path, *, target_id: int, frame_start: int, frame_count: int,
             width: int, height: int) -> dict[int, tuple[int, int, int, int]]:
    """Return one legal target box per replay frame in the official 0..999 scale."""
    wanted = set(range(frame_start, frame_start + frame_count))
    boxes: dict[int, tuple[int, int, int, int]] = {}
    for line in path.read_text().splitlines():
        parts = line.split(",")
        if len(parts) < 6:
            continue
        frame, identity = int(parts[0]), int(parts[1])
        if frame not in wanted or identity != target_id:
            continue
        x, y, w, h = (float(parts[i]) for i in range(2, 6))
        left = round(max(0.0, min(999.0, x / width * 999.0)))
        top = round(max(0.0, min(999.0, y / height * 999.0)))
        right = round(max(0.0, min(999.0, (x + w) / width * 999.0)))
        bottom = round(max(0.0, min(999.0, (y + h) / height * 999.0)))
        boxes[frame] = (left, top, right, bottom)
    missing = sorted(wanted.difference(boxes))
    if missing:
        raise RuntimeError(f"target id {target_id} is missing from frames {missing}")
    return boxes


def _summary(image_tokens: torch.Tensor) -> torch.Tensor:
    if image_tokens.ndim != 2 or image_tokens.shape[0] == 0:
        raise RuntimeError("G2 image-token summary cannot be computed from an empty tensor")
    return image_tokens.float().mean(dim=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--sequence-root", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--frame-start", type=int, default=82)
    parser.add_argument("--frame-count", type=int, default=4)
    parser.add_argument("--target-id", type=int, default=27)
    parser.add_argument("--width", type=int, default=1173)
    parser.add_argument("--height", type=int, default=880)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    runtime_root = Path(args.runtime_root).resolve()
    source_root = Path(args.source_root).resolve()
    os.environ.setdefault("GAM_DLM_FLEX_CUDAGRAPHS", "0")
    os.environ["LOCATEMOT_GROUNDANYTHING_ROOT"] = str(source_root)
    sys.path[:0] = [str(runtime_root), str(source_root)]
    from tools.groundjev.g2.compat import patch_transformers_causal_mask
    from tools.groundjev.g2.train_g2_dlm import load_model, _strict_load_trainable_state
    from train.dlm.data import DLMDataCollator

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("real G2 multi-frame replay requires CUDA")
    patch_transformers_causal_mask()
    start = time.perf_counter()
    sequence_root = Path(args.sequence_root).resolve()
    gt_path = Path(args.gt).resolve()
    boxes = _read_gt(gt_path, target_id=args.target_id, frame_start=args.frame_start,
                     frame_count=args.frame_count, width=args.width, height=args.height)
    frames = [args.frame_start + index for index in range(args.frame_count)]
    images = [sequence_root / "img1" / f"{frame:06d}.jpg" for frame in frames]
    if not all(image.is_file() for image in images):
        raise RuntimeError("one or more replay images are missing")

    model, processor, _, audit = load_model(Path(args.checkpoint).resolve(), device)
    saved = torch.load(Path(args.adapter).resolve(), map_location="cpu", weights_only=False)
    _strict_load_trainable_state(model, saved["model_trainable_state"])
    adapter_step = int(saved["step"])
    model.set_padding_free_packing(False)
    model.gradient_checkpointing_disable()
    model.eval()
    collator = DLMDataCollator(processor, max_length=32768, max_joint_length=65536,
                               model_family="groundinganything_qwen3")
    tokenizer = processor.tokenizer
    structural = {
        "box_start": int(tokenizer.convert_tokens_to_ids("<|box_start|>")),
        "box_end": int(tokenizer.convert_tokens_to_ids("<|box_end|>")),
    }
    coordinate_range = (
        int(tokenizer.convert_tokens_to_ids("<0>")),
        int(tokenizer.convert_tokens_to_ids("<999>")),
    )
    stop_ids = {structural["box_end"], int(tokenizer.convert_tokens_to_ids("<|im_end|>"))}
    if min(structural.values()) < 0 or min(coordinate_range) < 0:
        raise RuntimeError("G2 tokenizer lacks the official structured grounding contract")

    prepared: list[dict[str, Any]] = []
    query = "person"
    for image, frame in zip(images, frames):
        prepared.append(_prepare(
            model, collator, _row(image, query, [boxes[frame]]), device))
    hidden = int(prepared[0]["embeds"].shape[-1])
    input_dim = int(prepared[0]["image_tokens"].shape[-1])
    if any(int(item["image_tokens"].shape[-1]) != input_dim for item in prepared):
        raise RuntimeError("G2 image token dimensions changed across replay frames")
    projector = IdentityMemoryTokenProjector(input_dim, hidden, hidden_dim=512).to(device)

    rows: list[dict[str, Any]] = []
    for index, (frame, image, item) in enumerate(zip(frames, images, prepared)):
        conditions: dict[str, tuple[torch.Tensor, bool]] = {
            "C0_disabled": (torch.empty((0, hidden), device=device), False),
        }
        if index:
            previous_summary = _summary(prepared[index - 1]["image_tokens"])
            memory = projector({args.target_id: previous_summary}, ordered_ids=[args.target_id])
            conditions["C2_previous_frame_memory"] = (memory, True)
        frame_result: dict[str, Any] = {
            "frame": frame,
            "image": str(image),
            "target_id": args.target_id,
            "target_box_0_999": list(boxes[frame]),
            "previous_frame": frames[index - 1] if index else None,
            "conditions": {},
        }
        for name, (memory, enabled) in conditions.items():
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            condition_start = time.perf_counter()
            generated = _generate(model, item, memory, enabled=enabled,
                                  max_new_tokens=args.max_new_tokens, stop_ids=stop_ids,
                                  structural=structural, coordinate_range=coordinate_range)
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            text = tokenizer.decode(generated["token_ids"], skip_special_tokens=False)
            recovered, report = repair_gam_bbox_output(text)
            frame_result["conditions"][name] = {
                "memory_enabled": enabled,
                "memory_token_count": int(memory.shape[0]),
                "generated_token_count": len(generated["token_ids"]),
                "generated_text": text,
                "recovered_pairs": [[label, list(box)] for label, box in recovered],
                "repair_report": report,
                "prefill_calls": int(generated["prefill"]["prefill_calls"]),
                "cache_tokens": generated["cache_tokens"],
                "grammar_forced_box_end": generated["grammar_forced_box_end"],
                "elapsed_seconds": time.perf_counter() - condition_start,
            }
        rows.append(frame_result)

    receipt = {
        "status": "PHASE_C_REAL_G2_MULTIFRAME_MEMORY_REPLAY_COMPLETE",
        "project_root": str(PROJECT_ROOT),
        "sequence": sequence_root.name,
        "frame_start": args.frame_start,
        "frame_count": args.frame_count,
        "frames": frames,
        "target_id": args.target_id,
        "ground_truth_scope": "MOT20 TRAIN diagnostic; boxes only form answer suffix and are excluded from prompt prefill",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "adapter_step": adapter_step,
        "base_model_sha256": args.model_sha256,
        "visual_embed_calls": len(prepared),
        "one_visual_encode_per_frame": True,
        "current_visual_embed_reused_across_conditions": True,
        "independent_vlm_instances": 0,
        "memory_projector": "untrained IdentityMemoryTokenProjector; previous-frame whole-image summaries",
        "structured_termination": "official single-box four-coordinate boundary; no coordinates synthesized",
        "rows": rows,
        "g2_trainable_parameters": audit.get("trainable_parameters"),
        "elapsed_seconds": time.perf_counter() - start,
        "formal_g2_run_untouched": True,
        "scientific_claim": "real multi-frame G2 causal prefill and memory-reuse evidence only; no tracking or grounding gain claim",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
