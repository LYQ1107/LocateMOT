#!/usr/bin/env python3
"""Run a bounded dense G2 check through GroundAnything's official DLM decoder.

The earlier Phase-C diagnostics used a causal KV loop so that memory tokens
could be inserted explicitly.  That loop is not the released GroundAnything
decoder: G2 training uses the hybrid DLM/MDM path.  This evaluator therefore
keeps the frozen G2 adapter and real image inputs unchanged, but dispatches
generation through ``DLMInferenceEngine``/``HierarchyBlockDecoder`` with the
same structured termination contract used by the official server.

This is a decoder-contract diagnostic.  It does not claim a tracking gain and
does not modify the protected two-rank G2 training run.
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
# Keep this checkout first: the runtime tree has a regular ``locatemot``
# package, while this project intentionally exposes a namespace package with
# the shared output-repair utilities.  Reversing the order hides the local
# parser and produces a misleading import failure before model loading.
sys.path[:0] = [str(PROJECT_ROOT), str(RUNTIME_ROOT), str(SOURCE_ROOT)]

from locatemot.groundjev.native_output_repair import repair_gam_bbox_output  # noqa: E402
from tools.groundjev.coupled.evaluate_real_g2_dense_multibox import (  # noqa: E402
    _greedy_recall,
    _read_all_boxes,
    _row,
)


def _prepare_prompt(model: Any, collator: Any, row: dict[str, Any], device: torch.device) -> dict[str, Any]:
    batch = collator([row])
    tensor_batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }
    visual_dtype = next(model.base_model.model.visual.parameters()).dtype
    for key in ("pixel_values", "pixel_values_videos"):
        if key in tensor_batch:
            tensor_batch[key] = tensor_batch[key].to(dtype=visual_dtype)
    labels = tensor_batch["labels"]
    target = labels.ne(-100).nonzero(as_tuple=False)
    if target.numel() == 0:
        raise RuntimeError("collated row has no supervised assistant tokens")
    answer_start = int(target[0, 1])
    result: dict[str, Any] = {
        "input_ids": tensor_batch["input_ids"][:, :answer_start],
        "attention_mask": tensor_batch["attention_mask"][:, :answer_start],
        "mm_token_type_ids": tensor_batch["mm_token_type_ids"][:, :answer_start],
        "answer_start": answer_start,
    }
    for key in (
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
    ):
        if key in tensor_batch:
            result[key] = tensor_batch[key]
    return result


def _token_contract(tokenizer: Any) -> dict[str, Any]:
    structural = {
        name: int(tokenizer.convert_tokens_to_ids(token))
        for name, token in (
            ("object_ref_start", "<|object_ref_start|>"),
            ("object_ref_end", "<|object_ref_end|>"),
            ("box_start", "<|box_start|>"),
            ("box_end", "<|box_end|>"),
            ("quad_start", "<|quad_start|>"),
            ("quad_end", "<|quad_end|>"),
        )
    }
    coordinate_min = int(tokenizer.convert_tokens_to_ids("<0>"))
    coordinate_max = int(tokenizer.convert_tokens_to_ids("<999>"))
    separators = tokenizer.encode(",", add_special_tokens=False)
    if any(value < 0 for value in structural.values()):
        raise RuntimeError(f"missing structured token IDs: {structural}")
    if not 0 <= coordinate_min <= coordinate_max or len(separators) != 1:
        raise RuntimeError(
            f"invalid coordinate contract: range={(coordinate_min, coordinate_max)} separators={separators}"
        )
    eos = tokenizer.eos_token_id
    stop_ids = tuple(
        dict.fromkeys(
            int(value)
            for value in (eos, tokenizer.convert_tokens_to_ids("<|im_end|>"))
            if value is not None and int(value) >= 0
        )
    )
    return {
        "structural_token_ids": structural,
        "coordinate_token_id_range": (coordinate_min, coordinate_max),
        "coordinate_separator_token_ids": tuple(int(value) for value in separators),
        "stop_token_ids": stop_ids,
        "semantic_stop_token_ids": (
            structural["box_end"],
            structural["quad_end"],
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--g2-adapter", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--mode", choices=("hierarchy_dynamic", "hierarchy_step"), default="hierarchy_dynamic")
    parser.add_argument("--frames-per-sequence", type=int, default=1)
    args = parser.parse_args()

    os.environ.setdefault("GAM_DLM_FLEX_CUDAGRAPHS", "0")
    os.environ["LOCATEMOT_GROUNDANYTHING_ROOT"] = str(SOURCE_ROOT)
    from tools.groundjev.g2.compat import patch_transformers_causal_mask
    from tools.groundjev.g2.train_g2_dlm import _strict_load_trainable_state, load_model
    from train.dlm.data import DLMDataCollator
    from infer.dlm import DLMInferenceConfig, DLMInferenceEngine

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("official DLM dense evaluation requires CUDA")
    patch_transformers_causal_mask()
    start = time.perf_counter()
    model, processor, _, audit = load_model(Path(args.checkpoint).resolve(), device)
    saved = torch.load(Path(args.g2_adapter).resolve(), map_location="cpu", weights_only=False)
    _strict_load_trainable_state(model, saved["model_trainable_state"])
    model.set_padding_free_packing(False)
    model.gradient_checkpointing_disable()
    model.eval()
    model.config.use_cache = True
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    collator = DLMDataCollator(
        processor,
        max_length=32768,
        max_joint_length=65536,
        model_family="groundinganything_qwen3",
    )
    tokenizer = processor.tokenizer
    contract = _token_contract(tokenizer)
    config = DLMInferenceConfig(
        mode=args.mode,
        block_size=32,
        sub_block_size=8,
        denoise_steps=8,
        confidence_threshold=0.9,
        acceptance_policy="confidence",
        entropy_threshold=0.8,
        enforce_structured_termination=True,
        structured_max_coordinate_groups=0,
        commit_structural_boundaries=False,
    )
    engine = DLMInferenceEngine(model)
    specs = [
        (
            "MOT20-03",
            Path("/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-03"),
            Path("/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-03/gt/gt.txt"),
            82,
            1173,
            880,
        ),
        (
            "MOT20-05",
            Path("/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-05"),
            Path("/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-05/gt/gt.txt"),
            13,
            1654,
            1080,
        ),
    ]
    results: dict[str, Any] = {}
    visual_calls = 0
    for name, root, gt, first_frame, width, height in specs:
        frames = list(range(first_frame, first_frame + max(1, args.frames_per_sequence)))
        boxes = _read_all_boxes(gt, frames=frames, width=width, height=height)
        rows: list[dict[str, Any]] = []
        for frame in frames:
            prepared = _prepare_prompt(
                model,
                collator,
                _row(root / "img1" / f"{frame:06d}.jpg", boxes[frame]),
                device,
            )
            visual_calls += 1
            generation_args = {
                **contract,
                "input_ids": prepared["input_ids"],
                "attention_mask": prepared["attention_mask"],
                "max_new_tokens": args.max_new_tokens,
                "mm_token_type_ids": prepared.get("mm_token_type_ids"),
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
            # G2 restores the vision projector in FP32 while the native
            # multimodal stream is BF16.  The official training/inference
            # boundary enters CUDA autocast before ``_embed_clean``; preserve
            # that boundary around the DLM engine as well.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_ids = engine.generate(config, **generation_args)
            generated_ids = output_ids[0].tolist()
            text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            recovered, repair_report = repair_gam_bbox_output(text)
            predicted = [list(box) for _label, box in recovered]
            rows.append(
                {
                    "frame": frame,
                    "target_boxes": [list(box) for box in boxes[frame]],
                    "generated_text": text,
                    "generated_token_count": len(generated_ids),
                    "recovered_pairs": [[label, list(box)] for label, box in recovered],
                    "recall": _greedy_recall(predicted, boxes[frame]),
                    "repair_report": repair_report,
                    "generation_stats": dict(getattr(model, "_last_generation_stats", {})),
                }
            )
        results[name] = {
            "rows": rows,
            "mean_recall_at_iou_0_5": sum(
                row["recall"]["recall_at_iou_0_5"] for row in rows
            )
            / len(rows),
        }

    receipt = {
        "status": "PHASE_C_REAL_G2_OFFICIAL_DLM_DENSE_COMPLETE",
        "project_root": str(PROJECT_ROOT),
        "g2_checkpoint": str(Path(args.checkpoint).resolve()),
        "g2_adapter": str(Path(args.g2_adapter).resolve()),
        "adapter_step": int(saved["step"]),
        "base_model_sha256": args.model_sha256,
        "decoder": {
            "engine": "infer.dlm.DLMInferenceEngine",
            "mode": args.mode,
            "config": config.__dict__,
            "contract": contract,
        },
        "visual_embed_calls": visual_calls,
        "independent_vlm_instances": 0,
        "g2_frozen": True,
        "formal_g2_run_untouched": True,
        "results": results,
        "g2_trainable_parameters": audit.get("trainable_parameters"),
        "elapsed_seconds": time.perf_counter() - start,
        "scientific_claim": "official G2 DLM decoder contract diagnostic only; no tracking or coupling gain claim",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
