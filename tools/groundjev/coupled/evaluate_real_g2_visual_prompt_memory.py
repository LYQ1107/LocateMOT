#!/usr/bin/env python3
"""Evaluate GroundAnything's native visual-prompt path as causal identity memory."""
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

from locatemot.groundjev.native_output_repair import repair_gam_bbox_output  # noqa: E402
from tools.groundjev.coupled.real_g2_memory_prefill import (  # noqa: E402
    _answer,
    _generate,
    _prepare,
)
from tools.groundjev.coupled.train_real_g2_coupled_memory_jev import _iou, _read_boxes  # noqa: E402


def _visual_row(image: Path, reference_box: tuple[int, int, int, int], target_box: tuple[int, int, int, int]) -> dict[str, Any]:
    from grounding_anything.protocol import build_visual_prompt
    return {
        "id": f"groundjev_visual_memory_{image.stem}",
        "messages": [
            {"role": "user", "content": f"<image>\n{build_visual_prompt([reference_box])}", "loss": None},
            {"role": "assistant", "content": _answer("person", [target_box]), "loss": None},
        ],
        "images": [{"path": str(image), "bytes": None}],
        "source": "GroundJEV:real_visual_prompt_memory",
    }


def _dense_row(image: Path, target_box: tuple[int, int, int, int]) -> dict[str, Any]:
    return {
        "id": f"groundjev_visual_memory_dense_{image.stem}",
        "messages": [
            {"role": "user", "content": "<image>\nLocate all the instances that match the following categories: person.", "loss": None},
            {"role": "assistant", "content": _answer("person", [target_box]), "loss": None},
        ],
        "images": [{"path": str(image), "bytes": None}],
        "source": "GroundJEV:real_visual_prompt_memory",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--g2-adapter", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:7")
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
        raise RuntimeError("visual-prompt memory evaluation requires CUDA")
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
    collator = DLMDataCollator(processor, max_length=32768, max_joint_length=65536,
                               model_family="groundinganything_qwen3")
    specs = [
        ("MOT20-03", Path("/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-03"),
         Path("/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-03/gt/gt.txt"), 82, 27, 1173, 880),
        ("MOT20-05", Path("/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-05"),
         Path("/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-05/gt/gt.txt"), 13, 17, 1654, 1080),
    ]
    tokenizer = processor.tokenizer
    structural = {"box_start": int(tokenizer.convert_tokens_to_ids("<|box_start|>")),
                  "box_end": int(tokenizer.convert_tokens_to_ids("<|box_end|>"))}
    coordinate_range = (int(tokenizer.convert_tokens_to_ids("<0>")), int(tokenizer.convert_tokens_to_ids("<999>")))
    stop_ids = {structural["box_end"], int(tokenizer.convert_tokens_to_ids("<|im_end|>"))}
    results = {}
    visual_embed_calls = 0
    for name, root, gt, frame_start, target_id, width, height in specs:
        frames = list(range(frame_start, frame_start + 5))
        boxes = _read_boxes(gt, target_id=target_id, frame_start=frame_start, frame_count=5,
                            width=width, height=height)
        rows = []
        for index, frame in enumerate(frames):
            image = root / "img1" / f"{frame:06d}.jpg"
            row = _dense_row(image, boxes[frame]) if index == 0 else _visual_row(image, boxes[frames[index - 1]], boxes[frame])
            prepared = _prepare(model, collator, row, device)
            visual_embed_calls += 1
            generated = _generate(model, prepared, torch.empty((0, prepared["embeds"].shape[-1]), device=device),
                                  enabled=False, max_new_tokens=160, stop_ids=stop_ids,
                                  structural=structural, coordinate_range=coordinate_range)
            text = tokenizer.decode(generated["token_ids"], skip_special_tokens=False)
            recovered, report = repair_gam_bbox_output(text)
            output_box = list(recovered[0][1]) if recovered else None
            rows.append({"frame": frame, "condition": "C0_dense" if index == 0 else "C2_visual_prompt_previous_box",
                         "reference_box": list(boxes[frames[index - 1]]) if index else None,
                         "target_box_0_999": list(boxes[frame]), "generated_text": text,
                         "recovered_pairs": [[label, list(box)] for label, box in recovered],
                         "iou_vs_target": _iou(output_box, list(boxes[frame])) if output_box else 0.0,
                         "repair_report": report, "grammar_forced_box_end": generated["grammar_forced_box_end"]})
        values = [row["iou_vs_target"] for row in rows[1:]]
        results[name] = {"rows": rows, "c2_mean_iou": sum(values) / max(len(values), 1)}
    receipt = {
        "status": "PHASE_C_REAL_G2_VISUAL_PROMPT_MEMORY_COMPLETE",
        "project_root": str(PROJECT_ROOT), "g2_adapter": str(Path(args.g2_adapter).resolve()),
        "adapter_step": int(saved["step"]), "base_model_sha256": args.model_sha256,
        "g2_frozen": True, "formal_g2_run_untouched": True,
        "native_visual_prompt_protocol": "GroundAnything build_visual_prompt with previous legal box",
        "visual_embed_calls": visual_embed_calls, "independent_vlm_instances": 0,
        "results": results, "g2_trainable_parameters": audit.get("trainable_parameters"),
        "elapsed_seconds": time.perf_counter() - start,
        "scientific_claim": "bounded real visual-prompt memory diagnostic; no official MOT/RMOT/OVMOT metric claim",
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
