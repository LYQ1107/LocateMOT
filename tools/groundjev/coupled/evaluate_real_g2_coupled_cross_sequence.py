#!/usr/bin/env python3
"""Evaluate a trained coupled-memory pilot on a different MOT sequence.

The script never updates parameters.  It loads the frozen G2 adapter and the
small Phase-C coupled checkpoint, encodes each real frame once, and compares
memory-disabled grounding with a previous-frame memory token.  It also reads
the typed JEV MATCH/Yes-No output over the same shared image summaries.  The
result is a cross-sequence diagnostic, not an official MOT benchmark.
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

from locatemot.groundjev.coupled.g2_jev_readout import G2JEVReadout  # noqa: E402
from locatemot.groundjev.coupled.memory_tokens import (  # noqa: E402
    IdentityMemoryTokenProjector,
    MemoryTokenProjector,
)
from locatemot.groundjev.native_output_repair import repair_gam_bbox_output  # noqa: E402
from tools.groundjev.coupled.real_g2_memory_prefill import (  # noqa: E402
    _generate,
    _prepare,
    _row,
)


def _read_boxes(path: Path, *, target_id: int, frame_start: int, frame_count: int,
                width: int, height: int) -> dict[int, tuple[int, int, int, int]]:
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
        boxes[frame] = (
            round(max(0.0, min(999.0, x / width * 999.0))),
            round(max(0.0, min(999.0, y / height * 999.0))),
            round(max(0.0, min(999.0, (x + w) / width * 999.0))),
            round(max(0.0, min(999.0, (y + h) / height * 999.0))),
        )
    if missing := sorted(wanted.difference(boxes)):
        raise RuntimeError(f"target ID {target_id} missing from frames {missing}")
    return boxes


def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--g2-adapter", required=True)
    parser.add_argument("--coupled-checkpoint", required=True)
    parser.add_argument("--sequence-root", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--frame-start", type=int, default=13)
    parser.add_argument("--frame-count", type=int, default=5)
    parser.add_argument("--target-id", type=int, default=17)
    parser.add_argument("--width", type=int, default=1654)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
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
        raise RuntimeError("cross-sequence coupled evaluation requires CUDA")
    patch_transformers_causal_mask()
    start = time.perf_counter()
    frames = [args.frame_start + index for index in range(args.frame_count)]
    sequence_root = Path(args.sequence_root).resolve()
    boxes = _read_boxes(Path(args.gt).resolve(), target_id=args.target_id,
                        frame_start=args.frame_start, frame_count=args.frame_count,
                        width=args.width, height=args.height)
    images = [sequence_root / "img1" / f"{frame:06d}.jpg" for frame in frames]
    if not all(image.is_file() for image in images):
        raise RuntimeError("one or more cross-sequence images are missing")

    model, processor, _, audit = load_model(Path(args.checkpoint).resolve(), device)
    saved = torch.load(Path(args.g2_adapter).resolve(), map_location="cpu", weights_only=False)
    _strict_load_trainable_state(model, saved["model_trainable_state"])
    g2_adapter_step = int(saved["step"])
    coupled = torch.load(Path(args.coupled_checkpoint).resolve(), map_location="cpu", weights_only=False)
    model.set_padding_free_packing(False)
    model.gradient_checkpointing_disable()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    collator = DLMDataCollator(processor, max_length=32768, max_joint_length=65536,
                               model_family="groundinganything_qwen3")
    prepared = [_prepare(model, collator, _row(image, "person", [boxes[frame]]), device)
                for image, frame in zip(images, frames)]
    summaries = [item["image_tokens"].float().mean(dim=0) for item in prepared]
    input_dim = int(prepared[0]["image_tokens"].shape[-1])
    hidden = int(prepared[0]["embeds"].shape[-1])
    prompt_projector = IdentityMemoryTokenProjector(input_dim, hidden, hidden_dim=512).to(device)
    prompt_projector.load_state_dict(coupled["prompt_memory_projector"])
    jev_memory_projector = MemoryTokenProjector(input_dim, 256).to(device)
    jev_memory_projector.load_state_dict(coupled["jev_memory_projector"])
    readout = G2JEVReadout(input_dim, hidden_dim=256).to(device)
    readout.load_state_dict(coupled["typed_readout"])
    prompt_projector.eval(); jev_memory_projector.eval(); readout.eval()

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
    rows: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        conditions: dict[str, tuple[torch.Tensor, bool]] = {
            "C0_disabled": (torch.empty((0, hidden), device=device), False),
        }
        if index:
            memory = prompt_projector({args.target_id: summaries[index - 1]}, ordered_ids=[args.target_id])
            conditions["C2_previous_frame_memory"] = (memory, True)
        row: dict[str, Any] = {"frame": frame, "target_box_0_999": list(boxes[frame]),
                               "conditions": {}, "jev": None}
        for name, (memory, enabled) in conditions.items():
            generated = _generate(model, prepared[index], memory, enabled=enabled,
                                  max_new_tokens=160, stop_ids=stop_ids,
                                  structural=structural, coordinate_range=coordinate_range)
            text = tokenizer.decode(generated["token_ids"], skip_special_tokens=False)
            recovered, report = repair_gam_bbox_output(text)
            output_box = list(recovered[0][1]) if recovered else None
            row["conditions"][name] = {
                "memory_enabled": enabled,
                "memory_token_count": int(memory.shape[0]),
                "generated_text": text,
                "recovered_pairs": [[label, list(box)] for label, box in recovered],
                "iou_vs_target": _iou(output_box, list(boxes[frame])) if output_box else 0.0,
                "repair_status": report["status"],
                "grammar_forced_box_end": generated["grammar_forced_box_end"],
            }
        if index:
            with torch.no_grad():
                slot = jev_memory_projector(summaries[index - 1].unsqueeze(0))
                slots = readout.make_slot_tokens(slot, ["NEW", "NO_MATCH", "REJECT"])
                output = readout(summaries[index].unsqueeze(0), slots, action_start=1)
                probs = torch.softmax(output["identity_logits"][0], dim=-1).cpu().tolist()
                row["jev"] = {
                    "identity_options": [str(args.target_id), "NEW", "NO_MATCH", "REJECT"],
                    "identity_probabilities": probs,
                    "predicted_option": int(output["identity_logits"][0].argmax().cpu()),
                    "yes_probability": float(torch.sigmoid(output["yesno_logits"][0]).cpu()),
                    "admit_probability": float(torch.sigmoid(output["admit_logits"][0]).cpu()),
                    "lifecycle_option": int(output["lifecycle_logits"][0].argmax().cpu()),
                }
        rows.append(row)

    result = {
        "status": "PHASE_C_REAL_G2_COUPLED_CROSS_SEQUENCE_DIAGNOSTIC_COMPLETE",
        "project_root": str(PROJECT_ROOT),
        "sequence": sequence_root.name,
        "frames": frames,
        "target_id": args.target_id,
        "scope": "MOT20 TRAIN cross-sequence diagnostic relative to MOT20-03 coupling pilot; not official independent benchmark",
        "g2_checkpoint": str(Path(args.checkpoint).resolve()),
        "g2_adapter": str(Path(args.g2_adapter).resolve()),
        "g2_adapter_step": g2_adapter_step,
        "coupled_checkpoint": str(Path(args.coupled_checkpoint).resolve()),
        "coupled_training_scope": "MOT20-03 frames 82-85; frame 86 same-video temporal holdout",
        "base_model_sha256": args.model_sha256,
        "visual_embed_calls": len(prepared),
        "one_visual_encode_per_frame": True,
        "independent_vlm_instances": 0,
        "rows": rows,
        "mean_iou_by_condition": {
            name: sum(row["conditions"][name]["iou_vs_target"] for row in rows if name in row["conditions"])
            / max(sum(name in row["conditions"] for row in rows), 1)
            for name in ("C0_disabled", "C2_previous_frame_memory")
        },
        "formal_g2_run_untouched": True,
        "scientific_claim": "cross-sequence coupled-memory/JEV diagnostic only; no official MOT metric or generalization claim",
        "elapsed_seconds": time.perf_counter() - start,
        "g2_trainable_parameters": audit.get("trainable_parameters"),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
