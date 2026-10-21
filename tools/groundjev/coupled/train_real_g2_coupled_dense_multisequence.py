#!/usr/bin/env python3
"""Train a bounded shared-backbone coupling adapter on valid dense G2 rows.

Unlike the earlier target-only pilots, every row here uses the complete
GroundAnything dense answer for the frame (all legal person boxes inside one
``box_start``/``box_end`` group).  The frozen G2 backbone supplies one visual
summary per frame; a small identity/box-state projector conditions the causal
answer likelihood and a typed JEV readout is trained on the same summaries.
This is a finite coupling experiment, not a full MOT benchmark.
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

from locatemot.groundjev.coupled.g2_jev_readout import G2JEVReadout  # noqa: E402
from locatemot.groundjev.coupled.losses import structured_groundjev_losses  # noqa: E402
from locatemot.groundjev.coupled.memory_tokens import (  # noqa: E402
    IdentityBoxStateTokenProjector,
    MemoryTokenProjector,
)
from tools.groundjev.coupled.evaluate_real_g2_dense_multibox import (  # noqa: E402
    _read_all_boxes,
)
from tools.groundjev.coupled.train_real_g2_coupled_memory_jev import (  # noqa: E402
    _jev_forward,
    _jev_metrics,
    _mean_token,
    _teacher_forced_loss,
)
from tools.groundjev.coupled.real_g2_memory_prefill import _prepare, _row  # noqa: E402


def _sequence_spec(args: argparse.Namespace, prefix: str) -> dict[str, Any]:
    return {
        "name": getattr(args, f"{prefix}_name"),
        "root": Path(getattr(args, f"{prefix}_root")).resolve(),
        "gt": Path(getattr(args, f"{prefix}_gt")).resolve(),
        "frame_start": int(getattr(args, f"{prefix}_frame_start")),
        "frame_count": int(getattr(args, f"{prefix}_frame_count")),
        "target_id": int(getattr(args, f"{prefix}_target_id")),
        "width": int(getattr(args, f"{prefix}_width")),
        "height": int(getattr(args, f"{prefix}_height")),
    }


def _prepare_sequence(model: Any, collator: Any, spec: dict[str, Any], device: torch.device) -> dict[str, Any]:
    frames = [spec["frame_start"] + index for index in range(spec["frame_count"])]
    all_boxes = _read_all_boxes(
        spec["gt"], frames=frames, width=spec["width"], height=spec["height"]
    )
    def image_path(frame: int) -> Path:
        for width in (6, 8):
            candidate = spec["root"] / "img1" / f"{frame:0{width}d}.jpg"
            if candidate.is_file():
                return candidate
        raise RuntimeError(f"missing image frame {frame} in {spec['name']}")

    images = [image_path(frame) for frame in frames]
    if not all(image.is_file() for image in images):
        raise RuntimeError(f"missing dense image in {spec['name']}")
    prepared = [
        _prepare(model, collator, _row(image, "person", all_boxes[frame]), device)
        for image, frame in zip(images, frames)
    ]
    states = {
        frame: torch.tensor(
            [value / 999.0 for value in all_boxes[frame][0]], device=device, dtype=torch.float32
        )
        for frame in frames
    }
    return {
        **spec,
        "frames": frames,
        "images": images,
        "all_boxes": all_boxes,
        "prepared": prepared,
        "summaries": [_mean_token(item["image_tokens"]) for item in prepared],
        "states": states,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--g2-adapter", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--runtime-root", default=str(RUNTIME_ROOT))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--train-count", type=int, default=2)
    parser.add_argument("--coordinate-loss-weight", type=float, default=5.0)
    for prefix, default_name, default_root, default_gt, default_start, default_id, default_width, default_height in (
        ("a", "MOT20-03", "/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-03",
         "/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-03/gt/gt.txt", 82, 27, 1173, 880),
        ("b", "MOT20-05", "/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-05",
         "/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-05/gt/gt.txt", 13, 17, 1654, 1080),
    ):
        parser.add_argument(f"--{prefix}-name", default=default_name)
        parser.add_argument(f"--{prefix}-root", default=default_root)
        parser.add_argument(f"--{prefix}-gt", default=default_gt)
        parser.add_argument(f"--{prefix}-frame-start", type=int, default=default_start)
        parser.add_argument(f"--{prefix}-frame-count", type=int, default=3)
        parser.add_argument(f"--{prefix}-target-id", type=int, default=default_id)
        parser.add_argument(f"--{prefix}-width", type=int, default=default_width)
        parser.add_argument(f"--{prefix}-height", type=int, default=default_height)
    args = parser.parse_args()
    if args.steps < 1 or args.train_count < 1:
        raise ValueError("steps and train-count must be positive")
    if args.coordinate_loss_weight < 1.0:
        raise ValueError("coordinate-loss-weight must be at least 1")

    os.environ.setdefault("GAM_DLM_FLEX_CUDAGRAPHS", "0")
    os.environ["LOCATEMOT_GROUNDANYTHING_ROOT"] = str(Path(args.source_root).resolve())
    from tools.groundjev.g2.compat import patch_transformers_causal_mask
    from tools.groundjev.g2.train_g2_dlm import _strict_load_trainable_state, load_model
    from train.dlm.data import DLMDataCollator

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("dense coupled training requires CUDA")
    patch_transformers_causal_mask()
    start = time.perf_counter()
    model, processor, _, audit = load_model(Path(args.checkpoint).resolve(), device)
    saved = torch.load(Path(args.g2_adapter).resolve(), map_location="cpu", weights_only=False)
    _strict_load_trainable_state(model, saved["model_trainable_state"])
    adapter_step = int(saved["step"])
    model.set_padding_free_packing(False)
    model.gradient_checkpointing_disable()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    collator = DLMDataCollator(
        processor, max_length=32768, max_joint_length=65536, model_family="groundinganything_qwen3"
    )
    sequences = [
        _prepare_sequence(model, collator, _sequence_spec(args, "a"), device),
        _prepare_sequence(model, collator, _sequence_spec(args, "b"), device),
    ]
    for sequence in sequences:
        if args.train_count >= len(sequence["frames"]):
            raise ValueError("train-count must leave one temporal development frame")
    input_dim = int(sequences[0]["summaries"][0].shape[-1])
    hidden = int(sequences[0]["prepared"][0]["embeds"].shape[-1])
    if input_dim != hidden:
        raise RuntimeError("G2 visual summary and prompt hidden dimensions differ")
    prompt_projector = IdentityBoxStateTokenProjector(input_dim, hidden, hidden_dim=512).to(device)
    jev_memory_projector = MemoryTokenProjector(input_dim, 256).to(device)
    readout = G2JEVReadout(input_dim, hidden_dim=256).to(device)
    trainable = list(prompt_projector.parameters()) + list(jev_memory_projector.parameters()) + list(readout.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    coordinate_min = int(processor.tokenizer.convert_tokens_to_ids("<0>"))
    coordinate_max = int(processor.tokenizer.convert_tokens_to_ids("<999>"))
    coordinate_ids = set(range(coordinate_min, coordinate_max + 1))
    train_pairs = [
        (sequence_index, frame_index)
        for sequence_index in range(len(sequences))
        for frame_index in range(1, args.train_count + 1)
    ]
    jev_before = {
        sequence["name"]: _jev_metrics(
            readout, jev_memory_projector, sequence["summaries"], start_index=1
        )
        for sequence in sequences
    }
    history: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        grounding_values: list[float] = []
        jev_values: list[float] = []
        for sequence_index, frame_index in train_pairs:
            sequence = sequences[sequence_index]
            previous_frame = sequence["frames"][frame_index - 1]
            current = sequence["summaries"][frame_index]
            previous = sequence["summaries"][frame_index - 1]
            memory = prompt_projector(
                {sequence["target_id"]: previous},
                {sequence["target_id"]: sequence["states"][previous_frame]},
                ordered_ids=[sequence["target_id"]],
            )
            grounding = _teacher_forced_loss(
                model,
                sequence["prepared"][frame_index],
                memory,
                coordinate_token_ids=coordinate_ids,
                coordinate_loss_weight=args.coordinate_loss_weight,
            )
            jev_output = _jev_forward(readout, jev_memory_projector, previous, current)
            jev_losses = structured_groundjev_losses(
                jev_output,
                identity_targets=[0], identity_valid=[True],
                admit_targets=[1.0], admit_valid=[True],
                yesno_targets=[1.0], yesno_valid=[True],
                lifecycle_targets=[0], lifecycle_valid=[True],
                memory_targets=[1.0], memory_valid=[True],
            )
            jev_total = jev_losses["total"]
            if jev_total is None:
                raise RuntimeError("dense JEV loss unexpectedly empty")
            total = grounding + jev_total
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite dense coupled loss step={step} sequence={sequence['name']}")
            (total / len(train_pairs)).backward()
            grounding_values.append(float(grounding.detach().cpu()))
            jev_values.append(float(jev_total.detach().cpu()))
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            raise RuntimeError(f"non-finite dense coupled gradient step={step}")
        optimizer.step()
        history.append({
            "step": step,
            "grounding_loss": sum(grounding_values) / len(grounding_values),
            "jev_loss": sum(jev_values) / len(jev_values),
            "total_loss": sum(grounding_values) / len(grounding_values) + sum(jev_values) / len(jev_values),
            "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
            "train_pairs": len(train_pairs),
            "dense_answer_contract": True,
        })
    jev_after = {
        sequence["name"]: _jev_metrics(
            readout, jev_memory_projector, sequence["summaries"], start_index=1
        )
        for sequence in sequences
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f"coupled-dense-step{args.steps:06d}.pt"
    torch.save(
        {
            "format": "groundjev-real-g2-coupled-dense-multisequence-v1",
            "step": args.steps,
            "adapter_step": adapter_step,
            "model_sha256": args.model_sha256,
            "prompt_box_state_projector": prompt_projector.state_dict(),
            "jev_memory_projector": jev_memory_projector.state_dict(),
            "typed_readout": readout.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "dense_answer_contract": True,
            "train_pairs": [(sequences[s]["name"], sequences[s]["frames"][f]) for s, f in train_pairs],
        },
        checkpoint,
    )
    receipt = {
        "status": "PHASE_C_REAL_G2_DENSE_COUPLING_TRAINING_COMPLETE",
        "project_root": str(PROJECT_ROOT),
        "sequences": [
            {
                "name": sequence["name"],
                "frames": sequence["frames"],
                "train_frames": sequence["frames"][1 : args.train_count + 1],
                "development_frame": sequence["frames"][-1],
                "target_id": sequence["target_id"],
                "boxes_per_frame": {str(frame): len(sequence["all_boxes"][frame]) for frame in sequence["frames"]},
            }
            for sequence in sequences
        ],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "g2_adapter": str(Path(args.g2_adapter).resolve()),
        "coupled_checkpoint": str(checkpoint),
        "adapter_step": adapter_step,
        "base_model_sha256": args.model_sha256,
        "g2_frozen": True,
        "visual_embed_calls": sum(len(sequence["prepared"]) for sequence in sequences),
        "one_visual_encode_per_frame": True,
        "independent_vlm_instances": 0,
        "trainable_components": [
            "IdentityBoxStateTokenProjector (persistent visual summary + previous box state)",
            "MemoryTokenProjector (typed JEV identity slot)",
            "G2JEVReadout (MATCH/NEW/NO_MATCH/REJECT + Yes/No + lifecycle + memory)",
        ],
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "loss_weights": {"grounding_teacher_forced": 1.0, "jev_structured": 1.0, "coordinate_tokens": args.coordinate_loss_weight},
        "dense_answer_contract": "all legal boxes per frame; one object_ref and one box group; no target-only rows",
        "optimizer_steps": args.steps,
        "train_transition_count": len(train_pairs),
        "before_jev": jev_before,
        "after_jev": jev_after,
        "history": history,
        "formal_g2_run_untouched": True,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.perf_counter() - start,
        "scientific_claim": "bounded dense-answer shared-backbone coupling training; no official MOT/RMOT/OVMOT gain claim until independent video evaluation",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
