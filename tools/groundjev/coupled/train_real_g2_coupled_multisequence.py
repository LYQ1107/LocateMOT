#!/usr/bin/env python3
"""Train the bounded coupled memory/JEV pilot on two real MOT sequences.

This is the smallest follow-up to the single-sequence overfit diagnosis.  A
single frozen G2 backbone encodes both sequences; one shared memory projector
and one typed JEV readout see transitions from each sequence.  The final frame
of each sequence is held out temporally.  No independent VLM or G2 update is
introduced.
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
from locatemot.groundjev.coupled.losses import structured_groundjev_losses  # noqa: E402
from locatemot.groundjev.coupled.memory_tokens import (  # noqa: E402
    IdentityMemoryTokenProjector,
    MemoryTokenProjector,
)
from tools.groundjev.coupled.train_real_g2_coupled_memory_jev import (  # noqa: E402
    _grounding_snapshot,
    _jev_forward,
    _jev_metrics,
    _mean_token,
    _read_boxes,
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
    boxes = _read_boxes(spec["gt"], target_id=spec["target_id"],
                        frame_start=spec["frame_start"], frame_count=spec["frame_count"],
                        width=spec["width"], height=spec["height"])
    images = [spec["root"] / "img1" / f"{frame:06d}.jpg" for frame in frames]
    if not all(image.is_file() for image in images):
        raise RuntimeError(f"missing image in {spec['name']}")
    prepared = [_prepare(model, collator, _row(image, "person", [boxes[frame]]), device)
                for image, frame in zip(images, frames)]
    return {
        **spec,
        "frames": frames,
        "images": images,
        "boxes": boxes,
        "prepared": prepared,
        "summaries": [_mean_token(item["image_tokens"]) for item in prepared],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--g2-adapter", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--resume-coupled-checkpoint", default=None,
                        help="optional coupled checkpoint produced by this script")
    parser.add_argument("--train-count", type=int, default=3,
                        help="transitions 1..train-count per sequence; last frame stays held out")
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
        parser.add_argument(f"--{prefix}-frame-count", type=int, default=5)
        parser.add_argument(f"--{prefix}-target-id", type=int, default=default_id)
        parser.add_argument(f"--{prefix}-width", type=int, default=default_width)
        parser.add_argument(f"--{prefix}-height", type=int, default=default_height)
    args = parser.parse_args()
    if args.steps < 1 or args.train_count < 1:
        raise ValueError("steps and train-count must be positive")

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
        raise RuntimeError("multi-sequence coupled training requires CUDA")
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
    collator = DLMDataCollator(processor, max_length=32768, max_joint_length=65536,
                               model_family="groundinganything_qwen3")
    specs = [_sequence_spec(args, "a"), _sequence_spec(args, "b")]
    sequences = [_prepare_sequence(model, collator, spec, device) for spec in specs]
    if any(args.train_count >= len(sequence["frames"]) for sequence in sequences):
        raise ValueError("train-count must leave one final frame held out in every sequence")
    input_dim = int(sequences[0]["prepared"][0]["image_tokens"].shape[-1])
    hidden = int(sequences[0]["prepared"][0]["embeds"].shape[-1])
    if hidden != input_dim:
        raise RuntimeError("G2 image-token and prompt dimensions must agree")
    for sequence in sequences:
        if any(int(item["image_tokens"].shape[-1]) != input_dim for item in sequence["prepared"]):
            raise RuntimeError("G2 feature dimension changed between sequences")

    prompt_projector = IdentityMemoryTokenProjector(input_dim, hidden, hidden_dim=512).to(device)
    jev_memory_projector = MemoryTokenProjector(input_dim, 256).to(device)
    readout = G2JEVReadout(input_dim, hidden_dim=256).to(device)
    optimizer = torch.optim.AdamW(
        list(prompt_projector.parameters()) + list(jev_memory_projector.parameters())
        + list(readout.parameters()), lr=1e-3,
    )
    resume_state = None
    if args.resume_coupled_checkpoint:
        resume_state = torch.load(Path(args.resume_coupled_checkpoint).resolve(), map_location="cpu", weights_only=False)
        prompt_projector.load_state_dict(resume_state["prompt_memory_projector"])
        jev_memory_projector.load_state_dict(resume_state["jev_memory_projector"])
        readout.load_state_dict(resume_state["typed_readout"])
        if "optimizer_state" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer_state"])
    train_pairs = [(sequence_index, frame_index)
                   for sequence_index in range(len(sequences))
                   for frame_index in range(1, args.train_count + 1)]
    jev_before = {
        sequence["name"]: _jev_metrics(readout, jev_memory_projector, sequence["summaries"], start_index=1)
        for sequence in sequences
    }
    history = []
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    starting_step = int(resume_state.get("step", 0)) if resume_state is not None else 0
    for local_step in range(1, args.steps + 1):
        step = starting_step + local_step
        optimizer.zero_grad(set_to_none=True)
        grounding_values = []
        jev_values = []
        for sequence_index, frame_index in train_pairs:
            sequence = sequences[sequence_index]
            memory = prompt_projector(
                {sequence["target_id"]: sequence["summaries"][frame_index - 1]},
                ordered_ids=[sequence["target_id"]],
            )
            grounding = _teacher_forced_loss(model, sequence["prepared"][frame_index], memory)
            jev_output = _jev_forward(
                readout, jev_memory_projector,
                sequence["summaries"][frame_index - 1], sequence["summaries"][frame_index],
            )
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
                raise RuntimeError("multi-sequence JEV loss unexpectedly empty")
            total = grounding + jev_total
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite multi-sequence loss at step={step} sequence={sequence['name']} frame={sequence['frames'][frame_index]}")
            (total / len(train_pairs)).backward()
            grounding_values.append(float(grounding.detach().cpu()))
            jev_values.append(float(jev_total.detach().cpu()))
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(prompt_projector.parameters()) + list(jev_memory_projector.parameters())
            + list(readout.parameters()), 1.0,
        )
        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            raise RuntimeError(f"non-finite multi-sequence gradient at step={step}")
        optimizer.step()
        history.append({
            "step": step,
            "grounding_loss": sum(grounding_values) / len(grounding_values),
            "jev_loss": sum(jev_values) / len(jev_values),
            "total_loss": sum(grounding_values) / len(grounding_values)
            + sum(jev_values) / len(jev_values),
            "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
            "transitions": len(train_pairs),
        })

    jev_after = {
        sequence["name"]: _jev_metrics(readout, jev_memory_projector, sequence["summaries"], start_index=1)
        for sequence in sequences
    }
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
    before_projector = IdentityMemoryTokenProjector(input_dim, hidden, hidden_dim=512).to(device)
    before_grounding = {}
    after_grounding = {}
    for sequence in sequences:
        before_grounding[sequence["name"]] = _grounding_snapshot(
            model, sequence["prepared"], sequence["summaries"], before_projector,
            sequence["boxes"], sequence["frames"], structural, coordinate_range,
            stop_ids, tokenizer, device,
        )
        after_grounding[sequence["name"]] = _grounding_snapshot(
            model, sequence["prepared"], sequence["summaries"], prompt_projector,
            sequence["boxes"], sequence["frames"], structural, coordinate_range,
            stop_ids, tokenizer, device,
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "groundjev-real-g2-coupled-multisequence-pilot",
        "step": starting_step + args.steps,
        "adapter_step": adapter_step,
        "model_sha256": args.model_sha256,
        "prompt_memory_projector": prompt_projector.state_dict(),
        "jev_memory_projector": jev_memory_projector.state_dict(),
        "typed_readout": readout.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_pairs": [(sequences[s]["name"], sequences[s]["frames"][f]) for s, f in train_pairs],
        "resumed_from": str(Path(args.resume_coupled_checkpoint).resolve()) if args.resume_coupled_checkpoint else None,
    }, output / f"coupled-multisequence-step{starting_step + args.steps:06d}.pt")
    receipt = {
        "status": "PHASE_C_REAL_G2_COUPLED_MULTISEQUENCE_TRAINING_COMPLETE",
        "project_root": str(PROJECT_ROOT),
        "sequences": [{"name": sequence["name"], "frames": sequence["frames"],
                       "train_frames": sequence["frames"][1:args.train_count + 1],
                       "heldout_frame": sequence["frames"][-1], "target_id": sequence["target_id"]}
                      for sequence in sequences],
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "g2_adapter": str(Path(args.g2_adapter).resolve()),
        "adapter_step": adapter_step,
        "base_model_sha256": args.model_sha256,
        "g2_frozen": True,
        "visual_embed_calls": sum(len(sequence["prepared"]) for sequence in sequences),
        "one_visual_encode_per_frame": True,
        "independent_vlm_instances": 0,
        "trainable_components": [
            "IdentityMemoryTokenProjector (G2 input-space causal memory)",
            "MemoryTokenProjector (typed JEV identity slots)",
            "G2JEVReadout (dynamic MATCH/NEW/NO_MATCH/REJECT + Yes/No + lifecycle + memory)",
        ],
        "trainable_parameter_count": sum(
            parameter.numel() for module in (prompt_projector, jev_memory_projector, readout)
            for parameter in module.parameters()
        ),
        "loss_weights": {"grounding_teacher_forced": 1.0, "jev_structured": 1.0},
        "causal_teacher_forcing": True,
        "train_transition_count": len(train_pairs),
        "optimizer_steps_this_run": args.steps,
        "optimizer_step_start": starting_step,
        "optimizer_step_end": starting_step + args.steps,
        "before_jev": jev_before,
        "after_jev": jev_after,
        "history": history,
        "before_grounding": before_grounding,
        "after_grounding": after_grounding,
        "answer_tokens_excluded_from_prefill": True,
        "formal_g2_run_untouched": True,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.perf_counter() - start,
        "scientific_claim": "bounded two-sequence real GPU coupling training; no official MOT/RMOT/OVMOT metric claim",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
