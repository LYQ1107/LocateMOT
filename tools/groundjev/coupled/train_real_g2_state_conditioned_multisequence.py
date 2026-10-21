#!/usr/bin/env python3
"""Train a causal GroundJEV memory interface with persistent box state.

This is a bounded architecture ablation for the shared GroundAnything
backbone.  The previous-frame image summary and the previous legal box are
projected into the existing G2 causal prompt; typed JEV readout parameters are
trained on the same hidden evidence.  G2 itself stays frozen and no separate
MOT association model is introduced.
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
    IdentityBoxStateTokenProjector,
    IdentityMemoryTokenProjector,
    MemoryTokenProjector,
)
from locatemot.groundjev.native_output_repair import repair_gam_bbox_output  # noqa: E402
from tools.groundjev.coupled.real_g2_memory_prefill import _generate  # noqa: E402
from tools.groundjev.coupled.train_real_g2_coupled_memory_jev import (  # noqa: E402
    _iou,
    _jev_forward,
    _jev_metrics,
    _mean_token,
    _read_boxes,
    _teacher_forced_loss,
)
from tools.groundjev.coupled.train_real_g2_coupled_multisequence import (  # noqa: E402
    _prepare_sequence,
    _sequence_spec,
)


def _box_state(box: tuple[int, int, int, int], device: torch.device) -> torch.Tensor:
    return torch.tensor([value / 999.0 for value in box], device=device, dtype=torch.float32)


def _copy_identity_projector(target: IdentityBoxStateTokenProjector, state: dict[str, Any],
                             input_dim: int, hidden: int, device: torch.device) -> None:
    """Initialize the state interface to the prior image-only projector."""
    source = IdentityMemoryTokenProjector(input_dim, hidden, hidden_dim=512).to(device)
    source.load_state_dict(state)
    with torch.no_grad():
        target.project[0].weight[:input_dim].copy_(source.project[0].weight)
        target.project[0].weight[input_dim:].fill_(1.0)
        target.project[0].bias[:input_dim].copy_(source.project[0].bias)
        target.project[0].bias[input_dim:].zero_()
        target.project[1].weight[:, :input_dim].copy_(source.project[1].weight)
        target.project[1].weight[:, input_dim:].zero_()
        target.project[1].bias.copy_(source.project[1].bias)
        target.project[3].weight.copy_(source.project[3].weight)
        target.project[3].bias.copy_(source.project[3].bias)


def _snapshot(model: Any, sequences: list[dict[str, Any]], projector: IdentityBoxStateTokenProjector,
              structural: dict[str, int], coordinate_range: tuple[int, int], stop_ids: set[int],
              tokenizer: Any, device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for sequence in sequences:
        rows = []
        summaries = sequence["summaries"]
        frames = sequence["frames"]
        for index, frame in enumerate(frames):
            conditions: dict[str, tuple[torch.Tensor, bool]] = {
                "C0_disabled": (torch.empty((0, sequence["prepared"][index]["embeds"].shape[-1]), device=device), False)
            }
            if index:
                previous = frames[index - 1]
                memory = projector(
                    {sequence["target_id"]: summaries[index - 1]},
                    {sequence["target_id"]: _box_state(sequence["boxes"][previous], device)},
                    ordered_ids=[sequence["target_id"]],
                )
                conditions["C2_previous_box_state"] = (memory, True)
            row = {"frame": frame, "target_box_0_999": list(sequence["boxes"][frame]), "conditions": {}}
            for name, (memory, enabled) in conditions.items():
                generated = _generate(model, sequence["prepared"][index], memory, enabled=enabled,
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
                    "iou_vs_target": _iou(output_box, list(sequence["boxes"][frame])) if output_box else 0.0,
                    "repair_report": report,
                    "grammar_forced_box_end": generated["grammar_forced_box_end"],
                }
            rows.append(row)
        result[sequence["name"]] = {
            "rows": rows,
            "mean_iou_by_condition": {
                name: sum(row["conditions"][name]["iou_vs_target"] for row in rows if name in row["conditions"])
                / max(sum(name in row["conditions"] for row in rows), 1)
                for name in ("C0_disabled", "C2_previous_box_state")
            },
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--g2-adapter", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:7")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--train-count", type=int, default=3)
    parser.add_argument("--coordinate-loss-weight", type=float, default=5.0)
    parser.add_argument("--resume-coupled-checkpoint", required=True)
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
    if args.steps < 1 or args.train_count < 1 or args.coordinate_loss_weight < 1:
        raise ValueError("steps/train-count must be positive and coordinate weight must be >= 1")
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
        raise RuntimeError("state-conditioned coupled training requires CUDA")
    patch_transformers_causal_mask()
    start = time.perf_counter()
    model, processor, _, _audit = load_model(Path(args.checkpoint).resolve(), device)
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
        raise ValueError("train-count must leave one final frame held out")
    input_dim = int(sequences[0]["prepared"][0]["image_tokens"].shape[-1])
    hidden = int(sequences[0]["prepared"][0]["embeds"].shape[-1])
    state_projector = IdentityBoxStateTokenProjector(input_dim, hidden, hidden_dim=512).to(device)
    jev_memory_projector = MemoryTokenProjector(input_dim, 256).to(device)
    readout = G2JEVReadout(input_dim, hidden_dim=256).to(device)
    resume = torch.load(Path(args.resume_coupled_checkpoint).resolve(), map_location="cpu", weights_only=False)
    _copy_identity_projector(state_projector, resume["prompt_memory_projector"], input_dim, hidden, device)
    jev_memory_projector.load_state_dict(resume["jev_memory_projector"])
    readout.load_state_dict(resume["typed_readout"])
    optimizer = torch.optim.AdamW(
        list(state_projector.parameters()) + list(jev_memory_projector.parameters()) + list(readout.parameters()),
        lr=1e-3,
    )
    structural = {"box_start": int(processor.tokenizer.convert_tokens_to_ids("<|box_start|>")),
                  "box_end": int(processor.tokenizer.convert_tokens_to_ids("<|box_end|>"))}
    coordinate_ids = {int(processor.tokenizer.convert_tokens_to_ids(f"<{index}>")) for index in range(1000)}
    coordinate_range = (min(coordinate_ids), max(coordinate_ids))
    stop_ids = {structural["box_end"], int(processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))}
    summaries = [sequence["summaries"] for sequence in sequences]
    pairs = [(seq_index, frame_index) for seq_index in range(len(sequences))
             for frame_index in range(1, args.train_count + 1)]
    jev_before = {sequence["name"]: _jev_metrics(readout, jev_memory_projector,
                                                  sequence["summaries"], start_index=1)
                  for sequence in sequences}
    before_grounding = _snapshot(model, sequences, state_projector, structural, coordinate_range,
                                 stop_ids, processor.tokenizer, device)
    history = []
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        grounding_values, jev_values = [], []
        for seq_index, frame_index in pairs:
            sequence = sequences[seq_index]
            previous_frame = sequence["frames"][frame_index - 1]
            memory = state_projector(
                {sequence["target_id"]: sequence["summaries"][frame_index - 1]},
                {sequence["target_id"]: _box_state(sequence["boxes"][previous_frame], device)},
                ordered_ids=[sequence["target_id"]],
            )
            grounding = _teacher_forced_loss(model, sequence["prepared"][frame_index], memory,
                                             coordinate_token_ids=coordinate_ids,
                                             coordinate_loss_weight=args.coordinate_loss_weight)
            jev_output = _jev_forward(
                readout, jev_memory_projector,
                sequence["summaries"][frame_index - 1], sequence["summaries"][frame_index],
            )
            losses = structured_groundjev_losses(
                jev_output, identity_targets=[0], identity_valid=[True], admit_targets=[1.0],
                admit_valid=[True], yesno_targets=[1.0], yesno_valid=[True], lifecycle_targets=[0],
                lifecycle_valid=[True], memory_targets=[1.0], memory_valid=[True],
            )
            jev = losses["total"]
            if jev is None:
                raise RuntimeError("empty JEV loss")
            total = grounding + jev
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite state-conditioned loss at step {step}")
            (total / len(pairs)).backward()
            grounding_values.append(float(grounding.detach().cpu()))
            jev_values.append(float(jev.detach().cpu()))
        norm = torch.nn.utils.clip_grad_norm_(
            list(state_projector.parameters()) + list(jev_memory_projector.parameters()) + list(readout.parameters()), 1.0)
        if not torch.isfinite(torch.as_tensor(norm)):
            raise RuntimeError(f"non-finite state-conditioned gradient at step {step}")
        optimizer.step()
        history.append({"step": step, "grounding_loss": sum(grounding_values) / len(grounding_values),
                        "jev_loss": sum(jev_values) / len(jev_values),
                        "total_loss": sum(grounding_values) / len(grounding_values) + sum(jev_values) / len(jev_values),
                        "gradient_norm": float(torch.as_tensor(norm).detach().cpu()), "transitions": len(pairs)})
    jev_after = {sequence["name"]: _jev_metrics(readout, jev_memory_projector,
                                                 sequence["summaries"], start_index=1)
                 for sequence in sequences}
    after_grounding = _snapshot(model, sequences, state_projector, structural, coordinate_range,
                                stop_ids, processor.tokenizer, device)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f"coupled-state-step{args.steps:06d}.pt"
    torch.save({"format": "groundjev-real-g2-state-conditioned-coupling", "step": args.steps,
                "adapter_step": adapter_step, "model_sha256": args.model_sha256,
                "state_memory_projector": state_projector.state_dict(),
                "jev_memory_projector": jev_memory_projector.state_dict(),
                "typed_readout": readout.state_dict(), "optimizer_state": optimizer.state_dict(),
                "train_pairs": [(sequences[s]["name"], sequences[s]["frames"][f]) for s, f in pairs],
                "resumed_from": str(Path(args.resume_coupled_checkpoint).resolve())}, checkpoint)
    receipt = {
        "status": "PHASE_C_REAL_G2_STATE_CONDITIONED_COUPLING_COMPLETE",
        "project_root": str(PROJECT_ROOT), "adapter_step": adapter_step,
        "base_model_sha256": args.model_sha256, "resume_from_step": int(resume.get("step", 0)),
        "optimizer_steps": args.steps, "coordinate_loss_weight": args.coordinate_loss_weight,
        "sequences": [{"name": s["name"], "frames": s["frames"],
                       "train_frames": s["frames"][1:args.train_count + 1],
                       "heldout_frame": s["frames"][-1], "target_id": s["target_id"]} for s in sequences],
        "train_transition_count": len(pairs), "visual_embed_calls": sum(len(s["prepared"]) for s in sequences),
        "independent_vlm_instances": 0, "g2_frozen": True, "formal_g2_run_untouched": True,
        "trainable_components": ["IdentityBoxStateTokenProjector", "MemoryTokenProjector", "G2JEVReadout"],
        "trainable_parameter_count": sum(p.numel() for module in (state_projector, jev_memory_projector, readout) for p in module.parameters()),
        "before_jev": jev_before, "after_jev": jev_after, "before_grounding": before_grounding,
        "after_grounding": after_grounding, "history": history,
        "checkpoint": str(checkpoint), "checkpoint_sha256": __import__('hashlib').sha256(checkpoint.read_bytes()).hexdigest(),
        "elapsed_seconds": time.perf_counter() - start,
        "scientific_claim": "bounded two-sequence state-conditioned shared-backbone coupling; no official MOT/RMOT/OVMOT metric claim",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
