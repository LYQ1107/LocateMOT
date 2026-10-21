#!/usr/bin/env python3
"""Train a bounded real GroundJEV coupling pilot on one frozen G2 backbone.

The official G2 DLM is loaded once and frozen.  Its image/prompt embeddings
are prepared once per real frame.  A small identity-memory token projector and
the typed JEV readout are updated together: the same previous-frame summary
conditions a causal Grounding teacher-forcing loss and a MATCH/Yes-No/
lifecycle/memory decision loss over the current shared G2 tokens.  This is a
finite Phase-C/D bridge experiment, not a full end-to-end benchmark.
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
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locatemot.groundjev.coupled.causal_memory import causal_prefill_with_memory  # noqa: E402
from locatemot.groundjev.coupled.g2_jev_readout import G2JEVReadout  # noqa: E402
from locatemot.groundjev.coupled.losses import structured_groundjev_losses  # noqa: E402
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
    missing = sorted(wanted.difference(boxes))
    if missing:
        raise RuntimeError(f"target ID {target_id} missing from frames {missing}")
    return boxes


def _mean_token(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 2 or tokens.shape[0] == 0:
        raise RuntimeError("cannot summarize empty G2 image tokens")
    return tokens.float().mean(dim=0)


def _teacher_forced_loss(model: Any, prepared: dict[str, Any], memory: torch.Tensor) -> torch.Tensor:
    """Differentiate the causal answer likelihood into the memory projector."""
    target = prepared["answer_tokens"].to(device=prepared["embeds"].device)
    if target.numel() == 0:
        raise RuntimeError("prepared G2 row has no answer tokens for teacher forcing")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        prefill = causal_prefill_with_memory(
            model.language_model,
            model.lm_head,
            prepared["embeds"], prepared["mask"], prepared["positions"], memory,
            native_position_ids=model._native_position_ids,
            enabled=True,
            inference_mode=False,
        )
        cache = prefill["outputs"].past_key_values
        if cache is None:
            raise RuntimeError("training prefill did not return a KV cache")
        logits = prefill["logits"]
        losses = [F.cross_entropy(logits, target[:1])]
        axes = int(prefill["prepared"].position_ids.shape[0])
        batch_size = int(prefill["prepared"].inputs_embeds.shape[0])
        rope_deltas = getattr(model.multimodal_model, "rope_deltas", None)
        if rope_deltas is not None:
            rope_deltas = rope_deltas.detach().clone()
        for token_index in range(int(target.numel()) - 1):
            token = target[token_index].reshape(1, 1)
            position = int(cache.get_seq_length())
            positions = model._generation_position_ids(
                torch.tensor([position], device=token.device, dtype=torch.long),
                axes, batch_size, rope_deltas,
            )
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
            losses.append(F.cross_entropy(logits, target[token_index + 1:token_index + 2]))
            cache = outputs.past_key_values
    return torch.stack(losses).mean()


def _jev_forward(readout: G2JEVReadout, memory_projector: MemoryTokenProjector,
                 previous: torch.Tensor, current: torch.Tensor) -> dict[str, torch.Tensor | int]:
    previous_slot = memory_projector(previous.unsqueeze(0))
    slots = readout.make_slot_tokens(previous_slot, ["NEW", "NO_MATCH", "REJECT"])
    return readout(current.unsqueeze(0), slots, action_start=1)


def _jev_metrics(readout: G2JEVReadout, memory_projector: MemoryTokenProjector,
                 summaries: list[torch.Tensor], *, start_index: int) -> dict[str, float | int]:
    total = {"identity": 0, "identity_correct": 0, "admit": 0, "admit_correct": 0,
             "yesno": 0, "yesno_correct": 0, "lifecycle": 0, "lifecycle_correct": 0}
    with torch.no_grad():
        for index in range(max(1, start_index), len(summaries)):
            output = _jev_forward(readout, memory_projector, summaries[index - 1], summaries[index])
            total["identity"] += 1
            total["identity_correct"] += int(int(output["identity_logits"].argmax(-1)[0]) == 0)
            total["admit"] += 1
            total["admit_correct"] += int(bool(torch.sigmoid(output["admit_logits"])[0] >= 0.5))
            total["yesno"] += 1
            total["yesno_correct"] += int(bool(torch.sigmoid(output["yesno_logits"])[0] >= 0.5))
            total["lifecycle"] += 1
            total["lifecycle_correct"] += int(int(output["lifecycle_logits"].argmax(-1)[0]) == 0)
    for name in ("identity", "admit", "yesno", "lifecycle"):
        total[f"{name}_accuracy"] = total[f"{name}_correct"] / max(total[name], 1)
    return total


def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return intersection / max(area_a + area_b - intersection, 1)


def _grounding_snapshot(model: Any, prepared: list[dict[str, Any]], summaries: list[torch.Tensor],
                        prompt_projector: IdentityMemoryTokenProjector, boxes: dict[int, tuple[int, int, int, int]],
                        frames: list[int], structural: dict[str, int], coordinate_range: tuple[int, int],
                        stop_ids: set[int], tokenizer: Any, device: torch.device) -> dict[str, Any]:
    rows = []
    for index, frame in enumerate(frames):
        conditions: dict[str, tuple[torch.Tensor, bool]] = {
            "C0_disabled": (torch.empty((0, int(prepared[index]["embeds"].shape[-1])), device=device), False),
        }
        if index:
            memory = prompt_projector({0: summaries[index - 1]}, ordered_ids=[0])
            conditions["C2_previous_frame_memory"] = (memory, True)
        row: dict[str, Any] = {"frame": frame, "target_box_0_999": list(boxes[frame]), "conditions": {}}
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
                "repair_report": report,
                "grammar_forced_box_end": generated["grammar_forced_box_end"],
            }
        rows.append(row)
    return {"rows": rows, "mean_iou_by_condition": {
        name: sum(row["conditions"][name]["iou_vs_target"] for row in rows if name in row["conditions"])
        / max(sum(name in row["conditions"] for row in rows), 1)
        for name in ("C0_disabled", "C2_previous_frame_memory")
    }}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--sequence-root", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--frame-start", type=int, default=82)
    parser.add_argument("--frame-count", type=int, default=5)
    parser.add_argument("--train-count", type=int, default=3,
                        help="number of transitions after the first frame used for updates")
    parser.add_argument("--target-id", type=int, default=27)
    parser.add_argument("--width", type=int, default=1173)
    parser.add_argument("--height", type=int, default=880)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.frame_count < 2 or args.train_count < 1 or args.train_count > args.frame_count - 1:
        raise ValueError("train-count must cover at least one transition and fit in frame-count")

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
        raise RuntimeError("real coupled memory/JEV training requires CUDA")
    patch_transformers_causal_mask()
    start = time.perf_counter()
    frames = [args.frame_start + i for i in range(args.frame_count)]
    sequence_root = Path(args.sequence_root).resolve()
    gt_path = Path(args.gt).resolve()
    boxes = _read_boxes(gt_path, target_id=args.target_id, frame_start=args.frame_start,
                        frame_count=args.frame_count, width=args.width, height=args.height)
    images = [sequence_root / "img1" / f"{frame:06d}.jpg" for frame in frames]
    if not all(image.is_file() for image in images):
        raise RuntimeError("one or more real replay images are missing")

    model, processor, _, audit = load_model(Path(args.checkpoint).resolve(), device)
    saved = torch.load(Path(args.adapter).resolve(), map_location="cpu", weights_only=False)
    _strict_load_trainable_state(model, saved["model_trainable_state"])
    adapter_step = int(saved["step"])
    model.set_padding_free_packing(False)
    model.gradient_checkpointing_disable()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    collator = DLMDataCollator(processor, max_length=32768, max_joint_length=65536,
                               model_family="groundinganything_qwen3")
    prepared = []
    for image, frame in zip(images, frames):
        prepared.append(_prepare(model, collator, _row(image, "person", [boxes[frame]]), device))
    summaries = [_mean_token(item["image_tokens"]) for item in prepared]
    input_dim = int(prepared[0]["image_tokens"].shape[-1])
    hidden = int(prepared[0]["embeds"].shape[-1])
    if hidden != input_dim:
        raise RuntimeError("this pilot expects G2 image-token and prompt hidden dimensions to agree")

    prompt_projector = IdentityMemoryTokenProjector(input_dim, hidden, hidden_dim=512).to(device)
    jev_memory_projector = MemoryTokenProjector(input_dim, 256).to(device)
    readout = G2JEVReadout(input_dim, hidden_dim=256).to(device)
    optimizer = torch.optim.AdamW(
        list(prompt_projector.parameters()) + list(jev_memory_projector.parameters())
        + list(readout.parameters()), lr=1e-3,
    )
    train_indices = list(range(1, args.train_count + 1))
    structural = {
        "box_start": int(processor.tokenizer.convert_tokens_to_ids("<|box_start|>")),
        "box_end": int(processor.tokenizer.convert_tokens_to_ids("<|box_end|>")),
    }
    coordinate_range = (
        int(processor.tokenizer.convert_tokens_to_ids("<0>")),
        int(processor.tokenizer.convert_tokens_to_ids("<999>")),
    )
    stop_ids = {structural["box_end"], int(processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))}
    if min(structural.values()) < 0 or min(coordinate_range) < 0:
        raise RuntimeError("G2 tokenizer lacks the official structured grounding contract")

    jev_before = _jev_metrics(readout, jev_memory_projector, summaries, start_index=1)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    history = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        grounding_values = []
        jev_values = []
        gradient_inputs = []
        for index in train_indices:
            memory = prompt_projector({args.target_id: summaries[index - 1]}, ordered_ids=[args.target_id])
            grounding = _teacher_forced_loss(model, prepared[index], memory)
            jev_output = _jev_forward(
                readout, jev_memory_projector, summaries[index - 1], summaries[index])
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
                raise RuntimeError("JEV loss unexpectedly empty")
            total = grounding + jev_total
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite coupled loss at step {step} frame {frames[index]}")
            (total / len(train_indices)).backward()
            grounding_values.append(float(grounding.detach().cpu()))
            jev_values.append(float(jev_total.detach().cpu()))
            gradient_inputs.append(total.detach())
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            list(prompt_projector.parameters()) + list(jev_memory_projector.parameters())
            + list(readout.parameters()), 1.0,
        )
        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            raise RuntimeError(f"non-finite coupled gradient at step {step}")
        optimizer.step()
        history.append({
            "step": step,
            "grounding_loss": sum(grounding_values) / len(grounding_values),
            "jev_loss": sum(jev_values) / len(jev_values),
            "total_loss": sum(grounding_values) / len(grounding_values)
            + sum(jev_values) / len(jev_values),
            "gradient_norm": float(torch.as_tensor(gradient_norm).detach().cpu()),
        })

    jev_after = _jev_metrics(readout, jev_memory_projector, summaries, start_index=1)
    tokenizer = processor.tokenizer
    before_projector = IdentityMemoryTokenProjector(input_dim, hidden, hidden_dim=512).to(device)
    # Match the untrained control used by the prefill experiment: the before
    # C2 arm is a fresh projector, while C0 is independent of projector state.
    before_grounding = _grounding_snapshot(
        model, prepared, summaries, before_projector, boxes, frames,
        structural, coordinate_range, stop_ids, tokenizer, device,
    )
    after_grounding = _grounding_snapshot(
        model, prepared, summaries, prompt_projector, boxes, frames,
        structural, coordinate_range, stop_ids, tokenizer, device,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "groundjev-real-g2-coupled-memory-jev-pilot",
        "step": args.steps,
        "adapter_step": adapter_step,
        "model_sha256": args.model_sha256,
        "prompt_memory_projector": prompt_projector.state_dict(),
        "jev_memory_projector": jev_memory_projector.state_dict(),
        "typed_readout": readout.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "train_frames": [frames[index] for index in train_indices],
    }, output / f"coupled-adapter-step{args.steps:06d}.pt")
    receipt = {
        "status": "PHASE_C_REAL_G2_COUPLED_MEMORY_JEV_TRAINING_COMPLETE",
        "project_root": str(PROJECT_ROOT),
        "sequence": sequence_root.name,
        "frames": frames,
        "train_frames": [frames[index] for index in train_indices],
        "heldout_frame": frames[-1],
        "target_id": args.target_id,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "adapter_step": adapter_step,
        "base_model_sha256": args.model_sha256,
        "g2_frozen": True,
        "visual_embed_calls": len(prepared),
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
        "before_jev": jev_before,
        "after_jev": jev_after,
        "history": history,
        "before_grounding": before_grounding,
        "after_grounding": after_grounding,
        "native_g2_dlm_forward_count": len(prepared),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "elapsed_seconds": time.perf_counter() - start,
        "formal_g2_run_untouched": True,
        "scientific_claim": "bounded real GPU coupled training and causal-output evidence; no independent MOT/RMOT/OVMOT metric claim",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
