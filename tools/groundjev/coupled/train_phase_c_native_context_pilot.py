#!/usr/bin/env python3
"""Bounded multi-frame C1 pilot with native G2 clean context tokens.

This is a causal readout pilot: the official G2 DLM is frozen and each frame
is encoded once.  The downstream GroundJEV typed heads learn over native image
regions plus the same-frame clean-language context.  It is explicitly not a
full end-to-end G2/JEV training run or a tracking metric evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image

from locatemot.groundjev.coupled.g2_feature_bridge import G2FeatureBridge
from locatemot.groundjev.coupled.groundjev_model import GroundJEVModel
from locatemot.groundjev.coupled.losses import structured_groundjev_losses
from tools.groundjev.coupled.train_phase_b_choice_yes_pilot import (
    load_candidate_rows,
    load_gt,
    match_candidates,
)


def collect_examples(rows, bridge, gt_by_frame):
    examples = []
    id_to_token: dict[int, torch.Tensor] = {}
    for index, row in enumerate(rows):
        image_path = Path(row["image_path"]).resolve()
        image = Image.open(image_path).convert("RGB")
        boxes = [tuple(float(v) for v in candidate["box"]) for candidate in row["candidates"]]
        labels = [str(candidate.get("label", "person")) for candidate in row["candidates"]]
        state = bridge.encode(
            image, boxes, labels, frame_id=index, image_key=str(image_path), task_query="person",
        )
        gt_ids = match_candidates(boxes, gt_by_frame.get(int(row["frame"]), []))
        slot_ids = sorted(id_to_token)
        slot_position = {track_id: position for position, track_id in enumerate(slot_ids)}
        action_start = len(slot_ids)
        identity_targets = []
        admit_targets = []
        yesno_targets = []
        memory_targets = []
        seen: set[int] = set()
        for track_id in gt_ids:
            if track_id is None or track_id in seen:
                identity_targets.append(action_start + 2)
                admit_targets.append(0.0)
                yesno_targets.append(0.0)
                memory_targets.append(0.0)
            elif track_id in slot_position:
                identity_targets.append(slot_position[track_id])
                admit_targets.append(1.0)
                yesno_targets.append(1.0)
                memory_targets.append(1.0)
                seen.add(track_id)
            else:
                identity_targets.append(action_start)
                admit_targets.append(1.0)
                yesno_targets.append(0.0)
                memory_targets.append(1.0)
                seen.add(track_id)
        slot_features = (
            torch.stack([id_to_token[track_id] for track_id in slot_ids])
            if slot_ids else torch.empty((0, state.feature_dim))
        )
        examples.append({
            "frame": int(row["frame"]),
            "image_path": str(image_path),
            "features": state.vision_tokens.detach().cpu(),
            "context": state.contextual_tokens.detach().cpu(),
            "slot_ids": slot_ids,
            "slot_features": slot_features,
            "identity_targets": identity_targets,
            "admit_targets": admit_targets,
            "yesno_targets": yesno_targets,
            "memory_targets": memory_targets,
            "candidate_count": len(boxes),
        })
        for candidate_index, track_id in enumerate(gt_ids):
            if track_id is not None:
                id_to_token[int(track_id)] = state.vision_tokens[candidate_index].detach().cpu().clone()
    return examples


def forward_example(model, example, device):
    region = example["features"].to(device)
    slot_features = example["slot_features"].to(device)
    memory = model.memory_projector(slot_features) if slot_features.shape[0] else torch.empty(
        (0, model.hidden_dim), device=device,
    )
    slots = model.readout.make_slot_tokens(memory, ["NEW", "NO_MATCH", "REJECT"])
    context = example["context"].to(device)
    context_bias = model.context_projector(context).mean(dim=0)
    return model.readout(region, slots, action_start=example["slot_features"].shape[0], context_bias=context_bias)


def evaluate(model, examples, device):
    totals = {"identity": 0, "identity_correct": 0, "admit": 0, "admit_correct": 0}
    with torch.no_grad():
        for example in examples:
            outputs = forward_example(model, example, device)
            identity = outputs["identity_logits"].argmax(dim=-1).cpu().tolist()
            admit = torch.sigmoid(outputs["admit_logits"]).ge(0.5).cpu().tolist()
            for index, target in enumerate(example["identity_targets"]):
                totals["identity"] += 1
                totals["identity_correct"] += int(identity[index] == target)
                totals["admit"] += 1
                totals["admit_correct"] += int(bool(admit[index]) == bool(example["admit_targets"][index]))
    totals["identity_accuracy"] = totals["identity_correct"] / max(totals["identity"], 1)
    totals["admit_accuracy"] = totals["admit_correct"] / max(totals["admit"], 1)
    return totals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--runtime-root", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--train-frames", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ["LOCATEMOT_GROUNDANYTHING_ROOT"] = str(Path(args.source_root).resolve())
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the native context pilot")
    rows = load_candidate_rows(Path(args.candidate_manifest), args.max_frames)
    gt_by_frame = load_gt(Path(args.gt))
    if len(rows) < args.train_frames or args.train_frames <= 0:
        raise RuntimeError("native context pilot needs a non-empty train/dev split")
    start = time.perf_counter()
    bridge = G2FeatureBridge.from_native_clean_dlm(
        args.checkpoint, args.adapter, runtime_root=args.runtime_root, device=args.device,
        model_sha256=args.model_sha256, task_query="person",
    )
    examples = collect_examples(rows, bridge, gt_by_frame)
    model = GroundJEVModel(examples[0]["features"].shape[-1], hidden_dim=256).to(device)
    train_examples = examples[:args.train_frames]
    dev_examples = examples[args.train_frames:]
    before = {"train": evaluate(model, train_examples, device), "temporal_dev": evaluate(model, dev_examples, device)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    history = []
    step = 0
    for epoch in range(args.epochs):
        for example in train_examples:
            outputs = forward_example(model, example, device)
            losses = structured_groundjev_losses(
                outputs,
                identity_targets=example["identity_targets"],
                identity_valid=[True] * len(example["identity_targets"]),
                admit_targets=example["admit_targets"], admit_valid=[True] * len(example["admit_targets"]),
                yesno_targets=example["yesno_targets"], yesno_valid=[True] * len(example["yesno_targets"]),
                memory_targets=example["memory_targets"], memory_valid=[True] * len(example["memory_targets"]),
            )
            total = losses["total"]
            if total is None or not torch.isfinite(total):
                raise RuntimeError(f"non-finite native context pilot loss at epoch={epoch} step={step}")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(torch.as_tensor(gradient_norm)):
                raise RuntimeError(f"non-finite native context pilot gradient at step={step}")
            optimizer.step()
            step += 1
            history.append({"step": step, "epoch": epoch + 1, "loss": float(total.detach().cpu())})
    after = {"train": evaluate(model, train_examples, device), "temporal_dev": evaluate(model, dev_examples, device)}
    receipt = {
        "status": "PHASE_C_NATIVE_G2_CONTEXT_READOUT_PILOT_COMPLETE",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "model_sha256": args.model_sha256,
        "candidate_manifest": str(Path(args.candidate_manifest).resolve()),
        "gt_manifest": str(Path(args.gt).resolve()),
        "frames": len(examples), "train_frames": len(train_examples), "temporal_dev_frames": len(dev_examples),
        "candidate_count": sum(int(example["candidate_count"]) for example in examples),
        "teacher_forced_causal_memory": True,
        "frozen_native_g2": True,
        "native_dlm_forward_count": bridge.provider.forward_count,
        "bridge_forward_count": bridge.forward_count,
        "before": before, "after": after,
        "optimizer_steps": step,
        "history": history,
        "elapsed_seconds": time.perf_counter() - start,
        "single_backbone": True,
        "scientific_claim": "bounded native context readout learnability control; no tracking metric or generalization claim",
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
