#!/usr/bin/env python3
"""Bounded Phase-B pilot on real GroundAnything candidates and native G2 features.

The proposal boxes come from a prior native GroundAnything candidate export;
the G2 checkpoint is used again to pool the shared visual evidence.  MOT GT
is used only to create causal supervision and teacher-forced memory slots.  A
teacher-memory pilot is explicitly marked as such and is not a final tracker.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from locatemot.groundjev.coupled.dynamic_identity_slots import build_dynamic_slots
from locatemot.groundjev.coupled.g2_feature_bridge import G2FeatureBridge
from locatemot.groundjev.coupled.groundjev_model import GroundJEVModel
from locatemot.groundjev.coupled.g2_shared_runtime import CoupledGroundJEVRuntime
from locatemot.groundjev.coupled.identity_memory import IdentityMemoryState
from locatemot.groundjev.coupled.losses import structured_groundjev_losses


@dataclass
class FrameExample:
    frame: int
    image_path: str
    boxes: list[tuple[float, float, float, float]]
    labels: list[str]
    features: torch.Tensor
    gt_ids: list[int | None]
    slot_track_ids: list[int]
    slot_features: torch.Tensor
    option_tokens: list[str]
    identity_targets: list[int | None]
    admit_targets: list[float]
    yesno_targets: list[float]
    lifecycle_targets: list[int]
    memory_targets: list[float]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def load_gt(path: Path) -> dict[int, list[tuple[int, tuple[float, float, float, float]]]]:
    result: dict[int, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for line in path.read_text().splitlines():
        frame, track_id, x, y, w, h, mark, category, *_ = line.split(",")
        if int(float(mark)) != 1 or int(float(category)) != 1:
            continue
        box = (float(x), float(y), float(x) + float(w), float(y) + float(h))
        result.setdefault(int(frame), []).append((int(track_id), box))
    return result


def match_candidates(boxes, gt_rows, threshold=0.5):
    matched: list[int | None] = []
    for box in boxes:
        choices = [(iou(box, gt_box), track_id) for track_id, gt_box in gt_rows]
        best = max(choices, default=(0.0, None))
        matched.append(int(best[1]) if best[0] >= threshold and best[1] is not None else None)
    return matched


def load_candidate_rows(path: Path, max_frames: int):
    rows = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        rows.append(row)
        if len(rows) >= max_frames:
            break
    if not rows:
        raise RuntimeError("candidate manifest has no rows")
    return rows


def build_examples(rows, bridge, gt_by_frame, device, train_frames):
    # First pass: one G2 visual forward per frame, then freeze finite features.
    cached = []
    for index, row in enumerate(rows):
        image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        boxes = [tuple(float(v) for v in c["box"]) for c in row["candidates"]]
        labels = [str(c.get("label", "person")) for c in row["candidates"]]
        state = bridge.encode(image, boxes, labels, frame_id=index,
                              image_key=str(image_path), task_query="person")
        cached.append((int(row["frame"]), str(image_path), boxes, labels,
                       state.vision_tokens.detach().cpu(),
                       match_candidates(boxes, gt_by_frame.get(int(row["frame"]), []))))

    # Build dynamic options from causal previous-frame GT memory.  The memory
    # tokens are previous G2 region tokens, never future frames.
    id_to_token: dict[int, torch.Tensor] = {}
    examples = []
    for frame, image_path, boxes, labels, features, gt_ids in cached:
        slot_ids = sorted(id_to_token)
        slot_position = {track_id: position for position, track_id in enumerate(slot_ids)}
        option_tokens = [str(x) for x in slot_ids] + ["NEW", "NO_MATCH", "REJECT"]
        slot_features = (torch.stack([id_to_token[x] for x in slot_ids])
                         if slot_ids else torch.empty((0, features.shape[-1])))
        action_start = len(slot_ids)
        new_targets: dict[int, int] = {}
        seen_current: set[int] = set()
        identity_targets: list[int | None] = []
        admit_targets: list[float] = []
        yesno_targets: list[float] = []
        life_targets: list[int] = []
        memory_targets: list[float] = []
        for candidate_index, track_id in enumerate(gt_ids):
            if track_id is None:
                identity_targets.append(action_start + 2)  # explicit REJECT
                admit_targets.append(0.0); yesno_targets.append(0.0); memory_targets.append(0.0)
            elif track_id in seen_current:
                identity_targets.append(action_start + 2)
                admit_targets.append(0.0); yesno_targets.append(0.0); memory_targets.append(0.0)
            elif track_id in slot_position:
                identity_targets.append(slot_position[track_id])
                admit_targets.append(1.0); yesno_targets.append(1.0); memory_targets.append(1.0)
                seen_current.add(track_id)
            else:
                identity_targets.append(action_start)  # explicit NEW
                admit_targets.append(1.0); yesno_targets.append(0.0); memory_targets.append(1.0)
                seen_current.add(track_id)
                new_targets[track_id] = len(slot_ids)
            life_targets.append(0)  # ACTIVE is the only causal label in this pilot.
        examples.append(FrameExample(frame, image_path, boxes, labels, features, gt_ids,
                                     slot_ids, slot_features, option_tokens, identity_targets,
                                     admit_targets, yesno_targets, life_targets, memory_targets))
        for candidate_index, track_id in enumerate(gt_ids):
            if track_id is not None and track_id not in id_to_token:
                id_to_token[track_id] = features[candidate_index].clone()
            elif track_id is not None and track_id in id_to_token:
                id_to_token[track_id] = features[candidate_index].clone()
    return examples


def forward_example(model, example: FrameExample, device):
    region = example.features.to(device)
    if example.slot_features.shape[0]:
        memory = model.memory_projector(example.slot_features.to(device))
    else:
        memory = torch.empty((0, model.hidden_dim), device=device)
    action_order = ["NEW", "NO_MATCH", "REJECT"]
    slots = model.readout.make_slot_tokens(memory, action_order)
    return model.readout(region, slots)


def evaluate(model, examples, device):
    totals = {"identity": 0, "identity_correct": 0, "admit": 0, "admit_correct": 0,
              "new_targets": 0, "reject_targets": 0}
    with torch.no_grad():
        for example in examples:
            outputs = forward_example(model, example, device)
            pred = outputs["identity_logits"].argmax(-1).cpu().tolist()
            admit = (torch.sigmoid(outputs["admit_logits"]) >= 0.5).cpu().tolist()
            for i, target in enumerate(example.identity_targets):
                if target is not None:
                    totals["identity"] += 1
                    totals["identity_correct"] += int(pred[i] == target)
                    totals["new_targets"] += int(example.option_tokens[target] == "NEW")
                    totals["reject_targets"] += int(example.option_tokens[target] == "REJECT")
                totals["admit"] += 1
                totals["admit_correct"] += int(bool(admit[i]) == bool(example.admit_targets[i]))
    totals["identity_accuracy"] = totals["identity_correct"] / max(totals["identity"], 1)
    totals["admit_accuracy"] = totals["admit_correct"] / max(totals["admit"], 1)
    return totals


class CachedProvider:
    def __init__(self, examples, device):
        self.examples = examples
        self.device = device
        self.index = 0

    def __call__(self, image, boxes):
        example = self.examples[self.index]
        self.index += 1
        return example.features.to(self.device), example.features[:, None, :].to(self.device), {
            "feature_source": "cached_G2_visual_forward",
            "cached_frame": example.frame,
        }


def replay(model, examples, device, model_sha):
    provider = CachedProvider(examples, device)
    bridge = G2FeatureBridge(provider, model_sha256=model_sha, device=str(device))
    runtime = CoupledGroundJEVRuntime(bridge, model)
    rows = []
    for index, example in enumerate(examples):
        output = runtime.step(object(), example.boxes, example.labels, frame_id=index,
                              image_key=example.image_path, task_query="person")
        rows.append({"frame": example.frame, "assignments": output["assignments"],
                     "decision_count": len(output["decisions"]),
                     "memory_tracks": len(runtime.memory.tracks)})
    return rows, bridge.forward_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-manifest", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--checkpoint", required=True,
                        help="immutable GroundAnything base bundle used with the G2 adapter")
    parser.add_argument("--adapter", required=True,
                        help="G2 adapter checkpoint; prevents silently using G0 visual weights")
    parser.add_argument("--runtime-root", required=True,
                        help="GroundJEV runtime checkout containing the native DLM loader")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=32)
    parser.add_argument("--train-frames", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ["LOCATEMOT_GROUNDANYTHING_ROOT"] = str(Path(args.source_root).resolve())
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Phase-B pilot")
    rows = load_candidate_rows(Path(args.candidate_manifest), args.max_frames)
    gt_by_frame = load_gt(Path(args.gt))
    start = time.perf_counter()
    bridge = G2FeatureBridge.from_native_clean_dlm(
        args.checkpoint, args.adapter, runtime_root=args.runtime_root,
        device=args.device, model_sha256=args.model_sha256, task_query="person",
    )
    examples = build_examples(rows, bridge, gt_by_frame, device, args.train_frames)
    model = GroundJEVModel(examples[0].features.shape[-1], hidden_dim=256).to(device)
    train_examples = examples[:args.train_frames]
    dev_examples = examples[args.train_frames:]
    before = {"train": evaluate(model, train_examples, device),
              "dev_temporal_holdout": evaluate(model, dev_examples, device)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    step = 0; history = []
    for epoch in range(args.epochs):
        for example in train_examples:
            outputs = forward_example(model, example, device)
            losses = structured_groundjev_losses(
                outputs, identity_targets=example.identity_targets,
                identity_valid=[x is not None for x in example.identity_targets],
                admit_targets=example.admit_targets, admit_valid=[True] * len(example.admit_targets),
                yesno_targets=example.yesno_targets, yesno_valid=[True] * len(example.yesno_targets),
                lifecycle_targets=example.lifecycle_targets, lifecycle_valid=[True] * len(example.lifecycle_targets),
                memory_targets=example.memory_targets, memory_valid=[True] * len(example.memory_targets),
            )
            total = losses["total"]
            if total is None or not torch.isfinite(total):
                raise RuntimeError(f"non-finite pilot loss at epoch {epoch} step {step}")
            optimizer.zero_grad(set_to_none=True); total.backward(); optimizer.step(); step += 1
            if step % 32 == 0:
                history.append({"step": step, "epoch": epoch + 1,
                                "loss": float(total.detach().cpu())})
    after = {"train": evaluate(model, train_examples, device),
             "dev_temporal_holdout": evaluate(model, dev_examples, device)}
    replay_rows, replay_forwards = replay(model, dev_examples, device, args.model_sha256)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "groundjev-phase-b-choice-yes-pilot", "step": step,
                "model_sha256": args.model_sha256, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "candidate_manifest": str(Path(args.candidate_manifest).resolve()),
                "teacher_forced_causal_memory": True}, output / f"adapter-step{step:06d}.pt")
    receipt = {
        "status": "PHASE_B_REAL_G2_FEATURE_READOUT_PILOT_COMPLETE",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "adapter": str(Path(args.adapter).resolve()),
        "runtime_root": str(Path(args.runtime_root).resolve()),
        "model_sha256": args.model_sha256,
        "candidate_manifest": str(Path(args.candidate_manifest).resolve()),
        "gt_manifest": str(Path(args.gt).resolve()), "frames": len(examples),
        "train_frames": len(train_examples), "temporal_dev_frames": len(dev_examples),
        "candidate_count": sum(len(x.boxes) for x in examples),
        "proposal_source": "prior_native_GroundAnything_candidates_repo_external_to_G2",
        "teacher_forced_causal_memory": True,
        "g2_forward_count": bridge.forward_count,
        "native_dlm_forward_count": bridge.provider.forward_count,
        "pilot_optimizer_steps": step, "before": before, "after": after,
        "replay": {"frames": len(replay_rows), "feature_forwards": replay_forwards,
                    "rows": replay_rows},
        "elapsed_seconds": time.perf_counter() - start,
        "scientific_claim": "bounded training/development control; no independent generalization or HOTA claim",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    (output / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
