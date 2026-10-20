#!/usr/bin/env python3
"""Bounded fixed-candidate repair study on a registered TRAIN clip.

This is deliberately a training-scope diagnostic.  It uses the immutable
native GroundAnything region cache and a causal GT-derived memory only for
the *training* half of one MOT20 TRAIN clip.  The later contiguous frames are
held out before any optimization and are replayed through the real coupled
runtime.  The script never loads a second VLM and never changes the formal G2
run.

The repair addresses a measured domain/class imbalance: the earlier MOT17
pilot was trained with a much smaller reject pool, while the MOT20 cache has
many more native proposals per frame.  Identity-choice cross entropy is
weighted from the fixed training-half counts (inverse-frequency weights capped
at 4x); the cap and split are part of the receipt and are not tuned on the
held-out half.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from locatemot.groundjev.coupled.g2_feature_bridge import G2FeatureBridge
from locatemot.groundjev.coupled.groundjev_model import GroundJEVModel
from locatemot.groundjev.coupled.identity_memory import IdentityMemoryState
from locatemot.groundjev.coupled.dynamic_identity_slots import build_dynamic_slots
from tools.groundjev.coupled.evaluate_fixed_candidate_clip import (
    CachedNativeProvider,
    load_gt,
    load_rows,
    run_trackeval,
    sha256_file,
    write_trackeval_inputs,
)


MODEL_SHA = "3cf09355cde8cf4877cdc76e0a72301d572c1d6263b2831f8a03368285bb65e2"
ACTION_ORDER = ["NEW", "NO_MATCH", "REJECT"]


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(aa + bb - inter, 1e-6)


@dataclass
class Example:
    frame: int
    image_path: str
    boxes: list[tuple[float, float, float, float]]
    labels: list[str]
    features: torch.Tensor
    slot_features: torch.Tensor
    option_tokens: list[str]
    identity_targets: list[int]
    admit_targets: list[float]
    yesno_targets: list[float]
    lifecycle_targets: list[int]
    memory_targets: list[float]


def candidate_targets(
    boxes: list[tuple[float, float, float, float]],
    gt_rows: list[tuple[int, tuple[float, float, float, float]]],
    slot_ids: list[int],
) -> tuple[list[int], list[float], list[float], list[int], list[float], dict[int, int]]:
    position = {track_id: i for i, track_id in enumerate(slot_ids)}
    action_start = len(slot_ids)
    identity: list[int] = []
    admit: list[float] = []
    yesno: list[float] = []
    life: list[int] = []
    memory: list[float] = []
    seen: set[int] = set()
    births: dict[int, int] = {}
    for candidate_index, box in enumerate(boxes):
        best_iou, best_id = max(
            ((iou(box, gt_box), track_id) for track_id, gt_box in gt_rows),
            default=(0.0, None),
        )
        if best_iou < 0.5 or best_id is None or best_id in seen:
            identity.append(action_start + 2)  # explicit REJECT
            admit.append(0.0); yesno.append(0.0); memory.append(0.0)
        elif best_id in position:
            identity.append(position[best_id])
            admit.append(1.0); yesno.append(1.0); memory.append(1.0)
            seen.add(best_id)
        else:
            identity.append(action_start)  # explicit NEW
            admit.append(1.0); yesno.append(0.0); memory.append(1.0)
            seen.add(best_id); births[best_id] = candidate_index
        life.append(0)  # ACTIVE is the only label supported by this bounded study.
    return identity, admit, yesno, life, memory, births


def load_cached_examples(rows: list[dict[str, Any]], gt: dict[int, list[tuple[int, tuple[float, float, float, float]]]], train_frames: int, clip: Path) -> tuple[list[Example], dict[str, int]]:
    id_to_feature: dict[int, torch.Tensor] = {}
    examples: list[Example] = []
    class_counts: dict[str, int] = {}
    for row_index, row in enumerate(rows):
        feature_path = clip / str(row["native_region_features_npz"])
        with np.load(feature_path) as archive:
            features = torch.from_numpy(np.asarray(archive["regions"], dtype=np.float32).copy())
        boxes = [tuple(float(v) for v in c["box"]) for c in row.get("candidates", [])]
        labels = [str(c.get("label", "person")) for c in row.get("candidates", [])]
        if features.ndim != 2 or features.shape != (len(boxes), 2560):
            raise ValueError(f"feature shape mismatch at frame {row['frame']}: {tuple(features.shape)}")
        slot_ids = sorted(id_to_feature)
        slot_features = (torch.stack([id_to_feature[x] for x in slot_ids])
                         if slot_ids else torch.empty((0, 2560), dtype=torch.float32))
        identity, admit, yesno, life, memory, births = candidate_targets(
            boxes, gt.get(int(row["frame"]), []), slot_ids)
        options = [str(x) for x in slot_ids] + ACTION_ORDER
        if row_index < train_frames:
            for target, token in zip(identity, options_for_targets(identity, options)):
                class_counts[token] = class_counts.get(token, 0) + 1
        examples.append(Example(int(row["frame"]), str(row["image_path"]), boxes, labels,
                                features, slot_features, options, identity, admit, yesno,
                                life, memory))
        for track_id, candidate_index in births.items():
            id_to_feature[track_id] = features[candidate_index].detach().clone()
        # Existing IDs get the latest causal observation after the frame.
        for candidate_index, track_id in enumerate(match_ids(boxes, gt.get(int(row["frame"]), []))):
            if track_id is not None:
                id_to_feature[track_id] = features[candidate_index].detach().clone()
    return examples, class_counts


def match_ids(boxes, gt_rows):
    result = []
    for box in boxes:
        best_iou, best_id = max(((iou(box, gt_box), track_id) for track_id, gt_box in gt_rows), default=(0.0, None))
        result.append(int(best_id) if best_iou >= 0.5 and best_id is not None else None)
    return result


def options_for_targets(targets, options):
    return [options[int(x)] for x in targets]


def forward(model: GroundJEVModel, example: Example, device: torch.device) -> dict[str, Any]:
    region = example.features.to(device)
    memory = (model.memory_projector(example.slot_features.to(device))
              if example.slot_features.shape[0]
              else torch.empty((0, model.hidden_dim), device=device))
    slots = model.readout.make_slot_tokens(memory, ACTION_ORDER)
    return model.readout(region, slots)


def metrics(model, examples: list[Example], device: torch.device) -> dict[str, float | int]:
    total = correct = admit_total = admit_correct = 0
    with torch.no_grad():
        for example in examples:
            out = forward(model, example, device)
            pred = out["identity_logits"].argmax(-1).cpu().tolist()
            admit = (torch.sigmoid(out["admit_logits"]) >= 0.5).cpu().tolist()
            total += len(pred)
            correct += sum(int(a == b) for a, b in zip(pred, example.identity_targets))
            admit_total += len(admit)
            admit_correct += sum(int(bool(a) == bool(b)) for a, b in zip(admit, example.admit_targets))
    return {"identity": total, "identity_correct": correct,
            "identity_accuracy": correct / max(total, 1),
            "admit": admit_total, "admit_correct": admit_correct,
            "admit_accuracy": admit_correct / max(admit_total, 1)}


def class_weights(examples: list[Example], cap: float = 4.0) -> tuple[torch.Tensor, list[str]]:
    counts: dict[str, int] = {}
    options: set[str] = set()
    for ex in examples:
        options.update(ex.option_tokens)
        for target in ex.identity_targets:
            token = ex.option_tokens[target]
            counts[token] = counts.get(token, 0) + 1
    names = sorted(options, key=lambda x: (x not in ACTION_ORDER, x))
    total = sum(counts.values())
    weights = []
    for name in names:
        value = total / max(len(names) * counts.get(name, 1), 1)
        weights.append(min(float(cap), value))
    return torch.tensor(weights, dtype=torch.float32), names


def replay_and_eval(model, examples: list[Example], clip: Path, gt: dict[int, list[tuple[int, tuple[float, float, float, float]]]], output: Path, device: torch.device) -> dict[str, Any]:
    provider = CachedNativeProvider(clip, [
        {"frame": ex.frame, "image_path": ex.image_path, "candidates": [{"box": list(box)} for box in ex.boxes],
         "native_region_features_npz": str(Path(ex.image_path).name)} for ex in examples
    ], device)
    # CachedNativeProvider resolves the archive path from each row, so use a
    # tiny clip-relative adapter with the original archive references instead.
    class Provider:
        def __init__(self, exs): self.exs, self.i = exs, 0
        def encode_frame(self, image, boxes, *, image_key, task_query=None):
            del image, image_key, task_query
            ex = self.exs[self.i]; self.i += 1
            if len(boxes) != len(ex.boxes): raise ValueError("replay geometry changed")
            return {"region_tokens": ex.features.to(device), "local_tokens": None,
                    "contextual_tokens": None, "metadata": {"cached_replay": True}}
    del provider
    bridge = G2FeatureBridge(Provider(examples), model_sha256=MODEL_SHA, device=str(device))
    from locatemot.groundjev.coupled.g2_shared_runtime import CoupledGroundJEVRuntime
    runtime = CoupledGroundJEVRuntime(bridge, model)
    predictions = []
    with torch.no_grad():
        for ex in examples:
            result = runtime.step(None, ex.boxes, ex.labels, frame_id=ex.frame,
                                  image_key=ex.image_path, task_query="person")
            for candidate_index, track_id in result["assignments"].items():
                predictions.append({"frame": ex.frame, "track_id": int(track_id),
                                    "candidate_index": int(candidate_index),
                                    "box": list(ex.boxes[int(candidate_index)])})
    eval_root, pred_path = write_trackeval_inputs(output, "MOT20-03-repair-dev", [
        {"frame": ex.frame, "image_path": ex.image_path} for ex in examples
    ], gt, predictions)
    return {"metrics": run_trackeval(eval_root, "MOT20-03-repair-dev", len(examples)),
            "predictions": len(predictions), "frames_with_predictions": len({x["frame"] for x in predictions}),
            "feature_forwards": bridge.forward_count, "prediction_path": str(pred_path)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--train-frames", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    rows = load_rows(args.clip)
    if not 0 < args.train_frames < len(rows): raise ValueError("train split must leave a held-out suffix")
    gt = load_gt(args.gt, [int(row["frame"]) for row in rows])
    start = time.perf_counter(); device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    examples, counts = load_cached_examples(rows, gt, args.train_frames, args.clip)
    train, dev = examples[:args.train_frames], examples[args.train_frames:]
    model = GroundJEVModel(2560, hidden_dim=256).to(device)
    before = {"train": metrics(model, train, device), "temporal_dev": metrics(model, dev, device)}
    weights, names = class_weights(train)
    index = {name: i for i, name in enumerate(names)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history=[]; step=0
    for epoch in range(args.epochs):
        for ex in train:
            if not ex.boxes:
                continue
            out=forward(model,ex,device)
            target=torch.tensor(ex.identity_targets,device=device)
            row_weights=weights[[index[ex.option_tokens[int(t)]] for t in ex.identity_targets]].to(device)
            identity=F.cross_entropy(out["identity_logits"],target,weight=None,reduction="none")
            identity=(identity*row_weights).mean()
            admit=F.binary_cross_entropy_with_logits(out["admit_logits"],torch.tensor(ex.admit_targets,device=device))
            yesno=F.binary_cross_entropy_with_logits(out["yesno_logits"],torch.tensor(ex.yesno_targets,device=device))
            life=F.cross_entropy(out["lifecycle_logits"],torch.tensor(ex.lifecycle_targets,device=device))
            memory=F.binary_cross_entropy_with_logits(out["memory_logits"],torch.tensor(ex.memory_targets,device=device))
            loss=identity+admit+yesno+life+memory
            if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss at step {step}")
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); step += 1
            if step % 32 == 0: history.append({"step":step,"epoch":epoch+1,"loss":float(loss.detach().cpu())})
    after={"train":metrics(model,train,device),"temporal_dev":metrics(model,dev,device)}
    args.output.mkdir(parents=True)
    ckpt=args.output/f"adapter-step{step:06d}.pt"
    torch.save({"format":"groundjev-fixed-candidate-repair-v1","step":step,"model_sha256":MODEL_SHA,
                "model_state":model.state_dict(),"optimizer_state":optimizer.state_dict(),
                "train_frames":args.train_frames,"epochs":args.epochs,"identity_class_weights":dict(zip(names,weights.tolist()))},ckpt)
    replay=replay_and_eval(model,dev,args.clip,gt,args.output,device)
    receipt={"status":"FIXED_CANDIDATE_REPAIR_TRAIN_AND_DEV_COMPLETE","task":"MOT","dataset":"MOT20",
             "sequence":"MOT20-03","scope":"registered_train_only_temporal_holdout_not_benchmark_evaluation",
             "frames":len(rows),"train_frames":len(train),"temporal_dev_frames":len(dev),
             "candidate_count":sum(len(x.boxes) for x in examples),"candidate_recall_not_measured_by_model":True,
             "candidate_manifest_sha256":sha256_file(args.clip/"frame_candidates.jsonl"),"gt_path":str(args.gt),
             "backbone_model_sha256":MODEL_SHA,"checkpoint":str(ckpt),"checkpoint_sha256":sha256_file(ckpt),
             "class_counts":counts,"class_weights":dict(zip(names,weights.tolist())),"weight_cap":4.0,
             "optimizer_steps":step,"before":before,"after":after,"replay":replay,
             "second_vlm_loaded":False,"single_backbone":True,"elapsed_seconds":time.perf_counter()-start,
             "scientific_claim":"bounded TRAIN-scope repair diagnostic; no independent generalization claim"}
    (args.output/"official_metrics.json").write_text(json.dumps(receipt,indent=2,ensure_ascii=False)+"\n")
    (args.output/"training_history.json").write_text(json.dumps(history,indent=2)+"\n")
    print(json.dumps(receipt,ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
