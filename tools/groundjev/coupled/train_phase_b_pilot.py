#!/usr/bin/env python3
"""Bounded Phase-B pilot over real native GroundAnything candidates.

The backbone is frozen.  Real G2 visual region tokens are extracted once per
source image, then only the typed readout is optimized for a small registered
TRAIN-only pilot.  This is a learnability gate, not a complete tracker or
benchmark.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from PIL import Image

from locatemot.groundjev.coupled.dynamic_identity_slots import build_dynamic_slots
from locatemot.groundjev.coupled.g2_feature_bridge import G2FeatureBridge
from locatemot.groundjev.coupled.groundjev_model import GroundJEVCoupledModel
from locatemot.groundjev.coupled.losses import JointLossWeights, joint_losses
from locatemot.groundjev.coupled.shared_ground_state import SharedGroundState


BASE = Path("/data2/user/LocateMOT_GroundJEV/models/GroundAnything-DLM")
SOURCE = Path("/data2/user/LocateMOT_GroundJEV/external/GroundAnything")
ADAPTER = Path("/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final/checkpoints/adapter-step011000.pt")
CORPUS = Path("/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/native_yesno_v2_smoke.jsonl")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split", "train") not in ("train", "val"):
                continue
            # Validation rows are retained for an audit but never used by this
            # pilot; the source builder's split is part of the contract.
            if row.get("split", "train") != "train":
                continue
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def prepare_example(row: dict, bridge: G2FeatureBridge, device: torch.device) -> dict | None:
    image_path = row.get("current_image")
    if image_path and not Path(image_path).is_file() and not Path(image_path).is_absolute():
        candidate = Path("/data2/user/LocateMOT_GroundJEV") / image_path
        if candidate.is_file():
            image_path = str(candidate)
    if not image_path or not Path(image_path).is_file():
        return None
    candidates = row.get("native_candidates") or []
    boxes = [candidate.get("box") for candidate in candidates
             if isinstance(candidate, dict) and candidate.get("box") is not None]
    if not boxes:
        return None
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    state = bridge.encode(image, boxes, frame_id=row.get("frame_id", 0), image_key=str(image_path),
                          task_query=str(row.get("task", "")), metadata={
                              "dataset": str(row.get("video_id", "unknown")),
                              "split": "TRAIN",
                              "candidate_source": "native_groundanything_candidate_export",
                          })
    proposal = (row.get("identity_yesno_targets") or {}).get("proposal") or {}
    candidate_index = int(proposal.get("candidate_index", 0) or 0)
    candidate_index = min(max(candidate_index, 0), len(boxes) - 1)
    slots = build_dynamic_slots(row.get("past_track_memory") or [], max_history=31)
    target_token = row.get("identity_choice_target")
    identity_target = None
    if target_token is not None:
        for index, slot in enumerate(slots):
            if slot.token == str(target_token):
                identity_target = index
                break
    yesno = (row.get("identity_yesno_targets") or {}).get("yes")
    return {
        "region_tokens": state.region_tokens.detach(),
        "boxes": [list(map(float, box)) for box in boxes],
        "slots": [slots for _ in boxes],
        "candidate_index": candidate_index,
        "identity_target": identity_target,
        "yesno_target": float(yesno) if yesno is not None else None,
        "source_row": row.get("id"),
        "image": str(image_path),
        "state": state.audit_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=Path, default=CORPUS)
    parser.add_argument("--max-rows", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("Phase-B pilot requires CUDA")
    args.out.mkdir(parents=True, exist_ok=False)
    rows = load_rows(args.rows, args.max_rows)
    bridge = G2FeatureBridge(checkpoint=BASE, device=args.device,
                             g2_adapter_checkpoint=ADAPTER, local_source_root=SOURCE)
    device = torch.device(args.device)
    examples = []
    for row in rows:
        example = prepare_example(row, bridge, device)
        if example is not None:
            examples.append(example)
    if not examples:
        raise RuntimeError("no legal native TRAIN examples with local images")
    model = GroundJEVCoupledModel(bridge.feature_dimension, hidden_dim=256).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    logs = []
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total = None
        components = {}
        identity_logits = []
        identity_targets = []
        yesno_logits = []
        yesno_targets = []
        yesno_valid = []
        for example in examples:
            batch = model(
                SharedGroundState(
                    frame_id=example["state"]["frame_id"], image_key=example["image"],
                    model_sha256=example["state"]["model_sha256"],
                    boxes=tuple(tuple(box) for box in example["boxes"]),
                    vision_tokens=torch.empty(1, bridge.feature_dimension, 1, 1, device=device),
                    region_tokens=example["region_tokens"].to(device),
                    image_grid_thw=tuple(example["state"]["image_grid_thw"]),
                    spatial_layout=tuple(example["state"]["spatial_layout"]),
                    image_size=tuple(example["state"]["image_size"]),
                    task_query="",
                    metadata={"visual_forward_calls_for_frame": 1},
                ),
                example["slots"], memories={}, memory_enabled=True,
            )
            candidate = batch.decisions[example["candidate_index"]]
            identity_logits.append(candidate.identity_logits)
            identity_targets.append(example["identity_target"])
            if example["yesno_target"] is not None:
                yesno_logits.append(candidate.yesno_logit)
                yesno_targets.append(example["yesno_target"])
                yesno_valid.append(True)
        yesno_logits_tensor = torch.stack(yesno_logits) if yesno_logits else None
        yesno_targets_tensor = torch.tensor(yesno_targets, device=device) if yesno_targets else None
        yesno_valid_tensor = torch.tensor(yesno_valid, dtype=torch.bool, device=device) if yesno_valid else None
        losses = joint_losses(
            identity_logits=identity_logits,
            identity_targets=identity_targets,
            yesno_logits=yesno_logits_tensor,
            yesno_targets=yesno_targets_tensor,
            yesno_valid=yesno_valid_tensor,
            weights=JointLossWeights(grounding=0.0, identity_choice=1.0, yesno=1.0),
        )
        total = losses["total"]
        if total is None or not torch.isfinite(total):
            raise RuntimeError(f"non-finite Phase-B loss at step {step}")
        total.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if grad_norm <= 0 or not torch.isfinite(torch.tensor(grad_norm)):
            raise RuntimeError(f"invalid Phase-B gradient at step {step}: {grad_norm}")
        optimizer.step()
        logs.append({"step": step, "loss": float(total.detach()),
                     "identity_choice": float(losses["identity_choice"].detach()) if losses["identity_choice"] is not None else None,
                     "yesno": float(losses["yesno"].detach()) if losses["yesno"] is not None else None,
                     "gradient_norm": grad_norm})
    checkpoint = args.out / "readout_step_final.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "steps": args.steps, "backbone_frozen": True}, checkpoint)
    receipt = {
        "status": "PHASE_B_REAL_NATIVE_G2_READOUT_PILOT_COMPLETE",
        "backbone": "GroundAnything G2 step11000 projector visual path",
        "g2_adapter": str(ADAPTER), "g2_adapter_sha256": sha256(ADAPTER),
        "corpus": str(args.rows), "corpus_sha256": sha256(args.rows),
        "train_split_only": True, "examples": len(examples),
        "source_rows": [example["source_row"] for example in examples],
        "images": [example["image"] for example in examples],
        "visual_forward_calls": bridge.forward_calls,
        "typed_readout_parameters": sum(p.numel() for p in model.parameters()),
        "steps": args.steps, "logs": logs,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "elapsed_seconds": time.perf_counter() - start,
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "quality_claim": False,
        "note": "Frozen G2 visual evidence and native TRAIN candidates; bounded learnability pilot, not a MOT/RMOT/OVMOT benchmark.",
    }
    (args.out / "phase_b_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
