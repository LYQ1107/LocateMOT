#!/usr/bin/env python3
"""Run registered Phase-B readout ablations on one frozen G2 feature cache.

The three arms use exactly the same native TRAIN examples and precomputed
GroundAnything G2 region tokens.  B0 trains identity choice, B1 adds the
binary verifier loss, and B2 adds the available causal lifecycle/admission
supervision.  This is a bounded engineering/development study; it never reads
official TEST labels and does not report MOT metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image

from locatemot.groundjev.coupled.dynamic_identity_slots import build_dynamic_slots
from locatemot.groundjev.coupled.g2_feature_bridge import G2FeatureBridge
from locatemot.groundjev.coupled.groundjev_model import GroundJEVCoupledModel
from locatemot.groundjev.coupled.losses import JointLossWeights, joint_losses
from locatemot.groundjev.coupled.shared_ground_state import SharedGroundState
from locatemot.groundjev.coupled.typed_decisions import LIFECYCLE_CHOICES


BASE = Path("/data2/user/LocateMOT_GroundJEV/models/GroundAnything-DLM")
SOURCE = Path("/data2/user/LocateMOT_GroundJEV/external/GroundAnything")
ADAPTER = Path(
    "/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/"
    "formal_clean_coverage_v2_final/checkpoints/adapter-step011000.pt"
)
CORPUS = Path(
    "/data2/user/LocateMOT_GroundJEV/outputs/groundjev/coupled/"
    "joint_identity_v1_phaseb.jsonl"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_file():
        return path
    candidate = Path("/data2/user/LocateMOT_GroundJEV") / path
    return candidate if candidate.is_file() else None


def load_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split", "train") != "train":
                continue
            if not row.get("native_candidates"):
                continue
            if _image_path(row.get("current_image")) is None:
                continue
            rows.append(row)
    # Balance the fixed ablation set.  A first version selected only positive
    # identity rows, making Yes/No a trivial all-positive problem and leaving
    # B2 without a lifecycle target.  Keep the same deterministic strata for
    # every arm: identity +/- , lifecycle +/- , and native verifier +/-.
    strata: dict[str, list[dict]] = {key: [] for key in (
        "identity_pos", "identity_neg", "lifecycle_pos", "lifecycle_neg",
        "verifier_pos", "verifier_neg")}
    for row in rows:
        masks = row.get("label_validity_masks", {})
        target = (row.get("identity_yesno_targets") or {}).get("yes")
        family = str(row.get("proposal_source", ""))
        if masks.get("identity_choice"):
            key = "identity_pos" if float(target) > 0.5 else "identity_neg"
        elif masks.get("lifecycle"):
            key = "lifecycle_pos" if float(target) > 0.5 else "lifecycle_neg"
        else:
            key = "verifier_pos" if float(target) > 0.5 else "verifier_neg"
            if family not in {"native_rmot_candidate_export",
                              "groundanything_native_identity_candidate",
                              "groundanything_native_lifecycle_candidate"}:
                key = "verifier_pos" if float(target) > 0.5 else "verifier_neg"
        strata[key].append(row)
    for values in strata.values():
        values.sort(key=lambda row: str(row.get("id", "")))
    quota = max(1, int(limit) // len(strata))
    selected = []
    for key in ("identity_pos", "identity_neg", "lifecycle_pos", "lifecycle_neg",
                "verifier_pos", "verifier_neg"):
        selected.extend(strata[key][:quota])
    if len(selected) < int(limit):
        used = {str(row.get("id")) for row in selected}
        for row in rows:
            if str(row.get("id")) in used:
                continue
            selected.append(row)
            if len(selected) >= int(limit):
                break
    return selected[: int(limit)]


def prepare_example(row: dict, bridge: G2FeatureBridge) -> dict | None:
    image_path = _image_path(row.get("current_image"))
    candidates = row.get("native_candidates") or []
    boxes = [candidate.get("box") for candidate in candidates
             if isinstance(candidate, dict) and candidate.get("box") is not None]
    if image_path is None or not boxes:
        return None
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    state = bridge.encode(
        image, boxes, frame_id=row.get("frame_id", 0), image_key=str(image_path),
        task_query=str(row.get("task_query", row.get("task", ""))), metadata={
            "dataset": str(row.get("video_id", "unknown")),
            "split": "TRAIN",
            "candidate_source": "native_groundanything_candidate_export",
        })
    slots = build_dynamic_slots(row.get("past_track_memory") or [], max_history=31)
    proposal = (row.get("identity_yesno_targets") or {}).get("proposal") or {}
    candidate_index = int(proposal.get("candidate_index", 0) or 0)
    candidate_index = min(max(candidate_index, 0), len(boxes) - 1)
    target_token = row.get("identity_choice_target")
    identity_target = None
    if target_token is not None:
        for index, slot in enumerate(slots):
            if slot.token == str(target_token):
                identity_target = index
                break
    yes = (row.get("identity_yesno_targets") or {}).get("yes")
    life = row.get("lifecycle_targets") or {}
    lifecycle_target = None
    if life.get("valid") and life.get("value") in LIFECYCLE_CHOICES:
        lifecycle_target = LIFECYCLE_CHOICES.index(life["value"])
    return {
        "state": state.detached(),
        "boxes": [list(map(float, box)) for box in boxes],
        "slots": [slots for _ in boxes],
        "candidate_index": candidate_index,
        "identity_target": identity_target,
        "yesno_target": float(yes) if yes is not None else None,
        # For this pilot the registered binary proposal verifier is also the
        # candidate-admission target.  It is explicit supervision, not a GT
        # box or an inferred tracker label.
        "admit_target": float(yes) if yes is not None else None,
        "lifecycle_target": lifecycle_target,
        "source_row": row.get("id"),
        "family": row.get("proposal_source"),
        "image": str(image_path),
    }


def _stack(values: list[torch.Tensor], *, device: torch.device) -> torch.Tensor | None:
    return torch.stack(values).to(device) if values else None


def train_arm(mode: str, examples: list[dict], bridge: G2FeatureBridge,
              *, steps: int, out: Path, seed: int) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)
    device = torch.device(bridge.device)
    model = GroundJEVCoupledModel(bridge.feature_dimension, hidden_dim=256).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    logs = []
    started = time.perf_counter()
    for step in range(1, int(steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        identity_logits, identity_targets = [], []
        yesno_logits, yesno_targets, yesno_valid = [], [], []
        admit_logits, admit_targets, admit_valid = [], [], []
        lifecycle_logits, lifecycle_targets, lifecycle_valid = [], [], []
        for example in examples:
            batch = model(example["state"], example["slots"], memory_enabled=True)
            candidate = batch.decisions[example["candidate_index"]]
            identity_logits.append(candidate.identity_logits)
            identity_targets.append(example["identity_target"])
            if example["yesno_target"] is not None:
                yesno_logits.append(candidate.yesno_logit)
                yesno_targets.append(example["yesno_target"])
                yesno_valid.append(True)
                admit_logits.append(candidate.admission_logits[0])
                admit_targets.append(example["admit_target"])
                admit_valid.append(True)
            if example["lifecycle_target"] is not None:
                lifecycle_logits.append(candidate.lifecycle_logits)
                lifecycle_targets.append(example["lifecycle_target"])
                lifecycle_valid.append(True)
        device = next(model.parameters()).device
        yesno_logits_t = _stack(yesno_logits, device=device)
        yesno_targets_t = (torch.tensor(yesno_targets, device=device)
                           if yesno_targets else None)
        yesno_valid_t = (torch.tensor(yesno_valid, dtype=torch.bool, device=device)
                         if yesno_valid else None)
        admit_logits_t = _stack(admit_logits, device=device)
        admit_targets_t = (torch.tensor(admit_targets, device=device)
                           if admit_targets else None)
        admit_valid_t = (torch.tensor(admit_valid, dtype=torch.bool, device=device)
                         if admit_valid else None)
        lifecycle_logits_t = _stack(lifecycle_logits, device=device)
        lifecycle_targets_t = (torch.tensor(lifecycle_targets, dtype=torch.long, device=device)
                               if lifecycle_targets else None)
        lifecycle_valid_t = (torch.tensor(lifecycle_valid, dtype=torch.bool, device=device)
                             if lifecycle_valid else None)
        weights = JointLossWeights(
            grounding=0.0,
            identity_choice=1.0,
            yesno=0.0 if mode == "B0_choice_only" else 1.0,
            lifecycle=0.25 if mode == "B2_choice_yesno_lifecycle" else 0.0,
            memory=0.0,
        )
        losses = joint_losses(
            identity_logits=identity_logits,
            identity_targets=identity_targets,
            admit_logits=admit_logits_t if mode == "B2_choice_yesno_lifecycle" else None,
            admit_targets=admit_targets_t if mode == "B2_choice_yesno_lifecycle" else None,
            admit_valid=admit_valid_t if mode == "B2_choice_yesno_lifecycle" else None,
            lifecycle_logits=lifecycle_logits_t if mode == "B2_choice_yesno_lifecycle" else None,
            lifecycle_targets=lifecycle_targets_t if mode == "B2_choice_yesno_lifecycle" else None,
            lifecycle_valid=lifecycle_valid_t if mode == "B2_choice_yesno_lifecycle" else None,
            yesno_logits=yesno_logits_t,
            yesno_targets=yesno_targets_t,
            yesno_valid=yesno_valid_t,
            weights=weights,
        )
        total = losses["total"]
        if total is None or not torch.isfinite(total):
            raise RuntimeError(f"non-finite {mode} loss at step {step}")
        total.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        if grad_norm <= 0 or not torch.isfinite(torch.tensor(grad_norm)):
            raise RuntimeError(f"invalid {mode} gradient at step {step}: {grad_norm}")
        optimizer.step()
        if step in {1, int(steps) // 2, int(steps)}:
            logs.append({
                "step": step,
                "loss": float(total.detach()),
                "identity_choice": (float(losses["identity_choice"].detach())
                                     if losses["identity_choice"] is not None else None),
                "yesno": (float(losses["yesno"].detach())
                           if losses["yesno"] is not None else None),
                "lifecycle": (float(losses["lifecycle"].detach())
                              if losses["lifecycle"] is not None else None),
                "gradient_norm": grad_norm,
            })
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = out / "readout_step_final.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "mode": mode, "steps": int(steps), "backbone_frozen": True}, checkpoint)
    return {
        "mode": mode,
        "examples": len(examples),
        "steps": int(steps),
        "logs": logs,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "elapsed_seconds": time.perf_counter() - started,
        "typed_readout_parameters": sum(p.numel() for p in model.parameters()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rows", type=Path, default=CORPUS)
    parser.add_argument("--max-examples", type=int, default=24)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() and str(args.device).startswith("cuda"):
        raise RuntimeError("Phase-B ablations require CUDA")
    if args.out.exists():
        raise FileExistsError(args.out)
    bridge = G2FeatureBridge(checkpoint=BASE, device=args.device,
                             g2_adapter_checkpoint=ADAPTER, local_source_root=SOURCE)
    rows = load_rows(args.rows, args.max_examples)
    examples = []
    for row in rows:
        example = prepare_example(row, bridge)
        if example is not None:
            examples.append(example)
        if len(examples) >= args.max_examples:
            break
    if not examples:
        raise RuntimeError("no legal native TRAIN examples with local images")
    args.out.mkdir(parents=True, exist_ok=False)
    # Remove the temporary root marker before per-arm train_arm creates its
    # output directories; this keeps the fail-fast exists contract simple.
    args.out.rmdir()
    arms = []
    for mode in ("B0_choice_only", "B1_choice_yesno",
                 "B2_choice_yesno_lifecycle"):
        arms.append(train_arm(mode, examples, bridge, steps=args.steps,
                              out=args.out / mode, seed=20261021))
    receipt = {
        "status": "PHASE_B_REAL_NATIVE_G2_ABLATIONS_COMPLETE",
        "backbone": "GroundAnything G2 step11000 projector visual path",
        "g2_adapter": str(ADAPTER), "g2_adapter_sha256": sha256(ADAPTER),
        "corpus": str(args.rows), "corpus_sha256": sha256(args.rows),
        "train_split_only": True,
        "examples": len(examples),
        "source_rows": [example["source_row"] for example in examples],
        "families": [example["family"] for example in examples],
        "images": [example["image"] for example in examples],
        "visual_forward_calls": bridge.forward_calls,
        "arms": arms,
        "quality_claim": False,
        "official_test_labels_read": False,
        "future_evidence_in_model_input": False,
        "groundtruth_teacher_injection": False,
        "note": "Fixed native TRAIN candidates; bounded readout ablations, not a MOT/RMOT/OVMOT benchmark.",
    }
    (args.out / "phase_b_ablations_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
