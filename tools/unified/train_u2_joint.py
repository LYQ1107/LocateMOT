#!/usr/bin/env python3
"""Eligible-MOT-only initialized, frozen-foundation 4,000-step joint U2-B."""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.unified.build_u2_candidate_cache import DIRECTORY, PLAN, digest, write_json
from tools.unified.train_u2_mot_only import preflight as mot_preflight

RECIPE = ROOT / "outputs/unified/u2/protocol/joint_training_recipe.json"
OUT = ROOT / "outputs/unified/u2/joint"
EVALUATION_STEPS = (1000, 2000, 3000, 4000)


def recipe() -> dict:
    from locatemot.unified.data.u2_mot_clips import DATASET_PROBABILITIES
    return {"format": "locatemot-u2-joint-recipe-v1", "name": "U2_B_JOINT_MOT_OVMOT",
            "seed": 20261012, "world_size": 8, "clips_per_rank": 2, "global_clips": 16,
            "maximum_optimizer_steps": 4000, "evaluation_steps": list(EVALUATION_STEPS),
            "resume_checkpoint_every_steps": 500, "frames_per_clip": 8,
            "initialization": "strict load of every selected eligible MOT-only core tensor; no core reinitialization",
            "optimizer_initialization": "new U2-B AdamW moments; selected MOT-only weights preserved",
            "task_probabilities": {"mot": .65, "ovmot": .35},
            "mot_dataset_probabilities": DATASET_PROBABILITIES,
            "task_sampling": "one global task per step; independent video/start/stride per rank and clip",
            "mot_curriculum_counter": "6000 + joint step; retain eight frames and teacher probability .4",
            "ovmot_stride": "uniform source-image stride1..4; original physical filename/timestamp indices",
            "ovmot_supervision": "all frozen dense C-TAO base plus modal TAO TRAIN; labels outside candidate shards",
            "unknown_candidates": "keep inputs/model births; no automatic negative/NEW/alive/contrast label",
            "teacher_forcing_probability": .4,
            "foundation": "selected actual MM-GroundingDINO and shared BERT frozen; cached real features",
            "class_text": "selected shared U1 BERT/native projection256, full frozen1203 vocabulary",
            "classification_training": "base866 targets and softmax columns only; novel337 inference zero-shot",
            "task_embeddings": "two zero-initialized256-D embeddings in bounded semantic gate only; ordinary gate zero",
            "optimizer": {"type": "AdamW", "shared_core_lr": 5e-5, "ovmot_adapter_lr": 1e-4,
                          "candidate_and_semantic_projection_lr": 5e-5, "weight_decay": 1e-2, "gradient_clip": 1.},
            "precision": "FP32 parameters/losses; BF16 neural autocast",
            "identity_loss_weights": {"row": 1., "col": 1., "switch": .25, "alive": .25,
                                      "motion": .50, "identity_contrast": .25},
            "semantic_loss_weights": {"open_vocab_class": .50, "semantic_consistency": .25, "semantic_gate_reg": .10},
            "ordinary_negative_transfer_max_degradation": {"HOTA": .01, "AssA": .01},
            "ovmot_gate": {"TETA": 34., "AssocA": 37.}, "strong_gate": {"TETA": 36., "AssocA": 40.},
            "teta": {"official_unmodified": True, "exhaustive": False, "cluster_threshold": 50,
                     "max_candidates": 300, "scale": "0..100", "full_validation": True},
            "selection": ["higher TAO overall AssocA", "higher TETA", "higher ordinary macro AssA", "earlier step"],
            "bounded_u2_c_eligible_only_if": "ordinary retained, TETA>=31 and AssocA>=34, main OVMOT gate missed",
            "collective_wait_timeout_hours": 24, "formal_joint_training_launched": False,
            "official_test_labels_read": False}


def mot_reference(args) -> tuple[dict, dict, dict]:
    """Recompute all original gates rather than trusting a selected flag."""
    _, ordinary, simple = mot_preflight(args.cache_manifest, args.simple_baseline)
    if args.mot_only_selection is None or not args.mot_only_selection.is_file():
        raise ValueError("U2-B requires the frozen eligible MOT-only selection")
    provided = json.loads(args.mot_only_selection.read_text())
    if provided["status"] != "MOT_ONLY_SELECTED" or provided["selected"] is None:
        raise ValueError("MOT-only identity/ordinary gates failed; joint OVMOT is prohibited")
    from tools.unified.eval_u2_mot_only import select_mot_only
    actual = select_mot_only(args.mot_only_selection.parent, simple, args.cache_manifest)
    if actual != provided or actual["status"] != "MOT_ONLY_SELECTED":
        raise ValueError("MOT-only selection differs from all four complete original evaluation boundaries")
    return ordinary, simple, actual["selected"]


def preflight(args) -> tuple[dict, dict, dict, dict, dict, dict]:
    ordinary, simple, reference = mot_reference(args)
    for name in ("ovmot_cache", "ovmot_supervision", "class_text", "tao_gt_view"):
        path = getattr(args, name)
        if path is None or not path.is_file():
            raise ValueError(f"Joint optimization requires complete {name}")
    ovmot = json.loads(args.ovmot_cache.read_text())
    labels = json.loads(args.ovmot_supervision.read_text())
    foundation = ordinary["spec"]["foundation_sha256"]
    expected = {group["key"]: group for group in map(json.loads, PLAN.read_text().splitlines()) if group["dataset"] == "ovmot"}
    if (ovmot["status"] != "COMPLETE" or ovmot["spec"]["phase"] != "ovmot"
            or ovmot["spec"]["foundation_sha256"] != foundation
            or ovmot["spec"]["selection_sha256"] != ordinary["spec"]["selection_sha256"]
            or ovmot["spec"]["inventory_sha256"] != digest(PLAN)
            or {item["group"]["key"]: item["group"] for item in ovmot["groups"]} != expected
            or len(ovmot["groups"]) != len(expected)):
        raise ValueError("Joint OVMOT cache differs from full selected-foundation train/validation scope")
    if (labels["status"] != "COMPLETE" or labels["inventory_sha256"] != digest(PLAN)
            or labels["novel_training_gt_units"] != 0
            or {group["key"] for group in labels["groups"]} != {key for key, group in expected.items() if group["role"] == "train"}):
        raise ValueError("Joint identity labels must use the complete base-only original TRAIN scope")
    import torch
    text = torch.load(args.class_text, map_location="cpu", weights_only=False)
    if (text["reference_only"] or text["foundation_sha256"] != foundation
            or text["vocabulary_sha256"] != ovmot["spec"]["vocabulary_sha256"]
            or text["category_ids"] != list(range(1, 1204)) or text["embeddings"].shape != (1203, 256)):
        raise ValueError("Joint class features must come from the same selected frozen shared U1 BERT")
    gt = json.loads(args.tao_gt_view.read_text())
    raw = json.loads((DIRECTORY / "supervision_source_inventory.json").read_text())["ovmot"]["tao_val"]
    if (gt["status"] != "COMPLETE" or gt["source_sha256"] != raw["sha256"]
            or gt["view_images"] != raw["source_images"]
            or gt["view_sha256"] != digest(Path(gt["view_path"]))):
        raise ValueError("Joint TETA requires the complete explicitly mapped original TAO validation GT")
    from locatemot.unified.runtime.u2_teta_evaluation import verify_teta_source
    verify_teta_source()
    return ordinary, simple, reference, ovmot, labels, text


def optimizer_for_core(core):
    import torch
    adapter = list(core.ovmot.parameters())
    adapter_ids = {id(parameter) for parameter in adapter}
    shared = [parameter for parameter in core.parameters() if id(parameter) not in adapter_ids]
    return torch.optim.AdamW([{"params": shared, "lr": 5e-5, "name": "shared_core"},
                             {"params": adapter, "lr": 1e-4, "name": "ovmot_adapter"}], weight_decay=1e-2)


def initialize_from_mot(core, payload: dict, foundation: str) -> None:
    if (payload["format"] != "locatemot-u2-mot-only-checkpoint-v1"
            or payload["metadata"]["foundation_sha256"] != foundation):
        raise ValueError("Joint initialization is not the selected ordinary baseline foundation")
    core.load_state_dict(payload["core"], strict=True)


def save_checkpoint(model, optimizer, next_step: int, args, metadata: dict) -> Path:
    import torch
    import torch.distributed as dist
    rng_states = [None] * dist.get_world_size()
    dist.all_gather_object(rng_states, {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()})
    path = args.out / f"step_{next_step:04d}.pth"
    if dist.get_rank() == 0:
        payload = {"format": "locatemot-u2-joint-checkpoint-v1", "next_step": next_step,
                   "core": {key: value.detach().cpu() for key, value in model.module.core.state_dict().items()},
                   "optimizer": optimizer.state_dict(), "rank_rng_states": rng_states, "metadata": metadata}
        temporary = path.with_suffix(".pth.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
        write_json(args.out / "latest_checkpoint.json", {"path": str(path.resolve()), "next_step": next_step})
    dist.barrier()
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    for name in ("cache-manifest", "simple-baseline", "mot-only-selection", "ovmot-cache", "ovmot-supervision", "class-text", "tao-gt-view", "resume"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    registered = recipe()
    if args.register_only:
        if RECIPE.exists() and json.loads(RECIPE.read_text()) != registered:
            raise ValueError("The exact joint registration differs; no silent protocol change")
        write_json(RECIPE, registered)
        print(json.dumps({"status": "REGISTERED_NOT_LAUNCHED", "recipe": str(RECIPE)}))
        return 0
    try:
        ordinary, simple, reference, ovmot, labels, text = preflight(args)
    except (ValueError, FileNotFoundError) as error:
        print(json.dumps({"status": "HELD_BY_ELIGIBLE_MOT_ONLY_AND_SELECTED_CACHES", "reason": str(error),
                          "formal_joint_training_launched": False}), flush=True)
        return 2
    if not RECIPE.is_file() or json.loads(RECIPE.read_text()) != registered:
        raise ValueError("Register the exact joint experiment before optimizer launch")
    metadata = {"recipe_sha256": digest(RECIPE), "foundation_sha256": ordinary["spec"]["foundation_sha256"],
                "ordinary_cache_sha256": digest(args.cache_manifest), "ovmot_cache_sha256": digest(args.ovmot_cache),
                "base_supervision_sha256": digest(args.ovmot_supervision), "class_text_sha256": digest(args.class_text),
                "simple_baseline_sha256": digest(args.simple_baseline), "mot_only_selection_sha256": digest(args.mot_only_selection),
                "mot_only_checkpoint_sha256": reference["checkpoint_sha256"], "tao_gt_view_sha256": digest(args.tao_gt_view),
                "foundation_weights_modified": False, "world_size": 8,
                "implementation_sha256": {name: digest(ROOT / name) for name in (
                    "locatemot/unified/models/u2/shared_track_core.py", "locatemot/unified/models/u2/ovmot_adapter.py",
                    "locatemot/unified/data/u2_mot_clips.py", "locatemot/unified/data/u2_ovmot_clips.py",
                    "locatemot/unified/losses/u2_association.py", "locatemot/unified/losses/u2_semantic.py",
                    "locatemot/unified/runtime/u2_temporal_training.py", "locatemot/unified/runtime/u2_teta_evaluation.py",
                    "tools/unified/train_u2_joint.py", "tools/unified/eval_u2_joint.py")}}
    if args.preflight_only:
        print(json.dumps({"status": "READY", **metadata}))
        return 0
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from locatemot.unified.data.u2_mot_clips import CachedMOTClips
    from locatemot.unified.data.u2_ovmot_clips import CachedOVMOTClips, joint_task_for_step
    from locatemot.unified.runtime.u2_temporal_training import U2MOTTemporalTraining
    from tools.unified.eval_u2_joint import evaluate_checkpoint, select_joint
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(hours=24))
    if dist.get_world_size() != 8:
        raise ValueError("Formal joint U2-B requires eight GPU ranks")
    device = torch.device("cuda", local_rank)
    torch.manual_seed(registered["seed"])
    model = U2MOTTemporalTraining().to(device).train()
    initialize_from_mot(model.core, torch.load(reference["checkpoint"], map_location="cpu", weights_only=False), metadata["foundation_sha256"])
    model = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    optimizer = optimizer_for_core(model.module.core)
    torch.manual_seed(registered["seed"] + rank)
    next_step = 0
    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved["format"] != "locatemot-u2-joint-checkpoint-v1" or saved["metadata"] != metadata:
            raise ValueError("Joint resume differs from the original experiment/reference")
        model.module.core.load_state_dict(saved["core"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        next_step = saved["next_step"]
        torch.set_rng_state(saved["rank_rng_states"][rank]["cpu"])
        torch.cuda.set_rng_state(saved["rank_rng_states"][rank]["cuda"])
    source = json.loads((DIRECTORY / "supervision_source_inventory.json").read_text())["ordinary"]
    mot_sampler = CachedMOTClips(ordinary, source, seed=registered["seed"])
    ov_sampler = CachedOVMOTClips(ovmot, labels, seed=registered["seed"])
    class_text = text["embeddings"].to(device)
    base = torch.tensor([i for i, freq in enumerate(text["frequencies"]) if freq in {"f", "c"}], device=device)
    args.out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        write_json(args.out / "run_state.json", {"status": "RUNNING", "next_step": next_step,
                                               "maximum_optimizer_steps": 4000, **metadata})
    if next_step in EVALUATION_STEPS:
        evaluate_checkpoint(model.module.core, mot_sampler, ov_sampler, ordinary, ovmot, source,
                            class_text, reference, args, next_step, args.resume)
    for step in range(next_step, 4000):
        task = joint_task_for_step(step, registered["seed"])
        clips = mot_sampler.sample(6000 + step, rank, device) if task == "mot" else ov_sampler.sample(step, rank, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(clips, 6000 + step, task_type=task,
                           class_text=class_text if task == "ovmot" else None,
                           base_class_ids=base if task == "ovmot" else None)
        loss = output["losses"]["total"]
        loss.backward()
        finite = torch.tensor([int(torch.isfinite(loss) and all(
            p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))], device=device)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError(f"Nonfinite joint loss/gradient at original step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if rank == 0 and (step + 1) % 25 == 0:
            row = {"step": step + 1, "task": task, "clip_frames": len(clips[0]),
                   "teacher_forcing_probability": output["teacher_forcing_probability"],
                   "losses": {key: float(value.detach()) for key, value in output["losses"].items()}}
            with (args.out / "training_curve.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if (step + 1) % 500 == 0:
            checkpoint = save_checkpoint(model, optimizer, step + 1, args, metadata)
            if step + 1 in EVALUATION_STEPS:
                evaluate_checkpoint(model.module.core, mot_sampler, ov_sampler, ordinary, ovmot, source,
                                    class_text, reference, args, step + 1, checkpoint)
            if rank == 0:
                write_json(args.out / "run_state.json", {"status": "RUNNING", "next_step": step + 1,
                                                       "maximum_optimizer_steps": 4000, **metadata})
    if rank == 0:
        selected = select_joint(args.out, reference, metadata)
        write_json(args.out / "run_state.json", {"status": "COMPLETE", "optimizer_steps": 4000,
                                               "selection_status": selected["status"], **metadata})
        print(json.dumps({"status": selected["status"], "optimizer_steps": 4000}), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
