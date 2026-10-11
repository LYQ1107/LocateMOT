#!/usr/bin/env python3
"""Gated eight-GPU, 6,000-step MOT-only run of the exact shared core."""
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
from tools.unified.select_u2_perception_adaptation import verify_selected_foundation

RECIPE = ROOT / "outputs/unified/u2/protocol/mot_only_training_recipe.json"
OUT = ROOT / "outputs/unified/u2/mot_only"
EVALUATION_STEPS = (1000, 2500, 4000, 6000)


def recipe() -> dict:
    from locatemot.unified.data.u2_mot_clips import DATASET_PROBABILITIES
    return {"format": "locatemot-u2-mot-only-recipe-v1", "name": "U2_MOT_ONLY_BASELINE",
            "seed": 20261012, "world_size": 8, "clips_per_rank": 2, "global_clips": 16,
            "maximum_optimizer_steps": 6000, "evaluation_steps": list(EVALUATION_STEPS),
            "resume_checkpoint_every_steps": 500,
            "curriculum": [{"start": 0, "end_exclusive": 1000, "frames": 2},
                           {"start": 1000, "end_exclusive": 2500, "frames": 4},
                           {"start": 2500, "end_exclusive": 4000, "frames": 6},
                           {"start": 4000, "end_exclusive": 6000, "frames": 8}],
            "teacher_forcing": "1 through step 500; linear to .4 at 2500; .4 thereafter; Bernoulli per frame",
            "stride": "uniform integer 1..min(4,ceil(source_fps/7.5)); forward only",
            "dataset_probabilities": DATASET_PROBABILITIES,
            "dataset_sampling": "same dataset per global step, independent video/start/stride per rank and clip",
            "task_type": "mot", "foundation": "frozen selected MM-GroundingDINO, cached features only",
            "architecture": "SharedTrackCore(hidden_dim=256,max_age=30); same as joint core",
            "optimizer": {"type": "AdamW", "lr": 1e-4, "weight_decay": 1e-2, "gradient_clip": 1.},
            "precision": "FP32 parameters and six losses; BF16 neural autocast",
            "loss_weights": {"row": 1., "col": 1., "switch": .25, "alive": .25,
                             "motion": .50, "identity_contrast": .25},
            "column_loss": "positive-candidate CE over T+NEW plus NEW BCE: true births=1, existing/real negatives=0; ignored proposals excluded",
            "switch_margin": .5, "identity_contrast_temperature": .07,
            "query_drop_probability": .10, "past_negative_injection_probability": .20,
            "injected_query_count": "at most max(1,round(.10*active_tracks)) real past unmatched non-ignored proposals",
            "alive_target": "current eligible GT or an eligible GT observation within next 30 physical source frames; loss label only",
            "gt_runtime_identity_bookkeeping": "external, first known identity permanent until retirement; duplicate births cannot relabel a switch",
            "same_candidate_cache_as_simple": True,
            "eligibility": "macro AssA >= simple+.03 OR >=.50; every dataset HOTA >= simple-.02",
            "selection": ["higher macro AssA", "higher macro HOTA", "lower IDSW/GT tracks", "earlier step"],
            "formal_training_launched": False, "official_test_labels_read": False}


def preflight(cache_path: Path | None, simple_path: Path | None) -> tuple[dict, dict, dict]:
    registered = recipe()
    if cache_path is None or simple_path is None or not cache_path.is_file() or not simple_path.is_file():
        raise ValueError("MOT-only optimization requires the complete selected ordinary cache and all-five simple metrics")
    cache, simple = json.loads(cache_path.read_text()), json.loads(simple_path.read_text())
    spec = cache["spec"]
    if cache["status"] != "COMPLETE" or spec["phase"] != "ordinary":
        raise ValueError("MOT-only optimization cannot use an incomplete or OVMOT cache")
    verify_selected_foundation(Path(spec["selection"]), spec["foundation_sha256"])
    if digest(Path(spec["checkpoint"])) != spec["foundation_sha256"] or digest(PLAN) != spec["inventory_sha256"]:
        raise ValueError("Frozen candidate foundation or original frame scope differs")
    expected = {group["key"]: group for group in (json.loads(line) for line in PLAN.read_text().splitlines())
                if group["dataset"] in registered["dataset_probabilities"]}
    actual = {item["group"]["key"]: item["group"] for item in cache["groups"]}
    if actual != expected or len(cache["groups"]) != len(expected):
        raise ValueError("MOT-only cache does not cover every original ordinary train/dev group")
    if (simple["status"] != "COMPLETE" or simple["name"] != "U2_SIMPLE_BASELINE"
            or simple["candidate_cache_sha256"] != digest(cache_path)
            or simple["foundation_sha256"] != spec["foundation_sha256"]
            or set(simple["datasets"]) != set(registered["dataset_probabilities"])):
        raise ValueError("The simple reference was not measured on this exact complete candidate cache")
    for dataset, report in simple["datasets"].items():
        groups = [group for group in expected.values() if group["dataset"] == dataset and group["role"] == "dev"]
        if (report["status"] != "COMPLETE" or report["frames_evaluated"] != sum(len(g["frames"]) for g in groups)
                or report["sequences_evaluated"] != len(groups)):
            raise ValueError(f"Simple reference does not cover the registered {dataset} development scope")
    return registered, cache, simple


def save_checkpoint(model, optimizer, next_step: int, args, metadata: dict) -> Path:
    import torch
    import torch.distributed as dist
    rng = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}
    rng_states = [None] * dist.get_world_size()
    dist.all_gather_object(rng_states, rng)
    path = args.out / f"step_{next_step:04d}.pth"
    if dist.get_rank() == 0:
        payload = {"format": "locatemot-u2-mot-only-checkpoint-v1", "next_step": next_step,
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
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--simple-baseline", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.register_only:
        registered = recipe()
        if RECIPE.exists() and json.loads(RECIPE.read_text()) != registered:
            raise ValueError("The MOT-only registration differs; do not change the formal protocol silently")
        write_json(RECIPE, registered)
        print(json.dumps({"status": "REGISTERED_NOT_LAUNCHED", "recipe": str(RECIPE)}))
        return 0
    try:
        registered, cache, simple = preflight(args.cache_manifest, args.simple_baseline)
    except ValueError as error:
        print(json.dumps({"status": "HELD_BY_PERCEPTION_CACHE_AND_SIMPLE_GATES", "reason": str(error),
                          "formal_training_launched": False}), flush=True)
        return 2
    if not RECIPE.is_file() or json.loads(RECIPE.read_text()) != registered:
        raise ValueError("The exact MOT-only recipe must be registered before optimizer launch")
    metadata = {"recipe_sha256": digest(RECIPE), "foundation_sha256": cache["spec"]["foundation_sha256"],
                "candidate_cache_sha256": digest(args.cache_manifest), "simple_baseline_sha256": digest(args.simple_baseline),
                "task_type": "mot", "world_size": 8, "foundation_weights_modified": False,
                "implementation_sha256": {name: digest(ROOT / name) for name in (
                    "locatemot/unified/models/u2/shared_track_core.py",
                    "locatemot/unified/models/u2/query_interaction.py",
                    "locatemot/unified/data/u2_mot_clips.py",
                    "locatemot/unified/losses/u2_association.py",
                    "locatemot/unified/runtime/u2_temporal_training.py",
                    "tools/unified/train_u2_mot_only.py", "tools/unified/eval_u2_mot_only.py")}}
    if args.preflight_only:
        print(json.dumps({"status": "READY", **metadata}))
        return 0
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    from locatemot.unified.data.u2_mot_clips import CachedMOTClips
    from locatemot.unified.runtime.u2_temporal_training import U2MOTTemporalTraining
    from tools.unified.eval_u2_mot_only import evaluate_checkpoint, select_mot_only
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    # Other ranks wait while rank zero measures every original dev sequence.
    dist.init_process_group("nccl", timeout=timedelta(hours=24))
    if dist.get_world_size() != 8:
        raise ValueError("Formal MOT-only training requires eight GPU ranks")
    device = torch.device("cuda", local_rank)
    torch.manual_seed(registered["seed"])
    model = DistributedDataParallel(U2MOTTemporalTraining().to(device).train(), device_ids=[local_rank],
                                    find_unused_parameters=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    torch.manual_seed(registered["seed"] + rank)
    next_step = 0
    if args.resume is not None:
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        if saved["metadata"] != metadata or saved["format"] != "locatemot-u2-mot-only-checkpoint-v1":
            raise ValueError("Resume checkpoint belongs to a different registered experiment")
        model.module.core.load_state_dict(saved["core"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        next_step = saved["next_step"]
        torch.set_rng_state(saved["rank_rng_states"][rank]["cpu"])
        torch.cuda.set_rng_state(saved["rank_rng_states"][rank]["cuda"])
    supervision = json.loads((DIRECTORY / "supervision_source_inventory.json").read_text())["ordinary"]
    sampler = CachedMOTClips(cache, supervision, seed=registered["seed"])
    args.out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        write_json(args.out / "run_state.json", {"status": "RUNNING", "next_step": next_step,
                                               "maximum_optimizer_steps": 6000, **metadata})
    # Resume a saved evaluation boundary before producing another update.
    if next_step in EVALUATION_STEPS:
        evaluate_checkpoint(model.module.core, sampler, cache, supervision, args, next_step, args.resume)
    for step in range(next_step, 6000):
        clips = sampler.sample(step, rank, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(clips, step)
        loss = output["losses"]["total"]
        loss.backward()
        finite = torch.tensor([int(torch.isfinite(loss) and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()))], device=device)
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not finite.item():
            raise FloatingPointError(f"Nonfinite MOT loss/gradient at original optimizer step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if rank == 0 and (step + 1) % 25 == 0:
            row = {"step": step + 1, "dataset": sampler.dataset_for_step(step),
                   "clip_frames": len(clips[0]), "teacher_forcing_probability": output["teacher_forcing_probability"],
                   "losses": {key: float(value.detach()) for key, value in output["losses"].items()},
                   "rank0_teacher_frames": sum(d["teacher_forced"] for d in output["diagnostics"]),
                   "rank0_model_frames": sum(not d["teacher_forced"] for d in output["diagnostics"])}
            with (args.out / "training_curve.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if (step + 1) % 500 == 0:
            checkpoint = save_checkpoint(model, optimizer, step + 1, args, metadata)
            if step + 1 in EVALUATION_STEPS:
                evaluate_checkpoint(model.module.core, sampler, cache, supervision, args, step + 1, checkpoint)
            if rank == 0:
                write_json(args.out / "run_state.json", {"status": "RUNNING", "next_step": step + 1,
                                                       "maximum_optimizer_steps": 6000, **metadata})
    if rank == 0:
        selection = select_mot_only(args.out, simple, args.cache_manifest)
        write_json(args.out / "run_state.json", {"status": "COMPLETE", "optimizer_steps": 6000,
                                               "selection_status": selection["status"], **metadata})
        print(json.dumps({"status": selection["status"], "optimizer_steps": 6000}), flush=True)
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
