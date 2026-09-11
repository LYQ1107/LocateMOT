#!/usr/bin/env python3
"""Train the R1 residual sidecar on the frozen aligned cache.

The frozen L89E anchor and all visual/language caches stay outside the
checkpoint.  Labels are attached only after ``R1FeatureAssembler.prepare``
has completed for a query-frame.  The script supports the registered
single-process run and a bounded torchrun smoke without changing the tile
schedule.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.models.r1_aligned_track_conditioning import (  # noqa: E402
    FrozenL89EAnchor, R1AlignedTrackConditioning, R1Config, target_bag_loss,
)
from locatemot.rmot.r1_aligned_cache import R1AlignedCacheIndex, R1FeatureAssembler, write_json  # noqa: E402
from tools.r1_common import (  # noqa: E402
    ACCUMULATION, CATEGORIES, FIT_ROOTS, SEED, THREAD, check_manifest,
    file_meta, label_from_index, standard_flags, unit_key,
)

ANCHOR = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/train/joint40/checkpoint_l89_epoch004.pt")
RULE = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89E/outputs/l89e/dev/selection_attempt1/checkpoint_selection.json")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _dist_info() -> tuple[int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return world, rank, local


def _record_map(request: dict[str, Any], dataset: str) -> dict[str, dict[str, Any]]:
    values = request.get("unique_query_records", {}).get(dataset, {})
    result = {}
    for key, value in values.items():
        if str(key) != unit_key(value):
            raise AssertionError("R1 request key drift")
        # ``category`` is sampler metadata in the request manifest.  Keep it
        # here so the training loop can validate the registered strata, then
        # remove it immediately before feature construction; it is never
        # passed to the bank assembler or the sidecar.
        result[str(key)] = dict(value)
    return result


def _tile_records(request: dict[str, Any], dataset: str, epoch: int, rank: int, world: int,
                  record_map: dict[str, dict[str, Any]]) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = []
    for tile in request["tiles"][dataset]:
        if int(tile["epoch"]) != int(epoch) or int(tile["tile_index"]) % world != rank:
            continue
        keys = list(tile["query_unit_keys"])
        rows = [dict(record_map[key]) for key in keys]
        if [unit_key(row) for row in rows] != keys:
            raise AssertionError("R1 tile query order drift")
        result.append(rows)
    result.sort(key=lambda rows: int(record_map[unit_key(rows[0])]["frame_id"]) if rows else -1)
    # Sorting by frame would change the prebuilt request schedule.  Restore
    # the registered tile order using the first row's manifest tile index.
    order = {key: index for index, key in enumerate(
        tile["query_unit_keys"][0] for tile in request["tiles"][dataset]
        if int(tile["epoch"]) == int(epoch) and int(tile["tile_index"]) % world == rank)}
    result.sort(key=lambda rows: order[unit_key(rows[0])])
    return result


def _tiles_for_rank(request: dict[str, Any], dataset: str, epoch: int, rank: int, world: int,
                    record_map: dict[str, dict[str, Any]]) -> list[tuple[int, list[dict[str, Any]]]]:
    result = []
    for tile in request["tiles"][dataset]:
        if int(tile["epoch"]) != int(epoch) or int(tile["tile_index"]) % world != rank:
            continue
        rows = [dict(record_map[key]) for key in tile["query_unit_keys"]]
        if [unit_key(row) for row in rows] != list(tile["query_unit_keys"]):
            raise AssertionError("R1 tile query order drift")
        result.append((int(tile["tile_index"]), rows))
    return result


def _current_observation(frame: Any) -> torch.Tensor:
    if frame.history_observations.shape[1] != 8 or not bool(frame.history_mask[:, -1].all()):
        raise AssertionError("R1 current observation is not the causal final history slot")
    return frame.history_observations[:, -1]


def _forward_one(anchor: FrozenL89EAnchor, sidecar: torch.nn.Module, frame: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    n = frame.candidate_count
    q = frame.text_tokens.unsqueeze(0)
    z1 = frame.z1.unsqueeze(0)
    history = frame.history_observations
    mask = frame.history_mask
    frames = frame.history_frame_ids
    current = _current_observation(frame)
    with torch.no_grad():
        base = anchor(z1, q, frame.text_mask.unsqueeze(0), frame.text_global.unsqueeze(0),
                      frame.frame_global.unsqueeze(0), current, history, mask, frames, frame.frame_id)
    geometry_mask = torch.ones(frame.geometry.shape[:2], dtype=torch.bool, device=frame.geometry.device)
    output = sidecar(
        frame.z0.unsqueeze(0), frame.z1.unsqueeze(0), frame.z4.unsqueeze(0),
        frame.text_tokens.unsqueeze(0), frame.text_mask.unsqueeze(0), frame.text_global.unsqueeze(0),
        frame.raw_visual_tokens.unsqueeze(0), frame.geometry, geometry_mask, frame.boxes_norm,
        base["candidate_energy"], base["presence_logit"], base["null_logit"],
    )
    if output["final_energy"].shape != (1, n) or not all(bool(value.isfinite().all()) for value in output.values()):
        raise FloatingPointError("R1 sidecar output contract failed")
    return output, base


def _attach_after_feature(assembler: R1FeatureAssembler, dense: Any, frame: Any, record: dict[str, Any],
                          label_record: dict[str, Any]) -> dict[str, Any]:
    # ``prepare`` above has already assembled every feature.  Only now seek
    # the authoritative label line and open the L69 candidate-GT sidecar.
    raw_label = label_from_index(dense, label_record)
    batch = assembler.store.build_frame(record["dataset"], record["video"], int(record["query_id"]),
                                        int(record["frame_id"]), str(record["sentence"]))
    supervision = assembler.store.attach_frame_labels(batch, raw_label["target_ids"])
    if str(supervision["category"]) != str(record.get("category", supervision["category"])):
        raise AssertionError(f"R1 sampler/category drift: {record.get('unit_key')}")
    frame.supervision = supervision | {"label": raw_label}
    return frame.supervision


def _loss_for_frame(output: dict[str, torch.Tensor], supervision: dict[str, Any], config: R1Config) -> tuple[torch.Tensor, dict[str, Any]]:
    energy = output["final_energy"][0]
    loss, stats = target_bag_loss(energy, supervision["candidate_gt"], supervision["target_ids"],
                                  supervision["category"], margin=config.set_margin,
                                  duplicate_weight=config.duplicate_weight)
    residual_reg = output["delta_energy"].float().square().mean()
    if output["delta_presence"].numel():
        residual_reg = residual_reg + 0.25 * output["delta_presence"].float().square().mean()
    total = loss + float(config.residual_reg_weight) * residual_reg
    stats.update({"base_loss": float(loss.detach()), "residual_reg": float(residual_reg.detach()),
                  "loss": float(total.detach()), "finite": bool(torch.isfinite(total))})
    return total, stats


def _save_checkpoint(path: Path, sidecar: torch.nn.Module, optimizer: torch.optim.Optimizer,
                     scheduler: Any, *, epoch: int, optimizer_step: int, config: R1Config,
                     anchor_sha: str, rule: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"R1 checkpoint collision: {path}")
    module = sidecar.module if isinstance(sidecar, DistributedDataParallel) else sidecar
    package = {
        "format": "locatemot-r1-aligned-track-conditioning-checkpoint-v1", "status": "complete",
        "epoch": int(epoch), "optimizer_step": int(optimizer_step), "model_config": config.__dict__,
        "model_state_dict": {key: value.detach().cpu() for key, value in module.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
        "anchor_checkpoint_sha256": anchor_sha, "rule": rule, "metrics": metrics,
        "checkpoint_contains_anchor": False, "checkpoint_contains_detector": False,
        "candidate_deletion": False, "candidate_truncation": False,
        "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
        **standard_flags(training_run=True),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size,
            "epoch": int(epoch), "optimizer_step": int(optimizer_step)}


def _strict_reload_probe(path: Path, config: R1Config, probe: tuple[dict[str, torch.Tensor], dict[str, Any]] | None,
                         device: torch.device) -> dict[str, Any]:
    package = torch.load(path, map_location="cpu", weights_only=False)
    model = R1AlignedTrackConditioning(config).to(device)
    result = model.load_state_dict(package["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"R1 sidecar strict reload mismatch: {result}")
    output = {"strict": True, "missing_keys": [], "unexpected_keys": []}
    if probe is not None:
        # Probe is deliberately a compact set of sidecar inputs, not a cache;
        # the caller supplies no anchor parameters.
        with torch.no_grad():
            first = model(**probe[0])
        output["probe_finite"] = bool(all(torch.isfinite(value.float()).all() for value in first.values()))
    del model, package
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-manifest", type=Path, default=Path("outputs/r1/request_manifest_attempt1/request_manifest.json"))
    parser.add_argument("--aligned-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--tiles-per-epoch", type=int, default=3000)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="bounded wiring/smoke mode; zero runs all registered epochs")
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else WORK_ROOT / args.out).resolve()
    request_path = (args.request_manifest if args.request_manifest.is_absolute() else WORK_ROOT / args.request_manifest).resolve()
    cache_path = (args.aligned_cache if args.aligned_cache.is_absolute() else WORK_ROOT / args.aligned_cache).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R1 train output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([str(sys.executable), *sys.argv])
    started = time.perf_counter()
    world, rank, local_rank = _dist_info()
    device = torch.device(args.device if world == 1 else f"cuda:{local_rank}")
    status: dict[str, Any] = {
        "format": "locatemot-r1-training-status-v1", "status": "running", "command": command,
        "cwd": str(Path.cwd().resolve()), "thread": THREAD, "seed": SEED, "rank": rank, "world_size": world,
        "failure_root_cause": None, "next_action": "run legal-dev selection and fixed semantic diagnostics after clean R1 training",
        **standard_flags(training_run=True),
    }
    anchor = None
    assembler = None
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R1 training cwd: {Path.cwd()}")
        if int(args.epochs) != 6 or int(args.tiles_per_epoch) != 3000:
            raise ValueError("R1 registered schedule is exactly 6 epochs x 3000 tiles")
        manifest_sha = check_manifest()
        request = json.loads(request_path.read_text(encoding="utf-8"))
        if request.get("format") != "locatemot-r1-request-manifest-v1" or request.get("status") != "complete":
            raise AssertionError("invalid R1 request manifest")
        if int(request.get("seed", -1)) != SEED:
            raise AssertionError("R1 request seed drift")
        if not cache_path.is_dir():
            raise FileNotFoundError(cache_path)
        anchor = FrozenL89EAnchor(ANCHOR).to(device)
        anchor.eval()
        if any(parameter.requires_grad for parameter in anchor.parameters()):
            raise AssertionError("R1 anchor parameter was not frozen")
        aligned = R1AlignedCacheIndex(cache_path)
        assembler = R1FeatureAssembler(aligned, device)
        indexes = {}
        label_maps = {}
        for dataset, root in FIT_ROOTS.items():
            from tools.r0_common import DenseIndex
            indexes[dataset] = DenseIndex(root)
            label_maps[dataset] = {str(value["unit_key"]): value for value in indexes[dataset].label_records}
        rule = json.loads(RULE.read_text(encoding="utf-8"))
        # Frozen-anchor legal-dev audit: epoch-004 presence-only miss is
        # 3/400=.0075, below the preregistered .05 activation threshold.
        # Keep the decision explicit in the serialized model config.
        config = R1Config(presence_residual=False)
        models = {dataset: R1AlignedTrackConditioning(config).to(device) for dataset in ("refer_kitti_v1", "refer_kitti_v2")}
        for model in models.values():
            model.train()
        if world > 1:
            models = {dataset: DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
                      for dataset, model in models.items()}
        optimizers = {dataset: torch.optim.AdamW(models[dataset].parameters(), lr=5e-5, weight_decay=1e-2,
                                                  betas=(0.9, 0.999)) for dataset in models}
        # A small fixed warmup/cosine schedule is registered independently of
        # validation.  It is defined over the known optimizer-step budget.
        total_steps = max(1, (int(args.epochs) * int(args.tiles_per_epoch) * len(models)) // ACCUMULATION[world])
        warmup_steps = max(1, int(round(total_steps * 0.05)))
        schedulers = {}
        for dataset in models:
            def lr_lambda(step: int, warmup=warmup_steps, total=total_steps) -> float:
                if step < warmup:
                    return float(step + 1) / float(warmup)
                progress = min(1.0, float(step - warmup) / float(max(1, total - warmup)))
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            schedulers[dataset] = torch.optim.lr_scheduler.LambdaLR(optimizers[dataset], lr_lambda)
        if args.resume is not None:
            package = torch.load(args.resume.resolve(), map_location="cpu", weights_only=False)
            for dataset in models:
                module = models[dataset].module if isinstance(models[dataset], DistributedDataParallel) else models[dataset]
                module.load_state_dict(package["model_state_dict"], strict=True)
                optimizers[dataset].load_state_dict(package["optimizer_state_dict"])
                schedulers[dataset].load_state_dict(package["scheduler_state_dict"])
        record_maps = {dataset: _record_map(request, dataset) for dataset in models}
        losses: list[dict[str, Any]] = []
        sampling = {dataset: Counter() for dataset in models}
        checkpoints: list[dict[str, Any]] = []
        optimizer_steps = 0
        micro_steps = 0
        probe_for_reload = None
        for epoch in range(1, int(args.epochs) + 1):
            for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
                model = models[dataset]; optimizer = optimizers[dataset]; scheduler = schedulers[dataset]
                optimizer.zero_grad(set_to_none=True)
                for tile_index, records in _tiles_for_rank(request, dataset, epoch, rank, world, record_maps[dataset]):
                    tile_losses = []
                    tile_stats = []
                    for public in records:
                        feature_record = dict(public)
                        category = str(feature_record.pop("category", ""))
                        if category not in CATEGORIES:
                            raise AssertionError(f"invalid R1 sampler category: {category}")
                        frame = assembler.prepare(feature_record, attach_labels=False)
                        # A feature-complete frame is now available.  Only at
                        # this boundary is the fit label line read.
                        label_record = label_maps[dataset].get(unit_key(feature_record))
                        if label_record is None:
                            raise KeyError(f"R1 fit label key missing: {unit_key(feature_record)}")
                        supervision = _attach_after_feature(assembler, indexes[dataset], frame, {**feature_record, "category": category}, label_record)
                        output, base = _forward_one(anchor, model, frame)
                        loss, stats = _loss_for_frame(output, supervision, config)
                        output["final_energy"].retain_grad()
                        tile_losses.append(loss)
                        tile_stats.append(stats)
                        sampling[dataset][str(supervision["category"])] += 1
                        if probe_for_reload is None:
                            probe_for_reload = ({
                                "z0": frame.z0.unsqueeze(0).detach().clone(), "z1": frame.z1.unsqueeze(0).detach().clone(),
                                "z4": frame.z4.unsqueeze(0).detach().clone(), "text_tokens": frame.text_tokens.unsqueeze(0).detach().clone(),
                                "text_mask": frame.text_mask.unsqueeze(0).detach().clone(), "text_global": frame.text_global.unsqueeze(0).detach().clone(),
                                "raw_tokens": frame.raw_visual_tokens.unsqueeze(0).detach().clone(), "geometry": frame.geometry.detach().clone(),
                                "geometry_mask": torch.ones(frame.geometry.shape[:2], dtype=torch.bool, device=device),
                                "boxes_norm": frame.boxes_norm.detach().clone(), "base_candidate_energy": base["candidate_energy"].detach().clone(),
                                "base_presence": base["presence_logit"].detach().clone(), "base_null": base["null_logit"].detach().clone(),
                            }, supervision)
                        del frame, output, base
                    if not tile_losses:
                        raise AssertionError("empty R1 tile")
                    tile_loss = torch.stack(tile_losses).mean() / float(ACCUMULATION[world])
                    tile_loss.backward()
                    micro_steps += 1
                    if micro_steps % ACCUMULATION[world] == 0:
                        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
                        grad_values = [parameter.grad.detach() for parameter in trainable if parameter.grad is not None]
                        if not grad_values or not all(bool(torch.isfinite(value).all()) for value in grad_values):
                            raise FloatingPointError("R1 nonfinite/empty adapter gradient")
                        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
                        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                        optimizer_steps += 1
                        losses.append({"optimizer_step": optimizer_steps, "epoch": epoch, "dataset": dataset,
                                       "tile_index": tile_index, "loss": float(tile_loss.detach()) * float(ACCUMULATION[world]),
                                       "grad_norm": grad_norm, "finite": True, "stats": tile_stats})
                        if args.max_steps and optimizer_steps >= int(args.max_steps):
                            break
                    del tile_losses, tile_stats
                    if args.max_steps and optimizer_steps >= int(args.max_steps):
                        break
                if args.max_steps and optimizer_steps >= int(args.max_steps):
                    break
            if args.max_steps and optimizer_steps >= int(args.max_steps):
                break
            if rank == 0 and epoch in {1, 2, 4, 6}:
                for dataset in models:
                    checkpoints.append(_save_checkpoint(
                        out / f"checkpoint_r1_epoch{epoch:03d}_{dataset}.pt", models[dataset], optimizers[dataset],
                        schedulers[dataset], epoch=epoch, optimizer_step=optimizer_steps, config=config,
                        anchor_sha=sha256_file(ANCHOR), rule=rule, metrics={"sampling": dict(sampling[dataset])}))
        # A bounded run exits from the nested tile loop before the formal
        # epoch-save block.  Save one truthful smoke checkpoint here so the
        # requested wiring/reload gate cannot silently finish without reload
        # evidence.  This branch never runs for the registered formal run.
        if rank == 0 and args.max_steps and optimizer_steps > 0 and not checkpoints:
            for dataset in models:
                checkpoints.append(_save_checkpoint(
                    out / f"checkpoint_r1_smoke_step{optimizer_steps:04d}_{dataset}.pt",
                    models[dataset], optimizers[dataset], schedulers[dataset],
                    epoch=epoch, optimizer_step=optimizer_steps, config=config,
                    anchor_sha=sha256_file(ANCHOR), rule=rule,
                    metrics={"sampling": dict(sampling[dataset]), "stage": "fit_only_smoke"}))
        # A bounded smoke is also a valid strict-reload gate, but it is not a
        # semantic result.  The formal run reaches the registered epoch saves.
        if rank == 0:
            for dataset in models:
                module = models[dataset].module if isinstance(models[dataset], DistributedDataParallel) else models[dataset]
                if any(parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()) for parameter in module.parameters()):
                    raise FloatingPointError("R1 final gradient nonfinite")
            reload_audit = {}
            for checkpoint in checkpoints:
                reload_audit[checkpoint["path"]] = _strict_reload_probe(Path(checkpoint["path"]), config, probe_for_reload, device)
            write_json(out / "loss_trace.json", losses)
            write_json(out / "sampling_trace.json", {dataset: dict(values) for dataset, values in sampling.items()})
            metrics = {
                "format": "locatemot-r1-training-metrics-v1", "stage": "fit_only_smoke" if args.max_steps else "formal_fit",
                "status": "complete", "epochs_completed": epoch, "optimizer_steps": optimizer_steps,
                "finite_steps": len(losses), "nonzero_gradient_steps": len(losses), "world_size": world,
                "datasets": list(models), "sampling": {dataset: dict(values) for dataset, values in sampling.items()},
                "checkpoints": checkpoints, "reload_audit": reload_audit,
                "anchor_frozen": True, "anchor_sha256": sha256_file(ANCHOR), "rule_path": str(RULE),
                "persistent_raw_dense_cache": False, "candidate_deletion": False, "candidate_truncation": False,
                "elapsed_seconds": float(time.perf_counter() - started), "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)), "inputs": {"request_manifest": str(request_path), "aligned_cache": str(cache_path)},
                "flags": standard_flags(training_run=True),
            }
            metrics_name = "metrics_r1_smoke.json" if args.max_steps else "metrics_r1_formal.json"
            write_json(out / metrics_name, metrics)
            write_json(out / "provenance.json", {
                "format": "locatemot-r1-training-provenance-v1", "status": "complete", "command": command,
                "cwd": str(Path.cwd().resolve()), "thread": THREAD, "seed": SEED, "world_size": world,
                "manifest_sha256": manifest_sha, "anchor": {"path": str(ANCHOR), "sha256": sha256_file(ANCHOR)},
                "rule_path": str(RULE), "request_manifest": file_meta(request_path), "aligned_cache": str(cache_path),
                "labels_attached_after_feature_construction": True, "labels_in_cache": False,
                "anchor_parameters_in_optimizer": False, "detector_parameters_in_checkpoint": False,
                "persistent_raw_dense_cache": False, "failure_root_cause": None,
                "next_action": status["next_action"], **standard_flags(training_run=True),
            })
            write_json(out / "status.json", {**status, "status": "complete", "optimizer_steps": optimizer_steps,
                                              "checkpoints": checkpoints, "elapsed_seconds": float(time.perf_counter() - started),
                                              "outputs": {"metrics": str(out / metrics_name), "loss_trace": str(out / "loss_trace.json")}})
        if world > 1:
            dist.barrier()
        return 0
    except Exception as exc:
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        if rank == 0:
            status.update({"status": "incomplete", "elapsed_seconds": float(time.perf_counter() - started),
                           "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback": error,
                           "outputs": {"attempt": str(out)}})
            write_json(out / "status.json", status)
            (out / "INCOMPLETE.md").write_text("# R1 training incomplete\n\n```text\n" + error +
                                                 "```\n\nPreserve this attempt; repair only the first actionable contract error.\n", encoding="utf-8")
        raise
    finally:
        if assembler is not None:
            assembler.close()
        if anchor is not None:
            del anchor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
