#!/usr/bin/env python3
"""L84 paired mid-decoder probe and the single authorized no-refPE test.

The script intentionally keeps feature tensors process-local.  It rebuilds
the complete L69 row set from native frame pointers, attaches fit labels only
after state construction, and uses the identical canonical probe state and
schedule for every representation at a given seed.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from locatemot.evaluation.l83_target_bag_metrics import (  # noqa: E402
    aggregate_group_metrics,
    breakdowns,
    group_metrics,
)
from locatemot.models.l84_grounding_states import (  # noqa: E402
    SELECTED_STAGE_NAMES,
    capture_l84_states,
)
from locatemot.models.l84_paired_probe import L84PairedProbe, L84PairedProbeConfig  # noqa: E402
from locatemot.rmot.l83_target_bag_loss import l83_target_bag_loss  # noqa: E402
from locatemot.rmot.l80_data import L80BankStore  # noqa: E402

THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEEDS = (20260829, 20260830, 20260831)
BOOTSTRAP_SEED = 20260902
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
INIT_ROOT = ROOT / "outputs/l84/protocol"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path), "exists": path.exists(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "mtime_ns": path.stat().st_mtime_ns if path.exists() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def finite_gradients(parameters: Any) -> tuple[bool, float]:
    gradients = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return False, 0.0
    norm = float(torch.stack([value.float().norm() for value in gradients]).norm())
    ok = all(bool(torch.isfinite(value.float()).all()) for value in gradients)
    return bool(ok and math.isfinite(norm) and norm > 0.0), norm


def seed_for_stage(seed: int, rank: int) -> int:
    return int(seed + rank * 100003)


def set_paired_rng(seed: int, rank: int) -> None:
    value = seed_for_stage(seed, rank)
    random.seed(value)
    np.random.seed(value & 0xFFFFFFFF)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(value)
        torch.cuda.manual_seed_all(value)


def load_canonical(seed: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    path = INIT_ROOT / f"probe_init_seed{seed}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    package = torch.load(path, map_location="cpu")
    if package.get("format") != "locatemot-l84-canonical-probe-init-v1":
        raise AssertionError(f"canonical format drift: {path}")
    if int(package.get("seed")) != seed:
        raise AssertionError(f"canonical seed drift: {path}")
    state = {key: value.detach().cpu().clone() for key, value in package["state_dict"].items()}
    return state, {"path": str(path), "sha256": sha256_file(path), "state_keys": list(state)}


def group_key_from_row(row: dict[str, Any]) -> str:
    return f"{row['dataset']}|{row['video']}|{int(row['frame_id'])}"


def make_schedules(train_keys: list[str], world_size: int, out: Path) -> dict[int, dict[str, Any]]:
    schedules: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        path = out / "protocol" / f"train_schedule_seed{seed}.json"
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(train_keys)).tolist()
        global_order = [str(train_keys[index]) for index in permutation]
        local_orders = {
            str(rank): global_order[rank::world_size] for rank in range(world_size)
        }
        payload = {
            "format": "locatemot-l84-paired-train-schedule-v1",
            "seed": seed, "world_size": world_size,
            "global_group_order": global_order,
            "rank_local_group_order": local_orders,
            "group_count": len(global_order), "epochs": 10,
            "stage_shared_schedule": True,
            "command": " ".join([str(Path(__file__).resolve())]),
            "screening_gt_used": False, "official_test_labels_read": False,
            "hota_trackeval_run": False, "ordinary_mot_ovmot_touched": False,
        }
        if path.exists():
            existing = json.loads(path.read_text())
            if existing != payload:
                raise AssertionError(f"existing paired schedule differs: {path}")
        else:
            write_json(path, payload)
        schedules[seed] = payload
    return schedules


def build_group_states(
    group_keys: list[str], groups: dict[str, dict[str, Any]], device: torch.device,
    *, no_refpe_in_content: bool = False, selected_name: str | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    """Construct complete states before attaching fit labels."""
    from locatemot.rmot.l82_grounding_runtime import (  # imported after compat shim
        GroundingCandidateReferenceRuntime,
        install_clip_torchvision_compat,
    )
    from tools.l82_train_frozen_rank_probe import attach_fit_labels, load_fit_label_rows

    install_clip_torchvision_compat()
    runtime = GroundingCandidateReferenceRuntime(device)
    store = L80BankStore(max_history=8)
    label_rows: dict[str, dict[str, Any]] | None = None
    built: list[Any] = []
    stage_names: list[str] | None = None
    native_seconds = replay_seconds = 0.0
    started = time.perf_counter()

    for group_key in group_keys:
        group = groups[group_key]
        batches = [store.build_unit(row) for row in group["queries"]]
        if not batches:
            raise AssertionError(f"empty group {group_key}")
        first = batches[0]
        n = first.candidate_count
        if n <= 0:
            raise AssertionError(f"empty candidate set {group_key}")
        for batch in batches:
            if batch.candidate_count != n or batch.row_offsets != first.row_offsets:
                raise AssertionError(f"same-frame candidate row drift: {batch.unit_key}")
            if batch.history_frame_ids.numel() and bool((batch.history_frame_ids > int(batch.frame_id)).any()):
                raise AssertionError(f"future history: {batch.unit_key}")
            if len(batch.row_keys) != n or [int(key[-1]) for key in batch.row_keys] != batch.row_offsets:
                raise AssertionError(f"row key/order drift: {batch.unit_key}")

        runtime.encoder_events.clear()
        runtime.capture.clear()
        native_started = time.perf_counter()
        with torch.inference_mode():
            native = runtime.inference_detector(
                runtime.model, str(first.image_path), text_prompt=str(first.sentence), custom_entities=True)
        native_seconds += time.perf_counter() - native_started
        if len(runtime.encoder_events) != 1:
            raise AssertionError(f"native encoder event drift: {group_key}")
        event = runtime.encoder_events[-1]
        visual_feats = runtime.capture.get("visual_feats")
        sample_template = runtime.capture.get("sample_template")
        if visual_feats is None or not isinstance(sample_template, (list, tuple)) or len(sample_template) != 1:
            raise AssertionError(f"native reusable feature contract missing: {group_key}")
        image_shape = tuple(int(x) for x in native.metainfo["img_shape"][:2])
        scale_factor = native.metainfo["scale_factor"]
        sample_template = sample_template[0]
        per_stage: dict[str, list[torch.Tensor]] = defaultdict(list)
        row_audits: list[dict[str, Any]] = []
        for batch in batches:
            text_dict, caption, token_map = runtime.make_text_dict(
                runtime.model, str(batch.sentence), device, force_pad_to_max=True)
            sample = copy.deepcopy(sample_template)
            runtime.set_sample_text(sample, caption, token_map)
            before = len(runtime.encoder_events)
            replay_started = time.perf_counter()
            with torch.inference_mode():
                runtime.original_forward_transformer(visual_feats, text_dict, [sample])
            replay_seconds += time.perf_counter() - replay_started
            if len(runtime.encoder_events) != before + 1:
                raise AssertionError(f"replay encoder event drift: {batch.unit_key}")
            replay_event = runtime.encoder_events[-1]
            state = capture_l84_states(
                runtime.model, replay_event, batch.boxes.to(device), image_shape, scale_factor,
                no_refpe_in_content=no_refpe_in_content, selected_name=selected_name,
            )
            if state["candidate_count"] != n:
                raise AssertionError(f"state count drift: {batch.unit_key}")
            if stage_names is None:
                stage_names = list(state["states"])
            if list(state["states"]) != stage_names:
                raise AssertionError(f"stage order drift: {batch.unit_key}")
            for name, value in state["states"].items():
                if value.shape != (n, 256) or not bool(torch.isfinite(value).all()):
                    raise AssertionError(f"state shape/finite drift {name}: {batch.unit_key}")
                per_stage[name].append(value.float().cpu().contiguous())
            row_audits.append({
                "unit_key": str(batch.unit_key), "candidate_count": n,
                "row_offsets": [int(x) for x in batch.row_offsets],
                "row_keys_digest": hashlib.sha256(json.dumps([list(key) for key in batch.row_keys]).encode()).hexdigest(),
                "candidate_indices": [int(x) for x in batch.candidate_indices],
                "pool_ids": [int(x) for x in batch.pool_ids],
                "state_shapes": {name: list(value.shape) for name, value in state["states"].items()},
                "all_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
                "future_history_count": int((batch.history_frame_ids > int(batch.frame_id)).sum()),
            })
            runtime.encoder_events.clear()
            del state, replay_event, sample, text_dict
        feature_values = {name: torch.stack(values) for name, values in per_stage.items()}
        # Explicit label boundary: all complete feature/state rows now exist.
        if label_rows is None:
            label_rows = load_fit_label_rows()
        data = attach_fit_labels(batches, feature_values, label_rows, store, group, {
            "native_seconds": native_seconds, "image_shape": image_shape,
        })
        built.append(data)
        del batches, feature_values, row_audits, event, native, visual_feats, sample_template
        runtime.capture.clear()
        runtime.encoder_events.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not built or stage_names is None:
        raise AssertionError("no L84 groups built")
    model_info = dict(runtime.model_info)
    model_info.update({
        "selected_stage_names": stage_names,
        "no_refpe_in_content": no_refpe_in_content,
        "selected_name": selected_name,
        "native_seconds_sum": native_seconds,
        "replay_seconds_sum": replay_seconds,
        "feature_construction_wall_seconds": time.perf_counter() - started,
        "features_in_memory_only": True, "features_persistent": False,
        "candidate_deletion": False, "candidate_truncation": False,
        "future_history_count": 0,
    })
    runtime.close()
    del runtime, store, label_rows
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return built, model_info


def metric_rate(record: dict[str, Any], numerator: str, denominator: str) -> float | None:
    den = int(record.get(denominator, 0))
    return float(record.get(numerator, 0)) / den if den else None


def paired_bootstrap(
    zero_records: list[dict[str, Any]], candidate_records: list[dict[str, Any]],
    *, seed: int = BOOTSTRAP_SEED, resamples: int = 10000,
) -> dict[str, Any]:
    by_zero = {str(row["group_key"]): row for row in zero_records}
    by_candidate = {str(row["group_key"]): row for row in candidate_records}
    common = sorted(set(by_zero).intersection(by_candidate))
    if not common:
        raise AssertionError("no paired dev groups")
    differences: dict[str, np.ndarray] = {}
    for name, numerator, denominator in (
        ("bag_hard_improvement", "target_bag_hard_bad", "target_bag_hard_total"),
        ("hit_at1_improvement", "target_bag_hit_at1", "target_bag_query_total"),
        ("multi_exact_improvement", "multi_target_exact", "multi_target_total"),
    ):
        values = []
        for key in common:
            left = metric_rate(by_zero[key], numerator, denominator)
            right = metric_rate(by_candidate[key], numerator, denominator)
            if left is not None and right is not None:
                values.append(float(left - right) if name == "bag_hard_improvement" else float(right - left))
        if not values:
            differences[name] = np.empty(0, dtype=np.float64)
        else:
            differences[name] = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {
        "format": "locatemot-l84-paired-bootstrap-v1", "seed": seed,
        "resamples": resamples, "paired_group_count": len(common),
        "metrics": {},
    }
    for name, values in differences.items():
        if values.size == 0:
            result["metrics"][name] = {"count": 0, "finite": True}
            continue
        indices = rng.integers(0, values.size, size=(resamples, values.size))
        estimates = values[indices].mean(axis=1)
        result["metrics"][name] = {
            "count": int(values.size), "point": float(values.mean()),
            "ci95_lower": float(np.quantile(estimates, 0.025)),
            "ci95_upper": float(np.quantile(estimates, 0.975)),
            "finite": bool(np.isfinite(estimates).all()),
        }
    return result


def stable_gate(stage_metrics: dict[str, dict[int, Any]], stage: str) -> dict[str, Any]:
    rows = [stage_metrics[stage][seed] for seed in SEEDS]
    z0 = [stage_metrics["Z0"][seed] for seed in SEEDS]

    def value(row: dict[str, Any], field: str) -> float | None:
        return row["aggregate"].get(field)

    def v2(row: dict[str, Any], field: str) -> float | None:
        return row["breakdowns"].get("dataset", {}).get("refer_kitti_v2", {}).get(field)

    hard_delta = [float(value(a, "target_bag_hard_violation") - value(b, "target_bag_hard_violation")) for a, b in zip(z0, rows)]
    hit_delta = [float(value(b, "target_bag_hit_at1") - value(a, "target_bag_hit_at1")) for a, b in zip(z0, rows)]
    v2_hard_delta = [float(v2(a, "target_bag_hard_violation") - v2(b, "target_bag_hard_violation")) for a, b in zip(z0, rows)]
    v2_hit_delta = [float(v2(b, "target_bag_hit_at1") - v2(a, "target_bag_hit_at1")) for a, b in zip(z0, rows)]
    multi_delta = [float(value(b, "multi_target_exact_topT") - value(a, "multi_target_exact_topT")) for a, b in zip(z0, rows)]
    swap_delta = [float(value(b, "query_swap_pair_accuracy") - value(a, "query_swap_pair_accuracy")) for a, b in zip(z0, rows)]
    checks = {
        "A_aggregate_hard": bool(np.mean(hard_delta) >= 0.03 and all(x > 0.0 for x in hard_delta)),
        "B_hit_at1": bool(np.mean(hit_delta) >= 0.04 and sum(x > 0.0 for x in hit_delta) >= 2),
        "C_v2_hard_and_hit": bool(np.mean(v2_hard_delta) >= 0.03 and min(v2_hard_delta) >= -0.01 and np.mean(v2_hit_delta) >= 0.03),
        "D_query_swap": bool(np.mean(swap_delta) >= -0.03),
        "E_multi_target_exact": bool(np.mean(multi_delta) >= -0.03),
    }
    return {
        "stage": stage, "checks": checks, "passed_without_bootstrap": all(checks.values()),
        "deltas_by_seed": {
            "aggregate_hard_improvement": dict(zip(SEEDS, hard_delta)),
            "aggregate_hit_at1_improvement": dict(zip(SEEDS, hit_delta)),
            "v2_hard_improvement": dict(zip(SEEDS, v2_hard_delta)),
            "v2_hit_at1_improvement": dict(zip(SEEDS, v2_hit_delta)),
            "multi_exact_improvement": dict(zip(SEEDS, multi_delta)),
            "swap_accuracy_improvement": dict(zip(SEEDS, swap_delta)),
        },
        "means": {
            "aggregate_hard_improvement": float(np.mean(hard_delta)),
            "aggregate_hit_at1_improvement": float(np.mean(hit_delta)),
            "v2_hard_improvement": float(np.mean(v2_hard_delta)),
            "v2_hit_at1_improvement": float(np.mean(v2_hit_delta)),
            "multi_exact_improvement": float(np.mean(multi_delta)),
            "swap_accuracy_improvement": float(np.mean(swap_delta)),
        },
    }


def train_one_stage(
    stage: str, seed: int, data_by_key: dict[str, Any], local_order: list[str],
    local_dev_order: list[str], expected_dev_count: int, device: torch.device,
    out: Path, rank: int, world_size: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    canonical, canonical_info = load_canonical(seed)
    base = L84PairedProbe(L84PairedProbeConfig()).to(device=device, dtype=torch.float32)
    result = base.load_state_dict(canonical, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"canonical strict load failed: {result}")
    max_initial_diff = max(float((base.state_dict()[key].cpu() - value).abs().max()) for key, value in canonical.items())
    if max_initial_diff != 0.0:
        raise AssertionError(f"initial paired state drift {stage}/{seed}: {max_initial_diff}")
    set_paired_rng(seed, rank)
    model: torch.nn.Module = base
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(base, device_ids=[device.index], output_device=device.index)
    optimizer = torch.optim.AdamW(base.parameters(), lr=2e-4, weight_decay=1e-4)
    total_updates = len(local_order) * 10
    warmup = max(1, int(round(total_updates * 0.05)))

    def lr_factor(step: int) -> float:
        if step <= warmup:
            return float(step) / float(warmup)
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total_updates - warmup)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    trace: list[dict[str, Any]] = []
    max_grad = 0.0
    step = 0
    for epoch in range(1, 11):
        model.train()
        for key in local_order:
            step += 1
            data = data_by_key[key]
            optimizer.zero_grad(set_to_none=True)
            value = data.features[stage].to(device=device, dtype=torch.float32).clone()
            output = model(value)
            loss, parts = l83_target_bag_loss(
                output["interaction"], data.membership_mask.to(device), data.categories,
                data.target_ids, data.candidate_gt,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite L84 loss {stage}/{seed}/{step}")
            loss.backward()
            valid, grad_norm = finite_gradients(base.parameters())
            if not valid:
                raise FloatingPointError(f"invalid/nonzero L84 gradient {stage}/{seed}/{step}")
            clipped = float(torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0))
            if not math.isfinite(clipped) or clipped <= 0.0:
                raise FloatingPointError(f"invalid clipped L84 gradient {stage}/{seed}/{step}")
            optimizer.step()
            scheduler.step()
            max_grad = max(max_grad, clipped)
            trace.append({
                "seed": seed, "stage": stage, "epoch": epoch, "step": step,
                "loss": float(loss.detach().cpu()), "gradient_norm": grad_norm,
                "clipped_gradient_norm": clipped, "lr": float(optimizer.param_groups[0]["lr"]),
                **{key: (float(item.detach().cpu()) if torch.is_tensor(item) else item) for key, item in parts.items()},
                "finite": True, "nonzero_gradient": True,
            })
            del value, output, loss
    if step != total_updates:
        raise AssertionError(f"update count drift {stage}/{seed}: {step}/{total_updates}")

    model.eval()
    local_records: list[dict[str, Any]] = []
    local_aux: list[dict[str, Any]] = []
    with torch.inference_mode():
        for key in local_dev_order:
            data = data_by_key[key]
            value = data.features[stage].to(device=device, dtype=torch.float32).clone()
            first = model(value)
            second = model(value)
            scores = first["interaction"].float().cpu()
            repeat_noise = float((first["interaction"] - second["interaction"]).abs().max().cpu())
            record, auxiliary = group_metrics(data, scores)
            record["repeat_run_noise"] = repeat_noise
            local_records.append(record)
            local_aux.append(auxiliary)
            del value, first, second

    stage_dir = out / f"seed{seed}" / stage
    if rank == 0:
        stage_dir.mkdir(parents=True, exist_ok=True)
        package = {
            "format": "locatemot-l84-paired-probe-checkpoint-v1",
            "seed": seed, "stage": stage, "epoch": 10, "step": step,
            "model_config": {"input_dim": 256, "hidden": 256, "dropout": 0.05},
            "model_state_dict": {key: value.detach().cpu().clone() for key, value in base.state_dict().items()},
            "model_parameter_count": int(sum(value.numel() for value in base.parameters())),
            "canonical_initialization": canonical_info,
            "max_initial_parameter_diff": max_initial_diff,
            "schedule": {"local_group_count": len(local_order), "local_dev_group_count": len(local_dev_order)},
            "optimizer": {"name": "AdamW", "lr": 2e-4, "weight_decay": 1e-4, "clip_norm": 1.0, "warmup_fraction": 0.05, "cosine": True},
            "loss_contract": "L83 faithful target-bag loss, unchanged",
            "candidate_deletion": False, "candidate_truncation": False,
        }
        tmp = stage_dir / "checkpoint.pt.tmp"
        torch.save(package, tmp)
        tmp.replace(stage_dir / "checkpoint.pt")
    if world_size > 1:
        dist.barrier()
    gathered_records: list[list[dict[str, Any]] | None] | None = [None] * world_size if rank == 0 else None
    gathered_aux: list[list[dict[str, Any]] | None] | None = [None] * world_size if rank == 0 else None
    gathered_trace: list[list[dict[str, Any]] | None] | None = [None] * world_size if rank == 0 else None
    if world_size > 1:
        dist.gather_object(local_records, gathered_records, dst=0)
        dist.gather_object(local_aux, gathered_aux, dst=0)
        dist.gather_object(trace, gathered_trace, dst=0)
    else:
        gathered_records, gathered_aux, gathered_trace = [local_records], [local_aux], [trace]
    root_summary: dict[str, Any] | None = None
    if rank == 0:
        raw_records = [row for shard in (gathered_records or []) for row in (shard or [])]
        raw_auxiliaries = [row for shard in (gathered_aux or []) for row in (shard or [])]
        aux_by_key = {
            str(row["group_key"]): item
            for row, item in zip(raw_records, raw_auxiliaries)
        }
        records = sorted(raw_records, key=lambda row: str(row["group_key"]))
        # Auxiliary lists must follow the same deterministic order as records.
        auxiliaries = [aux_by_key[str(row["group_key"])] for row in records]
        aggregate = aggregate_group_metrics(records, auxiliaries)
        root_summary = {
            "seed": seed, "stage": stage, "aggregate": aggregate,
            "breakdowns": breakdowns(records, auxiliaries),
            "dev_group_count": len(records), "all_dev_groups_present": len({str(row["group_key"]) for row in records}) == expected_dev_count,
            "checkpoint": file_meta(stage_dir / "checkpoint.pt"),
            "trace_steps": sum(len(shard or []) for shard in (gathered_trace or [])),
            "max_gradient_norm": max_grad,
            "max_initial_parameter_diff": max_initial_diff,
        }
        write_json(stage_dir / "dev_metrics.json", root_summary)
        (stage_dir / "dev_group_metrics.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True, default=str) for row in records) + "\n"
        )
        write_json(stage_dir / "loss_trace.json", [item for shard in (gathered_trace or []) for item in (shard or [])])
    if world_size > 1:
        dist.barrier()
    del model, base, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return root_summary, local_records, local_aux, {"loss_trace": trace}


def contract_only(args: argparse.Namespace, out: Path, device: torch.device, groups: dict[str, dict[str, Any]], train_keys: list[str], dev_keys: list[str]) -> int:
    selected = train_keys[:2] + dev_keys[: max(2, args.max_groups - 2)] if args.max_groups else train_keys[:2] + dev_keys[:2]
    data, info = build_group_states(selected, groups, device)
    by_key = {item.group_key: item for item in data}
    checks = {
        "groups_built": len(data) == len(selected),
        "stage_names": list(data[0].features) == list(SELECTED_STAGE_NAMES),
        "all_candidate_rows_retained": all(item.candidate_count == next(iter(item.features.values())).shape[1] for item in data),
        "finite_features": all(bool(torch.isfinite(value).all()) for item in data for value in item.features.values()),
        "future_history_zero": all(not bool((item.features["Z0"] != item.features["Z0"]).any()) for item in data),
        "no_candidate_deletion": True, "no_candidate_truncation": True,
    }
    payload = {
        "format": "locatemot-l84-forward-contract-v1", "status": "complete" if all(checks.values()) else "contract_fail",
        "selected_group_keys": selected, "group_count": len(data),
        "stage_names": list(data[0].features) if data else [],
        "feature_shapes": {key: list(value.shape) for key, value in data[0].features.items()} if data else {},
        "checks": checks, "model_info": info,
        "labels_attached_after_complete_state": True,
        "candidate_deletion": False, "candidate_truncation": False,
        "screening_gt_used": False, "official_test_labels_read": False,
        "hota_trackeval_run": False, "ordinary_mot_ovmot_touched": False,
    }
    write_json(out / "contract.json", payload)
    write_json(out / "provenance.json", {"format": "locatemot-l84-forward-contract-provenance-v1", "command": " ".join([sys.executable, *sys.argv]), "inputs": {"manifest": file_meta(MANIFEST)}, **{key: payload[key] for key in ("status", "screening_gt_used", "official_test_labels_read", "ordinary_mot_ovmot_touched")}})
    write_json(out / "status.json", payload)
    del data, by_key
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return 0 if all(checks.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--max-groups", type=int, default=0)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L84 output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    command = " ".join([sys.executable, *sys.argv])
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if world_size > 4:
            raise RuntimeError(f"world size exceeds 4: {world_size}")
        if sha256_file(MANIFEST) != EXPECTED_MANIFEST:
            raise AssertionError("fixed manifest SHA drift")
        if not torch.cuda.is_available():
            raise RuntimeError("L84 requires CUDA")
        if world_size > 1:
            dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from tools.l82_train_frozen_rank_probe import load_groups
        groups, train_keys, dev_keys = load_groups()
        if args.contract_only:
            result = contract_only(args, out, device, groups, train_keys, dev_keys)
            if world_size > 1:
                dist.barrier(); dist.destroy_process_group()
            return result
        if args.max_groups:
            if args.max_groups < 4:
                raise ValueError("--max-groups must be >=4")
            train_keys = train_keys[: max(2, args.max_groups // 2)]
            dev_keys = dev_keys[: max(2, args.max_groups - len(train_keys))]
        all_keys = train_keys + dev_keys
        local_keys = all_keys[rank::world_size]
        train_set, dev_set = set(train_keys), set(dev_keys)
        local_train_keys = [key for key in local_keys if key in train_set]
        local_dev_keys = [key for key in local_keys if key in dev_set]
        if not local_train_keys or not local_dev_keys:
            raise AssertionError(f"rank {rank} split empty: {len(local_train_keys)}/{len(local_dev_keys)}")
        schedules = make_schedules(train_keys, world_size, ROOT / "outputs/l84")
        # Build each rank's complete state once; stages reuse only RAM tensors.
        data, model_info = build_group_states(local_keys, groups, device)
        data_by_key = {item.group_key: item for item in data}
        all_stage_metrics: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        all_stage_records: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(dict)
        all_stage_aux: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(dict)
        stage_run_started = time.perf_counter()
        for seed in SEEDS:
            global_order = schedules[seed]["global_group_order"]
            local_order = [key for key in global_order if key in data_by_key]
            local_dev_order = [key for key in local_dev_keys]
            for stage in SELECTED_STAGE_NAMES:
                summary, local_records, local_aux, _trace = train_one_stage(
                    stage, seed, data_by_key, local_order, local_dev_order,
                    len(dev_keys), device, out, rank, world_size,
                )
                if rank == 0 and summary is not None:
                    all_stage_metrics[stage][seed] = summary
                    # Root writes compact record copies for paired tests below.
                    all_stage_records[stage][seed] = local_records
                    all_stage_aux[stage][seed] = local_aux

        selected_original: str | None = None
        if rank == 0:
            stage_metrics_json = {
                stage: {str(seed): value for seed, value in seeds.items()}
                for stage, seeds in all_stage_metrics.items()
            }
            paired_bootstrap_json: dict[str, Any] = {}
            for stage in SELECTED_STAGE_NAMES:
                if stage == "Z0":
                    continue
                per_seed = {}
                for seed in SEEDS:
                    zero_path = out / f"seed{seed}" / "Z0" / "dev_group_metrics.jsonl"
                    candidate_path = out / f"seed{seed}" / stage / "dev_group_metrics.jsonl"
                    zero_records = [json.loads(line) for line in zero_path.read_text().splitlines() if line.strip()]
                    candidate_records = [json.loads(line) for line in candidate_path.read_text().splitlines() if line.strip()]
                    per_seed[str(seed)] = paired_bootstrap(zero_records, candidate_records)
                # A pooled test repeats the same paired group across the three
                # independently paired seeds, retaining the same group axis.
                pooled_zero: list[dict[str, Any]] = []
                pooled_candidate: list[dict[str, Any]] = []
                for seed in SEEDS:
                    pooled_zero.extend(json.loads(line) for line in (out / f"seed{seed}" / "Z0" / "dev_group_metrics.jsonl").read_text().splitlines() if line.strip())
                    pooled_candidate.extend(json.loads(line) for line in (out / f"seed{seed}" / stage / "dev_group_metrics.jsonl").read_text().splitlines() if line.strip())
                paired_bootstrap_json[stage] = {"per_seed": per_seed, "pooled": paired_bootstrap(pooled_zero, pooled_candidate)}
            write_json(out / "paired_stage_metrics.json", {
                "format": "locatemot-l84-paired-stage-metrics-v1",
                "status": "complete", "stage_metrics": stage_metrics_json,
                "stage_names": list(SELECTED_STAGE_NAMES), "seeds": list(SEEDS),
            })
            write_json(out / "paired_bootstrap.json", {
                "format": "locatemot-l84-paired-bootstrap-manifest-v1",
                "status": "complete", "bootstrap_seed": BOOTSTRAP_SEED,
                "resamples": 10000, "comparisons": paired_bootstrap_json,
            })
            stable = {
                stage: stable_gate(all_stage_metrics, stage)
                for stage in SELECTED_STAGE_NAMES if stage != "Z0"
            }
            for stage, value in stable.items():
                value["bootstrap_pooled"] = paired_bootstrap_json[stage]["pooled"]
                value["checks"]["F_paired_bootstrap_ci_lower"] = bool(
                    value["bootstrap_pooled"]["metrics"].get("bag_hard_improvement", {}).get("ci95_lower", -math.inf) > 0.0
                )
                value["passed"] = bool(value["passed_without_bootstrap"] and value["checks"]["F_paired_bootstrap_ci_lower"])
            qualifying = [stage for stage, value in stable.items() if value["passed"]]
            def stage_sort(stage: str) -> tuple[Any, ...]:
                means = stable[stage]["means"]
                # Sort ascending on the registered preference order: larger
                # improvements are represented by negative values, while the
                # final component keeps the earliest/simple stage on ties.
                return (
                    -means["aggregate_hard_improvement"],
                    -means["aggregate_hit_at1_improvement"],
                    -means["v2_hard_improvement"],
                    -means["v2_hit_at1_improvement"],
                    -means["multi_exact_improvement"],
                    SELECTED_STAGE_NAMES.index(stage),
                )
            selected_original = sorted(qualifying, key=stage_sort)[0] if qualifying else "Z0"
            write_json(out / "stable_gate.json", {
                "format": "locatemot-l84-stable-mid-layer-gate-v1",
                "status": "middecoder_verified_candidate" if qualifying else "middecoder_not_verified_use_z0_fallback",
                "checks": stable, "qualifying_stages": qualifying,
                "selected_original": selected_original,
                "thresholds": {"hard_improvement": 0.03, "hit_improvement": 0.04, "v2_hard_improvement": 0.03, "v2_hit_improvement": 0.03, "swap_drop_max": 0.03, "multi_drop_max": 0.03, "bootstrap_ci_lower": 0.0},
            })
            write_json(out / "training_summary.json", {
                "format": "locatemot-l84-paired-training-summary-v1",
                "status": "complete", "seed_count": len(SEEDS),
                "stage_count": len(SELECTED_STAGE_NAMES),
                "train_groups": len(train_keys), "dev_groups": len(dev_keys),
                "feature_model_info": model_info,
                "wall_seconds": time.perf_counter() - stage_run_started,
                "candidate_deletion": False, "candidate_truncation": False,
                "screening_gt_used": False, "official_test_labels_read": False,
                "hota_trackeval_run": False, "ordinary_mot_ovmot_touched": False,
            })
        if world_size > 1:
            holder = [selected_original]
            dist.broadcast_object_list(holder, src=0)
            selected_original = holder[0]

        final_status = "middecoder_not_verified_use_z0_fallback"
        final_selected = selected_original
        if selected_original != "Z0":
            # Free all original states before the one authorized no-refPE
            # rebuild.  The rebuild remains process-local and is not cached.
            del data, data_by_key
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if world_size > 1:
                dist.barrier()
            no_ref_data, no_ref_info = build_group_states(
                local_keys, groups, device, no_refpe_in_content=True,
                selected_name=selected_original,
            )
            no_ref_by_key = {item.group_key: item for item in no_ref_data}
            no_ref_stage = f"{selected_original}_no_refpe"
            no_ref_metrics: dict[int, dict[str, Any]] = {}
            for seed in SEEDS:
                global_order = schedules[seed]["global_group_order"]
                local_order = [key for key in global_order if key in no_ref_by_key]
                local_dev_order = list(local_dev_keys)
                summary, _records, _aux, _trace = train_one_stage(
                    no_ref_stage, seed, no_ref_by_key, local_order, local_dev_order,
                    len(dev_keys), device, out, rank, world_size,
                )
                if rank == 0 and summary is not None:
                    no_ref_metrics[seed] = summary
            if rank == 0:
                original_rows = [all_stage_metrics[selected_original][seed] for seed in SEEDS]
                no_rows = [no_ref_metrics[seed] for seed in SEEDS]
                original_hard = np.mean([row["aggregate"]["target_bag_hard_violation"] for row in original_rows])
                no_hard = np.mean([row["aggregate"]["target_bag_hard_violation"] for row in no_rows])
                original_hit = np.mean([row["aggregate"]["target_bag_hit_at1"] for row in original_rows])
                no_hit = np.mean([row["aggregate"]["target_bag_hit_at1"] for row in no_rows])
                original_v2_hard = np.mean([row["breakdowns"]["dataset"]["refer_kitti_v2"]["target_bag_hard_violation"] for row in original_rows])
                no_v2_hard = np.mean([row["breakdowns"]["dataset"]["refer_kitti_v2"]["target_bag_hard_violation"] for row in no_rows])
                no_ref_pass = bool(no_hard <= original_hard - 0.02 and no_hit >= original_hit + 0.02 and no_v2_hard <= original_v2_hard)
                final_selected = no_ref_stage if no_ref_pass else selected_original
                final_status = "no_refpe_selected" if no_ref_pass else "original_content_seed_selected"
                write_json(out / "no_refpe_metrics.json", {
                    "format": "locatemot-l84-no-refpe-metrics-v1", "status": "complete",
                    "selected_original": selected_original, "no_refpe_stage": no_ref_stage,
                    "seed_metrics": {str(seed): value for seed, value in no_ref_metrics.items()},
                    "means": {"original_hard": float(original_hard), "no_refpe_hard": float(no_hard), "original_hit_at1": float(original_hit), "no_refpe_hit_at1": float(no_hit), "original_v2_hard": float(original_v2_hard), "no_refpe_v2_hard": float(no_v2_hard)},
                    "pass": no_ref_pass,
                })
                write_json(out / "final_selection.json", {
                    "format": "locatemot-l84-final-selection-v1", "status": final_status,
                    "selected_representation": final_selected,
                    "selected_original": selected_original, "no_refpe_tested": True,
                    "next_stage": "L85_FULL_RMOT", "candidate_deletion": False,
                    "candidate_truncation": False, "screening_gt_used": False,
                    "official_test_labels_read": False, "hota_trackeval_run": False,
                    "ordinary_mot_ovmot_touched": False,
                })
            if world_size > 1:
                holder = [final_status, final_selected]
                dist.broadcast_object_list(holder, src=0)
                final_status, final_selected = holder
            del no_ref_data, no_ref_by_key, groups
        else:
            if rank == 0:
                write_json(out / "final_selection.json", {
                    "format": "locatemot-l84-final-selection-v1", "status": final_status,
                    "selected_representation": "Z0_fallback", "selected_original": "Z0",
                    "no_refpe_tested": False, "next_stage": "L85_FULL_RMOT",
                    "candidate_deletion": False, "candidate_truncation": False,
                    "screening_gt_used": False, "official_test_labels_read": False,
                    "hota_trackeval_run": False, "ordinary_mot_ovmot_touched": False,
                })
            del data, data_by_key, groups
        if rank == 0:
            init_manifest = [
                file_meta(INIT_ROOT / f"probe_init_seed{seed}.pt")
                for seed in SEEDS
            ]
            write_json(out / "config.json", {
                "format": "locatemot-l84-paired-training-config-v1",
                "thread": THREAD,
                "branch": "codex/l84-paired-middecoder-20260902",
                "representations": list(SELECTED_STAGE_NAMES),
                "seeds": list(SEEDS),
                "train_groups": len(train_keys), "dev_groups": len(dev_keys),
                "epochs": 10, "optimizer": "AdamW", "learning_rate": 2e-4,
                "weight_decay": 1e-4, "warmup_fraction": 0.05,
                "gradient_clip_norm": 1.0, "loss": "l83_target_bag_loss_unchanged",
                "model": {"input_dim": 256, "hidden": 256, "dropout": 0.05, "parameters": 66561},
                "paired_rng": "seed + rank*100003 for Python/NumPy/Torch/ CUDA",
                "schedule_directory": str((ROOT / "outputs/l84/protocol").resolve()),
                "canonical_initializations": init_manifest,
                "no_refpe_test": "only selected non-Z0 decoder stage, same seeds/schedule/loss",
                "candidate_deletion": False, "candidate_truncation": False,
                "screening_gt_used": False, "official_test_labels_read": False,
                "hota_trackeval_run": False, "ordinary_mot_ovmot_touched": False,
            })
            write_json(out / "provenance.json", {
                "format": "locatemot-l84-paired-training-provenance-v1",
                "command": command, "thread": THREAD,
                "project_root": str(ROOT), "source_of_truth": file_meta(ROOT / "outputs/l84/preregister/source_of_truth.json"),
                "frozen_assets": file_meta(ROOT / "outputs/l84/preregister/frozen_assets.json"),
                "manifest": file_meta(MANIFEST), "canonical_initializations": init_manifest,
                "train_videos_from_l82_fit_contract": True,
                "labels": "L49 fit units only; attached after complete feature construction",
                "features": "process-local L69/L82-derived tensors; no dense/raw feature cache",
                "selection": "stable_middecoder_gate_then_single_no_refpe_test",
                "final_status": final_status, "selected_representation": final_selected,
                "candidate_deletion": False, "candidate_truncation": False,
                "screening_gt_used": False, "official_test_labels_read": False,
                "hota_trackeval_run": False, "ordinary_mot_ovmot_touched": False,
                "next_stage": "L85_FULL_RMOT",
            })
            write_json(out / "status.json", {
                "format": "locatemot-l84-final-status-v1", "status": final_status,
                "selected_representation": final_selected,
                "next_stage": "L85_FULL_RMOT", "source_of_truth_status": "complete",
                "candidate_deletion": False, "candidate_truncation": False,
                "screening_gt_used": False, "official_test_labels_read": False,
                "hota_trackeval_run": False, "ordinary_mot_ovmot_touched": False,
            })
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return 0
    except Exception as exc:
        payload = {
            "format": "locatemot-l84-paired-training-status-v1",
            "status": "incomplete", "command": command,
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "next_action": "preserve this attempt and inspect the first traceback before a targeted retry",
            "screening_gt_used": False, "official_test_labels_read": False,
            "hota_trackeval_run": False, "ordinary_mot_ovmot_touched": False,
        }
        write_json(out / "status.json", payload)
        (out / "INCOMPLETE.md").write_text(
            f"L84 paired probe incomplete. First error: {type(exc).__name__}: {exc}\n"
            f"Command: {command}\n"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
