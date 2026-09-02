#!/usr/bin/env python3
"""L83 frozen-representation probe with faithful unique target-bag loss."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from locatemot.evaluation.l83_target_bag_metrics import aggregate_group_metrics, breakdowns, group_metrics
from locatemot.models.l83_faithful_rank_probe import L83FaithfulRankProbe, L83RankProbeConfig
from locatemot.rmot.l83_target_bag_loss import l83_target_bag_loss

THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
REPRESENTATIONS = ("l81_candidate_evidence", "l59_fused_roi", "l82_candidate_reference")
SPLIT = ROOT / "outputs/l82/protocol/fit_video_train_dev_split.json"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
BASELINE = ROOT / "outputs/l83/baselines/corrected_old_probe_attempt1/corrected_old_probe_metrics.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def meta(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns, "sha256": sha256(path)}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def finite_gradients(parameters: Any) -> tuple[bool, float]:
    values = [p.grad for p in parameters if p.grad is not None]
    if not values:
        return False, 0.0
    finite = all(bool(torch.isfinite(value).all()) for value in values)
    norm = float(torch.stack([value.detach().float().norm() for value in values]).norm())
    return bool(finite and math.isfinite(norm) and norm > 0.0), norm


def train_representation(
    representation: str, train_data: list[Any], dev_data: list[Any],
    device: torch.device, out: Path, rank: int, world_size: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    base = L83FaithfulRankProbe(L83RankProbeConfig()).to(device=device, dtype=torch.float32)
    model: torch.nn.Module = base
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(base, device_ids=[device.index], output_device=device.index)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    updates_per_epoch = len(train_data)
    total_updates = updates_per_epoch * 10
    warmup = max(1, int(round(total_updates * 0.05)))

    def schedule(step: int) -> float:
        if step <= warmup:
            return float(step) / float(warmup)
        progress = min(1.0, max(0.0, (step - warmup) / max(1, total_updates - warmup)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule(int(step)))
    trace: list[dict[str, Any]] = []
    step = 0
    max_grad = 0.0
    for epoch in range(1, 11):
        model.train()
        for data in train_data:
            step += 1
            optimizer.zero_grad(set_to_none=True)
            value = data.features[representation].to(device=device, dtype=torch.float32).clone()
            output = model(value)
            loss, parts = l83_target_bag_loss(
                output["interaction"], data.membership_mask.to(device), data.categories,
                data.target_ids, data.candidate_gt,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite L83 loss {representation} step {step}")
            loss.backward()
            valid, grad_norm = finite_gradients(base.parameters())
            if not valid:
                raise FloatingPointError(f"invalid/nonzero gradient failure {representation} step {step}")
            clipped = float(torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0))
            if not math.isfinite(clipped) or clipped <= 0.0:
                raise FloatingPointError(f"invalid clipped gradient {representation} step {step}")
            optimizer.step()
            scheduler.step()
            max_grad = max(max_grad, clipped)
            trace.append({
                "representation": representation, "epoch": epoch, "step": step,
                "loss": float(loss.detach().cpu()), "gradient_norm": grad_norm,
                "clipped_gradient_norm": clipped, "lr": float(optimizer.param_groups[0]["lr"]),
                **{key: (float(item.detach().cpu()) if torch.is_tensor(item) else item) for key, item in parts.items()},
                "finite": True, "nonzero_gradient": True,
            })
            del value, output, loss
    if step != total_updates:
        raise AssertionError(f"update count drift {representation}: {step}/{total_updates}")
    if world_size > 1:
        dist.barrier()
    model.eval()
    dev_records: list[dict[str, Any]] = []
    dev_aux: list[dict[str, Any]] = []
    with torch.inference_mode():
        for data in dev_data:
            value = data.features[representation].to(device=device, dtype=torch.float32).clone()
            first = model(value)
            second = model(value)
            scores = first["interaction"].float().cpu()
            repeat_noise = float((first["interaction"] - second["interaction"]).abs().max().cpu())
            record, auxiliary = group_metrics(data, scores)
            record["repeat_run_noise"] = repeat_noise
            dev_records.append(record)
            dev_aux.append(auxiliary)
            del value, first, second
    package = {
        "format": "locatemot-l83-faithful-rank-probe-checkpoint-v1",
        "stage": "phase_7_faithful_target_bag_probe", "representation": representation,
        "epoch": 10, "step": step, "seed": SEED,
        "model_config": {"input_dim": 256, "hidden": 256, "dropout": 0.05},
        "model_state_dict": base.state_dict(),
        "model_parameter_count": int(sum(p.numel() for p in base.parameters())),
        "loss_contract": {"bag_margin": 0.50, "query_margin": 0.25, "inactive_margin": 0.25, "weights": {"bag_cls": 1.0, "candidate_axis": 1.0, "query_axis": 0.50, "inactive": 0.25}},
        "metric_contract": {"target_bag_score": "max per unique candidate_gt target", "background": "singleton negative row", "row_metrics": "ROW_DIAGNOSTIC", "query_swap_auc": "independent rank-based ROC-AUC"},
        "candidate_deletion": False, "candidate_truncation": False,
    }
    path = out / "checkpoints" / f"{representation}_checkpoint_epoch10.pt"
    if rank == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(package, path)
    if world_size > 1:
        dist.barrier()
    info = {"representation": representation, "checkpoint": meta(path) if rank == 0 else {"path": str(path.resolve())}, "trace_steps": len(trace), "max_gradient_norm": max_grad, "parameter_report": base.parameter_report()}
    del model, base, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return info, dev_records, dev_aux, {"loss_trace": trace}


def gate_for_representation(name: str, old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    old_v2 = old["breakdowns"]["dataset"].get("refer_kitti_v2", {})
    new_v2 = new["breakdowns"]["dataset"].get("refer_kitti_v2", {})
    old_hard = old["aggregate"].get("target_bag_hard_violation")
    new_hard = new["aggregate"].get("target_bag_hard_violation")
    old_hit = old["aggregate"].get("target_bag_hit_at1")
    new_hit = new["aggregate"].get("target_bag_hit_at1")
    old_multi = old["aggregate"].get("multi_target_exact_topT")
    new_multi = new["aggregate"].get("multi_target_exact_topT")
    old_swap = old["aggregate"].get("query_swap_pair_accuracy")
    new_swap = new["aggregate"].get("query_swap_pair_accuracy")
    old_v2_hard = old_v2.get("target_bag_hard_violation")
    new_v2_hard = new_v2.get("target_bag_hard_violation")
    checks = {
        "G1_corrected_bag_hard": old_hard is not None and new_hard is not None and ((old_hard <= 0.75 and new_hard >= old_hard - 0.01) or (old_hard > 0.75 and old_hard - new_hard >= 0.05)),
        "G2_bag_hit_at1": old_hit is not None and new_hit is not None and new_hit - old_hit >= 0.08,
        "G3_multi_target_exact_topT": old_multi is not None and new_multi is not None and new_multi - old_multi >= 0.08,
        "G4_query_swap_pair_accuracy": old_swap is not None and new_swap is not None and new_swap >= old_swap - 0.02,
        "G5_v2_bag_hard": old_v2_hard is not None and new_v2_hard is not None and old_v2_hard - new_v2_hard >= 0.03,
        "G6_complete_finite_no_deletion": bool(new.get("all_dev_groups_present")) and new["aggregate"].get("finite") is True and new["aggregate"].get("candidate_deletion") is False and new["aggregate"].get("candidate_truncation") is False,
    }
    return {"representation": name, "checks": checks, "passed": all(checks.values()), "old": {"hard": old_hard, "hit_at1": old_hit, "multi_exact": old_multi, "swap": old_swap, "v2_hard": old_v2_hard}, "new": {"hard": new_hard, "hit_at1": new_hit, "multi_exact": new_multi, "swap": new_swap, "v2_hard": new_v2_hard}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 4:
        raise RuntimeError(f"world size exceeds four: {world_size}")
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256(MANIFEST) != EXPECTED_MANIFEST:
        raise AssertionError("fixed manifest SHA drift")
    if world_size > 1:
        dist.init_process_group(backend="nccl", init_method="env://")
    if not torch.cuda.is_available():
        raise RuntimeError("L83 faithful probe requires CUDA")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(SEED)
    np.random.seed(SEED + rank)
    from tools.l82_train_frozen_rank_probe import build_local_groups, load_groups
    groups, train_keys, dev_keys = load_groups()
    all_keys = train_keys + dev_keys
    local_keys = all_keys[rank::world_size]
    local_train = [key for key in train_keys if key in set(local_keys)]
    local_dev = [key for key in dev_keys if key in set(local_keys)]
    if not local_train or not local_dev:
        raise AssertionError(f"rank {rank} train/dev split empty: {len(local_train)}/{len(local_dev)}")
    build_started = time.perf_counter()
    data, build_info = build_local_groups(local_keys, groups, device)
    by_key = {item.group_key: item for item in data}
    train_data = [by_key[key] for key in local_train]
    dev_data = [by_key[key] for key in local_dev]
    local_summaries: dict[str, Any] = {}
    local_records: dict[str, list[dict[str, Any]]] = {}
    local_aux: dict[str, list[dict[str, Any]]] = {}
    local_traces: dict[str, list[dict[str, Any]]] = {}
    for representation in REPRESENTATIONS:
        info, records, auxiliaries, trace = train_representation(representation, train_data, dev_data, device, out, rank, world_size)
        local_summaries[representation] = info
        local_records[representation] = records
        local_aux[representation] = auxiliaries
        local_traces[representation] = trace["loss_trace"]

    gathered_records: dict[str, list[list[dict[str, Any]]]] = {}
    gathered_aux: dict[str, list[list[dict[str, Any]]]] = {}
    gathered_traces: dict[str, list[list[dict[str, Any]]]] = {}
    summaries_holder: list[dict[str, Any] | None] = [None] * world_size if rank == 0 else []
    build_holder: list[dict[str, Any] | None] = [None] * world_size if rank == 0 else []
    if world_size > 1:
        for representation in REPRESENTATIONS:
            holder = [None] * world_size if rank == 0 else None
            dist.gather_object(local_records[representation], holder, dst=0)
            if rank == 0:
                gathered_records[representation] = holder or []
            holder_aux = [None] * world_size if rank == 0 else None
            dist.gather_object(local_aux[representation], holder_aux, dst=0)
            if rank == 0:
                gathered_aux[representation] = holder_aux or []
            holder_trace = [None] * world_size if rank == 0 else None
            dist.gather_object(local_traces[representation], holder_trace, dst=0)
            if rank == 0:
                gathered_traces[representation] = holder_trace or []
        dist.gather_object(local_summaries, summaries_holder if rank == 0 else None, dst=0)
        dist.gather_object(build_info, build_holder if rank == 0 else None, dst=0)
    else:
        for representation in REPRESENTATIONS:
            gathered_records[representation] = [local_records[representation]]
            gathered_aux[representation] = [local_aux[representation]]
            gathered_traces[representation] = [local_traces[representation]]
        summaries_holder = [local_summaries]
        build_holder = [build_info]

    if rank == 0:
        with BASELINE.open() as handle:
            baseline = json.load(handle)
        metrics: dict[str, Any] = {}
        compact = []
        gates = []
        for representation in REPRESENTATIONS:
            records = [record for shard in gathered_records[representation] for record in (shard or [])]
            auxiliaries = [aux for shard in gathered_aux[representation] for aux in (shard or [])]
            old = baseline["metrics"][representation]
            metrics[representation] = {
                "aggregate": aggregate_group_metrics(records, auxiliaries),
                "breakdowns": breakdowns(records, auxiliaries),
                "group_records": len(records), "all_dev_groups_present": len({str(row["group_key"]) for row in records}) == len(dev_keys),
                "old_corrected_baseline": old["aggregate"],
            }
            gates.append(gate_for_representation(representation, old, metrics[representation]))
            compact.extend({key: value for key, value in record.items() if not key.startswith("_")} | {"representation": representation} for record in records)
        qualifying = [item["representation"] for item in gates if item["passed"] and item["representation"] in {"l59_fused_roi", "l82_candidate_reference"}]
        gate_status = "faithful_target_bag_training_gate_pass" if qualifying else "faithful_target_bag_training_gate_fail"
        gate = {
            "format": "locatemot-l83-faithful-target-bag-gate-v1", "status": gate_status,
            "stage": "phase_7_faithful_frozen_representation_probe", "checks_by_representation": gates,
            "qualifying_grounding_representations": qualifying,
            "thresholds": {"G1_hard_improvement": 0.05, "G1_floor_if_old_le_075": 0.01, "G2_bag_hit_at1": 0.08, "G3_multi_target_exact": 0.08, "G4_swap_pair_drop_max": 0.02, "G5_v2_hard_improvement": 0.03},
            "selection_tuple": ["lower corrected target-bag hard violation", "higher target-bag hit@1", "higher multi-target exact top-T", "higher query-swap margin", "higher V2 bag hit@1", "simpler representation"],
            "selection": "none until at least one GroundingDINO representation passes; no dev-based post-selection",
            "candidate_deletion": False, "candidate_truncation": False, "finite": all(item["all_dev_groups_present"] and item["aggregate"]["finite"] for item in metrics.values()),
            "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "training_run": True, "hota_trackeval_run": False, "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
        }
        write_json(out / "representation_metrics.json", {"format": "locatemot-l83-faithful-representation-metrics-v1", "status": gate_status, "metrics": metrics, "train_group_count": len(train_keys), "dev_group_count": len(dev_keys)})
        (out / "dev_group_metrics.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True, default=str) for row in compact) + "\n")
        for representation in REPRESENTATIONS:
            trace = [item for shard in gathered_traces[representation] for item in (shard or [])]
            (out / f"loss_trace_{representation}.json").write_text(json.dumps(trace, indent=2, sort_keys=True, default=str) + "\n")
        write_json(out / "faithful_gate.json", gate)
        write_json(out / "summary.json", {"format": "locatemot-l83-faithful-probe-summary-v1", "status": gate_status, "train_group_count": len(train_keys), "dev_group_count": len(dev_keys), "gpu_world_size": world_size, "feature_build_wall_seconds": time.perf_counter() - build_started, "build_info": build_holder, "representation_metrics": metrics, "gate": gate})
        write_json(out / "provenance.json", {"format": "locatemot-l83-faithful-probe-provenance-v1", "status": gate_status, "command": " ".join([sys.executable] + sys.argv), "cwd": str(ROOT), "luna_thread": THREAD, "seed": SEED, "inputs": {"manifest": {"path": str(MANIFEST), "sha256": sha256(MANIFEST)}, "split": {"path": str(SPLIT), "sha256": sha256(SPLIT)}, "corrected_baseline": {"path": str(BASELINE), "sha256": sha256(BASELINE)}}, "split": {"train_groups": len(train_keys), "dev_groups": len(dev_keys), "video_disjoint": True, "labels": "fit-only labels attached after complete representation construction"}, "loss": {"same_class_hard_negative_metadata": "unavailable", "hard_negative_fallback": "all non-referred target bags/background singleton bags"}, "resources": {"features_persistent": False, "raw_dense_cache_written": False, "gpu_world_size": world_size, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}, "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False, "candidate_deletion": False, "candidate_truncation": False, "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED", "l81_modified": False, "l82_modified": False, "uidm_shared_checkpoint_modified": False})
        write_json(out / "status.json", {"format": "locatemot-l83-faithful-probe-status-v1", "status": gate_status, "failure_root_cause": None if qualifying else "grounding_representation_target_separation_insufficient", "next_action": "run decoder sharpness audit only if a GroundingDINO representation passes all G1-G6" if qualifying else "stop L83 before decoder/factorized/task-composition phases and wait for supervisor", "command": " ".join([sys.executable] + sys.argv)})
    if world_size > 1:
        dist.barrier()
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    if rank == 0:
        print(json.dumps({"status": gate_status, "out": str(out)}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        rank = int(os.environ.get("RANK", "0"))
        out_arg = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else str(ROOT / "outputs/l83/train/faithful_bag")
        out = Path(out_arg)
        if not out.is_absolute():
            out = ROOT / out
        out.mkdir(parents=True, exist_ok=True)
        text = f"# INCOMPLETE L83 faithful probe\n\nrank={rank}\nfirst_error={type(exc).__name__}: {exc}\n\n```text\n{traceback.format_exc()}\n```\n"
        (out / f"INCOMPLETE.rank{rank}.md").write_text(text)
        if rank == 0:
            (out / "INCOMPLETE.md").write_text(text)
        raise
