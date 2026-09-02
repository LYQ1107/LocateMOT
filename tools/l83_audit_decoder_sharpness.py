#!/usr/bin/env python3
"""L83 pre-registered GroundingDINO decoder sharpness decomposition.

This audit is deliberately separate from L83 Phase 7's gate.  It runs the
same 66,561-parameter faithful target-bag probe on Z0, Zp and every fixed-
reference decoder layer.  Native iterative refinement is retained only as a
frozen diagnostic and never changes an external L69 candidate row.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from locatemot.evaluation.l83_target_bag_metrics import aggregate_group_metrics, breakdowns
from locatemot.models.l83_grounding_state_audit import capture_grounding_stages, compare_state_vectors
from locatemot.rmot.l80_data import L80BankStore

THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
BASELINE = ROOT / "outputs/l83/baselines/corrected_old_probe_attempt1/corrected_old_probe_metrics.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns, "sha256": sha256_file(path)}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def quantile_summary(values: list[float]) -> dict[str, Any]:
    finite = np.asarray([float(x) for x in values if math.isfinite(float(x))], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0, "finite": True}
    return {
        "count": int(finite.size), "mean": float(finite.mean()), "std": float(finite.std()),
        "p05": float(np.quantile(finite, 0.05)), "p50": float(np.quantile(finite, 0.50)),
        "p95": float(np.quantile(finite, 0.95)), "finite": True,
    }


def effective_rank(sample: torch.Tensor) -> float:
    if sample.shape[0] < 2:
        return 0.0
    centered = sample.float() - sample.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    denominator = float(energy.square().sum())
    if denominator <= 0.0:
        return 0.0
    return float(energy.sum().square() / denominator)


def distribution_summary(values: list[torch.Tensor], native_query: torch.Tensor) -> dict[str, Any]:
    flat = torch.cat([value.reshape(-1, value.shape[-1]).float().cpu() for value in values], dim=0)
    if not bool(torch.isfinite(flat).all()):
        raise FloatingPointError("nonfinite flattened decoder state")
    sample = flat[:1024]
    norms = flat.norm(dim=-1)
    native = native_query.float().cpu()
    native_norms = native.norm(dim=-1)
    native_centered = native - native.mean(dim=0, keepdim=True)
    native_singular = torch.linalg.svdvals(native_centered)
    native_energy = native_singular.square()
    native_denominator = float(native_energy.square().sum())
    state_centered = sample - sample.mean(dim=0, keepdim=True)
    state_singular = torch.linalg.svdvals(state_centered)
    state_energy = state_singular.square()
    state_denominator = float(state_energy.square().sum())
    native_basis = torch.linalg.svd(native_centered, full_matrices=False).Vh[:32]
    centered_sample = sample - native.mean(dim=0, keepdim=True)
    projection = centered_sample @ native_basis.T @ native_basis
    projection_ratio = float(projection.square().sum() / centered_sample.square().sum().clamp_min(1e-12))
    normalized_sample = torch.nn.functional.normalize(sample, dim=-1)
    normalized_native_mean = torch.nn.functional.normalize(native.mean(dim=0, keepdim=True), dim=-1)
    cosine_to_native_mean = (normalized_sample @ normalized_native_mean.T).reshape(-1)
    pair_sample = normalized_sample[:256]
    pair_matrix = pair_sample @ pair_sample.T
    upper = pair_matrix[torch.triu(torch.ones_like(pair_matrix, dtype=torch.bool), diagonal=1)]
    return {
        "count": int(flat.shape[0]), "dimension": int(flat.shape[-1]),
        "norm": quantile_summary(norms.tolist()),
        "dimension_mean": float(flat.mean()), "dimension_std": float(flat.std(unbiased=False)),
        "effective_rank": float(state_energy.sum().square() / state_denominator) if state_denominator > 0 else 0.0,
        "pair_cosine": quantile_summary(upper.tolist()),
        "cosine_to_native_query_mean": quantile_summary(cosine_to_native_mean.tolist()),
        "native_query": {
            "count": int(native.shape[0]), "dimension": int(native.shape[-1]),
            "norm": quantile_summary(native_norms.tolist()),
            "dimension_mean": float(native.mean()), "dimension_std": float(native.std(unbiased=False)),
            "effective_rank": float(native_energy.sum().square() / native_denominator) if native_denominator > 0 else 0.0,
        },
        "native_query_subspace": {
            "rank": 32, "projection_energy_ratio": projection_ratio,
            "residual_energy_ratio": float(1.0 - projection_ratio),
        },
        "norm_ratio_to_native_mean": float(norms.mean() / native_norms.mean().clamp_min(1e-12)),
    }


def candidate_pair_summary(values: list[torch.Tensor]) -> dict[str, Any]:
    pairs: list[float] = []
    for tensor in values:
        normalized = torch.nn.functional.normalize(tensor.float().cpu(), dim=-1)
        for query in normalized:
            if query.shape[0] < 2:
                continue
            matrix = query @ query.T
            upper = matrix[torch.triu(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)]
            pairs.extend(float(x) for x in upper[:256].tolist())
    return quantile_summary(pairs)


def native_query_stats(model: Any) -> torch.Tensor:
    query = getattr(model, "query_embedding", None)
    if query is None or not hasattr(query, "weight"):
        raise AssertionError("native query_embedding.weight is unavailable")
    value = query.weight.detach().float().cpu().clone()
    if value.ndim != 2 or not bool(torch.isfinite(value).all()):
        raise AssertionError(f"invalid native query embedding: {tuple(value.shape)}")
    return value


def build_group_states(
    group_keys: list[str], groups: dict[str, dict[str, Any]], device: torch.device,
) -> tuple[list[Any], dict[str, Any], list[dict[str, Any]], torch.Tensor]:
    """Build all states before opening fit labels, then attach fit labels."""
    from tools.l82_train_frozen_rank_probe import attach_fit_labels, load_fit_label_rows
    from locatemot.rmot.l82_grounding_runtime import GroundingCandidateReferenceRuntime, install_clip_torchvision_compat

    install_clip_torchvision_compat()
    runtime = GroundingCandidateReferenceRuntime(device)
    store = L80BankStore(max_history=8)
    label_rows: dict[str, dict[str, Any]] | None = None
    built: list[Any] = []
    native_diagnostics: list[dict[str, Any]] = []
    stage_names: list[str] | None = None
    started = time.perf_counter()
    total_native = 0.0
    total_replay = 0.0

    for group_key in group_keys:
        group = groups[group_key]
        batches = [store.build_unit(row) for row in group["queries"]]
        if not batches:
            raise AssertionError(f"empty group: {group_key}")
        first = batches[0]
        n = first.candidate_count
        if n <= 0 or len(first.row_offsets) != n:
            raise AssertionError(f"empty candidate set: {group_key}")
        if any(batch.row_offsets != first.row_offsets for batch in batches):
            raise AssertionError(f"same-frame row order drift: {group_key}")
        if any(batch.candidate_count != n for batch in batches):
            raise AssertionError(f"same-frame candidate count drift: {group_key}")
        runtime.encoder_events.clear()
        runtime.capture.clear()
        native_started = time.perf_counter()
        with torch.inference_mode():
            native = runtime.inference_detector(runtime.model, str(first.image_path), text_prompt=str(first.sentence), custom_entities=True)
        native_seconds = time.perf_counter() - native_started
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
        for batch in batches:
            text_dict, caption, token_map = runtime.make_text_dict(runtime.model, str(batch.sentence), device, force_pad_to_max=True)
            sample = copy.deepcopy(sample_template)
            runtime.set_sample_text(sample, caption, token_map)
            before = len(runtime.encoder_events)
            replay_started = time.perf_counter()
            with torch.inference_mode():
                runtime.original_forward_transformer(visual_feats, text_dict, [sample])
            total_replay += time.perf_counter() - replay_started
            if len(runtime.encoder_events) != before + 1:
                raise AssertionError(f"replay encoder event drift: {batch.unit_key}")
            replay_event = runtime.encoder_events[-1]
            state = capture_grounding_stages(runtime.model, replay_event, batch.boxes.to(device), image_shape, scale_factor)
            if state["candidate_count"] != n:
                raise AssertionError(f"state candidate count drift: {batch.unit_key}")
            if stage_names is None:
                stage_names = list(state["stages"].keys())
            if list(state["stages"].keys()) != stage_names:
                raise AssertionError(f"decoder layer count drift: {batch.unit_key}")
            for name, value in state["stages"].items():
                if value.shape != (n, 256) or not bool(torch.isfinite(value).all()):
                    raise AssertionError(f"state shape/finite drift {name}: {batch.unit_key}")
                per_stage[name].append(value.float().cpu().contiguous())
            native_layer_deltas = []
            for layer_index in range(len(state["native_stages"]) - 1):
                fixed_name = f"Z{layer_index + 1}"
                native_name = f"R{layer_index + 1}"
                native_layer_deltas.append(compare_state_vectors(state["stages"][fixed_name], state["native_stages"][native_name]))
            native_diagnostics.append({
                "group_key": str(group_key), "unit_key": str(batch.unit_key),
                "candidate_count": n, "fixed_hidden_shape": state["fixed_hidden_shape"],
                "native_hidden_shape": state["native_hidden_shape"],
                "native_reference_shape": state["native_reference_shape"],
                "native_vs_fixed_layer_deltas": native_layer_deltas,
                "all_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            })
            runtime.encoder_events.clear()
            del state, replay_event, sample, text_dict
        total_native += native_seconds
        feature_values = {name: torch.stack(values) for name, values in per_stage.items()}
        # This is the explicit label boundary: the current group's complete
        # raw decoder states exist before any fit label row is opened.
        if label_rows is None:
            label_rows = load_fit_label_rows()
        data = attach_fit_labels(batches, feature_values, label_rows, store, group, {
            "native_seconds": native_seconds, "image_shape": image_shape,
        })
        built.append(data)
        del batches, feature_values, event, native, visual_feats, sample_template
        runtime.capture.clear()
        runtime.encoder_events.clear()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if stage_names is None or not built:
        raise AssertionError("no decoder states built")
    native_query = native_query_stats(runtime.model)
    model_info = dict(runtime.model_info)
    model_info.update({
        "native_query_embedding_shape": list(native_query.shape),
        "encoder_source": "GroundingCandidateReferenceRuntime.encoder hook",
        "fixed_reference_decoder": True,
        "native_refinement_decoder": True,
        "state_stage_names": stage_names,
        "feature_construction_wall_seconds": time.perf_counter() - started,
        "native_seconds_sum": total_native,
        "replay_seconds_sum": total_replay,
        "features_persistent": False,
        "features_in_memory_only": True,
    })
    runtime.close()
    del runtime, store, label_rows
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return built, model_info, native_diagnostics, native_query


def score_entropy_for_dev(model_path: Path, dev_data: list[Any], device: torch.device) -> list[dict[str, Any]]:
    from locatemot.models.l83_faithful_rank_probe import L83FaithfulRankProbe, L83RankProbeConfig

    package = torch.load(model_path, map_location="cpu")
    model = L83FaithfulRankProbe(L83RankProbeConfig(**package["model_config"])).to(device=device, dtype=torch.float32)
    model.load_state_dict(package["model_state_dict"], strict=True)
    model.eval()
    rows = []
    with torch.inference_mode():
        for data in dev_data:
            value = data.features[model_path.stem.replace("_checkpoint_epoch10", "")].to(device=device, dtype=torch.float32).clone()
            scores = model(value)["interaction"].float().cpu()
            normalized_entropy = []
            for query_scores in scores:
                probabilities = torch.softmax(query_scores, dim=0)
                entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
                normalized_entropy.append(float(entropy / math.log(max(2, int(query_scores.numel())))))
            rows.append({
                "group_key": str(data.group_key),
                "score_entropy_normalized_mean": float(np.mean(normalized_entropy)) if normalized_entropy else None,
                "score_entropy_normalized_values": normalized_entropy,
                "score_std": float(scores.std(unbiased=False)),
                "finite": bool(torch.isfinite(scores).all()),
            })
            del value, scores
    del package, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows


def aggregate_native_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        for index, delta in enumerate(row["native_vs_fixed_layer_deltas"], 1):
            by_layer[index].append(float(delta["mean_l2"]))
    return {
        "unit_count": len(rows),
        "layer_mean_l2": {str(layer): quantile_summary(values) for layer, values in sorted(by_layer.items())},
        "all_rows_retained": all(bool(row["all_rows_retained"]) for row in rows),
        "candidate_deletion": any(bool(row["candidate_deletion"]) for row in rows),
        "candidate_truncation": any(bool(row["candidate_truncation"]) for row in rows),
    }


def stage_conclusion(stage_metrics: dict[str, Any]) -> dict[str, Any]:
    def bag_hit(metrics: dict[str, Any]) -> float | None:
        """Return the registered hit@1 field without changing the metric."""
        value = metrics.get("target_bag_hit_at1")
        if value is None:
            value = metrics.get("target_bag_recall_at1")
        return float(value) if value is not None else None

    names = list(stage_metrics)
    if "Z0" not in stage_metrics or "Zp" not in stage_metrics:
        return {"status": "decoder_sharpness_audit_inconclusive", "reason": "missing Z0/Zp"}
    z0 = stage_metrics["Z0"]["aggregate"]
    zp = stage_metrics["Zp"]["aggregate"]
    l59_reference = stage_metrics["Z0"].get("corrected_l59_baseline", {})
    l59_hard = l59_reference.get("target_bag_hard_violation")
    l59_hit = l59_reference.get("target_bag_recall_at1")
    l59_v2_hard = l59_reference.get("v2_target_bag_hard_violation")
    l59_v2_hit = l59_reference.get("v2_target_bag_hit_at1")
    fixed = [name for name in names if name.startswith("Z") and name[1:].isdigit()]
    fixed = sorted(fixed, key=lambda name: (0 if name == "Z0" else 1 if name == "Zp" else 2, int(name[1:]) if name[1:].isdigit() else -1))
    z0_hard = z0.get("target_bag_hard_violation")
    z0_hit = bag_hit(z0)
    zp_hard = zp.get("target_bag_hard_violation")
    zp_hit = bag_hit(zp)
    if all(value is not None for value in (z0_hard, z0_hit, zp_hard, zp_hit)) and zp_hard >= z0_hard + 0.03 and zp_hit <= z0_hit - 0.03:
        return {
            "status": "remove_refpe_from_content_keep_reference_as_query_pos",
            "evidence": {"Z0_hard": z0_hard, "Zp_hard": zp_hard, "Z0_hit_at1": z0_hit, "Zp_hit_at1": zp_hit},
        }
    qualifying = []
    for name in fixed:
        if name in {"Z0", "Zp"}:
            continue
        current = stage_metrics[name]["aggregate"]
        v2 = stage_metrics[name]["breakdowns"].get("dataset", {}).get("refer_kitti_v2", {})
        v2_hard = v2.get("target_bag_hard_violation")
        v2_hit = bag_hit(v2)
        current_hit = bag_hit(current)
        if all(value is not None for value in (current.get("target_bag_hard_violation"), current_hit, l59_hard, l59_hit, v2_hard, v2_hit, l59_v2_hard, l59_v2_hit)):
            if (current["target_bag_hard_violation"] <= l59_hard - 0.03 and
                    current_hit >= l59_hit + 0.05 and
                    v2_hard <= l59_v2_hard - 0.03 and
                    v2_hit >= l59_v2_hit + 0.05):
                qualifying.append(name)
    if qualifying:
        return {"status": "selected_semantic_layer", "selected_semantic_layer": sorted(qualifying, key=lambda name: int(name[1:]))[0], "qualifying_layers": qualifying}
    best = min(
        [name for name in fixed if stage_metrics[name]["aggregate"].get("target_bag_hard_violation") is not None],
        key=lambda name: (stage_metrics[name]["aggregate"]["target_bag_hard_violation"], -float(bag_hit(stage_metrics[name]["aggregate"]) or -1.0)),
    )
    best_value = stage_metrics[best]["aggregate"]
    no_layer_joint = all(
        name in {"Z0", "Zp"} or not (
            stage_metrics[name]["aggregate"].get("target_bag_hard_violation") is not None and
            bag_hit(stage_metrics[name]["aggregate"]) is not None and
            z0_hard is not None and z0_hit is not None and
            stage_metrics[name]["aggregate"]["target_bag_hard_violation"] < z0_hard and
            bag_hit(stage_metrics[name]["aggregate"]) > z0_hit
        ) for name in fixed
    )
    if best == "Z0" or no_layer_joint:
        return {
            "status": "decoder_not_authorized_for_primary_semantic_branch",
            "best_stage_by_hard_then_hit": best,
            "best_stage_metrics": best_value,
            "Z0_metrics": z0,
        }
    return {"status": "decoder_sharpness_audit_inconclusive", "best_stage_by_hard_then_hit": best, "best_stage_metrics": best_value}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-groups", type=int, default=0, help="small train+dev contract subset; zero means the full preregistered split")
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    existing = list(out.iterdir()) if out.exists() else []
    # A shell launcher may create an empty log file before Python starts.  It
    # is not an evidence collision; every other pre-existing entry remains a
    # hard stop so completed/failed attempts cannot be overwritten.
    allow_launcher_log = os.environ.get("L83_ALLOW_PRECREATED_LAUNCHER_LOG") == "1"
    if existing and not (allow_launcher_log and all(item.name == "launcher.log" for item in existing)):
        raise FileExistsError(f"refusing nonempty L83 decoder audit output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    command = " ".join([sys.executable] + sys.argv)
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if world_size > 4:
            raise RuntimeError(f"world size exceeds four: {world_size}")
        if not torch.cuda.is_available():
            raise RuntimeError("L83 decoder audit requires CUDA")
        if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        if world_size > 1:
            dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(SEED + rank)
        np.random.seed(SEED + rank)

        from tools.l82_train_frozen_rank_probe import load_groups
        from tools.l83_train_faithful_bag_probe import train_representation

        groups, train_keys, dev_keys = load_groups()
        all_keys = train_keys + dev_keys
        if args.max_groups:
            if args.max_groups < 2:
                raise ValueError("--max-groups must leave at least one train and one dev group")
            all_keys = train_keys[:1] + dev_keys[: max(1, args.max_groups - 1)]
            train_keys = train_keys[:1]
            dev_keys = dev_keys[: max(1, args.max_groups - 1)]
        local_keys = all_keys[rank::world_size]
        local_train_set = set(train_keys)
        local_train_keys = [key for key in local_keys if key in local_train_set]
        local_dev_keys = [key for key in local_keys if key in set(dev_keys)]
        if not local_train_keys or not local_dev_keys:
            raise AssertionError(f"rank split empty: {len(local_train_keys)}/{len(local_dev_keys)}")
        data, model_info, local_native_diag, native_query = build_group_states(local_keys, groups, device)
        by_key = {item.group_key: item for item in data}
        train_data = [by_key[key] for key in local_train_keys]
        dev_data = [by_key[key] for key in local_dev_keys]
        stage_names = list(train_data[0].features.keys())

        local_records: dict[str, list[dict[str, Any]]] = {}
        local_aux: dict[str, list[dict[str, Any]]] = {}
        local_traces: dict[str, list[dict[str, Any]]] = {}
        local_entropy: dict[str, list[dict[str, Any]]] = {}
        local_summaries: dict[str, Any] = {}
        for stage in stage_names:
            info, records, auxiliaries, trace = train_representation(stage, train_data, dev_data, device, out, rank, world_size)
            local_summaries[stage] = info
            local_records[stage] = records
            local_aux[stage] = auxiliaries
            local_traces[stage] = trace["loss_trace"]
            checkpoint = out / "checkpoints" / f"{stage}_checkpoint_epoch10.pt"
            if world_size > 1:
                dist.barrier()
            local_entropy[stage] = score_entropy_for_dev(checkpoint, dev_data, device)
            if world_size > 1:
                dist.barrier()

        if world_size > 1:
            gathered_records: dict[str, list[list[dict[str, Any]]]] = {}
            gathered_aux: dict[str, list[list[dict[str, Any]]]] = {}
            gathered_traces: dict[str, list[list[dict[str, Any]]]] = {}
            gathered_entropy: dict[str, list[list[dict[str, Any]]]] = {}
            gathered_summaries: list[dict[str, Any] | None] | None = [None] * world_size if rank == 0 else None
            gathered_model_info: list[dict[str, Any] | None] | None = [None] * world_size if rank == 0 else None
            gathered_native: list[list[dict[str, Any]] | None] | None = [None] * world_size if rank == 0 else None
            gathered_query: list[torch.Tensor | None] | None = [None] * world_size if rank == 0 else None
            for stage in stage_names:
                holder = [None] * world_size if rank == 0 else None
                dist.gather_object(local_records[stage], holder, dst=0)
                if rank == 0:
                    gathered_records[stage] = holder or []
                holder = [None] * world_size if rank == 0 else None
                dist.gather_object(local_aux[stage], holder, dst=0)
                if rank == 0:
                    gathered_aux[stage] = holder or []
                holder = [None] * world_size if rank == 0 else None
                dist.gather_object(local_traces[stage], holder, dst=0)
                if rank == 0:
                    gathered_traces[stage] = holder or []
                holder = [None] * world_size if rank == 0 else None
                dist.gather_object(local_entropy[stage], holder, dst=0)
                if rank == 0:
                    gathered_entropy[stage] = holder or []
            dist.gather_object(local_summaries, gathered_summaries, dst=0)
            dist.gather_object(model_info, gathered_model_info, dst=0)
            dist.gather_object(local_native_diag, gathered_native, dst=0)
            dist.gather_object(native_query, gathered_query, dst=0)
        else:
            gathered_records = {stage: [local_records[stage]] for stage in stage_names}
            gathered_aux = {stage: [local_aux[stage]] for stage in stage_names}
            gathered_traces = {stage: [local_traces[stage]] for stage in stage_names}
            gathered_entropy = {stage: [local_entropy[stage]] for stage in stage_names}
            gathered_summaries = [local_summaries]
            gathered_model_info = [model_info]
            gathered_native = [local_native_diag]
            gathered_query = [native_query]

        # Every rank must participate in this compact gather.  The actual
        # tensors remain local; only distribution summaries cross ranks.
        local_distribution = {}
        for stage in stage_names:
            values = [item.features[stage].detach().cpu() for item in data]
            local_distribution[stage] = {
                "distribution": distribution_summary(values, native_query),
                "candidate_pair_cosine": candidate_pair_summary(values),
            }
        if world_size > 1:
            distribution_holders: list[dict[str, Any] | None] = [None] * world_size if rank == 0 else None
            dist.gather_object(local_distribution, distribution_holders, dst=0)
        else:
            distribution_holders = [local_distribution]

        if rank == 0:
            all_native = [row for shard in (gathered_native or []) for row in (shard or [])]
            query_values = [value for value in (gathered_query or []) if value is not None]
            if not query_values:
                raise AssertionError("native query statistics were not gathered")
            native_query_cpu = query_values[0].float().cpu()
            if any(not torch.equal(native_query_cpu, value.float().cpu()) for value in query_values[1:] if value is not None):
                raise AssertionError("native query embedding differs across ranks")

            # Features remain in RAM only on each rank; rank zero receives only
            # compact summaries and group metrics below.
            stage_metrics: dict[str, Any] = {}
            distribution: dict[str, Any] = {}
            score_records: list[dict[str, Any]] = []
            for stage in stage_names:
                records = [row for shard in gathered_records[stage] for row in (shard or [])]
                auxiliaries = [row for shard in gathered_aux[stage] for row in (shard or [])]
                entropy_rows = [row for shard in gathered_entropy[stage] for row in (shard or [])]
                records_by_key = {str(row["group_key"]): row for row in records}
                entropy_by_key = {str(row["group_key"]): row for row in entropy_rows}
                for key, row in records_by_key.items():
                    row["score_entropy_normalized_mean"] = entropy_by_key.get(key, {}).get("score_entropy_normalized_mean")
                    row["score_std_from_reload"] = entropy_by_key.get(key, {}).get("score_std")
                    row["stage"] = stage
                    score_records.append(row)
                ordered_records = [records_by_key[key] for key in sorted(records_by_key)]
                ordered_aux = [auxiliaries[index] for index in np.argsort([str(row["group_key"]) for row in records]).tolist()] if len(auxiliaries) == len(records) else auxiliaries
                aggregate = aggregate_group_metrics(ordered_records, ordered_aux)
                stage_metrics[stage] = {
                    "aggregate": aggregate,
                    "breakdowns": breakdowns(ordered_records, ordered_aux),
                    "group_count": len(ordered_records),
                    "all_dev_groups_present": len({str(row["group_key"]) for row in ordered_records}) == len(dev_keys),
                    "score_entropy_normalized": quantile_summary([row["score_entropy_normalized_mean"] for row in ordered_records if row.get("score_entropy_normalized_mean") is not None]),
                    "checkpoint": file_meta(out / "checkpoints" / f"{stage}_checkpoint_epoch10.pt"),
                    "loss_trace_steps": sum(len(shard or []) for shard in gathered_traces[stage]),
                }
                # Do not persist the full stage tensors; reconstruct summaries
                # from the compact local data is intentionally omitted on rank
                # zero.  Every rank writes its distribution summary below via
                # a small gather of just statistics.

            distribution = {}
            for stage in stage_names:
                shards = [item[stage] for item in (distribution_holders or []) if item and stage in item]
                distribution[stage] = {
                    "shard_count": len(shards),
                    "local_summaries": shards,
                    "note": "statistics are per-rank summaries; feature tensors were not serialized",
                }
            baseline = json.loads(BASELINE.read_text())
            baseline_l59 = baseline["metrics"]["l59_fused_roi"]["aggregate"]
            baseline_l59_v2 = baseline["metrics"]["l59_fused_roi"]["breakdowns"]["dataset"]["refer_kitti_v2"]
            for stage in stage_metrics:
                stage_metrics[stage]["corrected_l59_baseline"] = {
                    "target_bag_hard_violation": baseline_l59.get("target_bag_hard_violation"),
                    "target_bag_recall_at1": baseline_l59.get("target_bag_hit_at1"),
                    "multi_target_exact_topT": baseline_l59.get("multi_target_exact_topT"),
                    "query_swap_pair_accuracy": baseline_l59.get("query_swap_pair_accuracy"),
                    "v2_target_bag_hard_violation": baseline_l59_v2.get("target_bag_hard_violation"),
                    "v2_target_bag_hit_at1": baseline_l59_v2.get("target_bag_hit_at1"),
                }
            conclusion = stage_conclusion(stage_metrics)
            write_json(out / "decoder_sharpness.json", {
                "format": "locatemot-l83-decoder-sharpness-audit-v1",
                "status": "complete",
                "command": command,
                "luna_thread": THREAD,
                "seed": SEED,
                "train_group_count": len(train_keys), "dev_group_count": len(dev_keys),
                "train_video_disjoint_dev": True,
                "stage_names": stage_names,
                "fixed_reference_contract": {"Z0": "L59 fused ROI visual_seed", "Zp": "visual_seed + pretrained reference position", "Z1_to_ZL": "fixed-reference decoder layer outputs"},
                "native_refinement_control": "R0..RL is frozen diagnostic only; refined references never replace L69 rows",
                "stage_metrics": stage_metrics,
                "distribution_shift": distribution,
                "native_refinement_summary": aggregate_native_diagnostics(all_native),
                "native_model_info": gathered_model_info[0] if gathered_model_info else model_info,
                "conclusion": conclusion,
                "candidate_deletion": False, "candidate_truncation": False,
                "features_persistent": False,
                "screening_gt_used": False, "official_test_labels_read": False,
                "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
                "decoder_audit_not_a_semantic_gate": True,
            })
            append_jsonl(out / "dev_group_metrics.jsonl", score_records)
            write_json(out / "provenance.json", {
                "format": "locatemot-l83-decoder-sharpness-provenance-v1",
                "status": "complete", "command": command, "cwd": str(ROOT), "luna_thread": THREAD,
                "inputs": {"manifest": file_meta(MANIFEST), "corrected_baseline": file_meta(BASELINE)},
                "protocol": {"representation_stages": stage_names, "probe": "L83 L83FaithfulRankProbe + faithful target-bag loss", "epochs": 10, "optimizer": "AdamW lr=2e-4 wd=1e-4", "seed": SEED, "fit_groups": len(train_keys), "dev_groups": len(dev_keys), "native_refinement": "diagnostic only"},
                "resources": {"gpu_world_size": world_size, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "features_persistent": False, "raw_dense_cache_written": False},
                "labels": "fit-only expression labels attached after complete state construction; no fixed validation labels",
                "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                "candidate_deletion": False, "candidate_truncation": False, "l81_modified": False, "l82_modified": False, "uidm_shared_checkpoint_modified": False,
                "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            })
            write_json(out / "status.json", {
                "format": "locatemot-l83-decoder-sharpness-status-v1", "status": "complete",
                "command": command, "failure_root_cause": None,
                "next_action": conclusion.get("status"), "decoder_sharpness_conclusion": conclusion,
                "screening_gt_used": False, "official_test_labels_read": False,
                "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            })
            write_json(out / "config.json", {
                "format": "locatemot-l83-decoder-sharpness-config-v1", "seed": SEED,
                "stages": stage_names, "model_probe": {"input_dim": 256, "hidden": 256, "dropout": 0.05, "parameter_count": 66561},
                "schedule": {"epochs": 10, "lr": 2e-4, "weight_decay": 1e-4, "warmup_fraction": 0.05, "cosine": True, "clip_norm": 1.0},
                "fixed_reference": True, "native_refinement_control": True, "candidate_deletion": False, "candidate_truncation": False,
            })
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        del data, train_data, dev_data
        gc.collect()
        return 0
    except Exception as exc:
        payload = {
            "format": "locatemot-l83-decoder-sharpness-status-v1", "status": "decoder_sharpness_audit_inconclusive",
            "command": command, "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "next_action": "preserve this attempt and inspect the first traceback before retrying",
            "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        }
        write_json(out / "status.json", payload)
        (out / "INCOMPLETE.md").write_text(f"L83 decoder sharpness audit incomplete. First error: {type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
