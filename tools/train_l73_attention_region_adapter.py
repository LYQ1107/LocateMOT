#!/usr/bin/env python3
"""Fit-only L73 post-fusion attention adapter contract and smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
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
from torch.nn import functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
L49_ROOT = ROOT / "outputs/l49/data"
L69_ROOT = ROOT / "outputs/l69/attempt9/budget40_features/kitti"
IMAGE_ROOT = ROOT / "data/kitti_tracking_training/image_02"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
FIT_DATASETS = {"refer_kitti_v1", "refer_kitti_v2"}
STRATA = ("positive", "multi_positive", "inactive", "present_uncovered")
SEED = 20260829

sys.path.insert(0, str(ROOT))
from locatemot.models.l73_attention_region_adapter import (  # noqa: E402
    L73AttentionRegionAdapter,
    adapter_config,
)
from tools.audit_l73_postfusion_attention import (  # noqa: E402
    L73Bank,
    capture_prefill,
    normalize_ids,
    read_jsonl,
    sentence_of,
    sha256_file,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def seed_all() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def fit_units() -> list[dict[str, Any]]:
    rows = read_jsonl(L49_ROOT / "train_units.jsonl")
    result = [dict(row) for row in rows
              if str(row.get("split")) == "fit" and str(row.get("dataset")) in FIT_DATASETS]
    if not result:
        raise AssertionError("empty V1/V2 fit unit set")
    return result


def build_metadata(units: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    banks: dict[str, L73Bank] = {}
    result: list[dict[str, Any]] = []
    try:
        for unit in units:
            video = str(unit["video"])
            bank = banks.get(video)
            if bank is None:
                bank = L73Bank(video)
                banks[video] = bank
            rows = bank.rows_for(int(unit["frame_id"]))
            if not rows:
                raise AssertionError(f"empty candidate set {unit.get('unit_key')}")
            result.append({
                "unit": unit,
                "video": video,
                "frame_id": int(unit["frame_id"]),
                "rows": rows,
                "source_category": str(unit.get("category", "unavailable")),
                "candidate_count": len(rows),
                "bank_path": str(bank.path),
                "bank_sha256": bank.sha256,
            })
    finally:
        for bank in banks.values():
            bank.close()
    source_categories = Counter(item["source_category"] for item in result)
    if not set(STRATA).issubset(source_categories):
        raise AssertionError(f"fit sampler lacks required strata: {dict(source_categories)}")
    return result, {
        "fit_unit_count": len(result),
        "fit_videos": sorted({item["video"] for item in result}),
        "fit_video_count": len({item["video"] for item in result}),
        "source_category_counts": dict(sorted(source_categories.items())),
        "candidate_rows_total": int(sum(item["candidate_count"] for item in result)),
        "same_class_hard_negative_metadata": "unavailable",
        "hard_negative_fallback": "all current-frame negative rows",
    }


def sample_order(metadata: list[dict[str, Any]], steps: int) -> list[int]:
    rng = random.Random(SEED)
    pools: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, item in enumerate(metadata):
        key = (str(item["unit"]["dataset"]), item["source_category"])
        pools[key].append(index)
    for bucket in pools.values():
        rng.shuffle(bucket)
    order: list[int] = []
    cursor = Counter()
    for step in range(steps):
        key = sorted(pools)[step % len(pools)]
        bucket = pools[key]
        order.append(bucket[cursor[key] % len(bucket)])
        cursor[key] += 1
    return order


def contract_order(metadata: list[dict[str, Any]]) -> list[int]:
    """Pick a fixed four-unit contract set without reading validation data."""
    desired = [
        ("refer_kitti_v1", "positive"),
        ("refer_kitti_v2", "multi_positive"),
        ("refer_kitti_v1", "inactive"),
        ("refer_kitti_v2", "present_uncovered"),
    ]
    result = []
    for dataset, category in desired:
        match = next((index for index, item in enumerate(metadata)
                      if str(item["unit"]["dataset"]) == dataset and
                      item["source_category"] == category), None)
        if match is None:
            raise AssertionError(f"contract stratum unavailable: {dataset}/{category}")
        result.append(match)
    return result


def load_components():
    from tools.audit_l73_postfusion_attention import load_model
    return load_model()


def stream_features(item: dict[str, Any], bank: L73Bank, la_model, processor, tokenizer):
    from PIL import Image

    unit = item["unit"]
    rows = item["rows"]
    image_path = IMAGE_ROOT / item["video"] / f"{int(unit['frame_id']):06d}.png"
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    image = Image.open(image_path).convert("RGB")
    boxes = bank.tensors["box"][rows].float().tolist()
    capture = capture_prefill(
        la_model, processor, tokenizer, image, sentence_of(unit), boxes,
        retain_vectors=True,
    )
    # Feature construction is complete before reading candidate sidecar labels.
    values = capture.pop("_value_vectors", None)
    query = capture.pop("_query_hidden", None)
    scores = capture.pop("_score_vector", None)
    if values is None or query is None or scores is None:
        raise AssertionError("L73 vectors missing")
    if len(values) != len(rows) or len(scores) != len(rows):
        raise AssertionError("candidate/vector count drift")
    if any(value is None for value in values) or any(score is None for score in scores):
        raise AssertionError("empty L73 region mapping in fit unit")
    region = torch.stack([value.detach().clone().float() for value in values])
    query_tensor = query.detach().clone().float().reshape(1, -1)
    score_tensor = torch.tensor([float(value) for value in scores], dtype=torch.float32).reshape(-1, 1).clone()
    if query_tensor.shape != (1, 2048) or region.shape != (len(rows), 2048):
        raise AssertionError(f"unexpected L73 vector shapes {tuple(region.shape)} {tuple(query_tensor.shape)}")
    if not (torch.isfinite(region).all() and torch.isfinite(query_tensor).all() and torch.isfinite(score_tensor).all()):
        raise AssertionError("nonfinite L73 vectors")
    candidate_indices = bank.tensors["candidate_index"].long().tolist()
    row_keys = []
    for local, candidate_row in enumerate(capture["candidate_rows"]):
        row_offset = int(rows[local])
        key = [str(unit["dataset"]), item["video"], int(unit["query_id"]), int(unit["frame_id"]), str(bank.path), row_offset]
        if int(candidate_row.get("row_offset", row_offset)) != row_offset:
            raise AssertionError("candidate row order drift")
        candidate_row.update({"row_key": key, "row_offset": row_offset,
                              "candidate_index": int(candidate_indices[row_offset])})
        row_keys.append(key)
    if len(row_keys) != len(set(tuple(key) for key in row_keys)):
        raise AssertionError("duplicate immutable row key")
    # Only now read the expression-level sidecar labels for the fit loss.
    targets = normalize_ids(unit.get("target_ids", []))
    labels_sidecar = bank.load_labels()
    labels = torch.tensor([
        labels_sidecar[row] is not None and str(labels_sidecar[row]) in targets for row in rows
    ], dtype=torch.float32)
    actual_category = (
        "multi_positive" if int(labels.sum()) > 1 else
        "positive" if int(labels.sum()) == 1 else
        "present_uncovered" if targets else "inactive"
    )
    capture_summary = {key: value for key, value in capture.items() if not key.startswith("_")}
    future_rows = sum(bank.future_rows(row, int(unit["frame_id"])) for row in rows)
    if future_rows:
        raise AssertionError(f"future history rows: {future_rows}")
    summary = {
        "unit_key": str(unit.get("unit_key")),
        "dataset": str(unit["dataset"]),
        "video": item["video"],
        "query_id": int(unit["query_id"]),
        "frame_id": int(unit["frame_id"]),
        "source_category": item["source_category"],
        "actual_category": actual_category,
        "category_mismatch": item["source_category"] != actual_category,
        "candidate_count": len(rows),
        "positive_count": int(labels.sum()),
        "target_ids": sorted(targets),
        "candidate_present": bool(labels.any()),
        "coverage_mask": not (bool(targets) and not bool(labels.any())),
        "row_keys": row_keys,
        "candidate_key_drift": 0,
        "candidate_truncation": False,
        "future_history_rows": future_rows,
        "labels_joined_after_feature_construction": True,
        "raw_dense_cache_written": False,
        "representation": {
            "region_shape": list(region.shape),
            "query_shape": list(query_tensor.shape),
            "attention_score_shape": list(score_tensor.shape),
            "mapping": "L73 last-layer text-to-image attention weighted value over overlap cells",
            "token_span_alignment": "UNALIGNED",
        },
        "capture": capture_summary,
    }
    del capture, image, boxes
    return {"region": region, "query": query_tensor, "attention_score": score_tensor, "labels": labels}, summary


def loss_for(model: L73AttentionRegionAdapter, features: dict[str, torch.Tensor], category: str):
    # clone() is intentional: capture_prefill returns inference-mode tensors.
    region = features["region"].to("cuda:0").clone()
    query = features["query"].to("cuda:0").clone()
    score = features["attention_score"].to("cuda:0").clone()
    labels = features["labels"].to("cuda:0").clone()
    output = model(query, region, score)
    logits = output["candidate_logits"]
    # Keep masked membership terms attached to the graph.  In particular,
    # present-uncovered units must not create fabricated candidate negatives,
    # but their query-conditioned presence/null auxiliary can still train.
    zero = logits.sum() * 0.0
    if category == "present_uncovered":
        membership, pairwise, minimum, brier = zero, zero, zero, zero
        null_loss = F.binary_cross_entropy_with_logits(
            output["null_logit"], logits.new_zeros((1,))
        )
        supervised = False
    elif category == "inactive":
        membership = F.softplus(logits + 1.0).mean()
        pairwise = minimum = zero
        brier = torch.sigmoid(logits).square().mean()
        null_loss = F.binary_cross_entropy_with_logits(output["null_logit"], logits.new_ones((1,)))
        supervised = True
    else:
        pos, neg = labels > 0.5, labels <= 0.5
        if not bool(pos.any()) or not bool(neg.any()):
            raise AssertionError("covered unit lacks positive or negative candidate")
        pos_logits, neg_logits = logits[pos], logits[neg]
        pos_weight = float(logits.numel()) / max(1, int(pos.sum()))
        neg_weight = float(logits.numel()) / max(1, int(neg.sum()))
        membership = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=torch.tensor(pos_weight / max(1.0, neg_weight), device=logits.device)
        )
        pairwise = F.softplus(0.10 - pos_logits[:, None] + neg_logits[None, :]).mean()
        minimum = F.softplus(0.10 - pos_logits.min() + neg_logits.max())
        brier = (torch.sigmoid(logits) - labels).square().mean()
        null_loss = F.binary_cross_entropy_with_logits(output["null_logit"], logits.new_zeros((1,)))
        supervised = True
    total = membership + 0.5 * pairwise + 0.5 * minimum + 0.1 * null_loss + 0.05 * brier
    return total, output, {
        "membership": float(membership.detach()), "pairwise": float(pairwise.detach()),
        "minimum_positive": float(minimum.detach()), "null": float(null_loss.detach()),
        "brier": float(brier.detach()), "total": float(total.detach()),
        "supervised_membership": supervised,
        "positive_count": int((labels > 0.5).sum()), "negative_count": int((labels <= 0.5).sum()),
        "present_uncovered_masked": category == "present_uncovered",
    }


def gradients(model: L73AttentionRegionAdapter) -> dict[str, Any]:
    rows = []
    finite = True
    nonzero = 0
    for name, parameter in model.named_parameters():
        value = 0.0 if parameter.grad is None else float(parameter.grad.detach().abs().max())
        if parameter.grad is not None:
            finite = finite and bool(torch.isfinite(parameter.grad).all())
        nonzero += int(value > 0.0)
        rows.append({"name": name, "max_abs": value})
    return {
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "finite": finite, "nonzero_parameter_tensors": nonzero, "parameters": rows,
    }


def checkpoint(path: Path, model, optimizer, step: int, config: dict[str, Any]) -> str:
    torch.save({"format": "locatemot-l73-attention-region-adapter-checkpoint-v1",
                "step": int(step), "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "config": config, "contains_frozen_model": False, "contains_raw_cache": False}, path)
    return sha256_file(path)


def run(args: argparse.Namespace) -> int:
    if ROOT != Path.cwd().resolve():
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise AssertionError("immutable manifest SHA mismatch")
    if not torch.cuda.is_available() or os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, "0"):
        raise RuntimeError("L73 requires CUDA_VISIBLE_DEVICES=0")
    seed_all()
    out = args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    units = fit_units()
    metadata, distribution = build_metadata(units)
    actual_steps = 4 if args.contract_only else int(args.steps)
    order = contract_order(metadata) if args.contract_only else sample_order(metadata, actual_steps)
    config = {
        "format": "locatemot-l73-attention-region-adapter-config-v1",
        "stage": "b0_contract" if args.contract_only else "b0_smoke",
        "seed": SEED, "steps": actual_steps, "fit_only": True,
        "fit_source": str(L49_ROOT / "train_units.jsonl"), "fit_datasets": sorted(FIT_DATASETS),
        "adapter": adapter_config(L73AttentionRegionAdapter(hidden=128)),
        "loss": {"balanced_membership_bce": True, "same_frame_pairwise": "all-negative fallback; metadata unavailable",
                 "all_positive_minimum_positive": True, "inactive_no_match": "softplus(logit + 1.0)",
                 "present_uncovered_membership": "masked", "null": "inactive-only", "brier_weight": 0.05},
        "sampling": {"seed": SEED, "dataset_category_round_robin": True, "complete_candidate_sets": True},
        "manifest_sha256": sha256_file(MANIFEST), "token_span_region_alignment": "UNALIGNED",
        "static_motion_mask": "UNALIGNED", "raw_dense_cache_written": False,
    }
    write_json(out / "config.json", config)
    started = time.perf_counter()
    model = L73AttentionRegionAdapter(hidden=128).to("cuda:0").float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    la_model = processor = tokenizer = None
    banks: dict[str, L73Bank] = {}
    losses, samples, grad_trace = [], [], []
    finite_steps = nonzero_steps = 0
    peak = 0
    checkpoint_path = None
    reload_ok = False
    try:
        la_model, processor, tokenizer, transformer_version = load_components()
        for step, metadata_index in enumerate(order, start=1):
            item = metadata[metadata_index]
            bank = banks.get(item["video"])
            if bank is None:
                bank = L73Bank(item["video"])
                banks[item["video"]] = bank
            features, summary = stream_features(item, bank, la_model, processor, tokenizer)
            optimizer.zero_grad(set_to_none=True)
            total, output, detail = loss_for(model, features, summary["actual_category"])
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            if args.contract_only:
                logit_grad = torch.autograd.grad(
                    total, output["candidate_logits"], retain_graph=True, allow_unused=False
                )[0]
                labels = features["labels"].to(logit_grad.device)
                positive = labels > 0.5
                negative = ~positive
                if summary["actual_category"] in ("positive", "multi_positive"):
                    if not bool(positive.any()) or not bool(negative.any()):
                        raise AssertionError("contract covered unit lacks positive/negative rows")
                    positive_nonzero = int((logit_grad[positive].abs() > 0).sum())
                    negative_nonzero = int((logit_grad[negative].abs() > 0).sum())
                    if positive_nonzero != int(positive.sum()) or negative_nonzero != int(negative.sum()):
                        raise AssertionError("covered candidate logit gradient is incomplete")
                    minimum_index = torch.nonzero(positive, as_tuple=False).reshape(-1)[
                        torch.argmin(output["candidate_logits"][positive])
                    ]
                    minimum_nonzero = bool(logit_grad[minimum_index].abs() > 0)
                    if not minimum_nonzero:
                        raise AssertionError("minimum-positive candidate has zero gradient")
                else:
                    positive_nonzero = int((logit_grad[positive].abs() > 0).sum())
                    negative_nonzero = int((logit_grad[negative].abs() > 0).sum())
                    minimum_nonzero = None
                detail["candidate_gradient_audit"] = {
                    "positive_total": int(positive.sum()),
                    "positive_nonzero": positive_nonzero,
                    "negative_total": int(negative.sum()),
                    "negative_nonzero": negative_nonzero,
                    "minimum_positive_nonzero": minimum_nonzero,
                }
                del logit_grad, labels, positive, negative
            total.backward()
            grad = gradients(model)
            if not grad["finite"] or grad["nonzero_parameter_tensors"] == 0:
                raise FloatingPointError(f"invalid adapter gradients at step {step}")
            optimizer.step()
            finite_steps += 1
            nonzero_steps += 1
            peak = max(peak, int(torch.cuda.max_memory_allocated()))
            losses.append({"step": step, **{key: detail[key] for key in ("membership", "pairwise", "minimum_positive", "null", "brier", "total")}})
            samples.append({"step": step, "metadata_index": metadata_index, **{key: summary[key] for key in ("unit_key", "dataset", "video", "frame_id", "source_category", "actual_category", "category_mismatch", "candidate_count", "positive_count", "candidate_key_drift", "candidate_truncation", "future_history_rows")}})
            grad_trace.append({
                "step": step,
                **grad,
                "candidate_gradient_audit": detail.get("candidate_gradient_audit"),
            })
            del features, output, total, detail, grad, summary
            torch.cuda.empty_cache()
        if args.contract_only:
            checkpoint_path = out / "checkpoint_l73_contract_step4.pt"
            checkpoint_sha = checkpoint(checkpoint_path, model, optimizer, actual_steps, config)
        else:
            checkpoint_path = out / f"checkpoint_l73_attention_region_step{args.steps}.pt"
            checkpoint_sha = checkpoint(checkpoint_path, model, optimizer, args.steps, config)
        package = torch.load(checkpoint_path, map_location="cuda:0", weights_only=False)
        reloaded = L73AttentionRegionAdapter(hidden=128).to("cuda:0").float()
        reloaded.load_state_dict(package["model"], strict=True)
        reload_ok = set(reloaded.state_dict()) == set(model.state_dict())
        del package, reloaded
        metrics_name = "metrics_l73_contract_step4.json" if args.contract_only else f"metrics_l73_step{args.steps}.json"
        write_json(out / metrics_name, {
            "format": "locatemot-l73-attention-region-contract-v1" if args.contract_only else "locatemot-l73-attention-region-smoke-v1",
            "status": "complete", "stage": config["stage"], "steps": actual_steps,
            "finite_steps": finite_steps, "nonzero_gradient_steps": nonzero_steps,
            "strict_reload": reload_ok, "checkpoint": str(checkpoint_path), "checkpoint_sha256": checkpoint_sha,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "loss_trace": losses, "gradient_audit": grad_trace, "sampling_trace": samples,
            "fit_distribution": distribution, "candidate_key_drift": 0, "candidate_truncation": False,
            "detector_frozen": all(not parameter.requires_grad for parameter in la_model.parameters()),
            "peak_cuda_bytes": peak, "elapsed_seconds": time.perf_counter() - started,
            "raw_dense_cache_written": False, "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        })
        write_json(out / "loss_trace.json", losses)
        write_json(out / "sampling_trace.json", {"records": samples, "fit_distribution": distribution})
        write_json(out / "reload_audit.json", {"format": "locatemot-l73-adapter-reload-audit-v1", "status": "complete", "strict_reload": reload_ok, "checkpoint_sha256": checkpoint_sha})
        write_json(out / "provenance.json", {
            "format": "locatemot-l73-attention-region-provenance-v1", "status": "complete", "cwd": str(ROOT),
            "command": " ".join(sys.argv), "interpreter": sys.executable, "torch": torch.__version__,
            "transformers": transformer_version, "device": "cuda:0", "seed": SEED, "fit_only": True,
            "fit_source": str(L49_ROOT / "train_units.jsonl"), "fit_distribution": distribution,
            "manifest_sha256": sha256_file(MANIFEST), "bank_sha256": {item["video"]: item["bank_sha256"] for item in metadata},
            "representation_source": "L73 frozen post-fusion final-layer text-to-image attention",
            "same_class_hard_negative_metadata": "unavailable", "token_span_region_alignment": "UNALIGNED",
            "static_motion_mask": "UNALIGNED", "raw_dense_feature_cache_written": False,
            "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False,
        })
        write_json(out / "status.json", {"format": "locatemot-l73-attention-region-run-v1", "status": "complete", "stage": config["stage"], "steps": actual_steps, "finite_steps": finite_steps, "nonzero_gradient_steps": nonzero_steps, "strict_reload": reload_ok, "checkpoint": str(checkpoint_path), "peak_cuda_bytes": peak, "elapsed_seconds": time.perf_counter() - started, "next_action": "review B0 implementation evidence before any semantic interpretation"})
        return 0
    finally:
        for bank in banks.values():
            bank.close()
        del la_model, processor, tokenizer, model, optimizer
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--contract-only", action="store_true")
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        # A non-empty attempt is immutable evidence. Do not overwrite its
        # status or traceback when a caller accidentally reuses the path.
        if args.out.exists() and any(args.out.iterdir()):
            return 1
        args.out.mkdir(parents=True, exist_ok=True)
        failure = {"format": "locatemot-l73-attention-region-run-v1", "status": "incomplete", "stage": "b0_contract" if args.contract_only else "b0_smoke", "steps": 1 if args.contract_only else args.steps, "cwd": str(Path.cwd()), "command": " ".join(sys.argv), "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "next_action": "preserve attempt and fix only the first actionable root cause", "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False}
        write_json(args.out / "status.json", failure)
        (args.out / "INCOMPLETE.md").write_text("# INCOMPLETE\n\nFirst actionable root cause: `" + failure["failure_root_cause"] + "`\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
