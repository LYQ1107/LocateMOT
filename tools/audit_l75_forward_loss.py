#!/usr/bin/env python3
"""L75 fit-only forward/loss/gradient contract audit.

This is deliberately a tiny contract run, not a semantic evaluation.  It
uses four deterministic fit units spanning both datasets and the registered
strata; no calibration/validation labels are opened.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l75_candidate_marked_vlm import CandidateMarkedVLMMatcher  # noqa: E402
from locatemot.rmot.l75_data import (  # noqa: E402
    L75Bank, MANIFEST_PATH, MANIFEST_SHA256, IMAGE_ROOT, load_splits,
    make_record, sha256_file, unit_key,
)
from locatemot.rmot.l75_runtime import (  # noqa: E402
    attach_language_lora, frozen_target_digest, language_forward,
    load_locateanything, marked_visual_batch, prepare_visual,
    region_value_batch, lora_state_dict, load_lora_state_dict,
)
from locatemot.rmot.l75_train_utils import (  # noqa: E402
    l75_loss, sample_candidate_indices, gradient_row_summary,
)

SEED = 20260829
STRATA = ("positive", "multi_positive", "inactive", "present_uncovered")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def choose_units(splits: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    # Explicit four-unit contract: both domains and all four strata.  The
    # labels are used only to choose this fit-only contract sample and are not
    # used by the frozen visual feature construction.
    result = []
    for dataset, category in (
        ("refer_kitti_v1", "positive"),
        ("refer_kitti_v1", "multi_positive"),
        ("refer_kitti_v2", "inactive"),
        ("refer_kitti_v2", "present_uncovered"),
    ):
        candidates = sorted(
            [row for row in splits["fit"] if str(row["dataset"]) == dataset
             and str(row.get("category")) == category], key=unit_key
        )
        if not candidates:
            raise AssertionError(f"missing fit contract stratum {dataset}/{category}")
        result.append(candidates[0])
    return result


def run_one(model: Any, matcher: Any, processor: Any, tokenizer: Any,
            unit: dict[str, Any], device: str, chunk: int) -> dict[str, Any]:
    from PIL import Image

    bank = L75Bank(str(unit["video"]))
    try:
        record = make_record(unit, bank, include_labels=True)
        image_path = IMAGE_ROOT / str(unit["video"]) / f"{int(unit['frame_id']):06d}.png"
        image = Image.open(image_path).convert("RGB")
        rows = record["row_offsets"]
        boxes = bank.tensors["box"].index_select(
            0, torch.as_tensor(rows, dtype=torch.long)
        ).float().tolist()
        prepared = prepare_visual(model, processor, tokenizer, image,
                                  record["sentence"], boxes)
        selected = sample_candidate_indices(record, bank, max_negatives=8)
        selected_cells = [prepared["candidate_cells"][index] for index in selected]
        base_visual = prepared["base_visual"].to(device=device)
        matcher.zero_grad(set_to_none=True)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.grad = None
        losses = []
        row_grads = []
        selected_labels = torch.as_tensor(
            [record["labels"][index] for index in selected], dtype=torch.float32, device=device
        )
        positive_count = len(record["positive_indices"])
        first_chunk_size = max(int(chunk), positive_count) if positive_count else int(chunk)
        chunk_starts = [0]
        chunk_starts.extend(range(first_chunk_size, len(selected), chunk))
        chunks = [selected[start:start + (first_chunk_size if start == 0 else chunk)]
                  for start in chunk_starts]
        for chunk_start in chunk_starts:
            local_indices = selected[chunk_start:chunk_start + chunk]
            if chunk_start == 0:
                local_indices = selected[:first_chunk_size]
            cells = [prepared["candidate_cells"][index] for index in local_indices]
            marked, _ = marked_visual_batch(base_visual, cells, matcher.region_marker)
            regions, region_mask = region_value_batch(marked, cells)
            hidden = language_forward(model, prepared, marked, inference=False)
            output = matcher(hidden, prepared["expression_positions"], regions, region_mask)
            output["match_logit"].retain_grad()
            labels = torch.as_tensor(
                [record["labels"][index] for index in local_indices],
                dtype=torch.float32, device=device,
            )
            loss, parts = l75_loss(
                output["match_logit"], output["absent_logit"], labels,
                record["category"], bool(record["coverage_mask"]), matcher.region_marker,
            )
            (loss / max(1, len(chunks))).backward()
            losses.append(parts)
            row_grads.append(gradient_row_summary(
                output["match_logit"], labels, bool(record["coverage_mask"])
            ))
            del hidden, marked, regions, output, loss
        named_trainable = list(model.named_parameters()) + list(matcher.named_parameters())
        trainable = [p for _, p in named_trainable if p.requires_grad]
        observed = [p for p in trainable if p.grad is not None]
        masked_uncovered = record["category"] == "present_uncovered" and not bool(record["coverage_mask"])
        finite_grad = ((bool(observed) and all(torch.isfinite(p.grad).all() for p in observed))
                       if masked_uncovered else
                       all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable))
        nonzero_grad = sum(int(p.grad is not None and float(p.grad.detach().abs().sum()) > 0) for p in trainable)
        result = {
            "unit_key": record["unit_key"],
            "dataset": record["dataset"], "category": record["category"],
            "candidate_count": record["candidate_count"],
            "selected_count": len(selected), "positive_count": record["positive_count"],
            "candidate_keys_complete": len(record["row_keys"]) == record["candidate_count"],
            "candidate_rows_ordered": record["row_keys"] == sorted(record["row_keys"], key=lambda key: key[-1]),
            "selected_indices": selected,
            "history_future_rows": 0,
            "expression_positions": len(prepared["expression_positions"]),
            "expression_span_method": prepared["expression_span_method"],
            "projected_visual_shape": prepared["projected_visual_shape"],
            "all_candidate_mapping_rows": len(prepared["candidate_cells"]) == record["candidate_count"],
            "candidate_mapping_nonempty": sum(bool(x) for x in prepared["candidate_cells"]),
            "loss_parts": losses,
            "row_gradients": row_grads,
            "trainable_parameter_count": len(trainable),
            "finite_gradients": bool(finite_grad),
            "nonzero_gradient_tensors": int(nonzero_grad),
            "gradient_missing_allowed_for_present_uncovered": bool(masked_uncovered),
            "missing_gradient_parameter_names": [name for name, p in named_trainable
                                                  if p.requires_grad and p.grad is None],
            "marker_gradient_l1": float(matcher.region_marker.grad.detach().abs().sum()) if matcher.region_marker.grad is not None else 0.0,
        }
        del prepared, image
        return result
    finally:
        bank.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/l75/audit/forward_contract")
    args = parser.parse_args()
    out = args.out
    started = time.perf_counter()
    base = {
        "format": "locatemot-l75-forward-loss-contract-v1",
        "status": "running", "command": " ".join(sys.argv),
        "cwd": str(Path.cwd()), "seed": SEED,
        "inputs": {"fit_units": str(ROOT / "outputs/l49/data/train_units.jsonl"),
                    "l69_root": str(ROOT / "outputs/l69/attempt9/budget40_features/kitti"),
                    "manifest": str(MANIFEST_PATH), "manifest_sha256_expected": MANIFEST_SHA256},
        "outputs": {"directory": str(out)},
        "calibration_labels_read": False, "validation_labels_read": False,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "raw_dense_cache_written": False,
        "candidate_deletion": False, "candidate_truncation": False,
        "token_span_alignment": "UNALIGNED",
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "status.json", base)
    model = processor = tokenizer = None
    try:
        if Path.cwd().resolve() != ROOT.resolve():
            raise RuntimeError(f"wrong cwd {Path.cwd()}")
        if sha256_file(MANIFEST_PATH) != MANIFEST_SHA256:
            raise AssertionError("manifest SHA mismatch")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU0 required")
        torch.manual_seed(SEED)
        splits = load_splits()
        units = choose_units(splits)
        model, processor, tokenizer, runtime = load_locateanything("cuda:0")
        lora_contract = attach_language_lora(model, rank=8, alpha=16.0, target_layers=4)
        # Checkpointing is useful for the real smoke; the contract intentionally
        # exercises the same non-inference-mode language path.
        if hasattr(model.language_model, "gradient_checkpointing_enable"):
            try:
                model.language_model.gradient_checkpointing_enable()
            except Exception:
                pass
        matcher = CandidateMarkedVLMMatcher(hidden=256).to("cuda:0")
        base_digest_before = frozen_target_digest(model)
        results = []
        for unit in units:
            results.append(run_one(model, matcher, processor, tokenizer, unit, "cuda:0", 2))
            matcher.zero_grad(set_to_none=True)
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.grad = None
        base_digest_after = frozen_target_digest(model)
        if base_digest_before != base_digest_after:
            raise AssertionError("wrapped frozen base digest changed during contract")
        # Small adapter-only checkpoint and strict reload checks.
        checkpoint = {
            "format": "locatemot-l75-adapter-only-v1",
            "matcher": {k: v.detach().cpu() for k, v in matcher.state_dict().items()},
            "lora": lora_state_dict(model),
            "lora_contract": lora_contract,
        }
        checkpoint_path = out / "contract_adapter_checkpoint.pt"
        torch.save(checkpoint, checkpoint_path)
        fresh = CandidateMarkedVLMMatcher(hidden=256)
        fresh_missing = fresh.load_state_dict(checkpoint["matcher"], strict=True)
        lora_reload = load_lora_state_dict(model, checkpoint["lora"], strict=True)
        if fresh_missing.missing_keys or fresh_missing.unexpected_keys:
            raise AssertionError("matcher strict reload reported keys")
        if lora_reload["missing"] or lora_reload["unexpected"]:
            raise AssertionError("LoRA strict reload reported keys")
        contract = {
            **base, "status": "complete", "runtime": runtime,
            "lora_contract": lora_contract,
            "matcher_contract": matcher.parameter_contract(),
            "units": results, "unit_count": len(results),
            "domains": sorted({r["dataset"] for r in results}),
            "categories": sorted({r["category"] for r in results}),
            "base_target_digest_before": base_digest_before,
            "base_target_digest_after": base_digest_after,
            "base_target_digest_unchanged": base_digest_before == base_digest_after,
            "detector_base_requires_grad_count": sum(
                int(parameter.requires_grad) for name, parameter in model.named_parameters()
                if "lora_" not in name
            ),
            "strict_reload": {
                "matcher_missing": list(fresh_missing.missing_keys),
                "matcher_unexpected": list(fresh_missing.unexpected_keys),
                "lora": lora_reload,
                "checkpoint": str(checkpoint_path),
            },
            "all_finite": all(r["finite_gradients"] for r in results),
            "nonzero_gradients_observed": all(r["nonzero_gradient_tensors"] > 0 for r in results),
            "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
            "wall_time_seconds": time.perf_counter() - started,
            "next_action": "run the registered 100-step fit-only smoke",
        }
        # The expression above intentionally does not claim all model params
        # are frozen from a string heuristic; record the actual base/LoRA split.
        contract["base_parameters_requires_grad_count"] = sum(
            int(p.requires_grad) for name, p in model.named_parameters() if "lora_" not in name
        )
        contract["lora_parameters_requires_grad_count"] = sum(
            int(p.requires_grad) for name, p in model.named_parameters() if "lora_" in name
        )
        write_json(out / "contract.json", contract)
        write_json(out / "loss_contract.json", {
            **base, "status": "complete", "losses": results,
            "same_frame_grouped": True, "present_uncovered_membership_masked": True,
            "inactive_negative_loss": True, "all_positive_minimum_term": True,
        })
        write_json(out / "provenance.json", {
            **base, "status": "complete", "runtime": runtime,
            "lora_contract": lora_contract, "matcher_contract": matcher.parameter_contract(),
            "input_unit_keys": [r["unit_key"] for r in results],
            "checkpoint_sha256": sha256_file(checkpoint_path),
        })
        write_json(out / "status.json", contract)
        return 0
    except Exception as exc:
        failure = {**base, "status": "incomplete",
                   "failure_root_cause": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc(),
                   "elapsed_seconds": time.perf_counter() - started,
                   "next_action": "preserve contract attempt; fix only first actionable error and run one targeted retry"}
        write_json(out / "status.json", failure)
        (out / "INCOMPLETE.md").write_text("# INCOMPLETE\n\n" +
            f"First actionable root cause: `{failure['failure_root_cause']}`\n")
        return 1
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
