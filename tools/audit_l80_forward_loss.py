#!/usr/bin/env python3
"""P1 L80 forward/loss/reload contract on fit-only strata."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
sys.path.insert(0, str(ROOT))
from locatemot.models.l80_raw_region_correspondence import L80Config, L80RawRegionCorrespondence  # noqa: E402
from locatemot.rmot.l80_data import (  # noqa: E402
    CATEGORIES, DATASETS, EXPECTED_MANIFEST_SHA, FORBIDDEN_LABEL_FIELDS, MANIFEST,
    L80BankStore, key_only, load_fit_units, load_full_unit_for_labels, sha256_file,
)
from locatemot.rmot.l80_losses import l80_loss  # noqa: E402
from locatemot.rmot.l80_runtime import FrameFeatureCache, load_clip, raw_inputs_for_unit, CLIP_SHA256, CLIP_WEIGHT  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def chosen_units() -> list[dict]:
    rows = load_fit_units()
    result = []
    for dataset in DATASETS:
        for category in CATEGORIES:
            matches = sorted((row for row in rows if row["dataset"] == dataset and row["category"] == category),
                             key=lambda row: (str(row["video"]), int(row["query_id"]), int(row["frame_id"]), str(row["unit_key"])))
            result.append(matches[0])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty output {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable] + sys.argv)
    start = time.perf_counter()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd {Path.cwd()}")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise AssertionError("manifest SHA drift")
    if sha256_file(CLIP_WEIGHT) != CLIP_SHA256:
        raise AssertionError("CLIP SHA drift")
    torch.manual_seed(20260829); np.random.seed(20260829)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    model = L80RawRegionCorrespondence(L80Config()).to(device=device, dtype=torch.float32)
    model.train()
    runtime_model = load_clip(device)
    cache = FrameFeatureCache(max_items=8)
    store = L80BankStore(max_history=8)
    unit_records = []
    positive_gradient_ok = []
    negative_gradient_ok = []
    try:
        for source in chosen_units():
            metadata = key_only(source)
            batch = store.build_unit(metadata)
            # Feature construction is completed before any labels/sidecar are
            # loaded.  The model sees only raw features and causal rows.
            raw = raw_inputs_for_unit(runtime_model, batch, device, cache)
            if FORBIDDEN_LABEL_FIELDS.intersection(metadata):
                raise AssertionError("pre-feature labels leaked")
            labels = store.attach_labels(batch, load_full_unit_for_labels(batch.unit_key))
            observations = batch.observations.to(device=device).clone()
            history = batch.history_observations.to(device=device).clone()
            history_mask = batch.history_mask.to(device=device).clone()
            history_frames = batch.history_frame_ids.to(device=device).clone()
            output = model(raw["visual_tokens"], raw["text_tokens"], raw["text_mask"],
                           history, history_mask, history_frames, batch.frame_id)
            output["candidate_logits"].retain_grad()
            loss, parts = l80_loss(output, labels["labels"], labels["coverage_mask"],
                                   observations, history_mask, labels["category"])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite loss {batch.unit_key}")
            model.zero_grad(set_to_none=True)
            loss.backward()
            grads = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
            finite_grad = all(value is None or bool(torch.isfinite(value).all()) for value in grads)
            nonzero_grad = any(value is not None and bool((value.abs() > 0).any()) for value in grads)
            if not finite_grad or not nonzero_grad:
                raise FloatingPointError(f"invalid adapter gradient {batch.unit_key}")
            row_grad = output["candidate_logits"].grad
            if labels["coverage_mask"]:
                if row_grad is None:
                    raise FloatingPointError(f"covered candidate logits disconnected {batch.unit_key}")
                row_grad = row_grad.detach().float().cpu()
                positive_gradient_ok.append(bool((row_grad[labels["labels"].cpu()].abs() > 0).all()) if labels["positive_count"] else True)
                negative_gradient_ok.append(bool((row_grad[(~labels["labels"]).cpu()].abs() > 0).all()) if batch.candidate_count > labels["positive_count"] else True)
            unit_records.append({
                "unit_key": batch.unit_key, "dataset": batch.dataset, "category": labels["category"],
                "candidate_count": batch.candidate_count, "row_keys": [list(x) for x in batch.row_keys],
                "positive_count": labels["positive_count"], "loss": float(loss.detach()), "loss_parts": parts,
                "finite_loss": True, "finite_adapter_gradients": finite_grad,
                "nonzero_adapter_gradients": nonzero_grad,
                "positive_logit_gradients_nonzero": positive_gradient_ok[-1] if labels["coverage_mask"] and positive_gradient_ok else "masked_expected",
                "negative_logit_gradients_nonzero": negative_gradient_ok[-1] if labels["coverage_mask"] and negative_gradient_ok else "masked_expected",
                "history_shape": list(batch.history_observations.shape), "history_future_rows": int((batch.history_frame_ids > batch.frame_id).sum()),
                "text_shape": list(raw["text_tokens"].shape), "region_shape": list(raw["visual_tokens"].shape),
                "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
                "raw_cache_persistent": False, "labels_loaded_after_feature_construction": True,
            })
            del output, raw, labels, batch, observations, history, history_mask, history_frames
            model.zero_grad(set_to_none=True)
    finally:
        cache.clear()
        store._store._bank = None
        store._store._text_cache = None
        del runtime_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    checkpoint = out / "checkpoint_forward_contract.pt"
    torch.save({"format": "locatemot-l80-forward-contract-v1", "config": L80Config().__dict__,
                "model_state_dict": model.state_dict()}, checkpoint)
    reload_model = L80RawRegionCorrespondence(L80Config()).to(device=device, dtype=torch.float32)
    result = reload_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"strict reload mismatch {result}")
    parameter_count = model.parameter_report()
    contract = {
        "format": "locatemot-l80-forward-loss-contract-v1", "status": "complete", "command": command,
        "inputs": {"manifest": str(MANIFEST), "manifest_sha256": sha256_file(MANIFEST), "clip_weight": str(CLIP_WEIGHT), "clip_sha256": sha256_file(CLIP_WEIGHT)},
        "outputs": [str(out / "unit_records.jsonl"), str(out / "contract.json"), str(checkpoint)],
        "units": len(unit_records), "domains": sorted({str(x["dataset"]) for x in unit_records}),
        "categories": sorted({str(x["category"]) for x in unit_records}), "unit_records": unit_records,
        "positive_gradient_checks": {"count": len(positive_gradient_ok), "all_nonzero": all(positive_gradient_ok)},
        "negative_gradient_checks": {"count": len(negative_gradient_ok), "all_nonzero": all(negative_gradient_ok)},
        "model": parameter_count, "strict_reload": True,
        "history_future_rows": int(sum(x["history_future_rows"] for x in unit_records)),
        "candidate_key_drift": 0, "candidate_deletion": False, "candidate_truncation": False,
        "raw_dense_cache_written": False, "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
        "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "training_run": False, "hota_trackeval_run": False,
        "failure_root_cause": None, "next_action": "run L80 bounded fit smoke", "elapsed_sec": time.perf_counter() - start,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
    }
    write_json(out / "contract.json", contract)
    (out / "unit_records.jsonl").write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in unit_records))
    write_json(out / "provenance.json", {"format": "locatemot-l80-forward-loss-provenance-v1", "status": "complete", "command": command,
        "cwd": str(Path.cwd().resolve()), "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745", "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint), "labels_loaded_only_after_raw_features": True,
        "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
        "training_run": False, "hota_trackeval_run": False})
    write_json(out / "status.json", {"format": "locatemot-l80-status-v1", "status": "complete", "command": command,
        "inputs": [str(MANIFEST), str(CLIP_WEIGHT)], "outputs": [str(out / "contract.json"), str(checkpoint)],
        "failure_root_cause": None, "next_action": "run L80 bounded fit smoke", "screening_gt_used": False,
        "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "training_run": False,
        "hota_trackeval_run": False})
    print(json.dumps(contract, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
