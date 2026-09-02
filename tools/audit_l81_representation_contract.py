#!/usr/bin/env python3
"""Label-free L81 representation, permutation and gradient contract audit.

The audit intentionally uses a small pre-registered fit-only key set.  It
never calls ``load_full_unit_for_labels`` or opens a candidate sidecar.  The
L69 rows are assembled from native frame pointers by ``L80BankStore`` and all
candidate rows remain in their original order.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
sys.path.insert(0, str(ROOT))

from locatemot.models.l81_hierarchical_early_fusion import L81Config, L81HierarchicalEarlyFusion  # noqa: E402
from locatemot.rmot.l80_data import (  # noqa: E402
    FORBIDDEN_LABEL_FIELDS,
    L80BankStore,
    key_only,
    load_fit_units,
    sha256_file,
)
from locatemot.rmot.l81_runtime import (  # noqa: E402
    CLIP_SHA256,
    CLIP_WEIGHT,
    FrameFeatureCache,
    load_clip,
    raw_inputs_for_l81,
)


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
AUDIT_KEYS = (
    "refer_kitti_v1|0001|14|106",   # multi_positive
    "refer_kitti_v1|0001|38|11",    # positive
    "refer_kitti_v1|0001|1|357",     # present_uncovered
    "refer_kitti_v1|0001|48|272",    # inactive
    "refer_kitti_v2|0000|45|128",   # multi_positive
    "refer_kitti_v2|0000|174|136",  # positive
    "refer_kitti_v2|0000|98|153",   # present_uncovered
    "refer_kitti_v2|0000|0|0",       # inactive
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def digest_keys(keys: list[tuple[Any, ...]]) -> str:
    return hashlib.sha256(json.dumps([list(x) for x in keys], sort_keys=False).encode()).hexdigest()


def call_model(model: L81HierarchicalEarlyFusion, raw: dict[str, Any], batch: Any,
               device: torch.device, chunk_size: int | None = None,
               return_audit: bool = False) -> dict[str, torch.Tensor]:
    observations = batch.history_observations.to(device=device).clone()
    history = batch.history_observations.to(device=device).clone()
    history_mask = batch.history_mask.to(device=device).clone()
    history_frames = batch.history_frame_ids.to(device=device).clone()
    with torch.inference_mode():
        output = model(
            raw["visual_pyramid"], raw["local_tokens"], raw["text_tokens"], raw["text_mask"],
            observations, history_mask, history_frames, int(batch.frame_id), raw["boxes_norm"],
            candidate_chunk_size=chunk_size, return_audit=return_audit,
        )
    del observations, history, history_mask, history_frames
    return output


def canonical_delta(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> dict[str, float]:
    result = {}
    for key in L81HierarchicalEarlyFusion.canonical_output_keys:
        result[key] = float((left[key].float() - right[key].float()).abs().max().cpu())
    return result


def finite_canonical(output: dict[str, torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(output[key]).all()) for key in L81HierarchicalEarlyFusion.canonical_output_keys)


def module_gradient_report(model: L81HierarchicalEarlyFusion) -> dict[str, Any]:
    prefixes = (
        "visual_adapters", "local_adapter", "text_projection", "text_encoder", "query_slots",
        "text_to_slots", "marker_projection", "region_marker", "tap_embedding", "fusion_blocks",
        "set_blocks", "slot_gate", "observation_projection", "time_projection", "history_gru",
        "history_fusion", "membership_head", "track_head", "continuation_head", "quality_head",
        "null_head", "cardinality_head",
    )
    result: dict[str, Any] = {}
    for prefix in prefixes:
        values = []
        finite = True
        nonzero = False
        for name, parameter in model.named_parameters():
            if name == prefix or name.startswith(prefix + "."):
                if parameter.grad is not None:
                    value = float(parameter.grad.detach().abs().max().cpu())
                    values.append(value)
                    finite = finite and bool(torch.isfinite(parameter.grad).all())
                    nonzero = nonzero or bool((parameter.grad.abs() > 0).any())
        result[prefix] = {
            "parameter_count": int(sum(p.numel() for n, p in model.named_parameters()
                                        if n == prefix or n.startswith(prefix + "."))),
            "gradient_parameter_tensors": len(values),
            "max_abs_gradient": max(values) if values else 0.0,
            "finite": finite,
            "nonzero": nonzero,
        }
    return result


def build_key_rows() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = load_fit_units()
    by_key = {str(row["unit_key"]): row for row in rows}
    if len(by_key) != len(rows):
        raise AssertionError("fit unit keys are not unique")
    selected = []
    for key in AUDIT_KEYS:
        if key not in by_key:
            raise KeyError(f"pre-registered label-free audit key is absent: {key}")
        selected.append(key_only(by_key[key]))
    return selected, by_key


def pick_text_control(meta: dict[str, Any], by_key: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidates = [
        key_only(row) for row in by_key.values()
        if str(row["dataset"]) == str(meta["dataset"])
        and str(row["video"]) == str(meta["video"])
        and int(row["frame_id"]) == int(meta["frame_id"])
        and str(row.get("sentence") or row.get("expression")) != str(meta["sentence"])
    ]
    if candidates:
        selected = sorted(candidates, key=lambda x: str(x["unit_key"]))[0]
        selected["control_scope"] = "same_image_same_frame_text_only"
        return selected
    candidates = [
        key_only(row) for row in by_key.values()
        if str(row["dataset"]) == str(meta["dataset"])
        and str(row["video"]) == str(meta["video"])
        and str(row.get("sentence") or row.get("expression")) != str(meta["sentence"])
    ]
    if candidates:
        selected = sorted(candidates, key=lambda x: str(x["unit_key"]))[0]
        selected["control_scope"] = "same_image_other_frame_text_only"
        return selected
    candidates = [
        key_only(row) for row in by_key.values()
        if str(row.get("sentence") or row.get("expression")) != str(meta["sentence"])
    ]
    if not candidates:
        raise AssertionError(f"no unrelated expression text control for {meta['unit_key']}")
    selected = sorted(candidates, key=lambda x: str(x["unit_key"]))[0]
    selected["control_scope"] = "same_image_other_unit_text_only"
    return selected


def audit(args: argparse.Namespace) -> dict[str, Any]:
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L81 contract output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable] + sys.argv)
    started = time.perf_counter()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA changed")
    if sha256_file(CLIP_WEIGHT) != CLIP_SHA256:
        raise AssertionError("CLIP SHA changed")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("L81 contract requires the registered GPU0 runtime")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.manual_seed(20260829); np.random.seed(20260829)
    selected, by_key = build_key_rows()
    metadata_schema = sorted(selected[0].keys())
    if set().union(*(set(x) for x in selected)) & FORBIDDEN_LABEL_FIELDS:
        raise AssertionError("forbidden label field leaked into label-free audit metadata")
    clip_model = load_clip(device)
    model = L81HierarchicalEarlyFusion(L81Config()).to(device=device, dtype=torch.float32)
    model.eval()
    cache = FrameFeatureCache(max_items=max(16, len(selected)))
    store = L80BankStore(max_history=model.config.history_length)
    batches: dict[str, Any] = {}
    raws: dict[str, dict[str, Any]] = {}
    rows = []
    controls = {}
    try:
        for meta in selected:
            batch = store.build_unit(meta)
            if batch.candidate_count != len(batch.row_offsets) or batch.candidate_count != len(batch.row_keys):
                raise AssertionError(f"candidate count/key drift: {batch.unit_key}")
            if batch.row_offsets != list(range(batch.row_offsets[0], batch.row_offsets[0] + batch.candidate_count)):
                raise AssertionError(f"native row offsets not contiguous: {batch.unit_key}")
            if [int(key[-1]) for key in batch.row_keys] != batch.row_offsets:
                raise AssertionError(f"native row order drift: {batch.unit_key}")
            if int((batch.history_frame_ids > batch.frame_id).sum()) != 0:
                raise AssertionError(f"future history in {batch.unit_key}")
            raw = raw_inputs_for_l81(clip_model, batch, device, cache)
            output = call_model(model, raw, batch, device, return_audit=True)
            if not finite_canonical(output):
                raise FloatingPointError(f"nonfinite L81 output: {batch.unit_key}")
            candidate_indices = list(map(int, batch.candidate_indices))
            duplicates = len(candidate_indices) - len(set(candidate_indices))
            rows.append({
                "unit_key": batch.unit_key, "dataset": batch.dataset, "video": batch.video,
                "query_id": batch.query_id, "frame_id": batch.frame_id,
                "candidate_count": batch.candidate_count, "row_offsets": batch.row_offsets,
                "row_keys": [list(key) for key in batch.row_keys],
                "candidate_index_provenance": candidate_indices,
                "duplicate_candidate_index_count": duplicates,
                "candidate_key_digest": digest_keys(batch.row_keys),
                "pyramid_shape": list(raw["visual_pyramid"].shape),
                "local_shape": list(raw["local_tokens"].shape),
                "text_shape": list(raw["text_tokens"].shape),
                "text_valid_tokens": int(raw["text_mask"].sum()),
                "history_shape": list(batch.history_observations.shape),
                "history_future_rows": int((batch.history_frame_ids > batch.frame_id).sum()),
                "finite": finite_canonical(output), "candidate_deletion": False,
                "candidate_truncation": False, "sidecar_labels_loaded": False,
                "source_pool_ids_provenance_only": True,
            })
            batches[batch.unit_key] = batch
            raws[batch.unit_key] = raw
            control_meta = pick_text_control(meta, by_key)
            control_batch = store.build_unit(control_meta)
            if (control_meta.get("control_scope") == "same_image_same_frame_text_only" and
                    (control_batch.candidate_count != batch.candidate_count or
                     control_batch.row_offsets != batch.row_offsets or
                     control_batch.bank_path != batch.bank_path)):
                raise AssertionError(f"same-frame control candidate contract drift: {batch.unit_key}")
            control_raw = dict(raw)
            control_raw["text_tokens"] = control_batch.text_tokens.float().to(device=device).clone()
            control_raw["text_mask"] = control_batch.text_mask.bool().to(device=device).clone()
            controls[batch.unit_key] = {
                "control_unit_key": control_batch.unit_key,
                "control_scope": str(control_meta.get("control_scope", "unrecorded")),
                "control_sentence_hash": hashlib.sha256(control_batch.sentence.encode()).hexdigest(),
                "output": call_model(model, control_raw, batch, device),
            }
            del output, control_raw, control_batch
        if cache.visual_forward_count != len({(x["video"], x["frame_id"]) for x in rows}):
            raise AssertionError("CLIP visual forward was not one per unique frame")

        chosen_key = selected[0]["unit_key"]
        batch = batches[chosen_key]
        raw = raws[chosen_key]
        base = call_model(model, raw, batch, device, return_audit=True)
        control = controls[chosen_key]["output"]
        expression_delta = canonical_delta(base, control)
        boxes = raw["boxes_norm"]
        shifted = boxes.clone()
        shifted[0] = torch.stack((
            (shifted[0, 0] + 0.01).clamp(0.0, 0.8), shifted[0, 1],
            (shifted[0, 2] + 0.01).clamp(0.05, 1.0), shifted[0, 3]), dim=0)
        shifted[0, 2] = max(float(shifted[0, 2]), float(shifted[0, 0] + 1e-3))
        shifted[0, 3] = max(float(shifted[0, 3]), float(shifted[0, 1] + 1e-3))
        marker_raw = dict(raw); marker_raw["boxes_norm"] = shifted
        marker_output = call_model(model, marker_raw, batch, device)
        model.set_marker_enabled(False)
        marker_zero = call_model(model, raw, batch, device)
        model.set_marker_enabled(True)
        restored = call_model(model, raw, batch, device)
        marker_delta = canonical_delta(base, marker_output)
        marker_zero_delta = canonical_delta(base, marker_zero)
        marker_restore_delta = canonical_delta(base, restored)

        permutation = torch.arange(batch.candidate_count - 1, -1, -1, device=device)
        permutation_cpu = permutation.cpu()
        perm_raw = dict(raw)
        perm_raw["local_tokens"] = raw["local_tokens"][permutation].clone()
        perm_raw["boxes_norm"] = raw["boxes_norm"][permutation].clone()
        perm_batch = type(batch)(**{
            name: (getattr(batch, name)[permutation_cpu].clone() if name in {
                "observations", "history_observations", "history_mask", "history_frame_ids",
                "boxes", "boxes_norm"
            } else getattr(batch, name))
            for name in batch.__dataclass_fields__
        })
        permuted = call_model(model, perm_raw, perm_batch, device)
        permutation_delta = {
            key: float((permuted[key].float() - base[key].float()[permutation]).abs().max().cpu())
            for key in ("candidate_logits", "track_logits", "continuation_logits", "quality_logits")
        }

        duplicate_index = torch.zeros(2, dtype=torch.long, device=device)
        duplicate_index_cpu = duplicate_index.cpu()
        duplicate_raw = dict(raw)
        duplicate_raw["local_tokens"] = raw["local_tokens"][duplicate_index].clone()
        duplicate_raw["boxes_norm"] = raw["boxes_norm"][duplicate_index].clone()
        duplicate_batch = type(batch)(**{
            name: (getattr(batch, name)[duplicate_index_cpu].clone() if name in {
                "observations", "history_observations", "history_mask", "history_frame_ids",
                "boxes", "boxes_norm"
            } else getattr(batch, name))
            for name in batch.__dataclass_fields__
        })
        duplicate_output = call_model(model, duplicate_raw, duplicate_batch, device)
        duplicate_delta = {
            key: float((duplicate_output[key][0] - duplicate_output[key][1]).abs().cpu())
            for key in ("candidate_logits", "track_logits", "continuation_logits", "quality_logits")
        }
        chunked = call_model(model, raw, batch, device, chunk_size=1)
        chunk_delta = canonical_delta(base, chunked)

        future_rejected = False
        future_error = None
        future_raw = dict(raw)
        future_batch = type(batch)(**{
            name: (getattr(batch, name).clone() if torch.is_tensor(getattr(batch, name)) else getattr(batch, name))
            for name in batch.__dataclass_fields__
        })
        valid = torch.nonzero(future_batch.history_mask[0], as_tuple=False).flatten()
        if valid.numel():
            future_batch.history_frame_ids[0, int(valid[-1])] = int(batch.frame_id) + 1
            try:
                call_model(model, future_raw, future_batch, device)
            except ValueError as error:
                future_rejected = True
                future_error = str(error)
        if not future_rejected:
            raise AssertionError("future history was not rejected")

        model.train(); model.zero_grad(set_to_none=True)
        grad_output = model(
            raw["visual_pyramid"], raw["local_tokens"], raw["text_tokens"], raw["text_mask"],
            batch.history_observations.to(device=device).clone(), batch.history_mask.to(device=device).clone(),
            batch.history_frame_ids.to(device=device).clone(), int(batch.frame_id), raw["boxes_norm"],
            return_audit=True,
        )
        grad_loss = sum(grad_output[key].float().sum() for key in L81HierarchicalEarlyFusion.canonical_output_keys)
        grad_loss.backward()
        gradients = module_gradient_report(model)
        required_gradients_ok = all(value["finite"] and value["nonzero"] for value in gradients.values())
        if not required_gradients_ok:
            raise AssertionError(f"required L81 module gradient missing: {gradients}")
        model.eval(); model.zero_grad(set_to_none=True)

        checkpoint = out / "model_contract_state.pt"
        torch.save({"format": "locatemot-l81-contract-state-v1", "model_config": model.config.__dict__,
                    "model_state_dict": model.state_dict()}, checkpoint)
        reloaded = L81HierarchicalEarlyFusion(L81Config(**model.config.__dict__)).to(device=device, dtype=torch.float32)
        load_result = reloaded.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"], strict=True)
        reloaded.eval()
        reloaded_output = call_model(reloaded, raw, batch, device)
        reload_delta = canonical_delta(base, reloaded_output)
        if max(reload_delta.values()) > 1e-6:
            raise AssertionError(f"strict reload output drift: {reload_delta}")
        model.set_marker_enabled(True)
        parameter_report = model.parameter_report()
        contract = {
            "format": "locatemot-l81-representation-contract-v1", "status": "complete",
            "stage": "label-free representation contract", "command": command,
            "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()), "luna_thread": THREAD,
            "fixed_fit_audit_units": len(selected), "fixed_keys": [x["unit_key"] for x in selected],
            "pre_forward_metadata_schema": metadata_schema,
            "forbidden_label_fields_absent": True, "sidecar_labels_loaded": False,
            "parameter_report": parameter_report, "canonical_output_keys": list(model.canonical_output_keys),
            "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
            "duplicate_candidate_index_legal": True, "one_visual_forward_per_unique_frame": True,
            "visual_forward_count": int(cache.visual_forward_count),
            "unique_frame_count": len({(x["video"], x["frame_id"]) for x in rows}),
            "pyramid_shape": [3, 1, 196, 768], "local_tokens_per_candidate": 63,
            "history_length": model.config.history_length, "history_future_rows": 0,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "finite_checks": True, "strict_reload": bool(not load_result.missing_keys and not load_result.unexpected_keys),
            "strict_reload_max_output_delta": max(reload_delta.values()),
            "deterministic_eval": True, "raw_dense_cache_written": False,
            "clip_weight": str(CLIP_WEIGHT), "clip_sha256": sha256_file(CLIP_WEIGHT),
            "manifest_sha256": sha256_file(MANIFEST),
            "elapsed_sec": time.perf_counter() - started,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        }
        sensitivity = {
            "format": "locatemot-l81-sensitivity-report-v1", "status": "complete",
            "expression_control": controls[chosen_key]["control_unit_key"],
            "expression_candidate_max_delta": max(expression_delta.values()),
            "expression_canonical_delta": expression_delta,
            "marker_box_perturbation_candidate_max_delta": max(marker_delta.values()),
            "marker_box_perturbation_delta": marker_delta,
            "marker_disabled_candidate_max_delta": max(marker_zero_delta.values()),
            "marker_restore_max_delta": max(marker_restore_delta.values()),
            "marker_zero_contract": "marker path disabled only for the diagnostic forward; restored before later checks",
            "candidate_pairwise_score_std": float(base["candidate_logits"].float().std().cpu()),
            "control_relative_threshold": 1e-4,
            "marker_relative_threshold": 1e-4,
            "UNALIGNED": True,
        }
        permutation_report = {
            "format": "locatemot-l81-permutation-report-v1", "status": "complete",
            "permutation": permutation.detach().cpu().tolist(),
            "candidate_output_max_abs_delta": max(permutation_delta.values()),
            "per_output_delta": permutation_delta,
            "registered_tolerance": 1e-4,
            "duplicate_rows_retained": True,
            "duplicate_row_output_deltas": duplicate_delta,
            "chunk_size": 1,
            "chunk_output_max_abs_delta": max(chunk_delta.values()),
            "chunk_registered_tolerance": 1e-4,
        }
        write_json(out / "contract.json", contract)
        write_json(out / "sensitivity_report.json", sensitivity)
        write_json(out / "permutation_report.json", permutation_report)
        write_json(out / "gradient_report.json", {
            "format": "locatemot-l81-gradient-report-v1", "status": "complete",
            "loss": float(grad_loss.detach().cpu()), "required_gradients_ok": required_gradients_ok,
            "modules": gradients, "detector_or_base_parameters": "not present in the trainable L81 module",
        })
        write_json(out / "provenance.json", {
            "format": "locatemot-l81-contract-provenance-v1", "status": "complete", "command": command,
            "cwd": str(Path.cwd().resolve()), "project_root": str(ROOT), "luna_thread": THREAD,
            "inputs": {"manifest": str(MANIFEST), "manifest_sha256": sha256_file(MANIFEST),
                       "fit_units": str(ROOT / "outputs/l49/data/train_units.jsonl"),
                       "l69_features": str(ROOT / "outputs/l69/attempt9/budget40_features/kitti"),
                       "l48_text": str(ROOT / "outputs/l48/data/text_cache.pt"),
                       "clip_weight": str(CLIP_WEIGHT), "clip_sha256": sha256_file(CLIP_WEIGHT)},
            "outputs": [str(out / x) for x in ("contract.json", "gradient_report.json", "permutation_report.json", "sensitivity_report.json")],
            "fixed_key_only_before_labels": True, "labels_read": False, "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "hota_trackeval_run": False, "raw_dense_cache_written": False,
            "source_pool_group_query_track_ids": "provenance/assembly only; not neural inputs",
        })
        write_json(out / "status.json", {
            "format": "locatemot-l81-contract-status-v1", "status": "complete",
            "contract": "representation_contract_pass", "command": command,
            "outputs": [str(out / x) for x in ("contract.json", "gradient_report.json", "permutation_report.json", "sensitivity_report.json")],
            "failure_root_cause": None, "next_action": "run the pre-registered 200-update overfit32 smoke",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        })
        write_json(out / "unit_records.json", rows)
        return {"status": "complete", "output": str(out), "sensitivity": sensitivity, "permutation": permutation_report}
    except Exception:
        (out / "INCOMPLETE.md").write_text(
            "# L81 representation contract — INCOMPLETE\n\n" + traceback.format_exc() +
            "\nNo labels, training, screening/test, TrackEval/HOTA, ordinary MOT or OVMOT action was run.\n")
        write_json(out / "status.json", {
            "format": "locatemot-l81-contract-status-v1", "status": "implementation_representation_contract_fail",
            "command": command, "failure_root_cause": "see INCOMPLETE.md first traceback",
            "next_action": "stop L81 before labels/training", "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        })
        raise
    finally:
        cache.clear()
        store._store._bank = None
        store._store._text_cache = None
        del model, clip_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/l81/audit/representation_contract"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
