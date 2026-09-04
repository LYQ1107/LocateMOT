#!/usr/bin/env python3
"""Mandatory L88 zero-init parity and differentiable gradient contract smoke."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch


WORK_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L88").resolve()
ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
MANIFEST = ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
L85_CACHE = ASSET_ROOT / "outputs/l85/features/fit_dev_eval_full_attempt2"

if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))
import locatemot.rmot as _rmot_package  # noqa: E402
if str(ASSET_ROOT / "locatemot" / "rmot") not in [str(x) for x in _rmot_package.__path__]:
    _rmot_package.__path__.append(str(ASSET_ROOT / "locatemot" / "rmot"))

from locatemot.models.l88_full_rmot import L88FullRMOT, L86Config  # noqa: E402
from locatemot.rmot.l86_losses import l86_loss  # noqa: E402
from locatemot.rmot.l87a_losses import l87a_loss  # noqa: E402
from locatemot.rmot.l88_clip_data import L88ClipStore  # noqa: E402
from locatemot.rmot.l88_grounding_runtime import (  # noqa: E402
    IMAGE_ROOT, L88GroundingRuntime, file_meta, sha256_file, forward_l88_z1,
)
from locatemot.rmot.l88_lora import adapter_grad_report, inject_lora  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def tensor_max_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise AssertionError(f"parity shape mismatch {tuple(left.shape)} / {tuple(right.shape)}")
    return float((left.float() - right.float()).abs().max())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=WORK_ROOT / "outputs/l88/audit/contract_smoke_attempt1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-tile", type=int, default=4)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L88 contract output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable, *sys.argv])
    started = time.perf_counter()
    runtime = None
    store = None
    model = None
    sidecar = None
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("L88 contract requires CUDA GPU0")
            torch.cuda.set_device(device)
        if not L85_CACHE.is_dir():
            raise FileNotFoundError(L85_CACHE)
        torch.manual_seed(20260829)
        store = L88ClipStore(L85_CACHE, load_cache_into_ram=True)
        if not store.train_keys:
            raise AssertionError("empty L88 fit group list")
        group_key = str(store.train_keys[0])
        frame = store.build_frame(group_key, temporal_enabled=False)
        bank_batch = store.bank_store.build_unit(store.groups[group_key]["queries"][0])
        candidate_count = len(frame.row_offsets)
        if candidate_count <= 0:
            raise AssertionError("contract group has no candidates")
        if int(args.query_tile) < 1:
            raise AssertionError(f"invalid smoke query tile {args.query_tile}")
        runtime = L88GroundingRuntime(device)
        model = runtime.model
        injector = inject_lora(model)
        manifest = injector.manifest()
        base_digest_before = injector.base_parameter_digest()
        image_path = IMAGE_ROOT / str(frame.video) / f"{int(frame.frame_id):06d}.png"
        cache_item = runtime.cache_frame(image_path)
        inputs = frame
        qcount = min(int(args.query_tile), len(frame.query_ids))
        selected_sentences = [str(frame.labels[q]["unit_key"] and frame.group_key) for q in range(qcount)]
        # FrameExample does not retain sentence per query; all queries in a
        # frame group share the native group sentence only when the group has
        # one query.  Use the L85 group metadata to recover the exact text.
        group = store.groups[group_key]
        sentences_by_qid = {int(row["query_id"]): str(row["sentence"]) for row in group["queries"]}
        selected_sentences = [sentences_by_qid[int(value)] for value in frame.query_ids[:qcount]]
        forward = forward_l88_z1(model, cache_item, bank_batch.boxes, selected_sentences, device,
                                 query_tile=qcount, autocast_bf16=False)
        z1 = forward["z1"]
        if z1.shape != (qcount, candidate_count, 256):
            raise AssertionError(f"L88 Z1 contract shape {tuple(z1.shape)}")
        reference_z1 = frame.z1[:qcount].to(device)
        # L85 serializes Z1 in FP16 and L86 promotes it back to FP32 when a
        # frame is loaded.  Report the direct FP32 difference as well as the
        # contract comparison after returning the adapted result to that
        # immutable storage precision; otherwise harmless FP16 rounding is
        # incorrectly treated as an architectural parity failure.
        raw_parity_delta = tensor_max_delta(z1.detach(), reference_z1)
        stored_parity_delta = tensor_max_delta(z1.detach().to(torch.float16), reference_z1)
        parity_delta = stored_parity_delta
        parity = {
            "group_key": group_key, "query_ids": [int(x) for x in frame.query_ids[:qcount]],
            "candidate_count": candidate_count, "query_tile": qcount,
            "adapted_z1_shape": list(z1.shape), "reference_z1_shape": list(reference_z1.shape),
            "raw_fp32_max_abs_delta": raw_parity_delta,
            "serialized_fp16_max_abs_delta": stored_parity_delta,
            "max_abs_delta": parity_delta, "comparison_storage_dtype": "float16 (immutable L85 Z1 cache)",
            "threshold": 1e-3,
            "finite": bool(torch.isfinite(z1.float()).all()), "pass": parity_delta <= 1e-3,
            "base_lora_update_zero": bool(manifest["zero_initialized_B"]),
            "reference_source": "immutable L85 Z1 cache; used only for parity, not training target",
        }
        if not parity["pass"]:
            raise AssertionError(f"L88 zero-init parity failed: {parity_delta}")

        sidecar = L88FullRMOT(L86Config()).to(device=device, dtype=torch.float32)
        sidecar.train()
        # Sidecar supervision is the existing L87-A/L86 contract and is used
        # only for this fit-group gradient smoke.
        labels = frame.labels[:qcount]
        current_obs = frame.current_observation.to(device)
        history_obs = frame.history_observations.to(device)
        history_mask = frame.history_mask.to(device)
        history_frames = frame.history_frame_ids.to(device)
        output = sidecar(
            z1.float(), frame.text_global[:qcount].to(device), frame.frame_global[:qcount].to(device),
            current_obs, history_obs, history_mask, history_frames, frame.frame_id,
            temporal_enabled=False)
        loss, loss_info = l87a_loss(output, labels, current_obs, [], temporal_enabled=False)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite L88 contract loss")
        loss.backward()
        lora_grads = adapter_grad_report(injector)
        sidecar_grad_norm = float(sum(float(parameter.grad.detach().float().norm())
                                      for parameter in sidecar.parameters() if parameter.grad is not None))
        sidecar_grad_entries = int(sum(parameter.grad is not None and bool(torch.isfinite(parameter.grad.float()).all())
                                       for parameter in sidecar.parameters()))
        lora_keys = {id(parameter) for parameter in injector.parameters()}
        unapproved = []
        for name, parameter in model.named_parameters():
            if id(parameter) in lora_keys:
                continue
            if parameter.grad is not None:
                unapproved.append(name)
        base_digest_after = injector.base_parameter_digest()
        if lora_grads["gradient_norm_sum"] <= 0 or sidecar_grad_norm <= 0:
            raise AssertionError(f"nonzero gradient contract failed {lora_grads['gradient_norm_sum']} {sidecar_grad_norm}")
        if unapproved:
            raise AssertionError(f"unapproved GroundingDINO gradients: {unapproved[:5]}")
        if base_digest_before != base_digest_after:
            raise AssertionError("base GroundingDINO parametrized weights changed in contract")

        adapter_state = injector.adapter_state_dict()
        sidecar_state = {key: value.detach().cpu().clone() for key, value in sidecar.state_dict().items()}
        sidecar_reload = L88FullRMOT(L86Config()).to(device=device, dtype=torch.float32)
        loaded_sidecar = sidecar_reload.load_state_dict(sidecar_state, strict=True)
        if loaded_sidecar.missing_keys or loaded_sidecar.unexpected_keys:
            raise AssertionError(f"sidecar strict reload mismatch: {loaded_sidecar}")
        model_info = runtime.model_info
        runtime.close(); runtime = None; model = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        runtime2 = L88GroundingRuntime(device)
        model2 = runtime2.model
        injector2 = inject_lora(model2)
        injector2.load_adapter_state_dict(adapter_state, strict=True)
        reload_manifest = injector2.manifest()
        if reload_manifest["targets"] != manifest["targets"]:
            raise AssertionError("L88 target manifest changed on strict reload")
        runtime2.close(); del runtime2, model2, injector2, sidecar_reload
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        write_json(out / "contract.json", {
            "format": "locatemot-l88-contract-smoke-v1", "status": "complete",
            "evidence_type": "one fit-group zero-init parity plus differentiable Stage-S loss smoke",
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD, "seed": 20260829,
            "runtime": model_info, "lora_manifest": manifest,
            "parity": parity, "loss": loss_info, "loss_value": float(loss.detach()),
            "gradients": {"lora": lora_grads, "sidecar_norm": sidecar_grad_norm,
                          "sidecar_finite_gradient_entries": sidecar_grad_entries,
                          "unapproved_groundingdino_gradients": unapproved},
            "base_parameter_digest_before": base_digest_before,
            "base_parameter_digest_after": base_digest_after,
            "strict_reload": {"sidecar": True, "lora": True, "target_manifest_equal": True},
            "candidate_rows": {"count": candidate_count, "row_offsets": [int(x) for x in frame.row_offsets],
                               "row_keys": [list(x) for x in frame.row_keys], "all_rows_retained": True,
                               "candidate_deletion": False, "candidate_truncation": False},
            "future_history_count": int((frame.history_frame_ids[frame.history_mask] > frame.frame_id).sum()),
            "raw_dense_cache_written": False,
            "inputs": {"l85_cache": str(L85_CACHE), "manifest_sha256": MANIFEST_SHA},
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "wall_seconds": time.perf_counter() - started, "failure_root_cause": None,
            "next_action": "freeze manifest/policy and build the query-independent cache before L88 training",
        })
        write_json(out / "provenance.json", json.loads((out / "contract.json").read_text()))
        write_json(out / "status.json", {"format": "locatemot-l88-contract-status-v1", "status": "complete",
                                         "parity_pass": True, "gradient_pass": True, "strict_reload": True,
                                         "screening_gt_used": False, "official_test_labels_read": False,
                                         "ordinary_mot_ovmot_touched": False, "training_run": False,
                                         "hota_trackeval_run": False})
        return 0
    except Exception:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L88 contract smoke — INCOMPLETE\n\n" + trace)
        write_json(out / "status.json", {"format": "locatemot-l88-contract-status-v1", "status": "incomplete",
                                         "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                         "failure_root_cause": "first traceback in INCOMPLETE.md",
                                         "screening_gt_used": False, "official_test_labels_read": False,
                                         "ordinary_mot_ovmot_touched": False, "training_run": False,
                                         "hota_trackeval_run": False})
        raise
    finally:
        if runtime is not None:
            runtime.close()
        if store is not None:
            store.close()
        del model, sidecar
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
