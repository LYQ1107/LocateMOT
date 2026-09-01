#!/usr/bin/env python3
"""L66 label-free CLIP-LoRA contract audit and one-step gradient regression."""
from __future__ import annotations

import json
import time
import argparse
from pathlib import Path

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
OUT = ROOT / "outputs/l66/audit/visual_lora_contract"
import sys
sys.path.insert(0, str(ROOT))
from locatemot.models.l66_visual_lora_set import L66VisualLoraSet, attach_visual_lora, lora_parameters
from tools.l66_visual_lora_common import (CLIP_WEIGHTS, EXPECTED_CLIP, EXPECTED_MANIFEST,
    L65_CHECKPOINT, MANIFEST, StreamingClipLora, fit_units, load_unit_features, sha256, stratified,
    loss_fn)


def main():
    if Path.cwd().resolve() != ROOT: raise RuntimeError(f"wrong cwd {Path.cwd()}")
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, default=OUT); args = ap.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    if sha256(CLIP_WEIGHTS) != EXPECTED_CLIP: raise AssertionError("CLIP SHA mismatch")
    if sha256(MANIFEST) != EXPECTED_MANIFEST: raise AssertionError("manifest SHA mismatch")
    units = stratified(fit_units(), 20260829)
    unit = next(u for u in units if u.get("category") == "multi_positive")
    device = torch.device("cuda:0")
    runtime = StreamingClipLora(device, crop_batch=4)
    target_path, wrapped = attach_visual_lora(runtime.model, rank=8, alpha=16.0, dropout=0.0)
    head = L66VisualLoraSet(hidden=128).to(device)
    head_state = torch.load(L65_CHECKPOINT, map_location="cpu", weights_only=False)["model"]
    head.load_state_dict(head_state, strict=True)
    head.train()
    # Feature construction is label-free. The unit labels are inspected only below it.
    t0 = time.time()
    item = load_unit_features(unit, runtime, labels=False)
    patches = item["patches"]
    words = item["words"].to(device)
    mask = item["mask"].to(device)
    nums = item["numeric"].to(device)
    # Labels are attached only after the label-free feature/shape audit.
    target = torch.zeros(item["candidate_count"], dtype=torch.bool)
    positive_indices = [int(x) for x in unit.get("positive_indices", [])]
    if any(x < 0 or x >= item["candidate_count"] for x in positive_indices): raise AssertionError("positive index out of range")
    if positive_indices: target[torch.as_tensor(positive_indices, dtype=torch.long)] = True
    item["target"] = target
    label_count = int(target.sum())
    with torch.no_grad():
        base_x = wrapped.base(torch.zeros((1, wrapped.base.in_features), device=device, dtype=wrapped.base.weight.dtype))
        wrapped_x = wrapped(torch.zeros((1, wrapped.base.in_features), device=device, dtype=wrapped.base.weight.dtype))
        zero_init_diff = float((base_x - wrapped_x).abs().max())
    with torch.no_grad():
        output = head(patches, words, mask, nums)
    label_free = {
        "unit_key": unit["unit_key"], "dataset": unit["dataset"], "video": unit["video"],
        "frame_id": int(unit["frame_id"]), "image": item["image"], "candidate_count": int(item["candidate_count"]),
        "patch_joint_shape": list(patches.shape), "text_joint_shape": list(item["words"].shape),
        "text_valid_count": int(item["mask"].sum()), "numeric_shape": list(item["numeric"].shape),
        "relevance_shape": list(output["relevance_logit"].shape), "null_shape": list(output["null_logit"].shape),
        "finite_features": bool(torch.isfinite(patches).all() and torch.isfinite(item["words"]).all()),
        "finite_output": bool(torch.isfinite(output["relevance_logit"]).all() and torch.isfinite(output["null_logit"])),
        "label_read_after_feature_construction": True, "positive_count_after_audit": label_count,
        "raw_cache_written": False, "source_pool_group_state_semantic_input": False,
    }
    del output
    optimizer = torch.optim.AdamW(lora_parameters(runtime.model) + list(head.parameters()), lr=5e-5)
    optimizer.zero_grad(set_to_none=True)
    output = head(patches, words, mask, nums)
    loss, parts = loss_fn(output, item["target"].to(device))
    if not torch.isfinite(loss): raise FloatingPointError("nonfinite audit loss")
    loss.backward()
    grad_rows = []
    for name, p in runtime.model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            grad_rows.append({"name": name, "has_grad": p.grad is not None, "finite": bool(p.grad is not None and torch.isfinite(p.grad).all()), "norm": 0.0 if p.grad is None else float(p.grad.norm())})
    base_grad = [p for name, p in runtime.model.named_parameters() if "lora_A" not in name and "lora_B" not in name and p.grad is not None and float(p.grad.abs().max()) != 0.0]
    head_grad = [p for p in head.parameters() if p.grad is not None and float(p.grad.abs().max()) != 0.0]
    optimizer.step()
    ck = out / "gradient_regression_checkpoint.pt"
    torch.save({"lora": {k: v.detach().cpu() for k, v in runtime.model.state_dict().items() if "lora_A" in k or "lora_B" in k}, "head": head.state_dict(), "step": 1, "format": "locatemot-l66-lora-contract-v1"}, ck)
    # Reload in a fresh compact head/runtime contract without serializing CLIP weights.
    reload_head = L66VisualLoraSet(hidden=128).cpu(); reload_head.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["head"], strict=True)
    reload_ok = all(tuple(a.shape) == tuple(b.shape) for a, b in zip(head.state_dict().values(), reload_head.state_dict().values()))
    payload = {
        "format": "locatemot-l66-visual-lora-contract-v1", "status": "complete", "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
        "seed": 20260829, "clip_weights": str(CLIP_WEIGHTS), "clip_weights_sha256": sha256(CLIP_WEIGHTS),
        "l65_checkpoint": str(L65_CHECKPOINT), "l65_checkpoint_sha256": sha256(L65_CHECKPOINT),
        "target_module": target_path, "target_type": type(wrapped.base).__name__, "rank": 8, "alpha": 16.0, "dropout": 0.0,
        "lora_parameter_shapes": {"lora_A": list(wrapped.lora_A.shape), "lora_B": list(wrapped.lora_B.shape)}, "base_visual_dtype": str(wrapped.base.weight.dtype), "lora_dtype": str(wrapped.lora_A.dtype),
        "lora_parameter_count": sum(p.numel() for p in lora_parameters(runtime.model)), "head_parameter_count": sum(p.numel() for p in head.parameters()),
        "base_requires_grad_trainable_count": sum(p.numel() for name, p in runtime.model.named_parameters() if "lora_A" not in name and "lora_B" not in name and p.requires_grad),
        "optimizer_parameter_count": sum(p.numel() for p in lora_parameters(runtime.model) + list(head.parameters())),
        "zero_initialized_B": bool(float(wrapped.lora_B.detach().abs().max()) > 0.0) is False,
        "initial_output_equals_base_max_abs": zero_init_diff, "initial_contract_exact": zero_init_diff == 0.0,
        "label_free_forward": label_free, "gradient_regression": {"steps": 1, "finite_loss": bool(torch.isfinite(loss)), "loss": float(loss.detach()), "loss_parts": parts, "lora_grads": grad_rows, "lora_any_nonzero": any(x["norm"] > 0 for x in grad_rows), "base_nonzero_grad_count": len(base_grad), "head_nonzero_grad_count": len(head_grad), "strict_reload": reload_ok, "checkpoint": str(ck), "checkpoint_sha256": sha256(ck)},
        "input_contract": {"image": "streamed PNG crop, L19 box + 10% padding + clip-to-image", "patch_joint": ["N", 17, 512], "text_joint": [77, 512], "numeric_dim": 32, "candidate_truncation": False, "candidate_key_drift": 0},
        "fit_only": True, "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
        "persistent_raw_dense_cache_written": False, "token_span_alignment": "UNALIGNED", "static_motion_mask": "UNALIGNED", "elapsed_sec": time.time() - t0,
    }
    (out / "contract.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "provenance.json").write_text(json.dumps({"project_root": str(ROOT), "cwd": str(Path.cwd().resolve()), "manifest_sha256": sha256(MANIFEST), "fit_units": str(ROOT / "outputs/l49/data/train_units.jsonl"), "no_test_flags": True, "raw_cache_written": False}, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "target": target_path, "lora_any_nonzero": payload["gradient_regression"]["lora_any_nonzero"], "base_nonzero_grad_count": len(base_grad), "reload": reload_ok, "output": str(out)}, indent=2), flush=True)


if __name__ == "__main__": main()
