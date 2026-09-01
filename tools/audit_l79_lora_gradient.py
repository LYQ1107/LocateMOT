#!/usr/bin/env python3
"""One-unit check that the private L79 CLIP LoRA path is actually differentiable."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
sys.path.insert(0, str(ROOT))

from locatemot.models.l79_hierarchical_correspondence import L79HierarchicalCorrespondence  # noqa: E402
from locatemot.rmot.l79_data import L79BankStore, key_only_unit, load_fit_units, MANIFEST, sha256_file  # noqa: E402
from locatemot.rmot.l79_runtime import CLIP_SHA256, CLIP_WEIGHT, load_clip_visual, lora_parameters, preprocess_full_frame, set_lora_enabled, visual_pyramid  # noqa: E402
from locatemot.rmot.l79_train_utils import compute_l79_loss  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command = " ".join([sys.executable] + sys.argv)
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd {Path.cwd()}")
        torch.manual_seed(20260829); np.random.seed(20260829)
        device = torch.device(f"cuda:{args.gpu}"); torch.cuda.set_device(args.gpu)
        rows = load_fit_units()
        unit = sorted([x for x in rows if str(x["category"]) == "positive" and str(x["dataset"]) == "refer_kitti_v2"],
                      key=lambda x: (str(x["video"]), int(x["frame_id"]), int(x["query_id"]), str(x["unit_key"])))[0]
        clip_model = load_clip_visual(device, enable_lora=False)
        set_lora_enabled(clip_model, True)
        model = L79HierarchicalCorrespondence().to(device=device, dtype=torch.float32)
        store = L79BankStore(max_history=16)
        batch = store.build_unit(key_only_unit(unit))
        image = preprocess_full_frame(batch.image_path, device, clip_model.visual.conv1.weight.dtype)
        pyramid = visual_pyramid(clip_model, image, with_grad=True)
        labels = store.attach_labels(batch, unit)
        outputs = model(batch.observations.cuda(), batch.history_observations.cuda(), batch.history_mask.cuda(),
                        batch.text_tokens.cuda(), batch.text_mask.cuda(), batch.boxes_norm.cuda(), pyramid)
        loss, details = compute_l79_loss(outputs, labels)
        loss.backward()
        lora = list(lora_parameters(clip_model))
        lora_nonzero = sum(int(p.grad is not None and bool((p.grad.float().abs() > 0).any())) for p in lora)
        lora_finite = all(p.grad is None or bool(torch.isfinite(p.grad.float()).all()) for p in lora)
        base_nonzero = 0
        for name, parameter in clip_model.named_parameters():
            if ".lora_A" not in name and ".lora_B" not in name and parameter.grad is not None:
                base_nonzero += int(bool((parameter.grad.float().abs() > 0).any()))
        if not bool(torch.isfinite(loss).all()) or lora_nonzero == 0 or not lora_finite or base_nonzero != 0:
            raise AssertionError(f"LoRA gradient contract failed: lora={lora_nonzero}, base={base_nonzero}, finite={lora_finite}")
        result = {"format": "locatemot-l79-lora-gradient-contract-v1", "status": "complete", "stage": "L79-pre-P2",
                  "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()), "command": command,
                  "unit_key": batch.unit_key, "dataset": batch.dataset, "category": labels["category"],
                  "candidate_count": batch.candidate_count, "loss": float(loss.detach().cpu()), "loss_details": details,
                  "lora_blocks": [8, 9, 10, 11], "rank": 32, "alpha": 16.0, "lora_dtype": "float32",
                  "lora_enabled": True, "lora_parameter_tensors": len(lora), "lora_nonzero_gradient_tensors": lora_nonzero,
                  "lora_gradients_finite": lora_finite, "base_nonzero_gradient_tensors": base_nonzero,
                  "base_requires_grad_false": all(not p.requires_grad for name, p in clip_model.named_parameters() if ".lora_A" not in name and ".lora_B" not in name),
                  "candidate_deletion": False, "candidate_truncation": False, "history_future_rows": int((batch.history_frame_ids > batch.frame_id).sum()),
                  "manifest_sha256": sha256_file(MANIFEST), "clip_sha256": sha256_file(CLIP_WEIGHT), "expected_clip_sha256": CLIP_SHA256,
                  "no_persistent_raw_dense_cache": True, "screening_gt_used": False, "official_test_labels_read": False,
                  "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False, "wall_time_seconds": time.perf_counter() - started,
                  "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device))}
        write_json(out / "contract.json", result)
        write_json(out / "status.json", {"format": "locatemot-l79-lora-gradient-status-v1", "status": "complete", "failure_root_cause": None,
                                          "next_action": "run registered P2 fit schedule"})
        print(json.dumps({"status": "complete", "out": str(out)}, indent=2), flush=True)
        return 0
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        write_json(out / "status.json", {"format": "locatemot-l79-lora-gradient-status-v1", "status": "incomplete", "command": command,
                                          "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback": tb,
                                          "next_action": "preserve attempt; fix only the first actionable LoRA runtime error"})
        (out / "INCOMPLETE.md").write_text("# L79 LoRA gradient contract incomplete\n\n```text\n" + tb + "```\n")
        print(tb, file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
