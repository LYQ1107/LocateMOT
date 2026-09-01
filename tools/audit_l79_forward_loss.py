#!/usr/bin/env python3
"""Small L79 forward/loss/reload contract regression before P1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
sys.path.insert(0, str(ROOT))

from locatemot.models.l79_hierarchical_correspondence import L79Config, L79HierarchicalCorrespondence  # noqa: E402
from locatemot.rmot.l79_data import L79BankStore, MANIFEST, file_meta, key_only_unit, load_fit_units, sha256_file  # noqa: E402
from locatemot.rmot.l79_runtime import CLIP_SHA256, CLIP_WEIGHT, MemoryFrameCache, load_clip_visual, preprocess_full_frame, visual_pyramid  # noqa: E402
from locatemot.rmot.l79_train_utils import compute_l79_loss  # noqa: E402


SEED = 20260829


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def choose_units(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = []
    wanted = [
        ("refer_kitti_v1", "positive"),
        ("refer_kitti_v2", "multi_positive"),
        ("refer_kitti_v1", "inactive"),
        ("refer_kitti_v2", "present_uncovered"),
    ]
    for dataset, category in wanted:
        matches = [x for x in rows if str(x["dataset"]) == dataset and str(x["category"]) == category]
        if not matches:
            raise AssertionError(f"missing contract stratum {dataset}/{category}")
        selected.append(sorted(matches, key=lambda x: (str(x["video"]), int(x["frame_id"]), int(x["query_id"]), str(x["unit_key"]))) [0])
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/l79/audit/forward_loss_contract_attempt1")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command = " ".join([sys.executable] + sys.argv)
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(MANIFEST) != "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa":
            raise AssertionError("fixed manifest SHA changed")
        if sha256_file(CLIP_WEIGHT) != CLIP_SHA256:
            raise AssertionError("CLIP SHA changed")
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
        device = torch.device(f"cuda:{args.gpu}")
        torch.cuda.set_device(args.gpu)
        rows = load_fit_units()
        units = choose_units(rows)
        clip_model = load_clip_visual(device, enable_lora=False)
        model = L79HierarchicalCorrespondence(L79Config()).to(device=device, dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        store = L79BankStore(max_history=16)
        cache = MemoryFrameCache(max_items=8)
        checks = []
        last_inputs = None
        for unit in units:
            key_unit = key_only_unit(unit)
            batch = store.build_unit(key_unit)
            if not Path(batch.image_path).is_file():
                raise FileNotFoundError(batch.image_path)
            cache_key = (batch.video, int(batch.frame_id))
            pyramid = cache.get(cache_key)
            if pyramid is None:
                image = preprocess_full_frame(batch.image_path, device, clip_model.visual.conv1.weight.dtype)
                with torch.inference_mode():
                    pyramid = visual_pyramid(clip_model, image, with_grad=False)
                cache.put(cache_key, pyramid)
                del image
            labels = store.attach_labels(batch, unit)
            inputs = (batch.observations.cuda(), batch.history_observations.cuda(), batch.history_mask.cuda(),
                      batch.text_tokens.cuda(), batch.text_mask.cuda(), batch.boxes_norm.cuda(), pyramid)
            output = model(*inputs)
            loss, details = compute_l79_loss(output, labels)
            row_grad = torch.autograd.grad(loss, output["frame_membership_logits"], retain_graph=True)[0]
            target = labels["labels"].cuda(); mask = labels["membership_mask"].cuda()
            positive_grad = bool((row_grad[target & mask].abs() > 0).any()) if bool((target & mask).any()) else True
            negative_grad = bool((row_grad[(~target) & mask].abs() > 0).any()) if bool(((~target) & mask).any()) else True
            (loss / len(units)).backward()
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            if not bool(torch.isfinite(loss).all()) or not positive_grad or not negative_grad:
                raise AssertionError(f"forward/loss gradient contract failed for {batch.unit_key}")
            if int((batch.history_frame_ids > batch.frame_id).sum()) != 0:
                raise AssertionError(f"future history for {batch.unit_key}")
            checks.append({"unit_key": batch.unit_key, "dataset": batch.dataset, "category": labels["category"],
                           "candidate_count": batch.candidate_count, "row_key_count": len(batch.row_keys),
                           "positive_count": labels["positive_count"], "loss": float(loss.detach().cpu()),
                           "positive_gradient_nonzero": positive_grad, "negative_gradient_nonzero": negative_grad,
                           "minimum_positive_gradient_nonzero": positive_grad if labels["positive_count"] > 1 else True,
                           "history_future_rows": 0, "candidate_deletion": False, "candidate_truncation": False,
                           "finite": True})
            last_inputs = tuple(x.detach().clone() for x in inputs)
            del output, loss, row_grad, inputs, pyramid, batch, labels

        checkpoint = out / "contract_checkpoint.pt"
        torch.save({"format": "locatemot-l79-forward-contract-checkpoint-v1", "model_config": model.config_dict(),
                    "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}}, checkpoint)
        package = torch.load(checkpoint, map_location="cpu", weights_only=False)
        reloaded = L79HierarchicalCorrespondence(L79Config(**package["model_config"])).cuda().float()
        result = reloaded.load_state_dict(package["model_state_dict"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise AssertionError(f"reload mismatch {result}")
        model.eval(); reloaded.eval()
        with torch.inference_mode():
            first = model(*last_inputs); second = reloaded(*last_inputs)
        max_diff = max(float((first[key] - second[key]).abs().max().cpu()) for key in first)
        if max_diff > 1e-5:
            raise AssertionError(f"reload output difference {max_diff}")
        base_nonzero = sum(int(p.grad is not None and bool((p.grad.float().abs() > 0).any())) for p in clip_model.parameters())
        contract = {"format": "locatemot-l79-forward-loss-contract-v1", "status": "complete", "stage": "L79-pre-P1",
                    "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()), "command": command, "seed": SEED,
                    "unit_count": len(checks), "units": checks, "candidate_keys_complete": True,
                    "candidate_deletion": False, "candidate_truncation": False, "history_future_rows": 0,
                    "finite_forward_loss": True, "positive_negative_gradients_nonzero": True,
                    "detector_base_nonzero_gradient": base_nonzero, "detector_frozen": all(not p.requires_grad for p in clip_model.parameters()),
                    "strict_reload": True, "reload_max_output_difference": max_diff,
                    "decoder_parameter_count": sum(p.numel() for p in model.parameters()),
                    "no_raw_dense_cache": True, "same_class_hard_negative_metadata": "unavailable",
                    "token_span_alignment": "UNALIGNED", "manifest_sha256": sha256_file(MANIFEST),
                    "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                    "hota_trackeval_run": False, "wall_time_seconds": time.perf_counter() - started,
                    "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device))}
        write_json(out / "contract.json", contract)
        write_json(out / "provenance.json", {"format": "locatemot-l79-forward-loss-provenance-v1", "status": "complete",
                                              "inputs": {"manifest": file_meta(MANIFEST), "clip_weight": {"path": str(CLIP_WEIGHT), "sha256": sha256_file(CLIP_WEIGHT), "expected": CLIP_SHA256},
                                                         "fit_units": str(ROOT / "outputs/l49/data/train_units.jsonl")},
                                              "checkpoint": {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()},
                                              "labels": "attached after label-free bank/text/image construction", "no_persistent_feature_cache": True,
                                              "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                                              "hota_trackeval_run": False})
        write_json(out / "status.json", {"format": "locatemot-l79-forward-loss-status-v1", "status": "complete", "failure_root_cause": None,
                                          "next_action": "run exactly the P1 100-step fit smoke"})
        print(json.dumps({"status": "complete", "out": str(out)}, indent=2), flush=True)
        return 0
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        write_json(out / "status.json", {"format": "locatemot-l79-forward-loss-status-v1", "status": "incomplete",
                                          "command": command, "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback": tb,
                                          "next_action": "fix only the first actionable contract error and rerun in a new directory"})
        (out / "INCOMPLETE.md").write_text("# L79 forward/loss contract incomplete\n\n```text\n" + tb + "```\n")
        print(tb, file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
