#!/usr/bin/env python3
"""L51 B0 train-only streaming raw-image/token adapter smoke.

This file intentionally reads only L49 fit units and train-side L19 banks.  It
does not load calibration, validation, screening or official-test records.
Raw crops are encoded inside each step and are never serialized.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
DATA = ROOT / "outputs/l49/data"
BANK_ROOT = ROOT / "outputs/l19/dual_banks_features/kitti"
TEXT_CACHE = ROOT / "outputs/l48/data/text_cache.pt"
L28_ROOT = ROOT / "outputs/l28/track_sequence_bank_final"
L29_CHECKPOINT = ROOT / "outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"
FAST_MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
CLIP_WEIGHTS = Path("/home/lwr/.cache/clip/ViT-B-16.pt")
from locatemot.models.l51_streaming_crop_adapter import L51StreamingCropAdapter
from locatemot.rmot.l49_data import load_bank, sha256_file
from tools.train_l49_kitti_rmot import L29Teacher


def load_units():
    units = [json.loads(x) for x in (DATA / "train_units.jsonl").read_text().splitlines() if x.strip()]
    if not units:
        raise RuntimeError("empty L49 train unit manifest")
    if any(str(x.get("split")) != "fit" for x in units):
        raise AssertionError("L51 B0 received a non-fit unit")
    return units


def choose_units(units, per_bucket=4):
    categories = ("multi_positive", "positive", "present_uncovered", "inactive")
    buckets = defaultdict(list)
    for u in units:
        if str(u.get("category")) in categories:
            buckets[(str(u["dataset"]), str(u["category"]))].append(u)
    chosen = []
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        for category in categories:
            values = sorted(buckets[(dataset, category)],
                            key=lambda x: (str(x["video"]), int(x["frame_id"]), int(x["query_id"])))
            if len(values) < per_bucket:
                raise RuntimeError(f"not enough {dataset}/{category} fit units: {len(values)}")
            # Spread selected units over videos instead of a single center window.
            by_video = defaultdict(list)
            for value in values:
                by_video[str(value["video"])].append(value)
            videos = sorted(by_video)
            selected = []
            for i in range(per_bucket):
                v = videos[i % len(videos)]
                selected.append(by_video[v][i // len(videos)])
            chosen.extend(selected)
    if len(chosen) != 32:
        raise AssertionError(f"expected 32 stratified units, got {len(chosen)}")
    return chosen


def crop_box(box, width, height, padding=0.10):
    x1, y1, x2, y2 = [float(x) for x in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    return (max(0, int(math.floor(x1 - padding * bw))),
            max(0, int(math.floor(y1 - padding * bh))),
            min(width, int(math.ceil(x2 + padding * bw))),
            min(height, int(math.ceil(y2 + padding * bh))))


def decode_crop_count(item):
    """Decode real observation crops without invoking the broken CPU CLIP kernel."""
    count = 0
    for box, frame in zip(item["boxes"].tolist(), item["frames"].tolist()):
        path = ROOT / "data/kitti_tracking_training/image_02" / item["video"] / f"{int(frame):06d}.png"
        with Image.open(path) as image:
            image = image.convert("RGB")
            crop = crop_box(box, image.width, image.height)
            if crop[2] <= crop[0] or crop[3] <= crop[1]:
                raise ValueError(f"invalid crop {path} {box} {crop}")
            image.crop(crop).resize((224, 224))
            count += 1
    return count


def numeric_features(tensors, rows):
    history = tensors["history_clip"][rows].float()
    summary = torch.stack((history.mean(1), history.std(1),
                           history.norm(dim=1) / 100.0,
                           history.abs().mean(1)), 1)
    return torch.cat((tensors["geometry"][rows].float(),
                      tensors["motion"][rows].float(),
                      tensors["lifecycle"][rows].float(),
                      tensors["context"][rows].float(),
                      tensors["objectness"][rows].float().reshape(-1, 1),
                      summary), 1)


def materialize_units(selected, text_payload, teacher):
    """Read each selected bank once, retain only frozen row metadata/tensors."""
    grouped = defaultdict(list)
    for i, unit in enumerate(selected):
        grouped[(str(unit["dataset"]), str(unit["video"]))].append((i, unit))
    result = [None] * len(selected)
    for (dataset, video), entries in sorted(grouped.items()):
        bank = load_bank(dataset, video)
        tensors = bank["tensors"]
        for index, unit in entries:
            begin, end = int(unit["begin"]), int(unit["end"])
            rows = torch.arange(begin, end, dtype=torch.long)
            if len(rows) != int(unit["candidate_count"]):
                raise AssertionError(f"candidate count drift: {unit['unit_key']}")
            text_index = text_payload["sentence_to_index"].get(unit["sentence"])
            if text_index is None:
                raise KeyError(f"missing sentence in frozen text cache: {unit['sentence']}")
            labels = torch.zeros(len(rows), dtype=torch.bool)
            pos = torch.as_tensor(unit.get("positive_indices", []), dtype=torch.long)
            if len(pos):
                if int(pos.max()) >= len(rows) or int(pos.min()) < 0:
                    raise AssertionError(f"positive index drift: {unit['unit_key']}")
                labels[pos] = True
            teacher_score = teacher.score(unit, bank)
            if teacher_score is None or len(teacher_score) != len(rows):
                raise AssertionError(f"missing/misaligned L29 teacher: {unit['unit_key']}")
            item = {
                "unit_key": str(unit["unit_key"]), "dataset": dataset, "video": video,
                "query_id": int(unit["query_id"]), "sentence": str(unit["sentence"]),
                "frame_id": int(unit["frame_id"]), "category": str(unit["category"]),
                "begin": begin, "end": end, "y": labels,
                "boxes": tensors["box"][rows].float().contiguous(),
                "frames": tensors["frame"][rows].long().contiguous(),
                "objectness": tensors["objectness"][rows].float().contiguous(),
                "frozen_clip": tensors["clip"][rows].float().contiguous(),
                "numeric": numeric_features(tensors, rows).contiguous(),
                "teacher": teacher_score.float().contiguous(),
                "image_size": list(bank["metadata"].get("image_size", [])),
                "text_index": int(text_index),
            }
            result[index] = item
        del bank
        gc.collect()
    if any(x is None for x in result):
        raise AssertionError("unit materialization incomplete")
    return result


class StreamingClipPatches:
    def __init__(self, device):
        if not CLIP_WEIGHTS.is_file():
            raise FileNotFoundError(f"frozen CLIP weights unavailable: {CLIP_WEIGHTS}")
        import clip
        self.device = torch.device(device)
        self.model, self.preprocess = clip.load(str(CLIP_WEIGHTS), device=self.device)
        # OpenAI CLIP's half-precision CPU path is not supported reliably by
        # the local PyTorch build; CPU preflight is diagnostic only.
        if self.device.type == "cpu":
            self.model.float()
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, item):
        pixels = []
        for box, frame in zip(item["boxes"].tolist(), item["frames"].tolist()):
            path = ROOT / "data/kitti_tracking_training/image_02" / item["video"] / f"{int(frame):06d}.png"
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                image = image.convert("RGB")
                crop = crop_box(box, image.width, image.height)
                if crop[2] <= crop[0] or crop[3] <= crop[1]:
                    raise ValueError(f"invalid crop {path} {box} {crop}")
                pixels.append(self.preprocess(image.crop(crop)))
        visual = self.model.visual
        result = []
        for start in range(0, len(pixels), 32):
            pixel = torch.stack(pixels[start:start + 32]).to(self.device)
            pixel = pixel.float() if self.device.type == "cpu" else pixel.to(dtype=visual.conv1.weight.dtype)
            x = visual.conv1(pixel).reshape(pixel.shape[0], visual.conv1.out_channels, -1).permute(0, 2, 1)
            cls = visual.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
            x = torch.cat((cls, x), 1) + visual.positional_embedding.to(x.dtype)
            x = visual.ln_pre(x).permute(1, 0, 2)
            x = visual.transformer(x).permute(1, 0, 2)[:, 1:]
            side = int(round(x.shape[1] ** 0.5))
            if side * side != x.shape[1]:
                raise AssertionError(f"unexpected CLIP patch count {x.shape[1]}")
            # Keep spatial tokens while reducing only the smoke memory footprint.
            x = F.adaptive_avg_pool2d(x.transpose(1, 2).reshape(x.shape[0], x.shape[2], side, side), (2, 2))
            result.append(x.flatten(2).transpose(1, 2).float())
            del pixel, x
        return torch.cat(result, 0)


def balanced_bce(logits, target):
    target = target.bool()
    pieces = []
    if target.any():
        pieces.append(F.binary_cross_entropy_with_logits(logits[target], torch.ones_like(logits[target])))
    if (~target).any():
        pieces.append(F.binary_cross_entropy_with_logits(logits[~target], torch.zeros_like(logits[~target])))
    return torch.stack(pieces).mean() if pieces else logits.new_zeros(())


def losses(out, item, device):
    score = out["final_logit"]
    y = item["y"].to(device)
    pos = torch.nonzero(y, as_tuple=False).flatten()
    neg = torch.nonzero(~y, as_tuple=False).flatten()
    zero = score.new_zeros(())
    if len(neg):
        pre = neg[torch.argsort(item["objectness"].to(device)[neg], descending=True)[:min(48, len(neg))]]
        with torch.no_grad():
            hard = pre[torch.argsort(score.detach()[pre], descending=True)[:min(12, len(pre))]]
    else:
        hard = neg
    bce = balanced_bce(score, y)
    pairwise = F.softplus(0.2 + score[hard][None, :] - score[pos][:, None]).mean() if len(pos) and len(hard) else zero
    listwise = torch.logsumexp(score, 0) - torch.logsumexp(score[pos], 0) if len(pos) else zero
    min_positive = F.softplus(0.2 + score[hard].max() - score[pos]).mean() if len(pos) and len(hard) else zero
    teacher = item["teacher"].to(device)
    teacher_distill = F.smooth_l1_loss(score - score.mean(), teacher - teacher.mean())
    residual_reg = out["residual"].square().mean()
    inactive = balanced_bce(score, torch.zeros_like(y)) if not y.any() else zero
    total = bce + pairwise + 0.5 * listwise + 0.5 * min_positive + 0.25 * teacher_distill + 0.05 * residual_reg + 0.25 * inactive
    return total, {
        "total": float(total.detach()), "frame_balanced_bce": float(bce.detach()),
        "pairwise_hard": float(pairwise.detach()), "multi_positive_listwise": float(listwise.detach()),
        "min_positive": float(min_positive.detach()), "teacher_distillation": float(teacher_distill.detach()),
        "residual_l2": float(residual_reg.detach()), "inactive_bce": float(inactive.detach()),
        "positive_count": int(len(pos)), "negative_count": int(len(neg)), "hard_negative_count": int(len(hard)),
    }, pos, hard


def forward_item(model, encoder, item, text, device):
    patch = encoder.encode(item).to(device)
    tokens = text["token_hidden"][item["text_index"]].float().to(device)
    mask = text["attention_mask"][item["text_index"]].bool().to(device)
    out = model(patch, tokens, mask, item["frozen_clip"].to(device), item["numeric"].to(device), item["teacher"].to(device))
    return out, patch


def common_provenance(selected, seed):
    return {
        "format": "locatemot-l51-b0-streaming-crop-adapter-v1",
        "stage": "B0",
        "project_root": str(ROOT),
        "seed": int(seed),
        "train_manifest": str((DATA / "train_units.jsonl").resolve()),
        "train_manifest_sha256": sha256_file(DATA / "train_units.jsonl"),
        "fit_only": True,
        "selected_unit_count": len(selected),
        "selected_domains": sorted(set(str(x["dataset"]) for x in selected)),
        "selected_videos": sorted(set(str(x["video"]) for x in selected)),
        "text_cache": str(TEXT_CACHE.resolve()),
        "text_cache_sha256": sha256_file(TEXT_CACHE),
        "fixed_manifest_sha256": sha256_file(FAST_MANIFEST),
        "official_test_labels_read": False,
        "screening_gt_used": False,
        "ordinary_mot_ovmot_touched": False,
        "semantic_inputs_excluded": ["source_id", "pool_id", "group_id", "state_key", "query_id"],
        "token_span_region_alignment": "UNALIGNED",
        "static_motion_language_mask": "UNALIGNED/not claimed",
        "raw_cache_written": False,
        "raw_crop_contract": {
            "box_source": "L19 candidate observation box",
            "padding": 0.10,
            "boundary": "clip",
            "encoder": "frozen CLIP ViT-B/16",
            "weights": str(CLIP_WEIGHTS),
            "weights_sha256": sha256_file(CLIP_WEIGHTS),
            "patch_tokens_after_fixed_2x2_pool": 4,
        },
        "l29_checkpoint": str(L29_CHECKPOINT.resolve()),
        "l29_checkpoint_sha256": sha256_file(L29_CHECKPOINT),
        "l19_bank_root": str(BANK_ROOT.resolve()),
        "l28_cache_root": str(L28_ROOT.resolve()),
    }


def preflight(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    units = choose_units(load_units())
    text = torch.load(TEXT_CACHE, map_location="cpu", weights_only=False)
    teacher = L29Teacher(text, torch.device("cpu"))
    materialized = materialize_units(units, text, teacher)
    # Actual crop decode is CPU-safe; the verified GPU CLIP API is audited
    # separately because this local CPU build crashes in CLIP's transformer.
    sample = materialized[0]
    crop_count = decode_crop_count(sample)
    if crop_count != len(sample["y"]):
        raise AssertionError("CPU crop/candidate alignment failed")
    generator = torch.Generator(device="cpu").manual_seed(7)
    patch = torch.randn((len(sample["y"]), 4, 768), generator=generator)
    model = L51StreamingCropAdapter(hidden=128, heads=4, layers=2).cpu().train()
    tokens = text["token_hidden"][sample["text_index"]].float()
    mask = text["attention_mask"][sample["text_index"]].bool()
    with torch.no_grad():
        initial = model(patch, tokens, mask, sample["frozen_clip"], sample["numeric"], sample["teacher"])
    initial_diff = float((initial["final_logit"] - sample["teacher"]).abs().max())
    if initial_diff != 0.0:
        raise AssertionError(f"initial teacher contract diff={initial_diff}")
    # Artificial multi-positive check on a real complete set.
    multi = next(x for x in materialized if int(x["y"].sum()) > 1 and int((~x["y"]).sum()) > 0)
    mp = torch.randn((len(multi["y"]), 4, 768), generator=generator)
    mt = text["token_hidden"][multi["text_index"]].float(); mm = text["attention_mask"][multi["text_index"]].bool()
    out = model(mp, mt, mm, multi["frozen_clip"], multi["numeric"], multi["teacher"])
    out["final_logit"].retain_grad()
    loss, parts, pos, hard = losses(out, multi, torch.device("cpu"))
    loss.backward()
    grad = out["final_logit"].grad.abs()
    pos_fraction = float((grad[pos] > 1e-10).float().mean()) if len(pos) else 0.0
    hard_fraction = float((grad[hard] > 1e-10).float().mean()) if len(hard) else 0.0
    if not math.isfinite(float(loss)) or pos_fraction != 1.0 or hard_fraction != 1.0:
        raise AssertionError(f"preflight gradient failure loss={float(loss)} pos={pos_fraction} hard={hard_fraction}")
    out_dir = Path(args.out_root); out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = common_provenance(units, args.seed)
    payload.update({
        "status": "pass", "mode": "cpu_single_batch_preflight",
        "sample_unit_key": sample["unit_key"], "sample_candidate_count": len(sample["y"]),
        "multi_positive_unit_key": multi["unit_key"], "multi_positive_count": int(multi["y"].sum()),
        "hard_negative_count": int(len(hard)), "crop_decode": True, "crop_count": crop_count,
        "model_contract_synthetic_patch_tokens": True,
        "patch_shape": list(patch.shape), "initial_residual_max_abs": float(initial["residual"].abs().max()),
        "initial_base_vs_final_max_abs": initial_diff, "loss_finite": True,
        "multi_positive_gradient_fraction": pos_fraction, "hard_gradient_fraction": hard_fraction,
        "loss_parts": parts,
    })
    (out_dir / "b0_preflight.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out_root); out = out if out.is_absolute() else ROOT / out
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    try:
        units = choose_units(load_units())
        text = torch.load(TEXT_CACHE, map_location="cpu", weights_only=False)
        teacher = L29Teacher(text, torch.device("cpu"))
        materialized = materialize_units(units, text, teacher)
        device = torch.device(args.device)
        if device.type != "cuda":
            raise RuntimeError("B0 training requires the authorized single GPU; use --preflight-only for CPU")
        encoder = StreamingClipPatches(device)
        model = L51StreamingCropAdapter(hidden=128, heads=4, layers=2).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        provenance = common_provenance(units, args.seed)
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        config = {"seed": args.seed, "steps": args.steps, "device": str(device), "model": model.config,
                  "sampling": {"unit_count": len(materialized), "schedule": "each stratified unit once, then seeded uniform"},
                  "loss": {"frame_balanced_bce": 1.0, "pairwise_hard": 1.0, "multi_positive_listwise": 0.5,
                           "min_positive": 0.5, "teacher_distillation": 0.25, "residual_l2": 0.05,
                           "inactive_bce": 0.25, "objectness_prefilter": 48, "current_score_hard_topk": 12},
                  "official_test_labels_read": False, "screening_gt_used": False,
                  "ordinary_mot_ovmot_touched": False}
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        sampling = Counter(); trace=[]; grad_trace=[]; crop_count=0
        start = time.time(); peak = 0
        # Initialization audit on the first complete candidate set.
        first = materialized[0]
        with torch.inference_mode():
            initial_out, initial_patch = forward_item(model.eval(), encoder, first, text, device)
            initial_diff = float((initial_out["final_logit"] - first["teacher"].to(device)).abs().max())
            initial_residual = float(initial_out["residual"].abs().max())
        if initial_diff != 0.0 or initial_residual != 0.0:
            raise AssertionError(f"initial teacher contract failed diff={initial_diff} residual={initial_residual}")
        del initial_out, initial_patch
        rng = np.random.default_rng(args.seed)
        model.train()
        for step in range(1, args.steps + 1):
            item = materialized[step - 1] if step <= len(materialized) else materialized[int(rng.integers(len(materialized)))]
            sampling[(item["dataset"], item["video"], item["category"])] += 1
            opt.zero_grad(set_to_none=True)
            out_values, patch = forward_item(model, encoder, item, text, device)
            out_values["final_logit"].retain_grad()
            total, parts, pos, hard = losses(out_values, item, device)
            if not torch.isfinite(total):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            total.backward()
            score_grad = out_values["final_logit"].grad.detach().abs()
            pos_frac = float((score_grad[pos] > 1e-10).float().mean()) if len(pos) else 0.0
            hard_frac = float((score_grad[hard] > 1e-10).float().mean()) if len(hard) else 0.0
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            finite_grads = all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters())
            if not finite_grads or not math.isfinite(grad_norm):
                raise FloatingPointError(f"nonfinite gradient at step {step}")
            group_grad = {}
            for name, p in model.named_parameters():
                group = "image_adapter" if name.startswith("image_proj") else "text_adapter" if name.startswith("text_proj") else "residual" if name.startswith("residual_head") else "other"
                group_grad[group] = max(group_grad.get(group, 0.0), float(p.grad.detach().abs().max()) if p.grad is not None else 0.0)
            opt.step()
            crop_count += int(patch.shape[0]); peak = max(peak, int(torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0))
            row = {"step": step, **parts, "gradient_norm": grad_norm,
                   "positive_grad_fraction": pos_frac, "hard_grad_fraction": hard_frac,
                   "residual_mean": float(out_values["residual"].detach().mean()),
                   "residual_max_abs": float(out_values["residual"].detach().abs().max()),
                   "group_grad_max": group_grad, "finite": True}
            trace.append(row); grad_trace.append(row)
            del out_values, patch, total
        elapsed = time.time() - start
        checkpoint = out / f"checkpoint_l51_b0_step{args.steps}.pt"
        torch.save({"format": "locatemot-l51-b0-checkpoint-v1", "stage": "B0", "model": model.state_dict(),
                    "optimizer": opt.state_dict(), "config": config, "provenance": provenance}, checkpoint)
        reload_model = L51StreamingCropAdapter(hidden=128, heads=4, layers=2).cpu()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        reload_model.load_state_dict(payload["model"], strict=True); reload_model.eval()
        reload_finite = all(torch.isfinite(v).all().item() for v in reload_model.state_dict().values() if torch.is_floating_point(v))
        if not reload_finite: raise FloatingPointError("reload state is nonfinite")
        (out / "sampling_trace.json").write_text(json.dumps({"counts": {"|".join(k): v for k,v in sampling.items()}, "unit_keys": [x["unit_key"] for x in materialized], "domains": sorted(set(x["dataset"] for x in materialized)), "videos": sorted(set(x["video"] for x in materialized))}, indent=2) + "\n")
        (out / "loss_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        metrics = {"format": "locatemot-l51-b0-metrics-v1", "stage": "B0", "status": "pass",
                   "seed": args.seed, "steps": args.steps, "finite_steps": len(trace), "nonzero_gradient_steps": sum(x["gradient_norm"] > 0 for x in trace),
                   "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
                   "checkpoint_reload": True, "reload_state_finite": reload_finite,
                   "crop_count": crop_count, "sampled_domain_count": len(set(x["dataset"] for x in materialized)),
                   "sampled_video_count": len(set(x["video"] for x in materialized)), "candidate_frame_key_drift": 0,
                   "base_vs_initial_residual_diff": initial_diff, "initial_residual_max_abs": initial_residual,
                   "residual_max_abs_over_run": max(x["residual_max_abs"] for x in trace),
                   "positive_gradient_fraction_min": min(x["positive_grad_fraction"] for x in trace if x["positive_count"]),
                   "hard_gradient_fraction_min": min(x["hard_grad_fraction"] for x in trace if x["hard_negative_count"]),
                   "image_adapter_nonzero_gradient_steps": sum(x["group_grad_max"].get("image_adapter", 0) > 0 for x in trace),
                   "residual_nonzero_gradient_steps": sum(x["group_grad_max"].get("residual", 0) > 0 for x in trace),
                   "loss_mean": {k: float(np.mean([x[k] for x in trace])) for k in ("total", "frame_balanced_bce", "pairwise_hard", "multi_positive_listwise", "min_positive", "teacher_distillation", "residual_l2", "inactive_bce")},
                   "gradient_norm_mean": float(np.mean([x["gradient_norm"] for x in trace])),
                   "peak_memory_bytes": peak, "elapsed_sec": elapsed, "steps_per_sec": args.steps / max(elapsed, 1e-9),
                   "official_test_labels_read": False, "screening_gt_used": False, "ordinary_mot_ovmot_touched": False,
                   "raw_cache_written": False, "token_span_region_alignment": "UNALIGNED", "static_motion_language_mask": "UNALIGNED/not claimed"}
        (out / "metrics_l51_b0.json").write_text(json.dumps(metrics, indent=2) + "\n")
        (out / "gradient_audit.json").write_text(json.dumps({"finite_steps": metrics["finite_steps"], "nonzero_gradient_steps": metrics["nonzero_gradient_steps"], "image_adapter_nonzero_gradient_steps": metrics["image_adapter_nonzero_gradient_steps"], "residual_nonzero_gradient_steps": metrics["residual_nonzero_gradient_steps"], "positive_gradient_fraction_min": metrics["positive_gradient_fraction_min"], "hard_gradient_fraction_min": metrics["hard_gradient_fraction_min"]}, indent=2) + "\n")
        (out / "reload_audit.json").write_text(json.dumps({"checkpoint": str(checkpoint.resolve()), "strict_load": True, "state_finite": reload_finite}, indent=2) + "\n")
        print(json.dumps(metrics, indent=2), flush=True)
    except Exception as exc:
        (out / "INCOMPLETE.md").write_text(f"# L51 B0 incomplete\n\nFirst actionable root cause: `{type(exc).__name__}: {exc}`\n")
        raise


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-root", required=True); ap.add_argument("--steps", type=int, default=100); ap.add_argument("--seed", type=int, default=20260829); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--preflight-only", action="store_true"); args = ap.parse_args()
    if Path.cwd().resolve() != ROOT: raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if args.preflight_only: preflight(args)
    else: train(args)


if __name__ == "__main__": main()
