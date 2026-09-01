#!/usr/bin/env python3
"""L49 semantic warm-up followed by bounded identity/NULL/sequence training.

The command intentionally runs as one blocking job and saves independently
reloadable checkpoints at the required milestones.  It consumes only the
L49 train unit manifest and frozen L19 observations.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder  # noqa: E402
from locatemot.models.l49_kitti_rmot import L49KittiRMOT, brier_loss  # noqa: E402
from locatemot.rmot.l49_data import (  # noqa: E402
    L29_CHECKPOINT, TEXT_CACHE, load_bank, sha256_file, unit_features,
)
from tools.train_l28_track_set_decoder import state_at  # noqa: E402

DATA = ROOT / "outputs/l49/data"
CONTRACT = ROOT / "outputs/l49/audit/kitti_data_contract.json"
L48_INIT = ROOT / "outputs/l48/train/semantic_smoke100/checkpoint_semantic_step100.pt"
L28_CACHE = ROOT / "outputs/l28/track_sequence_bank_final"
SAVE_STEPS = (100, 250, 500, 1000, 2500, 5000)


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class BankStore:
    """Share a bank between V1/V2 because both domains use the same frozen file."""
    def __init__(self, limit: int = 2):
        self.limit = int(limit)
        self.cache: OrderedDict[str, dict] = OrderedDict()

    def get(self, dataset: str, video: str):
        key = str(video)
        if key not in self.cache:
            self.cache[key] = load_bank(dataset, key)
            if len(self.cache) > self.limit:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return self.cache[key]


def balanced_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if not len(logits):
        return logits.new_zeros(())
    target = target.bool()
    terms = []
    if target.any():
        terms.append(F.binary_cross_entropy_with_logits(logits[target], torch.ones_like(logits[target])))
    if (~target).any():
        terms.append(F.binary_cross_entropy_with_logits(logits[~target], torch.zeros_like(logits[~target])))
    return torch.stack(terms).mean() if terms else logits.new_zeros(())


def build_teacher_cache(bank: dict) -> dict:
    """Build the L29 input cache from frozen L19 observations only."""
    tensors = bank["tensors"]
    count = int(tensors["track_id"].numel())
    by_track = defaultdict(list)
    for row, track in enumerate(tensors["track_id"].long().tolist()):
        by_track[int(track)].append(row)
    tracks = sorted(by_track)
    ordered = [row for track in tracks for row in by_track[track]]
    order = torch.as_tensor(ordered, dtype=torch.long)
    required = ("clip", "history_clip", "uidm_h", "geometry", "motion", "lifecycle", "objectness")
    if any(name not in tensors for name in required):
        missing = [name for name in required if name not in tensors]
        raise KeyError(f"L29 teacher cache missing fields: {missing}")
    feature = torch.cat([
        tensors[name].float().reshape(count, -1)
        for name in required
    ], 1).half()[order].contiguous()
    return {
        "track_ids": torch.as_tensor(tracks, dtype=torch.long),
        "track_ptr": torch.as_tensor([0] + list(np.cumsum([len(by_track[t]) for t in tracks])), dtype=torch.long),
        "obs_features": feature,
        "obs_frame": tensors["frame"].long()[order].to(torch.int32),
        "obs_gt_ids": [None] * len(ordered),
    }


def valid_track_indices(cache: dict, cutoff: int):
    ptr = cache["track_ptr"].numpy()
    frames = cache["obs_frame"].numpy()
    return [index for index in range(len(ptr) - 1)
            if np.any(frames[int(ptr[index]):int(ptr[index + 1])] <= int(cutoff))]


class L29Teacher:
    """Lazy, frozen L29 control used at low weight during L49 training."""
    def __init__(self, text_payload: dict, device: torch.device):
        self.device = device
        self.text = text_payload
        self.model = L29FrameMembershipSetDecoder().to(device)
        checkpoint = torch.load(L29_CHECKPOINT, map_location=device, weights_only=False)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        self.cache: OrderedDict[str, dict] = OrderedDict()

    def _cache_for(self, bank: dict):
        key = str(bank["path"])
        if key not in self.cache:
            if not (L28_CACHE / f"{bank['path'].stem}.pt").exists():
                return None
            self.cache[key] = build_teacher_cache(bank)
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return self.cache[key]

    @torch.inference_mode()
    def score(self, unit: dict, bank: dict) -> torch.Tensor | None:
        cache = self._cache_for(bank)
        if cache is None:
            return None
        index = self.text["sentence_to_index"].get(unit["sentence"])
        if index is None:
            return None
        frame = int(unit["frame_id"])
        obs, mask, obs_time, _, _ = state_at(cache, frame, history=8)
        if not len(obs):
            return None
        text = self.text["token_hidden"][index].to(self.device)
        text_mask = self.text["attention_mask"][index].bool().to(self.device)
        encoded = self.model.encode_observations(obs.to(self.device), mask.to(self.device), obs_time.to(self.device))
        output = self.model.forward_encoded(encoded, encoded[1], text, text_mask)
        tracks = self.model_track_ids(cache, frame)
        values = {int(track): float(value) for track, value in
                  zip(tracks, output["current_membership_logits"].float().cpu().tolist())}
        tensor = bank["tensors"]
        rows = range(int(unit["begin"]), int(unit["end"]))
        return torch.as_tensor([values.get(int(tensor["track_id"][row]), -20.0) for row in rows], dtype=torch.float32)

    @staticmethod
    def model_track_ids(cache: dict, frame: int):
        indices = valid_track_indices(cache, frame)
        return cache["track_ids"][torch.as_tensor(indices, dtype=torch.long)].tolist()


def pair_losses(score: torch.Tensor, target: torch.Tensor):
    pos = torch.nonzero(target, as_tuple=False).flatten()
    neg = torch.nonzero(~target, as_tuple=False).flatten()
    zero = score.new_zeros(())
    with torch.no_grad():
        hard = neg[torch.argsort(score.detach()[neg], descending=True)[:min(24, len(neg))]] if len(neg) else neg
    pairwise = F.softplus(.25 + score[hard][None, :] - score[pos][:, None]).mean() if len(pos) and len(hard) else zero
    listwise = torch.logsumexp(score, 0) - torch.logsumexp(score[pos], 0) if len(pos) and len(score) else zero
    min_positive = F.softplus(.25 + score[hard].max() - score[pos]).mean() if len(pos) and len(hard) else zero
    return pairwise, listwise, min_positive, pos, hard


def unit_loss(model: L49KittiRMOT, values: dict[str, torch.Tensor], device: torch.device,
              phase: str, teacher_score: torch.Tensor | None):
    move = {key: value.to(device, non_blocking=True) for key, value in values.items()
            if key not in ("target",)}
    target = values["target"].to(device, non_blocking=True)
    out = model(move["clip"], move["history_clip"], move["geometry"], move["motion"],
                move["context"], move["lifecycle"], move["objectness"], move["text"],
                move["text_mask"], move["relation"], move["history_sequence"],
                move["history_mask"], stage=phase)
    score = out["final_logit"]
    score.retain_grad()
    semantic = out["semantic_logit"]
    pairwise, listwise, min_positive, pos, hard = pair_losses(score, target)
    membership = balanced_bce(score, target)
    semantic_control = balanced_bce(semantic, target)
    null_target = score.new_full(out["null_logit"].shape, float(not bool(target.any())))
    null_loss = F.binary_cross_entropy_with_logits(out["null_logit"], null_target)
    history_valid = move["history_mask"].any(-1)
    continuation_target = target & history_valid
    continuation_loss = F.binary_cross_entropy_with_logits(out["continuation_logit"], continuation_target.float())
    sequence_consistency = F.smooth_l1_loss(score, semantic.detach())
    calibration = brier_loss(score, target)
    inactive_loss = balanced_bce(score, torch.zeros_like(target)) if not target.any() else score.new_zeros(())
    teacher_loss = score.new_zeros(())
    if teacher_score is not None and len(teacher_score) == len(score):
        teacher_loss = F.smooth_l1_loss(score - score.mean(), teacher_score.to(device) - teacher_score.to(device).mean())
    if phase in ("semantic", "semantic_warmup"):
        total = semantic_control + pairwise + .5 * listwise + .5 * min_positive + .15 * null_loss + .1 * calibration + .1 * teacher_loss
    else:
        total = membership + pairwise + .5 * listwise + .5 * min_positive + .2 * continuation_loss + .2 * null_loss + .1 * sequence_consistency + .1 * calibration + .1 * teacher_loss + .1 * inactive_loss
    parts = {
        "total": float(total.detach()), "membership": float(membership.detach()),
        "semantic_control": float(semantic_control.detach()), "pairwise": float(pairwise.detach()),
        "listwise_all_positive": float(listwise.detach()), "min_positive": float(min_positive.detach()),
        "continuation": float(continuation_loss.detach()), "null": float(null_loss.detach()),
        "sequence_consistency": float(sequence_consistency.detach()), "calibration_brier": float(calibration.detach()),
        "inactive": float(inactive_loss.detach()), "teacher_distillation": float(teacher_loss.detach()),
        "positive_count": int(len(pos)), "negative_count": int(len(torch.nonzero(~target, as_tuple=False))),
        "hard_negative_count": int(len(hard)), "null_target": float(null_target.detach()),
    }
    return total, parts, out, pos, hard


def save_checkpoint(out: Path, model: L49KittiRMOT, optimizer, step: int,
                    args, contract_sha: str, text_sha: str, trace, counts, start: float,
                    device: torch.device):
    checkpoint = out / f"checkpoint_l49_step{step}.pt"
    metadata = {
        "format": "locatemot-l49-kitti-rmot-v1", "stage": "C1-long-training",
        "seed": args.seed, "steps": step, "warmup_steps": args.warmup,
        "model_config": model.config(), "model": model.state_dict(),
        "data_contract_sha256": contract_sha, "text_cache_sha256": text_sha,
        "init_l48_checkpoint": str(L48_INIT.resolve()),
        "init_l48_checkpoint_sha256": sha256_file(L48_INIT),
        "l29_checkpoint": str(L29_CHECKPOINT.resolve()),
        "l29_checkpoint_sha256": sha256_file(L29_CHECKPOINT),
        "train_only": True, "screening_gt_used": False,
        "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
        "checkpoint_step": step, "elapsed_sec": time.time() - start,
    }
    torch.save(metadata, checkpoint)
    reload_model = L49KittiRMOT(hidden=model.hidden, heads=4, history_length=model.history_length).to(device)
    reload_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"], strict=True)
    reload_model.eval()
    with torch.inference_mode():
        finite = all(torch.isfinite(value).all().item() for value in reload_model.state_dict().values()
                     if torch.is_floating_point(value))
    if not finite:
        raise FloatingPointError(f"checkpoint reload has nonfinite state at step {step}")
    window = [row for row in trace if int(row["step"]) > max(0, step - 100)]
    metrics = {
        "format": "locatemot-l49-training-checkpoint-metrics-v1", "stage": "C0/C1",
        "step": step, "seed": args.seed, "device": str(device),
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_reload": True, "reload_state_finite": finite,
        "window_steps": len(window), "loss_mean": {
            key: float(np.mean([row[key] for row in window])) for key in
            ("total", "membership", "semantic_control", "pairwise", "listwise_all_positive",
             "min_positive", "continuation", "null", "sequence_consistency",
             "calibration_brier", "inactive", "teacher_distillation")},
        "gradient_norm": {"mean": float(np.mean([row["gradient_norm"] for row in window])),
                          "max": float(np.max([row["gradient_norm"] for row in window])),
                          "nonzero_steps": int(sum(row["gradient_norm"] > 0 for row in window))},
        "sample_counts": {key: int(value) for key, value in sorted(counts.items())},
        "elapsed_sec": time.time() - start, "train_only": True,
        "screening_gt_used": False, "official_test_labels_read": False,
    }
    (out / f"metrics_l49_step{step}.json").write_text(json.dumps(metrics, indent=2) + "\n")
    del reload_model
    gc.collect()
    return checkpoint, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="outputs/l49/train/joint_long5000")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260829)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-prob", type=float, default=.15)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")
    out = Path(args.out_root)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    started = time.time()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    try:
        contract = json.loads(CONTRACT.read_text())
        if contract.get("decision") != "enter_B0":
            raise RuntimeError("L49 data contract is not enter_B0")
        units = load_jsonl(DATA / "train_units.jsonl")
        if not units or {x["dataset"] for x in units} != {"refer_kitti_v1", "refer_kitti_v2"}:
            raise RuntimeError("L49 train units do not contain exactly V1 and V2")
        text_payload = torch.load(TEXT_CACHE, map_location="cpu", weights_only=False)
        if not {x["sentence"] for x in units}.issubset(text_payload["sentence_to_index"]):
            raise RuntimeError("L49 train units contain a sentence absent from frozen text cache")
        data_contract_sha = sha256_file(CONTRACT); text_sha = sha256_file(TEXT_CACHE)
        device = torch.device(args.device if args.device != "cpu" or not torch.cuda.is_available() else "cpu")
        model = L49KittiRMOT(hidden=256, heads=4, history_length=8).to(device)
        init = torch.load(L48_INIT, map_location="cpu", weights_only=False)
        model.semantic.load_state_dict(init["model"], strict=True)
        del init
        teacher = L29Teacher(text_payload, device)
        # Stage C0 only updates semantic parameters.  Auxiliary heads are not
        # unfrozen until the semantic warm-up boundary.
        semantic_params = list(model.semantic_parameters())
        auxiliary_params = list(model.auxiliary_parameters())
        for parameter in semantic_params:
            parameter.requires_grad_(True)
        for parameter in auxiliary_params:
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW([
            {"params": semantic_params, "lr": 2e-4},
            {"params": auxiliary_params, "lr": 0.0},
        ], weight_decay=1e-4)
        rng = random.Random(args.seed)
        by_domain = defaultdict(list)
        for unit in units:
            by_domain[unit["dataset"]].append(unit)
        domains = ("refer_kitti_v1", "refer_kitti_v2")
        store = BankStore(limit=2)
        trace = []
        counts = Counter()
        saved = {}
        last_phase = "semantic_warmup"
        model.train()
        for step in range(1, args.steps + 1):
            if step == args.warmup + 1:
                for parameter in semantic_params:
                    parameter.requires_grad_(False)
                for parameter in auxiliary_params:
                    parameter.requires_grad_(True)
                optimizer.param_groups[0]["lr"] = 0.0
                optimizer.param_groups[1]["lr"] = 5e-5
                last_phase = "identity_continuation_null_sequence"
            phase = "semantic_warmup" if step <= args.warmup else last_phase
            domain = domains[(step - 1) % len(domains)]
            unit = rng.choice(by_domain[domain])
            category = str(unit["category"])
            counts[f"{domain}|{category}"] += 1
            bank = store.get(domain, unit["video"])
            values = unit_features(unit, bank, text_payload, history=8)
            teacher_value = None
            teacher_used = False
            if args.teacher_prob > 0 and rng.random() < args.teacher_prob and step > args.warmup:
                teacher_value = teacher.score(unit, bank)
                teacher_used = teacher_value is not None
            optimizer.zero_grad(set_to_none=True)
            enabled = device.type == "cuda"
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled):
                loss, parts, output, pos, hard = unit_loss(model, values, device, phase, teacher_value)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            loss.backward()
            if output["final_logit"].grad is not None:
                grad = output["final_logit"].grad
                parts["positive_grad_nonzero"] = float((grad[pos].abs() > 1e-12).float().mean().detach().cpu()) if len(pos) else 0.0
                parts["hard_grad_nonzero"] = float((grad[hard].abs() > 1e-12).float().mean().detach().cpu()) if len(hard) else 0.0
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
            if not np.isfinite(grad_norm):
                raise FloatingPointError(f"nonfinite gradient at step {step}")
            optimizer.step()
            parts.update({"step": step, "domain": domain, "category": category,
                          "phase": phase, "gradient_norm": grad_norm,
                          "teacher_used": teacher_used,
                          "residual_identity_abs_max": float(output["deltas"]["identity"].detach().abs().max().cpu()),
                          "residual_continuation_abs_max": float(output["deltas"]["continuation"].detach().abs().max().cpu()),
                          "residual_sequence_abs_max": float(output["deltas"]["sequence"].detach().abs().max().cpu()),
                          "semantic_mean": float(output["semantic_logit"].detach().mean().cpu()),
                          "final_mean": float(output["final_logit"].detach().mean().cpu())})
            trace.append(parts)
            if step in SAVE_STEPS and step <= args.steps:
                saved[step], _ = save_checkpoint(out, model, optimizer, step, args,
                                                  data_contract_sha, text_sha, trace, counts,
                                                  started, device)
                print(json.dumps({"event": "checkpoint", "step": step,
                                  "path": str(saved[step]), "phase": phase}, ensure_ascii=False), flush=True)
        (out / "loss_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        final = {
            "format": "locatemot-l49-joint-training-summary-v1", "stage": "C1",
            "seed": args.seed, "steps": args.steps, "warmup_steps": args.warmup,
            "checkpoints": {str(k): str(v.resolve()) for k, v in saved.items()},
            "checkpoint_sha256": {str(k): sha256_file(v) for k, v in saved.items()},
            "train_units_available": len(units), "sample_counts": dict(counts),
            "loss_mean": {key: float(np.mean([row[key] for row in trace])) for key in
                          ("total", "membership", "semantic_control", "pairwise", "listwise_all_positive",
                           "min_positive", "continuation", "null", "sequence_consistency",
                           "calibration_brier", "inactive", "teacher_distillation")},
            "gradient_norm": {"mean": float(np.mean([row["gradient_norm"] for row in trace])),
                              "max": float(np.max([row["gradient_norm"] for row in trace])),
                              "nonzero_steps": int(sum(row["gradient_norm"] > 0 for row in trace))},
            "teacher_used_steps": int(sum(bool(row["teacher_used"]) for row in trace)),
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
            "elapsed_sec": time.time() - started, "data_contract_sha256": data_contract_sha,
            "text_cache_sha256": text_sha, "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "token_span_region_alignment": "UNALIGNED", "static_motion_language_mask": "UNALIGNED/not claimed",
        }
        (out / "metrics_l49_training_summary.json").write_text(json.dumps(final, indent=2) + "\n")
        print(json.dumps(final, indent=2), flush=True)
    except Exception as exc:
        (out / "INCOMPLETE.md").write_text("# L49 training incomplete\n\n" + f"First actionable error: `{type(exc).__name__}: {exc}`\n")
        raise


if __name__ == "__main__":
    main()
