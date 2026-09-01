#!/usr/bin/env python3
"""L46 B0: train-only smoke for the RMOT query/region/track decoder.

The script materializes only 32 deterministic train units.  Frozen L19 CLIP
region vectors, L28 histories and L26 word-token rows are read in RAM and are
never written as a new cache.  L29 is loaded only to produce an auxiliary
teacher target; its current-frame emission is not the L46 learned output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
from locatemot.models.l46_end_to_end_query_region_track import (
    L46EndToEndQueryRegionTrackDecoder,
)
from tools.audit_l46_end_to_end_contract import load_bank as load_l46_bank
from tools.audit_l44_integrated_contract import (
    FAST, L19, L28, L29, V5, TRAIN_VIDEOS, load_queries, sha256,
    valid_teacher_indices,
)
from tools.train_l28_track_set_decoder import state_at
from tools.train_l42_current_frame_grounding import make_units, numeric_for


def balanced_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if logits.numel() == 0:
        return logits.new_zeros(())
    target = target.bool()
    parts = []
    if target.any():
        parts.append(F.binary_cross_entropy_with_logits(
            logits[target], torch.ones_like(logits[target])))
    if (~target).any():
        parts.append(F.binary_cross_entropy_with_logits(
            logits[~target], torch.zeros_like(logits[~target])))
    return torch.stack(parts).mean() if parts else logits.new_zeros(())


def history_for(cache, bank, rows, frame, query, history_len=8):
    """Build causal per-candidate history without using pool/source IDs."""
    feature = cache["obs_features"]
    ptr = cache["track_ptr"].tolist()
    frames = cache["obs_frame"].tolist()
    gt_ids = cache["obs_gt_ids"]
    track_to_index = {int(t): i for i, t in enumerate(cache["track_ids"].tolist())}
    values = torch.zeros((len(rows), history_len, int(feature.shape[1])), dtype=torch.float32)
    mask = torch.zeros((len(rows), history_len), dtype=torch.bool)
    times = torch.zeros((len(rows), history_len), dtype=torch.float32)
    labels = torch.zeros((len(rows), history_len), dtype=torch.bool)
    continuation = torch.zeros(len(rows), dtype=torch.bool)
    target = query["target"]
    for i, row in enumerate(rows):
        ti = track_to_index.get(int(bank["track"][row]))
        if ti is None:
            continue
        begin, end = int(ptr[ti]), int(ptr[ti + 1])
        eligible = [j for j in range(begin, end) if int(frames[j]) <= int(frame)]
        chosen = eligible[-history_len:]
        offset = history_len - len(chosen)
        if not chosen:
            continue
        values[i, offset:] = feature[torch.as_tensor(chosen)].float()
        mask[i, offset:] = True
        times[i, offset:] = torch.as_tensor(
            np.asarray([frames[j] for j in chosen], dtype=np.float32) /
            max(1.0, float(frame) + 1.0))
        for j, obs_index in enumerate(chosen, offset):
            gid = gt_ids[obs_index]
            labels[i, j] = gid is not None and str(gid) in target.get(int(frames[obs_index]), set())
        earlier = [j for j in chosen if int(frames[j]) < int(frame)]
        continuation[i] = any(
            gt_ids[j] is not None and str(gt_ids[j]) in target.get(int(frames[j]), set())
            for j in earlier)
    return values, mask, times, labels, continuation


def teacher_for(l29_model, cache, query, frame, bank, rows, text_hidden,
                text_mask, device):
    obs, obs_mask, obs_time, _, _ = state_at(cache, int(frame), history=8)
    valid = valid_teacher_indices(cache, int(frame))
    with torch.inference_mode():
        encoded = l29_model.encode_observations(
            obs.to(device), obs_mask.to(device), obs_time.to(device))
        output = l29_model.forward_encoded(
            encoded, encoded[1], text_hidden.to(device), text_mask.to(device))
    logits = output["current_membership_logits"].float().cpu()
    valid_ids = cache["track_ids"][torch.as_tensor(valid)].tolist()
    by_track = {int(track): float(score)
                for track, score in zip(valid_ids, logits.tolist())}
    result = torch.as_tensor(
        [by_track.get(int(bank["track"][row]), -20.0) for row in rows],
        dtype=torch.float32)
    if not torch.isfinite(result).all():
        raise RuntimeError("nonfinite or missing L29 teacher mapping")
    return result


def choose_hard(y, objectness, scores, prelimit=96, limit=24):
    neg = np.flatnonzero(~np.asarray(y, dtype=bool))
    if not len(neg):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    order = np.argsort(-np.asarray(objectness)[neg], kind="stable")
    pre = neg[order[:min(prelimit, len(neg))]]
    hard = pre[np.argsort(-np.asarray(scores)[pre], kind="stable")[:min(limit, len(pre))]]
    easy = np.setdiff1d(neg, hard, assume_unique=False)
    return hard, easy[:min(24, len(easy))]


def pairwise_teacher_terms(teacher, scores, pos, hard, zero):
    if not len(pos) or not len(hard):
        return zero, zero, {"pairs": 0, "teacher_correct": 0,
                            "teacher_error": 0, "teacher_correct_flips": 0,
                            "teacher_error_corrections": 0}
    td = teacher[pos, None] - teacher[hard][None, :]
    sd = scores[pos, None] - scores[hard][None, :]
    correct = td > 0
    error = ~correct
    preserve = F.relu(-sd[correct]).mean() if correct.any() else zero
    # A small error-correction term is permitted only for teacher-error pairs;
    # the larger preservation term prevents free global re-ranking.
    correct_error = F.softplus(0.1 - sd[error]).mean() if error.any() else zero
    counts = {
        "pairs": int(td.numel()), "teacher_correct": int(correct.sum()),
        "teacher_error": int(error.sum()),
        "teacher_correct_flips": int((correct & (sd < 0)).sum()),
        "teacher_error_corrections": int((error & (sd > 0)).sum()),
    }
    return preserve, correct_error, counts


def unit_loss(model, unit, text_hidden, text_mask, device):
    region = unit["region"].to(device).float()
    numeric = unit["numeric"].to(device).float()
    history = unit["history"].to(device).float()
    history_mask = unit["history_mask"].to(device)
    history_time = unit["history_time"].to(device).float()
    teacher = unit["teacher"].to(device).float()
    qidx = unit["text_local_index"]
    qh = text_hidden[qidx].to(device).float()
    qm = text_mask[qidx].to(device).bool()
    y = unit["y"].to(device).bool()
    n = int(region.shape[0])
    candidate_mask = torch.ones(n, dtype=torch.bool, device=device)

    out = model(region, qh, numeric, history, history_mask, history_time,
                candidate_mask=candidate_mask, text_mask=qm, teacher=teacher)
    scores = out["membership_logits"]
    scores.retain_grad()
    pos = torch.nonzero(y, as_tuple=False).flatten()
    with torch.no_grad():
        hard_np, easy_np = choose_hard(
            y.cpu().numpy(), unit["objectness"].float().numpy(),
            scores.detach().float().cpu().numpy())
    hard = torch.as_tensor(hard_np, dtype=torch.long, device=device)
    easy = torch.as_tensor(easy_np, dtype=torch.long, device=device)
    zero = scores.new_zeros(())
    pos_scores = scores[pos] if len(pos) else scores[:0]
    hard_scores = scores[hard] if len(hard) else scores[:0]
    easy_scores = scores[easy] if len(easy) else scores[:0]

    membership = balanced_bce(scores, y)
    hard_bce = (F.binary_cross_entropy_with_logits(
        hard_scores, torch.zeros_like(hard_scores)) if len(hard) else zero)
    easy_bce = (F.binary_cross_entropy_with_logits(
        easy_scores, torch.zeros_like(easy_scores)) if len(easy) else zero)
    pairwise = (F.softplus(.2 + hard_scores[None, :] - pos_scores[:, None]).mean()
                if len(pos) and len(hard) else zero)
    # Every positive is present in this mean, including the weakest positive.
    listwise = (-F.log_softmax(scores, dim=0)[pos].mean() if len(pos) else zero)
    min_positive = (F.softplus(.2 + hard_scores.max() - pos_scores.min())
                    if len(pos) and len(hard) else zero)
    seq_target = y.float()
    sequence = F.binary_cross_entropy_with_logits(
        out["sequence_logits"], seq_target)
    continuation_target = unit["continuation_y"].to(device).float()
    continuation = F.binary_cross_entropy_with_logits(
        out["continuation_logits"], continuation_target)
    history_y = unit["history_y"].to(device).bool()
    hist_logits = out["history_membership_logits"]
    history_loss = balanced_bce(hist_logits[history_mask], history_y[history_mask])
    null_target = torch.tensor(float(not y.any()), device=device)
    null_loss = F.binary_cross_entropy_with_logits(out["null_logit"], null_target)
    inactive = balanced_bce(scores, torch.zeros_like(y)) if not y.any() else zero
    brier = (torch.sigmoid(scores) - y.float()).pow(2).mean()
    teacher_distill = F.huber_loss(scores, teacher, delta=1.0)
    teacher_order, teacher_error, rank = pairwise_teacher_terms(
        teacher, scores, pos, hard, zero)
    valid_hist = history_mask[:, 1:] & history_mask[:, :-1]
    temporal = ((hist_logits[:, 1:] - hist_logits[:, :-1])[valid_hist].abs().mean()
                if valid_hist.any() else zero)
    set_drift = (scores.mean() - teacher.mean()).pow(2)
    total = (
        membership + 0.5 * hard_bce + 0.1 * easy_bce + pairwise
        + 0.5 * listwise + 0.5 * min_positive + 0.5 * sequence
        + 0.25 * teacher_distill + 1.0 * teacher_order + 0.25 * teacher_error
        + 0.25 * history_loss + 0.2 * continuation + 0.2 * null_loss
        + 0.2 * inactive + 0.05 * temporal + 0.05 * brier + 0.02 * set_drift
    )
    part = {
        "total": float(total.detach()), "membership_bce": float(membership.detach()),
        "hard_bce": float(hard_bce.detach()), "easy_bce": float(easy_bce.detach()),
        "pairwise": float(pairwise.detach()), "listwise": float(listwise.detach()),
        "min_positive": float(min_positive.detach()), "sequence_bce": float(sequence.detach()),
        "teacher_distillation": float(teacher_distill.detach()),
        "teacher_order": float(teacher_order.detach()),
        "teacher_error_correction": float(teacher_error.detach()),
        "history_membership": float(history_loss.detach()),
        "continuation": float(continuation.detach()), "null": float(null_loss.detach()),
        "inactive": float(inactive.detach()), "temporal": float(temporal.detach()),
        "brier": float(brier.detach()), "set_score_drift": float(set_drift.detach()),
        "positive_count": int(y.sum()), "hard_count": int(len(hard)),
        "easy_count": int(len(easy)), "candidate_count": int(len(y)),
        "multi_positive_unit": int(y.sum() > 1), "inactive_unit": int(not y.any()),
        "text_region_attention_entropy": float(out["text_region_attention_entropy"]),
        "set_attention_entropy": float(out["set_attention_entropy"]),
        **rank,
    }
    return total, part, out, y, hard


def amp_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                              cache_enabled=False)
    return nullcontext()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--units", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd().resolve()}")
    out = Path(args.out_root)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    queries = load_queries()
    banks = {video: load_l46_bank(video) for video in TRAIN_VIDEOS}
    meta = make_units(queries, banks, args.units)
    needed_text = sorted({q[0]["text_index"] for q in meta})
    text_blob = torch.load(V5 / "text_tokens.pt", map_location="cpu", weights_only=False)
    # Keep only selected train-query rows in fp32; the full text cache is not
    # retained after unit construction.
    text_hidden = text_blob["token_hidden"][needed_text].float().clone()
    text_mask = text_blob["attention_mask"][needed_text].bool().clone()
    text_remap = {old: i for i, old in enumerate(needed_text)}
    del text_blob

    selected_videos = sorted({q[0]["video"] for q in meta})
    caches = {video: torch.load(L28 / f"{video}.pt", map_location="cpu",
                               weights_only=False) for video in selected_videos}
    l29_model = L29FrameMembershipSetDecoder().to(device)
    l29_model.load_state_dict(torch.load(L29, map_location=device,
                                         weights_only=False)["model"], strict=True)
    l29_model.eval()
    units = []
    for query, fi, y_np in meta:
        bank = banks[query["video"]]
        begin, end = int(bank["ptr"][fi]), int(bank["ptr"][fi + 1])
        rows = list(range(begin, end)); frame = int(bank["frame_ids"][fi])
        history, hmask, htime, hy, cy = history_for(
            caches[query["video"]], bank, rows, frame, query, history_len=8)
        # L19's pooled CLIP region vector is the frozen current-region token.
        # It is not copied into a new bank; P=1 is explicit in the model input.
        region = bank["clip"][rows].float().unsqueeze(1).cpu()
        units.append({
            "query": query, "frame": frame, "y": torch.as_tensor(y_np, dtype=torch.bool),
            "text_local_index": text_remap[query["text_index"]],
            "objectness": bank["objectness"][rows].float().cpu(),
            "numeric": numeric_for(bank, rows).float().cpu(),
            "region": region, "history": history.cpu(),
            "history_mask": hmask.cpu(), "history_time": htime.cpu(),
            "history_y": hy.cpu(), "continuation_y": cy.cpu(),
            "teacher": teacher_for(
                l29_model, caches[query["video"]], query, frame, bank, rows,
                text_hidden[ text_remap[query["text_index"]] ],
                text_mask[text_remap[query["text_index"]]], device),
            "category": ("multi_positive" if int(y_np.sum()) > 1 else
                         "positive" if bool(y_np.any()) else
                         "inactive" if not query["target"].get(frame, set()) else "other"),
        })
    del l29_model, caches, banks

    model = L46EndToEndQueryRegionTrackDecoder(
        region_dim=512, text_dim=768, numeric_dim=36, track_dim=1432,
        hidden=256, heads=8, layers=2, history_len=8, dropout=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    trace = []; gradients = []; positive_grads = []; hard_grads = []
    rank_total = Counter(); attention = []; start = time.time(); model.train()
    for step in range(1, args.steps + 1):
        unit = units[int(rng.integers(len(units)))]
        optimizer.zero_grad(set_to_none=True)
        with amp_context(device):
            loss, part, output, y, hard = unit_loss(
                model, unit, text_hidden, text_mask, device)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite L46 loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        if not np.isfinite(grad_norm):
            raise FloatingPointError(f"nonfinite L46 gradient at step {step}")
        score_grad = output["membership_logits"].grad.detach().abs()
        pos = torch.nonzero(y, as_tuple=False).flatten()
        positive_grads.append(float((score_grad[pos] > 1e-10).float().mean()) if len(pos) else 0.0)
        hard_grads.append(float((score_grad[hard] > 1e-10).float().mean()) if len(hard) else 0.0)
        optimizer.step()
        gradients.append(grad_norm)
        attention.append({"text_region": part["text_region_attention_entropy"],
                          "set": part["set_attention_entropy"]})
        for key in ("pairs", "teacher_correct", "teacher_error",
                    "teacher_correct_flips", "teacher_error_corrections"):
            rank_total[key] += int(part[key])
        trace.append({key: part[key] for key in (
            "total", "membership_bce", "hard_bce", "easy_bce", "pairwise",
            "listwise", "min_positive", "sequence_bce", "teacher_distillation",
            "teacher_order", "teacher_error_correction", "history_membership",
            "continuation", "null", "inactive", "temporal", "brier",
            "set_score_drift", "positive_count", "hard_count", "easy_count")})

    checkpoint = out / f"checkpoint_l46_end_to_end_step{args.steps}.pt"
    categories = {name: sum(int(unit["category"] == name) for unit in units)
                  for name in ("multi_positive", "positive", "inactive", "other")}
    config = {
        "format": "locatemot-l46-end-to-end-query-region-track-v1",
        "stage": "L46-B0-train-only-smoke", "seed": args.seed,
        "steps": args.steps, "device": str(device), "sampled_unit_count": len(units),
        "sample_categories": categories, "train_video_count": len(TRAIN_VIDEOS),
        "train_query_count": len(queries), "selected_train_videos": selected_videos,
        "screening_gt_used_for_fit": False, "fixed_fast_manifest_used_for_training": False,
        "fixed_fast_manifest_sha256": sha256(FAST),
        "semantic_inputs_excluded": ["source_id", "pool_id", "group_id", "state_key"],
        "token_span_region_alignment": "UNALIGNED; no verified token/span boxes",
        "motion_language_decomposition": "not claimed; no verified motion-language mask",
        "model_config": model.config,
        "frozen_feature_contract": {
            "region": "L19 frozen CLIP pooled region vector as one current-region token",
            "region_dim": 512, "history": "L28 obs_features", "history_dim": 1432,
            "history_len": 8, "numeric_dim": 36, "persistent_new_cache": False,
        },
        "teacher": {"checkpoint": str(L29.resolve()), "sha256": sha256(L29),
                    "role": "auxiliary distillation/control only; L46 membership is learned"},
        "loss_contract": {
            "frame_balanced_membership_bce": True,
            "same_frame_hard_negative": "objectness top-96 prefilter, current membership top-24",
            "hard_pairwise_margin": 0.2,
            "multi_positive_all_gradient": True,
            "inactive_null": True, "continuation_auxiliary": True,
            "teacher_distillation_and_order": True, "brier": True,
            "set_competition": True,
        },
        "loss_mean": {key: float(np.mean([row[key] for row in trace])) for key in trace[0]},
        "loss_first": trace[0], "loss_last": trace[-1],
        "gradient_norm": {"mean": float(np.mean(gradients)),
                          "max": float(np.max(gradients)),
                          "nonzero_steps": int(np.count_nonzero(np.asarray(gradients) > 0))},
        "gradient_audit": {
            "positive_membership_logit_nonzero_fraction_mean": float(np.mean(positive_grads)),
            "hard_membership_logit_nonzero_fraction_mean": float(np.mean(hard_grads)),
            "all_positive_units_present": categories["multi_positive"] > 0,
            "all_sample_categories_present": all(categories[name] > 0 for name in categories),
        },
        "rank_diagnostics": {
            **dict(rank_total),
            "teacher_correct_flip_ratio": rank_total["teacher_correct_flips"] / max(1, rank_total["teacher_correct"]),
            "teacher_error_correction_ratio": rank_total["teacher_error_corrections"] / max(1, rank_total["teacher_error"]),
        },
        "attention_diagnostics": {
            "text_region_entropy_mean": float(np.mean([x["text_region"] for x in attention])),
            "set_entropy_mean": float(np.mean([x["set"] for x in attention])),
        },
        "resource": {
            "elapsed_sec": time.time() - start,
            "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0,
        },
    }
    torch.save({"model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "config": config}, checkpoint)
    reload_model = L46EndToEndQueryRegionTrackDecoder(**model.config)
    reload_model.load_state_dict(torch.load(checkpoint, map_location="cpu",
                                             weights_only=False)["model"], strict=True)
    config["checkpoint"] = str(checkpoint.resolve())
    config["checkpoint_reload"] = True
    config["finite_loss"] = bool(all(np.isfinite([row["total"] for row in trace])))
    (out / f"metrics_l46_smoke{args.steps}.json").write_text(
        json.dumps(config, indent=2) + "\n")
    (out / "loss_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (out / "README.md").write_text(
        "# L46 B0 train-only smoke\n\n"
        "RMOT-only decoder. Frozen L19/L28 features and train expressions are used; "
        "no screening GT or new dense cache is written.\n")
    print(json.dumps(config, indent=2), flush=True)


if __name__ == "__main__":
    main()
