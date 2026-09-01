#!/usr/bin/env python3
"""Train-only L44 integrated query/region/track decoder smoke.

The smoke materializes only a small, category-balanced set of train units in
RAM.  Raw crop patch tokens are produced by the existing frozen CLIP encoder
and are never written as a new feature bank.  The fixed screening manifest is
not read for fitting or selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l44_integrated_query_region_track_decoder import L44IntegratedQueryRegionTrackDecoder
from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
from tools.audit_l44_integrated_contract import (L19, L28, L29, SPLIT, V5,
                                                  TRAIN_VIDEOS, load_bank,
                                                  load_queries, sha256)
from tools.l40_raw_data import WEIGHTS
from tools.train_l42_current_frame_grounding import (StreamingCropPatchEncoder,
                                                      make_units, numeric_for)

FAST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"


def balanced_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Equalize positive and negative groups inside one frame/unit."""
    if logits.numel() == 0:
        return logits.new_zeros(())
    target = target.bool()
    pieces = []
    if target.any():
        pieces.append(F.binary_cross_entropy_with_logits(
            logits[target], torch.ones_like(logits[target])))
    if (~target).any():
        pieces.append(F.binary_cross_entropy_with_logits(
            logits[~target], torch.zeros_like(logits[~target])))
    return torch.stack(pieces).mean() if pieces else logits.new_zeros(())


def teacher_for(model, cache, query, frame, bank, rows, text_hidden,
                text_mask, device):
    """Map L29 current membership by valid cache track id, never row position."""
    from tools.train_l28_track_set_decoder import state_at
    ptr = cache["track_ptr"].numpy()
    frames = cache["obs_frame"].numpy()
    valid = [i for i in range(len(ptr) - 1)
             if np.any(frames[int(ptr[i]):int(ptr[i + 1])] <= int(frame))]
    obs, obs_mask, obs_time, _, _ = state_at(cache, int(frame), history=8)
    with torch.inference_mode():
        encoded = model.encode_observations(obs.to(device), obs_mask.to(device),
                                            obs_time.to(device))
        result = model.forward_encoded(
            encoded, encoded[1], text_hidden.to(device), text_mask.to(device))
    logits = result["current_membership_logits"].float().cpu()
    valid_ids = cache["track_ids"][torch.as_tensor(valid)].tolist()
    by_track = {int(track): float(score)
                for track, score in zip(valid_ids, logits.tolist())}
    values = [by_track.get(int(bank["track"][r]), -20.0) for r in rows]
    if not all(np.isfinite(values)):
        raise RuntimeError("nonfinite or missing L29 teacher mapping")
    return torch.as_tensor(values, dtype=torch.float32)


def history_for(cache, bank, rows, frame, query, history_len=8):
    """Build causal history and expression-level history labels per candidate."""
    feature = cache["obs_features"]
    ptr = cache["track_ptr"].tolist()
    frames = cache["obs_frame"].tolist()
    gt_ids = cache["obs_gt_ids"]
    track_to_index = {int(t): i for i, t in enumerate(cache["track_ids"].tolist())}
    n = len(rows)
    values = torch.zeros((n, history_len, int(feature.shape[1])), dtype=torch.float32)
    mask = torch.zeros((n, history_len), dtype=torch.bool)
    times = torch.zeros((n, history_len), dtype=torch.float32)
    labels = torch.zeros((n, history_len), dtype=torch.bool)
    continuation = torch.zeros(n, dtype=torch.bool)
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
            if gid is not None and str(gid) in target.get(int(frames[obs_index]), set()):
                labels[i, j] = True
        earlier = [j for j in chosen if int(frames[j]) < int(frame)]
        continuation[i] = any(
            gt_ids[j] is not None and str(gt_ids[j]) in target.get(int(frames[j]), set())
            for j in earlier
        )
    return values, mask, times, labels, continuation


def choose_hard(y, objectness, scores, limit=24, prelimit=96):
    neg = np.flatnonzero(~np.asarray(y, dtype=bool))
    if not len(neg):
        return np.empty(0, dtype=np.int64)
    pre = neg[np.argsort(-np.asarray(objectness)[neg], kind="stable")[:prelimit]]
    return pre[np.argsort(-np.asarray(scores)[pre], kind="stable")[:limit]]


def rank_counts(teacher, student, y, hard):
    pos = np.flatnonzero(y)
    hard = np.asarray(hard, dtype=np.int64)
    if not len(pos) or not len(hard):
        return {"pairs": 0, "teacher_correct": 0, "teacher_error": 0,
                "teacher_correct_flips": 0, "teacher_error_corrections": 0}
    td = teacher[pos, None] - teacher[hard][None, :]
    sd = student[pos, None] - student[hard][None, :]
    correct = td > 0
    error = ~correct
    return {
        "pairs": int(td.size),
        "teacher_correct": int(correct.sum()),
        "teacher_error": int(error.sum()),
        "teacher_correct_flips": int((correct & (sd < 0)).sum()),
        "teacher_error_corrections": int((error & (sd > 0)).sum()),
    }


def unit_loss(model, unit, text_hidden, text_mask, device):
    patch = unit["patch"].to(device).float()
    numeric = unit["numeric"].to(device).float()
    history = unit["history"].to(device).float()
    history_mask = unit["history_mask"].to(device)
    history_time = unit["history_time"].to(device).float()
    teacher = unit["teacher"].to(device).float()
    candidate_mask = torch.ones(patch.shape[0], dtype=torch.bool, device=device)
    qh = text_hidden[unit["query"]["text_index"]].to(device).float()
    qm = text_mask[unit["query"]["text_index"]].to(device).bool()
    y = unit["y"].to(device).bool()

    # The preliminary no-grad pass is nested inside the caller's BF16
    # autocast context.  With the default autocast weight cache, that pass can
    # cache detached casts of parameters that are used only by the final
    # residual head; the following trainable pass would then silently lose
    # those parameter gradients.  Disable the cache for both passes so online
    # mining remains no-grad without changing the trainable graph.
    amp_kwargs = {"device_type": "cuda", "dtype": torch.bfloat16,
                  "enabled": device.type == "cuda", "cache_enabled": False}
    with torch.autocast(**amp_kwargs):
        with torch.no_grad():
            prelim = model(patch, qh, numeric, history, history_mask, history_time,
                           teacher, candidate_mask, qm)["final_membership_logits"]
            hard = choose_hard(y.cpu().numpy(), unit["objectness"].numpy(),
                               prelim.detach().cpu().numpy())
        out = model(patch, qh, numeric, history, history_mask, history_time,
                    teacher, candidate_mask, qm)
    scores = out["final_membership_logits"]
    scores.retain_grad()
    out["residual"].retain_grad()
    pos = torch.nonzero(y, as_tuple=False).flatten()
    hidx = torch.as_tensor(hard, dtype=torch.long, device=device)
    zero = scores.new_zeros(())
    pos_scores = scores[pos] if len(pos) else scores[:0]
    hard_scores = scores[hidx] if len(hidx) else scores[:0]

    membership = balanced_bce(scores, y)
    hard_bce = (F.binary_cross_entropy_with_logits(
        hard_scores, torch.zeros_like(hard_scores)) if len(hidx) else zero)
    pairwise = (F.softplus(.2 + hard_scores[None, :] - pos_scores[:, None]).mean()
                if len(pos) and len(hidx) else zero)
    listwise = (torch.logsumexp(scores, 0) - torch.logsumexp(pos_scores, 0)
                if len(pos) else zero)
    min_positive = (F.binary_cross_entropy_with_logits(
        pos_scores, torch.ones_like(pos_scores)) if len(pos) else zero)

    teacher_distill = F.huber_loss(scores, teacher, delta=1.0)
    teacher_order_terms = []
    teacher_error_terms = []
    if len(pos) and len(hidx):
        td = teacher[pos, None] - teacher[hidx][None, :]
        sd = scores[pos, None] - scores[hidx][None, :]
        correct = td > 0
        error = ~correct
        if correct.any():
            teacher_order_terms.append(F.relu(-sd[correct]).mean())
        if error.any():
            teacher_error_terms.append(F.softplus(.1 - sd[error]).mean())
    teacher_order = torch.stack(teacher_order_terms).mean() if teacher_order_terms else zero
    teacher_error = torch.stack(teacher_error_terms).mean() if teacher_error_terms else zero

    history_mask_flat = history_mask
    history_logits = out["history_membership_logits"]
    history_y = unit["history_y"].to(device).bool()
    history_loss = balanced_bce(history_logits[history_mask_flat],
                                history_y[history_mask_flat])
    cont_target = unit["continuation_y"].to(device).float()
    continuation = F.binary_cross_entropy_with_logits(
        out["continuation_logits"], cont_target)
    null_target = torch.as_tensor(float(not y.any()), device=device)
    null_loss = F.binary_cross_entropy_with_logits(out["null_logit"], null_target)
    inactive = (balanced_bce(scores, torch.zeros_like(y)) if not y.any() else zero)
    same = history_mask[:, 1:] & history_mask[:, :-1]
    temporal = ((history_logits[:, 1:] - history_logits[:, :-1])[same].abs().mean()
                if same.any() else zero)
    brier = (torch.sigmoid(scores) - y.float()).pow(2).mean()
    drift = (scores - teacher).pow(2).mean()
    residual_l2 = out["residual"].pow(2).mean()
    residual_mean = out["residual"].mean().pow(2)

    # Teacher-order preservation is intentionally at least as important as the
    # small teacher-error correction; this is the safeguard learned scorers
    # lacked in L37/L42.
    total = (
        membership + .5 * hard_bce + pairwise + .5 * listwise + .5 * min_positive
        + .25 * teacher_distill + 1.0 * teacher_order + .5 * teacher_error
        + .25 * history_loss + .2 * continuation + .2 * null_loss + .2 * inactive
        + .05 * temporal + .05 * brier + .05 * drift + .02 * residual_l2
        + .02 * residual_mean
    )
    counts = rank_counts(teacher.detach().cpu().numpy(),
                         scores.detach().cpu().numpy(), y.cpu().numpy(), hard)
    part = {
        "total": float(total.detach()), "membership_bce": float(membership.detach()),
        "hard_bce": float(hard_bce.detach()), "pairwise": float(pairwise.detach()),
        "listwise": float(listwise.detach()), "min_positive": float(min_positive.detach()),
        "teacher_distillation": float(teacher_distill.detach()),
        "teacher_order": float(teacher_order.detach()),
        "teacher_error_correction": float(teacher_error.detach()),
        "history_membership": float(history_loss.detach()),
        "continuation": float(continuation.detach()), "null": float(null_loss.detach()),
        "inactive": float(inactive.detach()), "temporal": float(temporal.detach()),
        "brier": float(brier.detach()), "teacher_drift": float(drift.detach()),
        "residual_l2": float(residual_l2.detach()), "residual_mean_penalty": float(residual_mean.detach()),
        "positive_count": int(y.sum()), "hard_count": int(len(hard)),
        "candidate_count": int(len(y)), "null_unit": int(not y.any()),
        "multi_positive_unit": int(y.sum() > 1), "continuation_positive": int(cont_target.sum()),
        **counts,
    }
    return total, part, out, y, hidx


def build_units(queries, banks, caches, text_hidden, text_mask, teacher_fn, encoder,
                device, limit=32):
    meta = make_units(queries, banks, limit)
    units = []
    for query, fi, y_np in meta:
        bank = banks[query["video"]]
        begin, end = int(bank["ptr"][fi]), int(bank["ptr"][fi + 1])
        rows = list(range(begin, end))
        frame = int(bank["frame_ids"][fi])
        cache = caches[query["video"]]
        history, hmask, htime, hy, cy = history_for(
            cache, bank, rows, frame, query, history_len=8)
        units.append({
            "query": query,
            "frame": frame,
            "y": torch.as_tensor(y_np, dtype=torch.bool),
            "objectness": bank["objectness"][rows].cpu(),
            "numeric": numeric_for(bank, rows).cpu(),
            "patch": encoder.encode(query["video"], bank, rows).cpu(),
            "history": history.cpu(), "history_mask": hmask.cpu(),
            "history_time": htime.cpu(), "history_y": hy.cpu(),
            "continuation_y": cy.cpu(),
            "teacher": teacher_fn(cache=cache, query=query, frame=frame,
                                   bank=bank, rows=rows,
                                   text_hidden=text_hidden[query["text_index"]].float(),
                                   text_mask=text_mask[query["text_index"]],
                                   device=device),
            "category": ("multi_positive" if int(y_np.sum()) > 1 else
                         "positive" if bool(y_np.any()) else
                         "inactive" if not query["target"].get(frame, set()) else "other"),
        })
    return units, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="outputs/l44/train/integrated_smoke100")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--units", type=int, default=32)
    args = ap.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd().resolve()}")
    out_dir = Path(args.out_root)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    if out_dir.exists():
        raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    queries = load_queries()
    banks = {v: load_bank(v) for v in TRAIN_VIDEOS}
    caches = {v: torch.load(L28 / f"{v}.pt", map_location="cpu", weights_only=False)
              for v in TRAIN_VIDEOS}
    txt = torch.load(V5 / "text_tokens.pt", map_location="cpu", weights_only=False)
    text_hidden = txt["token_hidden"]
    text_mask = txt["attention_mask"].bool()

    l29 = L29FrameMembershipSetDecoder().to(device)
    l29.load_state_dict(torch.load(L29, map_location=device, weights_only=False)["model"], strict=True)
    l29.eval()
    encoder = StreamingCropPatchEncoder(device, batch_size=32)
    # Build the small in-memory unit set once.  The teacher mapper is passed as
    # a closure-shaped callable so its input/output contract is explicit.
    def teacher_fn(**kwargs):
        return teacher_for(l29, **kwargs)
    meta = make_units(queries, banks, args.units)
    units = []
    for query, fi, y_np in meta:
        bank = banks[query["video"]]
        begin, end = int(bank["ptr"][fi]), int(bank["ptr"][fi + 1])
        rows = list(range(begin, end)); frame = int(bank["frame_ids"][fi])
        cache = caches[query["video"]]
        history, hmask, htime, hy, cy = history_for(cache, bank, rows, frame, query, 8)
        units.append({
            "query": query, "frame": frame,
            "y": torch.as_tensor(y_np, dtype=torch.bool),
            "objectness": bank["objectness"][rows].cpu(),
            "numeric": numeric_for(bank, rows).cpu(),
            "patch": encoder.encode(query["video"], bank, rows).cpu(),
            "history": history.cpu(), "history_mask": hmask.cpu(),
            "history_time": htime.cpu(), "history_y": hy.cpu(),
            "continuation_y": cy.cpu(),
            "teacher": teacher_fn(cache=cache, query=query, frame=frame,
                                   bank=bank, rows=rows,
                                   text_hidden=text_hidden[query["text_index"]].float(),
                                   text_mask=text_mask[query["text_index"]], device=device),
            "category": ("multi_positive" if int(y_np.sum()) > 1 else
                         "positive" if bool(y_np.any()) else
                         "inactive" if not query["target"].get(frame, set()) else "other"),
        })
    del encoder, l29, caches, banks

    model = L44IntegratedQueryRegionTrackDecoder(
        image_dim=768, text_dim=768, numeric_dim=36, track_dim=1432,
        hidden=256, heads=8, layers=2, history_len=8, residual_bound=.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    trace = []; gradients = []; positive_grad = []; hard_grad = []
    residual_max = []; residual_mean = []; rank_sum = Counter()
    started = time.time(); model.train()
    amp = device.type == "cuda"
    for step in range(1, args.steps + 1):
        unit = units[int(rng.integers(len(units)))]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            loss, part, model_out, y, hidx = unit_loss(model, unit, text_hidden,
                                                       text_mask, device)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"nonfinite L44 loss at step {step}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        if not np.isfinite(grad_norm):
            raise FloatingPointError(f"nonfinite L44 gradient at step {step}")
        sg = model_out["final_membership_logits"].grad.detach().abs()
        pos = torch.nonzero(y, as_tuple=False).flatten()
        positive_grad.append(float((sg[pos] > 1e-10).float().mean()) if len(pos) else 0.0)
        hard_grad.append(float((sg[hidx] > 1e-10).float().mean()) if len(hidx) else 0.0)
        optimizer.step()
        gradients.append(grad_norm)
        residual = model_out["residual"].detach().float()
        residual_max.append(float(residual.abs().max()) if residual.numel() else 0.0)
        residual_mean.append(float(residual.mean()) if residual.numel() else 0.0)
        for key in ("pairs", "teacher_correct", "teacher_error",
                    "teacher_correct_flips", "teacher_error_corrections"):
            rank_sum[key] += int(part[key])
        trace.append({k: part[k] for k in (
            "total", "membership_bce", "hard_bce", "pairwise", "listwise",
            "min_positive", "teacher_distillation", "teacher_order",
            "teacher_error_correction", "history_membership", "continuation",
            "null", "inactive", "temporal", "brier", "teacher_drift",
            "residual_l2", "residual_mean_penalty")})

    ckpt = out_dir / f"checkpoint_l44_integrated_step{args.steps}.pt"
    categories = {k: sum(int(u["category"] == k) for u in units)
                  for k in ("multi_positive", "positive", "inactive", "other")}
    payload = {
        "format": "locatemot-l44-integrated-query-region-track-decoder-v1",
        "stage": "L44-B0-train-only-smoke", "seed": args.seed,
        "steps": args.steps, "device": str(device),
        "train_video_count": len(TRAIN_VIDEOS), "train_query_count": len(queries),
        "sampled_unit_count": len(units), "sample_categories": categories,
        "screening_gt_used_for_fit": False, "fixed_fast_manifest_used_for_training": False,
        "fast_manifest_sha256": sha256(FAST),
        "semantic_inputs_excluded": ["source_id", "pool_id", "group_id", "state_key"],
        "token_span_region_alignment": "UNALIGNED; no verified token/span boxes",
        "motion_language_decomposition": "not claimed; no verified motion-language mask",
        "model_config": model.config,
        "raw_feature_contract": {
            "image_encoder": "frozen CLIP ViT-B/16 patch tokens",
            "weights": str(WEIGHTS), "weights_sha256": sha256(WEIGHTS),
            "patch_tokens_per_candidate": 4, "pixel_storage": "transient RAM only",
            "numeric_dim": 36, "history_feature_dim": 1432, "history_len": 8,
        },
        "teacher": {
            "checkpoint": str(L29.resolve()), "sha256": sha256(L29),
            "role": "frozen current-membership anchor/distillation/order control",
        },
        "loss_contract": {
            "frame_balanced_membership_bce": True,
            "online_hard_negative": "objectness top-96 then current final score top-24",
            "hard_pairwise_margin": .2, "all_positive_listwise_and_min_positive": True,
            "teacher_huber_weight": .25, "teacher_order_weight": 1.0,
            "teacher_error_correction_weight": .5, "continuation_auxiliary": True,
            "inactive_null": True, "temporal_auxiliary": True,
            "calibration_brier": True, "residual_bound": .5,
        },
        "loss_mean": {k: float(np.mean([x[k] for x in trace])) for k in trace[0]},
        "loss_first": trace[0], "loss_last": trace[-1],
        "gradient_norm": {"mean": float(np.mean(gradients)),
                          "max": float(np.max(gradients)),
                          "nonzero_steps": int(np.count_nonzero(np.asarray(gradients) > 0))},
        "gradient_audit": {
            "positive_final_logit_nonzero_fraction_mean": float(np.mean(positive_grad)),
            "hard_final_logit_nonzero_fraction_mean": float(np.mean(hard_grad)),
            "all_positive_units_present": categories["multi_positive"] > 0,
        },
        "rank_diagnostics": {
            **dict(rank_sum),
            "teacher_correct_flip_ratio": rank_sum["teacher_correct_flips"] / max(1, rank_sum["teacher_correct"]),
            "teacher_error_correction_ratio": rank_sum["teacher_error_corrections"] / max(1, rank_sum["teacher_error"]),
        },
        "residual_diagnostics": {
            "max_abs_over_steps": float(max(residual_max, default=0.0)),
            "mean_abs_over_steps": float(np.mean(np.abs(residual_mean))),
            "mean_over_steps": float(np.mean(residual_mean)),
            "bound_satisfied": bool(max(residual_max, default=0.0) <= .5 + 1e-6),
        },
        "resource": {
            "elapsed_sec": time.time() - started,
            "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if amp else 0,
        },
    }
    torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": payload}, ckpt)
    reload_model = L44IntegratedQueryRegionTrackDecoder(**model.config)
    reload_model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model"], strict=True)
    payload["checkpoint"] = str(ckpt.resolve()); payload["checkpoint_reload"] = True
    payload["finite_loss"] = all(np.isfinite([x["total"] for x in trace]))
    (out_dir / f"metrics_l44_smoke{args.steps}.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out_dir / "loss_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    (out_dir / "config.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
