"""Train and sanity-check the Stage L19 ungated retriever.

The trainer is intentionally separate from the L18 CLI so old checkpoints and
the L18 proxy protocol remain reproducible.  It reuses the frozen bank loader
and adds decoupled targets, source/state-balanced item sampling, verified
query swaps, and explicit truncated BPTT.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l16_track_selector import expression_family_vector
from locatemot.models.l19_ungated_retriever import L19UngatedRetriever
from tools.train_l18_carr import (
    BankStore, TextStore, frame_features, l19_frame_targets, load_items,
    l19_track_membership_index,
)


def expression_text(entry: dict) -> str:
    return str(entry.get("sentence", entry.get("expression", "")))


def query_identity_set(entry: dict) -> set[str]:
    result = set()
    for values in entry.get("label", {}).values():
        result.update(str(value) for value in values)
    return result


def item_category(item: dict, bank: dict) -> tuple[str, dict]:
    """Categorize an item using only train annotations and frozen bank rows."""
    tensors = bank["tensors"]
    labels = bank.get("candidate_gt", [])
    ptr = tensors["frame_ptr"].tolist()
    frame_to_index = bank["frame_to_index"]
    pool = tensors.get("pool_id")
    pool_values = pool.numpy() if pool is not None else np.zeros(len(labels), np.int64)
    target_frames = 0
    main_only = reserve_only = uncovered = False
    hard = False
    for frame, ids in item["entry"].get("label", {}).items():
        if not ids:
            continue
        frame_index = frame_to_index.get(int(frame))
        if frame_index is None:
            continue
        target_frames += 1
        begin, end = ptr[frame_index], ptr[frame_index + 1]
        target_ids = {str(value) for value in ids}
        row_labels = labels[begin:end]
        main = any(value is not None and str(value) in target_ids and
                   pool_values[begin + local] == 0
                   for local, value in enumerate(row_labels))
        reserve = any(value is not None and str(value) in target_ids and
                      pool_values[begin + local] == 1
                      for local, value in enumerate(row_labels))
        if main:
            main_only = True
        elif reserve:
            reserve_only = True
        else:
            uncovered = True
        positive = sum(value is not None and str(value) in target_ids
                       for value in row_labels)
        if len(target_ids) > 1 or (positive and len(row_labels) - positive >= 2):
            hard = True
    if reserve_only and not main_only:
        primary = "reserve_positive"
    elif uncovered and not main_only and not reserve_only:
        primary = "present_uncovered"
    elif main_only:
        primary = "main_positive"
    else:
        primary = "ordinary_negative"
    metadata = {
        "primary": primary, "has_target_frames": bool(target_frames),
        "hard_negative": bool(hard), "identity_count": len(query_identity_set(item["entry"])),
        "has_main_covered": bool(main_only),
        "has_reserve_covered": bool(reserve_only),
        "has_present_uncovered": bool(uncovered),
    }
    return primary, metadata


def build_item_buckets(items_by_domain: dict, store: BankStore) -> tuple[dict, dict]:
    """Scan each bank once and form reusable source/state sampling buckets."""
    buckets = defaultdict(list)
    metadata = {}
    grouped = defaultdict(list)
    for domain, items in items_by_domain.items():
        for item in items:
            grouped[(item["bank_dataset"], item["video"])].append((domain, item))
    for key, values in sorted(grouped.items()):
        bank = store.get(*key)
        for domain, item in values:
            primary, row = item_category(item, bank)
            token = (domain, item["video"], expression_text(item["entry"]))
            metadata[token] = {**row, "domain": domain, "video": item["video"]}
            buckets[primary].append(item)
            # Secondary memberships deliberately oversample mixed trajectories
            # that contain a reserve rescue or a present-but-uncovered frame;
            # the target construction itself remains unchanged.
            if row["has_reserve_covered"] and primary != "reserve_positive":
                buckets["reserve_positive"].append(item)
            if row["has_present_uncovered"] and primary != "present_uncovered":
                buckets["present_uncovered"].append(item)
            if row["hard_negative"]:
                buckets["hard_negative"].append(item)
    return dict(buckets), metadata


def choose_item(items_by_domain: dict, buckets: dict, rng: random.Random) -> tuple[dict, str]:
    """Prefer reserve/uncovered states but fall back without changing labels."""
    choices = [
        ("reserve_positive", 0.32),
        ("present_uncovered", 0.22),
        ("main_positive", 0.20),
        ("hard_negative", 0.16),
        ("ordinary_negative", 0.10),
    ]
    available = [(name, weight) for name, weight in choices if buckets.get(name)]
    total = sum(weight for _name, weight in available)
    value = rng.random() * max(total, 1e-6)
    for name, weight in available:
        value -= weight
        if value <= 0:
            return rng.choice(buckets[name]), name
    domain = "kitti" if rng.random() < 0.5 else "dance"
    return rng.choice(items_by_domain[domain]), "fallback"


def choose_verified_swap(base: dict, by_video: dict[str, list[dict]],
                         rng: random.Random) -> tuple[dict, dict | None]:
    """Choose a different expression whose identity labels actually differ."""
    candidates = []
    base_text = expression_text(base["entry"])
    base_ids = query_identity_set(base["entry"])
    for candidate in by_video.get(base["video"], []):
        text = expression_text(candidate["entry"])
        if text == base_text:
            continue
        candidate_ids = query_identity_set(candidate["entry"])
        if candidate_ids != base_ids:
            candidates.append((candidate, candidate_ids))
    if not candidates:
        return base, None
    chosen, chosen_ids = rng.choice(candidates)
    info = {
        "original_query": base_text,
        "swapped_query": expression_text(chosen["entry"]),
        "original_gt_identity": sorted(base_ids),
        "swapped_gt_identity": sorted(chosen_ids),
        "identity_changed": True,
        "all_negative_forced": False,
        "labels_recomputed_from_swapped_entry": True,
    }
    return chosen, info


def item_episode(item: dict, store: BankStore, text_store: TextStore,
                  rng: random.Random, sequence_length: int, burn_in: int,
                  device: torch.device) -> tuple:
    bank = store.get(item["bank_dataset"], item["video"])
    if "l19_track_membership" not in bank:
        bank["l19_track_membership"] = l19_track_membership_index(bank)
    entry = item["entry"]
    frame_count = len(bank["tensors"]["frame_ids"])
    total_length = min(frame_count, int(sequence_length))
    loss_length = max(1, total_length - int(burn_in))
    positive_frames = [
        bank["frame_to_index"][int(frame)]
        for frame, ids in entry.get("label", {}).items()
        if ids and int(frame) in bank["frame_to_index"]
    ]
    if positive_frames and rng.random() < 0.75:
        anchor = rng.choice(positive_frames)
        start = max(0, min(frame_count - total_length,
                           anchor - rng.randrange(max(1, loss_length))))
        mode = "positive_covered"
    else:
        start = rng.randrange(max(1, frame_count - total_length + 1))
        mode = "random_or_empty"
    query = torch.as_tensor(np.asarray(entry["spec"], np.float32), device=device)
    text = expression_text(entry)
    family = expression_family_vector(text).to(device)
    tokens, mask = text_store.get(text, device)
    episode = []
    for frame_index in range(start, start + total_length):
        features, track_ids, begin, end = frame_features(
            bank, frame_index, device)
        frame_id = int(bank["tensors"]["frame_ids"][frame_index])
        targets = l19_frame_targets(
            bank, begin, end, entry, frame_id, bank["l19_track_membership"])
        episode.append({
            "features": features, "track_ids": track_ids,
            "membership": torch.as_tensor(targets["membership"], device=device),
            "presence": torch.as_tensor(targets["presence"], device=device),
            "current_match": torch.as_tensor(targets["current_match"], device=device),
            "state": int(targets["state"]), "active": bool(targets["active"]),
            "source": torch.as_tensor(targets["source"], device=device),
            "group": torch.as_tensor(targets["group"], device=device),
            "frame": frame_id,
            "target_ids": targets["target_ids"],
        })
    return item, query, family, tokens, mask, episode, mode


def detach_state(state: dict) -> dict:
    result = {}
    for key, value in state.items():
        if isinstance(value, dict):
            result[key] = {name: tensor.detach() for name, tensor in value.items()}
        else:
            result[key] = value.detach()
    return result


def balanced_bce(logits: torch.Tensor, target: torch.Tensor,
                 source: torch.Tensor | None = None,
                 kind: str = "membership") -> torch.Tensor:
    if not len(logits):
        return logits.new_zeros(())
    if kind == "focal":
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probability = torch.sigmoid(logits)
        pt = torch.where(target > 0.5, probability, 1.0 - probability)
        alpha = torch.where(target > 0.5, logits.new_tensor(0.75),
                            logits.new_tensor(0.25))
        return (alpha * (1.0 - pt).pow(2.0) * bce).mean()
    positive = target > 0.5
    negative = ~positive
    weights = torch.ones_like(target)
    if positive.any():
        if source is None:
            weights[positive] = 0.5 / positive.float().sum().clamp_min(1.0)
        else:
            main = positive & (source == 0)
            reserve = positive & (source == 1)
            if main.any():
                weights[main] = 0.5 / main.float().sum().clamp_min(1.0)
            if reserve.any():
                # Reserve positives get equal source mass, not a capped scalar
                # positive weight that still lets the main source dominate.
                weights[reserve] = 0.5 / reserve.float().sum().clamp_min(1.0)
    if negative.any():
        weights[negative] = 0.5 / negative.float().sum().clamp_min(1.0)
    return (weights * F.binary_cross_entropy_with_logits(
        logits, target, reduction="none")).sum()


def ranking_loss(logits: torch.Tensor, target: torch.Tensor,
                 source: torch.Tensor | None = None, margin: float = 0.20):
    positive = logits[target > 0.5]
    negative = logits[target <= 0.5]
    if not len(positive) or not len(negative):
        return logits.new_zeros(())
    negative = torch.topk(negative, min(len(negative), max(8, 3 * len(positive)))).values
    loss = F.softplus(margin - positive[:, None] + negative[None, :]).mean()
    if source is not None:
        reserve_positive = logits[(target > 0.5) & (source == 1)]
        main_negative = logits[(target <= 0.5) & (source == 0)]
        if len(reserve_positive) and len(main_negative):
            hard = torch.topk(main_negative,
                              min(len(main_negative), max(4, len(reserve_positive)))).values
            loss = 0.5 * loss + 0.5 * F.softplus(
                margin - reserve_positive[:, None] + hard[None, :]).mean()
    return loss


def coverage_loss(logits: torch.Tensor, state: int,
                  class_weights: torch.Tensor) -> torch.Tensor:
    target = torch.as_tensor([state], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits.reshape(1, -1), target,
                           weight=class_weights.to(logits.device))


def run_episode(model: L19UngatedRetriever, query, family, tokens, mask,
                episode: list[dict], burn_in: int, bptt_chunk: int,
                loss_mode: str, class_weights: torch.Tensor,
                collect: bool = False):
    state = {}
    last_seen = {}
    query_context = model.query_context(tokens, query, family, mask)
    chunk_losses, frame_losses = [], []
    stats = Counter()
    for index, row in enumerate(episode):
        frame = row["frame"]
        if index < int(burn_in):
            with torch.no_grad():
                output = model(row["features"], query, family, row["track_ids"],
                               state, query_tokens=tokens, query_mask=mask,
                               query_context=query_context)
            state = output["state"]
            last_seen.update({int(track_id): index for track_id in row["track_ids"].detach().cpu().tolist()})
            state = {key: value for key, value in state.items()
                     if index - last_seen.get(int(key), index) <= 12}
            continue
        output = model(row["features"], query, family, row["track_ids"], state,
                       query_tokens=tokens, query_mask=mask,
                       query_context=query_context)
        state = output["state"]
        for track_id in row["track_ids"].detach().cpu().tolist():
            last_seen[int(track_id)] = index
        state = {key: value for key, value in state.items()
                 if index - last_seen.get(int(key), index) <= 12}
        membership = balanced_bce(output["membership_logits"], row["membership"],
                                  row["source"], loss_mode)
        presence = balanced_bce(output["presence_logits"], row["presence"],
                                None, loss_mode)
        coverage = coverage_loss(output["coverage_logits"], row["state"],
                                 class_weights)
        final = balanced_bce(output["logits"], row["membership"],
                             row["source"], loss_mode)
        rank = ranking_loss(output["logits"], row["membership"], row["source"])
        value = (0.65 * membership + 0.55 * presence + 0.25 * coverage +
                 0.85 * final + 0.35 * rank)
        frame_losses.append(value)
        stats["membership"] += float(membership.detach())
        stats["presence"] += float(presence.detach())
        stats["coverage"] += float(coverage.detach())
        stats["final"] += float(final.detach())
        stats["rank"] += float(rank.detach())
        stats["frames"] += 1
        stats["reserve_positive"] += int(((row["membership"] > 0.5) &
                                           (row["source"] == 1)).sum())
        stats["main_positive"] += int(((row["membership"] > 0.5) &
                                       (row["source"] == 0)).sum())
        stats["present_uncovered"] += int(row["state"] == 3)
        stats[f"state_{row['state']}"] += 1
        if len(frame_losses) >= int(bptt_chunk):
            chunk_losses.append(torch.stack(frame_losses).mean())
            frame_losses = []
            state = detach_state(state)
    if frame_losses:
        chunk_losses.append(torch.stack(frame_losses).mean())
    if not chunk_losses:
        chunk_losses = [next(model.parameters()).sum() * 0.0]
    denominator = max(1, int(stats["frames"]))
    for key in list(stats):
        if key not in {"frames", "reserve_positive", "main_positive",
                       "present_uncovered"} and not key.startswith("state_"):
            stats[key] /= denominator
    return torch.stack(chunk_losses).mean(), chunk_losses, stats


@torch.no_grad()
def proxy_validate(model, items: list[dict], store: BankStore,
                   text_store: TextStore, rng: random.Random, episodes: int,
                   sequence_length: int, burn_in: int, bptt_chunk: int,
                   device: torch.device, class_weights: torch.Tensor) -> dict:
    model.eval()
    losses, counts = [], Counter()
    for _ in range(int(episodes)):
        item = rng.choice(items)
        item, query, family, tokens, mask, episode, _ = item_episode(
            item, store, text_store, rng, sequence_length, burn_in, device)
        loss, _chunks, stats = run_episode(
            model, query, family, tokens, mask, episode, burn_in, bptt_chunk,
            "balanced", class_weights)
        losses.append(float(loss))
        counts.update(stats)
    return {"loss": float(np.mean(losses)) if losses else None,
            "episodes": int(episodes), "stats": dict(counts)}


def gradient_check(model: L19UngatedRetriever, device: torch.device) -> dict:
    """Check that frame t has a nonzero gradient through frame t-1 state."""
    model.zero_grad(set_to_none=True)
    n = 2
    features = {
        "clip": torch.randn(n, 512, device=device),
        "history_clip": torch.randn(n, 512, device=device),
        "pbd": torch.randn(n, 2048, device=device),
        "uidm_h": torch.randn(n, 384, device=device),
        "geometry": torch.randn(n, 7, device=device),
        "motion": torch.randn(n, 8, device=device),
        "context": torch.randn(n, 8, device=device),
        "lifecycle": torch.randn(n, 8, device=device),
        "objectness": torch.rand(n, device=device),
        "pool_id": torch.tensor([0, 1], dtype=torch.long, device=device),
    }
    query = torch.randn(512, device=device)
    family = torch.zeros(8, device=device)
    ids = torch.tensor([11, 12], dtype=torch.long, device=device)
    first = model(features, query, family, ids, {})
    for value in first["state"].values():
        value.retain_grad()
    second = model(features, query, family, ids, first["state"])
    second["logits"].sum().backward()
    gradients = [value.grad for value in first["state"].values()
                 if value.grad is not None]
    norm = float(sum(value.abs().sum() for value in gradients)) if gradients else 0.0
    model.zero_grad(set_to_none=True)
    return {"nonzero": bool(norm > 1e-9), "state_gradient_l1": norm,
            "checked_tracks": len(gradients)}


def save_checkpoint(path: Path, model, optimizer, scheduler, args, protocol,
                    step: int, validation: dict | None, sampler_stats: dict,
                    gradient: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(), "model_name": "l19",
        "cfg": {"hidden": args.hidden, "heads": args.heads,
                "sequence_length": args.sequence_length, "burn_in": args.burn_in,
                "bptt_chunk": args.bptt_chunk, "seed": args.seed,
                "use_slots": args.use_slots, "holistic_only": args.holistic_only,
                "coverage_mode": args.coverage_mode, "loss_mode": args.loss_mode,
                "reserve_identity_bank": str((ROOT / args.bank_root).resolve()),
                "balanced_sampling": True, "true_query_swap": args.swap_prob > 0},
        "protocol": protocol, "step": int(step), "validation_proxy": validation,
        "bank_root": str((ROOT / args.bank_root).resolve()),
        "text_root": str((ROOT / args.text_root).resolve()),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "sampler_stats": sampler_stats, "temporal_gradient_check": gradient,
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(checkpoint, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/l19/checkpoints/l19_ungated_balanced.pt")
    parser.add_argument("--bank-root", default="outputs/l19/dual_banks")
    parser.add_argument("--text-root", default="outputs/l18/data/text_cache")
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--burn-in", type=int, default=24)
    parser.add_argument("--bptt-chunk", type=int, default=24)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cache-size", type=int, default=2)
    parser.add_argument("--swap-prob", type=float, default=0.20)
    parser.add_argument("--loss-mode", choices=("balanced", "focal"), default="balanced")
    parser.add_argument("--use-slots", action="store_true")
    parser.add_argument("--no-holistic-only", dest="holistic_only", action="store_false")
    parser.set_defaults(holistic_only=True)
    parser.add_argument("--coverage-mode", choices=("aux_only", "soft_residual"),
                        default="aux_only")
    parser.add_argument("--resume", default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.sequence_length <= args.burn_in or args.bptt_chunk < 1:
        raise ValueError("sequence length must exceed burn-in and chunk must be positive")
    if args.smoke:
        args.steps = min(args.steps, 150)
        args.validate_every = min(args.validate_every, 50)
        args.validation_episodes = min(args.validation_episodes, 8)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(0)
    items, protocol = load_items()
    store = BankStore((ROOT / args.bank_root).resolve(), args.cache_size)
    text_store = TextStore((ROOT / args.text_root).resolve())
    buckets, bucket_metadata = build_item_buckets(items["train"], store)
    by_video = defaultdict(list)
    for domain in ("kitti", "dance"):
        for item in items["train"][domain]:
            by_video[item["video"]].append(item)
    if not buckets.get("reserve_positive") or not buckets.get("present_uncovered"):
        raise RuntimeError("balanced L19 buckets lack reserve/uncovered examples")
    model = L19UngatedRetriever(
        args.hidden, args.heads, use_slots=args.use_slots,
        holistic_only=args.holistic_only, coverage_mode=args.coverage_mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    updates = max(1, args.steps // args.accumulate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates, eta_min=args.lr * 0.05)
    start_step = 0
    if args.resume:
        checkpoint = torch.load(Path(args.resume), map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_step = int(checkpoint.get("step", 0))
    gradient = gradient_check(model, device)
    class_weights = torch.tensor([0.35, 1.0, 4.0, 4.0], dtype=torch.float32,
                                 device=device)
    sampler_stats = {"bucket_counts": {key: len(value) for key, value in buckets.items()},
                     "sampled": Counter(), "swap_samples": [],
                     "gradient_check": gradient}
    rng = random.Random(args.seed + 1009)
    history = []
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        item, sampled_bucket = choose_item(items["train"], buckets, rng)
        sampler_stats["sampled"][sampled_bucket] += 1
        swap_info = None
        original_item = item
        if args.swap_prob > 0 and rng.random() < args.swap_prob:
            item, swap_info = choose_verified_swap(item, by_video, rng)
            if swap_info is not None:
                sampler_stats["sampled"]["true_query_swap"] += 1
                if len(sampler_stats["swap_samples"]) < 64:
                    sampler_stats["swap_samples"].append(swap_info)
        item, query, family, tokens, mask, episode, mode = item_episode(
            item, store, text_store, rng, args.sequence_length, args.burn_in, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            loss, chunks, stats = run_episode(
                model, query, family, tokens, mask, episode, args.burn_in,
                args.bptt_chunk, args.loss_mode, class_weights)
            scaled = loss / max(1, args.accumulate)
        scaled.backward()
        if step % args.accumulate == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        row = {"step": step, "loss": float(loss.detach()),
               "bucket": sampled_bucket, "mode": mode,
               "query": expression_text(item["entry"]),
               "original_query": expression_text(original_item["entry"]),
               "chunks": len(chunks), "stats": dict(stats)}
        history.append(row)
        if step == 1 or step % 25 == 0:
            print(f"[l19-train] step={step}/{args.steps} loss={row['loss']:.4f} "
                  f"bucket={sampled_bucket} reserve+={int(stats['reserve_positive'])} "
                  f"uncovered={int(stats['present_uncovered'])}", flush=True)
        validation = None
        if step % args.validate_every == 0 or step == args.steps:
            validation = proxy_validate(
                model, items["train_val"]["kitti"] + items["train_val"]["dance"],
                store, text_store, random.Random(args.seed + step),
                args.validation_episodes, args.sequence_length, args.burn_in,
                args.bptt_chunk, device, class_weights)
            row["validation"] = validation
            out = (ROOT / args.out).resolve()
            save_checkpoint(out.with_name(f"{out.stem}_step{step}.pt"), model,
                            optimizer, scheduler, args, protocol, step,
                            validation, sampler_stats, gradient)
            print(f"[l19-val] step={step} loss={validation['loss']:.5f} "
                  f"stats={validation['stats']}", flush=True)
    out = (ROOT / args.out).resolve()
    save_checkpoint(out, model, optimizer, scheduler, args, protocol, args.steps,
                    history[-1].get("validation") if history else None,
                    sampler_stats, gradient)
    serial_sampler = dict(sampler_stats)
    serial_sampler["sampled"] = dict(serial_sampler["sampled"])
    out.with_suffix(out.suffix + ".json").write_text(json.dumps({
        "args": vars(args), "steps": args.steps,
        "wall_seconds": time.time() - started,
        "bucket_metadata_count": len(bucket_metadata),
        "sampler_stats": serial_sampler, "history": history,
    }, indent=2) + "\n")
    print(f"[l19-train] done seconds={time.time()-started:.1f} "
          f"gradient={gradient}", flush=True)


if __name__ == "__main__":
    main()
