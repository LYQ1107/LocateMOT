"""Train the Stage L19 learned FlexHook-style correspondence model.

The input bank is frozen: LocalAnything supplies the main observations and
the L19 GroundingDINO reserve supplies high-recall observations.  The only
learned object here is expression-to-tracklet correspondence plus a temporal
association embedding.  This diagnostic intentionally does not instantiate
the L18 CARR coverage head or any heuristic linker.
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

from locatemot.models.l19_flexhook_correspondence import (  # noqa: E402
    L19FlexHookCorrespondence,
)
from tools.train_l18_carr import BankStore, TextStore, load_items  # noqa: E402
from tools.train_l19 import (  # noqa: E402
    balanced_bce,
    build_item_buckets,
    choose_item,
    choose_verified_swap,
    detach_state,
    item_episode,
    ranking_loss,
)


def expression_text(entry: dict) -> str:
    return str(entry.get("sentence", entry.get("expression", "")))


def detach_association(state: dict[int, torch.Tensor]) -> dict[int, torch.Tensor]:
    return {key: value.detach() for key, value in state.items()}


def association_loss(output: dict, row: dict,
                     previous: dict[int, torch.Tensor]) -> torch.Tensor:
    """Temporal same-track consistency without a hand-written linker.

    Track IDs are supplied by the frozen LocalAnything/DINO bank.  The model
    learns the embedding used for this consistency term; no box IoU or CLIP
    similarity enters the association objective.
    """
    embedding = output.get("association_embedding")
    if embedding is None or not len(embedding):
        return output["logits"].new_zeros(())
    terms = []
    ids = row["track_ids"].detach().cpu().tolist()
    for index, raw_id in enumerate(ids):
        track_id = int(raw_id)
        if track_id in previous:
            terms.append(1.0 - F.cosine_similarity(
                embedding[index:index + 1], previous[track_id].reshape(1, -1),
                dim=-1).squeeze(0))
        previous[track_id] = embedding[index]
    if not terms:
        return output["logits"].new_zeros(())
    return torch.stack(terms).mean()


def run_episode(model: L19FlexHookCorrespondence, query, family, tokens, mask,
                episode: list[dict], burn_in: int, bptt_chunk: int,
                loss_mode: str, collect: bool = False):
    state = {}
    previous_association = {}
    last_seen = {}
    query_context = model.query_context(tokens, query, family, mask)
    chunk_losses, frame_losses = [], []
    stats = Counter()
    for index, row in enumerate(episode):
        if index < int(burn_in):
            with torch.no_grad():
                output = model(
                    row["features"], query, family, row["track_ids"], state,
                    query_tokens=tokens, query_mask=mask,
                    query_context=query_context,
                )
            state = output["state"]
            last_seen.update({
                int(track_id): index
                for track_id in row["track_ids"].detach().cpu().tolist()
            })
            state = {
                key: value for key, value in state.items()
                if index - last_seen.get(int(key), index) <= 12
            }
            previous_association = {
                key: value for key, value in previous_association.items()
                if index - last_seen.get(int(key), index) <= 12
            }
            continue

        output = model(
            row["features"], query, family, row["track_ids"], state,
            query_tokens=tokens, query_mask=mask,
            query_context=query_context,
        )
        state = output["state"]
        last_seen.update({
            int(track_id): index
            for track_id in row["track_ids"].detach().cpu().tolist()
        })
        state = {
            key: value for key, value in state.items()
            if index - last_seen.get(int(key), index) <= 12
        }
        previous_association = {
            key: value for key, value in previous_association.items()
            if index - last_seen.get(int(key), index) <= 12
        }

        source = row["source"]
        track_target = row["membership"]
        observation_target = row["current_match"]
        track = balanced_bce(
            output["membership_logits"], track_target, source, loss_mode)
        observation = balanced_bce(
            output["observation_logits"], observation_target, source, loss_mode)
        final = balanced_bce(
            output["logits"], observation_target, source, loss_mode)
        rank = ranking_loss(
            output["logits"], observation_target, source, margin=0.20)
        presence = balanced_bce(
            output["presence_logits"], row["presence"], None, loss_mode)
        association = association_loss(output, row, previous_association)
        value = (0.45 * track + 0.65 * observation + 0.85 * final +
                 0.30 * rank + 0.15 * association + 0.10 * presence)
        frame_losses.append(value)
        stats["track"] += float(track.detach())
        stats["observation"] += float(observation.detach())
        stats["final"] += float(final.detach())
        stats["rank"] += float(rank.detach())
        stats["association"] += float(association.detach())
        stats["presence"] += float(presence.detach())
        stats["frames"] += 1
        stats["main_positive"] += int(((observation_target > 0.5) &
                                        (source == 0)).sum())
        stats["reserve_positive"] += int(((observation_target > 0.5) &
                                           (source == 1)).sum())
        stats["membership_positive"] += int((track_target > 0.5).sum())
        stats[f"state_{row['state']}"] += 1
        if len(frame_losses) >= int(bptt_chunk):
            chunk_losses.append(torch.stack(frame_losses).mean())
            frame_losses = []
            state = detach_state(state)
            previous_association = detach_association(previous_association)
    if frame_losses:
        chunk_losses.append(torch.stack(frame_losses).mean())
    if not chunk_losses:
        chunk_losses = [next(model.parameters()).sum() * 0.0]
    denominator = max(1, int(stats["frames"]))
    for key in ("track", "observation", "final", "rank", "association", "presence"):
        stats[key] /= denominator
    return torch.stack(chunk_losses).mean(), chunk_losses, stats


@torch.no_grad()
def proxy_validate(model: L19FlexHookCorrespondence, items: list[dict],
                   store: BankStore, text_store: TextStore, rng: random.Random,
                   episodes: int, sequence_length: int, burn_in: int,
                   bptt_chunk: int, device: torch.device) -> dict:
    model.eval()
    losses, counts = [], Counter()
    for _ in range(int(episodes)):
        item = rng.choice(items)
        item, query, family, tokens, mask, episode, _ = item_episode(
            item, store, text_store, rng, sequence_length, burn_in, device)
        loss, _chunks, stats = run_episode(
            model, query, family, tokens, mask, episode, burn_in, bptt_chunk,
            "balanced")
        losses.append(float(loss))
        counts.update(stats)
    return {
        "loss": float(np.mean(losses)) if losses else None,
        "episodes": int(episodes), "stats": dict(counts),
    }


def gradient_check(model: L19FlexHookCorrespondence,
                   device: torch.device) -> dict:
    """Verify gradients cross one causal state transition."""
    previous_detach = model.detach_state
    model.detach_state = False
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
    tokens = torch.randn(77, 512, device=device)
    mask = torch.ones(77, dtype=torch.bool, device=device)
    context = model.query_context(tokens, query, family, mask)
    first = model(features, query, family, ids, {}, query_tokens=tokens,
                  query_mask=mask, query_context=context)
    for value in first["state"].values():
        value["memory"].retain_grad()
    second = model(features, query, family, ids, first["state"],
                   query_tokens=tokens, query_mask=mask,
                   query_context=context)
    second["logits"].sum().backward()
    gradients = [value["memory"].grad for value in first["state"].values()
                 if value["memory"].grad is not None]
    norm = float(sum(value.abs().sum() for value in gradients)) if gradients else 0.0
    model.zero_grad(set_to_none=True)
    model.detach_state = previous_detach
    return {
        "nonzero": bool(norm > 1e-9), "state_gradient_l1": norm,
        "checked_tracks": len(gradients),
    }


def save_checkpoint(path: Path, model, optimizer, scheduler, args, protocol,
                    step: int, validation: dict | None, sampler_stats: dict,
                    gradient: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "model_name": "l19_flexhook_correspondence",
        "cfg": {
            "hidden": args.hidden, "heads": args.heads,
            "dropout": args.dropout, "token_dim": 512,
            "temporal_points": args.temporal_points,
            "hook_points": args.hook_points,
            "sequence_length": args.sequence_length,
            "burn_in": args.burn_in, "bptt_chunk": args.bptt_chunk,
            "seed": args.seed, "loss_mode": args.loss_mode,
            "reserve_identity_bank": str((ROOT / args.bank_root).resolve()),
            "learned_correspondence": True,
            "c_hook_pcd": True, "temporal_feature_map_sampling": True,
            "carr_coverage_gate": False, "heuristic_linker": False,
        },
        "protocol": protocol, "step": int(step),
        "validation_proxy": validation,
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
    parser.add_argument("--out", default=
                        "outputs/l19/checkpoints/l19_flexhook_correspondence_diag.pt")
    parser.add_argument("--bank-root", default="outputs/l19/dual_banks_features")
    parser.add_argument("--text-root", default="outputs/l18/data/text_cache")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--burn-in", type=int, default=24)
    parser.add_argument("--bptt-chunk", type=int, default=24)
    parser.add_argument("--temporal-points", type=int, default=8)
    parser.add_argument("--hook-points", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--cache-size", type=int, default=1)
    parser.add_argument("--swap-prob", type=float, default=0.20)
    parser.add_argument("--loss-mode", choices=("balanced", "focal"),
                        default="balanced")
    args = parser.parse_args()
    if args.sequence_length <= args.burn_in or args.bptt_chunk < 1:
        raise ValueError("sequence length must exceed burn-in and chunk positive")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(0)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    items, protocol = load_items()
    store = BankStore((ROOT / args.bank_root).resolve(), args.cache_size)
    text_store = TextStore((ROOT / args.text_root).resolve())
    buckets, bucket_metadata = build_item_buckets(items["train"], store)
    by_video = defaultdict(list)
    for domain in ("kitti", "dance"):
        for item in items["train"][domain]:
            by_video[item["video"]].append(item)
    if not buckets.get("reserve_positive") or not buckets.get("present_uncovered"):
        raise RuntimeError("correspondence buckets lack reserve/uncovered examples")

    model = L19FlexHookCorrespondence(
        args.hidden, args.heads, dropout=args.dropout,
        temporal_points=args.temporal_points, hook_points=args.hook_points,
    ).to(device)
    model.detach_state = False
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    updates = max(1, args.steps // args.accumulate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates, eta_min=args.lr * 0.05)
    gradient = gradient_check(model, device)
    sampler_stats = {
        "bucket_counts": {key: len(value) for key, value in buckets.items()},
        "sampled": Counter(), "swap_samples": [],
        "gradient_check": gradient,
    }
    rng = random.Random(args.seed + 1009)
    history = []
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        model.train()
        item, sampled_bucket = choose_item(items["train"], buckets, rng)
        sampler_stats["sampled"][sampled_bucket] += 1
        original_item = item
        swap_info = None
        if args.swap_prob > 0 and rng.random() < args.swap_prob:
            item, swap_info = choose_verified_swap(item, by_video, rng)
            if swap_info is not None:
                sampler_stats["sampled"]["true_query_swap"] += 1
                if len(sampler_stats["swap_samples"]) < 64:
                    sampler_stats["swap_samples"].append(swap_info)
        item, query, family, tokens, mask, episode, mode = item_episode(
            item, store, text_store, rng, args.sequence_length, args.burn_in,
            device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            loss, chunks, stats = run_episode(
                model, query, family, tokens, mask, episode, args.burn_in,
                args.bptt_chunk, args.loss_mode)
            (loss / max(1, args.accumulate)).backward()
        if step % args.accumulate == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        row = {
            "step": step, "loss": float(loss.detach()),
            "bucket": sampled_bucket, "mode": mode,
            "query": expression_text(item["entry"]),
            "original_query": expression_text(original_item["entry"]),
            "chunks": len(chunks), "stats": dict(stats),
            "true_query_swap": swap_info is not None,
        }
        history.append(row)
        if step == 1 or step % 25 == 0:
            print(
                f"[l19-correspondence] step={step}/{args.steps} "
                f"loss={row['loss']:.4f} bucket={sampled_bucket} "
                f"main+={int(stats['main_positive'])} "
                f"reserve+={int(stats['reserve_positive'])}", flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            validation = proxy_validate(
                model, items["train_val"]["kitti"] + items["train_val"]["dance"],
                store, text_store, random.Random(args.seed + step),
                args.validation_episodes, args.sequence_length, args.burn_in,
                args.bptt_chunk, device)
            row["validation"] = validation
            out = (ROOT / args.out).resolve()
            save_checkpoint(
                out.with_name(f"{out.stem}_step{step}.pt"), model, optimizer,
                scheduler, args, protocol, step, validation, sampler_stats,
                gradient)
            print(
                f"[l19-correspondence-val] step={step} "
                f"loss={validation['loss']:.5f} stats={validation['stats']}",
                flush=True)
    out = (ROOT / args.out).resolve()
    save_checkpoint(
        out, model, optimizer, scheduler, args, protocol, args.steps,
        history[-1].get("validation") if history else None, sampler_stats,
        gradient)
    serial_sampler = dict(sampler_stats)
    serial_sampler["sampled"] = dict(serial_sampler["sampled"])
    out.with_suffix(out.suffix + ".json").write_text(json.dumps({
        "args": vars(args), "steps": args.steps,
        "wall_seconds": time.time() - started,
        "bucket_metadata_count": len(bucket_metadata),
        "sampler_stats": serial_sampler, "history": history,
    }, indent=2) + "\n")
    print(
        f"[l19-correspondence] done seconds={time.time() - started:.1f} "
        f"gradient={gradient}", flush=True)


if __name__ == "__main__":
    main()
