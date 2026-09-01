"""Train the L18 controlled FlexHook port or CARR on frozen banks."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.flexhook_bank_port import FlexHookBankPort  # noqa: E402
from locatemot.models.l18_coverage_retrieve_repair import (  # noqa: E402
    L18CARRRetriever, STATE_NAMES,
)
from locatemot.models.l16_track_selector import expression_family_vector  # noqa: E402


PROTOCOL = ROOT / "outputs/l16/data/protocol/split_manifest.json"
KITTI_OLD = ROOT / "outputs/l11/data/rmot_kitti/expressions.json"
KITTI_NEW = ROOT / "outputs/l16/data/kitti_missing/records/expressions.json"
DANCE = ROOT / "outputs/l16/data/protocol/refer_dance_expressions.json"
L16_ROOT = ROOT / "outputs/l16/track_banks_dedup"
L16_BASE_ROOT = ROOT / "outputs/l16/track_banks"
FEATURE_NAMES = (
    "clip", "history_clip", "pbd", "uidm_h", "geometry", "motion",
    "context", "lifecycle", "objectness", "pool_id",
)


class TextStore:
    def __init__(self, root: Path):
        self.sentences = json.loads((root / "sentences.json").read_text())
        self.index = {text: i for i, text in enumerate(self.sentences)}
        self.tokens = np.load(root / "tokens.npy", mmap_mode="r")
        self.masks = np.load(root / "masks.npy", mmap_mode="r")

    def get(self, text: str, device: torch.device):
        index = self.index.get(str(text))
        if index is None:
            return (torch.zeros((77, 512), device=device),
                    torch.ones(77, dtype=torch.bool, device=device))
        return (torch.as_tensor(np.array(self.tokens[index], copy=True),
                                device=device),
                torch.as_tensor(np.array(self.masks[index], copy=True),
                                device=device))


class BankStore:
    def __init__(self, root: Path, cache_size: int = 2):
        self.root = root
        self.cache_size = int(cache_size)
        self.cache = OrderedDict()

    def get(self, dataset: str, video: str) -> dict:
        key = (dataset, video)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        path = self.root / dataset / f"{video}.pt"
        if not path.exists():
            fallback = L16_ROOT / dataset / f"{video}.pt"
            if not fallback.exists():
                fallback = L16_BASE_ROOT / dataset / f"{video}.pt"
            if not fallback.exists() and dataset == "dance_eval":
                fallback = L16_BASE_ROOT / "dance_train" / f"{video}.pt"
            if not fallback.exists():
                raise FileNotFoundError(path)
            path = fallback
        bank = torch.load(path, map_location="cpu", weights_only=False)
        labels_path = path.with_suffix(".labels.json")
        if labels_path.exists():
            labels = json.loads(labels_path.read_text())["candidate_gt"]
        else:
            labels = [None] * len(bank["tensors"]["track_id"])
        bank["candidate_gt"] = labels
        bank["frame_to_index"] = {
            int(frame): index
            for index, frame in enumerate(bank["tensors"]["frame_ids"].tolist())
        }
        self.cache[key] = bank
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return bank


def expression_text(entry: dict) -> str:
    return str(entry.get("sentence", entry.get("expression", "")))


def load_items():
    protocol = json.loads(PROTOCOL.read_text())
    kitti = json.loads(KITTI_OLD.read_text())
    kitti.update(json.loads(KITTI_NEW.read_text()))
    dance = json.loads(DANCE.read_text())
    split_items = defaultdict(lambda: defaultdict(list))
    for domain, metadata, split_map, bank_dataset in (
            ("kitti", kitti, protocol["kitti_v2"], "kitti"),
            ("dance", dance, protocol["refer_dance"], "dance_train")):
        lookup = {video: split for split in ("train", "train_val")
                  for video in split_map[split]}
        for video, entries in metadata.items():
            if video not in lookup:
                continue
            for entry in entries:
                labels = entry.get("label", {})
                item = {
                    "domain": domain, "bank_dataset": bank_dataset,
                    "video": video, "entry": entry,
                    "nonempty": any(bool(value) for value in labels.values()),
                }
                split_items[lookup[video]][domain].append(item)
    return split_items, protocol


def frame_features(bank: dict, frame_index: int, device: torch.device):
    tensors = bank["tensors"]
    start = int(tensors["frame_ptr"][frame_index])
    end = int(tensors["frame_ptr"][frame_index + 1])
    features = {}
    for name in FEATURE_NAMES:
        if name in tensors:
            features[name] = tensors[name][start:end].to(
                device, non_blocking=True)
        elif name == "pool_id":
            features[name] = torch.zeros(end - start, dtype=torch.long,
                                         device=device)
    return features, tensors["track_id"][start:end].to(device), start, end


def frame_target(bank: dict, begin: int, end: int, entry: dict,
                 frame_id: int):
    labels = entry.get("label", {})
    target_ids = {str(value) for value in labels.get(
        str(frame_id), labels.get(frame_id, []))}
    candidate_gt = bank["candidate_gt"][begin:end]
    target = np.asarray([
        float(value is not None and str(value) in target_ids)
        for value in candidate_gt
    ], np.float32)
    pool = bank["tensors"].get("pool_id")
    if pool is None:
        pool_values = np.zeros(end - begin, np.int64)
    else:
        pool_values = pool[begin:end].numpy()
    main_covered = bool(np.any((target > 0.5) & (pool_values == 0)))
    reserve_covered = bool(np.any((target > 0.5) & (pool_values == 1)))
    if not target_ids:
        state = 0
    elif main_covered:
        state = 1
    elif reserve_covered:
        state = 2
    else:
        state = 3
    return target, state, bool(target_ids), main_covered, reserve_covered


def l19_track_membership_index(bank: dict) -> dict[int, set[str]]:
    """Build query-independent track-to-GT history for L19 supervision.

    The L18 ``frame_target`` is intentionally retained.  L19 membership is
    different: a candidate is positive when its causal track has carried one
    of the query's identities anywhere in the bank, while presence is only the
    current observation's valid-GT flag.  The split is used by the new L19
    trainer and is harmless to the frozen L18 path.
    """
    labels = bank.get("candidate_gt", [])
    tensors = bank["tensors"]
    track_ids = tensors["track_id"].tolist()
    result = defaultdict(set)
    for track_id, label in zip(track_ids, labels):
        if label is not None:
            result[int(track_id)].add(str(label))
    return result


def l19_frame_targets(bank: dict, begin: int, end: int, entry: dict,
                      frame_id: int,
                      track_membership: dict[int, set[str]] | None = None):
    """Return decoupled membership/presence/coverage targets for one frame."""
    labels = entry.get("label", {})
    target_ids = {str(value) for value in labels.get(
        str(frame_id), labels.get(frame_id, []))}
    candidate_gt = bank.get("candidate_gt", [None] * len(
        bank["tensors"]["track_id"]))[begin:end]
    tensors = bank["tensors"]
    track_ids = tensors["track_id"][begin:end].tolist()
    source = tensors.get("pool_id")
    source = (source[begin:end].numpy() if source is not None else
              np.zeros(end - begin, np.int64))
    if track_membership is None:
        track_membership = l19_track_membership_index(bank)
    membership = np.asarray([
        float(bool(target_ids.intersection(track_membership.get(
            int(track_id), set())))) for track_id in track_ids], np.float32)
    # Presence is query-independent: all currently valid GT-overlapping
    # observations are positive, including distractors for this expression.
    presence = np.asarray([float(value is not None)
                           for value in candidate_gt], np.float32)
    current_match = np.asarray([
        float(value is not None and str(value) in target_ids)
        for value in candidate_gt], np.float32)
    main_covered = bool(np.any((current_match > 0.5) & (source == 0)))
    reserve_covered = bool(np.any((current_match > 0.5) & (source == 1)))
    if not target_ids:
        state = 0
    elif main_covered:
        state = 1
    elif reserve_covered:
        state = 2
    else:
        state = 3
    groups = tensors.get("observation_group_id")
    group_values = (groups[begin:end].numpy().astype(np.int64)
                    if groups is not None else
                    np.arange(begin, end, dtype=np.int64))
    return {
        "membership": membership, "presence": presence,
        "current_match": current_match, "state": state,
        "active": bool(target_ids), "source": source.astype(np.int64),
        "group": group_values,
        "main_covered": main_covered, "reserve_covered": reserve_covered,
        "target_ids": sorted(target_ids),
    }


def item_episode(item: dict, store: BankStore, text_store: TextStore,
                  rng: random.Random, sequence_length: int, burn_in: int,
                  device: torch.device):
    bank = store.get(item["bank_dataset"], item["video"])
    entry = item["entry"]
    frame_count = len(bank["tensors"]["frame_ids"])
    loss_length = max(1, int(sequence_length) - int(burn_in))
    total_length = min(frame_count, int(sequence_length))
    labels = entry.get("label", {})
    positive_frames = [
        bank["frame_to_index"][int(frame)]
        for frame, ids in labels.items()
        if ids and int(frame) in bank["frame_to_index"]
    ]
    if positive_frames and rng.random() < 0.70:
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
        features, track_ids, begin, end = frame_features(bank, frame_index, device)
        frame_id = int(bank["tensors"]["frame_ids"][frame_index])
        target, state, active, main, reserve = frame_target(
            bank, begin, end, entry, frame_id)
        episode.append((features, track_ids,
                        torch.as_tensor(target, device=device), state,
                        active, frame_id))
    return item, query, family, tokens, mask, episode, mode


def ranking_loss(logits: torch.Tensor, target: torch.Tensor, margin=0.25):
    positive = logits[target > 0.5]
    negative = logits[target <= 0.5]
    if not len(positive) or not len(negative):
        return logits.new_zeros(())
    negative = torch.topk(negative, min(len(negative), max(8, 2 * len(positive)))).values
    return F.softplus(margin - positive[:, None] + negative[None, :]).mean()


def membership_loss(logits, target, state):
    if not len(logits):
        return logits.new_zeros(())
    positive = target > 0.5
    pos = int(positive.sum())
    neg = max(1, int((~positive).sum()))
    weight = min(6.0, neg / max(1, pos)) if pos else 1.0
    weights = torch.where(positive, logits.new_tensor(weight),
                          logits.new_tensor(1.0))
    if state == 3:
        weights = weights * 0.25
    if state == 0:
        weights = weights * 0.50
    return (weights * F.binary_cross_entropy_with_logits(
        logits, target, reduction="none")).mean()


def run_episode(model, model_name, query, family, tokens, mask, episode,
                burn_in: int, use_coverage=True):
    state = {}
    query_context = model.query_context(tokens, query, family, mask)
    losses = []
    stats = defaultdict(float)
    previous = {}
    for index, (features, track_ids, target, state_target, _active, _frame) in enumerate(episode):
        if index < int(burn_in):
            with torch.no_grad():
                output = model(features, query, family, track_ids, state,
                               query_tokens=tokens, query_mask=mask,
                               query_context=query_context)
            state = output["state"]
            continue
        output = model(features, query, family, track_ids, state,
                       query_tokens=tokens, query_mask=mask,
                       query_context=query_context)
        state = output["state"]
        if model_name == "flexhook":
            m_loss = membership_loss(output["membership_logits"], target,
                                     state_target)
            rank = ranking_loss(output["membership_logits"], target)
            loss = m_loss + 0.30 * rank
            stats["membership"] += float(m_loss.detach())
            stats["rank"] += float(rank.detach())
        else:
            m_loss = membership_loss(output["membership_logits"], target,
                                     state_target)
            presence_target = target
            p_loss = F.binary_cross_entropy_with_logits(
                output["presence_logits"], presence_target)
            c_loss = F.cross_entropy(
                output["coverage_logits"].reshape(1, -1),
                torch.as_tensor([state_target], device=target.device))
            rank = ranking_loss(output["membership_logits"], target)
            # Train the score that is actually emitted by CARR.  The
            # auxiliary heads above alone leave coverage_scale,
            # presence_scale, reserve_bias, and the temperature effectively
            # uncalibrated at inference time.
            final_loss = membership_loss(output["logits"], target,
                                         state_target)
            final_rank = ranking_loss(output["logits"], target)
            consistency_terms = []
            for local, raw_id in enumerate(track_ids.detach().cpu().tolist()):
                if target[local] > 0.5 and int(raw_id) in previous:
                    consistency_terms.append(
                        (torch.sigmoid(output["membership_logits"][local]) -
                         previous[int(raw_id)]).abs())
                if target[local] > 0.5:
                    previous[int(raw_id)] = torch.sigmoid(
                        output["membership_logits"][local])
            temporal = torch.stack(consistency_terms).mean() \
                if consistency_terms else output["logits"].new_zeros(())
            loss = (0.65 * m_loss + 0.70 * p_loss + 0.80 * c_loss +
                    0.75 * final_loss + 0.30 * rank +
                    0.30 * final_rank + 0.20 * temporal)
            stats["membership"] += float(m_loss.detach())
            stats["presence"] += float(p_loss.detach())
            stats["coverage"] += float(c_loss.detach())
            stats["rank"] += float(rank.detach())
            stats["final"] += float(final_loss.detach())
            stats["temporal"] += float(temporal.detach())
            stats["state_%s" % STATE_NAMES[state_target]] += 1.0
        losses.append(loss)
    if not losses:
        # A short video can be entirely burn-in; keep a differentiable zero.
        losses = [next(model.parameters()).sum() * 0.0]
    loss = torch.stack(losses).mean()
    denominator = max(1, len(losses))
    for key in list(stats):
        stats[key] /= denominator
    return loss, stats


class EpisodeTrainer(nn.Module):
    def __init__(self, model, model_name):
        super().__init__()
        self.model = model
        self.model_name = model_name

    def forward(self, query, family, tokens, mask, episode, burn_in):
        loss, stats = run_episode(self.model, self.model_name, query, family,
                                  tokens, mask, episode, burn_in)
        values = torch.stack([
            loss.detach(), loss.new_tensor(stats.get("membership", 0.0)),
            loss.new_tensor(stats.get("presence", 0.0)),
            loss.new_tensor(stats.get("coverage", 0.0)),
            loss.new_tensor(stats.get("rank", 0.0)),
        ])
        return loss, values


@torch.no_grad()
def validate(model, model_name, items, store, text_store, seed, episodes,
             sequence_length, burn_in, device):
    model.eval()
    rng = random.Random(seed)
    losses = []
    state_counts = defaultdict(int)
    score_rows = []
    target_rows = []
    for _ in range(episodes):
        domain = "kitti" if rng.random() < 0.5 else "dance"
        pool = items[domain]
        item = pool[rng.randrange(len(pool))]
        item, query, family, tokens, mask, episode, _mode = item_episode(
            item, store, text_store, rng, sequence_length, burn_in, device)
        loss, _ = run_episode(model, model_name, query, family, tokens, mask,
                              episode, burn_in)
        losses.append(float(loss))
        for _f, _ids, target, state_target, _active, _frame in episode[burn_in:]:
            state_counts[STATE_NAMES[state_target]] += 1
            target_rows.append(target.detach().cpu())
    targets = torch.cat(target_rows) if target_rows else torch.zeros(0)
    return {
        "loss": float(np.mean(losses)) if losses else None,
        "episodes": episodes, "examples": int(len(targets)),
        "positive_rate": float(targets.mean()) if len(targets) else 0.0,
        "state_counts": dict(state_counts),
    }


def make_model(args, device):
    if args.model == "flexhook":
        return FlexHookBankPort(
            args.hidden, args.heads, token_dim=512,
            use_slots=not args.no_slots,
            holistic_only=args.holistic_only).to(device)
    return L18CARRRetriever(
        args.hidden, args.heads, token_dim=512,
        use_slots=not args.no_slots,
        holistic_only=args.holistic_only,
        use_coverage=not args.no_coverage).to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("flexhook", "carr"), required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bank-root", default="outputs/l18/dual_banks")
    parser.add_argument("--text-root", default="outputs/l18/data/text_cache")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--sequence-length", type=int, default=96)
    parser.add_argument("--burn-in", type=int, default=24)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--cache-size", type=int, default=2)
    parser.add_argument("--no-slots", action="store_true")
    parser.add_argument("--holistic-only", action="store_true")
    parser.add_argument("--no-coverage", action="store_true")
    args = parser.parse_args()
    if args.sequence_length < 64:
        raise ValueError("L18 formal training requires sequence length >=64")
    if args.burn_in >= args.sequence_length:
        raise ValueError("burn-in must be shorter than sequence length")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = torch.device("cuda", local_rank)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get(
            "L18_GPU", "1")
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    items, protocol = load_items()
    if not items["train"]["kitti"] or not items["train"]["dance"]:
        raise RuntimeError("joint train mixture is empty")
    store = BankStore((ROOT / args.bank_root).resolve(), args.cache_size)
    text_store = TextStore((ROOT / args.text_root).resolve())
    raw_model = make_model(args, device)
    wrapper = EpisodeTrainer(raw_model, args.model)
    model = DistributedDataParallel(
        wrapper, device_ids=[device.index],
        # Some episodes legitimately omit a branch (e.g. the FlexHook
        # control's presence head), so let DDP handle dynamic reachability.
        find_unused_parameters=True) \
        if distributed else wrapper
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    updates = max(1, args.steps // args.accumulate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates, eta_min=args.lr * 0.05)
    rng = random.Random(args.seed + 1009 * rank)
    out = (ROOT / args.out).resolve()
    if rank == 0:
        out.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    history = []
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        domain = "kitti" if rng.random() < 0.5 else "dance"
        pool = items["train"][domain]
        item = pool[rng.randrange(len(pool))]
        # Same-video query swaps are naturally hard negatives for the bank;
        # sample them explicitly without changing the dataset annotation.
        if rng.random() < 0.15:
            same_video = [x for x in pool if x["video"] == item["video"]]
            if same_video:
                item = same_video[rng.randrange(len(same_video))]
        item, query, family, tokens, mask, episode, mode = item_episode(
            item, store, text_store, rng, args.sequence_length, args.burn_in,
            device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            loss, values = model(query, family, tokens, mask, episode,
                                 args.burn_in)
            scaled = loss / args.accumulate
        scaled.backward()
        if step % args.accumulate == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        row = {"step": step, "loss": float(loss.detach()),
               "domain": domain, "sampling": mode}
        row.update({name: float(value) for name, value in zip(
            ("membership", "presence", "coverage", "rank"),
            values[1:].detach().cpu())})
        if rank == 0 and (step == 1 or step % 50 == 0):
            print(f"[l18-train] model={args.model} step={step}/{args.steps} "
                  f"loss={row['loss']:.4f} domain={domain}", flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            validation = validate(
                raw_model, args.model, items["train_val"], store, text_store,
                args.seed + step, args.validation_episodes,
                args.sequence_length, args.burn_in, device) if rank == 0 else None
            if distributed:
                values_obj = [validation]
                dist.broadcast_object_list(values_obj, src=0)
                validation = values_obj[0]
            row["validation"] = validation
            checkpoint = {
                "model": raw_model.state_dict(), "model_name": args.model,
                "cfg": {"hidden": args.hidden, "heads": args.heads,
                        "sequence_length": args.sequence_length,
                        "burn_in": args.burn_in, "seed": args.seed,
                        "world_size": world_size,
                        "no_slots": args.no_slots,
                        "holistic_only": args.holistic_only,
                        "no_coverage": args.no_coverage},
                "protocol": protocol, "validation_proxy": validation,
                "step": step, "bank_root": str((ROOT / args.bank_root).resolve()),
                "text_root": str((ROOT / args.text_root).resolve()),
            }
            if rank == 0:
                torch.save(checkpoint, out.with_name(f"{out.stem}_step{step}.pt"))
                print(f"[l18-val] model={args.model} step={step} "
                      f"loss={validation['loss']:.5f} states={validation['state_counts']}",
                      flush=True)
                if validation["loss"] is not None and validation["loss"] < best_loss:
                    best_loss = validation["loss"]
                    torch.save(checkpoint, out)
        history.append(row)
    if rank == 0:
        out.with_suffix(out.suffix + ".json").write_text(json.dumps({
            "args": vars(args), "world_size": world_size,
            "best_validation_proxy_loss": best_loss,
            "wall_seconds": time.time() - started, "history": history,
        }, indent=2) + "\n")
        print(f"[l18-train] done model={args.model} seconds={time.time()-started:.1f}",
              flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
