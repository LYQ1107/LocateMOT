"""Train the joint Stage L16 causal track selector on train-only banks."""
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
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l16_track_selector import (  # noqa: E402
    L16TrackSelector, expression_family_vector,
)

PROTOCOL = ROOT / "outputs/l16/data/protocol/split_manifest.json"
BANK_ROOT = Path(os.environ.get(
    "L16_BANK_ROOT", str(ROOT / "outputs/l16/track_banks_dedup")))
KITTI_OLD = ROOT / "outputs/l11/data/rmot_kitti/expressions.json"
KITTI_NEW = ROOT / "outputs/l16/data/kitti_missing/records/expressions.json"
DANCE = ROOT / "outputs/l16/data/protocol/refer_dance_expressions.json"
FEATURE_NAMES = (
    "clip", "history_clip", "pbd", "uidm_h", "geometry", "motion",
    "context", "lifecycle", "objectness",
)


class BankStore:
    def __init__(self, cache_size=64):
        self.cache_size = int(cache_size)
        self.cache = OrderedDict()

    def get(self, dataset, video):
        key = (dataset, video)
        if key not in self.cache:
            path = BANK_ROOT / dataset / f"{video}.pt"
            bank = torch.load(path, map_location="cpu", weights_only=False)
            labels = json.loads(path.with_suffix(".labels.json").read_text())[
                "candidate_gt"]
            frame_ids = bank["tensors"]["frame_ids"].tolist()
            bank["frame_to_index"] = {int(frame): index
                                      for index, frame in enumerate(frame_ids)}
            bank["candidate_gt"] = labels
            self.cache[key] = bank
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)
        else:
            self.cache.move_to_end(key)
        return self.cache[key]


def load_expressions():
    protocol = json.loads(PROTOCOL.read_text())
    kitti = json.loads(KITTI_OLD.read_text())
    for video, values in json.loads(KITTI_NEW.read_text()).items():
        kitti[video] = values
    dance = json.loads(DANCE.read_text())
    split_items = defaultdict(lambda: defaultdict(list))
    for domain, metadata, split_map, bank_dataset in (
        ("kitti", kitti, protocol["kitti_v2"], "kitti"),
        ("dance", dance, protocol["refer_dance"], "dance_train"),
    ):
        lookup = {video: split for split in ("train", "train_val")
                  for video in split_map[split]}
        for video, entries in metadata.items():
            if video not in lookup:
                continue
            for entry in entries:
                label = entry.get("label", {})
                nonempty = any(bool(ids) for ids in label.values())
                item = {
                    "domain": domain, "bank_dataset": bank_dataset,
                    "video": video, "entry": entry, "nonempty": nonempty,
                }
                split_items[lookup[video]][domain].append(item)
    return split_items, protocol


def frame_features(bank, frame_index, device):
    tensors = bank["tensors"]
    start = int(tensors["frame_ptr"][frame_index])
    end = int(tensors["frame_ptr"][frame_index + 1])
    features = {name: tensors[name][start:end].to(device, non_blocking=True)
                for name in FEATURE_NAMES}
    return features, tensors["track_id"][start:end].to(device), start, end


def sample_episode(items, store, rng, sequence_length, empty_probability,
                   device):
    domain = "kitti" if rng.random() < 0.5 else "dance"
    pool = items[domain]
    want_empty = rng.random() < empty_probability
    choices = [item for item in pool if item["nonempty"] != want_empty]
    if not choices:
        choices = pool
    item = choices[rng.randrange(len(choices))]
    entry = item["entry"]
    bank = store.get(item["bank_dataset"], item["video"])
    frame_count = len(bank["tensors"]["frame_ids"])
    labels = entry.get("label", {})
    positive_frames = [int(frame) for frame, ids in labels.items()
                       if ids and int(frame) in bank["frame_to_index"]]
    if positive_frames and not want_empty:
        anchor = bank["frame_to_index"][rng.choice(positive_frames)]
    else:
        anchor = rng.randrange(frame_count)
    offset = rng.randrange(sequence_length)
    start_index = max(0, min(frame_count - sequence_length, anchor - offset))
    end_index = min(frame_count, start_index + sequence_length)
    query = torch.as_tensor(np.asarray(entry["spec"], np.float32), device=device)
    text = entry.get("sentence", entry.get("expression", ""))
    family = expression_family_vector(text).to(device)
    episode = []
    for frame_index in range(start_index, end_index):
        features, track_ids, begin, end = frame_features(bank, frame_index, device)
        frame_id = int(bank["tensors"]["frame_ids"][frame_index])
        targets = {str(value) for value in labels.get(str(frame_id),
                                                       labels.get(frame_id, []))}
        candidate_gt = bank["candidate_gt"][begin:end]
        target = torch.tensor([
            1.0 if value is not None and str(value) in targets else 0.0
            for value in candidate_gt], device=device)
        episode.append((features, track_ids, target, frame_id))
    return item, query, family, episode


def grouped_loss(logits, target, margin=0.35, hard_negatives=True):
    """All ranking comparisons are confined to this one query/frame."""
    if not len(logits):
        zero = logits.new_zeros(())
        return zero, {"bce": zero, "rank": zero}
    positive = target > 0.5
    negatives = ~positive
    positive_count = int(positive.sum())
    negative_count = int(negatives.sum())
    positive_weight = min(12.0, negative_count / max(1, positive_count))
    weights = torch.where(positive, torch.full_like(target, positive_weight),
                          torch.ones_like(target))
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    pt = torch.where(positive, probability, 1.0 - probability)
    bce = (weights * (1.0 - pt).pow(1.5) * raw).mean()
    if positive_count and negative_count:
        pos = logits[positive]
        neg = logits[negatives]
        if hard_negatives:
            neg = torch.topk(neg, min(len(neg), max(4, 2 * len(pos)))).values
        rank = F.softplus(margin - pos[:, None] + neg[None, :]).mean()
    else:
        rank = logits.new_zeros(())
    return bce + 0.45 * rank, {"bce": bce.detach(), "rank": rank.detach()}


def run_episode(model, query, family, episode, use_belief=True,
                use_cross_track=True, use_motion=True, hard_negatives=True):
    state = {}
    total = query.new_zeros(())
    groups = 0
    stats = defaultdict(float)
    previous = {}
    predictions, targets = [], []
    for features, track_ids, target, _frame_id in episode:
        output = model(
            features, query, family, track_ids, state,
            use_belief=use_belief, use_cross_track=use_cross_track,
            use_motion=use_motion)
        state = output["state"]
        logits = output["logits"]
        loss, parts = grouped_loss(logits, target,
                                   hard_negatives=hard_negatives)
        consistency = logits.new_zeros(())
        terms = []
        for index, track_id in enumerate(track_ids.detach().cpu().tolist()):
            if target[index] > 0.5 and int(track_id) in previous:
                terms.append((torch.sigmoid(logits[index]) -
                              previous[int(track_id)]).abs())
            if target[index] > 0.5:
                previous[int(track_id)] = torch.sigmoid(logits[index])
        if terms:
            consistency = torch.stack(terms).mean()
        loss = loss + 0.10 * consistency
        total = total + loss
        groups += 1
        stats["bce"] += float(parts["bce"])
        stats["rank"] += float(parts["rank"])
        stats["consistency"] += float(consistency.detach())
        predictions.append(logits.detach())
        targets.append(target.detach())
    total = total / max(1, groups)
    for key in list(stats):
        stats[key] /= max(1, groups)
    return total, stats, predictions, targets


class EpisodeTrainer(torch.nn.Module):
    """Make one causal multi-frame episode one DDP forward/backward unit."""

    def __init__(self, selector):
        super().__init__()
        self.selector = selector

    def forward(self, query, family, episode):
        loss, stats, _, _ = run_episode(
            self.selector, query, family, episode)
        values = torch.stack([
            loss.detach().new_tensor(stats["bce"]),
            loss.detach().new_tensor(stats["rank"]),
            loss.detach().new_tensor(stats["consistency"]),
        ])
        return loss, values


@torch.no_grad()
def validate(model, items, store, seed, episodes, sequence_length, device):
    model.eval()
    rng = random.Random(seed)
    all_logits, all_targets = [], []
    losses = []
    domain_loss = defaultdict(list)
    for _ in range(episodes):
        item, query, family, episode = sample_episode(
            items, store, rng, sequence_length, 0.20, device)
        loss, _, logits, targets = run_episode(model, query, family, episode)
        losses.append(float(loss))
        domain_loss[item["domain"]].append(float(loss))
        all_logits.extend([value.cpu() for value in logits])
        all_targets.extend([value.cpu() for value in targets])
    logits = torch.cat(all_logits) if all_logits else torch.zeros(0)
    targets = torch.cat(all_targets) if all_targets else torch.zeros(0)
    best = {"threshold": 0.5, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    probabilities = torch.sigmoid(logits)
    for threshold in np.linspace(0.05, 0.95, 37):
        pred = probabilities >= threshold
        true = targets > 0.5
        tp = int((pred & true).sum())
        fp = int((pred & ~true).sum())
        fn = int((~pred & true).sum())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        if f1 > best["f1"]:
            best = {"threshold": float(threshold), "f1": f1,
                    "precision": precision, "recall": recall}
    return {
        "loss": float(np.mean(losses)),
        "domain_loss": {key: float(np.mean(value))
                        for key, value in domain_loss.items()},
        "examples": int(len(targets)), "positive_rate": float(targets.mean()),
        **best,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--empty-probability", type=float, default=0.15)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--validation-episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--cache-size", type=int, default=64)
    args = parser.parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        device = torch.device("cuda", local_rank)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    items, protocol = load_expressions()
    if not items["train"]["kitti"] or not items["train"]["dance"]:
        raise RuntimeError("joint train mixture is empty")
    store = BankStore(args.cache_size)
    raw_model = L16TrackSelector(args.hidden, args.heads).to(device)
    episode_model = EpisodeTrainer(raw_model)
    model = DistributedDataParallel(episode_model, device_ids=[device.index]) \
        if distributed else episode_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    optimizer_updates = max(1, args.steps // args.accumulate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=optimizer_updates, eta_min=args.lr * 0.05)
    rng = random.Random(args.seed + 1009 * rank)
    out = Path(args.out)
    if rank == 0:
        out.parent.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        model.train()
        item, query, family, episode = sample_episode(
            items["train"], store, rng, args.sequence_length,
            args.empty_probability, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            loss, stat_values = model(query, family, episode)
            scaled = loss / args.accumulate
        stats = {key: float(value) for key, value in zip(
            ("bce", "rank", "consistency"), stat_values.detach().cpu())}
        scaled.backward()
        if step % args.accumulate == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        row = {"step": step, "train_loss": float(loss.detach()),
               "domain": item["domain"], "video": item["video"], **stats}
        if rank == 0 and (step == 1 or step % 50 == 0):
            print(f"[l16-selector] step={step}/{args.steps} "
                  f"loss={row['train_loss']:.4f} domain={item['domain']} "
                  f"bce={stats['bce']:.4f} rank={stats['rank']:.4f}", flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            validation = validate(
                raw_model, items["train_val"], store,
                args.seed + step, args.validation_episodes,
                args.sequence_length, device) if rank == 0 else None
            if distributed:
                values = [validation]
                dist.broadcast_object_list(values, src=0)
                validation = values[0]
            row["validation"] = validation
            if rank == 0:
                print(f"[l16-selector] val step={step} loss={validation['loss']:.4f} "
                      f"f1={validation['f1']:.4f} threshold={validation['threshold']:.3f} "
                      f"P/R={validation['precision']:.3f}/{validation['recall']:.3f}",
                      flush=True)
            checkpoint = {
                "model": raw_model.state_dict(),
                "cfg": {"hidden": args.hidden, "heads": args.heads,
                        "sequence_length": args.sequence_length,
                        "seed": args.seed, "generic_bank": "v1",
                        "world_size": world_size},
                "calibration": validation, "step": step,
                "protocol": protocol, "history_tail": history[-100:] + [row],
            }
            if rank == 0:
                torch.save(checkpoint, out.with_name(f"step{step}.pt"))
            if rank == 0 and validation["loss"] < best_loss:
                best_loss = validation["loss"]
                torch.save(checkpoint, out)
        history.append(row)
    if rank == 0:
        log = out.with_suffix(out.suffix + ".json")
        log.write_text(json.dumps({
            "args": vars(args), "world_size": world_size,
            "train_counts": {
                domain: len(values) for domain, values in items["train"].items()},
            "validation_counts": {
                domain: len(values) for domain, values in items["train_val"].items()},
            "best_validation_loss": best_loss,
            "wall_seconds": time.time() - started, "history": history,
        }, indent=2) + "\n")
        print(f"[l16-selector] done best={out} seconds={time.time()-started:.1f}",
              flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
