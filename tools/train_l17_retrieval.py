"""Train controlled iKUN and precision-first L17 selectors on frozen banks."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.ikun_bank_port import (  # noqa: E402
    IKunBankPort, pseudo_frequency_offset,
)
from locatemot.models.ikun_rn50_port import IKunRN50BankPort  # noqa: E402
from locatemot.models.l16_track_selector import expression_family_vector  # noqa: E402
from locatemot.models.l17_track_retriever import L17TrackSetRetriever  # noqa: E402
from locatemot.rmot.ikun_cache import RN50FeatureStore  # noqa: E402
from tools.train_l16_track_selector import (  # noqa: E402
    BankStore, FEATURE_NAMES, frame_features, load_expressions,
)


def expression_text(entry: dict) -> str:
    return entry.get("sentence", entry.get("expression", ""))


def build_calibration_table(items: dict) -> dict:
    rows = {}
    for domain_items in items.values():
        for item in domain_items:
            entry = item["entry"]
            text = expression_text(entry)
            positive = sum(len(value) for value in entry.get("label", {}).values())
            if text not in rows:
                rows[text] = [list(map(float, entry["spec"])), 0]
            rows[text][1] += positive
    total = sum(value[1] for value in rows.values())
    total = max(1, total)
    return {
        "texts": list(rows),
        "features": [rows[text][0] for text in rows],
        "probabilities": [rows[text][1] / total for text in rows],
        "source": "iKUN pseudo-frequency calibration over L17 train expressions",
    }


def sample_episode(items, store, rng, sequence_length, empty_probability,
                   positive_center_probability, device, model_name=None,
                   feature_store=None):
    domain = "kitti" if rng.random() < 0.5 else "dance"
    pool = items[domain]
    empties = [item for item in pool if not item["nonempty"]]
    nonempty = [item for item in pool if item["nonempty"]]
    want_empty = bool(empties) and rng.random() < empty_probability
    choices = empties if want_empty else nonempty
    if not choices:
        choices = pool
    item = choices[rng.randrange(len(choices))]
    entry = item["entry"]
    bank = store.get(item["bank_dataset"], item["video"])
    frame_count = len(bank["tensors"]["frame_ids"])
    length = min(sequence_length, frame_count)
    labels = entry.get("label", {})
    positive_frames = [
        bank["frame_to_index"][int(frame)] for frame, ids in labels.items()
        if ids and int(frame) in bank["frame_to_index"]]
    if positive_frames and rng.random() < positive_center_probability:
        anchor = rng.choice(positive_frames)
        start = max(0, min(frame_count - length,
                           anchor - rng.randrange(max(1, length))))
        mode = "positive_center"
    else:
        start = rng.randrange(max(1, frame_count - length + 1))
        mode = "full_video_random" if item["nonempty"] else "empty_query"
    query = torch.as_tensor(np.asarray(entry["spec"], np.float32), device=device)
    family = expression_family_vector(expression_text(entry)).to(device)
    episode = []
    for frame_index in range(start, start + length):
        features, track_ids, begin, end = frame_features(bank, frame_index, device)
        if model_name == "ikun_rn50":
            features.update(feature_store.frame_features(
                item["bank_dataset"], item["video"], frame_index,
                begin, end, device))
        frame_id = int(bank["tensors"]["frame_ids"][frame_index])
        targets = {str(value) for value in labels.get(
            str(frame_id), labels.get(frame_id, []))}
        candidate_gt = bank["candidate_gt"][begin:end]
        target = torch.tensor([
            1.0 if value is not None and str(value) in targets else 0.0
            for value in candidate_gt], device=device)
        episode.append((features, track_ids, target, bool(targets), frame_id))
    return item, query, family, episode, mode


def ranking_loss(logits: torch.Tensor, target: torch.Tensor,
                 margin: float = 0.30) -> torch.Tensor:
    positive = logits[target > 0.5]
    negative = logits[target <= 0.5]
    if not len(positive) or not len(negative):
        return logits.new_zeros(())
    negative = torch.topk(negative, min(len(negative), max(4, 2 * len(positive)))).values
    return F.softplus(margin - positive[:, None] + negative[None, :]).mean()


def run_episode(model, model_name, query, family, episode,
                use_global=True, use_kum=True, use_null=True,
                stateless=False, use_identity=True):
    state = {}
    losses = []
    stats = defaultdict(float)
    for features, track_ids, target, gt_active, _frame_id in episode:
        if stateless:
            state = {}
        if model_name in ("ikun", "ikun_rn50"):
            if model_name == "ikun_rn50":
                output = model(features, query, track_ids, state,
                               use_global=use_global, use_kum=use_kum)
            else:
                output = model(features, query, track_ids, state)
            state = output["state"]
            logits = output["logits"]
            if len(logits):
                raw = F.binary_cross_entropy_with_logits(
                    logits, target, reduction="none")
                probability = torch.sigmoid(logits)
                pt = torch.where(
                    target > 0.5, probability, 1.0 - probability)
                loss = ((1.0 - pt).pow(2.0) * raw).mean()
            else:
                loss = sum(parameter.sum() * 0.0
                           for parameter in model.parameters())
            stats["focal"] += float(loss.detach())
        else:
            output = model(features, query, family, track_ids, state,
                           use_null=use_null, use_identity=use_identity)
            state = output["state"]
            logits = output["logits"]
            positive = target > 0.5
            weights = torch.where(positive, torch.full_like(target, 3.0),
                                  torch.ones_like(target))
            classification = (weights * F.binary_cross_entropy_with_logits(
                logits, target, reduction="none")).mean() if len(logits) \
                else output["null_logit"].new_zeros(())
            rank = ranking_loss(output["membership_logits"], target)
            candidate_available = float(bool(positive.any()))
            null_target = output["null_logit"].new_tensor(1.0 - candidate_available)
            null = F.binary_cross_entropy_with_logits(
                output["null_logit"], null_target) if use_null \
                else output["null_logit"].new_zeros(())
            loss = classification + 0.35 * rank + \
                (0.70 * null if use_null else 0.0)
            stats["classification"] += float(classification.detach())
            stats["rank"] += float(rank.detach())
            stats["null"] += float(null.detach())
            stats["uncovered_active"] += float(gt_active and not candidate_available)
        losses.append(loss)
    total = torch.stack(losses).mean()
    for key in list(stats):
        stats[key] /= max(1, len(losses))
    return total, stats


class EpisodeTrainer(nn.Module):
    def __init__(self, selector, model_name, use_global=True, use_kum=True,
                 use_null=True, stateless=False, use_identity=True):
        super().__init__()
        self.selector = selector
        self.model_name = model_name
        self.use_global = use_global
        self.use_kum = use_kum
        self.use_null = use_null
        self.stateless = stateless
        self.use_identity = use_identity

    def forward(self, query, family, episode):
        loss, stats = run_episode(
            self.selector, self.model_name, query, family, episode,
            use_global=self.use_global, use_kum=self.use_kum,
            use_null=self.use_null, stateless=self.stateless,
            use_identity=self.use_identity)
        ordered = ("focal", "classification", "rank", "null", "uncovered_active")
        values = torch.stack([loss.detach().new_tensor(stats.get(key, 0.0))
                              for key in ordered])
        return loss, values


def fixed_validation_items(items: dict, per_domain: int, seed: int) -> dict:
    rng = random.Random(seed)
    selected = {}
    for domain, values in items.items():
        nonempty = [item for item in values if item["nonempty"]]
        rng.shuffle(nonempty)
        # Official V1/V2 queries are non-empty and Refer-Dance HOTA explicitly
        # excludes empty-GT query/video pairs. Empty queries remain a training
        # negative, but must not distort checkpoint or threshold selection.
        selected[domain] = nonempty[:per_domain]
    return selected


@torch.no_grad()
def infer_item(model, model_name, item, store, device, calibration_table,
               feature_store=None, use_global=True, use_kum=True,
               stateless=False, use_null=True, use_identity=True):
    bank = store.get(item["bank_dataset"], item["video"])
    entry = item["entry"]
    query = torch.as_tensor(np.asarray(entry["spec"], np.float32), device=device)
    family = expression_family_vector(expression_text(entry)).to(device)
    labels = entry.get("label", {})
    state = {}
    score_rows, target_rows, active_rows = [], [], []
    offset = pseudo_frequency_offset(query, calibration_table) \
        if model_name in ("ikun", "ikun_rn50") else 0.0
    for frame_index, frame_id in enumerate(bank["tensors"]["frame_ids"].tolist()):
        features, track_ids, begin, end = frame_features(bank, frame_index, device)
        if model_name == "ikun_rn50":
            features.update(feature_store.frame_features(
                item["bank_dataset"], item["video"], frame_index,
                begin, end, device))
        target_ids = {str(value) for value in labels.get(
            str(frame_id), labels.get(frame_id, []))}
        target = torch.tensor([
            1.0 if value is not None and str(value) in target_ids else 0.0
            for value in bank["candidate_gt"][begin:end]], device=device)
        if stateless:
            state = {}
        if model_name == "ikun":
            output = model(features, query, track_ids, state)
        elif model_name == "ikun_rn50":
            output = model(features, query, track_ids, state,
                           use_global=use_global, use_kum=use_kum)
        else:
            output = model(features, query, family, track_ids, state,
                           use_null=use_null, use_identity=use_identity)
        state = output["state"]
        score_rows.append((output["logits"].float() + offset).cpu())
        target_rows.append(target.cpu())
        active_rows.append(torch.full_like(target.cpu(), float(bool(target_ids))))
    return (torch.cat(score_rows), torch.cat(target_rows),
            torch.cat(active_rows))


def threshold_metrics(scores, targets, active, threshold):
    predicted = scores >= threshold
    truth = targets > 0.5
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    f05 = 1.25 * precision * recall / max(1e-9, 0.25 * precision + recall)
    gmean = float((precision * recall) ** 0.5)
    det_a = tp / max(1, tp + fp + fn)
    inactive_fp = int((predicted & (active <= 0.5)).sum())
    return {"precision": precision, "recall": recall, "f1": f1,
            "f05": f05, "gmean": gmean, "tp": tp, "fp": fp, "fn": fn,
            "det_a_proxy": det_a,
            "inactive_fp_fraction": inactive_fp / max(1, int(predicted.sum()))}


@torch.no_grad()
def validate_full(model, model_name, selected, store, device,
                  calibration_table, feature_store=None, use_global=True,
                  use_kum=True, stateless=False, use_null=True,
                  use_identity=True):
    model.eval()
    values = {}
    all_scores = []
    for domain, domain_items in selected.items():
        scores, targets, active = [], [], []
        for item in domain_items:
            score, target, frame_active = infer_item(
                model, model_name, item, store, device, calibration_table,
                feature_store=feature_store, use_global=use_global,
                use_kum=use_kum, stateless=stateless, use_null=use_null,
                use_identity=use_identity)
            scores.append(score)
            targets.append(target)
            active.append(frame_active)
        values[domain] = (torch.cat(scores), torch.cat(targets), torch.cat(active))
        all_scores.append(values[domain][0])
    combined = torch.cat(all_scores)
    quantiles = torch.quantile(
        combined, torch.linspace(0.001, 0.999, 161)).unique()
    best = None
    for threshold in quantiles.tolist():
        metrics = {domain: threshold_metrics(*row, threshold)
                   for domain, row in values.items()}
        det_as = [row["det_a_proxy"] for row in metrics.values()]
        objective = min(det_as) + 0.25 * sum(det_as) / len(det_as)
        candidate = {"threshold_logit": float(threshold),
                     "objective": objective, "domains": metrics}
        if best is None or candidate["objective"] > best["objective"]:
            best = candidate
    best["queries"] = {domain: len(rows) for domain, rows in selected.items()}
    best["examples"] = {domain: int(len(values[domain][0])) for domain in values}
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["ikun", "ikun_rn50", "l17"], required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--accumulate", type=int, default=2)
    parser.add_argument("--empty-probability", type=float, default=0.20)
    parser.add_argument("--positive-center-probability", type=float, default=0.35)
    parser.add_argument("--validate-every", type=int, default=500)
    parser.add_argument("--validation-per-domain", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--cache-size", type=int, default=64)
    parser.add_argument("--feature-cache-root",
                        default="outputs/l17/ikun_rn50_cache")
    parser.add_argument("--no-global", action="store_true")
    parser.add_argument("--no-kum", action="store_true")
    parser.add_argument("--stateless", action="store_true")
    parser.add_argument("--holistic-query", action="store_true")
    parser.add_argument("--no-null", action="store_true")
    parser.add_argument("--positive-centered", action="store_true")
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    items, protocol = load_expressions()
    store = BankStore(args.cache_size)
    feature_store = RN50FeatureStore(args.feature_cache_root, args.cache_size) \
        if args.model == "ikun_rn50" else None
    calibration_table = build_calibration_table(items["train"])
    validation_items = fixed_validation_items(
        items["train_val"], args.validation_per_domain, args.seed + 17)
    if args.model == "ikun":
        raw_model = IKunBankPort(args.hidden, args.heads).to(device)
    elif args.model == "ikun_rn50":
        raw_model = IKunRN50BankPort(args.hidden, args.heads).to(device)
    else:
        raw_model = L17TrackSetRetriever(
            args.hidden, holistic_query=args.holistic_query).to(device)
    episode_model = EpisodeTrainer(
        raw_model, args.model, use_global=not args.no_global,
        use_kum=not args.no_kum, use_null=not args.no_null,
        stateless=args.stateless, use_identity=not args.stateless)
    model = DistributedDataParallel(
        episode_model, device_ids=[device.index],
        find_unused_parameters=bool(
            args.holistic_query or args.no_null or args.stateless)
    ) if distributed else episode_model
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    updates = max(1, args.steps // args.accumulate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=updates, eta_min=args.lr * 0.05)
    rng = random.Random(args.seed + 1009 * rank)
    output = Path(args.out)
    if rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
    best_objective = -1.0
    history = []
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        item, query, family, episode, sampling = sample_episode(
            items["train"], store, rng, args.sequence_length,
            0.0 if args.positive_centered else args.empty_probability,
            1.0 if args.positive_centered else args.positive_center_probability,
            device,
            model_name=args.model, feature_store=feature_store)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            loss, stat_values = model(query, family, episode)
            scaled = loss / args.accumulate
        scaled.backward()
        if step % args.accumulate == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        names = ("focal", "classification", "rank", "null", "uncovered_active")
        row = {"step": step, "loss": float(loss.detach()),
               "domain": item["domain"], "sampling": sampling}
        row.update({name: float(value) for name, value in zip(
            names, stat_values.detach().cpu())})
        if rank == 0 and (step == 1 or step % 50 == 0):
            print(f"[l17-train] model={args.model} step={step}/{args.steps} "
                  f"loss={row['loss']:.4f} domain={row['domain']} "
                  f"sample={sampling}", flush=True)
        if step % args.validate_every == 0 or step == args.steps:
            validation = validate_full(
                raw_model, args.model, validation_items, store, device,
                calibration_table, feature_store=feature_store,
                use_global=not args.no_global, use_kum=not args.no_kum,
                stateless=args.stateless, use_null=not args.no_null,
                use_identity=not args.stateless) if rank == 0 else None
            if distributed:
                values = [validation]
                dist.broadcast_object_list(values, src=0)
                validation = values[0]
            row["validation"] = validation
            checkpoint = {
                "model": raw_model.state_dict(), "model_name": args.model,
                "cfg": {"hidden": args.hidden, "heads": args.heads,
                        "sequence_length": args.sequence_length,
                        "seed": args.seed, "world_size": world_size,
                        "holistic_query": args.holistic_query,
                        "no_null": args.no_null,
                        "positive_centered": args.positive_centered,
                        "stateless": args.stateless},
                "calibration": validation,
                "ikun_frequency_table": calibration_table
                if args.model in ("ikun", "ikun_rn50") else None,
                "protocol": protocol, "step": step,
            }
            if rank == 0:
                torch.save(checkpoint, output.with_name(f"{output.stem}_step{step}.pt"))
                print(f"[l17-val] model={args.model} step={step} "
                      f"objective={validation['objective']:.5f} "
                      f"threshold={validation['threshold_logit']:.4f} "
                      f"domains={validation['domains']}", flush=True)
                if validation["objective"] > best_objective:
                    best_objective = validation["objective"]
                    torch.save(checkpoint, output)
        history.append(row)
    if rank == 0:
        output.with_suffix(output.suffix + ".json").write_text(json.dumps({
            "args": vars(args), "world_size": world_size,
            "best_objective": best_objective,
            "validation_queries": {key: len(value)
                                   for key, value in validation_items.items()},
            "wall_seconds": time.time() - started, "history": history,
        }, indent=2) + "\n")
        print(f"[l17-train] done model={args.model} best={best_objective:.5f} "
              f"seconds={time.time()-started:.1f}", flush=True)
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
