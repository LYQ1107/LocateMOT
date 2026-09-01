"""Train the minimal RMOT candidate scorer on the fixed L20 fast bank.

This is deliberately a candidate-level development trainer.  It uses the
64 calibration queries for optimization and the 96 screening queries only as
a held-out validation report.  It never invokes a tracker, grouping,
membership/source acceptance, NULL scalar, or TrackEval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.rmot_candidate_scorer import RMOTCandidateScorer  # noqa: E402


MANIFEST_DEFAULT = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
BANK_ROOT_DEFAULT = ROOT / "outputs/l19/dual_banks_features"
METADATA_PATHS = (
    ROOT / "outputs/l11/data/rmot_kitti/expressions.json",
    ROOT / "outputs/l16/data/kitti_missing/records/expressions.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, np.float64).reshape(-1)
    labels = np.asarray(labels, bool).reshape(-1)
    valid = np.isfinite(scores)
    scores, labels = scores[valid], labels[valid]
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return float((ranks[labels].sum() - positive * (positive + 1) / 2.0) /
                 (positive * negative))


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float | None:
    scores = np.asarray(scores, np.float64).reshape(-1)
    labels = np.asarray(labels, bool).reshape(-1)
    valid = np.isfinite(scores)
    scores, labels = scores[valid], labels[valid]
    if not labels.any():
        return None
    order = np.argsort(-scores, kind="stable")
    ordered = labels[order].astype(np.float64)
    cumulative = np.cumsum(ordered)
    positions = np.flatnonzero(ordered)
    return float(np.mean(cumulative[positions] / (positions + 1.0)))


def scalar_stats(values: list[float] | np.ndarray) -> dict:
    values = np.asarray(values, np.float64).reshape(-1)
    if not len(values):
        return {"count": 0}
    return {
        "count": int(len(values)), "mean": float(values.mean()),
        "std": float(values.std()), "min": float(values.min()),
        "max": float(values.max()),
    }


def load_metadata() -> dict[tuple[str, str], dict]:
    result = {}
    for path in METADATA_PATHS:
        for video, entries in json.loads(path.read_text()).items():
            for entry in entries:
                expression = str(entry.get("expression",
                                      entry.get("sentence", "")))
                result[(str(video), expression)] = entry
    return result


def load_bank(path: Path) -> dict:
    bank = torch.load(path, map_location="cpu", weights_only=False)
    tensors = bank["tensors"]
    labels_path = path.with_suffix(".labels.json")
    if not labels_path.exists():
        raise FileNotFoundError(labels_path)
    labels = json.loads(labels_path.read_text())["candidate_gt"]
    n = len(tensors["track_id"])
    required = {"frame_ptr", "frame_ids", "track_id", "clip", "history_clip",
                "geometry", "motion", "objectness", "pool_id"}
    missing = sorted(required - set(tensors))
    if missing or len(labels) != n:
        raise ValueError(f"invalid bank {path}: missing={missing}, labels={len(labels)} n={n}")
    bank["candidate_gt"] = labels
    bank["bank_sha256"] = sha256_file(path)
    bank["labels_sha256"] = sha256_file(labels_path)
    return bank


def make_refs(rows: list[dict], metadata: dict, banks: dict[str, dict]) -> list[dict]:
    refs = []
    for row in rows:
        video = str(row["video"])
        expression = str(row["expression"])
        entry = metadata[(video, expression)]
        bank = banks[video]
        tensors, labels = bank["tensors"], bank["candidate_gt"]
        target_by_frame = {
            int(key): {str(value) for value in values}
            for key, values in entry.get("label", {}).items()
        }
        frame_ids = tensors["frame_ids"].tolist()
        ptr = tensors["frame_ptr"].tolist()
        pool = tensors["pool_id"].numpy()
        for frame_index, frame_id in enumerate(frame_ids):
            begin, end = int(ptr[frame_index]), int(ptr[frame_index + 1])
            target_ids = target_by_frame.get(int(frame_id), set())
            positive = np.asarray([
                value is not None and str(value) in target_ids
                for value in labels[begin:end]
            ], dtype=bool)
            main = bool(np.any(positive & (pool[begin:end] == 0)))
            reserve = bool(np.any(positive & (pool[begin:end] == 1)))
            null = not target_ids or not main and not reserve
            refs.append({
                "query_index": int(row["query_index"]), "video": video,
                "split": str(row["split"]), "spec": np.asarray(
                    entry["spec"], np.float32), "frame_index": frame_index,
                "frame_id": int(frame_id), "begin": begin, "end": end,
                "positive": positive, "null": bool(null),
            })
    return refs


HARD_PREFILTER = 48
HARD_TOPK = 12


def _online_hard_scores(model, ref: dict, bank: dict, candidates: np.ndarray,
                        device: torch.device, static_only: bool = False) -> np.ndarray:
    """Score the objectness-prefiltered negatives without a gradient path."""
    tensors = bank["tensors"]
    index = torch.as_tensor(ref["begin"] + candidates, dtype=torch.long)
    count = len(index)
    query = torch.as_tensor(ref["spec"], dtype=torch.float32).reshape(1, -1)
    values = {
        "query": query.expand(count, -1).to(device),
        "static_query": query.expand(count, -1).to(device),
        "motion_query": query.expand(count, -1).to(device),
        "current": tensors["clip"].float().index_select(0, index).to(device),
        "history": tensors["history_clip"].float().index_select(0, index).to(device),
        "geometry": tensors["geometry"].float().index_select(0, index).to(device),
        "motion": tensors["motion"].float().index_select(0, index).to(device),
        "objectness": tensors["objectness"].float().index_select(0, index).reshape(count, 1).to(device),
        "delta": torch.zeros(count, 1, device=device),
    }
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = score_batch(model, values, device, static_only).float().cpu().numpy()
    if was_training:
        model.train()
    return output


def select_rows(ref: dict, bank: dict, rng: random.Random,
                positive_limit: int, negative_limit: int,
                model=None, device: torch.device | None = None,
                static_only: bool = False, hard_prefilter: int = HARD_PREFILTER,
                hard_topk: int = HARD_TOPK) -> tuple[np.ndarray, np.ndarray]:
    positive = np.flatnonzero(ref["positive"])
    negative = np.flatnonzero(~ref["positive"])
    if len(positive) > positive_limit:
        positive = np.asarray(rng.sample(positive.tolist(), positive_limit))
    objectness = bank["tensors"]["objectness"][
        ref["begin"]:ref["end"]].float().numpy().reshape(-1)
    prefilter_count = min(len(negative), int(hard_prefilter))
    prefiltered = negative[np.argsort(-objectness[negative], kind="stable")[:prefilter_count]]
    hard_count = min(len(prefiltered), int(hard_topk))
    if model is not None and device is not None and len(prefiltered):
        current_scores = _online_hard_scores(
            model, ref, bank, prefiltered, device, static_only)
        hard = prefiltered[np.argsort(-current_scores, kind="stable")[:hard_count]]
    else:
        hard = prefiltered[:hard_count]
    remaining = np.setdiff1d(negative, hard, assume_unique=False)
    random_count = min(len(remaining), max(0, negative_limit - len(hard)))
    random_part = np.asarray(rng.sample(remaining.tolist(), random_count), dtype=np.int64)
    negative_selected = np.concatenate([hard, random_part])
    rows = np.concatenate([positive, negative_selected]).astype(np.int64)
    hard_mask = np.zeros(len(rows), dtype=bool)
    hard_mask[len(positive):len(positive) + len(hard)] = True
    return rows, hard_mask


def batch_from_refs(refs: list[dict], banks: dict[str, dict],
                    rng: random.Random, device: torch.device,
                    positive_limit: int, negative_limit: int,
                    model=None, static_only: bool = False,
                    hard_prefilter: int = HARD_PREFILTER,
                    hard_topk: int = HARD_TOPK) -> tuple[dict, list[tuple[int, int, bool]]]:
    pieces = {key: [] for key in ("query", "static_query", "motion_query",
                                  "current", "history", "geometry", "motion",
                                  "objectness", "delta", "target", "hard")}
    groups = []
    offset = 0
    for ref in refs:
        bank = banks[ref["video"]]
        rows, hard_mask = select_rows(
            ref, bank, rng, positive_limit, negative_limit, model, device,
            static_only, hard_prefilter, hard_topk)
        tensors = bank["tensors"]
        absolute = torch.as_tensor(ref["begin"] + rows, dtype=torch.long)
        count = len(rows)
        query = torch.as_tensor(ref["spec"], dtype=torch.float32).reshape(1, -1)
        pieces["query"].append(query.expand(count, -1))
        pieces["static_query"].append(query.expand(count, -1))
        pieces["motion_query"].append(query.expand(count, -1))
        pieces["current"].append(tensors["clip"].float().index_select(0, absolute))
        pieces["history"].append(tensors["history_clip"].float().index_select(0, absolute))
        pieces["geometry"].append(tensors["geometry"].float().index_select(0, absolute))
        pieces["motion"].append(tensors["motion"].float().index_select(0, absolute))
        pieces["objectness"].append(tensors["objectness"].float().index_select(0, absolute).reshape(count, 1))
        pieces["delta"].append(torch.zeros(count, 1))
        pieces["target"].append(torch.as_tensor(ref["positive"][rows], dtype=torch.float32))
        pieces["hard"].append(torch.as_tensor(hard_mask, dtype=torch.bool))
        groups.append((offset, offset + count, bool(ref["null"])))
        offset += count
    values = {key: torch.cat(value).to(device) for key, value in pieces.items()}
    return values, groups


def score_batch(model: nn.Module, values: dict, device: torch.device,
                static_only: bool = False) -> torch.Tensor:
    output = model(
        values["query"], values["static_query"], values["motion_query"],
        values["current"], values["history"], values["geometry"],
        frame_delta=values["delta"], motion_feature=values["motion"],
        objectness=values["objectness"],
    )
    return output["static_logit"] if static_only else output["final_candidate_logit"]


def train_step(model, optimizer, refs, banks, rng, device, args):
    values, groups = batch_from_refs(
        refs, banks, rng, device, args.positive_limit, args.negative_limit,
        model, args.static_only, args.hard_prefilter, args.hard_topk)
    logits = score_batch(model, values, device, args.static_only)
    target = values["target"]
    hard_mask = values["hard"].bool()
    positive_terms, hard_terms, easy_terms = [], [], []
    pair_terms, listwise_terms, null_terms = [], [], []
    for begin, end, is_null in groups:
        group_logits, group_target = logits[begin:end], target[begin:end]
        positives = group_logits[group_target > 0.5]
        negatives = group_logits[group_target <= 0.5]
        group_hard = hard_mask[begin:end]
        hard_negatives = group_logits[(group_target <= 0.5) & group_hard]
        easy_negatives = group_logits[(group_target <= 0.5) & ~group_hard]
        if len(positives):
            positive_terms.append(nn.functional.binary_cross_entropy_with_logits(
                positives, torch.ones_like(positives)))
        if len(hard_negatives):
            hard_terms.append(nn.functional.binary_cross_entropy_with_logits(
                hard_negatives, torch.zeros_like(hard_negatives)))
        if len(easy_negatives):
            easy_terms.append(nn.functional.binary_cross_entropy_with_logits(
                easy_negatives, torch.zeros_like(easy_negatives)))
        if len(hard_negatives):
            pair_count = min(len(hard_negatives), args.pairwise_topk)
            pair_order = torch.topk(
                hard_negatives, pair_count, largest=True, sorted=False).indices
            pair_negatives = hard_negatives[pair_order]
        else:
            pair_negatives = negatives
        if len(positives) and len(pair_negatives):
            pair_terms.append(nn.functional.softplus(
                args.pair_margin - (positives[:, None] - pair_negatives[None, :])).mean())
            listwise_terms.append(
                torch.logsumexp(torch.cat((positives, pair_negatives)), dim=0) -
                torch.logsumexp(positives, dim=0))
        if is_null and len(negatives):
            null_terms.append(nn.functional.binary_cross_entropy_with_logits(
                negatives, torch.zeros_like(negatives)))
    positive_bce = torch.stack(positive_terms).mean() if positive_terms else logits.sum() * 0.0
    hard_bce = torch.stack(hard_terms).mean() if hard_terms else logits.sum() * 0.0
    easy_bce = torch.stack(easy_terms).mean() if easy_terms else logits.sum() * 0.0
    # Frame-balanced BCE: easy negatives are auxiliary and cannot dominate.
    candidate_bce = positive_bce + hard_bce + 0.10 * easy_bce
    pairwise = torch.stack(pair_terms).mean() if pair_terms else logits.sum() * 0.0
    listwise = torch.stack(listwise_terms).mean() if listwise_terms else logits.sum() * 0.0
    null_bce = torch.stack(null_terms).mean() if null_terms else logits.sum() * 0.0
    total = (candidate_bce + args.pair_weight * pairwise +
             args.listwise_weight * listwise + args.null_weight * null_bce)
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip))
    optimizer.step()
    return {
        "total": float(total.detach().cpu()),
        "candidate_bce": float(candidate_bce.detach().cpu()),
        "positive_bce": float(positive_bce.detach().cpu()),
        "hard_negative_bce": float(hard_bce.detach().cpu()),
        "easy_negative_bce": float(easy_bce.detach().cpu()),
        "pairwise_margin": float(pairwise.detach().cpu()),
        "listwise_ranking": float(listwise.detach().cpu()),
        "null_candidate_bce": float(null_bce.detach().cpu()),
        "sampled_candidates": int(len(target)), "sampled_positive": int(target.sum().item()),
        "sampled_hard_negative_buckets": int(sum(
            int(torch.count_nonzero(hard_mask[g[0]:g[1]]) > 0) for g in groups)),
        "null_buckets": int(sum(int(g[2]) for g in groups)),
        "grad_norm": grad_norm,
    }


def evaluate(model, refs: list[dict], banks: dict[str, dict], device: torch.device,
             eval_batch: int, video_codes: dict[str, int],
             static_only: bool = False,
             hard_prefilter: int = HARD_PREFILTER,
             hard_topk: int = HARD_TOPK) -> dict:
    score_parts, label_parts, source_parts, objectness_parts = [], [], [], []
    frame_parts, query_parts, null_parts = [], [], []
    model.eval()
    with torch.inference_mode():
        for query_index in sorted({ref["query_index"] for ref in refs}):
            query_refs = [ref for ref in refs if ref["query_index"] == query_index]
            query_refs.sort(key=lambda ref: ref["frame_index"])
            if not query_refs:
                continue
            bank = banks[query_refs[0]["video"]]
            tensors = bank["tensors"]
            for start in range(0, len(query_refs), 8):
                frame_batch = query_refs[start:start + 8]
                index_parts, frame_id_parts, null_id_parts = [], [], []
                for ref in frame_batch:
                    index = np.arange(ref["begin"], ref["end"], dtype=np.int64)
                    index_parts.append(index)
                    frame_id_parts.append(np.full(len(index), ref["frame_id"], np.int32))
                    null_id_parts.append(np.full(len(index), int(ref["null"]), np.uint8))
                all_indices = np.concatenate(index_parts)
                frame_ids = np.concatenate(frame_id_parts)
                null_ids = np.concatenate(null_id_parts)
                query = torch.as_tensor(query_refs[0]["spec"], dtype=torch.float32).reshape(1, -1)
                for row_start in range(0, len(all_indices), eval_batch):
                    row_indices = torch.as_tensor(all_indices[row_start:row_start + eval_batch])
                    count = len(row_indices)
                    values = {
                        "query": query.expand(count, -1).to(device),
                        "static_query": query.expand(count, -1).to(device),
                        "motion_query": query.expand(count, -1).to(device),
                        "current": tensors["clip"].float().index_select(0, row_indices).to(device),
                        "history": tensors["history_clip"].float().index_select(0, row_indices).to(device),
                        "geometry": tensors["geometry"].float().index_select(0, row_indices).to(device),
                        "motion": tensors["motion"].float().index_select(0, row_indices).to(device),
                        "objectness": tensors["objectness"].float().index_select(0, row_indices).reshape(count, 1).to(device),
                        "delta": torch.zeros(count, 1, device=device),
                    }
                    score = score_batch(model, values, device, static_only).float().cpu().numpy()
                    absolute = all_indices[row_start:row_start + eval_batch]
                    # Positive labels are kept in the frame refs; this avoids
                    # re-reading any GT source during model evaluation.
                    label_lookup = np.concatenate([
                        ref["positive"] for ref in frame_batch
                    ])
                    source_lookup = np.concatenate([
                        tensors["pool_id"][ref["begin"]:ref["end"]].numpy()
                        for ref in frame_batch
                    ])
                    objectness_lookup = np.concatenate([
                        tensors["objectness"][ref["begin"]:ref["end"]].float().numpy().reshape(-1)
                        for ref in frame_batch
                    ])
                    score_parts.append(score)
                    label_parts.append(label_lookup[row_start:row_start + eval_batch])
                    source_parts.append(source_lookup[row_start:row_start + eval_batch])
                    objectness_parts.append(objectness_lookup[row_start:row_start + eval_batch])
                    frame_parts.append(frame_ids[row_start:row_start + eval_batch])
                    query_parts.append(np.full(len(score), query_index, np.int64))
                    null_parts.append(null_ids[row_start:row_start + eval_batch])
    scores = np.concatenate(score_parts).astype(np.float32)
    labels = np.concatenate(label_parts).astype(bool)
    source = np.concatenate(source_parts).astype(np.int8)
    objectness = np.concatenate(objectness_parts).astype(np.float32)
    frame = np.concatenate(frame_parts).astype(np.int32)
    query = np.concatenate(query_parts).astype(np.int64)
    null = np.concatenate(null_parts).astype(bool)
    # Every query in this fast manifest belongs to one video. Reconstruct the
    # namespace from the refs rather than using source or a global frame ID.
    query_video = {ref["query_index"]: video_codes[ref["video"]] for ref in refs}
    video = np.asarray([query_video[int(value)] for value in query], np.int8)
    frame_key = query * 1_000_000_000 + video.astype(np.int64) * 1_000_000 + frame.astype(np.int64)
    order = np.argsort(frame_key, kind="stable")
    sorted_key = frame_key[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1], True])
    top1_hits = top5_hits = positive_frames = 0
    top1_selected = {0: 0, 1: 0}
    top1_correct = {0: 0, 1: 0}
    top5_selected = {0: 0, 1: 0}
    top5_correct = {0: 0, 1: 0}
    null_max, margins, top1_margins, positive_ranks, hard_negative_scores = [], [], [], [], []
    easy_negative_scores = []
    zero_selected = 0
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        indices = order[left:right]
        frame_scores, frame_labels = scores[indices], labels[indices]
        frame_source = source[indices]
        frame_objectness = objectness[indices]
        ranking = np.argsort(-frame_scores, kind="stable")
        positives = np.flatnonzero(frame_labels)
        negatives = np.flatnonzero(~frame_labels)
        zero_selected += int(np.count_nonzero(frame_scores >= 0.0))
        if len(positives):
            positive_frames += 1
            top1_hits += int(np.any(frame_labels[ranking[:1]]))
            top5_hits += int(np.any(frame_labels[ranking[:5]]))
            ranks = np.empty(len(frame_labels), np.int32)
            ranks[ranking] = np.arange(1, len(ranking) + 1)
            positive_ranks.extend(ranks[positives].tolist())
            if len(negatives):
                prefilter = negatives[np.argsort(-frame_objectness[negatives], kind="stable")[:min(len(negatives), hard_prefilter)]]
                hard = prefilter[np.argsort(-frame_scores[prefilter], kind="stable")[:min(len(prefilter), hard_topk)]]
                easy = np.asarray([index for index in negatives if index not in set(hard.tolist())], dtype=np.int64)
                margins.append(float(frame_scores[positives].min() - frame_scores[hard].max()))
                top1_margins.append(float(frame_scores[positives].max() - frame_scores[hard].max()))
                hard_negative_scores.append(float(frame_scores[hard].max()))
                easy_negative_scores.extend(frame_scores[easy].tolist())
        elif null[indices[0]]:
            null_max.append(float(frame_scores.max()))
        for source_id in (0, 1):
            pool = np.flatnonzero(frame_source == source_id)
            if not len(pool):
                continue
            source_order = pool[np.argsort(-frame_scores[pool], kind="stable")]
            top1_selected[source_id] += 1
            top1_correct[source_id] += int(frame_labels[source_order[:1]].sum())
            top5_selected[source_id] += min(5, len(source_order))
            top5_correct[source_id] += int(frame_labels[source_order[:5]].sum())
    null_frame_count = int(sum(
        bool(null[indices[0]]) for left, right in zip(boundaries[:-1], boundaries[1:])
        for indices in (order[left:right],)
    ))
    return {
        "candidate_count": int(len(labels)), "positive_count": int(labels.sum()),
        "negative_count": int((~labels).sum()), "frame_count": int(len(boundaries) - 1),
        "positive_frame_count": int(positive_frames), "null_frame_count": null_frame_count,
        "roc_auc": auc(scores, labels), "pr_auc": average_precision(scores, labels),
        "top1_frame_recall": float(top1_hits / max(1, positive_frames)),
        "top5_frame_recall": float(top5_hits / max(1, positive_frames)),
        "top1_hit_count": int(top1_hits), "top5_hit_count": int(top5_hits),
        "positive_score": scalar_stats(scores[labels]),
        "negative_score": scalar_stats(scores[~labels]),
        "positive_rank": scalar_stats(positive_ranks),
        "hard_negative_score": scalar_stats(hard_negative_scores),
        "easy_negative_score": scalar_stats(easy_negative_scores),
        "positive_hard_negative_margin": scalar_stats(margins),
        "top1_positive_hard_negative_margin": scalar_stats(top1_margins),
        "hard_negative_violation_rate": float(
            np.mean(np.asarray(margins) < 0.0)) if margins else None,
        "null_highest_candidate_score": scalar_stats(null_max),
        "source_internal_precision": {
            name: {
                "top1": float(top1_correct[source_id] / max(1, top1_selected[source_id])),
                "top5": float(top5_correct[source_id] / max(1, top5_selected[source_id])),
                "top1_correct": top1_correct[source_id], "top1_selected": top1_selected[source_id],
                "top5_correct": top5_correct[source_id], "top5_selected": top5_selected[source_id],
            } for source_id, name in ((0, "main"), (1, "reserve"))
        },
        "zero_threshold_acceptance_rate": float(zero_selected / max(1, len(scores))),
        "zero_threshold_predictions_per_positive": float(zero_selected / max(1, int(labels.sum()))),
    }


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(MANIFEST_DEFAULT))
    parser.add_argument("--bank-root", default=str(BANK_ROOT_DEFAULT))
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--listwise-weight", type=float, default=0.5)
    parser.add_argument("--hard-prefilter", type=int, default=HARD_PREFILTER)
    parser.add_argument("--hard-topk", type=int, default=HARD_TOPK)
    parser.add_argument("--pairwise-topk", type=int, default=HARD_TOPK)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-frames", type=int, default=8)
    parser.add_argument("--positive-limit", type=int, default=8)
    parser.add_argument("--negative-limit", type=int, default=24)
    parser.add_argument("--eval-batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--pair-weight", type=float, default=0.5)
    parser.add_argument("--null-weight", type=float, default=0.25)
    parser.add_argument("--pair-margin", type=float, default=0.5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    args = parser.parse_args()
    if args.steps not in (50, 100, 250, 500, 1000):
        raise ValueError("steps must be one of 50, 100, 250, 500, or 1000")
    manifest_path, bank_root, out_root = Path(args.manifest), Path(args.bank_root), Path(args.out_root)
    if not manifest_path.is_absolute(): manifest_path = ROOT / manifest_path
    if not bank_root.is_absolute(): bank_root = ROOT / bank_root
    if not out_root.is_absolute(): out_root = ROOT / out_root
    if out_root.exists():
        raise FileExistsError(f"refusing to overwrite: {out_root}")
    manifest = json.loads(manifest_path.read_text())
    rows = sorted(manifest["queries"], key=lambda value: int(value["query_index"]))
    if len(rows) != 160 or manifest.get("selection_uses_model_scores", True):
        raise ValueError("expected the fixed score-independent 160-query manifest")
    if sum(row["split"] == "calibration" for row in rows) != 64 or \
            sum(row["split"] == "screening" for row in rows) != 96:
        raise ValueError("fixed manifest split counts are not 64/96")
    metadata = load_metadata()
    videos = sorted({str(row["video"]) for row in rows})
    banks = {video: load_bank(bank_root / "kitti" / f"{video}.pt") for video in videos}
    video_codes = {video: index + 1 for index, video in enumerate(videos)}
    train_rows = [row for row in rows if row["split"] == "calibration"]
    val_rows = [row for row in rows if row["split"] == "screening"]
    train_refs, val_refs = make_refs(train_rows, metadata, banks), make_refs(val_rows, metadata, banks)
    if not train_refs or not val_refs:
        raise ValueError("empty training or validation frame set")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA scorer training but CUDA is unavailable")
    device = torch.device(args.device)
    model = RMOTCandidateScorer(hidden=256, dropout=0.10).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)
    model.eval()
    initial_val = evaluate(model, val_refs, banks, device, args.eval_batch,
                           video_codes, args.static_only,
                           args.hard_prefilter, args.hard_topk)
    history, loss_rows = [], []
    model.train()
    for step in range(1, args.steps + 1):
        sampled = rng.sample(train_refs, min(args.batch_frames, len(train_refs)))
        row = train_step(model, optimizer, sampled, banks, rng, device, args)
        scheduler.step()
        row["step"] = step
        loss_rows.append(row)
        if step == 1 or step % 50 == 0 or step == args.steps:
            print(json.dumps({"step": step, **row}, sort_keys=True), flush=True)
    model.eval()
    train_report = evaluate(model, train_refs, banks, device, args.eval_batch,
                            video_codes, args.static_only,
                            args.hard_prefilter, args.hard_topk)
    val_report = evaluate(model, val_refs, banks, device, args.eval_batch,
                          video_codes, args.static_only,
                          args.hard_prefilter, args.hard_topk)
    out_root.mkdir(parents=True, exist_ok=False)
    checkpoint = out_root / f"checkpoint_step{args.steps}.pt"
    torch.save({
        "format": "locatemot-rmot-candidate-scorer-v1", "step": args.steps,
        "seed": args.seed, "manifest_sha256": sha256_file(manifest_path),
        "bank_sha256": {video: banks[video]["bank_sha256"] for video in videos},
        "labels_sha256": {video: banks[video]["labels_sha256"] for video in videos},
        "config": vars(args), "model": model.state_dict(),
        "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
    }, checkpoint)
    initial_auc, final_auc = initial_val["roc_auc"], val_report["roc_auc"]
    initial_pr, final_pr = initial_val["pr_auc"], val_report["pr_auc"]
    initial_margin = initial_val["positive_hard_negative_margin"]["mean"]
    final_margin = val_report["positive_hard_negative_margin"]["mean"]
    gate = {
        "auc_gt_055": bool(final_auc is not None and final_auc > 0.55),
        "val_auc_improved": bool(final_auc is not None and initial_auc is not None and final_auc > initial_auc),
        "val_pr_auc_improved": bool(final_pr is not None and initial_pr is not None and final_pr > initial_pr),
        "positive_hard_negative_margin_improved": bool(final_margin > initial_margin),
        "all_candidates_accepted": bool(val_report["zero_threshold_acceptance_rate"] > 0.995),
        "passed_250": bool(final_auc is not None and final_auc > 0.55 and
                           final_auc > (initial_auc or -1.0) and
                           final_pr is not None and final_pr > (initial_pr or -1.0) and
                           final_margin > initial_margin and
                           val_report["zero_threshold_acceptance_rate"] <= 0.995),
        "candidate_gate_for_fast_trackeval": {
            "auc_gt_065": bool(final_auc is not None and final_auc > 0.65),
            "top1_recall_gt_050": bool(val_report["top1_frame_recall"] > 0.50),
            "zero_threshold_predictions_not_explosive": bool(
                val_report["zero_threshold_predictions_per_positive"] < 3.0),
        },
    }
    payload = {
        "format": "locatemot-rmot-candidate-scorer-training-v1",
        "provenance": {
            "project_root": str(ROOT), "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256_file(manifest_path), "query_count": len(rows),
            "train_split": "calibration (64 queries)", "val_split": "screening (96 queries)",
            "bank_root": str(bank_root.resolve()), "bank_sha256": {
                video: banks[video]["bank_sha256"] for video in videos},
            "labels_sha256": {video: banks[video]["labels_sha256"] for video in videos},
            "official_eval_used": False, "trackeval_used": False,
            "tracker_modified": False, "old_checkpoint_modified": False,
            "grouping": False, "membership": False, "source_acceptance": False,
            "null_scalar_subtraction": False, "temporal_gru": False,
        },
        "config": vars(args), "initial_validation": initial_val,
        "train": train_report, "validation": val_report,
        "loss": {key: float(np.mean([row[key] for row in loss_rows]))
                 for key in ("total", "candidate_bce", "positive_bce",
                             "hard_negative_bce", "easy_negative_bce",
                             "pairwise_margin", "listwise_ranking",
                             "null_candidate_bce", "grad_norm")},
        "sampling": {key: int(sum(row[key] for row in loss_rows))
                     for key in ("sampled_candidates", "sampled_positive",
                                 "sampled_hard_negative_buckets", "null_buckets")},
        "gate": gate, "checkpoint": str(checkpoint),
        "static_only": bool(args.static_only),
        "motion_branch_status": "disabled in final score; static_query and motion_query share the same spec; frame_delta is zero",
    }
    atomic_json(out_root / f"metrics_step{args.steps}.json", payload)
    (out_root / "README.md").write_text(
        f"# Minimal RMOT scorer step-{args.steps}\n\n"
        "Calibration is optimization data; screening is held-out validation. "
        "No tracker, grouping, official evaluation, or TrackEval was used.\n\n"
        f"Gate passed: `{gate['passed_250']}`\n")
    print(json.dumps({"status": "complete", "output": str(out_root), "gate": gate}, indent=2), flush=True)


if __name__ == "__main__":
    main()
