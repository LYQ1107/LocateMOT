#!/usr/bin/env python3
"""B1: cross-video/domain held-out semantic gate for L48 B0."""
from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l29_frame_membership_set_decoder import (  # noqa: E402
    L29FrameMembershipSetDecoder,
)
from locatemot.models.l48_joint_rmot import L48SemanticMatcher  # noqa: E402
from locatemot.rmot.l48_data import L29_CHECKPOINT, load_bank  # noqa: E402
from tools.train_l28_track_set_decoder import state_at  # noqa: E402

DATA = ROOT / "outputs/l48/data"
CONTRACT = ROOT / "outputs/l48/audit/joint_data_contract.json"
TEXT_CACHE = DATA / "text_cache.pt"
CHECKPOINT = ROOT / "outputs/l48/train/semantic_smoke100/checkpoint_semantic_step100.pt"
OUT = ROOT / "outputs/l48/eval/semantic_gate.json"


def load_jsonl(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def relation_features(boxes: torch.Tensor, image_size):
    boxes = boxes.float()
    width = float(image_size[0]) if len(image_size) >= 2 else max(1.0, float(boxes[:, 2].max()))
    height = float(image_size[1]) if len(image_size) >= 2 else max(1.0, float(boxes[:, 3].max()))
    scale = boxes.new_tensor([width, height, width, height])
    norm = boxes / scale
    center = (norm[:, :2] + norm[:, 2:]) * .5
    if len(boxes) <= 1:
        return boxes.new_zeros((len(boxes), 4))
    delta = center[:, None] - center[None, :]
    dist2 = delta.square().sum(-1)
    dist2.fill_diagonal_(float("inf"))
    nearest = dist2.argmin(-1)
    other = norm[nearest]
    left = torch.maximum(norm[:, None, 0], other[None, :, 0])
    top = torch.maximum(norm[:, None, 1], other[None, :, 1])
    right = torch.minimum(norm[:, None, 2], other[None, :, 2])
    bottom = torch.minimum(norm[:, None, 3], other[None, :, 3])
    inter = (right - left).clamp_min(0) * (bottom - top).clamp_min(0)
    area = (norm[:, 2] - norm[:, 0]).clamp_min(0) * (norm[:, 3] - norm[:, 1]).clamp_min(0)
    other_area = (other[:, 2] - other[:, 0]).clamp_min(0) * (other[:, 3] - other[:, 1]).clamp_min(0)
    idx = torch.arange(len(boxes))
    inter_n = inter[idx, nearest]
    iou = inter_n / (area + other_area[nearest] - inter_n).clamp_min(1e-6)
    return torch.cat((delta[idx, nearest], iou[:, None], dist2[idx, nearest, None].sqrt()), -1)


def build_state_cache(bank):
    tensors = bank["tensors"]
    count = int(tensors["track_id"].numel())
    by_track = defaultdict(list)
    for row, track in enumerate(tensors["track_id"].long().tolist()):
        by_track[int(track)].append(row)
    tracks = sorted(by_track)
    ordered = [row for track in tracks for row in by_track[track]]
    order = torch.as_tensor(ordered, dtype=torch.long)
    feature = torch.cat([
        tensors[name].float().reshape(count, -1)
        for name in ("clip", "history_clip", "uidm_h", "geometry", "motion", "lifecycle", "objectness")
    ], 1).half()[order].contiguous()
    return {
        "track_ids": torch.as_tensor(tracks, dtype=torch.long),
        "track_ptr": torch.as_tensor([0] + list(np.cumsum([len(by_track[t]) for t in tracks])), dtype=torch.long),
        "obs_features": feature,
        "obs_frame": tensors["frame"].long()[order].to(torch.int32),
        "obs_gt_ids": [None] * len(ordered),
    }


def valid_track_indices(cache, cutoff):
    ptr = cache["track_ptr"].numpy()
    frames = cache["obs_frame"].numpy()
    return [index for index in range(len(ptr) - 1)
            if np.any(frames[int(ptr[index]):int(ptr[index + 1])] <= int(cutoff))]


def baseline_scores(teacher, cache, bank, unit, text, device):
    frame = int(unit["frame_id"])
    obs, obs_mask, obs_time, _, _ = state_at(cache, frame, history=8)
    text_index = text["sentence_to_index"][unit["sentence"]]
    with torch.inference_mode():
        encoded = teacher.encode_observations(obs.to(device), obs_mask.to(device), obs_time.to(device))
        out = teacher.forward_encoded(encoded, encoded[1], text["token_hidden"][text_index].to(device),
                                      text["attention_mask"][text_index].to(device))
    tracks = cache["track_ids"][valid_track_indices(cache, frame)].tolist()
    values = {int(track): float(value) for track, value in
              zip(tracks, out["current_membership_logits"].float().cpu().tolist())}
    tensors = bank["tensors"]
    rows = range(int(unit["begin"]), int(unit["end"]))
    score = np.asarray([values.get(int(tensors["track_id"][r]), -20.0) for r in rows], dtype=np.float32)
    return score


def semantic_scores(model, bank, unit, text, device):
    tensors = bank["tensors"]
    sl = slice(int(unit["begin"]), int(unit["end"]))
    idx = text["sentence_to_index"][unit["sentence"]]
    relation = relation_features(tensors["box"][sl], unit["image_size"]).to(device)
    with torch.inference_mode():
        out = model(tensors["clip"][sl].to(device), tensors["history_clip"][sl].to(device),
                   tensors["geometry"][sl].to(device), tensors["motion"][sl].to(device),
                   tensors["context"][sl].to(device), tensors["lifecycle"][sl].to(device),
                   tensors["objectness"][sl].to(device), text["token_hidden"][idx].to(device),
                   text["attention_mask"][idx].to(device), relation)
    return out["semantic_logit"].float().cpu().numpy()


def fit_threshold(records):
    """Fit one candidate threshold from fit-only labels; no held-out labels."""
    values = np.concatenate([x["score"] for x in records if len(x["score"])])
    labels = np.concatenate([x["label"] for x in records if len(x["label"])])
    if not len(values):
        return {"threshold": 0.0, "objective": "fit_candidate_f1", "fit_units": 0}
    candidates = np.unique(values)
    if len(candidates) > 256:
        candidates = np.quantile(values, np.linspace(0, 1, 256))
    best = None
    for threshold in candidates.tolist() + [float(values.max()) + 1e-6, float(values.min()) - 1e-6]:
        chosen = values >= threshold
        tp = int((chosen & labels).sum()); fp = int((chosen & ~labels).sum()); fn = int((~chosen & labels).sum())
        f1 = 2.0 * tp / max(1.0, 2.0 * tp + fp + fn)
        key = (f1, -float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), tp, fp, fn)
    return {"threshold": best[1], "objective": "fit_candidate_f1",
            "fit_units": len(records), "fit_tp": best[2], "fit_fp": best[3], "fit_fn": best[4],
            "screening_or_validation_labels_used": False}


def auc_score(values, labels):
    values = np.asarray(values, dtype=np.float64); labels = np.asarray(labels, dtype=bool)
    pos = values[labels]; neg = values[~labels]
    if not len(pos) or not len(neg):
        return None
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64); ranks[order] = np.arange(1, len(values) + 1)
    return float((ranks[labels].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def average_precision(values, labels):
    values = np.asarray(values, dtype=np.float64); labels = np.asarray(labels, dtype=bool)
    total = int(labels.sum())
    if not total: return None
    order = np.argsort(-values, kind="stable"); hits = labels[order]
    cumulative = np.cumsum(hits)
    return float((cumulative[hits] / np.arange(1, len(hits) + 1)[hits]).sum() / total)


def summarize(records, threshold):
    if not records:
        return {"frame_units": 0, "threshold": threshold}
    flat_s = np.concatenate([x["score"] for x in records])
    flat_y = np.concatenate([x["label"] for x in records])
    tp = fp = fn = 0; empty = 0; null_false = 0; top1 = top5 = 0
    positive_units = multi_units = multi_recall_sum = 0
    strict_margins = []; best_margins = []; avg_margins = []; violations = []
    accepted_by_source = defaultdict(lambda: [0, 0])
    for rec in records:
        score = rec["score"]; y = rec["label"]
        chosen = score >= threshold
        tp += int((chosen & y).sum()); fp += int((chosen & ~y).sum()); fn += int((~chosen & y).sum())
        empty += int(not chosen.any())
        null_false += int(not y.any() and chosen.any())
        for source, source_mask in rec["sources"].items():
            if not np.any(source_mask): continue
            selected = chosen & source_mask
            accepted_by_source[source][0] += int(selected.sum())
            accepted_by_source[source][1] += int((selected & y).sum())
        pos = np.flatnonzero(y)
        neg = np.flatnonzero(~y)
        if not len(pos):
            continue
        positive_units += 1
        order = np.argsort(-score, kind="stable")
        top1 += int(y[order[:1]].any()); top5 += int(y[order[:5]].any())
        if len(pos) > 1:
            multi_units += 1
            multi_recall_sum += float((chosen & y).sum() / len(pos))
        if len(neg):
            strict = float(score[pos].min() - score[neg].max())
            best = float(score[pos].max() - score[neg].max())
            avg = float(score[pos].mean() - score[neg].max())
            strict_margins.append(strict); best_margins.append(best); avg_margins.append(avg)
            violations.append(strict < 0)
    def dist(values):
        if not values: return {"count": 0, "mean": None, "median": None, "q10": None, "q90": None}
        x = np.asarray(values, dtype=np.float64)
        return {"count": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
                "q10": float(np.quantile(x, .1)), "q90": float(np.quantile(x, .9))}
    source_metrics = {key: {"accepted": value[0], "true_positive": value[1],
                            "precision": value[1] / max(1, value[0])}
                      for key, value in sorted(accepted_by_source.items())}
    return {
        "frame_units": len(records), "candidate_count": int(len(flat_y)),
        "positive_count": int(flat_y.sum()), "positive_frame_units": positive_units,
        "roc_auc": auc_score(flat_s, flat_y), "pr_auc": average_precision(flat_s, flat_y),
        "top1_frame_recall": top1 / max(1, positive_units), "top5_frame_recall": top5 / max(1, positive_units),
        "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
        "false_positive_candidates_per_frame": fp / max(1, len(records)),
        "empty_output_rate": empty / max(1, len(records)),
        "null_frame_false_acceptance": null_false / max(1, len(records)),
        "predictions_per_positive": (tp + fp) / max(1, int(flat_y.sum())),
        "multi_positive_frame_count": multi_units,
        "multi_positive_recall": multi_recall_sum / max(1, multi_units),
        "strict_min_positive_margin": dist(strict_margins),
        "best_positive_margin": dist(best_margins),
        "average_positive_margin": dist(avg_margins),
        "hard_violation_rate": float(np.mean(violations)) if violations else None,
        "source_precision": source_metrics,
        "threshold": float(threshold),
        "unit_summaries": [{"dataset": x["dataset"], "video": x["video"], "query_id": x["query_id"],
                            "frame_id": x["frame_id"], "positive_count": int(x["label"].sum()),
                            "top1_positive": bool(x["label"][np.argsort(-x["score"])[:1]].any()) if x["label"].any() else None,
                            "selected_count": int((x["score"] >= threshold).sum()),
                            "max_score": float(x["score"].max()) if len(x["score"]) else None}
                           for x in records],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")
    started = time.time()
    calibration = load_jsonl(DATA / "calibration_units.jsonl")
    validation = load_jsonl(DATA / "val_units.jsonl")
    text = torch.load(TEXT_CACHE, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    teacher = L29FrameMembershipSetDecoder().to(device)
    teacher.load_state_dict(torch.load(L29_CHECKPOINT, map_location=device, weights_only=False)["model"], strict=True)
    teacher.eval()
    model = L48SemanticMatcher(hidden=256).to(device)
    model.load_state_dict(torch.load(Path(args.checkpoint), map_location=device, weights_only=False)["model"], strict=True)
    model.eval()
    all_units = [("calibration", x) for x in calibration] + [("validation", x) for x in validation]
    by_video = defaultdict(list)
    for split, unit in all_units:
        by_video[(unit["dataset"], unit["video"])].append((split, unit))
    scored = {"l29_teacher": {"calibration": [], "validation": []},
              "l48_semantic": {"calibration": [], "validation": []}}
    # Grouping by bank keeps one frozen bank in memory at a time and prevents
    # accidental cross-video candidate comparison.
    for (dataset, video), values in sorted(by_video.items()):
        bank = load_bank(dataset, video)
        state_cache = build_state_cache(bank)
        for split, unit in values:
            count = int(unit["end"] - unit["begin"])
            label = np.zeros(count, dtype=bool)
            label[np.asarray(unit["positive_indices"], dtype=np.int64)] = True
            pool = bank["tensors"].get("pool_id")
            if pool is None:
                source_values = np.zeros(count, dtype=np.int64)
                sources = {"unknown": np.ones(count, dtype=bool)}
            else:
                source_values = pool[int(unit["begin"]):int(unit["end"])].long().numpy()
                sources = {"main": source_values == 0, "reserve": source_values != 0}
            common = {"dataset": dataset, "video": video, "query_id": int(unit["query_id"]),
                      "frame_id": int(unit["frame_id"]), "label": label, "sources": sources,
                      "source_values": source_values}
            teacher_score = baseline_scores(teacher, state_cache, bank, unit, text, device)
            semantic_score = semantic_scores(model, bank, unit, text, device)
            if len(teacher_score) != count or len(semantic_score) != count:
                raise RuntimeError(f"score length mismatch {dataset}/{video}/{unit['frame_id']}")
            scored["l29_teacher"][split].append({**common, "score": teacher_score})
            scored["l48_semantic"][split].append({**common, "score": semantic_score})
        del state_cache, bank
        gc.collect()
    thresholds = {}
    summaries = {}
    per_domain = {}
    for model_name in scored:
        thresholds[model_name] = {}
        for domain in sorted({x["dataset"] for x in calibration}):
            train_records = [x for x in scored[model_name]["calibration"] if x["dataset"] == domain]
            val_records = [x for x in scored[model_name]["validation"] if x["dataset"] == domain]
            thresholds[model_name][domain] = fit_threshold(train_records)
            threshold = thresholds[model_name][domain]["threshold"]
            summaries.setdefault(model_name, {})[domain] = summarize(val_records, threshold)
    for domain in sorted(summaries["l29_teacher"]):
        base = summaries["l29_teacher"][domain]
        new = summaries["l48_semantic"][domain]
        per_domain[domain] = {
            "baseline": base, "semantic": new,
            "delta": {
                "top1": new["top1_frame_recall"] - base["top1_frame_recall"],
                "top5": new["top5_frame_recall"] - base["top5_frame_recall"],
                "recall": new["recall"] - base["recall"],
                "precision": new["precision"] - base["precision"],
                "hard_violation_rate": new["hard_violation_rate"] - base["hard_violation_rate"],
                "multi_positive_recall": new["multi_positive_recall"] - base["multi_positive_recall"],
                "empty_output_rate": new["empty_output_rate"] - base["empty_output_rate"],
            },
        }
    hard_improved = sum(
        per_domain[d]["delta"]["hard_violation_rate"] <= -.05
        for d in per_domain
    )
    per_domain_gate = {}
    for domain, result in per_domain.items():
        delta = result["delta"]
        per_domain_gate[domain] = {
            "top1_or_recall_no_substantive_drop": delta["top1"] >= -.05 or delta["recall"] >= -.05,
            "recall_not_collapsed": delta["recall"] >= -.05 and result["semantic"]["empty_output_rate"] < .90,
            "multi_positive_drop_within_003": delta["multi_positive_recall"] >= -.03,
            "not_output_empty": result["semantic"]["empty_output_rate"] < .90,
        }
    gate = {
        "decision": "B1_pass" if hard_improved >= 2 and all(all(v.values()) for v in per_domain_gate.values()) else "B1_failed_stage_stop",
        "hard_violation_domains_improved_by_at_least_005": int(hard_improved),
        "per_domain": per_domain_gate,
        "requirements": {
            "relative_to_frozen_l29": True,
            "hard_violation_decrease_in_at_least_two_domains": True,
            "no_substantive_top1_or_recall_drop": True,
            "multi_positive_recall_drop_le_003": True,
            "not_output_empty": True,
        },
    }
    payload = {
        "format": "locatemot-l48-semantic-gate-v1", "stage": "B1",
        "started_at_unix": started, "completed_at_unix": time.time(),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(Path(args.checkpoint)),
        "l29_checkpoint": str(L29_CHECKPOINT.resolve()),
        "l29_checkpoint_sha256": sha256_file(L29_CHECKPOINT),
        "data_contract_sha256": sha256_file(CONTRACT),
        "text_cache": str(TEXT_CACHE.resolve()), "text_cache_sha256": sha256_file(TEXT_CACHE),
        "calibration_unit_count": len(calibration), "validation_unit_count": len(validation),
        "calibration_labels_fit_only": True, "validation_labels_used_for_final_statistics_only": True,
        "screening_gt_read": False, "official_test_labels_read": False,
        "thresholds": thresholds, "per_domain": per_domain, "gate": gate,
        "semantic_inputs_excluded": ["source_id", "pool_id", "group_id", "state_key", "query_id_as_feature"],
        "token_span_region_alignment": "UNALIGNED", "static_motion_language_mask": "UNALIGNED/not claimed",
        "ordinary_mot_ovmot_touched": False,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"decision": gate["decision"], "hard_domains": hard_improved,
                      "per_domain": {d: r["delta"] for d, r in per_domain.items()},
                      "elapsed_sec": time.time() - started}, indent=2), flush=True)


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    main()
