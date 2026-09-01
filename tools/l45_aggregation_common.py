"""Pure CPU helpers for the L45 frozen pair-residual aggregation probes."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def iter_replay(cache, metadata):
    teacher = cache["teacher"]
    labels = cache["labels"].astype(bool)
    objectness = cache["objectness"]
    source = cache["source"]
    track_id = cache["track_id"]
    residual = cache["residual"]
    for item in metadata:
        cs, ce = int(item["candidate_start"]), int(item["candidate_end"])
        ps, pe = int(item["pair_start"]), int(item["pair_end"])
        n = ce - cs
        matrix = residual[ps:pe].reshape(n, n).astype(np.float64, copy=False)
        yield item, {
            "teacher": teacher[cs:ce].astype(np.float64, copy=False),
            "label": labels[cs:ce],
            "objectness": objectness[cs:ce].astype(np.float64, copy=False),
            "source": source[cs:ce],
            "track_id": track_id[cs:ce],
            "residual": matrix,
        }


def valid_pair_mask(n):
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    return mask


def aggregate_mean(teacher, residual):
    n = len(teacher)
    mask = valid_pair_mask(n)
    delta = np.where(mask, residual, 0.0).sum(1) / max(1, n - 1)
    return teacher + delta, delta


def aggregate_weighted(teacher, residual, tau=0.5):
    n = len(teacher)
    mask = valid_pair_mask(n)
    weights = np.exp(-np.abs(teacher[:, None] - teacher[None, :]) / float(tau))
    weights *= mask
    denom = weights.sum(1).clip(min=1e-12)
    delta = (weights * residual).sum(1) / denom
    return teacher + delta, delta


def aggregate_zero_mean(teacher, residual, tau=0.5):
    """Center all valid edges, then center candidate deltas per frame."""
    n = len(teacher)
    mask = valid_pair_mask(n)
    edge_mean = float(residual[mask].mean()) if mask.any() else 0.0
    centered = np.where(mask, residual - edge_mean, 0.0)
    weights = np.exp(-np.abs(teacher[:, None] - teacher[None, :]) / float(tau))
    weights *= mask
    denom = weights.sum(1).clip(min=1e-12)
    delta = (weights * centered).sum(1) / denom
    delta -= delta.mean() if len(delta) else 0.0
    return teacher + delta, delta, edge_mean


def aggregate_oracle(teacher, residual, labels, tau=0.5):
    """GT-privileged sign-correction diagnostic, never a model output.

    Only teacher-error positive/negative edges are retained.  Their measured
    magnitude is oriented toward the GT-positive endpoint, then the same
    teacher-margin weighting and per-frame zero-mean normalization as Probe 3
    are applied.  This is intentionally an upper-bound diagnostic, not a
    legal prediction rule.
    """
    n = len(teacher)
    mask = valid_pair_mask(n)
    pos = labels[:, None]
    neg = ~labels[:, None]
    error_pos_to_neg = pos & (~labels[None, :]) & (teacher[:, None] <= teacher[None, :])
    error_neg_to_pos = (~labels[:, None]) & labels[None, :] & (teacher[:, None] >= teacher[None, :])
    oriented = np.zeros_like(residual, dtype=np.float64)
    oriented[error_pos_to_neg] = np.abs(residual[error_pos_to_neg])
    oriented[error_neg_to_pos] = -np.abs(residual[error_neg_to_pos])
    weights = np.exp(-np.abs(teacher[:, None] - teacher[None, :]) / float(tau))
    weights *= mask
    denom = weights.sum(1).clip(min=1e-12)
    delta = (weights * oriented).sum(1) / denom
    delta -= delta.mean() if len(delta) else 0.0
    return teacher + delta, delta


def fixed_teacher_hard(labels, objectness, teacher, limit=24, prelimit=96):
    negative = np.flatnonzero(~np.asarray(labels, dtype=bool))
    if not len(negative):
        return np.empty(0, dtype=np.int64)
    pre = negative[np.argsort(-np.asarray(objectness)[negative], kind="stable")[:prelimit]]
    return pre[np.argsort(-np.asarray(teacher)[pre], kind="stable")[:limit]]


def rank_stats(teacher, student, labels, hard=None):
    pos = np.flatnonzero(labels)
    neg = np.flatnonzero(~labels) if hard is None else np.asarray(hard, dtype=np.int64)
    if not len(pos) or not len(neg):
        return {"pairs": 0, "teacher_correct": 0, "teacher_error": 0,
                "teacher_correct_flips": 0, "teacher_error_corrections": 0}
    td = teacher[pos, None] - teacher[neg][None, :]
    sd = student[pos, None] - student[neg][None, :]
    correct = td > 0
    return {
        "pairs": int(td.size), "teacher_correct": int(correct.sum()),
        "teacher_error": int((~correct).sum()),
        "teacher_correct_flips": int((correct & (sd < 0)).sum()),
        "teacher_error_corrections": int((~correct & (sd > 0)).sum()),
    }


def add_counts(total, item):
    for key, value in item.items():
        total[key] = total.get(key, 0) + int(value)


def distribution(values):
    if not values:
        return {"count": 0, "mean": None, "median": None,
                "q10": None, "q90": None, "max": None}
    x = np.asarray(values, dtype=np.float64)
    return {"count": int(len(x)), "mean": float(x.mean()),
            "median": float(np.median(x)), "q10": float(np.quantile(x, .1)),
            "q90": float(np.quantile(x, .9)), "max": float(x.max())}


def metric_summary(records, threshold):
    tp = fp = fn = selected = empty = null_accept = 0
    top1, top5, strict, best, average = [], [], [], [], []
    multi_recall, fp_frame = [], []
    transitions = defaultdict(list)
    source = {"main": [0, 0, 0], "reserve": [0, 0, 0]}
    for record in records:
        y = np.asarray(record["label"], dtype=bool)
        score = np.asarray(record["score"], dtype=np.float64)
        chosen = score >= float(threshold)
        tp += int((chosen & y).sum()); fp += int((chosen & ~y).sum())
        fn += int((~chosen & y).sum()); selected += int(chosen.sum())
        empty += int(not chosen.any())
        null_accept += int(not y.any() and chosen.any())
        fp_frame.append(int((chosen & ~y).sum()))
        transitions[int(record["query_index"])].append(
            (int(record["frame"]), set(record["track_id"][chosen].tolist())))
        order = np.argsort(-score, kind="stable")
        if y.any():
            top1.append(float(y[order[:1]].any())); top5.append(float(y[order[:5]].any()))
            pos, neg = score[y], score[~y]
            if len(neg):
                strict.append(float(pos.min() - neg.max()))
                best.append(float(pos.max() - neg.max()))
                average.append(float(pos.mean() - neg.max()))
            if y.sum() > 1:
                multi_recall.append(float((chosen & y).sum() / max(1, int(y.sum()))))
        for sid, name in ((0, "main"), (1, "reserve")):
            smask = np.asarray(record["source"]) == sid
            source[name][0] += int((chosen & smask).sum())
            source[name][1] += int((y & smask).sum())
            source[name][2] += int((chosen & smask & y).sum())
    switches = 0
    for sequence in transitions.values():
        sequence.sort(); previous = set()
        for _, current in sequence:
            if previous and current and current != previous:
                switches += 1
            previous = current
    source_metrics = {
        name: {"selected": vals[0], "positive": vals[1],
               "true_positive": vals[2],
               "precision": vals[2] / max(1, vals[0]),
               "recall": vals[2] / max(1, vals[1])}
        for name, vals in source.items()
    }
    positive_total = int(sum(np.asarray(x["label"], bool).sum() for x in records))
    return {
        "frame_units": int(len(records)),
        "candidate_rows": int(sum(len(x["label"]) for x in records)),
        "positive_rows": positive_total, "selected": selected, "tp": tp,
        "fp": fp, "fn": fn, "precision": tp / max(1, tp + fp),
        "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "top1_frame_recall": float(np.mean(top1)) if top1 else None,
        "top5_frame_recall": float(np.mean(top5)) if top5 else None,
        "strict_min_positive_margin": distribution(strict),
        "best_positive_margin": distribution(best),
        "average_positive_margin": distribution(average),
        "hard_violation_rate": float(np.mean(np.asarray(strict) < 0)) if strict else None,
        "multi_positive_frame_count": int(len(multi_recall)),
        "multi_positive_recall": float(np.mean(multi_recall)) if multi_recall else None,
        "false_positive_candidates_per_frame": float(np.mean(fp_frame)) if fp_frame else None,
        "empty_output_rate": empty / max(1, len(records)),
        "null_frame_false_acceptance": null_accept / max(1, len(records)),
        "predictions_per_gt_positive": selected / max(1, positive_total),
        "source_precision": source_metrics,
        "identity_switch_proxy": int(switches),
    }


def choose_teacher_threshold(records):
    values = np.concatenate([np.asarray(x["score"], float)
                             for x in records if len(x["score"])])
    labels = np.concatenate([np.asarray(x["label"], bool)
                             for x in records if len(x["label"])])
    if not len(values) or not labels.any():
        raise RuntimeError("calibration has no usable teacher labels")
    best = None
    for threshold in np.unique(np.quantile(values, np.linspace(.01, .995, 160))):
        chosen = values >= float(threshold)
        tp = int((chosen & labels).sum()); fp = int((chosen & ~labels).sum())
        fn = int((~chosen & labels).sum())
        p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-12, p + r)
        item = (f1, p, r, -float(threshold), float(threshold), tp, fp, fn)
        if best is None or item > best:
            best = item
    return {"threshold": best[4],
            "source": "single_L29_teacher_calibration_only_balanced_F1",
            "precision": best[1], "recall": best[2], "f1": best[0],
            "tp": best[5], "fp": best[6], "fn": best[7],
            "calibration_rows": int(len(labels)),
            "calibration_positive_rows": int(labels.sum())}
