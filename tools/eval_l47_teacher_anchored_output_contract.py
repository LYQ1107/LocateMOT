#!/usr/bin/env python3
"""Evaluate the L47 contract on held-out train videos and fixed 100 units.

Threshold fitting is performed once on a structural sample of the 12 fit
videos using frozen L29 teacher scores.  Validation and screening scores are
then evaluated with that frozen threshold; screening labels are read only for
the final, predeclared statistics and never influence a choice.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l47_teacher_anchored_output_contract import (  # noqa: E402
    L47TeacherAnchoredOutputContract,
)
from tools.l47_data import (  # noqa: E402
    FAST,
    FIT_VIDEOS,
    L28,
    L29,
    VAL_VIDEOS,
    build_l19_cache,
    build_unit,
    evenly_spaced_refs,
    fast_entries,
    fast_refs,
    hard_indices,
    pair_contract,
    load_bank,
    load_l29,
    load_queries,
    load_text,
    sha256,
)

L27_SCORE_ROOT = ROOT / "outputs/l27/fast_rmot_validation_retry/scores/A_C1_S2000"


def metric_distribution(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "q10": None, "q90": None}
    x = np.asarray(values, dtype=np.float64)
    return {
        "count": int(x.size), "mean": float(x.mean()),
        "median": float(np.median(x)), "q10": float(np.quantile(x, .10)),
        "q90": float(np.quantile(x, .90)),
    }


def l27_cache_name(entry):
    digest = hashlib.sha1(entry["expression"].encode()).hexdigest()[:12]
    return f"{entry['video']}_{digest}.npz"


def load_l27_screen_labels(entry):
    """Load immutable L27 row labels/source for formal screening comparison."""
    path = L27_SCORE_ROOT / l27_cache_name(entry)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}, path


def align_l27_labels(entry, bank, unit, teacher_score):
    cache, path = load_l27_screen_labels(entry)
    frame = int(unit["frame"])
    cache_rows = np.flatnonzero(cache["frame"] == frame)
    cache_tracks = cache["track_id"][cache_rows].astype(np.int64)
    track_to_cache = {int(track): int(row) for track, row in zip(cache_tracks, cache_rows)}
    bank_tracks = bank["track"][unit["begin"]:unit["end"]].numpy().astype(np.int64)
    if set(track_to_cache) != set(bank_tracks):
        raise RuntimeError(f"L27/cache row alignment mismatch {entry['video']}/{frame}")
    aligned = np.asarray([track_to_cache[int(track)] for track in bank_tracks], dtype=np.int64)
    labels = cache["label"][aligned].astype(bool)
    source = cache["source"][aligned].astype(np.int64)
    hard = hard_indices(labels, unit["objectness"].numpy(), teacher_score.numpy())
    correct, error = pair_contract(labels, teacher_score.numpy(), hard)
    unit["labels"] = torch.as_tensor(labels, dtype=torch.bool)
    unit["hard_indices"] = hard if isinstance(hard, torch.Tensor) else torch.as_tensor(hard, dtype=torch.long)
    unit["teacher_correct_pairs"] = correct
    unit["teacher_error_pairs"] = error
    return source, path


def score_unit(model, unit, text_hidden, text_mask, device):
    with torch.inference_mode():
        out = model(
            unit["clip"].to(device), unit["history_clip"].to(device),
            unit["numeric"].to(device),
            text_hidden[unit["query_index"]].to(device),
            text_mask[unit["query_index"]].to(device),
            unit["teacher"].to(device),
        )
    return {
        "teacher": unit["teacher"].numpy().astype(np.float32),
        "map": out["score_map"].float().cpu().numpy(),
        "residual": out["residual"].float().cpu().numpy(),
        "score": out["final_score"].float().cpu().numpy(),
        "label": unit["labels"].numpy().astype(bool),
        "objectness": unit["objectness"].numpy().astype(np.float32),
        "hard_indices": unit["hard_indices"].numpy().astype(np.int64),
        "video": unit["video"], "expression": unit["expression"],
        "query_index": unit["query_index"], "frame": unit["frame"],
    }


def output_threshold_metrics(records, key, threshold):
    selected = 0
    tp = fp = fn = 0
    empty = null_accept = 0
    fp_per_frame = []
    frame_top1 = []
    frame_top5 = []
    multi_recalls = []
    strict = []
    best = []
    average = []
    model_hard_strict = []
    teacher_correct_pairs = teacher_error_pairs = 0
    teacher_correct_flips = teacher_error_corrections = 0
    source = {0: [0, 0, 0], 1: [0, 0, 0]}
    residual_abs = []
    residual_mean = []
    scale_values = []
    offset_values = []

    for record in records:
        score = record[key]
        y = record["label"]
        chosen = score >= threshold
        selected += int(chosen.sum())
        tp += int((chosen & y).sum())
        fp += int((chosen & ~y).sum())
        fn += int((~chosen & y).sum())
        empty += int(not chosen.any())
        null_accept += int(not y.any() and chosen.any())
        fp_per_frame.append(int((chosen & ~y).sum()))
        pos = np.flatnonzero(y)
        order = np.argsort(-score, kind="stable")
        if len(pos):
            frame_top1.append(float(y[order[:1]].any()))
            frame_top5.append(float(y[order[:5]].any()))
            if len(pos) > 1:
                multi_recalls.append(float((chosen & y).sum() / len(pos)))
            hard = record["hard_indices"]
            if len(hard):
                strict.append(float(score[pos].min() - score[hard].max()))
                best.append(float(score[pos].max() - score[hard].max()))
                average.append(float(score[pos].mean() - score[hard].max()))
                negative = np.flatnonzero(~y)
                full_hard = negative[np.argsort(-score[negative], kind="stable")[:min(24, len(negative))]]
                if len(full_hard):
                    model_hard_strict.append(float(score[pos].min() - score[full_hard].max()))
                td = record["teacher"][pos, None] - record["teacher"][hard][None, :]
                sd = score[pos, None] - score[hard][None, :]
                teacher_correct = td > 0
                teacher_correct_pairs += int(teacher_correct.sum())
                teacher_error_pairs += int((~teacher_correct).sum())
                teacher_correct_flips += int((teacher_correct & (sd < 0)).sum())
                teacher_error_corrections += int((~teacher_correct & (sd > 0)).sum())
        for sid in (0, 1):
            # source is provenance only; candidate rows are intentionally not
            # available to the model and are attached by the caller.
            pool = record["source"] == sid
            source[sid][0] += int((chosen & pool).sum())
            source[sid][1] += int((y & pool).sum())
            source[sid][2] += int((chosen & pool & y).sum())
        if "residual" in record:
            residual_abs.extend(np.abs(record["residual"]).tolist())
            residual_mean.append(float(record["residual"].mean()))
        if "scale" in record:
            scale_values.append(float(record["scale"]))
            offset_values.append(float(record["offset"]))

    source_metrics = {
        ("main" if sid == 0 else "reserve"): {
            "selected": values[0], "positive": values[1],
            "true_positive": values[2],
            "precision": float(values[2] / max(1, values[0])),
            "recall": float(values[2] / max(1, values[1])),
        }
        for sid, values in source.items()
    }
    pairs = teacher_correct_pairs + teacher_error_pairs
    return {
        "frame_units": len(records),
        "candidate_rows": int(sum(len(r["label"]) for r in records)),
        "positive_rows": int(sum(int(r["label"].sum()) for r in records)),
        "selected": selected, "tp": tp, "fp": fp, "fn": fn,
        "precision": float(tp / max(1, tp + fp)),
        "recall": float(tp / max(1, tp + fn)),
        "top1_frame_recall": float(np.mean(frame_top1)) if frame_top1 else None,
        "top5_frame_recall": float(np.mean(frame_top5)) if frame_top5 else None,
        "multi_positive_frame_count": len(multi_recalls),
        "multi_positive_recall": float(np.mean(multi_recalls)) if multi_recalls else None,
        "false_positive_candidates_per_frame": float(np.mean(fp_per_frame)) if fp_per_frame else None,
        "empty_output_rate": float(empty / max(1, len(records))),
        "null_frame_false_acceptance": float(null_accept / max(1, len(records))),
        "predictions_per_gt_positive": float(selected / max(1, tp + fn)),
        "strict_min_positive_margin": metric_distribution(strict),
        "best_positive_margin": metric_distribution(best),
        "average_positive_margin": metric_distribution(average),
        "contract_hard_violation_rate": float(np.mean(np.asarray(strict) < 0)) if strict else None,
        "full_frame_model_hard_margin": metric_distribution(model_hard_strict),
        "full_frame_model_hard_violation_rate": float(np.mean(np.asarray(model_hard_strict) < 0)) if model_hard_strict else None,
        "rank_flip": {
            "pairs": pairs,
            "teacher_correct_pairs": teacher_correct_pairs,
            "teacher_error_pairs": teacher_error_pairs,
            "teacher_correct_flips": teacher_correct_flips,
            "teacher_error_corrections": teacher_error_corrections,
            "teacher_correct_flip_rate": teacher_correct_flips / max(1, teacher_correct_pairs),
            "teacher_error_correction_rate": teacher_error_corrections / max(1, teacher_error_pairs),
        },
        "source_precision": source_metrics,
        "residual": {
            "max_abs": float(max(residual_abs)) if residual_abs else None,
            "mean_abs": float(np.mean(residual_abs)) if residual_abs else None,
            "mean": float(np.mean(residual_mean)) if residual_mean else None,
            "scale_min": float(min(scale_values)) if scale_values else None,
            "scale_max": float(max(scale_values)) if scale_values else None,
            "offset_min": float(min(offset_values)) if offset_values else None,
            "offset_max": float(max(offset_values)) if offset_values else None,
        },
    }


def add_output_fields(record, output, model):
    record.update(output)
    record["source"] = record.pop("_source")
    # These are unit-level scalars, replicated only in the metrics record;
    # they do not become candidate semantic features.
    record["scale"] = float(output.get("scale", model._score_map(torch.as_tensor(record["teacher"]))[0])) if "scale" in output else float(model._score_map(torch.as_tensor(record["teacher"]))[0].detach())
    record["offset"] = float(output.get("offset", model._score_map(torch.as_tensor(record["teacher"]))[1])) if "offset" in output else float(model._score_map(torch.as_tensor(record["teacher"]))[1].detach())
    return record


def evaluate_refs(refs, banks, caches, teacher, model, text_hidden, text_mask,
                  device):
    records = []
    for query, frame_index in refs:
        bank = banks[query["video"]]
        unit = build_unit(query, frame_index, bank, caches[query["video"]],
                          teacher, text_hidden, text_mask)
        output = score_unit(model, unit, text_hidden, text_mask, device)
        begin, end = unit["begin"], unit["end"]
        output["source"] = bank["pool"][begin:end].numpy().astype(np.int64)
        # Keep scalar map diagnostics directly from the model.  Re-evaluate
        # only the scalar values through the frozen parameters, not a crop.
        with torch.inference_mode():
            raw = model(
                unit["clip"].to(device), unit["history_clip"].to(device),
                unit["numeric"].to(device), text_hidden[unit["query_index"]].to(device),
                text_mask[unit["query_index"]].to(device), unit["teacher"].to(device),
            )
        output["scale"] = float(raw["scale"].float().cpu())
        output["offset"] = float(raw["frame_offset"].float().cpu())
        records.append(output)
    return records


def make_group(videos, queries, cap, use_l28_cache=True):
    banks = {video: load_bank(video) for video in videos}
    caches = {}
    for video in videos:
        path = L28 / f"{video}.pt"
        if use_l28_cache and path.exists():
            caches[video] = torch.load(path, map_location="cpu", weights_only=False)
        else:
            caches[video] = build_l19_cache(banks[video])
    refs = evenly_spaced_refs(queries, banks, videos, cap)
    return refs, banks, caches


def fit_threshold(records):
    values = np.concatenate([r["teacher"] for r in records if len(r["teacher"])])
    labels = np.concatenate([r["label"] for r in records if len(r["label"])])
    candidates = np.unique(np.quantile(values, np.linspace(.005, .995, 256)))
    best = None
    for threshold in candidates.tolist():
        chosen = values >= threshold
        tp = int((chosen & labels).sum()); fp = int((chosen & ~labels).sum())
        fn = int((~chosen & labels).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        item = (f1, -float(threshold), float(threshold), tp, fp, fn)
        if best is None or item > best:
            best = item
    return {
        "threshold": float(best[2]), "row_f1": float(best[0]),
        "tp": best[3], "fp": best[4], "fn": best[5],
        "source": "fit-video structural calibration using frozen L29 teacher only",
        "fit_units": len(records), "screening_labels_used": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fit-calibration-cap", type=int, default=128)
    parser.add_argument("--validation-cap", type=int, default=100)
    parser.add_argument("--screening-cap", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Use a previously frozen calibration threshold; no new fit.")
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    torch.set_num_threads(1)
    text_hidden, text_mask = load_text()
    teacher = load_l29(device)
    state = torch.load(Path(args.checkpoint), map_location=device, weights_only=False)
    model = L47TeacherAnchoredOutputContract(**state["config"]["model_config"]).to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    queries = load_queries()

    fit_refs, fit_banks, fit_caches = make_group(FIT_VIDEOS, queries,
                                                  args.fit_calibration_cap)
    fit_records = evaluate_refs(fit_refs, fit_banks, fit_caches, teacher, model,
                                text_hidden, text_mask, device)
    if args.threshold is None:
        threshold = fit_threshold(fit_records)
    else:
        threshold = {
            "threshold": float(args.threshold),
            "source": "explicitly frozen L29 calibration threshold; no new fit",
            "fit_units": len(fit_records),
            "screening_labels_used": False,
            "new_threshold_search": False,
        }
    del fit_banks, fit_caches, fit_records

    val_refs, val_banks, val_caches = make_group(VAL_VIDEOS, queries,
                                                 args.validation_cap)
    val_records = evaluate_refs(val_refs, val_banks, val_caches, teacher, model,
                                text_hidden, text_mask, device)
    screen_entries = fast_entries()
    screen_videos = sorted({x["video"] for x in screen_entries if x["split"] == "screening"})
    screen_banks = {video: load_bank(video) for video in screen_videos}
    screen_caches = {video: build_l19_cache(screen_banks[video]) for video in screen_videos}
    screen_refs = fast_refs(screen_entries, screen_banks, split="screening",
                            cap=args.screening_cap)
    screen_records = evaluate_refs(screen_refs, screen_banks, screen_caches,
                                   teacher, model, text_hidden, text_mask, device)

    # Rebuild only the screening label/source views from the immutable L27
    # cache.  The score replay above is unchanged; this alignment makes the
    # formal L47 baseline exactly comparable to L46 while keeping labels out
    # of checkpoint/threshold/branch selection.
    # evaluate_refs has already produced one record per fixed ref in order.
    aligned_screen_records = []
    for record, (entry, frame_index) in zip(screen_records, screen_refs):
        bank = screen_banks[entry["video"]]
        cache = screen_caches[entry["video"]]
        unit = build_unit(entry, frame_index, bank, cache, teacher,
                          text_hidden, text_mask)
        source, cache_path = align_l27_labels(entry, bank, unit, unit["teacher"])
        record["label"] = unit["labels"].numpy().astype(bool)
        record["hard_indices"] = unit["hard_indices"].numpy().astype(np.int64)
        record["source"] = source
        record["l27_label_cache"] = str(cache_path.resolve())
        aligned_screen_records.append(record)
    screen_records = aligned_screen_records

    variants = ("teacher", "map", "score")
    results = {
        "validation": {key: output_threshold_metrics(val_records, key, threshold["threshold"])
                        for key in variants},
        "screening_100": {key: output_threshold_metrics(screen_records, key, threshold["threshold"])
                          for key in variants},
    }
    # The explicit zero-residual control is numerically identical to the
    # teacher.  Keep a separate record to make the contract auditable.
    results["validation"]["zero_residual_control"] = results["validation"]["teacher"]
    results["screening_100"]["zero_residual_control"] = results["screening_100"]["teacher"]

    l29_baseline = {"top1": 0.5152, "recall": 0.7014, "hard_violation": 0.8333,
                    "precision": 0.2937, "fp_per_frame": 4.69}
    payload = {
        "format": "locatemot-l47-teacher-anchored-evaluation-v1",
        "stage": "L47-B1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256(Path(args.checkpoint)),
        "teacher_checkpoint": str(L29.resolve()),
        "teacher_checkpoint_sha256": sha256(L29),
        "manifest": {"path": str(FAST.resolve()), "sha256": sha256(FAST),
                     "expected_sha256": "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"},
        "fit_videos": list(FIT_VIDEOS), "validation_videos": list(VAL_VIDEOS),
        "fit_calibration": threshold,
        "validation_units": len(val_records), "screening_units": len(screen_records),
        "screening_query_count": len([x for x in screen_entries if x["split"] == "screening"]),
        "screening_gt_used_only_for_final_statistics": True,
        "screening_gt_used_for_threshold_or_choice": False,
        "screening_label_provenance": {
            "root": str(L27_SCORE_ROOT.resolve()),
            "immutable": True,
            "per_record_cache_paths": sorted({r["l27_label_cache"] for r in screen_records}),
            "used_for_final_statistics_only": True,
        },
        "l29_fixed_baseline_reference": l29_baseline,
        "results": results,
        "gate": {
            "top1_tolerance": 0.02, "recall_tolerance": 0.03,
            "hard_violation_required_absolute_drop": 0.05,
            "precision_max_drop": 0.01, "fp_per_frame_max_increase": 0.10,
            "teacher_correct_flip_max_rate": 0.01,
            "multi_positive_recall_max_drop": 0.03,
        },
        "semantic_inputs_excluded": ["source_id", "pool_id", "group_id", "state_key", "query_index_as_feature"],
        "token_span_region_verified": False,
        "motion_language_decomposition": "not claimed",
        "elapsed_sec": time.time() - start,
    }
    # Formal decision is filled after the data is visible to the report; do not
    # use screening values to select a branch or threshold.
    payload["decision"] = "b1_metrics_written_pending_report_gate"
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (out.parent / "README.md").write_text(
        "# L47 B1 candidate gate\n\n"
        "The threshold was fit on fit-video labels and then frozen. Validation uses "
        "three different train videos; screening is a fixed structural 100-unit final report.\n"
    )
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
