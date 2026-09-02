#!/usr/bin/env python3
"""Calibration-only internal-dev selection for the L85 factorized sidecar.

This tool uses only the label-free compact L85 cache for model inputs.  Fit
labels are attached after a complete cached group has been scored.  It selects
one checkpoint and one global dataset-level emission rule on the registered
video-disjoint internal dev split; the fixed 16/24 evaluation is performed by
the separate L85 evaluator after this file has frozen its selection.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from locatemot.models.l85_full_rmot import L85Config, L85FullRMOT  # noqa: E402
from locatemot.rmot.l80_data import L80BankStore, key_only, load_fit_units  # noqa: E402
from locatemot.rmot.l85_fullvideo_bank import (  # noqa: E402
    EXPECTED_MANIFEST_SHA,
    MANIFEST,
    sha256_file,
)
from locatemot.rmot.l85_runtime import (  # noqa: E402
    build_groups,
    load_fit_train_dev_groups,
)


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260829
CACHE_FORMAT = "locatemot-l85-z1-semantic-cache-v1"
FORBIDDEN_LABEL_FIELDS = {
    "target_ids", "positive_indices", "positive_count", "category", "labels",
    "target_present", "candidate_present", "coverage_mask", "sidecar_candidate_gt",
    "null_target", "label_source",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=False, default=str).encode()).hexdigest()


def checkpoint_step(package: dict[str, Any], path: Path) -> int:
    if "step" in package:
        return int(package["step"])
    digits = "".join(ch if ch.isdigit() else " " for ch in path.stem).split()
    return int(digits[-1]) if digits else 0


def checkpoint_norm(package: dict[str, Any]) -> float:
    value = 0.0
    for tensor in package.get("model_state_dict", {}).values():
        if torch.is_tensor(tensor):
            value += float(tensor.float().pow(2).sum())
    return math.sqrt(value)


def history_for_final_stage(batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce the registered J-stage causal last-four-observation view."""
    history = torch.zeros_like(batch.history_observations)
    mask = torch.zeros_like(batch.history_mask)
    frames = torch.full_like(batch.history_frame_ids, -1)
    for row in range(batch.candidate_count):
        valid = torch.nonzero(batch.history_mask[row], as_tuple=False).flatten().tolist()[-4:]
        if valid:
            length = len(valid)
            history[row, :length] = batch.history_observations[row, valid]
            mask[row, :length] = True
            frames[row, :length] = batch.history_frame_ids[row, valid]
    if bool((frames[mask] > int(batch.frame_id)).any()):
        raise AssertionError(f"future history in {batch.unit_key}")
    return history, mask, frames


def load_checkpoint(path: Path, device: torch.device) -> tuple[L85FullRMOT, dict[str, Any]]:
    package = torch.load(path, map_location="cpu", weights_only=False)
    config = L85Config(**package["model_config"])
    model = L85FullRMOT(config).to(device=device, dtype=torch.float32)
    loaded = model.load_state_dict(package["model_state_dict"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise AssertionError(f"strict L85 load failed: {loaded}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    info = {
        "path": str(path.resolve()), "sha256": sha256_file(path),
        "step": checkpoint_step(package, path), "epoch": int(package.get("epoch", 0)),
        "parameter_norm": checkpoint_norm(package), "model_config": config.__dict__,
        "strict_reload": True,
    }
    return model, info


def cache_index(cache_root: Path) -> dict[str, dict[str, Any]]:
    rows = [json.loads(line) for line in (cache_root / "manifest.jsonl").read_text().splitlines() if line.strip()]
    if any(row.get("candidate_deletion") or row.get("candidate_truncation") for row in rows):
        raise AssertionError("cache manifest contains deletion/truncation")
    result = {str(row["group_key"]): row for row in rows}
    if len(result) != len(rows):
        raise AssertionError("duplicate cache group key")
    return result


def score_group(item: dict[str, Any], group: dict[str, Any], store: L80BankStore,
                model: L85FullRMOT, device: torch.device) -> list[dict[str, Any]]:
    if item.get("format") != "locatemot-l85-z1-semantic-group-v1" or item.get("labels_in_cache"):
        raise AssertionError(f"invalid label-free cache item {item.get('group_key')}")
    query_keys = [str(value) for value in item["query_unit_keys"]]
    if len(query_keys) != int(item["z1"].shape[0]) or query_keys != [str(row["unit_key"]) for row in group["queries"]]:
        raise AssertionError(f"cache query order drift {item['group_key']}")
    batches = [store.build_unit(key_only(row)) for row in group["queries"]]
    first = batches[0]
    if first.candidate_count != int(item["candidate_count"]):
        raise AssertionError(f"candidate count drift {item['group_key']}")
    first_structure = [(key[0], key[1], key[3], key[4], key[5]) for key in first.row_keys]
    for batch in batches:
        structure = [(key[0], key[1], key[3], key[4], key[5]) for key in batch.row_keys]
        if structure != first_structure:
            raise AssertionError(f"candidate structure drift {batch.unit_key}")
        if int((batch.history_frame_ids > int(batch.frame_id)).sum()) != 0:
            raise AssertionError(f"future history in {batch.unit_key}")
    history, history_mask, history_frames = history_for_final_stage(first)
    z1 = item["z1"].float().clone().to(device=device)
    presence_input = torch.cat((item["text_global"].float(), item["frame_global"].float()), dim=-1).clone().to(device=device)
    current = first.observations.float().clone().to(device=device)
    history = history.float().clone().to(device=device)
    history_mask = history_mask.clone().to(device=device)
    history_frames = history_frames.clone().to(device=device)
    with torch.inference_mode():
        output = model(z1, presence_input, current, history, history_mask, history_frames,
                       int(first.frame_id), temporal_enabled=True)
    scores = output["membership"].float().cpu().numpy()
    presence = output["presence"].float().cpu().numpy()
    null = output["null_logit"].float().cpu().numpy()
    if not (np.isfinite(scores).all() and np.isfinite(presence).all() and np.isfinite(null).all()):
        raise FloatingPointError(f"nonfinite dev score {item['group_key']}")
    records = []
    for index, batch in enumerate(batches):
        records.append({
            "format": "locatemot-l85-dev-score-v1", "unit_key": str(batch.unit_key),
            "dataset": str(batch.dataset), "video": str(batch.video), "query_id": int(batch.query_id),
            "frame_id": int(batch.frame_id), "group_key": str(item["group_key"]),
            "candidate_count": int(batch.candidate_count), "row_offsets": [int(x) for x in batch.row_offsets],
            "row_keys": [list(key) for key in batch.row_keys],
            "candidate_indices": [int(x) for x in batch.candidate_indices],
            "track_ids": [int(x) for x in batch.track_ids], "pool_ids": [int(x) for x in batch.pool_ids],
            "score": scores[index].astype(np.float64).tolist(),
            "presence": float(presence[index]), "null_logit": float(null[index]),
            "history_future_rows": int((batch.history_frame_ids > int(batch.frame_id)).sum()),
            "candidate_rows_retained": int(batch.candidate_count),
            "candidate_deletion": False, "candidate_truncation": False, "labels_attached": False,
            "finite_scores": True,
        })
    del output, z1, presence_input, current, history, history_mask, history_frames, batches
    return records


def score_group_reuse_first(item: dict[str, Any], group: dict[str, Any], store: L80BankStore,
                            model: L85FullRMOT, device: torch.device) -> list[dict[str, Any]]:
    """Score a full-video frame without rebuilding identical bank history per query.

    All queries in a frame share the query-independent L69 candidate rows and
    causal observation history.  The normal dev scorer retains its exhaustive
    per-query contract checks; this full-video adapter performs the equivalent
    frame identity check once, then derives each query's row key by changing
    only its query id.  Text/Z1 outputs remain per query and every row is
    still emitted in native order.
    """
    if item.get("format") != "locatemot-l85-z1-semantic-group-v1" or item.get("labels_in_cache"):
        raise AssertionError(f"invalid label-free frame item {item.get('group_key')}")
    rows = [dict(row) for row in group["queries"]]
    query_keys = [str(value) for value in item["query_unit_keys"]]
    if len(query_keys) != int(item["z1"].shape[0]) or query_keys != [str(row["unit_key"]) for row in rows]:
        raise AssertionError(f"frame query order drift {item['group_key']}")
    if not rows:
        raise AssertionError(f"empty frame group {item['group_key']}")
    first = store.build_unit(key_only(rows[0]))
    if int(first.candidate_count) != int(item["candidate_count"]):
        raise AssertionError(f"candidate count drift {item['group_key']}")
    first_structure = [(key[0], key[1], key[3], key[4], key[5]) for key in first.row_keys]
    for row in rows:
        if (str(row["dataset"]), str(row["video"]), int(row["frame_id"])) != (
                str(first.dataset), str(first.video), int(first.frame_id)):
            raise AssertionError(f"frame identity drift {row['unit_key']}")
    if int((first.history_frame_ids > int(first.frame_id)).sum()) != 0:
        raise AssertionError(f"future history in {first.unit_key}")
    history, history_mask, history_frames = history_for_final_stage(first)
    z1 = item["z1"].float().clone().to(device=device)
    presence_input = torch.cat((item["text_global"].float(), item["frame_global"].float()), dim=-1).clone().to(device=device)
    current = first.observations.float().clone().to(device=device)
    history = history.float().clone().to(device=device)
    history_mask = history_mask.clone().to(device=device)
    history_frames = history_frames.clone().to(device=device)
    with torch.inference_mode():
        output = model(z1, presence_input, current, history, history_mask, history_frames,
                       int(first.frame_id), temporal_enabled=True)
    scores = output["membership"].float().cpu().numpy()
    presence = output["presence"].float().cpu().numpy()
    null = output["null_logit"].float().cpu().numpy()
    if not (np.isfinite(scores).all() and np.isfinite(presence).all() and np.isfinite(null).all()):
        raise FloatingPointError(f"nonfinite full-video score {item['group_key']}")
    records = []
    for index, row in enumerate(rows):
        row_keys = [(str(first.dataset), str(first.video), int(row["query_id"]), int(first.frame_id),
                     str(first.bank_path), int(offset)) for offset in first.row_offsets]
        if [(key[0], key[1], key[3], key[4], key[5]) for key in row_keys] != first_structure:
            raise AssertionError(f"candidate structure drift {row['unit_key']}")
        values = scores[index]
        records.append({
            "format": "locatemot-l85-dev-score-v1", "unit_key": str(row["unit_key"]),
            "dataset": str(row["dataset"]), "video": str(row["video"]), "query_id": int(row["query_id"]),
            "frame_id": int(first.frame_id), "group_key": str(item["group_key"]),
            "candidate_count": int(first.candidate_count), "row_offsets": [int(x) for x in first.row_offsets],
            "row_keys": [list(key) for key in row_keys], "candidate_indices": [int(x) for x in first.candidate_indices],
            "track_ids": [int(x) for x in first.track_ids], "pool_ids": [int(x) for x in first.pool_ids],
            "score": values.astype(np.float64).tolist(), "presence": float(presence[index]),
            "null_logit": float(null[index]), "history_future_rows": int((first.history_frame_ids > int(first.frame_id)).sum()),
            "candidate_rows_retained": int(first.candidate_count), "candidate_deletion": False,
            "candidate_truncation": False, "labels_attached": False, "finite_scores": True,
        })
    del output, z1, presence_input, current, history, history_mask, history_frames, first
    return records


def attach_label(record: dict[str, Any], full: dict[str, Any], store: L80BankStore) -> dict[str, Any]:
    batch = store.build_unit(key_only(full))
    sidecar = json.loads((Path(batch.bank_path).with_suffix(".labels.json")).read_text())
    candidate_gt = sidecar.get("candidate_gt")
    if not isinstance(candidate_gt, list) or max(record["row_offsets"], default=-1) >= len(candidate_gt):
        raise AssertionError(f"sidecar contract failure {record['unit_key']}")
    targets = {str(value) for value in full.get("target_ids", [])}
    labels = [bool(candidate_gt[int(offset)] is not None and str(candidate_gt[int(offset)]) in targets)
              for offset in record["row_offsets"]]
    target_present = bool(targets)
    candidate_present = bool(any(labels))
    category = ("inactive" if not target_present else
                "present_uncovered" if not candidate_present else
                "multi_positive" if sum(labels) > 1 else "positive")
    result = dict(record)
    result.update({
        "labels": labels, "positive_count": int(sum(labels)), "positive_indices": [i for i, x in enumerate(labels) if x],
        "target_ids": sorted(targets), "target_present": target_present,
        "candidate_present": candidate_present, "category": category,
        "coverage_mask": not (target_present and not candidate_present),
        "sidecar_labels_loaded": True, "label_source": str(Path(batch.bank_path).with_suffix(".labels.json").resolve()),
        "labels_attached_after_scoring": True,
    })
    if len(labels) != int(record["candidate_count"]):
        raise AssertionError(f"label length drift {record['unit_key']}")
    return result


def metric(records: list[dict[str, Any]], candidate_threshold: float,
           presence_threshold: float, null_margin: float) -> dict[str, Any]:
    tp = fp = fn = selected_rows = positive_rows = 0
    top1 = top5 = top_units = empty = 0
    hard: list[bool] = []
    strict: list[float] = []
    best: list[float] = []
    average: list[float] = []
    multi_hit: list[float] = []
    multi_exact: list[float] = []
    inactive = inactive_accept = inactive_fp_rows = 0
    present = present_uncovered = 0
    score_values: list[float] = []
    for row in records:
        scores = np.asarray(row["score"], dtype=np.float64)
        labels = np.asarray(row["labels"], dtype=bool)
        if scores.shape != labels.shape or not np.isfinite(scores).all():
            raise AssertionError(f"score/label drift {row['unit_key']}")
        accepted = (float(row["presence"]) >= float(presence_threshold) and
                    float(row["presence"]) - float(row["null_logit"]) >= float(null_margin))
        selected = (scores >= float(candidate_threshold))
        if not accepted:
            selected = np.zeros_like(selected, dtype=bool)
        tp += int((selected & labels).sum()); fp += int((selected & ~labels).sum()); fn += int((~selected & labels).sum())
        selected_rows += int(selected.sum()); positive_rows += int(labels.sum())
        score_values.extend(scores.tolist())
        category = str(row["category"])
        if category == "inactive":
            inactive += 1; inactive_accept += int(bool(selected.any())); inactive_fp_rows += int((selected & ~labels).sum())
        elif category == "present_uncovered":
            present_uncovered += 1
        else:
            present += 1
        if labels.any():
            order = np.argsort(-scores, kind="stable")
            top1 += int(bool(labels[order[:1]].any())); top5 += int(bool(labels[order[:5]].any())); top_units += 1
        empty += int(not selected.any())
        pos = np.flatnonzero(labels); neg = np.flatnonzero(~labels)
        if len(pos) and len(neg):
            strict_value = float(scores[pos].min() - scores[neg].max())
            strict.append(strict_value); best.append(float(scores[pos].max() - scores[neg].max()))
            average.append(float(scores[pos].mean() - scores[neg].max())); hard.append(strict_value < 0.0)
        if len(pos) > 1:
            multi_hit.append(float(selected[pos].sum() / len(pos))); multi_exact.append(float(selected[pos].all()))
    def stats(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "mean": None, "p50": None, "p90": None}
        arr = np.asarray(values, dtype=np.float64)
        return {"count": int(arr.size), "mean": float(arr.mean()), "p50": float(np.quantile(arr, .5)),
                "p90": float(np.quantile(arr, .9))}
    return {
        "units": len(records), "candidate_rows": int(sum(len(x["labels"]) for x in records)),
        "positive_rows": positive_rows, "selected_rows": selected_rows,
        "true_positive_rows": tp, "false_positive_rows": fp, "false_negative_rows": fn,
        "candidate_precision": tp / max(1, selected_rows), "candidate_recall": tp / max(1, tp + fn),
        "fp_per_frame": fp / max(1, len(records)), "predictions_per_positive": selected_rows / max(1, positive_rows),
        "top1": top1 / max(1, top_units), "top5": top5 / max(1, top_units),
        "hard_violation": float(np.mean(hard)) if hard else None,
        "strict_margin": stats(strict), "best_margin": stats(best), "average_margin": stats(average),
        "multi_positive_recall": float(np.mean(multi_hit)) if multi_hit else None,
        "multi_target_exact": float(np.mean(multi_exact)) if multi_exact else None,
        "minimum_positive_coverage": float(np.mean(multi_hit)) if multi_hit else None,
        "empty_rate": empty / max(1, len(records)), "inactive_units": inactive,
        "inactive_false_acceptance": inactive_accept / max(1, inactive),
        "inactive_false_positive_rows": inactive_fp_rows, "present_units": present,
        "present_uncovered_units": present_uncovered, "score_distribution": stats(score_values),
        "candidate_threshold": float(candidate_threshold), "presence_threshold": float(presence_threshold),
        "null_margin": float(null_margin), "candidate_rows_retained": True,
        "candidate_deletion": False, "candidate_truncation": False,
    }


def threshold_grid(start: float = -2.0, stop: float = 2.0, step: float = .1) -> list[float]:
    return [round(float(x), 10) for x in np.arange(start, stop + step * .5, step)]


def fit_dev_rule(records: list[dict[str, Any]], candidates: list[float], null_margins: list[float]) -> dict[str, Any]:
    """Search the fixed global rule without repeatedly rebuilding records."""
    scores = np.concatenate([np.asarray(row["score"], dtype=np.float64) for row in records])
    labels = np.concatenate([np.asarray(row["labels"], dtype=bool) for row in records])
    unit_ids = np.concatenate([np.full(len(row["score"]), index, dtype=np.int64)
                               for index, row in enumerate(records)])
    presence = np.asarray([float(row["presence"]) for row in records], dtype=np.float64)
    null = np.asarray([float(row["null_logit"]) for row in records], dtype=np.float64)
    best: dict[str, Any] | None = None
    best_key: tuple[float, int, float, float, float] | None = None
    for presence_threshold in candidates:
        for null_margin in null_margins:
            unit_gate = ((presence >= float(presence_threshold)) &
                         (presence - null >= float(null_margin)))
            row_gate = unit_gate[unit_ids]
            for candidate_threshold in candidates:
                selected = (scores >= float(candidate_threshold)) & row_gate
                tp = int((selected & labels).sum())
                fp = int((selected & ~labels).sum())
                fn = int((~selected & labels).sum())
                f1 = 2.0 * tp / max(1.0, 2.0 * tp + fp + fn)
                # The threshold rule is a fixed observed-score F1 rule.  The
                # tie order is explicit and independent of the later model
                # selection tuple.
                key = (float(f1), -fp, float(candidate_threshold),
                       float(presence_threshold), float(null_margin))
                if best_key is None or key > best_key:
                    best_key = key
                    best = {"candidate_threshold": float(candidate_threshold),
                            "presence_threshold": float(presence_threshold),
                            "null_margin": float(null_margin), "tp": tp, "fp": fp,
                            "fn": fn, "f1": float(f1),
                            "objective": "exact observed candidate-level F1 on internal dev",
                            "tie_rule": "higher F1, fewer FP rows, higher candidate threshold, then higher presence threshold, then higher null margin"}
    assert best is not None
    return best


def selection_key(value: dict[str, Any]) -> tuple[Any, ...]:
    metric_value = value["metrics"]
    return (
        float(metric_value["hard_violation"] if metric_value["hard_violation"] is not None else 1.0),
        -float(metric_value["top1"]),
        -float(metric_value["multi_target_exact"] if metric_value["multi_target_exact"] is not None else 0.0),
        float(metric_value["inactive_false_acceptance"]),
        int(value["step"]),
        float(value["parameter_norm"]),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L85 dev output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable] + sys.argv)
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST_SHA:
        raise AssertionError("fixed manifest SHA drift")
    cache_root = (args.cache if args.cache.is_absolute() else ROOT / args.cache).resolve()
    summary = json.loads((cache_root / "summary.json").read_text())
    if summary.get("format") != CACHE_FORMAT or summary.get("status") != "complete" or summary.get("labels_in_cache"):
        raise AssertionError("invalid L85 semantic cache")
    cache_rows = cache_index(cache_root)
    groups, train_keys, dev_keys = load_fit_train_dev_groups()
    fit_rows = load_fit_units()
    labels_by_key = {str(row["unit_key"]): row for row in fit_rows}
    if len(labels_by_key) != 5314:
        raise AssertionError("fit label count drift")
    checkpoint_specs: list[tuple[str, Path]] = []
    for value in args.checkpoint:
        if "=" not in value:
            raise ValueError("--checkpoint requires NAME=PATH")
        name, path_value = value.split("=", 1)
        path = Path(path_value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint_specs.append((str(name), path))
    if not checkpoint_specs or len({name for name, _ in checkpoint_specs}) != len(checkpoint_specs):
        raise AssertionError("checkpoint names missing/duplicated")
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device)
    thresholds = threshold_grid()
    null_margins = [0.0, .25, .50, .75, 1.0]
    all_candidates: list[dict[str, Any]] = []
    dev_trace: list[dict[str, Any]] = []
    store = L80BankStore(max_history=8)
    try:
        for name, path in checkpoint_specs:
            model, info = load_checkpoint(path, device)
            raw_records: list[dict[str, Any]] = []
            for group_key in dev_keys:
                if group_key not in cache_rows:
                    raise KeyError(f"dev cache group missing: {group_key}")
                item = torch.load(cache_rows[group_key]["path"], map_location="cpu", weights_only=False)
                current = score_group(item, groups[group_key], store, model, device)
                for record in current:
                    if record["unit_key"] not in labels_by_key:
                        raise KeyError(record["unit_key"])
                    raw_records.append(attach_label(record, labels_by_key[record["unit_key"]], store))
                del item, current
            if len(raw_records) != sum(len(groups[key]["queries"]) for key in dev_keys):
                raise AssertionError("dev group score count drift")
            rule = fit_dev_rule(raw_records, thresholds, null_margins)
            best_for_checkpoint = {"checkpoint": name, "step": int(info["step"]),
                                   "parameter_norm": float(info["parameter_norm"]),
                                   "metrics": metric(raw_records, rule["candidate_threshold"],
                                                      rule["presence_threshold"], rule["null_margin"]),
                                   "rule_fit": rule}
            best_for_checkpoint["checkpoint_info"] = info
            all_candidates.append(best_for_checkpoint)
            dev_trace.append({"checkpoint": name, "checkpoint_info": info,
                              "selected_rule": {key: best_for_checkpoint["metrics"][key]
                                                for key in ("candidate_threshold", "presence_threshold", "null_margin")},
                              "metrics": best_for_checkpoint["metrics"],
                              "dev_units": len(raw_records), "dev_group_count": len(dev_keys),
                              "labels_attached_after_scores": True})
            del model, raw_records
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        selected = sorted(all_candidates, key=selection_key)[0]
        selection = {
            "format": "locatemot-l85-dev-selection-v1", "status": "complete",
            "selection_source": "video-disjoint internal dev labels only",
            "selection_tuple": ["lower target-bag hard violation", "higher target-bag hit@1",
                                "higher multi-target exact", "lower inactive false acceptance",
                                "higher dev full-video HOTA (unavailable at this selection pass)", "earlier epoch",
                                "smaller parameter norm"],
            "selected": selected, "candidates": all_candidates,
            "threshold_grid": {"candidate": thresholds, "presence": thresholds, "null_margin": null_margins},
            "threshold_objective": "fixed global rule; dev target-bag F1 proxy with registered diagnostics and simpler-rule tie",
            "dev_full_video_hota_used_for_selection": False,
            "dev_full_video_hota_reason": "full-video TrackEval is run as a separate frozen diagnostic after this selection; no sparse-unit surrogate is called full-video",
            "train_group_count": len(train_keys), "dev_group_count": len(dev_keys),
            "fit_labels_only_for_dev": True, "fixed_validation_read": False,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        }
        write_json(out / "checkpoint_selection.json", selection)
        write_json(out / "dev_metrics.json", {"format": "locatemot-l85-dev-metrics-v1", "status": "complete",
                  "command": command, "seed": SEED, "dev_trace": dev_trace,
                  "manifest_sha256": sha256_file(MANIFEST), "cache_summary_sha256": sha256_file(cache_root / "summary.json"),
                  "candidate_deletion": False, "candidate_truncation": False,
                  "screening_gt_used": False, "official_test_labels_read": False,
                  "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        write_json(out / "provenance.json", {"format": "locatemot-l85-dev-provenance-v1", "status": "complete",
                  "command": command, "cwd": str(ROOT), "luna_thread": THREAD, "seed": SEED,
                  "inputs": {"cache": str(cache_root), "cache_summary_sha256": sha256_file(cache_root / "summary.json"),
                             "fit_units": str(ROOT / "outputs/l49/data/train_units.jsonl"),
                             "split": str(ROOT / "outputs/l82/protocol/fit_video_train_dev_split.json"),
                             "manifest": str(MANIFEST), "manifest_sha256": sha256_file(MANIFEST)},
                  "outputs": [str(out / name) for name in ("checkpoint_selection.json", "dev_metrics.json")],
                  "label_attachment": "after complete label-free cache group scores",
                  "candidate_set": "complete L69 rows; no top-k/NMS/deletion", "history_stage": "J last four causal observations",
                  "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
                  "screening_gt_used": False, "official_test_labels_read": False,
                  "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                  "dev_full_video_hota_used_for_selection": False})
        write_json(out / "status.json", {"format": "locatemot-l85-dev-status-v1", "status": "complete",
                  "selected_checkpoint": selected["checkpoint"], "selected_step": int(selected["step"]),
                  "next_action": "freeze dev selection, then run fixed internal validation and full-video TrackEval diagnostics",
                  "screening_gt_used": False, "official_test_labels_read": False,
                  "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        return selection
    except Exception:
        write_json(out / "status.json", {"format": "locatemot-l85-dev-status-v1", "status": "incomplete",
                  "command": command, "failure_root_cause": "first traceback from invoking process",
                  "screening_gt_used": False, "official_test_labels_read": False,
                  "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        (out / "INCOMPLETE.md").write_text("# L85 dev selection — INCOMPLETE\n\nThe invoking process retained the first traceback.\n")
        raise
    finally:
        store._store._bank = None; store._store._text_cache = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
