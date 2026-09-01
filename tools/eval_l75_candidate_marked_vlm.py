#!/usr/bin/env python3
"""L75 fixed checkpoint-selection and 16-calibration/24-validation evaluator.

The large LocateAnything model is loaded once and remains frozen.  Candidate
rows are rebuilt from the L69 frame pointer, and every row is scored in small
chunks; no score, proposal, or candidate tensor is cached on disk.  The
checkpoint is selected from calibration evidence only.  Validation unit
labels are not joined until that selection has been written in memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
sys.path.insert(0, str(ROOT))

from locatemot.models.l75_candidate_marked_vlm import CandidateMarkedVLMMatcher  # noqa: E402
from locatemot.rmot.l75_data import (  # noqa: E402
    IMAGE_ROOT,
    L62_RECORDS,
    L75Bank,
    MANIFEST_PATH,
    MANIFEST_SHA256,
    make_record,
    sha256_file,
    unit_key,
)
from locatemot.rmot.l75_runtime import (  # noqa: E402
    attach_language_lora,
    frozen_target_digest,
    language_forward,
    load_locateanything,
    load_lora_state_dict,
    marked_visual_batch,
    prepare_visual,
    region_value_batch,
)

SEED = 20260829
DATASETS = ("refer_kitti_v1", "refer_kitti_v2")
L29_THRESHOLD = -1.030576229095459
EXPECTED_L29 = {
    "candidate_recall": 0.7333333333333333,
    "candidate_precision": 0.0830188679245283,
    "fp_per_frame": 10.125,
    "predictions_per_positive": 8.833333333333334,
    "hard_violation": 0.9166666666666666,
    "multi_positive_recall": 0.8194444444444443,
}
TRAIN_DIR = ROOT / "outputs/l75/train/joint_fit5000_attempt3"
DEFAULT_OUT = ROOT / "outputs/l75/eval/semantic_16cal24val"
CHECKPOINT_STEPS = (500, 1000, 2000, 5000)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None,
                "max": None, "p50": None, "p95": None}
    return {
        "count": int(array.size), "mean": float(array.mean()),
        "std": float(array.std()), "min": float(array.min()),
        "max": float(array.max()), "p50": float(np.quantile(array, .50)),
        "p95": float(np.quantile(array, .95)),
    }


def fit_threshold(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    values = np.unique(np.concatenate([
        np.asarray(row[score_key], dtype=np.float64) for row in rows
    ]))
    if values.size == 0 or not np.isfinite(values).all():
        raise AssertionError("calibration score values are empty/nonfinite")
    candidates = values.tolist() + [float(values.min()) - 1e-6,
                                    float(values.max()) + 1e-6]
    best: tuple[tuple[float, int, float], float] | None = None
    best_counts: tuple[int, int, int] | None = None
    for threshold in candidates:
        tp = fp = fn = 0
        for row in rows:
            score = np.asarray(row[score_key], dtype=np.float64)
            label = np.asarray(row["labels"], dtype=bool)
            selected = score >= float(threshold)
            tp += int((selected & label).sum())
            fp += int((selected & ~label).sum())
            fn += int((~selected & label).sum())
        f1 = 2.0 * tp / max(1.0, 2.0 * tp + fp + fn)
        # Registered rule: maximize candidate F1, then minimize FP, then use
        # the higher observed threshold.  No validation value is considered.
        key = (f1, -fp, float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold))
            best_counts = (tp, fp, fn)
    assert best is not None and best_counts is not None
    return {
        "threshold": best[1],
        "objective": "candidate-level F1 over all calibration candidate rows",
        "tie_rule": "higher F1, fewer FP, then higher threshold",
        "counts": {"tp": best_counts[0], "fp": best_counts[1], "fn": best_counts[2]},
        "calibration_units": len(rows),
        "validation_used": False,
    }


def metric(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Candidate-only frame metric; NULL is deliberately not a filter."""
    tp = fp = fn = selected = positives = empty = 0
    top1 = top5 = 0
    target_present_units = candidate_present_units = 0
    present_uncovered_units = 0
    inactive_false_acceptance = 0
    inactive_fp_rows = 0
    strict: list[float] = []
    best: list[float] = []
    average: list[float] = []
    violations: list[bool] = []
    multi: list[float] = []
    values: list[float] = []
    null_values: list[float] = []
    complete = True
    for row in rows:
        score = np.asarray(row["scores"], dtype=np.float64)
        label = np.asarray(row["labels"], dtype=bool)
        if score.size != label.size or not np.isfinite(score).all():
            raise AssertionError(f"score/label length or finite failure: {row['unit_key']}")
        if len(row["candidate_keys"]) != score.size:
            raise AssertionError(f"candidate key length failure: {row['unit_key']}")
        values.extend(score.tolist())
        if row.get("absent_logit") is not None:
            absent = float(row["absent_logit"])
            if not math.isfinite(absent):
                raise AssertionError(f"nonfinite absent logit: {row['unit_key']}")
            null_values.append(absent)
        target_present = bool(row.get("target_present", bool(label.any())))
        candidate_present = bool(row.get("candidate_present", bool(label.any())))
        if target_present:
            target_present_units += 1
        if candidate_present:
            candidate_present_units += 1
        if row.get("category") == "present_uncovered":
            present_uncovered_units += 1
        selected_mask = score >= float(threshold)
        row_tp = int((selected_mask & label).sum())
        row_fp = int((selected_mask & ~label).sum())
        row_fn = int((~selected_mask & label).sum())
        tp += row_tp; fp += row_fp; fn += row_fn
        selected += int(selected_mask.sum()); positives += int(label.sum())
        empty += int(not selected_mask.any())
        if row.get("category") == "inactive":
            inactive_false_acceptance += int(selected_mask.any())
            inactive_fp_rows += row_fp
        positive = np.flatnonzero(label)
        negative = np.flatnonzero(~label)
        if target_present and positive.size:
            order = np.argsort(-score, kind="stable")
            top1 += int(bool(label[order[:1]].any()))
            top5 += int(bool(label[order[:5]].any()))
        if positive.size and negative.size:
            negative_max = float(score[negative].max())
            strict_value = float(score[positive].min() - negative_max)
            strict.append(strict_value)
            best.append(float(score[positive].max() - negative_max))
            average.append(float(score[positive].mean() - negative_max))
            violations.append(strict_value < 0.0)
        if positive.size > 1:
            multi.append(float((selected_mask & label).sum() / positive.size))
        complete = complete and bool(row.get("candidate_keys_complete", False))
    inactive_units = sum(row.get("category") == "inactive" for row in rows)
    return {
        "units": len(rows),
        "candidate_rows": int(sum(len(row["labels"]) for row in rows)),
        "positive_rows": int(positives),
        "target_present_units": int(target_present_units),
        "candidate_present_units": int(candidate_present_units),
        "candidate_coverage": candidate_present_units / max(1, target_present_units),
        "present_uncovered_units": int(present_uncovered_units),
        "top1": top1 / max(1, candidate_present_units),
        "top5": top5 / max(1, candidate_present_units),
        "top1_denominator_candidate_present_units": int(candidate_present_units),
        "top5_denominator_candidate_present_units": int(candidate_present_units),
        "candidate_precision": tp / max(1, selected),
        "candidate_recall": tp / max(1, tp + fn),
        "fp_per_frame": fp / max(1, len(rows)),
        "predictions_per_positive": selected / max(1, positives),
        "selected_rows": int(selected),
        "false_positive_rows": int(fp),
        "hard_violation": float(np.mean(violations)) if violations else None,
        "strict_margin": stats(strict),
        "best_margin": stats(best),
        "average_margin": stats(average),
        "multi_positive_recall": float(np.mean(multi)) if multi else None,
        "multi_positive_units": len(multi),
        "empty_rate": empty / max(1, len(rows)),
        "inactive_false_acceptance": inactive_false_acceptance / max(1, inactive_units),
        "inactive_false_positive_rows": int(inactive_fp_rows),
        "inactive_units": int(inactive_units),
        "score_mean": float(np.mean(values)) if values else None,
        "score_std": float(np.std(values)) if values else None,
        "score_stats": stats(values),
        "absent_logit_stats_diagnostic": stats(null_values),
        "threshold": float(threshold),
        "complete_finite_keys": bool(complete),
        "null_head_used_for_primary": False,
        "candidate_filter_used": False,
    }


def make_old_control_row(old: dict[str, Any], field: str = "l29") -> dict[str, Any]:
    labels = [bool(value) for value in old["label"]]
    category = str(old.get("category", "unknown"))
    return {
        "unit_key": str(old["unit_key"]), "dataset": str(old["dataset"]),
        "video": str(old["video"]), "frame_id": int(old["frame_id"]),
        "category": category, "labels": labels,
        "target_present": category != "inactive" or bool(any(labels)),
        "candidate_present": bool(any(labels)),
        "candidate_keys": [[str(old["unit_key"]), i] for i in range(len(labels))],
        "scores": [float(value) for value in old[field]],
        "absent_logit": float(old.get("null_logit", 0.0)),
        "candidate_keys_complete": True,
    }


def read_unit_keys(path: Path) -> list[str]:
    keys = []
    for line in path.read_text().splitlines():
        if line.strip():
            # Extract only the immutable ordering key before checkpoint
            # selection; do not materialize validation labels at this point.
            match = re.search(r'"unit_key"\s*:\s*"([^"]+)"', line)
            if match is None:
                raise AssertionError(f"missing unit_key in {path}")
            keys.append(match.group(1))
    return keys


def read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checkpoint_norm(package: dict[str, Any]) -> float:
    total = 0.0
    for value in package.get("matcher", {}).values():
        if torch.is_tensor(value):
            total += float(value.float().pow(2).sum())
    for value in package.get("lora", {}).values():
        if torch.is_tensor(value):
            total += float(value.float().pow(2).sum())
    return math.sqrt(total)


def load_checkpoint_package(path: Path) -> dict[str, Any]:
    package = torch.load(path, map_location="cpu", weights_only=False)
    if package.get("format") != "locatemot-l75-adapter-only-v1":
        raise AssertionError(f"unexpected checkpoint format: {path}")
    required = {"matcher", "lora", "lora_contract", "step", "config"}
    missing = sorted(required.difference(package))
    if missing:
        raise AssertionError(f"checkpoint missing keys {missing}: {path}")
    contract = package["lora_contract"]
    if int(contract.get("rank", -1)) != 8 or float(contract.get("alpha", -1)) != 16.0:
        raise AssertionError(f"checkpoint LoRA contract drift: {path}")
    return package


def load_unit_for_eval(unit: dict[str, Any], bank: L75Bank,
                       include_labels: bool = True) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the complete row mapping before joining labels."""
    record = make_record(unit, bank, include_labels=include_labels)
    image_path = IMAGE_ROOT / str(unit["video"]) / f"{int(unit['frame_id']):06d}.png"
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    image = Image.open(image_path).convert("RGB")
    if record.get("image_size_declared") and record["image_size_declared"] != [image.width, image.height]:
        raise AssertionError(f"image size mismatch {record['unit_key']}")
    rows = record["row_offsets"]
    boxes = bank.tensors["box"].index_select(
        0, torch.as_tensor(rows, dtype=torch.long)
    ).float().tolist()
    # This call only consumes box geometry and the expression.  Labels have
    # already been joined in record, but are not passed to representation code.
    return record, {"image": image, "boxes": boxes}


def score_prepared(model: Any, matcher: Any, prepared: dict[str, Any],
                   candidate_count: int, chunk: int) -> tuple[list[float], float]:
    base_visual = prepared["base_visual"].to("cuda:0")
    scores: list[float] = []
    absent_values: list[float] = []
    cells = prepared["candidate_cells"]
    if len(cells) != int(candidate_count):
        raise AssertionError("prepared candidate count drift")
    for start in range(0, candidate_count, int(chunk)):
        local_cells = cells[start:start + int(chunk)]
        with torch.inference_mode():
            marked, _ = marked_visual_batch(base_visual, local_cells, matcher.region_marker)
            regions, region_mask = region_value_batch(marked, local_cells)
            hidden = language_forward(model, prepared, marked, inference=True)
            output = matcher(hidden, prepared["expression_positions"], regions, region_mask)
            current = output["match_logit"].float().cpu().tolist()
            absent = output["absent_logit"].float().cpu().reshape(-1).tolist()
        if len(current) != len(local_cells) or len(absent) != len(local_cells):
            raise AssertionError("model output/candidate chunk drift")
        scores.extend(float(value) for value in current)
        absent_values.extend(float(value) for value in absent)
        del marked, regions, region_mask, hidden, output
    if len(scores) != candidate_count or not np.isfinite(np.asarray(scores)).all():
        raise AssertionError("nonfinite or incomplete candidate scores")
    # The matcher absent head is candidate-batch repeated.  The mean is only
    # diagnostic; it is never used to suppress candidate rows.
    absent_logit = float(np.mean(absent_values)) if absent_values else 0.0
    if not math.isfinite(absent_logit):
        raise AssertionError("nonfinite absent diagnostic")
    del base_visual
    return scores, absent_logit


def evaluate_units(model: Any, matcher: Any, units: list[dict[str, Any]],
                   include_labels: bool, chunk: int,
                   fixed_old: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    for unit in units:
        video = str(unit["video"])
        bank = L75Bank(video)
        try:
            record, image_payload = load_unit_for_eval(unit, bank, include_labels=include_labels)
            prepared = prepare_visual(model, PROCESSOR, TOKENIZER,
                                      image_payload["image"], record["sentence"], image_payload["boxes"])
            # PROCESSOR/TOKENIZER are assigned by main.  The explicit globals
            # keep this helper free of a second model/processor construction.
            scores, absent_logit = score_prepared(model, matcher, prepared,
                                                  record["candidate_count"], chunk)
            row = {
                "format": "locatemot-l75-score-record-v1",
                "unit_key": record["unit_key"], "dataset": record["dataset"],
                "video": record["video"], "query_id": record["query_id"],
                "frame_id": record["frame_id"], "category": record["category"],
                "declared_category": record["declared_category"],
                "candidate_count": record["candidate_count"],
                "candidate_keys": record["row_keys"],
                "candidate_index_provenance": record["candidate_index_provenance"],
                "track_id_provenance": record["track_id_provenance"],
                "pool_id_provenance": record["pool_id_provenance"],
                "raw_rank_provenance": record["raw_rank_provenance"],
                "duplicate_candidate_index": record["duplicate_candidate_index"],
                "labels": [int(value) for value in record["labels"]],
                "target_ids": record["target_ids"],
                "target_present": bool(record["target_ids"]),
                "candidate_present": bool(record["candidate_present"]),
                "coverage_mask": bool(record["coverage_mask"]),
                "present_uncovered_not_negative": bool(record["present_uncovered_not_negative"]),
                "scores": scores, "absent_logit": absent_logit,
                "candidate_keys_complete": len(record["row_keys"]) == record["candidate_count"],
                "candidate_rows_ordered": record["row_keys"] == sorted(record["row_keys"], key=lambda key: key[-1]),
                "candidate_truncation": False, "candidate_deletion": False,
                "expression_span_method": prepared["expression_span_method"],
                "expression_token_count": len(prepared["expression_positions"]),
                "prompt_sha256": prepared["prompt_sha256"],
                "mapping_nonempty_count": sum(bool(cells) for cells in prepared["candidate_cells"]),
                "mapping_row_count": len(prepared["candidate_cells"]),
                "candidate_row_offsets": record["row_offsets"],
                "bank_path": record["bank_path"], "bank_sha256": record["bank_sha256"],
            }
            if fixed_old is not None:
                old = fixed_old[row["unit_key"]]
                row["l29"] = [float(value) for value in old["l29"]]
                row["old_l29_label"] = [int(value) for value in old["label"]]
                row["old_l29_candidate_count"] = len(old["label"])
            evaluated.append(row)
            del prepared, image_payload, record
        finally:
            bank.close()
    return evaluated


def attach_checkpoint(model: Any, matcher: Any, package: dict[str, Any]) -> dict[str, Any]:
    matcher_result = matcher.load_state_dict(package["matcher"], strict=True)
    lora_result = load_lora_state_dict(model, package["lora"], strict=True)
    if matcher_result.missing_keys or matcher_result.unexpected_keys:
        raise AssertionError("matcher state mismatch")
    model.language_model.model.eval()
    matcher.eval()
    return {"matcher_missing": list(matcher_result.missing_keys),
            "matcher_unexpected": list(matcher_result.unexpected_keys),
            "lora": lora_result}


def slice_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(f"dataset:{row['dataset']}", []).append(row)
        groups.setdefault(f"category:{row['category']}", []).append(row)
        groups.setdefault(f"video:{row['video']}", []).append(row)
    return {name: metric(group, threshold) for name, group in sorted(groups.items())}


# Set by main immediately after local runtime construction.  No model state is
# stored here; the objects live only for the active evaluation process.
PROCESSOR: Any = None
TOKENIZER: Any = None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--candidate-chunk", type=int, default=4)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    train_dir = (args.train_dir if args.train_dir.is_absolute() else ROOT / args.train_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base = {
        "format": "locatemot-l75-semantic-evaluation-v1", "status": "running",
        "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
        "command": " ".join(sys.argv), "seed": SEED,
        "calibration_units": 16, "validation_units": 24,
        "checkpoint_selection_before_validation": True,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
        "candidate_rows_retained": True, "candidate_truncation": False,
        "candidate_deletion": False, "raw_dense_feature_cache_written": False,
        "token_span_alignment": "UNALIGNED",
    }
    write_json(out / "status.json", base)
    model = matcher = None
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256_file(MANIFEST_PATH) != MANIFEST_SHA256:
            raise AssertionError("fixed manifest SHA mismatch")
        if not train_dir.exists():
            raise FileNotFoundError(train_dir)
        checkpoint_paths = []
        packages: dict[int, dict[str, Any]] = {}
        for step in CHECKPOINT_STEPS:
            path = train_dir / f"checkpoint_l75_candidate_marked_step{step}.pt"
            if not path.exists():
                raise FileNotFoundError(path)
            package = load_checkpoint_package(path)
            if int(package["step"]) != step:
                raise AssertionError(f"checkpoint step mismatch: {path}")
            packages[step] = package
            checkpoint_paths.append({"step": step, "path": str(path),
                                     "sha256": sha256_file(path),
                                     "adapter_norm": checkpoint_norm(package)})
        train_metrics_path = train_dir / "metrics_l75_step5000.json"
        train_metrics = json.loads(train_metrics_path.read_text()) if train_metrics_path.exists() else {}
        if train_metrics.get("status") != "complete":
            raise AssertionError("formal fit metrics are not complete")
        # Only unit keys are read from the immutable 40-row order before the
        # calibration selection.  Validation labels are joined below, after
        # checkpoint/threshold strategy selection is complete.
        fixed_keys = read_unit_keys(L62_RECORDS)
        if len(fixed_keys) != 40 or len(set(fixed_keys)) != 40:
            raise AssertionError("immutable fixed order must contain 40 unique keys")
        calibration_rows = read_records(ROOT / "outputs/l49/data/calibration_units.jsonl")
        cal_lookup = {unit_key(row): row for row in calibration_rows}
        cal_units = []
        for order, key in enumerate(fixed_keys[:16]):
            if key not in cal_lookup:
                raise AssertionError(f"fixed calibration key absent from L49 calibration: {key}")
            item = dict(cal_lookup[key]); item["fixed_eval_order"] = order
            item["fixed_eval_split"] = "calibration"
            cal_units.append(item)
        global PROCESSOR, TOKENIZER
        model, PROCESSOR, TOKENIZER, runtime = load_locateanything("cuda:0")
        lora_contract = attach_language_lora(model, rank=8, alpha=16.0, target_layers=4)
        matcher = CandidateMarkedVLMMatcher(hidden=256).to("cuda:0")
        matcher.eval()
        base_digest_before = frozen_target_digest(model)

        # Calibration-only checkpoint panel.  This is intentionally run before
        # any validation unit is opened or labels are joined.
        cal_by_step: dict[int, list[dict[str, Any]]] = {step: [] for step in CHECKPOINT_STEPS}
        cal_start = time.perf_counter()
        for unit in cal_units:
            bank = L75Bank(str(unit["video"]))
            try:
                record, image_payload = load_unit_for_eval(unit, bank, include_labels=True)
                prepared = prepare_visual(model, PROCESSOR, TOKENIZER,
                                          image_payload["image"], record["sentence"], image_payload["boxes"])
                for step in CHECKPOINT_STEPS:
                    attach_checkpoint(model, matcher, packages[step])
                    scores, absent_logit = score_prepared(model, matcher, prepared,
                                                          record["candidate_count"], args.candidate_chunk)
                    cal_by_step[step].append({
                        "unit_key": record["unit_key"], "dataset": record["dataset"],
                        "video": record["video"], "frame_id": record["frame_id"],
                        "category": record["category"], "labels": [bool(x) for x in record["labels"]],
                        "target_present": bool(record["target_ids"]),
                        "candidate_present": bool(record["candidate_present"]),
                        "scores": scores, "absent_logit": absent_logit,
                        "candidate_keys": record["row_keys"],
                        "candidate_keys_complete": len(record["row_keys"]) == record["candidate_count"],
                    })
                del prepared, image_payload, record
            finally:
                bank.close()
        cal_selection_rows = []
        for step in CHECKPOINT_STEPS:
            threshold_info = fit_threshold(cal_by_step[step], "scores")
            cal_metric = metric(cal_by_step[step], threshold_info["threshold"])
            cal_selection_rows.append({
                "step": step, "checkpoint": checkpoint_paths[CHECKPOINT_STEPS.index(step)],
                "threshold": threshold_info, "calibration_metric": cal_metric,
                "strict_margin_mean": cal_metric["strict_margin"]["mean"],
                "minimum_positive_coverage": cal_metric["multi_positive_recall"],
                "adapter_norm": checkpoint_norm(packages[step]),
                "eligible_from_fit_contract": True,
            })
        def selection_key(item: dict[str, Any]) -> tuple[float, float, float, float, int, float]:
            cm = item["calibration_metric"]
            return (
                float(cm["hard_violation"] if cm["hard_violation"] is not None else 1e9),
                -float(item["minimum_positive_coverage"] if item["minimum_positive_coverage"] is not None else -1e9),
                float(cm["inactive_false_acceptance"]),
                # The fourth registered key is candidate false-positive
                # volume at the observed calibration threshold.  Do not use
                # predictions/positive here: that is a different quantity
                # and would make the checkpoint choice diverge from the
                # pre-registered rule in reports/l75_checkpoint_selection_rule.md.
                float(cm["false_positive_rows"]),
                int(item["step"]),
                float(item["adapter_norm"]),
            )
        selected = sorted(cal_selection_rows, key=selection_key)[0]
        selected_step = int(selected["step"])
        selected_package = packages[selected_step]
        selected_threshold = float(selected["threshold"]["threshold"])
        checkpoint_selection = {
            "rule_source": str(ROOT / "reports/l75_checkpoint_selection_rule.md"),
            "validation_used": False,
            "selection_tuple_minimize": [
                "calibration hard_violation", "negative calibration minimum-positive coverage",
                "calibration inactive false acceptance", "calibration false-positive volume",
                "checkpoint step (earlier)", "adapter+marker L2 norm",
            ],
            "selected_step": selected_step,
            "selected_checkpoint": selected["checkpoint"],
            "selected_checkpoint_sha256": selected["checkpoint"]["sha256"],
            "candidate_threshold": selected_threshold,
            "calibration_elapsed_seconds": time.perf_counter() - cal_start,
            "all_checkpoint_calibration": cal_selection_rows,
        }
        write_json(out / "checkpoint_selection.json", checkpoint_selection)

        # Strategy is now fixed.  Only here do we open validation unit metadata
        # and the immutable L62 rows containing validation labels.
        validation_rows = read_records(ROOT / "outputs/l49/data/validation_units.jsonl")
        val_lookup = {unit_key(row): row for row in validation_rows}
        val_units = []
        for order, key in enumerate(fixed_keys[16:], start=16):
            if key not in val_lookup:
                raise AssertionError(f"fixed validation key absent from L49 validation: {key}")
            item = dict(val_lookup[key]); item["fixed_eval_order"] = order
            item["fixed_eval_split"] = "validation"
            val_units.append(item)
        old_records = read_records(L62_RECORDS)
        if [str(row["unit_key"]) for row in old_records] != fixed_keys:
            raise AssertionError("immutable L62 order changed")
        fixed_old = {str(row["unit_key"]): row for row in old_records}
        attach_checkpoint(model, matcher, selected_package)
        selected_cal = evaluate_units(model, matcher, cal_units, True, args.candidate_chunk, fixed_old)
        selected_val = evaluate_units(model, matcher, val_units, True, args.candidate_chunk, fixed_old)
        final_rows = selected_cal + selected_val
        if [row["unit_key"] for row in final_rows] != fixed_keys:
            raise AssertionError("L75 final row order drift")
        if any(len(row["scores"]) != int(row["candidate_count"]) for row in final_rows):
            raise AssertionError("L75 score/candidate count drift")

        old_cal = [make_old_control_row(row) for row in old_records[:16]]
        old_val = [make_old_control_row(row) for row in old_records[16:]]
        old_l29_cal = metric(old_cal, L29_THRESHOLD)
        old_l29_val = metric(old_val, L29_THRESHOLD)
        l29_check = {
            name: {"observed": float(old_l29_val[name]), "expected": expected,
                   "abs_error": abs(float(old_l29_val[name]) - expected),
                   "within_tolerance": abs(float(old_l29_val[name]) - expected) <= 1e-10}
            for name, expected in EXPECTED_L29.items()
        }
        if not all(item["within_tolerance"] for item in l29_check.values()):
            raise AssertionError(f"immutable L29 control mismatch: {l29_check}")
        l75_cal = metric(selected_cal, selected_threshold)
        l75_val = metric(selected_val, selected_threshold)
        slices = {
            "calibration": slice_metrics(selected_cal, selected_threshold),
            "validation": slice_metrics(selected_val, selected_threshold),
        }
        complete = all(
            row["candidate_keys_complete"] and row["candidate_rows_ordered"]
            and not row["candidate_truncation"] and not row["candidate_deletion"]
            and len(row["scores"]) == row["candidate_count"] == len(row["labels"])
            and np.isfinite(np.asarray(row["scores"], dtype=np.float64)).all()
            for row in final_rows
        )
        checks = {
            "hard_violation_decrease_ge_0.05": (
                l75_val["hard_violation"] is not None and old_l29_val["hard_violation"] is not None
                and l75_val["hard_violation"] <= old_l29_val["hard_violation"] - 0.05
            ),
            "recall_floor": l75_val["candidate_recall"] >= 0.7233333,
            "precision_floor": l75_val["candidate_precision"] >= 0.0830188679,
            "fp_per_frame_floor": l75_val["fp_per_frame"] <= 11.125,
            "predictions_per_positive_floor": l75_val["predictions_per_positive"] <= 4.069,
            "multi_positive_floor": l75_val["multi_positive_recall"] is not None
            and l75_val["multi_positive_recall"] >= 0.7894444,
            "inactive_false_acceptance_not_universal": l75_val["inactive_false_acceptance"] < 1.0,
            "complete_finite_keys": complete,
            "candidate_deletion_false": all(not row["candidate_deletion"] for row in final_rows),
            "candidate_truncation_false": all(not row["candidate_truncation"] for row in final_rows),
            "both_domains_reported": sorted({row["dataset"] for row in final_rows}) == list(DATASETS),
        }
        gate = {
            "format": "locatemot-l75-semantic-gate-v1",
            "status": "semantic_gate_pass" if all(checks.values()) else "semantic_gate_fail",
            "decision": "pass" if all(checks.values()) else "fail",
            "checks": checks,
            "selected_checkpoint_step": selected_step,
            "selected_checkpoint_sha256": selected["checkpoint"]["sha256"],
            "threshold": selected_threshold,
            "threshold_fit": "16 calibration units only; observed-score candidate F1",
            "validation_selection_used": False,
            "null_head_primary_used": False,
            "null_head_contract": "diagnostic absent_logit only; no NULL suppression",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
        }
        base_digest_after = frozen_target_digest(model)
        if base_digest_after != base_digest_before:
            raise AssertionError("frozen LocateAnything base digest changed during eval")
        provenance = {
            **base, "status": "complete", "runtime": runtime,
            "manifest": str(MANIFEST_PATH), "manifest_sha256": sha256_file(MANIFEST_PATH),
            "l62_records": str(L62_RECORDS), "l62_records_sha256": sha256_file(L62_RECORDS),
            "train_dir": str(train_dir), "train_metrics": str(train_metrics_path),
            "train_metrics_sha256": sha256_file(train_metrics_path),
            "checkpoint_selection": str(out / "checkpoint_selection.json"),
            "selected_checkpoint": selected["checkpoint"],
            "lora_contract": lora_contract,
            "matcher_contract": matcher.parameter_contract(),
            "base_target_digest_before": base_digest_before,
            "base_target_digest_after": base_digest_after,
            "base_target_digest_unchanged": base_digest_before == base_digest_after,
            "calibration_order": fixed_keys[:16], "validation_order": fixed_keys[16:],
            "candidate_rows_calibration": int(sum(len(row["labels"]) for row in selected_cal)),
            "candidate_rows_validation": int(sum(len(row["labels"]) for row in selected_val)),
            "old_l29_candidate_rows_validation": int(sum(len(row["label"]) for row in old_records[16:])),
            "l69_bank_rows_rebuilt_from_frame_ptr": True,
            "old_l49_ranges_used_for_candidate_addressing": False,
            "old_positive_indices_used_for_candidate_addressing": False,
            "candidate_rows_retained": True, "candidate_truncation": False,
            "candidate_deletion": False, "persistent_raw_dense_cache_written": False,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
            "token_span_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "elapsed_seconds": time.perf_counter() - started,
        }
        methods = {
            "l29_teacher_immutable": {"calibration": old_l29_cal, "validation": old_l29_val,
                                      "threshold": {"value": L29_THRESHOLD,
                                                     "source": "accepted immutable L62 control contract"}},
            "l75_candidate_marked_selected": {"checkpoint": selected["checkpoint"],
                                               "calibration": l75_cal, "validation": l75_val,
                                               "slices": slices,
                                               "primary_output": "candidate-only match_logit",
                                               "absent_logit": "diagnostic only"},
        }
        semantic = {
            "format": base["format"], "status": "complete", "provenance": provenance,
            "checkpoint_selection": checkpoint_selection,
            "methods": methods, "l29_reproduction": l29_check, "gate": gate,
            "training_fit_summary": {
                "formal_metrics": train_metrics,
                "fit_100_to_500_monitor": next((x.get("fit_100_to_500_monitor")
                    for x in train_metrics.get("checkpoint_summaries", []) if x.get("step") == 500), None),
                "note": "fit diagnostics are separate from semantic validation; no validation used",
            },
        }
        write_json(out / "semantic.json", semantic)
        write_json(out / "gate_decision.json", gate)
        out.joinpath("score_records.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                    for row in final_rows)
        )
        write_json(out / "provenance.json", provenance)
        write_json(out / "status.json", {**base, "status": "complete",
                                          "selected_checkpoint_step": selected_step,
                                          "gate_status": gate["status"],
                                          "elapsed_seconds": time.perf_counter() - started,
                                          "failure_root_cause": None,
                                          "next_action": "stop L75 if semantic gate fails; no screening/test"})
        print(json.dumps({"status": gate["status"], "selected_step": selected_step,
                          "validation": l75_val, "checks": checks,
                          "output": str(out)}, indent=2), flush=True)
        return 0
    except Exception as exc:
        failure = {**base, "status": "INCOMPLETE",
                   "failure_root_cause": f"{type(exc).__name__}: {exc}",
                   "next_action": "preserve this evaluator attempt; fix only the first actionable root cause and retry in a new directory",
                   "traceback": traceback.format_exc()}
        write_json(out / "status.json", failure)
        out.joinpath("INCOMPLETE.md").write_text(
            "# L75 evaluator INCOMPLETE\n\n"
            f"First actionable root cause: `{type(exc).__name__}: {exc}`\n\n"
            "```text\n" + traceback.format_exc() + "```\n"
        )
        raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
