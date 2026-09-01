#!/usr/bin/env python3
"""Build the L49 fit/validation error matrix before official test is read.

This is an analysis-only evaluator.  It uses the frozen L49 checkpoints and
the already generated validation score cache; it never reads an official test
label.  The selected checkpoint and calibration thresholds come from the
existing validation-selection artifact and are not re-selected here.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l29_frame_membership_set_decoder import (  # noqa: E402
    L29FrameMembershipSetDecoder,
)
from locatemot.rmot.l49_data import (  # noqa: E402
    L29_CHECKPOINT,
    TEXT_CACHE,
    history_sequence,
    load_bank,
    sha256_file,
    unit_features,
)
from tools.eval_l49_validation import (  # noqa: E402
    BankStore,
    checkpoint_paths,
    jsonable_record,
    l29_score,
    make_records,
    summarize,
)
from tools.train_l28_track_set_decoder import state_at  # noqa: E402
from tools.train_l49_kitti_rmot import (  # noqa: E402
    build_teacher_cache,
    valid_track_indices,
)

DATA = ROOT / "outputs/l49/data"
VAL = ROOT / "outputs/l49/val"
TRAIN_ROOT = ROOT / "outputs/l49/train/joint_long5000"
SELECTION = VAL / "selected_checkpoint.json"
METRICS = VAL / "validation_metrics.json"


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def record_key(record: dict) -> tuple[str, str, int, int]:
    return (str(record["dataset"]), str(record["video"]),
            int(record["query_id"]), int(record["frame_id"]))


def unit_key(unit: dict) -> tuple[str, str, int, int]:
    return (str(unit["dataset"]), str(unit["video"]),
            int(unit["query_id"]), int(unit["frame_id"]))


def load_score_records(path: Path, split: str, checkpoint_step: int | None = None):
    records = []
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") != split:
                continue
            if checkpoint_step is not None and int(row["checkpoint_step"]) != int(checkpoint_step):
                continue
            row.pop("split", None)
            row.pop("checkpoint_step", None)
            for key in ("score", "semantic_score", "identity_score", "continuation_score"):
                row[key] = np.asarray(row[key], dtype=np.float32)
            row["label"] = np.asarray(row["label"], dtype=bool)
            row["sources"] = {key: np.asarray(value, dtype=bool)
                               for key, value in row["sources"].items()}
            records.append(row)
    return records


def make_baseline_records(units, store, text, teacher, device):
    grouped = defaultdict(list)
    for unit in units:
        grouped[(unit["dataset"], unit["video"])].append(unit)
    cache_by_path = OrderedDict()
    result = []
    for (dataset, video), values in sorted(grouped.items()):
        bank = store.get(dataset, video)
        cache_key = str(bank["path"])
        if cache_key not in cache_by_path:
            cache_by_path[cache_key] = build_teacher_cache(bank)
            if len(cache_by_path) > 1:
                cache_by_path.popitem(last=False)
        cache = cache_by_path[cache_key]
        for unit in values:
            score = l29_score(teacher, cache, bank, unit, text, device)
            label = np.zeros(int(unit["end"] - unit["begin"]), dtype=bool)
            label[np.asarray(unit["positive_indices"], dtype=np.int64)] = True
            result.append({
                "dataset": unit["dataset"], "video": str(unit["video"]),
                "query_id": int(unit["query_id"]), "expression": unit.get("expression", ""),
                "sentence": unit["sentence"], "frame_id": int(unit["frame_id"]),
                "category": unit["category"], "candidate_count": len(label),
                "positive_count": int(label.sum()), "score": score,
                "semantic_score": score, "identity_score": np.zeros_like(score),
                "continuation_score": np.zeros_like(score), "null_logit": 0.0,
                "label": label,
                "sources": {key: value.copy() for key, value in
                            __import__("tools.eval_l49_validation", fromlist=["source_masks"])
                            .source_masks(bank, int(unit["begin"]), int(unit["end"])).items()},
            })
    return result


def annotate_history(records, units, store):
    by_key = {unit_key(unit): unit for unit in units}
    by_video = {}
    for record in records:
        key = record_key(record)
        unit = by_key.get(key)
        if unit is None:
            record["history_length_bucket"] = "unknown"
            continue
        cache_key = (unit["dataset"], str(unit["video"]))
        if cache_key not in by_video:
            by_video[cache_key] = store.get(*cache_key)
        bank = by_video[cache_key]
        _sequence, mask = history_sequence(bank, int(unit["begin"]), int(unit["end"]), length=8)
        lengths = mask.long().sum(-1).numpy()
        mean_length = float(lengths.mean()) if len(lengths) else 0.0
        if mean_length <= 1:
            bucket = "0-1"
        elif mean_length <= 4:
            bucket = "2-4"
        elif mean_length <= 8:
            bucket = "5-8"
        else:
            bucket = "9+"
        record["history_length_bucket"] = bucket
        record["history_length_mean"] = mean_length
    return records


def expression_type(sentence: str) -> str:
    text = str(sentence).lower()
    if any(word in text for word in ("left", "right", "front", "ahead", "before", "behind")):
        return "space"
    if any(word in text for word in
           ("black", "white", "silver", "red", "light", "green", "color", "colour")):
        return "appearance"
    if any(word in text for word in
           ("moving", "walk", "walking", "park", "parked", "standing", "direction",
            "transit", "opposite", "same", "heading", "abandoned")):
        return "action"
    if any(word in text for word in
           ("car", "cars", "vehicle", "vehicles", "auto", "autos", "automobile",
            "automobiles", "pedestrian", "pedestrians", "person", "persons", "people",
            "men", "males", "women", "females", "individual")):
        return "category"
    return "other"


def candidate_bucket(count: int) -> str:
    if count <= 10:
        return "1-10"
    if count <= 20:
        return "11-20"
    if count <= 40:
        return "21-40"
    if count <= 80:
        return "41-80"
    return "81+"


def length_bucket(sentence: str) -> str:
    count = len(str(sentence).split())
    if count <= 5:
        return "1-5"
    if count <= 10:
        return "6-10"
    return "11+"


def margin_bucket(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "unavailable"
    if value < -2:
        return "<-2"
    if value < -1:
        return "[-2,-1)"
    if value < -0.25:
        return "[-1,-.25)"
    if value < 0:
        return "[-.25,0)"
    return ">=0"


def frame_margin(score, label):
    pos = np.flatnonzero(label)
    neg = np.flatnonzero(~label)
    if not len(pos) or not len(neg):
        return None
    return float(np.asarray(score)[pos].min() - np.asarray(score)[neg].max())


def compact_summary(records, threshold):
    summary = summarize(records, threshold)
    keep = (
        "frame_units", "candidate_rows", "positive_rows", "positive_frame_units",
        "precision", "recall", "top1_frame_recall", "top5_frame_recall",
        "false_positive_candidates_per_frame", "empty_output_rate",
        "null_frame_false_acceptance", "predictions_per_positive",
        "multi_positive_frame_count", "multi_positive_recall", "hard_violation_rate",
        "strict_min_positive_margin", "best_positive_margin", "average_positive_margin",
        "source_precision", "null_max_score", "threshold",
    )
    return {key: summary.get(key) for key in keep}


def rank_flip_decomposition(records, baseline_by_key):
    names = {
        "semantic_only": lambda rec: np.asarray(rec["semantic_score"], dtype=np.float64),
        "semantic_plus_identity": lambda rec: np.asarray(rec["semantic_score"], dtype=np.float64)
        + .20 * np.tanh(np.asarray(rec["identity_score"], dtype=np.float64)),
        "final": lambda rec: np.asarray(rec["score"], dtype=np.float64),
    }
    output = {}
    for name, scorer in names.items():
        counts = defaultdict(int)
        for record in records:
            baseline = baseline_by_key.get(record_key(record))
            if baseline is None:
                continue
            label = np.asarray(record["label"], dtype=bool)
            pos = np.flatnonzero(label); neg = np.flatnonzero(~label)
            if not len(pos) or not len(neg):
                continue
            teacher = np.asarray(baseline["score"], dtype=np.float64)
            student = scorer(record)
            teacher_order = (teacher[pos, None] > teacher[None, neg]).reshape(-1)
            student_order = (student[pos, None] > student[None, neg]).reshape(-1)
            counts["pair_count"] += int(len(teacher_order))
            counts["teacher_correct_pairs"] += int(teacher_order.sum())
            counts["student_correct_pairs"] += int(student_order.sum())
            counts["teacher_correct_student_flip"] += int((teacher_order & ~student_order).sum())
            counts["teacher_error_student_correction"] += int((~teacher_order & student_order).sum())
            counts["total_pair_rank_flip"] += int((teacher_order != student_order).sum())
            counts["frame_units"] += 1
        pairs = max(1, counts["pair_count"])
        output[name] = {
            **{key: int(value) for key, value in counts.items()},
            "teacher_correct_student_flip_rate": counts["teacher_correct_student_flip"] / pairs,
            "teacher_error_student_correction_rate": counts["teacher_error_student_correction"] / pairs,
            "total_pair_rank_flip_rate": counts["total_pair_rank_flip"] / pairs,
        }
    return output


def matrix_buckets(records, baseline_by_key, thresholds):
    annotated = []
    for record in records:
        base = baseline_by_key.get(record_key(record))
        teacher_margin = frame_margin(base["score"], record["label"]) if base else None
        final_margin = frame_margin(record["score"], record["label"])
        primary = expression_type(record.get("sentence", ""))
        flags = {
            "expression_type": primary,
            "candidate_count": candidate_bucket(int(record["candidate_count"])),
            "teacher_margin": margin_bucket(teacher_margin),
            "final_margin": margin_bucket(final_margin),
            "domain": str(record["dataset"]),
            "expression_length": length_bucket(record.get("sentence", "")),
            "history_length": str(record.get("history_length_bucket", "unknown")),
        }
        if int(record.get("positive_count", 0)) > 1:
            flags["multi_positive"] = "multi_positive"
        if int(record.get("positive_count", 0)) == 0:
            flags["inactive_null"] = "inactive_null"
        annotated.append((record, flags))
    result = {}
    for dimension in ("expression_type", "candidate_count", "teacher_margin", "final_margin",
                      "domain", "expression_length", "history_length", "multi_positive",
                      "inactive_null"):
        groups = defaultdict(list)
        for record, flags in annotated:
            if dimension in flags:
                # V1 and V2 use distinct frozen calibration thresholds.  Keep
                # the domain in every non-domain bucket so a cross-domain
                # bucket is never summarized with the wrong threshold.
                groups[(str(record["dataset"]), flags[dimension])].append(record)
        result[dimension] = {
            f"{domain}|{bucket}": compact_summary(values,
                                                   thresholds.get(domain, 0.0))
            for (domain, bucket), values in sorted(groups.items()) if values
        }
    return result


def add_internal_metadata(records, units, store):
    by_key = {unit_key(unit): unit for unit in units}
    for record in records:
        unit = by_key.get(record_key(record))
        if unit is not None:
            record["unit_key"] = unit["unit_key"]
            record["history_length_bucket"] = record.get("history_length_bucket", "unknown")
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", default=str(VAL / "error_matrix_pretest.json"))
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")
    started = time.time()
    selection = json.loads(SELECTION.read_text())
    selected_step = int(selection["selected_step"])
    thresholds = {key: float(value["threshold"])
                  for key, value in selection["selected_thresholds"].items()}
    train_units = load_jsonl(DATA / "train_units.jsonl")
    validation_units = load_jsonl(DATA / "validation_units.jsonl")
    validation_scores_path = VAL / "validation_scores.jsonl"
    validation_selected = load_score_records(validation_scores_path, "validation", selected_step)
    if not validation_selected:
        raise RuntimeError("selected validation score cache is empty")
    text = torch.load(TEXT_CACHE, map_location="cpu", weights_only=False)
    device = torch.device(args.device if args.device != "cpu" or not torch.cuda.is_available() else "cpu")

    # Build the frozen L29 reference once.  This is an evaluation reference,
    # not a train/test selection signal.
    teacher = L29FrameMembershipSetDecoder().to(device)
    teacher.load_state_dict(torch.load(L29_CHECKPOINT, map_location=device,
                                       weights_only=False)["model"], strict=True)
    teacher.eval()
    baseline_store = BankStore(limit=1)
    baseline_fit = make_baseline_records(train_units, baseline_store, text, teacher, device)
    baseline_val = make_baseline_records(validation_units, baseline_store, text, teacher, device)
    baseline_fit_by_key = {record_key(x): x for x in baseline_fit}
    baseline_val_by_key = {record_key(x): x for x in baseline_val}
    history_store = BankStore(limit=1)
    annotate_history(validation_selected, validation_units, history_store)
    annotate_history(baseline_val, validation_units, history_store)

    fit_summaries = {}
    selected_fit = None
    fit_baseline_by_key = baseline_fit_by_key
    for checkpoint in checkpoint_paths():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = payload.get("model_config", {})
        from locatemot.models.l49_kitti_rmot import L49KittiRMOT
        model = L49KittiRMOT(hidden=int(cfg.get("hidden", 256)), heads=int(cfg.get("heads", 4)),
                             history_length=int(cfg.get("history_length", 8))).to(device)
        model.load_state_dict(payload["model"], strict=True)
        step = int(payload.get("checkpoint_step", checkpoint.stem.split("step")[-1]))
        stage = "semantic_warmup" if step <= int(payload.get("warmup_steps", 1000)) else "identity_continuation_null_sequence"
        fit_records = make_records(model, step, train_units, history_store, text, device, stage)
        annotate_history(fit_records, train_units, history_store)
        for record in fit_records:
            record["unit_key"] = next((u["unit_key"] for u in train_units
                                        if unit_key(u) == record_key(record)), "")
        fit_summaries[str(step)] = {
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint),
            "stage": stage,
            "thresholds_frozen_from_selected_validation": thresholds,
            "per_domain": {
                domain: compact_summary([x for x in fit_records if x["dataset"] == domain],
                                         thresholds.get(domain, 0.0))
                for domain in ("refer_kitti_v1", "refer_kitti_v2")
            },
        }
        if step == selected_step:
            selected_fit = fit_records
        del model, payload, fit_records
        gc.collect()
    if selected_fit is None:
        raise RuntimeError(f"selected step {selected_step} not found in checkpoints")

    # The detailed matrix is evaluated with the frozen validation-selected
    # checkpoint and thresholds.  Fit and validation are separate domains;
    # official test is deliberately an explicit not-run state at this stage.
    detailed = {}
    for name, records, baseline in (
        ("fit", selected_fit, baseline_fit_by_key),
        ("validation", validation_selected, baseline_val_by_key),
    ):
        detailed[name] = {
            "checkpoint_step": selected_step,
            "per_domain": {
                domain: compact_summary([x for x in records if x["dataset"] == domain],
                                         thresholds.get(domain, 0.0))
                for domain in ("refer_kitti_v1", "refer_kitti_v2")
            },
            "buckets": matrix_buckets(records, baseline, thresholds),
            "rank_flip_decomposition": rank_flip_decomposition(records, baseline),
        }

    # Save selected fit/baseline records for the post-test matrix update.  The
    # values are frozen predictions, not a new cache or a training input.
    fit_path = VAL / "fit_scores_selected.jsonl"
    base_fit_path = VAL / "fit_baseline_scores_selected.jsonl"
    base_val_path = VAL / "validation_baseline_scores.jsonl"
    with fit_path.open("w") as handle:
        for record in selected_fit:
            handle.write(json.dumps({"split": "fit", "checkpoint_step": selected_step,
                                     **jsonable_record(record)}, allow_nan=False) + "\n")
    with base_fit_path.open("w") as handle:
        for record in baseline_fit:
            handle.write(json.dumps({"split": "fit", "checkpoint_step": "L29",
                                     **jsonable_record(record)}, allow_nan=False) + "\n")
    with base_val_path.open("w") as handle:
        for record in baseline_val:
            handle.write(json.dumps({"split": "validation", "checkpoint_step": "L29",
                                     **jsonable_record(record)}, allow_nan=False) + "\n")

    output = {
        "format": "locatemot-l49-error-matrix-v1",
        "matrix_status": "pretest_fit_validation_complete_test_not_read",
        "started_at_unix": started,
        "completed_at_unix": time.time(),
        "selected_checkpoint_step": selected_step,
        "selected_checkpoint": selection["selected_checkpoint"],
        "selected_checkpoint_sha256": selection["selected_checkpoint_sha256"],
        "selection_rule_inherited": selection["selection_rule"],
        "thresholds_inherited_from_calibration": thresholds,
        "screening_or_test_labels_used": False,
        "official_test_labels_read": False,
        "provenance": {
            "train_units": str((DATA / "train_units.jsonl").resolve()),
            "train_units_sha256": sha256_file(DATA / "train_units.jsonl"),
            "validation_units": str((DATA / "validation_units.jsonl").resolve()),
            "validation_units_sha256": sha256_file(DATA / "validation_units.jsonl"),
            "validation_scores": str(validation_scores_path.resolve()),
            "validation_scores_sha256": sha256_file(validation_scores_path),
            "l29_checkpoint": str(L29_CHECKPOINT.resolve()),
            "l29_checkpoint_sha256": sha256_file(L29_CHECKPOINT),
            "text_cache": str(TEXT_CACHE.resolve()),
            "text_cache_sha256": sha256_file(TEXT_CACHE),
            "device": str(device),
        },
        "fit_checkpoint_curves": fit_summaries,
        "selected_detailed_matrix": detailed,
        "test": {
            "status": "not_run",
            "official_test_labels_read": False,
            "reason": "deferred until this matrix and validation selection are frozen",
        },
        "saved_selected_score_records": {
            "fit": str(fit_path.resolve()),
            "fit_baseline": str(base_fit_path.resolve()),
            "validation_baseline": str(base_val_path.resolve()),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    (VAL / "error_matrix_pretest.md").write_text(
        "# L49 fit/validation error matrix (pre-test)\n\n"
        f"Selected frozen checkpoint: step {selected_step}.\n\n"
        "This artifact contains fit and video-disjoint validation breakdowns; "
        "official test labels have not been read. The `test` field is explicitly `not_run` "
        "and will be populated only after the frozen official test/TrackEval pass.\n"
    )
    print(json.dumps({"matrix": str(out.resolve()), "selected_step": selected_step,
                      "fit_units": len(train_units), "validation_units": len(validation_units),
                      "elapsed_sec": time.time() - started,
                      "official_test_labels_read": False}, indent=2), flush=True)


if __name__ == "__main__":
    main()
