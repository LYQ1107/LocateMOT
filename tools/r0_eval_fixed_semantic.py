#!/usr/bin/env python3
"""Post-selection fixed-slice diagnostics for the isolated R0 sidecar.

This is not a new model-selection procedure.  V1/V2 checkpoints and the
registered zero-logit rule are frozen by legal dev evidence before this tool
reads the fixed 16-calibration/24-validation labels.  Current predictions
are built from native L69 frame pointers; old L29 rows are retained only as
an immutable sparse control and are never overwritten.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.models.r0_track_grounding import R0Config, R0TrackGroundingHead  # noqa: E402
from locatemot.rmot.r0_dense_data import R0BankStore  # noqa: E402
from locatemot.rmot.r0_safe_target_source import (  # noqa: E402
    load_safe_v1_records_with_manifest,
    load_safe_v2_records_with_manifest,
)
from tools.r0_common import (  # noqa: E402
    ASSET_ROOT as COMMON_ASSET_ROOT,
    DEFAULT_LANGUAGE_ROOTS,
    MergedLanguageCache,
    R0RuntimeData,
    VisualCacheIndex,
    check_manifest,
    file_meta,
    model_forward,
    sha256_file,
    write_json,
)


THREAD = "01a02014-fce8-7f51-8414-e7ed6ab44745"
SEED = 20260909
FIXED_SCORE_RECORDS = ASSET_ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
CALIBRATION_UNITS = ASSET_ROOT / "outputs/l49/data/calibration_units.jsonl"
VALIDATION_UNITS = ASSET_ROOT / "outputs/l49/data/validation_units.jsonl"
L69_ROOT = ASSET_ROOT / "outputs/l69/attempt9/budget40_features/kitti"
L29_THRESHOLD = -1.030576229095459
EXPECTED_L29 = {
    "candidate_recall": 0.7333333333333333,
    "candidate_precision": 0.0830188679245283,
    "fp_per_frame": 10.125,
    "predictions_per_positive": 8.833333333333334,
    "hard_violation": 0.9166666666666666,
    "multi_positive_recall": 0.8194444444444444,
}


class FixedIndex:
    """Small label-free query identity index used by R0RuntimeData."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.query_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
        for row in records:
            key = (str(row["dataset"]), str(row["video"]), int(row["query_id"]))
            value = {"dataset": key[0], "video": key[1], "query_id": key[2], "sentence": str(row["sentence"])}
            previous = self.query_by_key.get(key)
            if previous is not None and previous["sentence"] != value["sentence"]:
                raise AssertionError(f"fixed sentence drift: {key}")
            self.query_by_key[key] = value

    def query(self, record: dict[str, Any]) -> dict[str, Any]:
        key = (str(record["dataset"]), str(record["video"]), int(record["query_id"]))
        value = self.query_by_key.get(key)
        if value is None or value["sentence"] != str(record["sentence"]):
            raise KeyError(f"fixed query identity missing/drifted: {key}")
        return value


class MergedVisualIndex:
    """Read-only union of the registered cache and the two cal-video items."""

    def __init__(self, roots: Iterable[Path]) -> None:
        self.indexes = [VisualCacheIndex(Path(root).resolve()) for root in roots]
        self.entries: dict[tuple[str, str, int], tuple[VisualCacheIndex, dict[str, Any]]] = {}
        for index in self.indexes:
            for key, value in index.entries.items():
                if key in self.entries:
                    raise AssertionError(f"duplicate fixed semantic visual frame: {key}")
                self.entries[key] = (index, value)

    def get(self, dataset: str, video: str, frame_id: int) -> dict[str, Any]:
        key = (str(dataset), str(video), int(frame_id))
        value = self.entries.get(key)
        if value is None:
            raise KeyError(f"fixed semantic visual cache miss: {key}")
        return value[0].get(*key)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def fixed_records() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    old = _read_jsonl(FIXED_SCORE_RECORDS)
    if len(old) != 40:
        raise AssertionError(f"fixed L62 row count drift: {len(old)}")
    metadata: dict[str, dict[str, Any]] = {}
    for path, expected_split in ((CALIBRATION_UNITS, "calibration"), (VALIDATION_UNITS, "validation")):
        for value in _read_jsonl(path):
            key = str(value["unit_key"])
            if key in metadata:
                raise AssertionError(f"duplicate L49 fixed unit: {key}")
            # The full L49 object is discarded immediately after this
            # metadata-only projection.  No old positive range is used for
            # current R0 rows.
            metadata[key] = {
                "unit_key": key, "dataset": str(value["dataset"]), "video": str(value["video"]),
                "query_id": int(value["query_id"]), "frame_id": int(value["frame_id"]),
                "sentence": str(value["sentence"]), "split": expected_split,
                "old_begin": int(value["begin"]), "old_end": int(value["end"]),
                "old_bank_path": str(value["bank_path"]), "old_label_path": str(value["label_path"]),
                "old_target_ids": [str(item) for item in value["target_ids"]],
                "old_positive_indices": [int(item) for item in value["positive_indices"]],
                "old_category": str(value["category"]),
            }
    ordered: list[dict[str, Any]] = []
    for position, old_row in enumerate(old):
        key = str(old_row["unit_key"])
        if key not in metadata:
            raise AssertionError(f"fixed key missing from L49 split metadata: {key}")
        dataset, video, query_id, frame_id = key.split("|", 3)
        meta = metadata[key]
        if (dataset, video, int(query_id), int(frame_id)) != (
            meta["dataset"], meta["video"], meta["query_id"], meta["frame_id"]):
            raise AssertionError(f"fixed key field drift: {key}")
        meta = dict(meta)
        meta["fixed_position"] = position
        meta["old_record"] = old_row
        ordered.append(meta)
    if [value["split"] for value in ordered[:16]] != ["calibration"] * 16:
        raise AssertionError("fixed first-16 calibration order drift")
    if [value["split"] for value in ordered[16:]] != ["validation"] * 24:
        raise AssertionError("fixed last-24 validation order drift")
    return ordered, metadata


def load_old_candidate_gt(meta: dict[str, Any], cache: dict[str, list[Any]]) -> list[Any]:
    path = str(Path(meta["old_label_path"]).resolve())
    if path not in cache:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        values = payload.get("candidate_gt")
        if not isinstance(values, list):
            raise AssertionError(f"old candidate_gt missing: {path}")
        cache[path] = values
    values = cache[path][int(meta["old_begin"]):int(meta["old_end"])]
    if len(values) != len(meta["old_record"]["l29"]):
        raise AssertionError(f"old candidate_gt length drift: {meta['unit_key']}")
    return [None if value is None else str(value) for value in values]


def safe_target_map(records: list[dict[str, Any]]) -> tuple[dict[tuple[str, str, int], Any], dict[str, Any]]:
    pairs = {(str(value["dataset"]), str(value["video"])) for value in records}
    v1 = {(dataset, video) for dataset, video in pairs if dataset == "refer_kitti_v1"}
    v2 = {(dataset, video) for dataset, video in pairs if dataset == "refer_kitti_v2"}
    loaded = []
    if v1:
        loaded.append(load_safe_v1_records_with_manifest(v1, purpose="audit"))
    if v2:
        loaded.append(load_safe_v2_records_with_manifest(v2, purpose="audit"))
    result: dict[tuple[str, str, int], Any] = {}
    manifests: list[dict[str, Any]] = []
    for value in loaded:
        manifests.append(value.manifest)
        for record in value.records:
            key = (record.dataset, record.video, int(record.query_id))
            if key in result:
                raise AssertionError(f"duplicate fixed safe target key: {key}")
            result[key] = record
    for value in records:
        if (value["dataset"], value["video"], int(value["query_id"])) not in result:
            raise AssertionError(f"safe target missing fixed key: {value['unit_key']}")
    return result, {"component_manifests": manifests, "labels_loaded_after_prediction": True}


def _finite(values: list[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def metric_rows(rows: list[dict[str, Any]], *, threshold: float, use_presence: bool,
                stratify: bool = True) -> dict[str, Any]:
    if not rows:
        raise AssertionError("empty fixed metric rows")
    units = len(rows)
    candidate_rows = selected = tp = fp = fn = positives = 0
    target_units = top1 = top5 = empty = 0
    hard_total = hard_bad = 0
    multi_values: list[float] = []
    margins: list[float] = []
    best_margins: list[float] = []
    average_margins: list[float] = []
    inactive_units = inactive_accept = inactive_fp = 0
    present_uncovered = 0
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scores = [float(value) for value in row["score"]]
        labels = [bool(value) for value in row["labels"]]
        if len(scores) != len(labels) or len(scores) != int(row["candidate_count"]):
            raise AssertionError(f"fixed metric row length drift: {row['unit_key']}")
        if not _finite(scores):
            raise FloatingPointError(f"nonfinite fixed score: {row['unit_key']}")
        presence = float(row.get("presence_logit", 1.0))
        if not math.isfinite(presence):
            raise FloatingPointError(f"nonfinite fixed presence: {row['unit_key']}")
        gate = (presence >= 0.0) if use_presence else True
        selected_mask = [bool(gate and value >= float(threshold)) for value in scores]
        candidate_rows += len(scores); selected += sum(selected_mask); positives += sum(labels)
        tp += sum(int(a and b) for a, b in zip(selected_mask, labels))
        fp += sum(int(a and not b) for a, b in zip(selected_mask, labels))
        fn += sum(int((not a) and b) for a, b in zip(selected_mask, labels))
        empty += int(not any(selected_mask))
        positive = [index for index, value in enumerate(labels) if value]
        negative = [index for index, value in enumerate(labels) if not value]
        if positive:
            order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
            target_units += 1
            top1 += int(labels[order[0]])
            top5 += int(any(labels[index] for index in order[:5]))
        if positive and negative:
            minimum = min(scores[index] for index in positive)
            maximum = max(scores[index] for index in negative)
            margins.append(minimum - maximum)
            best_margins.append(max(scores[index] for index in positive) - maximum)
            average_margins.append(sum(scores[index] for index in positive) / len(positive) - maximum)
            hard_total += 1; hard_bad += int(maximum >= minimum)
        if len(positive) > 1:
            multi_values.append(sum(selected_mask[index] for index in positive) / len(positive))
        category = str(row["category"])
        if category == "inactive":
            inactive_units += 1; inactive_accept += int(any(selected_mask)); inactive_fp += sum(selected_mask)
        if category == "present_uncovered":
            present_uncovered += 1
        by_dataset[str(row["dataset"])].append(row)
        by_category[category].append(row)

    def ratio(a: int | float, b: int | float) -> float:
        return float(a) / float(b) if float(b) else 0.0

    def mean(values: list[float]) -> float | None:
        return float(sum(values) / len(values)) if values else None

    def summary(values: list[float]) -> dict[str, Any]:
        if not values:
            return {"count": 0, "mean": None, "p50": None, "min": None, "max": None}
        ordered = sorted(values)
        return {"count": len(values), "mean": mean(values), "p50": ordered[(len(ordered) - 1) // 2],
                "min": ordered[0], "max": ordered[-1]}

    result: dict[str, Any] = {
        "units": units, "candidate_rows": candidate_rows, "selected_rows": selected,
        "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
        "finite_scores": True, "threshold": float(threshold), "use_presence_gate": bool(use_presence),
        "candidate_precision": ratio(tp, selected), "candidate_recall": ratio(tp, tp + fn),
        "fp_per_frame": ratio(fp, units), "predictions_per_positive": ratio(selected, positives),
        "top1": ratio(top1, target_units), "top5": ratio(top5, target_units),
        "hard_violation": ratio(hard_bad, hard_total), "hard_total": hard_total, "hard_bad": hard_bad,
        "strict_margin": summary(margins), "best_margin": summary(best_margins),
        "average_margin": summary(average_margins), "multi_positive_recall": mean(multi_values),
        "multi_positive_units": len(multi_values), "empty_rate": ratio(empty, units),
        "inactive_units": inactive_units, "inactive_false_acceptance": ratio(inactive_accept, inactive_units),
        "inactive_false_positive_rows": inactive_fp, "present_uncovered_units": present_uncovered,
        "score_distribution": {
            "count": len([value for row in rows for value in row["score"]]),
            "mean": mean([float(value) for row in rows for value in row["score"]]),
            "min": min(float(value) for row in rows for value in row["score"]),
            "max": max(float(value) for row in rows for value in row["score"]),
        },
    }
    if stratify:
        result["per_dataset"] = {
            key: metric_rows(value, threshold=threshold, use_presence=use_presence, stratify=False)
            for key, value in sorted(by_dataset.items())
        }
        result["per_category"] = {
            key: metric_rows(value, threshold=threshold, use_presence=use_presence, stratify=False)
            for key, value in sorted(by_category.items())
        }
    return result


def checkpoint_model(path: Path, device: torch.device) -> tuple[R0TrackGroundingHead, dict[str, Any]]:
    package = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    if package.get("format") != "locatemot-r0-tcgh-checkpoint-v1":
        raise AssertionError(f"invalid fixed semantic checkpoint: {path}")
    if package.get("detector_state_included") or package.get("tracker_state_included"):
        raise AssertionError(f"forbidden state in fixed semantic checkpoint: {path}")
    model = R0TrackGroundingHead(R0Config(**package["model_config"])).to(device=device, dtype=torch.float32)
    result = model.load_state_dict(package["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise AssertionError(f"fixed semantic strict reload failed: {result}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {"path": str(path.resolve()), "sha256": sha256_file(path),
                   "dataset": str(package["dataset"]), "epoch": int(package["epoch"]),
                   "global_step": int(package["global_step"]), "strict_reload": True,
                   "model_config": package["model_config"]}


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty fixed semantic output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    command = " ".join([sys.executable, *sys.argv])
    base: dict[str, Any] = {
        "format": "locatemot-r0-fixed-semantic-v1", "status": "incomplete",
        "command": command, "cwd": str(Path.cwd().resolve()), "luna_thread": THREAD,
        "seed": SEED, "fixed_source": str(FIXED_SCORE_RECORDS.resolve()),
        "manifest_sha256": check_manifest(),
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "training_run": False,
        "hota_trackeval_run": False, "candidate_deletion": False,
        "candidate_truncation": False, "threshold_fitted": False,
        "checkpoint_selection_frozen_before_fixed_validation": True,
        "failure_root_cause": None, "next_action": "complete internal R0 TrackEval and final report",
    }
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong fixed semantic cwd: {Path.cwd()}")
        records, _metadata = fixed_records()
        index = FixedIndex(records)
        visual = MergedVisualIndex([args.visual_cache, args.calibration_visual_cache])
        language = MergedLanguageCache([Path(value) for value in args.language_root])
        device = torch.device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("fixed semantic CUDA unavailable")
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        runtime = R0RuntimeData(index, visual, language, device)
        checkpoint_paths = {
            "refer_kitti_v1": Path(args.v1_checkpoint).resolve(),
            "refer_kitti_v2": Path(args.v2_checkpoint).resolve(),
        }
        prediction_rows: list[dict[str, Any]] = []
        for dataset, checkpoint_path in checkpoint_paths.items():
            model, info = checkpoint_model(checkpoint_path, device)
            for record in [value for value in records if value["dataset"] == dataset]:
                model_record = {
                    "dataset": record["dataset"], "video": record["video"], "query_id": int(record["query_id"]),
                    "frame_id": int(record["frame_id"]), "sentence": record["sentence"],
                    "candidate_count": int(visual.get(dataset, record["video"], record["frame_id"])["candidate_count"]),
                    "bank_path": str((L69_ROOT / f"{record['video']}.pt").resolve()),
                    "unit_key": record["unit_key"],
                }
                with torch.inference_mode():
                    sample = runtime.prepare(model_record, attach_labels=False)
                    output = model_forward(model, sample)
                    scores = output["membership_logit"][0].detach().float().cpu().tolist()
                    presence = float(output["coverage_presence_logit"][0].detach().float().cpu())
                if len(scores) != int(sample["candidate_count"]) or not _finite(scores) or not math.isfinite(presence):
                    raise AssertionError(f"fixed R0 score/key drift: {record['unit_key']}")
                prediction_rows.append({
                    "unit_key": record["unit_key"], "dataset": dataset, "video": record["video"],
                    "query_id": int(record["query_id"]), "frame_id": int(record["frame_id"]),
                    "split": record["split"], "category_source": "attached_after_prediction",
                    "candidate_count": int(sample["candidate_count"]),
                    "row_offsets": list(sample["batch"].row_offsets),
                    "row_keys": list(sample["row_keys"]),
                    "candidate_indices": list(sample["batch"].candidate_indices),
                    "track_ids": list(sample["batch"].track_ids), "pool_ids": list(sample["batch"].pool_ids),
                    "r0_score": scores, "r0_presence_logit": presence,
                    "checkpoint_info": info, "labels_used_for_prediction": False,
                    "candidate_rows_retained": True, "candidate_deletion": False,
                    "candidate_truncation": False,
                })
                del sample, output
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Only now attach fixed-slice labels for diagnostic metrics.
        safe_map, safe_manifest = safe_target_map(records)
        store = R0BankStore()
        old_gt_cache: dict[str, list[Any]] = {}
        r0_rows: list[dict[str, Any]] = []
        l29_rows: list[dict[str, Any]] = []
        for prediction in prediction_rows:
            record = next(value for value in records if value["unit_key"] == prediction["unit_key"])
            safe = safe_map[(record["dataset"], record["video"], int(record["query_id"]))]
            batch = store.build_frame(record["dataset"], record["video"], int(record["query_id"]),
                                      int(record["frame_id"]), record["sentence"])
            supervision = store.attach_frame_labels(batch, safe.target.get(int(record["frame_id"]), ()))
            if len(supervision["labels"]) != int(prediction["candidate_count"]):
                raise AssertionError(f"fixed R0 label/candidate drift: {record['unit_key']}")
            r0_row = {
                "unit_key": record["unit_key"], "dataset": record["dataset"], "video": record["video"],
                "query_id": int(record["query_id"]), "frame_id": int(record["frame_id"]),
                "category": supervision["category"], "candidate_count": int(prediction["candidate_count"]),
                "score": list(prediction["r0_score"]), "presence_logit": float(prediction["r0_presence_logit"]),
                "labels": [bool(value) for value in supervision["labels"].tolist()],
                "candidate_gt": list(supervision["candidate_gt"]), "target_ids": list(supervision["target_ids"]),
                "candidate_present": bool(supervision["candidate_present"]),
                "row_keys": list(prediction["row_keys"]),
            }
            r0_rows.append(r0_row)
            old = record["old_record"]
            old_labels = [bool(value) for value in old["label"]]
            old_gt = load_old_candidate_gt(record, old_gt_cache)
            l29_rows.append({
                "unit_key": record["unit_key"], "dataset": record["dataset"], "video": record["video"],
                "category": record["old_category"], "candidate_count": len(old["l29"]),
                "score": [float(value) for value in old["l29"]], "presence_logit": 1.0,
                "labels": old_labels, "candidate_gt": old_gt,
                "target_ids": list(record["old_target_ids"]), "candidate_present": bool(any(old_labels)),
                "row_keys": list(range(len(old["l29"]))),
            })
        if len(r0_rows) != 40 or len(l29_rows) != 40:
            raise AssertionError("fixed semantic row count drift")
        metrics = {
            "l29_immutable_control": metric_rows(l29_rows, threshold=L29_THRESHOLD, use_presence=False),
            "r0_fixed_zero_logit": metric_rows(r0_rows, threshold=0.0, use_presence=True),
        }
        l29 = metrics["l29_immutable_control"]
        l29_reproduction = {
            key: {"observed": float(l29[key]), "expected": float(value),
                  "abs_diff": abs(float(l29[key]) - float(value)),
                  "pass": abs(float(l29[key]) - float(value)) <= 1e-9}
            for key, value in EXPECTED_L29.items()
        }
        score_path = out / "score_records.jsonl"
        with score_path.open("w", encoding="utf-8") as handle:
            for prediction, r0_row, l29_row in zip(prediction_rows, r0_rows, l29_rows):
                handle.write(json.dumps({
                    "format": "locatemot-r0-fixed-score-record-v1", "unit_key": prediction["unit_key"],
                    "dataset": prediction["dataset"], "video": prediction["video"],
                    "query_id": int(prediction["query_id"]), "frame_id": int(prediction["frame_id"]),
                    "split": prediction["split"], "r0": r0_row, "l29": l29_row,
                    "candidate_rows_retained": True, "candidate_deletion": False,
                    "candidate_truncation": False, "labels_attached_after_prediction": True,
                }, ensure_ascii=False, sort_keys=True) + "\n")
        if not all(value["pass"] for value in l29_reproduction.values()):
            raise AssertionError(f"immutable L29 reproduction failed: {l29_reproduction}")
        payload = {
            **base, "status": "complete", "unit_count": 40, "calibration_count": 16,
            "validation_count": 24, "score_records": str(score_path.resolve()),
            "score_records_sha256": sha256_file(score_path), "metrics": metrics,
            "l29_reproduction": l29_reproduction, "r0_checkpoints": {
                dataset: {"path": str(path), "sha256": sha256_file(path)}
                for dataset, path in checkpoint_paths.items()
            },
            "safe_target_manifest": safe_manifest,
            "visual_cache_roots": [str(Path(args.visual_cache).resolve()), str(Path(args.calibration_visual_cache).resolve())],
            "language_cache_entries": language.entry_count,
            "candidate_key_count": sum(len(value["row_keys"]) for value in prediction_rows),
            "candidate_keys_complete": True, "all_scores_finite": True,
            "labels_attached_after_prediction": True, "calibration_labels_used_for_selection": False,
            "validation_labels_used_for_selection": False,
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
            "wall_seconds": time.perf_counter() - started,
            "next_action": "run internal R0 TrackEval and finalize R0A report",
        }
        write_json(out / "semantic.json", payload)
        write_json(out / "provenance.json", payload | {"format": "locatemot-r0-fixed-semantic-provenance-v1"})
        write_json(out / "status.json", payload)
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# R0 fixed semantic — INCOMPLETE\n\n" + trace, encoding="utf-8")
        payload = {**base, "failure_root_cause": f"{type(exc).__name__}: {exc}",
                   "traceback_path": str((out / "INCOMPLETE.md").resolve()),
                   "wall_seconds": time.perf_counter() - started}
        write_json(out / "provenance.json", payload); write_json(out / "status.json", payload)
        return 2
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-checkpoint", type=Path, required=True)
    parser.add_argument("--v2-checkpoint", type=Path, required=True)
    parser.add_argument("--visual-cache", type=Path, required=True)
    parser.add_argument("--calibration-visual-cache", type=Path, required=True)
    parser.add_argument("--language-root", type=Path, action="append",
                        default=[Path(value) for value in DEFAULT_LANGUAGE_ROOTS])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
