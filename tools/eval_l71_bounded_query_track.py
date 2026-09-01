#!/usr/bin/env python3
"""Fixed 16-calibration/24-validation evaluation for L71.

L71 deliberately has no NULL head.  The evaluator therefore reports
candidate-only correspondence and inactive false acceptance; it does not add
post-hoc NULL suppression.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from tools.l71_common import (  # noqa: E402
    L71Bank,
    load_l49_splits,
    load_l62_order,
    load_text_cache,
    safe_torch_load,
    sha256_file,
    unit_tensors,
    unit_key,
    write_json,
)
from locatemot.models.l71_bounded_query_track import L71BoundedQueryTrack  # noqa: E402


INDEX = ROOT / "outputs/l71/audit/data_contract_retry3/unit_records.jsonl"
L62_RECORDS = ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
L70_RECORDS = ROOT / "outputs/l70/eval/semantic_16cal24val_retry1/score_records.jsonl"
L70_SEMANTIC = ROOT / "outputs/l70/eval/semantic_16cal24val_retry1/semantic.json"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
SEED = 20260829
L29_THRESHOLD = -1.030576229095459


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(array.size), "mean": float(array.mean()), "std": float(array.std()),
        "min": float(array.min()), "max": float(array.max()),
    }


def fit_threshold(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = np.unique(np.concatenate([np.asarray(row[field], dtype=np.float64) for row in rows]))
    candidates = values.tolist() + [float(values.min()) - 1e-6, float(values.max()) + 1e-6]
    best: tuple[tuple[float, int, float], float] | None = None
    for threshold in candidates:
        tp = fp = fn = 0
        for row in rows:
            score = np.asarray(row[field], dtype=np.float64)
            label = np.asarray(row["labels"], dtype=bool)
            selected = score >= threshold
            tp += int((selected & label).sum())
            fp += int((selected & ~label).sum())
            fn += int((~selected & label).sum())
        f1 = 2.0 * tp / max(1.0, 2.0 * tp + fp + fn)
        key = (f1, -fp, float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    assert best is not None
    return {
        "threshold": best[1],
        "objective": "candidate-level F1 on the 16 calibration units",
        "tie_rule": "higher F1, fewer FP, then higher threshold",
        "validation_used": False,
    }


def metric(rows: list[dict[str, Any]], field: str, threshold: float) -> dict[str, Any]:
    tp = fp = fn = selected = positives = top1 = top5 = empty = 0
    target_present_units = candidate_present_units = present_uncovered_units = 0
    inactive_false_accept = inactive_fp_rows = 0
    strict: list[float] = []
    best: list[float] = []
    average: list[float] = []
    violations: list[bool] = []
    multi: list[float] = []
    query_track_recall: list[float] = []
    values: list[float] = []
    complete = True
    for row in rows:
        score = np.asarray(row[field], dtype=np.float64)
        label = np.asarray(row["labels"], dtype=bool)
        if score.size != label.size or score.size != int(row["candidate_count"]) or not np.isfinite(score).all():
            raise AssertionError(f"score/label/key length failure for {row['unit_key']} / {field}")
        values.extend(score.tolist())
        target_present = bool(row["target_present"])
        if target_present:
            target_present_units += 1
        if bool(row["candidate_present"]):
            candidate_present_units += 1
        if str(row["category"]) == "present_uncovered":
            present_uncovered_units += 1
        selected_mask = score >= threshold
        row_tp = int((selected_mask & label).sum())
        row_fp = int((selected_mask & ~label).sum())
        row_fn = int((~selected_mask & label).sum())
        tp += row_tp
        fp += row_fp
        fn += row_fn
        selected += int(selected_mask.sum())
        positives += int(label.sum())
        empty += int(not selected_mask.any())
        if str(row["category"]) == "inactive":
            inactive_false_accept += int(selected_mask.any())
            inactive_fp_rows += row_fp
        if target_present and label.any():
            order = np.argsort(-score, kind="stable")
            top1 += int(bool(label[order[:1]].any()))
            top5 += int(bool(label[order[:5]].any()))
        if label.any():
            query_track_recall.append(row_tp / float(label.sum()))
        positive = np.flatnonzero(label)
        negative = np.flatnonzero(~label)
        if positive.size and negative.size:
            negative_max = float(score[negative].max())
            strict_value = float(score[positive].min() - negative_max)
            strict.append(strict_value)
            best.append(float(score[positive].max() - negative_max))
            average.append(float(score[positive].mean() - negative_max))
            violations.append(strict_value < 0)
        if positive.size > 1:
            multi.append(float((selected_mask & label).sum() / positive.size))
    units = len(rows)
    return {
        "units": units,
        "candidate_rows": int(sum(int(row["candidate_count"]) for row in rows)),
        "positive_rows": positives,
        "target_present_units": target_present_units,
        "candidate_present_units": candidate_present_units,
        "present_uncovered_units": present_uncovered_units,
        "top1": top1 / max(1, target_present_units),
        "top5": top5 / max(1, target_present_units),
        "candidate_recall": tp / max(1, tp + fn),
        "candidate_precision": tp / max(1, selected),
        "fp_per_frame": fp / max(1, units),
        "predictions_per_positive": selected / max(1, positives),
        "hard_violation": float(np.mean(violations)) if violations else None,
        "strict_margin": stats(strict),
        "best_margin": stats(best),
        "average_margin": stats(average),
        "multi_positive_recall": float(np.mean(multi)) if multi else None,
        "query_track_recall": stats(query_track_recall),
        "empty_rate": empty / max(1, units),
        "inactive_false_acceptance": inactive_false_accept / max(1, sum(str(row["category"]) == "inactive" for row in rows)),
        "inactive_false_positive_rows": inactive_fp_rows,
        "score_mean": float(np.mean(values)) if values else None,
        "score_std": float(np.std(values)) if values else None,
        "identity_switches": "not_implemented_for_frame_correspondence_isolation",
        "complete_finite": complete,
        "threshold": float(threshold),
    }


def old_control_rows(path: Path, field: str, threshold: float) -> tuple[list[dict[str, Any]], float]:
    rows = read_jsonl(path)
    result: list[dict[str, Any]] = []
    for row in rows:
        labels = [bool(value) for value in row["label"]]
        result.append({
            "unit_key": row["unit_key"],
            "dataset": row["dataset"],
            "video": row["video"],
            "frame_id": int(row["frame_id"]),
            "category": row.get("category", "unknown"),
            "candidate_count": len(labels),
            "labels": labels,
            "target_present": bool(any(labels) or row.get("category") == "present_uncovered"),
            "candidate_present": bool(any(labels)),
            "score": [float(value) for value in row[field]],
            "candidate_keys": row.get("key_audit", {}),
        })
    if len(result) != 40:
        raise AssertionError(f"historical control must have 40 rows: {path}")
    return result, threshold


def historical_l70_rows() -> tuple[list[dict[str, Any]], float]:
    rows = read_jsonl(L70_RECORDS)
    semantic = json.loads(L70_SEMANTIC.read_text())
    threshold = float(semantic["l70"]["candidate_threshold_fit"]["threshold"])
    result: list[dict[str, Any]] = []
    for row in rows:
        labels = [bool(value) for value in row["labels"]]
        result.append({
            "unit_key": row["unit_key"], "dataset": row["dataset"], "video": row["video"],
            "frame_id": int(row["frame_id"]), "category": row["category"],
            "candidate_count": len(labels), "labels": labels,
            "target_present": bool(any(labels) or row["category"] == "present_uncovered"),
            "candidate_present": bool(any(labels)), "score": [float(value) for value in row["l70_score"]],
            "candidate_keys": row.get("candidate_keys", []),
        })
    if len(result) != 40:
        raise AssertionError("historical L70 score record count is not 40")
    return result, threshold


def load_fixed_l71_index() -> list[dict[str, Any]]:
    rows = read_jsonl(INDEX)
    indexed = {str(row["unit_key"]): row for row in rows if row.get("index_role") == "fixed_eval"}
    old = load_l62_order()
    keys = [str(row["unit_key"]) for row in old]
    if len(indexed) != 40 or len(set(keys)) != 40 or set(keys) != set(indexed):
        raise AssertionError("L71 fixed key set does not match immutable L62 order")
    result = [indexed[key] for key in keys]
    if [str(row["unit_key"]) for row in result] != keys:
        raise AssertionError("L71 fixed key order drift")
    return result


def load_model(checkpoint: Path, device: torch.device) -> L71BoundedQueryTrack:
    package = safe_torch_load(checkpoint)
    config = package.get("config", {})
    model = L71BoundedQueryTrack(
        obs_dim=int(config.get("obs_dim", 1432)),
        text_dim=int(config.get("text_dim", 768)),
        hidden=int(config.get("hidden", 192)),
        max_history=int(config.get("max_history", 8)),
        temperature=float(config.get("temperature", 0.07)),
    ).to(device)
    model.load_state_dict(package["model"], strict=True)
    model.eval()
    return model


def evaluate_checkpoints(records: list[dict[str, Any]], checkpoints: dict[str, Path], device: torch.device) -> list[dict[str, Any]]:
    text_cache = load_text_cache()
    models = {name: load_model(path, device) for name, path in checkpoints.items()}
    by_video: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, record in enumerate(records):
        by_video[str(record["video"])].append((index, record))
    evaluated: list[dict[str, Any] | None] = [None] * len(records)
    bank_hashes: dict[str, str] = {}
    for video in sorted(by_video):
        bank = L71Bank(video)
        bank_hashes[video] = bank.sha256
        try:
            for index, record in by_video[video]:
                data_cpu = unit_tensors(record, bank, text_cache)
                data = {key: value.to(device, non_blocking=True) for key, value in data_cpu.items()}
                current = data_cpu["current"]
                frozen_cosine = F.cosine_similarity(current[:, :512], current[:, 512:1024], dim=1).numpy()
                if not np.isfinite(frozen_cosine).all():
                    raise AssertionError(f"nonfinite frozen history cosine: {record['unit_key']}")
                values: dict[str, list[float]] = {}
                with torch.inference_mode():
                    for name, model in models.items():
                        output = model(data)
                        logits = output["correspondence_logits"].float().cpu()
                        if logits.numel() != int(record["candidate_count"]):
                            raise AssertionError(f"candidate output count drift: {record['unit_key']}")
                        values[name] = logits.numpy().astype(float).tolist()
                n = int(record["candidate_count"])
                if any(len(value) != n for value in values.values()) or len(record["row_keys"]) != n:
                    raise AssertionError(f"candidate row length drift: {record['unit_key']}")
                if [int(key[-1]) for key in record["row_keys"]] != [int(row) for row in record["row_offsets"]]:
                    raise AssertionError(f"candidate row order drift: {record['unit_key']}")
                evaluated[index] = {
                    "format": "locatemot-l71-score-record-v1",
                    "unit_key": record["unit_key"], "dataset": record["dataset"],
                    "video": record["video"], "frame_id": int(record["frame_id"]),
                    "query_id": int(record["query_id"]), "category": record["category"],
                    "candidate_count": n, "candidate_keys": record["row_keys"],
                    "labels": [bool(value) for value in record["labels"]],
                    "target_present": bool(record["target_ids"]),
                    "candidate_present": bool(record["candidate_present"]),
                    "present_uncovered": record["category"] == "present_uncovered",
                    "frozen_history_cosine": frozen_cosine.astype(float).tolist(),
                    "null_head": "not_implemented_for_isolation",
                    **values,
                    "row_contract": {
                        "candidate_count": n, "key_count": len(record["row_keys"]),
                        "ordered": True, "candidate_truncation": False,
                        "candidate_deletion": False, "old_l49_begin_end_used": False,
                        "old_l49_positive_indices_used": False,
                    },
                }
                del data, data_cpu
        finally:
            bank.close()
    if any(row is None for row in evaluated):
        raise AssertionError("missing fixed evaluation row")
    return [row for row in evaluated if row is not None]


def generic_rows(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return [{**row, "score": row[field]} for row in rows]


def split_metrics(rows: list[dict[str, Any]], field: str, threshold: float) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[f"dataset:{row['dataset']}"].append(row)
        groups[f"category:{row['category']}"].append(row)
    return {
        name: metric(group, field, threshold)
        for name, group in sorted(groups.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint100", type=Path, required=True)
    parser.add_argument("--checkpoint250", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    base = {
        "format": "locatemot-l71-semantic-evaluation-v1",
        "status": "running", "project_root": str(ROOT), "cwd": os.getcwd(),
        "command": " ".join(sys.argv), "seed": SEED,
        "inputs": {"unit_index": str(INDEX), "l62_records": str(L62_RECORDS), "l70_records": str(L70_RECORDS)},
        "outputs": {"root": str(args.out)},
        "calibration_units": 16, "validation_units": 24,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        "raw_dense_feature_cache_written": False,
        "null_head": "not_implemented_for_isolation",
    }
    write_json(args.out / "status.json", base)
    try:
        if ROOT.resolve() != Path.cwd().resolve():
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        manifest_sha = sha256_file(MANIFEST)
        if manifest_sha != EXPECTED_MANIFEST:
            raise AssertionError(f"manifest SHA mismatch: {manifest_sha}")
        records = load_fixed_l71_index()
        old_l62 = load_l62_order()
        old_l70 = read_jsonl(L70_RECORDS)
        if [str(row["unit_key"]) for row in old_l62] != [str(row["unit_key"]) for row in old_l70]:
            raise AssertionError("L70 historical records are not in the immutable 40-unit order")
        device = torch.device(args.device)
        if device.type == "cuda" and device.index not in (None, 0):
            raise RuntimeError(f"L71 evaluation requires GPU0, got {device}")
        checkpoints = {"step100": args.checkpoint100, "step250": args.checkpoint250}
        for path in checkpoints.values():
            if not path.exists():
                raise FileNotFoundError(path)
        evaluated = evaluate_checkpoints(records, checkpoints, device)
        cal, val = evaluated[:16], evaluated[16:]
        methods: dict[str, Any] = {}
        thresholds: dict[str, Any] = {}
        for name in ("step100", "step250"):
            fitted = fit_threshold(cal, name)
            thresholds[name] = fitted
            methods[name] = {
                "calibration": metric(cal, name, fitted["threshold"]),
                "validation": metric(val, name, fitted["threshold"]),
                "slices": split_metrics(evaluated, name, fitted["threshold"]),
            }
        frozen = generic_rows(evaluated, "frozen_history_cosine")
        frozen_threshold = fit_threshold(frozen[:16], "score")
        methods["frozen_observation_history_cosine_control"] = {
            "calibration": metric(frozen[:16], "score", frozen_threshold["threshold"]),
            "validation": metric(frozen[16:], "score", frozen_threshold["threshold"]),
            "threshold": frozen_threshold,
            "meaning": "expression-independent clip/history_clip identity diagnostic; not a query correspondence result",
        }
        l29_rows, l29_threshold = old_control_rows(L62_RECORDS, "l29", L29_THRESHOLD)
        l70_rows, l70_threshold = historical_l70_rows()
        methods["l29_teacher_immutable"] = {
            "calibration": metric(l29_rows[:16], "score", l29_threshold),
            "validation": metric(l29_rows[16:], "score", l29_threshold),
            "threshold_source": "accepted immutable L62/L64 contract",
        }
        methods["l70_historical_immutable"] = {
            "calibration": metric(l70_rows[:16], "score", l70_threshold),
            "validation": metric(l70_rows[16:], "score", l70_threshold),
            "threshold_source": "accepted L70 semantic evaluation calibration fit",
        }
        base_val = methods["l29_teacher_immutable"]["validation"]
        final_val = methods["step250"]["validation"]
        l29_domains = {
            name.split(":", 1)[1]: value
            for name, value in split_metrics(l29_rows[16:], "score", l29_threshold).items()
            if name.startswith("dataset:")
        }
        l71_domains = {
            name.split(":", 1)[1]: value
            for name, value in methods["step250"]["slices"].items()
            if name.startswith("dataset:")
        }
        domain_hard = {
            dataset: {
                "l29_hard_violation": l29_domains[dataset]["hard_violation"],
                "l71_hard_violation": l71_domains[dataset]["hard_violation"],
                "decrease": None if l29_domains[dataset]["hard_violation"] is None or l71_domains[dataset]["hard_violation"] is None else l29_domains[dataset]["hard_violation"] - l71_domains[dataset]["hard_violation"],
            }
            for dataset in sorted(set(l29_domains) & set(l71_domains))
        }
        checks = {
            "hard_negative_decrease_ge_0.05": final_val["hard_violation"] is not None and base_val["hard_violation"] is not None and final_val["hard_violation"] <= base_val["hard_violation"] - 0.05,
            "recall_drop_le_0.01": final_val["candidate_recall"] >= base_val["candidate_recall"] - 0.01,
            "precision_at_least_l29": final_val["candidate_precision"] >= base_val["candidate_precision"],
            "fp_per_frame_le_11.125": final_val["fp_per_frame"] <= 11.125,
            "predictions_per_positive_le_4.069": final_val["predictions_per_positive"] <= 4.069,
            "multi_positive_preserved": final_val["multi_positive_recall"] is not None and base_val["multi_positive_recall"] is not None and final_val["multi_positive_recall"] >= base_val["multi_positive_recall"] - 0.03,
            "score_std_non_degenerate": final_val["score_std"] is not None and final_val["score_std"] > 1e-3,
            "both_domain_hard_separation_improved": all(value["decrease"] is not None and value["decrease"] >= 0.05 for value in domain_hard.values()),
            "complete_keys_no_deletion": all(
                row["row_contract"]["candidate_count"] == row["row_contract"]["key_count"] == len(row["step250"]) == len(row["labels"])
                and not row["row_contract"]["candidate_truncation"] and not row["row_contract"]["candidate_deletion"]
                and np.isfinite(np.asarray(row["step250"], dtype=np.float64)).all()
                for row in evaluated
            ),
        }
        gate = {
            "format": "locatemot-l71-bounded-correspondence-gate-v1",
            "status": "bounded_correspondence_probe_pass" if all(checks.values()) else "bounded_correspondence_probe_fail",
            "decision": "pass" if all(checks.values()) else "fail",
            "formal_step": "step250",
            "step100_role": "learning-curve control only; not selected by validation",
            "checks": checks,
            "domain_hard_negative": domain_hard,
            "null_head": "not_implemented_for_isolation; inactive false acceptance is reported, no NULL filter applied",
            "selection": {"threshold": "fit separately on first 16 calibration units for each fixed checkpoint", "validation_used_for_selection": False, "candidate_set": "complete L69 rows; no top-k/NMS/deletion"},
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
        }
        serial = []
        for row in evaluated:
            serial.append({
                "format": row["format"], "unit_key": row["unit_key"], "dataset": row["dataset"], "video": row["video"],
                "query_id": row["query_id"], "frame_id": row["frame_id"], "category": row["category"],
                "candidate_count": row["candidate_count"], "candidate_keys": row["candidate_keys"],
                "labels": [int(value) for value in row["labels"]], "target_present": row["target_present"],
                "candidate_present": row["candidate_present"], "present_uncovered": row["present_uncovered"],
                "frozen_history_cosine": row["frozen_history_cosine"], "step100": row["step100"], "step250": row["step250"],
                "null_head": row["null_head"], "row_contract": row["row_contract"],
            })
        (args.out / "score_records.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in serial))
        provenance = {
            **base, "status": "complete", "manifest": str(MANIFEST), "manifest_sha256": manifest_sha,
            "index_sha256": sha256_file(INDEX), "l62_records_sha256": sha256_file(L62_RECORDS),
            "l70_records_sha256": sha256_file(L70_RECORDS),
            "checkpoints": {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in checkpoints.items()},
            "candidate_rows": int(sum(row["candidate_count"] for row in evaluated)),
            "candidate_rows_retained": True, "candidate_truncation": False, "candidate_deletion": False,
            "fixed_order": True, "history_causal": True, "text_tokens_preserved": True,
            "old_l49_begin_end_used": False, "old_l49_positive_indices_used": False,
            "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
            "token_region_alignment": "UNALIGNED", "null_head": "not_implemented_for_isolation",
            "elapsed_seconds": time.perf_counter() - started,
        }
        write_json(args.out / "semantic.json", {"format": base["format"], "status": "complete", "methods": methods, "thresholds": thresholds, "gate": gate, "provenance": provenance})
        write_json(args.out / "gate_decision.json", gate)
        write_json(args.out / "provenance.json", provenance)
        write_json(args.out / "status.json", {**base, "status": "complete", "failure_root_cause": None, "next_action": "write the L71 evidence report; do not run screening or TrackEval unless the correspondence gate passes and a new instruction authorizes it"})
        print(json.dumps({"status": gate["status"], "validation_step250": final_val, "checks": checks, "out": str(args.out)}), flush=True)
        return 0
    except Exception as exc:
        write_json(args.out / "status.json", {**base, "status": "INCOMPLETE", "failure_root_cause": f"{type(exc).__name__}: {exc}", "next_action": "fix only the first evaluator root cause and rerun in a new output directory"})
        (args.out / "INCOMPLETE.md").write_text("# L71 semantic evaluation INCOMPLETE\n\n" + f"First actionable root cause: `{type(exc).__name__}: {exc}`\n\n```text\n" + traceback.format_exc() + "```\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
