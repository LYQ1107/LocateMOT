#!/usr/bin/env python3
"""Fixed 16-calibration/24-validation semantic evaluation for L70.

The L70 candidate rows are rebuilt from the L69 bank and therefore are not
assumed to have the old L19/L62 row count.  L29/M0/M54 are immutable controls
read from the accepted L62 record file and are never written back to it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from l70_common import (  # noqa: E402
    L69Bank,
    load_l49_splits,
    load_l62_order,
    load_text_cache,
    safe_torch_load,
    unit_key,
    unit_tensors,
    write_json,
)
from locatemot.models.l70_persistent_set_decoder import L70PersistentSetDecoder  # noqa: E402

INDEX = ROOT / "outputs/l70/audit/data_contract_retry2/unit_records.jsonl"
CHECKPOINT = ROOT / "outputs/l70/train/persistent_set_smoke100/checkpoint_l70_step100.pt"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
OUT = ROOT / "outputs/l70/eval/semantic_16cal24val"


def sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stats(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {"count": int(array.size), "mean": float(array.mean()), "std": float(array.std()),
            "min": float(array.min()), "max": float(array.max())}


def read_index() -> list[dict[str, Any]]:
    return [json.loads(line) for line in INDEX.read_text().splitlines() if line.strip()]


def fit_threshold(rows: list[dict[str, Any]], score_key: str) -> dict[str, Any]:
    values = np.unique(np.concatenate([np.asarray(row[score_key], dtype=np.float64) for row in rows]))
    candidates = values.tolist() + [float(values.min()) - 1e-6, float(values.max()) + 1e-6]
    best: tuple[tuple[float, int, float], float] | None = None
    for threshold in candidates:
        tp = fp = fn = 0
        for row in rows:
            score = np.asarray(row[score_key], dtype=np.float64)
            label = np.asarray(row["labels"], dtype=bool)
            selected = score >= threshold
            tp += int((selected & label).sum())
            fp += int((selected & ~label).sum())
            fn += int((~selected & label).sum())
        f1 = 2.0 * tp / max(1.0, 2.0 * tp + fp + fn)
        # Pre-registered calibration rule: exact observed candidate F1,
        # then fewer false positives, then the higher threshold.
        key = (f1, -fp, float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    assert best is not None
    return {
        "threshold": best[1],
        "objective": "candidate-level F1 on calibration units",
        "tie_rule": "higher F1, fewer FP, then higher threshold",
        "validation_used": False,
    }


def fit_null_rule(rows: list[dict[str, Any]], score_key: str, threshold: float) -> dict[str, Any]:
    values = np.unique([float(row["null_logit"]) for row in rows])
    candidates = values.tolist() + [float(values.min()) - 1e-6, float(values.max()) + 1e-6]
    best: tuple[tuple[float, int, float], float] | None = None
    for null_threshold in candidates:
        predicted: list[bool] = []
        truth: list[bool] = []
        for row in rows:
            score = np.asarray(row[score_key], dtype=np.float64)
            target_present = bool(row["target_present"])
            candidate_above = bool((score >= threshold).any())
            suppress = float(row["null_logit"]) >= float(null_threshold) and not candidate_above
            predicted.append(candidate_above and not suppress)
            truth.append(target_present)
        tp = sum(a and b for a, b in zip(predicted, truth))
        fp = sum(a and not b for a, b in zip(predicted, truth))
        fn = sum((not a) and b for a, b in zip(predicted, truth))
        f1 = 2.0 * tp / max(1.0, 2.0 * tp + fp + fn)
        inactive_false = sum(a and not b for a, b in zip(predicted, truth))
        key = (f1, -inactive_false, float(null_threshold))
        if best is None or key > best[0]:
            best = (key, float(null_threshold))
    assert best is not None
    return {
        "null_threshold": best[1],
        "rule": "suppress all candidate rows iff null_logit >= null_threshold and no candidate score reaches candidate threshold",
        "objective": "frame target-presence F1 on calibration units",
        "tie_rule": "higher F1, fewer inactive false acceptances, then higher null threshold",
        "validation_used": False,
    }


def metric(rows: list[dict[str, Any]], score_key: str, threshold: float,
           null_threshold: float | None = None) -> dict[str, Any]:
    tp = fp = fn = selected = positives = top1 = top5 = empty = 0
    inactive_false_accept = 0
    inactive_fp_rows = 0
    target_present_units = 0
    candidate_present_units = 0
    present_uncovered_units = 0
    strict: list[float] = []
    best: list[float] = []
    average: list[float] = []
    violation: list[bool] = []
    multi: list[float] = []
    continuation_selected = continuation_positive = 0
    values: list[float] = []
    complete = True
    for row in rows:
        score = np.asarray(row[score_key], dtype=np.float64)
        label = np.asarray(row["labels"], dtype=bool)
        if score.size != label.size or not np.isfinite(score).all():
            complete = False
            raise AssertionError(f"score/label failure for {row['unit_key']} / {score_key}")
        values.extend(score.tolist())
        target_present = bool(row["target_present"])
        if target_present:
            target_present_units += 1
        if bool(row.get("candidate_present", label.any())):
            candidate_present_units += 1
        if str(row.get("category", "")) == "present_uncovered":
            present_uncovered_units += 1
        null_suppress = (
            null_threshold is not None
            and float(row["null_logit"]) >= float(null_threshold)
            and not bool((score >= threshold).any())
        )
        selected_mask = (score >= threshold) & (not null_suppress)
        row_tp = int((selected_mask & label).sum())
        row_fp = int((selected_mask & ~label).sum())
        row_fn = int((~selected_mask & label).sum())
        tp += row_tp; fp += row_fp; fn += row_fn; selected += int(selected_mask.sum()); positives += int(label.sum())
        empty += int(not selected_mask.any())
        if str(row.get("category", "")) == "inactive":
            inactive_false_accept += int(selected_mask.any())
            inactive_fp_rows += row_fp
        if target_present and label.any():
            order = np.argsort(-score, kind="stable")
            top1 += int(bool(label[order[:1]].any()))
            top5 += int(bool(label[order[:5]].any()))
        positive = np.flatnonzero(label)
        negative = np.flatnonzero(~label)
        if positive.size and negative.size:
            negative_max = float(score[negative].max())
            strict_value = float(score[positive].min() - negative_max)
            strict.append(strict_value)
            best.append(float(score[positive].max() - negative_max))
            average.append(float(score[positive].mean() - negative_max))
            violation.append(strict_value < 0)
        if positive.size > 1:
            multi.append(float((selected_mask & label).sum() / positive.size))
        cont = np.asarray(row.get("continuation_target", [0] * len(score)), dtype=bool)
        if cont.size == score.size:
            continuation_positive += int(cont.sum())
            continuation_selected += int((selected_mask & cont).sum())
    units = len(rows)
    return {
        "units": units,
        "candidate_rows": int(sum(len(row["labels"]) for row in rows)),
        "positive_rows": positives,
        "target_present_units": target_present_units,
        "candidate_present_units": candidate_present_units,
        "present_uncovered_units": present_uncovered_units,
        "top1": top1 / max(1, target_present_units),
        "top5": top5 / max(1, target_present_units),
        "candidate_precision": tp / max(1, selected),
        "candidate_recall": tp / max(1, tp + fn),
        "fp_per_frame": fp / max(1, units),
        "predictions_per_positive": selected / max(1, positives),
        "hard_violation": float(np.mean(violation)) if violation else None,
        "strict_margin": stats(strict),
        "best_margin": stats(best),
        "average_margin": stats(average),
        "multi_positive_recall": float(np.mean(multi)) if multi else None,
        "empty_rate": empty / max(1, units),
        "null_false_acceptance": inactive_false_accept / max(1, sum(str(row.get("category", "")) == "inactive" for row in rows)),
        "inactive_false_positive_rows": inactive_fp_rows,
        "score_mean": float(np.mean(values)) if values else None,
        "score_std": float(np.std(values)) if values else None,
        "continuation_positive_rows": continuation_positive,
        "continuation_selected_recall": continuation_selected / max(1, continuation_positive),
        "threshold": float(threshold),
        "null_threshold": None if null_threshold is None else float(null_threshold),
        "complete_finite": complete,
    }


def control_rows(old: list[dict[str, Any]], field: str, threshold: float) -> list[dict[str, Any]]:
    result = []
    for row in old:
        labels = [bool(value) for value in row["label"]]
        result.append({
            "unit_key": row["unit_key"], "dataset": row["dataset"], "video": row["video"],
            "frame_id": int(row["frame_id"]), "category": row.get("category", "unknown"),
            "labels": labels, "target_present": bool(any(labels) or row.get("category") == "present_uncovered"),
            "candidate_present": bool(any(labels)), "null_logit": float(row.get("null_logit", 0.0)),
            "score": [float(x) for x in row[field]], "continuation_target": [0] * len(labels),
            "key_audit": row.get("key_audit", {}),
        })
    return result


def serial_row(row: dict[str, Any]) -> dict[str, Any]:
    output = {
        "format": "locatemot-l70-score-record-v1",
        "unit_key": row["unit_key"], "dataset": row["dataset"], "video": row["video"],
        "frame_id": row["frame_id"], "category": row["category"],
        "candidate_count": len(row["labels"]),
        "candidate_keys": row["candidate_keys"], "labels": [int(x) for x in row["labels"]],
        "l70_membership": row["l70_membership"], "l70_track": row["l70_track"],
        "l70_continuation": row["l70_continuation"], "l70_score": row["l70_score"],
        "null_logit": row["null_logit"], "continuation_target": row["continuation_target"],
        "l29": row["l29"], "m0": row["m0"], "m54": row["m54"],
        "old_control_label": row["old_control_label"],
        "row_contract": row["row_contract"],
    }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    base = {
        "format": "locatemot-l70-semantic-evaluation-v1", "status": "running",
        "project_root": str(ROOT), "cwd": os.getcwd(), "command": " ".join(sys.argv),
        "seed": 20260829, "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
        "raw_dense_feature_cache_written": False,
    }
    write_json(args.out / "status.json", base)
    try:
        if Path.cwd().resolve() != ROOT.resolve():
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        if sha256(MANIFEST) != EXPECTED_MANIFEST:
            raise AssertionError("manifest SHA mismatch")
        if not args.checkpoint.exists():
            raise FileNotFoundError(args.checkpoint)
        index = read_index()
        indexed = {str(row["unit_key"]): row for row in index if row.get("split") != "fit"}
        old = load_l62_order()
        if len(old) != 40 or len(indexed) != 40:
            raise AssertionError(f"fixed evaluation count old={len(old)} L70={len(indexed)}")
        old_keys = [str(row["unit_key"]) for row in old]
        if set(old_keys) != set(indexed):
            raise AssertionError("fixed L62 key set does not match L70 index")
        records = [indexed[key] for key in old_keys]
        if [str(row["unit_key"]) for row in records] != old_keys:
            raise AssertionError("fixed order drift")
        text_cache = load_text_cache()
        package = safe_torch_load(args.checkpoint)
        config = package.get("config", {})
        model_args = {key: config[key] for key in ("obs_dim", "text_dim", "hidden", "heads", "layers", "max_history", "dropout") if key in config}
        model = L70PersistentSetDecoder(**model_args).to(args.device)
        missing, unexpected = model.load_state_dict(package["model"], strict=False)
        if missing or unexpected:
            raise AssertionError(f"checkpoint keys missing={missing} unexpected={unexpected}")
        model.eval()
        by_video: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for i, record in enumerate(records):
            by_video.setdefault(str(record["video"]), []).append((i, record))
        evaluated: list[dict[str, Any] | None] = [None] * len(records)
        start = time.perf_counter()
        for video in sorted(by_video):
            bank = L69Bank(video)
            try:
                for i, record in by_video[video]:
                    data = unit_tensors(record, bank, text_cache)
                    with torch.inference_mode():
                        output = model({key: value.to(args.device) for key, value in data.items()})
                    n = int(record["candidate_count"])
                    membership = output["membership_logits"].float().cpu().numpy()
                    track = output["track_logits"].float().cpu().numpy()
                    continuation = output["continuation_logits"].float().cpu().numpy()
                    null_logit = float(output["null_logit"].float().cpu().reshape(-1)[0])
                    if not (len(membership) == len(track) == len(continuation) == n == len(record["labels"]) == len(record["row_keys"])):
                        raise AssertionError(f"candidate length drift: {record['unit_key']}")
                    if not np.isfinite(membership).all() or not np.isfinite(track).all() or not np.isfinite(continuation).all() or not np.isfinite(null_logit):
                        raise AssertionError(f"nonfinite model output: {record['unit_key']}")
                    score = membership + 0.25 * track + 0.10 * continuation
                    evaluated[i] = {
                        "unit_key": record["unit_key"], "dataset": record["dataset"], "video": record["video"],
                        "frame_id": int(record["frame_id"]), "category": record["category"],
                        "labels": [bool(x) for x in record["labels"]], "target_present": bool(record["target_ids"]),
                        "candidate_present": bool(record["candidate_present"]), "null_logit": null_logit,
                        "candidate_keys": record["row_keys"], "continuation_target": [bool(x) for x in record["continuation_target"]],
                        "l70_membership": membership.astype(float).tolist(), "l70_track": track.astype(float).tolist(),
                        "l70_continuation": continuation.astype(float).tolist(), "l70_score": score.astype(float).tolist(),
                        "labels_l70": [bool(x) for x in record["labels"]],
                        "row_contract": {"candidate_count": n, "key_count": len(record["row_keys"]), "ordered": True,
                                         "candidate_truncation": False, "candidate_deletion": False,
                                         "old_l49_begin_end_used": False},
                        "old_control_label": [int(x) for x in old[i]["label"]],
                        "l29": [float(x) for x in old[i]["l29"]], "m0": [float(x) for x in old[i]["m0"]],
                        "m54": [float(x) for x in old[i]["m54"]],
                    }
                    del data, output
            finally:
                bank.close()
        if any(row is None for row in evaluated):
            raise AssertionError("missing evaluated row")
        l70_rows = [row for row in evaluated if row is not None]
        l70_rows = [{**row, "labels": row["labels_l70"], "score": row["l70_score"]} for row in l70_rows]
        cal, val = l70_rows[:16], l70_rows[16:]
        l70_threshold = fit_threshold(cal, "score")
        l70_null = fit_null_rule(cal, "score", l70_threshold["threshold"])
        l70_methods = {
            "candidate_only": {"calibration": metric(cal, "score", l70_threshold["threshold"]),
                                "validation": metric(val, "score", l70_threshold["threshold"])},
            "candidate_plus_null": {"calibration": metric(cal, "score", l70_threshold["threshold"], l70_null["null_threshold"]),
                                     "validation": metric(val, "score", l70_threshold["threshold"], l70_null["null_threshold"])},
            "candidate_threshold_fit": l70_threshold,
            "null_rule_fit": l70_null,
        }
        slice_groups: dict[str, list[dict[str, Any]]] = {}
        for row in l70_rows:
            slice_groups.setdefault(f"dataset:{row['dataset']}", []).append(row)
            slice_groups.setdefault(f"category:{row['category']}", []).append(row)
        l70_methods["slices"] = {
            name: {
                "units": len(group),
                "candidate_only": metric(group, "score", l70_threshold["threshold"]),
                "candidate_plus_null": metric(group, "score", l70_threshold["threshold"], l70_null["null_threshold"]),
            }
            for name, group in sorted(slice_groups.items())
        }
        l29 = control_rows(old, "l29", -1.030576229095459)
        m0 = control_rows(old, "m0", 0.024281445890665054)
        m54 = control_rows(old, "m54", 0.017827898263931274)
        controls = {
            "l29_teacher_immutable": {"calibration": metric(l29[:16], "score", -1.030576229095459),
                                      "validation": metric(l29[16:], "score", -1.030576229095459),
                                      "threshold_source": "accepted L64/L62 control contract"},
            "l53_m0_immutable": {"calibration": metric(m0[:16], "score", 0.024281445890665054),
                                 "validation": metric(m0[16:], "score", 0.024281445890665054),
                                 "threshold_source": "accepted L62 historical calibration"},
            "l54_continuous_immutable": {"calibration": metric(m54[:16], "score", 0.017827898263931274),
                                         "validation": metric(m54[16:], "score", 0.017827898263931274),
                                         "threshold_source": "accepted L62 historical calibration"},
        }
        base_val = controls["l29_teacher_immutable"]["validation"]
        current_val = l70_methods["candidate_plus_null"]["validation"]
        complete = all(row["row_contract"]["candidate_count"] == row["row_contract"]["key_count"] == len(row["l70_score"]) == len(row["labels_l70"]) and row["row_contract"]["candidate_truncation"] is False for row in evaluated)
        checks = {
            "hard_violation_decrease_ge_0.05": current_val["hard_violation"] is not None and base_val["hard_violation"] is not None and current_val["hard_violation"] <= base_val["hard_violation"] - 0.05,
            "recall_drop_le_0.01": current_val["candidate_recall"] >= base_val["candidate_recall"] - 0.01,
            "precision_ge_l29": current_val["candidate_precision"] >= 0.0830188679,
            "fp_per_frame_le_11.125": current_val["fp_per_frame"] <= 11.125,
            "predictions_per_positive_le_4.069": current_val["predictions_per_positive"] <= 4.069,
            "multi_positive_preserved": current_val["multi_positive_recall"] is not None and current_val["multi_positive_recall"] >= 0.8194444444444444 - 0.03,
            "null_not_universal": current_val["null_false_acceptance"] < 1.0,
            "complete_keys_no_deletion": complete,
        }
        gate = {"format": "locatemot-l70-semantic-gate-v1", "status": "semantic_gate_pass" if all(checks.values()) else "semantic_gate_fail",
                "decision": "pass" if all(checks.values()) else "fail", "checks": checks,
                "selection": {"checkpoint": "step100 fixed before validation", "threshold": "fit on first 16 calibration only",
                              "null_rule": "fit on first 16 calibration only", "validation_used_for_selection": False,
                              "candidate_set": "complete L69 rows; no top-k/NMS/deletion"},
                "screening_gt_used": False, "official_test_labels_read": False,
                "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True}
        (args.out / "score_records.jsonl").write_text("".join(json.dumps(serial_row(row), ensure_ascii=False, separators=(",", ":")) + "\n" for row in evaluated if row is not None))
        elapsed = time.perf_counter() - start
        provenance = {**base, "status": "complete", "checkpoint": str(args.checkpoint.resolve()),
                      "checkpoint_sha256": sha256(args.checkpoint), "index": str(INDEX),
                      "l62_control_records": str(ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"),
                      "l62_control_records_sha256": sha256(ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"),
                      "manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST),
                      "calibration_units": 16, "validation_units": 24,
                      "l70_candidate_rows": int(sum(len(row["labels_l70"]) for row in evaluated)),
                      "l29_candidate_rows": int(sum(len(row["old_control_label"]) for row in evaluated)),
                      "candidate_rows_retained": True, "candidate_truncation": False,
                      "old_l49_begin_end_used": False, "old_l49_positive_indices_used": False,
                      "text_tokens_preserved": True, "history_causal": True,
                      "token_region_alignment": "UNALIGNED", "elapsed_seconds": elapsed}
        write_json(args.out / "semantic.json", {"format": base["format"], "status": "complete", "provenance": provenance,
                                                "controls": controls, "l70": l70_methods, "gate": gate})
        write_json(args.out / "gate_decision.json", {**gate, "methods": {"l29": base_val, "l70": current_val},
                                                     "provenance": {"calibration_units": 16, "validation_units": 24,
                                                                     "screening_gt_used": False, "official_test_labels_read": False}})
        write_json(args.out / "provenance.json", provenance)
        write_json(args.out / "status.json", {**base, "status": "complete", "elapsed_seconds": elapsed,
                                              "failure_root_cause": None, "next_action": "stop L70 because semantic gate is a fixed evidence decision"})
        print(json.dumps({"status": gate["status"], "validation": current_val, "checks": checks, "out": str(args.out)}), flush=True)
        return 0
    except Exception as exc:
        write_json(args.out / "status.json", {**base, "status": "INCOMPLETE", "failure_root_cause": f"{type(exc).__name__}: {exc}",
                                              "next_action": "fix first evaluator root cause and rerun in a new output directory"})
        (args.out / "INCOMPLETE.md").write_text(
            "# L70 semantic evaluator INCOMPLETE\n\n"
            f"First actionable root cause: `{type(exc).__name__}: {exc}`\n\n"
            "```text\n" + traceback.format_exc() + "```\n"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
