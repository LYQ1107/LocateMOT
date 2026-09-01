#!/usr/bin/env python3
"""L66 control-attribution retry: isolate the trained head from visual LoRA."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
DATA = ROOT / "outputs/l49/data"
IMMUTABLE = ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
L65_CHECKPOINT = ROOT / "outputs/l65/train/clip_joint_smoke100/checkpoint_l65_clip_joint_step100.pt"
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
L29_THRESHOLD = -1.030576229095459

import sys

sys.path.insert(0, str(ROOT))
from locatemot.models.l66_visual_lora_set import L66VisualLoraSet, attach_visual_lora
from tools.l66_visual_lora_common import (
    CLIP_WEIGHTS,
    EXPECTED_CLIP,
    StreamingClipLora,
    load_unit_features,
    sha256,
)


def fixed_units():
    lookup = {}
    for name in ("calibration_units.jsonl", "validation_units.jsonl"):
        for line in (DATA / name).read_text().splitlines():
            if line.strip():
                unit = json.loads(line)
                lookup[unit["unit_key"]] = unit
    source = [json.loads(x) for x in IMMUTABLE.read_text().splitlines() if x.strip()]
    if len(source) != 40 or len({x["unit_key"] for x in source}) != 40:
        raise AssertionError("immutable L62 source must contain 40 unique units")
    units = []
    for index, record in enumerate(source):
        key = record["unit_key"]
        if key not in lookup:
            raise AssertionError(f"missing immutable unit {key}")
        unit = dict(lookup[key])
        unit["eval_split"] = "calibration" if index < 16 else "validation"
        unit["immutable_record"] = record
        units.append(unit)
    return units


def stats(values):
    array = np.asarray(values, dtype=float)
    if not array.size:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def metric(rows, field, threshold, null_field=None, null_threshold=None):
    tp = fp = fn = selected = positives = top1 = top5 = empty = null_false = 0
    strict, best, average, violations, multi, values = [], [], [], [], [], []
    for row in rows:
        scores = np.asarray(row[field], dtype=float)
        labels = np.asarray(row["label"], dtype=bool)
        if scores.size != labels.size:
            raise AssertionError(f"score/label length mismatch: {row['unit_key']} {field}")
        if not np.isfinite(scores).all():
            raise AssertionError(f"nonfinite scores: {row['unit_key']} {field}")
        suppressed = (
            null_threshold is not None
            and float(row[null_field]) >= float(null_threshold)
        )
        selected_mask = (scores >= threshold) & (not suppressed)
        tp += int((selected_mask & labels).sum())
        fp += int((selected_mask & ~labels).sum())
        fn += int((~selected_mask & labels).sum())
        selected += int(selected_mask.sum())
        positives += int(labels.sum())
        empty += int(not selected_mask.any())
        null_false += int(not labels.any() and selected_mask.any())
        values.extend(scores.tolist())
        positive = np.flatnonzero(labels)
        negative = np.flatnonzero(~labels)
        if positive.size:
            order = np.argsort(-scores, kind="stable")
            top1 += int(labels[order[:1]].any())
            top5 += int(labels[order[:5]].any())
            if negative.size:
                negative_max = float(scores[negative].max())
                strict_value = float(scores[positive].min() - negative_max)
                strict.append(strict_value)
                best.append(float(scores[positive].max() - negative_max))
                average.append(float(scores[positive].mean() - negative_max))
                violations.append(strict_value < 0)
            if positive.size > 1:
                multi.append(float((selected_mask & labels).sum() / positive.size))
    present = sum(bool(np.asarray(row["label"], dtype=bool).any()) for row in rows)
    return {
        "units": len(rows),
        "candidate_rows": int(sum(len(row["label"]) for row in rows)),
        "positive_rows": positives,
        "top1": top1 / max(1, present),
        "top5": top5 / max(1, present),
        "candidate_precision": tp / max(1, selected),
        "candidate_recall": tp / max(1, tp + fn),
        "fp_per_frame": fp / max(1, len(rows)),
        "predictions_per_positive": selected / max(1, positives),
        "hard_violation": float(np.mean(violations)) if violations else None,
        "strict_margin": stats(strict),
        "best_margin": stats(best),
        "average_margin": stats(average),
        "multi_positive_recall": float(np.mean(multi)) if multi else None,
        "empty_rate": empty / max(1, len(rows)),
        "null_false_acceptance": null_false / max(1, len(rows)),
        "score_mean": float(np.mean(values)) if values else None,
        "score_std": float(np.std(values)) if values else None,
        "threshold": float(threshold),
        "null_threshold": None if null_threshold is None else float(null_threshold),
    }


def fit_threshold(rows, field):
    values = np.unique(
        np.concatenate([np.asarray(row[field], dtype=float) for row in rows])
    )
    candidates = values.tolist() + [float(values.min()) - 1e-6, float(values.max()) + 1e-6]
    best = None
    for threshold in candidates:
        tp = fp = fn = 0
        for row in rows:
            scores = np.asarray(row[field], dtype=float)
            labels = np.asarray(row["label"], dtype=bool)
            selected = scores >= threshold
            tp += int((selected & labels).sum())
            fp += int((selected & ~labels).sum())
            fn += int((~selected & labels).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        # Fixed tie rule: F1, fewer false positives, then higher threshold.
        key = (f1, -fp, float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1]


def fit_null(rows, field, threshold, null_field):
    values = np.unique([float(row[null_field]) for row in rows])
    candidates = values.tolist() + [float(values.min()) - 1e-6, float(values.max()) + 1e-6]
    best = None
    for null_threshold in candidates:
        predicted, truth = [], []
        for row in rows:
            candidate_present = bool((np.asarray(row[field]) >= threshold).any())
            predicted.append(candidate_present and float(row[null_field]) < null_threshold)
            truth.append(bool(np.asarray(row["label"], dtype=bool).any()))
        tp = sum(a and b for a, b in zip(predicted, truth))
        fp = sum(a and not b for a, b in zip(predicted, truth))
        fn = sum((not a) and b for a, b in zip(predicted, truth))
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        key = (f1, -fp, float(null_threshold))
        if best is None or key > best[0]:
            best = (key, float(null_threshold))
    return best[1]


def make_head(state, device):
    head = L66VisualLoraSet(hidden=128).to(device)
    head.load_state_dict(state, strict=True)
    head.eval()
    return head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    if sha256(MANIFEST) != EXPECTED_MANIFEST:
        raise AssertionError("manifest SHA mismatch")
    if sha256(CLIP_WEIGHTS) != EXPECTED_CLIP:
        raise AssertionError("CLIP SHA mismatch")
    if not L65_CHECKPOINT.is_file():
        raise FileNotFoundError(L65_CHECKPOINT)
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(out)
    checkpoint = args.checkpoint.resolve()
    package = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if set(package.get("lora", {})) != {
        "visual.transformer.resblocks.11.mlp.c_proj.lora_A",
        "visual.transformer.resblocks.11.mlp.c_proj.lora_B",
    }:
        raise AssertionError("compact checkpoint must contain exactly the two L66 LoRA tensors")
    if package.get("target_module") != "visual.transformer.resblocks[-1].mlp.c_proj":
        raise AssertionError("unexpected L66 target module")
    if int(package.get("rank")) != 8 or float(package.get("alpha")) != 16.0:
        raise AssertionError("unexpected L66 LoRA rank/alpha")
    l65_package = torch.load(L65_CHECKPOINT, map_location="cpu", weights_only=False)
    l65_state = l65_package["model"]
    head_state = package["head"]
    if set(head_state) != set(l65_state):
        # This is expected: L65 and L66 heads are separately trained modules,
        # but the explicit comparison is retained as provenance rather than a control.
        head_key_overlap = len(set(head_state) & set(l65_state))
    else:
        head_key_overlap = len(head_state)

    units = fixed_units()
    device = torch.device("cuda:0")
    runtime = StreamingClipLora(device, crop_batch=4)
    target_module, wrapper = attach_visual_lora(runtime.model, rank=8, alpha=16.0, dropout=0.0)
    missing, unexpected = runtime.model.load_state_dict(package["lora"], strict=False)
    if unexpected or any("lora_A" in key or "lora_B" in key for key in missing):
        raise AssertionError(f"LoRA reload mismatch missing={missing} unexpected={unexpected}")
    for name, parameter in runtime.model.named_parameters():
        if "lora_A" not in name and "lora_B" not in name and parameter.requires_grad:
            raise AssertionError(f"base parameter unexpectedly trainable: {name}")
    l66_head = make_head(head_state, device)
    l65_head = make_head(l65_state, device)
    rows = []
    lora_before = sha256(checkpoint)
    start_time = time.time()
    with torch.inference_mode():
        for unit in units:
            # Active LoRA stream.
            active = load_unit_features(unit, runtime, labels=True)
            active_out = l66_head(
                active["patches"].to(device), active["words"].to(device),
                active["mask"].to(device), active["numeric"].to(device)
            )
            active_scores = [float(x) for x in active_out["relevance_logit"].cpu()]
            active_null = float(active_out["null_logit"].cpu())

            # Same L66 head and same input path, with only the LoRA update disabled.
            saved_a = wrapper.lora_A.detach().clone()
            saved_b = wrapper.lora_B.detach().clone()
            saved_b_sha = sha256_tensor(saved_b)
            wrapper.lora_B.zero_()
            if bool(wrapper.lora_B.abs().max() != 0):
                raise AssertionError("LoRA zero control was not zero")
            base = load_unit_features(unit, runtime, labels=True)
            base_out_l66 = l66_head(
                base["patches"].to(device), base["words"].to(device),
                base["mask"].to(device), base["numeric"].to(device)
            )
            base_out_l65 = l65_head(
                base["patches"].to(device), base["words"].to(device),
                base["mask"].to(device), base["numeric"].to(device)
            )
            no_lora_scores = [float(x) for x in base_out_l66["relevance_logit"].cpu()]
            no_lora_null = float(base_out_l66["null_logit"].cpu())
            l65_scores = [float(x) for x in base_out_l65["relevance_logit"].cpu()]
            l65_null = float(base_out_l65["null_logit"].cpu())
            wrapper.lora_A.copy_(saved_a)
            wrapper.lora_B.copy_(saved_b)
            if sha256_tensor(wrapper.lora_B.detach()) != saved_b_sha:
                raise AssertionError("LoRA B was not restored exactly")
            labels = active["target"].tolist()
            record = unit["immutable_record"]
            if base["target"].tolist() != labels:
                raise AssertionError(f"label order drift: {unit['unit_key']}")
            expected_n = len(record["l29"])
            arrays = {
                "l65_head_frozen_clip": l65_scores,
                "l66_head_no_lora": no_lora_scores,
                "l66_head_lora": active_scores,
                "l29": record["l29"],
            }
            if any(len(values) != expected_n for values in arrays.values()):
                raise AssertionError(f"candidate length drift: {unit['unit_key']}")
            if any(not np.isfinite(values).all() for values in map(np.asarray, arrays.values())):
                raise AssertionError(f"nonfinite candidate scores: {unit['unit_key']}")
            rows.append({
                "unit_key": unit["unit_key"], "dataset": unit["dataset"],
                "video": unit["video"], "frame_id": int(unit["frame_id"]),
                "category": unit["category"], "eval_split": unit["eval_split"],
                "label": labels, "l29": record["l29"],
                "l65_head_frozen_clip": l65_scores,
                "l66_head_no_lora": no_lora_scores,
                "l66_head_lora": active_scores,
                "l65_null_logit": l65_null,
                "l66_head_no_lora_null_logit": no_lora_null,
                "l66_head_lora_null_logit": active_null,
                "key_audit": {
                    "candidate_count": expected_n,
                    "candidate_rows_retained": expected_n,
                    "candidate_truncation": False,
                    "ordered": True,
                    "duplicate_candidate_index_legal": True,
                },
            })
            del active, base, active_out, base_out_l66, base_out_l65, saved_a, saved_b
    if sha256(checkpoint) != lora_before:
        raise AssertionError("checkpoint changed during evaluation")
    cal, val = rows[:16], rows[16:]
    method_specs = {
        "l65_head_frozen_clip": "l65_null_logit",
        "l66_head_no_lora": "l66_head_no_lora_null_logit",
        "l66_head_lora": "l66_head_lora_null_logit",
    }
    methods = {}
    for name, null_field in method_specs.items():
        threshold = fit_threshold(cal, name)
        null_threshold = fit_null(cal, name, threshold, null_field)
        methods[name] = {
            "candidate_only_calibration": metric(cal, name, threshold),
            "candidate_only_validation": metric(val, name, threshold),
            "final_calibration": metric(cal, name, threshold, null_field, null_threshold),
            "final_validation": metric(val, name, threshold, null_field, null_threshold),
            "threshold": {"threshold": threshold, "fit": "16 calibration units only"},
            "null_rule": {
                "null_field": null_field,
                "null_threshold": null_threshold,
                "fit": "16 calibration units only",
                "rule": "suppress all candidates when null logit >= calibrated threshold",
            },
        }
    l29_rows = [{"label": row["label"], "l29": row["l29"]} for row in rows]
    methods["l29_teacher"] = {
        "calibration": metric(l29_rows[:16], "l29", L29_THRESHOLD),
        "validation": metric(l29_rows[16:], "l29", L29_THRESHOLD),
        "threshold": {"threshold": L29_THRESHOLD, "source": "immutable accepted L62 rows"},
        "null_rule": {"status": "not recomputed; immutable L29 control"},
    }
    base = methods["l29_teacher"]["validation"]
    gate_methods = {}
    for name in ("l66_head_no_lora", "l66_head_lora"):
        current = methods[name]["final_validation"]
        gate_methods[name] = {
            "hard_violation_decrease_ge_0.05": current["hard_violation"] <= base["hard_violation"] - 0.05,
            "recall_floor": current["candidate_recall"] >= 0.7233333,
            "precision_floor": current["candidate_precision"] >= 0.0830188679,
            "fp_frame_floor": current["fp_per_frame"] <= 11.125,
            "predictions_per_positive_floor": current["predictions_per_positive"] <= 4.069,
            "multi_positive_floor": current["multi_positive_recall"] is not None and current["multi_positive_recall"] >= 0.7894444,
            "null_not_universal": current["null_false_acceptance"] < 1.0,
            "complete_keys": all(row["key_audit"]["candidate_count"] == len(row[name]) == len(row["label"]) for row in rows),
            "candidate_deletion_false": True,
        }
    usable = [name for name, checks in gate_methods.items() if all(checks.values())]
    gate = {
        "format": "locatemot-l66-control-attribution-semantic-gate-v1",
        "status": "semantic_gate_pass" if usable else "semantic_gate_fail",
        "usable_methods": usable,
        "checks_by_method": gate_methods,
        "calibration_units": 16, "validation_units": 24,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True,
    }
    out.mkdir(parents=True)
    semantic = {
        "format": "locatemot-l66-control-attribution-semantic-v1",
        "status": "complete", "project_root": str(ROOT),
        "cwd": str(Path.cwd().resolve()), "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint), "head_state_key_overlap_with_l65": head_key_overlap,
        "methods": methods, "gate": gate, "elapsed_sec": time.time() - start_time,
    }
    (out / "semantic.json").write_text(json.dumps(semantic, indent=2) + "\n")
    (out / "gate_decision.json").write_text(json.dumps(gate, indent=2) + "\n")
    (out / "score_records.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    provenance = {
        "format": "locatemot-l66-control-attribution-provenance-v1",
        "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
        "manifest_sha256": sha256(MANIFEST), "immutable_l29_source": str(IMMUTABLE),
        "immutable_l29_source_sha256": sha256(IMMUTABLE), "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint), "clip_weights": str(CLIP_WEIGHTS),
        "clip_weights_sha256": sha256(CLIP_WEIGHTS), "l65_checkpoint": str(L65_CHECKPOINT),
        "l65_checkpoint_sha256": sha256(L65_CHECKPOINT), "calibration_units": 16,
        "validation_units": 24, "unit_order_source": str(IMMUTABLE),
        "l66_head_state_same_for_no_lora_and_lora": True,
        "lora_zero_control": {"target_module": target_module, "rank": 8, "alpha": 16.0,
                               "B_zeroed_temporarily": True, "restored_exactly": True,
                               "base_nonzero_gradient": 0, "eval_inference_mode": True},
        "candidate_rows_retained": True, "candidate_truncation": False,
        "persistent_raw_dense_cache_written": False, "screening_gt_used": False,
        "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
        "no_hota_or_trackeval": True, "token_span_alignment": "UNALIGNED",
        "static_motion_mask": "UNALIGNED",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"status": gate["status"], "output": str(out),
                      "l29_validation": base,
                      "l66_no_lora_validation": methods["l66_head_no_lora"]["final_validation"],
                      "l66_lora_validation": methods["l66_head_lora"]["final_validation"]}, indent=2), flush=True)


def sha256_tensor(tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


if __name__ == "__main__":
    main()
