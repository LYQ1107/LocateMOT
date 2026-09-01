#!/usr/bin/env python3
"""Fixed 16-calibration/24-validation evaluation for L64 raw patch probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
DATA = ROOT / "outputs/l49/data"
IMMUTABLE = ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
L29_THRESHOLD = -1.030576229095459

import sys
sys.path.insert(0, str(ROOT))
from locatemot.models.l64_raw_patch_set import L64RawPatchSet
from tools.l64_raw_patch_common import StreamingOpenAIClip, sha256


def units40():
    by_key = {}
    for filename in ("calibration_units.jsonl", "validation_units.jsonl"):
        for line in (DATA / filename).read_text().splitlines():
            if line.strip():
                row = json.loads(line); by_key[row["unit_key"]] = row
    rows = [json.loads(x) for x in IMMUTABLE.read_text().splitlines() if x.strip()]
    if len(rows) != 40 or len({r["unit_key"] for r in rows}) != 40:
        raise AssertionError("immutable unit count/key contract")
    out = []
    for i, record in enumerate(rows):
        if record["unit_key"] not in by_key:
            raise KeyError(record["unit_key"])
        unit = dict(by_key[record["unit_key"]])
        unit["eval_split"] = "calibration" if i < 16 else "validation"
        unit["control"] = {k: record[k] for k in ("l29", "m0", "m54")}
        unit["label"] = record["label"]
        out.append(unit)
    return out


def numeric(t, begin, end):
    return torch.cat((t["geometry"][begin:end].float(), t["motion"][begin:end].float(),
                      t["lifecycle"][begin:end].float(), t["context"][begin:end].float(),
                      t["objectness"][begin:end].float().reshape(-1, 1)), dim=1)


def dist(values):
    a = np.asarray(values, dtype=float)
    return {"count": int(a.size), "mean": float(a.mean()) if a.size else None,
            "std": float(a.std()) if a.size else None, "min": float(a.min()) if a.size else None,
            "max": float(a.max()) if a.size else None}


def metric(rows, threshold, null_threshold=None, field="score"):
    tp = fp = fn = selected = top1 = top5 = empty = null_false = positives = 0
    strict, best, average, violations, multi = [], [], [], [], []
    score_values = []
    for row in rows:
        s = np.asarray(row[field], dtype=float); y = np.asarray(row["label"], dtype=bool)
        if len(s) != len(y) or not np.isfinite(s).all():
            raise AssertionError(f"score/label contract {row['unit_key']}")
        suppress = null_threshold is not None and float(row.get("null_logit", -np.inf)) >= float(null_threshold)
        z = (s >= float(threshold)) & (not suppress)
        tp += int((z & y).sum()); fp += int((z & ~y).sum()); fn += int((~z & y).sum()); selected += int(z.sum()); positives += int(y.sum())
        empty += int(not z.any()); null_false += int(not y.any() and z.any()); score_values.extend(s.tolist())
        pos = np.flatnonzero(y); neg = np.flatnonzero(~y)
        if len(pos):
            order = np.argsort(-s, kind="stable"); top1 += int(y[order[:1]].any()); top5 += int(y[order[:5]].any())
            if len(neg):
                d = float(s[pos].min() - s[neg].max()); strict.append(d); best.append(float(s[pos].max() - s[neg].max())); average.append(float(s[pos].mean() - s[neg].max())); violations.append(d < 0)
            if len(pos) > 1: multi.append(float((z & y).sum() / len(pos)))
    present = sum(bool(np.asarray(r["label"], dtype=bool).any()) for r in rows)
    return {"units": len(rows), "candidate_rows": int(sum(len(r["label"]) for r in rows)), "positive_rows": positives,
            "top1": top1 / max(1, present), "top5": top5 / max(1, present),
            "candidate_precision": tp / max(1, selected), "candidate_recall": tp / max(1, tp + fn),
            "fp_per_frame": fp / max(1, len(rows)), "predictions_per_positive": selected / max(1, positives),
            "hard_violation": float(np.mean(violations)) if violations else None,
            "strict_margin": dist(strict), "best_margin": dist(best), "average_margin": dist(average),
            "multi_positive_recall": float(np.mean(multi)) if multi else None,
            "empty_rate": empty / max(1, len(rows)), "null_false_acceptance": null_false / max(1, len(rows)),
            "score_distribution": dist(score_values), "threshold": float(threshold),
            "null_threshold": None if null_threshold is None else float(null_threshold)}


def fit_threshold(rows, field="score"):
    values = np.unique(np.concatenate([np.asarray(r[field], dtype=float) for r in rows]))
    candidates = values.tolist() + [float(values.min()) - 1e-6, float(values.max()) + 1e-6]
    best = None
    for threshold in candidates:
        tp = fp = fn = 0
        for row in rows:
            s = np.asarray(row[field]); y = np.asarray(row["label"], dtype=bool); z = s >= threshold
            tp += int((z & y).sum()); fp += int((z & ~y).sum()); fn += int((~z & y).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        key = (f1, -fp, -float(threshold))
        if best is None or key > best[0]: best = (key, float(threshold))
    return {"threshold": best[1], "objective": "candidate-level F1 on 16 calibration units; tie lower FP then lower threshold"}


def fit_null(rows, candidate_threshold):
    values = np.unique([float(r["null_logit"]) for r in rows])
    best = None
    for threshold in values.tolist() + [float(values.min()) - 1e-6, float(values.max()) + 1e-6]:
        decisions = []; truths = []
        for row in rows:
            y = np.asarray(row["label"], dtype=bool); decisions.append(bool((np.asarray(row["score"]) >= candidate_threshold).any()) and float(row["null_logit"]) < threshold); truths.append(bool(y.any()))
        tp = sum(a and b for a, b in zip(decisions, truths)); fp = sum(a and not b for a, b in zip(decisions, truths)); fn = sum((not a) and b for a, b in zip(decisions, truths)); f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        key = (f1, -fp, -float(threshold))
        if best is None or key > best[0]: best = (key, float(threshold))
    return {"null_threshold": best[1], "rule": "suppress all candidate rows iff null_logit >= threshold", "objective": "frame-presence F1 on 16 calibration units; ties lower FP then lower threshold"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", required=True); ap.add_argument("--out", required=True); args = ap.parse_args()
    if Path.cwd().resolve() != ROOT: raise RuntimeError(f"wrong cwd {Path.cwd()}")
    out = Path(args.out); out = out if out.is_absolute() else ROOT / out; out = out.resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    if sha256(MANIFEST) != EXPECTED_MANIFEST: raise AssertionError("manifest mismatch")
    data = units40(); model = L64RawPatchSet(hidden=128).cuda(); checkpoint = Path(args.checkpoint).resolve(); payload = torch.load(checkpoint, map_location="cuda:0", weights_only=False); model.load_state_dict(payload["model"], strict=True); model.eval()
    encoder = StreamingOpenAIClip("cuda:0", batch_size=32)
    rows = []; start = time.time()
    for unit in data:
        bank = torch.load(Path(unit["bank_path"]), map_location="cpu", weights_only=False); t = bank["tensors"]; begin, end = int(unit["begin"]), int(unit["end"]); n = end - begin
        boxes = t["box"][begin:end].float(); patches, image = encoder.encode_unit(unit["video"], int(unit["frame_id"]), boxes.tolist()); words, mask = encoder.text_tokens(unit["sentence"]); nums = numeric(t, begin, end)
        if patches.shape != (n, 16, 768) or not torch.isfinite(patches).all() or not torch.isfinite(words).all(): raise AssertionError(f"raw feature contract {unit['unit_key']}")
        with torch.inference_mode(): output = model(patches.cuda(), words.cuda(), mask.cuda(), nums.cuda())
        score = output["relevance_logit"].float().cpu().numpy(); labels = np.asarray(unit["label"], dtype=int)
        if len(score) != n or len(labels) != n: raise AssertionError(f"candidate length {unit['unit_key']}")
        rows.append({"unit_key": unit["unit_key"], "dataset": unit["dataset"], "video": unit["video"], "frame_id": int(unit["frame_id"]), "category": unit["category"], "eval_split": unit["eval_split"], "label": labels.tolist(), "score": score.tolist(), "null_logit": float(output["null_logit"].cpu()), "l29": list(map(float, unit["control"]["l29"])), "m0": list(map(float, unit["control"]["m0"])), "m54": list(map(float, unit["control"]["m54"])), "key_audit": {"candidate_count": n, "candidate_rows_retained": n, "candidate_truncation": False, "ordered": True, "image": str(image), "row_offsets": [begin, end]}})
        del bank, patches, words, mask, nums, output, boxes
    del encoder, model
    cal, val = rows[:16], rows[16:]
    methods = {}
    for name, field in (("l29_teacher", "l29"), ("l53_m0", "m0"), ("l54_continuous", "m54")):
        threshold = L29_THRESHOLD if name == "l29_teacher" else fit_threshold(cal, field)["threshold"]
        methods[name] = {"calibration": metric(cal, threshold, None, field), "validation": metric(val, threshold, None, field), "threshold": {"threshold": threshold, "source": "immutable L62 threshold" if name == "l29_teacher" else "calibration-only"}}
    candidate_threshold = fit_threshold(cal)["threshold"]; null_rule = fit_null(cal, candidate_threshold)
    methods["l64_raw_patch"] = {"candidate_only_calibration": metric(cal, candidate_threshold), "candidate_only_validation": metric(val, candidate_threshold), "threshold": {"threshold": candidate_threshold, "objective": "calibration-only candidate F1"}, "null_rule": null_rule, "final_calibration": metric(cal, candidate_threshold, null_rule["null_threshold"]), "final_validation": metric(val, candidate_threshold, null_rule["null_threshold"])}
    base = methods["l29_teacher"]["validation"]; cur = methods["l64_raw_patch"]["final_validation"]
    checks = {"hard_violation_decrease_ge_0.05": cur["hard_violation"] <= base["hard_violation"] - 0.05, "recall_drop_le_0.01": cur["candidate_recall"] >= base["candidate_recall"] - 0.01, "precision_ge_0.0830188679": cur["candidate_precision"] >= 0.0830188679, "fp_per_frame_le_11.125": cur["fp_per_frame"] <= 11.125, "predictions_per_positive_le_4.069": cur["predictions_per_positive"] <= 4.069, "multi_positive_preserved": cur["multi_positive_recall"] is not None and cur["multi_positive_recall"] >= base["multi_positive_recall"] - 0.03, "null_not_universal": cur["null_false_acceptance"] < 1.0, "complete_keys": all(r["key_audit"]["candidate_count"] == len(r["score"]) == len(r["label"]) for r in rows), "candidate_deletion_false": all(r["key_audit"]["candidate_rows_retained"] == r["key_audit"]["candidate_count"] for r in rows)}
    gate = {"format": "locatemot-l64-raw-patch-semantic-gate-v1", "status": "semantic_gate_pass" if all(checks.values()) else "semantic_gate_fail", "checks": checks, "calibration_units": 16, "validation_units": 24, "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "no_hota_or_trackeval": True}
    out.mkdir(parents=True); (out / "semantic.json").write_text(json.dumps({"format": "locatemot-l64-raw-patch-semantic-v1", "status": "complete", "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()), "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint), "source_control": str(IMMUTABLE), "source_control_sha256": sha256(IMMUTABLE), "methods": methods, "gate": gate, "elapsed_sec": time.time() - start}, indent=2) + "\n"); (out / "gate_decision.json").write_text(json.dumps(gate, indent=2) + "\n"); (out / "score_records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows)); (out / "provenance.json").write_text(json.dumps({"project_root": str(ROOT), "cwd": str(Path.cwd().resolve()), "manifest_sha256": sha256(MANIFEST), "checkpoint_sha256": sha256(checkpoint), "calibration_units": 16, "validation_units": 24, "threshold_fit_on_calibration_only": True, "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "persistent_raw_dense_cache_written": False, "candidate_rows_retained": True, "image_weight": "/home/lwr/.cache/clip/ViT-B-16.pt", "image_weight_sha256": sha256(Path("/home/lwr/.cache/clip/ViT-B-16.pt"))}, indent=2) + "\n"); print(json.dumps({"status": gate["status"], "validation": cur, "checks": checks, "output": str(out)}, indent=2), flush=True)


if __name__ == "__main__": main()
