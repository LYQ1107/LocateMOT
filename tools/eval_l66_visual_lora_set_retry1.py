#!/usr/bin/env python3
"""L66 continuation: fixed 16-calibration/24-validation semantic gate."""
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
L29_THRESHOLD = -1.030576229095459
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
import sys
sys.path.insert(0, str(ROOT))
from locatemot.models.l66_visual_lora_set import L66VisualLoraSet, attach_visual_lora
from tools.l66_visual_lora_common import (CLIP_WEIGHTS, EXPECTED_CLIP, L65_CHECKPOINT,
    StreamingClipLora, load_unit_features, sha256)


def fixed_units():
    lookup = {}
    for fn in ("calibration_units.jsonl", "validation_units.jsonl"):
        for line in (DATA / fn).read_text().splitlines():
            if line.strip():
                u = json.loads(line); lookup[u["unit_key"]] = u
    records = [json.loads(x) for x in IMMUTABLE.read_text().splitlines() if x.strip()]
    if len(records) != 40 or len({r["unit_key"] for r in records}) != 40:
        raise AssertionError("immutable 40-unit contract")
    result = []
    for i, record in enumerate(records):
        if record["unit_key"] not in lookup: raise AssertionError(f"missing unit {record['unit_key']}")
        u = dict(lookup[record["unit_key"]]); u["eval_split"] = "calibration" if i < 16 else "validation"; u["record"] = record; result.append(u)
    return result


def _stats(values):
    a = np.asarray(values, dtype=float)
    return {"count": int(a.size), "mean": float(a.mean()) if a.size else None, "std": float(a.std()) if a.size else None, "min": float(a.min()) if a.size else None, "max": float(a.max()) if a.size else None}


def metric(rows, field, threshold, null_field=None, null_threshold=None):
    tp = fp = fn = selected = positives = top1 = top5 = empty = null_false = 0
    strict, best, average, violations, multi, values = [], [], [], [], [], []
    for row in rows:
        scores = np.asarray(row[field], dtype=float); labels = np.asarray(row["label"], dtype=bool)
        if scores.size != labels.size or not np.isfinite(scores).all(): raise AssertionError(f"finite/length {row['unit_key']} {field}")
        suppressed = null_threshold is not None and float(row[null_field]) >= float(null_threshold)
        selected_mask = (scores >= threshold) & (not suppressed)
        tp += int((selected_mask & labels).sum()); fp += int((selected_mask & ~labels).sum()); fn += int((~selected_mask & labels).sum()); selected += int(selected_mask.sum()); positives += int(labels.sum()); empty += int(not selected_mask.any()); null_false += int(not labels.any() and selected_mask.any()); values.extend(scores.tolist())
        p = np.flatnonzero(labels); n = np.flatnonzero(~labels)
        if p.size:
            order = np.argsort(-scores, kind="stable"); top1 += int(labels[order[:1]].any()); top5 += int(labels[order[:5]].any())
            if n.size:
                d = float(scores[p].min() - scores[n].max()); strict.append(d); best.append(float(scores[p].max() - scores[n].max())); average.append(float(scores[p].mean() - scores[n].max())); violations.append(d < 0)
            if p.size > 1: multi.append(float((selected_mask & labels).sum() / p.size))
    present = sum(bool(np.asarray(row["label"], dtype=bool).any()) for row in rows)
    return {"units": len(rows), "candidate_rows": int(sum(len(row["label"]) for row in rows)), "positive_rows": positives, "top1": top1 / max(1, present), "top5": top5 / max(1, present), "candidate_precision": tp / max(1, selected), "candidate_recall": tp / max(1, tp + fn), "fp_per_frame": fp / max(1, len(rows)), "predictions_per_positive": selected / max(1, positives), "hard_violation": float(np.mean(violations)) if violations else None, "strict_margin": _stats(strict), "best_margin": _stats(best), "average_margin": _stats(average), "multi_positive_recall": float(np.mean(multi)) if multi else None, "empty_rate": empty / max(1, len(rows)), "null_false_acceptance": null_false / max(1, len(rows)), "score_mean": float(np.mean(values)) if values else None, "score_std": float(np.std(values)) if values else None, "threshold": float(threshold), "null_threshold": None if null_threshold is None else float(null_threshold)}


def fit_threshold(rows, field):
    values = np.unique(np.concatenate([np.asarray(row[field], dtype=float) for row in rows])); best = None
    for threshold in values.tolist() + [float(values.min()) - 1e-6, float(values.max()) + 1e-6]:
        tp = fp = fn = 0
        for row in rows:
            scores = np.asarray(row[field]); labels = np.asarray(row["label"], dtype=bool); selected = scores >= threshold; tp += int((selected & labels).sum()); fp += int((selected & ~labels).sum()); fn += int((~selected & labels).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn); key = (f1, -fp, -float(threshold))
        if best is None or key > best[0]: best = (key, float(threshold))
    return best[1]


def fit_null(rows, field, threshold, null_field):
    values = np.unique([float(row[null_field]) for row in rows]); best = None
    for null_threshold in values.tolist() + [float(values.min()) - 1e-6, float(values.max()) + 1e-6]:
        predicted, truth = [], []
        for row in rows:
            predicted.append(bool((np.asarray(row[field]) >= threshold).any()) and float(row[null_field]) < null_threshold); truth.append(bool(np.asarray(row["label"], dtype=bool).any()))
        tp = sum(a and b for a, b in zip(predicted, truth)); fp = sum(a and not b for a, b in zip(predicted, truth)); fn = sum((not a) and b for a, b in zip(predicted, truth)); f1 = 2 * tp / max(1, 2 * tp + fp + fn); key = (f1, -fp, -float(null_threshold))
        if best is None or key > best[0]: best = (key, float(null_threshold))
    return best[1]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True, type=Path); parser.add_argument("--out", required=True, type=Path); args = parser.parse_args()
    if Path.cwd().resolve() != ROOT: raise RuntimeError(f"wrong cwd {Path.cwd()}")
    if sha256(MANIFEST) != EXPECTED_MANIFEST: raise AssertionError("manifest SHA mismatch")
    if sha256(CLIP_WEIGHTS) != EXPECTED_CLIP: raise AssertionError("CLIP SHA mismatch")
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    units = fixed_units(); checkpoint = args.checkpoint.resolve(); package = torch.load(checkpoint, map_location="cpu", weights_only=False); device = torch.device("cuda:0")
    runtime = StreamingClipLora(device, crop_batch=4); _, wrapper = attach_visual_lora(runtime.model, rank=8, alpha=16.0, dropout=0.0)
    missing, unexpected = runtime.model.load_state_dict(package["lora"], strict=False)
    if unexpected or any("lora_A" in key or "lora_B" in key for key in missing): raise AssertionError(f"LoRA reload mismatch missing={missing} unexpected={unexpected}")
    model = L66VisualLoraSet(hidden=128).to(device); model.load_state_dict(package["head"], strict=True); model.eval()
    control = L66VisualLoraSet(hidden=128).to(device); control.load_state_dict(torch.load(L65_CHECKPOINT, map_location="cpu", weights_only=False)["model"], strict=True); control.eval()
    rows = []; start = time.time()
    with torch.no_grad():
        for unit in units:
            item = load_unit_features(unit, runtime, labels=True); lora_out = model(item["patches"].to(device), item["words"].to(device), item["mask"].to(device), item["numeric"].to(device)); lora_scores = [float(x) for x in lora_out["relevance_logit"].cpu()]; lora_null = float(lora_out["null_logit"].cpu()); saved_b = wrapper.lora_B.detach().clone(); wrapper.lora_B.zero_(); base_item = load_unit_features(unit, runtime, labels=True); base_out = control(base_item["patches"].to(device), base_item["words"].to(device), base_item["mask"].to(device), base_item["numeric"].to(device)); wrapper.lora_B.copy_(saved_b); labels = item["target"].tolist();
            if base_item["target"].tolist() != labels or len(lora_scores) != int(unit["candidate_count"]) or len(labels) != len(unit["record"]["l29"]): raise AssertionError(f"candidate/label order {unit['unit_key']}")
            rows.append({"unit_key":unit["unit_key"],"dataset":unit["dataset"],"video":unit["video"],"frame_id":int(unit["frame_id"]),"category":unit["category"],"eval_split":unit["eval_split"],"label":labels,"l66_lora":lora_scores,"l65_no_lora":[float(x) for x in base_out["relevance_logit"].cpu()],"l66_null_logit":lora_null,"l65_null_logit":float(base_out["null_logit"].cpu()),"l29":unit["record"]["l29"],"key_audit":{"candidate_count":len(labels),"candidate_rows_retained":len(labels),"candidate_truncation":False,"ordered":True,"duplicate_candidate_index_legal":True}})
            del item, base_item, lora_out, base_out, saved_b
    cal, val = rows[:16], rows[16:]; methods = {}
    for name, null_field in (("l65_no_lora", "l65_null_logit"), ("l66_lora", "l66_null_logit")):
        threshold = fit_threshold(cal, name); null_threshold = fit_null(cal, name, threshold, null_field)
        methods[name] = {"candidate_only_calibration":metric(cal,name,threshold),"candidate_only_validation":metric(val,name,threshold),"final_calibration":metric(cal,name,threshold,null_field,null_threshold),"final_validation":metric(val,name,threshold,null_field,null_threshold),"threshold":{"threshold":threshold,"fit":"16 calibration units only"},"null_rule":{"null_threshold":null_threshold,"fit":"16 calibration units only","null_field":null_field,"rule":"suppress all candidates when null logit >= threshold"}}
    l29 = [{"label":row["label"],"l29":row["l29"]} for row in rows]; methods["l29_teacher"] = {"calibration":metric(l29[:16],"l29",L29_THRESHOLD),"validation":metric(l29[16:],"l29",L29_THRESHOLD),"threshold":{"threshold":L29_THRESHOLD,"source":"accepted immutable L62 records"}}
    base = methods["l29_teacher"]["validation"]; gates = {}
    for name, field, null_field, use_null in (("l66_lora_candidate_only","l66_lora","l66_null_logit",False),("l66_lora_final_null","l66_lora","l66_null_logit",True)):
        method = methods[field]; current = metric(val, field, method["threshold"]["threshold"], null_field if use_null else None, method["null_rule"]["null_threshold"] if use_null else None)
        gates[name] = {"hard_violation_decrease_ge_0.05":current["hard_violation"] is not None and current["hard_violation"] <= base["hard_violation"] - .05,"recall_floor":current["candidate_recall"] >= .7233333,"precision_floor":current["candidate_precision"] >= .0830188679,"fp_frame_floor":current["fp_per_frame"] <= 11.125,"predictions_per_positive_floor":current["predictions_per_positive"] <= 4.069,"multi_positive_floor":current["multi_positive_recall"] is not None and current["multi_positive_recall"] >= .7894444,"null_not_universal":current["null_false_acceptance"] < 1.0,"complete_keys":all(row["key_audit"]["candidate_count"] == len(row[field]) == len(row["label"]) for row in rows),"candidate_deletion_false":True}
    usable = [name for name, checks in gates.items() if all(checks.values())]; gate = {"format":"locatemot-l66-visual-lora-semantic-gate-v1","status":"semantic_gate_pass" if usable else "semantic_gate_fail","usable_methods":usable,"checks_by_method":gates,"calibration_units":16,"validation_units":24,"screening_gt_used":False,"official_test_labels_read":False,"ordinary_mot_ovmot_touched":False,"no_hota_or_trackeval":True}
    out.mkdir(parents=True); (out/"semantic.json").write_text(json.dumps({"format":"locatemot-l66-visual-lora-semantic-v1","status":"complete","project_root":str(ROOT),"cwd":str(Path.cwd().resolve()),"checkpoint":str(checkpoint),"checkpoint_sha256":sha256(checkpoint),"methods":methods,"gate":gate,"elapsed_sec":time.time()-start},indent=2)+"\n"); (out/"gate_decision.json").write_text(json.dumps(gate,indent=2)+"\n"); (out/"score_records.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows)); (out/"provenance.json").write_text(json.dumps({"project_root":str(ROOT),"cwd":str(Path.cwd().resolve()),"manifest_sha256":sha256(MANIFEST),"immutable_l29_source":str(IMMUTABLE),"immutable_l29_source_sha256":sha256(IMMUTABLE),"checkpoint":str(checkpoint),"checkpoint_sha256":sha256(checkpoint),"clip_weights":str(CLIP_WEIGHTS),"clip_weights_sha256":sha256(CLIP_WEIGHTS),"calibration_units":16,"validation_units":24,"threshold_fit_calibration_only":True,"null_fit_calibration_only":True,"candidate_rows_retained":True,"candidate_truncation":False,"persistent_raw_dense_cache_written":False,"screening_gt_used":False,"official_test_labels_read":False,"ordinary_mot_ovmot_touched":False,"token_span_alignment":"UNALIGNED","static_motion_mask":"UNALIGNED"},indent=2)+"\n"); print(json.dumps({"status":gate["status"],"usable_methods":usable,"output":str(out),"l66_final_validation":methods["l66_lora"]["final_validation"]},indent=2),flush=True)


if __name__ == "__main__": main()
