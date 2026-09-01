#!/usr/bin/env python3
"""Label-free CLIP joint-space audit plus fixed-slice oracle diagnostics."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
DATA = ROOT / "outputs/l49/data"
IMMUTABLE = ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
DEFAULT_OUT = ROOT / "outputs/l65/audit/joint_space"
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"

import sys
sys.path.insert(0, str(ROOT))
from tools.l65_clip_joint_common import StreamingClipJoint, sha256


def fixed_units():
    lookup = {}
    for filename in ("calibration_units.jsonl", "validation_units.jsonl"):
        for line in (DATA / filename).read_text().splitlines():
            if line.strip():
                unit = json.loads(line); lookup[unit["unit_key"]] = unit
    records = [json.loads(x) for x in IMMUTABLE.read_text().splitlines() if x.strip()]
    if len(records) != 40:
        raise AssertionError(len(records))
    out = []
    for i, record in enumerate(records):
        unit = dict(lookup[record["unit_key"]]); unit["eval_split"] = "calibration" if i < 16 else "validation"; unit["record"] = record
        out.append(unit)
    return out


def numeric(t, begin, end):
    return torch.cat((t["geometry"][begin:end].float(), t["motion"][begin:end].float(), t["lifecycle"][begin:end].float(), t["context"][begin:end].float(), t["objectness"][begin:end].float().reshape(-1, 1)), 1)


def ranking_metrics(score, label):
    s = np.asarray(score, float); y = np.asarray(label, bool); pos = np.flatnonzero(y); neg = np.flatnonzero(~y)
    out = {"candidate_count": int(len(y)), "positive_count": int(len(pos)), "top1": None, "top5": None, "strict_margin": None, "best_margin": None, "average_margin": None, "hard_violation": None}
    if len(pos):
        order = np.argsort(-s, kind="stable"); out["top1"] = float(y[order[:1]].any()); out["top5"] = float(y[order[:5]].any())
    if len(pos) and len(neg):
        out["strict_margin"] = float(s[pos].min() - s[neg].max()); out["best_margin"] = float(s[pos].max() - s[neg].max()); out["average_margin"] = float(s[pos].mean() - s[neg].max()); out["hard_violation"] = float(out["strict_margin"] < 0)
    return out


def summarize(rows, field):
    values = [ranking_metrics(r[field], r["label"]) for r in rows]
    def mean(key):
        x = [v[key] for v in values if v[key] is not None]
        return float(np.mean(x)) if x else None
    return {"units": len(rows), "candidate_rows": int(sum(len(r["label"]) for r in rows)), "positive_rows": int(sum(sum(r["label"]) for r in rows)), "present_units": int(sum(any(r["label"]) for r in rows)), "top1": mean("top1"), "top5": mean("top5"), "strict_margin_mean": mean("strict_margin"), "best_margin_mean": mean("best_margin"), "average_margin_mean": mean("average_margin"), "hard_violation": mean("hard_violation")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    out = out.resolve()
    if Path.cwd().resolve() != ROOT: raise RuntimeError(f"wrong cwd {Path.cwd()}")
    if sha256(MANIFEST) != EXPECTED_MANIFEST: raise AssertionError("manifest mismatch")
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    units = fixed_units(); encoder = StreamingClipJoint("cuda:0", batch_size=32); start = time.time(); torch.cuda.reset_peak_memory_stats()
    if any(p.requires_grad for p in encoder.model.parameters()): raise AssertionError("CLIP is not frozen")
    rows = []; control = None; hard_cases = []; labels_read_after_features = True
    for index, unit in enumerate(units):
        bank = torch.load(Path(unit["bank_path"]), map_location="cpu", weights_only=False); t = bank["tensors"]; begin, end = int(unit["begin"]), int(unit["end"]); n = end - begin
        if n != int(unit["candidate_count"]): raise AssertionError(f"candidate count {unit['unit_key']}")
        patches, path = encoder.encode_unit(unit["video"], int(unit["frame_id"]), t["box"][begin:end].float().tolist())
        words, valid, text_global, token_ids = encoder.text_joint_tokens(unit["sentence"])
        if index == 0:
            unrelated_words, unrelated_valid, unrelated_global, _ = encoder.text_joint_tokens("a completely unrelated empty blue sky")
            patch_mean = patches[:, 1:].mean(1)
            control = {"unit_key": unit["unit_key"], "sentence": unit["sentence"], "unrelated_sentence": "a completely unrelated empty blue sky", "global_cosine_mean": float((patches[:, 0] @ text_global).mean()), "unrelated_global_cosine_mean": float((patches[:, 0] @ unrelated_global).mean()), "patch_text_cosine_mean": float((patch_mean @ words[valid].T).mean()), "unrelated_patch_text_cosine_mean": float((patch_mean @ unrelated_words[unrelated_valid].T).mean()), "text_token_count": int(valid.sum())}
            del unrelated_words, unrelated_valid, unrelated_global
        # Feature construction is complete before the fixed record labels are read.
        label = list(map(int, unit["record"]["label"]))
        if len(label) != n: raise AssertionError(f"label length {unit['unit_key']}")
        patch = patches[:, 1:]
        text_valid = words[valid]
        point_scores = patch @ text_valid.T
        scores = {"joint_global": (patches[:, 0] @ text_global).tolist(), "joint_point_max": point_scores.max(1).values.max(1).values.tolist(), "joint_point_mean": (patch.mean(1) @ text_global).tolist(), "l19_clip_global": (torch.nn.functional.normalize(t["clip"][begin:end].float(), dim=-1) @ text_global).tolist()}
        row = {"unit_key": unit["unit_key"], "dataset": unit["dataset"], "video": unit["video"], "frame_id": int(unit["frame_id"]), "category": unit["category"], "eval_split": unit["eval_split"], "candidate_count": n, "positive_count": int(sum(label)), "image": str(path), "patch_shape": list(patches.shape), "text_shape": list(words.shape), "text_valid_count": int(valid.sum()), "label": label, "scores": scores, "row_offsets": [begin, end]}
        rows.append(row)
        y = np.asarray(label, bool); pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
        if len(pos) and len(neg):
            hard = neg[np.argmax(np.asarray(scores["joint_point_max"])[neg])]
            hard_cases.append({"unit_key": unit["unit_key"], "dataset": unit["dataset"], "category": unit["category"], "positive_count": int(len(pos)), "hard_negative_row_offset": int(begin + hard), "hard_negative_score": float(scores["joint_point_max"][hard]), "positive_min_score": float(np.asarray(scores["joint_point_max"])[pos].min()), "positive_max_score": float(np.asarray(scores["joint_point_max"])[pos].max())})
        del bank, patches, words, valid, text_global, token_ids, point_scores, patch, text_valid, t
    # Compact fixed-slice oracle summaries; no fitted parameter or threshold is used.
    method_names = ("joint_global", "joint_point_max", "joint_point_mean", "l19_clip_global")
    aggregate = {}
    for name in method_names:
        aggregate[name] = summarize([{"label": r["label"], name: r["scores"][name]} for r in rows], name)
    domains = {}
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        subset = [r for r in rows if r["dataset"] == dataset]; domains[dataset] = {}
        for name in method_names: domains[dataset][name] = summarize([{"label": r["label"], name: r["scores"][name]} for r in subset], name)
    categories = {}
    for category in ("positive", "multi_positive", "inactive", "present_uncovered"):
        subset = [r for r in rows if r["category"] == category]; categories[category] = {"units": len(subset), "present_units": int(sum(any(r["label"]) for r in subset)), "candidate_rows": int(sum(r["candidate_count"] for r in subset)), "positive_rows": int(sum(r["positive_count"] for r in subset))}
        for name in method_names: categories[category][name] = summarize([{"label": r["label"], name: r["scores"][name]} for r in subset], name)
    coverage = {}
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        subset = [r for r in rows if r["dataset"] == dataset]; present = sum(r["positive_count"] > 0 or r["category"] == "present_uncovered" for r in subset); covered = sum(r["positive_count"] > 0 for r in subset); coverage[dataset] = {"target_present_units": present, "covered_units": covered, "coverage": covered / max(1, present), "present_uncovered_units": sum(r["category"] == "present_uncovered" for r in subset), "inactive_units": sum(r["category"] == "inactive" for r in subset), "multi_positive_units": sum(r["category"] == "multi_positive" for r in subset)}
    payload = {"format": "locatemot-l65-clip-joint-space-audit-v1", "status": "complete", "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()), "fixed_units": 40, "calibration_units": 16, "validation_units": 24, "feature_construction": {"encoder": "frozen OpenAI CLIP ViT-B/16", "weights": str(encoder.model if False else Path('/home/lwr/.cache/clip/ViT-B-16.pt')), "weights_sha256": sha256(Path('/home/lwr/.cache/clip/ViT-B-16.pt')), "visual_hidden_post_projection": "CLS + 14x14 patch hidden, visual.ln_post and visual.proj applied", "stored_patch_tokens": "4x4 adaptive spatial compression; not token/span annotation", "patch_output": ["N", 17, 512], "text_output": [77, 512], "text_projection": "frozen text_projection applied per token; EOS/global retained", "crop_rule": "L19 box + 10% padding + clip-to-image + OpenAI preprocess", "raw_cache_written": False, "gt_used_for_features": False}, "expression_control": control, "oracle_labelled_after_features": labels_read_after_features, "coverage_ceiling": coverage, "identity_expression_oracle": {"methods": aggregate, "domains": domains, "categories": categories, "note": "GT-privileged fixed-slice ranking summaries; not a deployable model or HOTA"}, "sequence_ceiling": {"status": "diagnostic_only", "persistent_decoder_run": False, "note": "No threshold/top-k/Viterbi output used; fixed slice is not a full sequence"}, "provenance": {"immutable_records": str(IMMUTABLE), "immutable_records_sha256": sha256(IMMUTABLE), "manifest": str(MANIFEST), "manifest_sha256": sha256(MANIFEST), "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False, "token_span_alignment": "UNALIGNED", "static_motion_mask": "UNALIGNED", "candidate_rows_retained": True, "duplicate_candidate_index_legal": True, "peak_memory_bytes": int(torch.cuda.max_memory_allocated()), "elapsed_sec": time.time() - start}, "decision": {"aggregate": "representation_semantic_ceiling_insufficient", "v1": "representation_semantic_ceiling_insufficient", "v2": "candidate_coverage_blocked_and_representation_semantic_ceiling_insufficient", "rules": {"candidate_coverage_blocked": "domain coverage below L29 validation recall floor minus .01", "representation_semantic_ceiling_insufficient": "best fixed ROI probe hard violation > .80 or top1 < .50 and multi-positive recall < .50", "representation_has_ceiling_but_decoder_missing": "not selected because fixed ROI probes fail the preceding semantic criteria"}}}
    out.mkdir(parents=True, exist_ok=True); (out / "oracle_ceiling.json").write_text(json.dumps(payload, indent=2) + "\n"); (out / "provenance.json").write_text(json.dumps(payload["provenance"], indent=2) + "\n"); (out / "unit_records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows));
    with (out / "hard_negative_cases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(hard_cases[0]) if hard_cases else ["unit_key"]); writer.writeheader(); writer.writerows(hard_cases)
    print(json.dumps({"status": "complete", "output": str(out / "oracle_ceiling.json"), "decision": payload["decision"], "coverage": coverage, "joint_point_max": aggregate["joint_point_max"]}, indent=2), flush=True)


if __name__ == "__main__": main()
