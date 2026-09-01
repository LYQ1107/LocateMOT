#!/usr/bin/env python3
"""L41 B2: bounded raw relational continuation diagnostic over frozen L29."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
from locatemot.models.l41_raw_relational_identity import L41RawRelationalIdentity
from tools.audit_l29_emission_contract import build_cache as build_l19_sequence_cache
from tools.eval_l27_fast_rmot import frame_groups, make_entries
from tools.eval_l37_expression_track_set import choose_threshold, metrics
from tools.eval_l38_candidate_gate import build_records
from tools.l41_raw_data import RAW_ROOT, WEIGHTS, StreamingClipPatchEncoder, pad_patches, relation_features, sha256
from tools.summarize_l27_fast_rmot import load_caches
from tools.train_l26_crossmodal_adapter import V5 as TEXT_ROOT
from tools.train_l28_track_set_decoder import state_at

L29_CHECKPOINT = ROOT / "outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"
L38_CHECKPOINT = ROOT / "outputs/l38/train/emission_residual_smoke100_retry/checkpoint_l38_emission_residual_step100.pt"
L41_CHECKPOINT = ROOT / "outputs/l41/train/relational_smoke100_retry/checkpoint_l41_raw_relational_identity_step100.pt"
SCORE_ROOT = ROOT / "outputs/l27/fast_rmot_validation_retry"
L28_ROOT = ROOT / "outputs/l28/track_sequence_bank_final"
FAST_MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"


def make_obs(t, row, video):
    numeric = torch.cat((t["geometry"][row].float(), t["motion"][row].float(), t["lifecycle"][row].float(), t["objectness"][row].float().reshape(1)))
    frame = int(t["frame"][row]); return {"row": int(row), "frame": frame, "box": t["box"][row].float().tolist(), "numeric": numeric, "source": int(t["pool_id"][row]), "gt": set(), "image": str(RAW_ROOT / video / f"{frame:06d}.png")}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--cap", type=int, default=100); ap.add_argument("--crop-batch", type=int, default=32); args = ap.parse_args(); assert Path.cwd().resolve() == ROOT
    out = Path(args.out); out = out if out.is_absolute() else ROOT / out; out.parent.mkdir(parents=True, exist_ok=True); started = time.time(); device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    entries = make_entries(); arrays = load_caches(SCORE_ROOT, entries, ("A_C1_S2000",))["A_C1_S2000"]
    text_manifest = json.loads((TEXT_ROOT / "text_manifest.json").read_text())["expressions"]; text_index = {(str(x["video"]), str(x["expression"])): int(x["query_index"]) for x in text_manifest}
    for e in entries: e["text_index"] = text_index[(str(e["video"]), str(e["expression"]))]
    text = torch.load(TEXT_ROOT / "text_tokens.pt", map_location="cpu", weights_only=False); hidden = text["token_hidden"].float(); mask = text["attention_mask"].bool(); del text
    seq_cache = {}
    for v in sorted({str(e["video"]) for e in entries}):
        p = L28_ROOT / f"{v}.pt"; seq_cache[v] = torch.load(p, map_location="cpu", weights_only=False) if p.exists() else build_l19_sequence_cache(v)
    l29 = L29FrameMembershipSetDecoder().to(device); l29.load_state_dict(torch.load(L29_CHECKPOINT, map_location=device, weights_only=False)["model"]); l29.eval()
    l38 = torch.load(L38_CHECKPOINT, map_location=device, weights_only=False); from locatemot.models.l38_bounded_emission_residual import L38BoundedEmissionResidual; residual = L38BoundedEmissionResidual(hidden=96, history=8).to(device); residual.load_state_dict(l38["model"]); residual.eval()
    # Reuse L38's immutable frame-unit construction solely to obtain the frozen L29 rows.
    records, counts, _ = build_records(entries, arrays, seq_cache, hidden, mask, l29, residual, device, args.cap)
    baseline = choose_threshold(records["calibration"]["l29"]); threshold = float(baseline["threshold"])
    screening = records["screening"]["l29"]; calibration = records["calibration"]["l29"]
    bank_cache = {}; track_rows = {}
    needed_keys = set()
    for rec in calibration + screening:
        for track in rec["track_id"].tolist(): needed_keys.add((str(rec["video"]), int(rec["frame"]), int(track)))
    pseudo_left = []; pseudo_right = []; keys = []
    for video in sorted({x[0] for x in needed_keys}):
        bank = torch.load(ROOT / "outputs/l19/dual_banks_features/kitti" / f"{video}.pt", map_location="cpu", weights_only=False)["tensors"]; bank_cache[video] = bank; by_track = defaultdict(list)
        for row, track in enumerate(bank["track_id"].tolist()): by_track[int(track)].append(row)
        for track in by_track: by_track[track].sort(key=lambda r: int(bank["frame"][r]))
        bank_cache[video] = (bank, by_track)
    for video, frame, track in sorted(needed_keys):
        bank, by_track = bank_cache[video]; rows = by_track.get(track, []); current = [r for r in rows if int(bank["frame"][r]) == frame]; prior = [r for r in rows if int(bank["frame"][r]) < frame]
        if not current or not prior: continue
        a = make_obs(bank, prior[-1], video); b = make_obs(bank, current[-1], video); pseudo_left.append({"obs": [a], "frames": {a["frame"]}, "source": a["source"]}); pseudo_right.append({"obs": [b], "frames": {b["frame"]}, "source": b["source"]}); keys.append((video, frame, track))
    pseudo = pseudo_left + pseudo_right; encoder = StreamingClipPatchEncoder(device=device, weights=WEIGHTS, batch_size=args.crop_batch); patches = encoder.encode(pseudo, range(len(pseudo))); patch_map = {i: x for i, x in enumerate(patches)}; del patches, encoder
    model = L41RawRelationalIdentity(hidden=96, history=8).to(device); model.load_state_dict(torch.load(L41_CHECKPOINT, map_location=device, weights_only=False)["model"]); model.eval(); scores = {k: 0.0 for k in keys}
    with torch.inference_mode():
        for start_id in range(0, len(keys), 128):
            chunk_keys = keys[start_id:start_id + 128]; ai = list(range(start_id, start_id + len(chunk_keys))); bi = [len(pseudo_left) + x for x in ai]; la, lm = pad_patches(pseudo, ai, patch_map, device); rb, rm = pad_patches(pseudo, bi, patch_map, device); rel = torch.stack([relation_features(pseudo[x], pseudo[len(pseudo_left) + x]) for x in ai]).to(device); out_score = model(la, rb, rel, lm, rm)["logit"].float().cpu().tolist(); scores.update({k: float(v) for k, v in zip(chunk_keys, out_score)})
    def add_cont(records_in):
        result = []
        for r in records_in:
            z = copy.deepcopy(r); vals = []
            for track in r["track_id"].tolist(): vals.append(scores.get((str(r["video"]), int(r["frame"]), int(track)), 0.0))
            z["score"] = r["score"].astype(np.float32) + 0.05 * np.tanh(np.asarray(vals, np.float32)); result.append(z)
        return result
    fusion = {"calibration": add_cont(calibration), "screening": add_cont(screening)}; metrics_base = metrics(screening, threshold, None); metrics_fusion = metrics(fusion["screening"], threshold, None)
    relation_vals = np.asarray(list(scores.values()), float); payload = {"format": "locatemot-l41-restricted-emission-v1", "stage": "L41-B2", "checkpoint": str(L41_CHECKPOINT.resolve()), "checkpoint_sha256": sha256(L41_CHECKPOINT), "teacher_checkpoint": str(L29_CHECKPOINT.resolve()), "teacher_checkpoint_sha256": sha256(L29_CHECKPOINT), "manifest": str(FAST_MANIFEST.resolve()), "manifest_sha256": sha256(FAST_MANIFEST), "cap": args.cap, "counts": counts, "calibration_threshold": baseline, "threshold_reused_without_change": True, "continuation_bound": 0.05, "continuation_formula": "frozen_L29_current_membership + 0.05*tanh(L41_pair_logit)", "candidate_continuation_rows": len(scores), "continuation_stats": {"mean": float(relation_vals.mean()) if len(relation_vals) else None, "min": float(relation_vals.min()) if len(relation_vals) else None, "max": float(relation_vals.max()) if len(relation_vals) else None}, "screening_gt_used_for_selection": False, "semantic_inputs_excluded": ["expression", "source_id", "pool_id", "group_id", "state_key"], "baseline_l29": metrics_base, "bounded_fusion": metrics_fusion, "gate": {"top1_not_drop_gt_0.02": bool(metrics_fusion["top1_frame_recall"] >= metrics_base["top1_frame_recall"] - 0.02), "recall_not_drop_gt_0.03": bool(metrics_fusion["recall"] >= metrics_base["recall"] - 0.03), "hard_violation_improved": bool(metrics_fusion["hard_violation_rate"] < metrics_base["hard_violation_rate"]), "precision_not_drop_gt_0.02": bool(metrics_fusion["precision"] >= metrics_base["precision"] - 0.02), "fp_not_increase_gt_5pct": bool(metrics_fusion["false_positive_candidates_per_frame"] <= metrics_base["false_positive_candidates_per_frame"] * 1.05)}, "decision": "pass" if metrics_fusion["top1_frame_recall"] >= metrics_base["top1_frame_recall"] - 0.02 and metrics_fusion["recall"] >= metrics_base["recall"] - 0.03 and metrics_fusion["hard_violation_rate"] < metrics_base["hard_violation_rate"] and metrics_fusion["precision"] >= metrics_base["precision"] - 0.02 and metrics_fusion["false_positive_candidates_per_frame"] <= metrics_base["false_positive_candidates_per_frame"] * 1.05 else "fail", "screening_trackeval_run": False, "elapsed_sec": time.time() - started}
    out.write_text(json.dumps(payload, indent=2) + "\n"); print(json.dumps({"out": str(out), "decision": payload["decision"], "gate": payload["gate"], "baseline": metrics_base, "fusion": metrics_fusion}, indent=2), flush=True)


if __name__ == "__main__": main()
