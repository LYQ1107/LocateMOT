#!/usr/bin/env python3
"""L38 A1: compare frozen L29 emission and bounded residual on 100 units."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
from locatemot.models.l38_bounded_emission_residual import L38BoundedEmissionResidual
from tools.audit_l29_emission_contract import build_cache as build_l19_sequence_cache
from tools.eval_l27_fast_rmot import frame_groups, make_entries
from tools.summarize_l27_fast_rmot import load_caches
from tools.train_l26_crossmodal_adapter import V5 as TEXT_ROOT
from tools.train_l28_track_set_decoder import state_at
from tools.eval_l37_expression_track_set import choose_threshold, metrics

SCORE_ROOT = ROOT / "outputs/l27/fast_rmot_validation_retry"
L28_ROOT = ROOT / "outputs/l28/track_sequence_bank_final"
L29_CHECKPOINT = ROOT / "outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"
L38_CHECKPOINT = ROOT / "outputs/l38/train/emission_residual_smoke100_retry/checkpoint_l38_emission_residual_step100.pt"
FAST_MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def valid_indices(cache, cutoff):
    ptr = cache["track_ptr"].numpy(); frames = cache["obs_frame"].numpy()
    return [i for i in range(len(ptr) - 1)
            if np.any(frames[int(ptr[i]):int(ptr[i + 1])] <= int(cutoff))]


def build_records(entries, arrays, seq_cache, text_hidden, text_mask, l29, l38,
                  device, screen_cap):
    available = []
    for e in entries:
        if e["split"] != "screening": continue
        d = arrays[(e["video"], e["expression"])]
        available.extend((str(e["video"]), str(e["expression"]), int(f))
                         for f, _ in frame_groups(d))
    available.sort()
    selected = {available[i] for i in np.linspace(0, len(available) - 1,
                                                   min(screen_cap, len(available)), dtype=int)}
    needed = []
    for e in entries:
        d = arrays[(e["video"], e["expression"])]
        for frame, _ in frame_groups(d):
            unit = (str(e["video"]), str(e["expression"]), int(frame))
            if e["split"] == "calibration" or unit in selected:
                needed.append((unit, e))
    by_frame = defaultdict(list)
    for unit, e in needed: by_frame[(unit[0], unit[2])].append((unit, e))
    records = {"calibration": {"l29": [], "l38": [], "control": []},
               "screening": {"l29": [], "l38": [], "control": []}}
    diag = {"screening_residual": [], "screening_teacher_error": [],
            "screening_continuation_logit": []}
    for (video, frame), pairs in sorted(by_frame.items()):
        cache = seq_cache[video]
        obs, om, ot, _, _ = state_at(cache, frame, history=8)
        valid = valid_indices(cache, frame)
        track_ids = cache["track_ids"][torch.as_tensor(valid)].numpy().astype(np.int64)
        with torch.inference_mode():
            encoded = l29.encode_observations(obs.to(device), om.to(device), ot.to(device))
            for unit, e in pairs:
                qh = text_hidden[int(e["text_index"])].to(device)
                qm = text_mask[int(e["text_index"])].to(device)
                teacher_out = l29.forward_encoded(encoded, encoded[1], qh, qm)
                teacher = teacher_out["current_membership_logits"].float()
                out = l38(obs.to(device), om.to(device), ot.to(device), qh, qm, teacher)
                residual = out["residual_score"].float()
                final = out["final_score"].float()
                diag["screening_residual"].extend(residual.cpu().numpy().tolist())
                diag["screening_teacher_error"].append(float((out["teacher_score"] - teacher).abs().max().cpu()))
                diag["screening_continuation_logit"].extend(out["continuation_logit"].float().cpu().numpy().tolist())
                map_teacher = {int(t): float(v) for t, v in zip(track_ids, teacher.cpu().numpy())}
                map_final = {int(t): float(v) for t, v in zip(track_ids, final.cpu().numpy())}
                d = arrays[(e["video"], e["expression"])]
                idx = np.flatnonzero(d["frame"] == frame); tracks = d["track_id"][idx].astype(np.int64)
                base = {"video": video, "expression": str(e["expression"]),
                        "query_index": int(e["query_index"]), "frame": int(frame),
                        "track_id": tracks, "label": d["label"][idx].astype(bool),
                        "source": d["source"][idx].astype(np.int8)}
                teacher_rows = np.asarray([map_teacher.get(int(t), -20.) for t in tracks], np.float32)
                final_rows = np.asarray([map_final.get(int(t), -20.) for t in tracks], np.float32)
                kind = "calibration" if e["split"] == "calibration" else "screening"
                records[kind]["l29"].append({**base, "score": teacher_rows, "null_logit": None})
                records[kind]["l38"].append({**base, "score": final_rows, "null_logit": None})
                records[kind]["control"].append({**base, "score": teacher_rows, "null_logit": None})
    return records, {"screening_units_selected": len(selected),
                     "screening_units_available": len(available),
                     "calibration_units": len(records["calibration"]["l29"])}, diag


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--checkpoint", default=str(L38_CHECKPOINT))
    ap.add_argument("--out", default="outputs/l38/eval/candidate_gate_100.json")
    ap.add_argument("--cap", type=int, default=100); ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(); assert Path.cwd().resolve() == ROOT
    entries = make_entries(); arrays = load_caches(SCORE_ROOT, entries, ("A_C1_S2000",))["A_C1_S2000"]
    text_manifest = json.loads((TEXT_ROOT / "text_manifest.json").read_text())["expressions"]
    text_index = {(str(x["video"]), str(x["expression"])): int(x["query_index"]) for x in text_manifest}
    for e in entries: e["text_index"] = text_index[(str(e["video"]), str(e["expression"]))]
    text = torch.load(TEXT_ROOT / "text_tokens.pt", map_location="cpu", weights_only=False)
    hidden = text["token_hidden"].float(); mask = text["attention_mask"].bool(); del text
    seq_cache = {}
    for v in sorted({str(e["video"]) for e in entries}):
        path = L28_ROOT / f"{v}.pt"
        seq_cache[v] = torch.load(path, map_location="cpu", weights_only=False) if path.exists() else build_l19_sequence_cache(v)
    device = torch.device(args.device)
    l29 = L29FrameMembershipSetDecoder().to(device)
    l29.load_state_dict(torch.load(L29_CHECKPOINT, map_location=device, weights_only=False)["model"]); l29.eval()
    l38 = L38BoundedEmissionResidual(hidden=96, history=8).to(device)
    l38.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)["model"]); l38.eval()
    records, counts, diag = build_records(entries, arrays, seq_cache, hidden, mask, l29, l38, device, args.cap)
    threshold = choose_threshold(records["calibration"]["l29"])
    t = threshold["threshold"]
    strategy = {name: {"threshold": threshold, "null_gate": "disabled",
                       "screening": metrics(records["screening"][name], t, None)}
                for name in ("l29", "control", "l38")}
    residual = np.asarray(diag["screening_residual"], np.float64)
    cont = np.asarray(diag["screening_continuation_logit"], np.float64)
    payload = {"format": "locatemot-l38-frozen-emission-candidate-gate-v1",
               "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": sha(Path(args.checkpoint)),
               "teacher_checkpoint": str(L29_CHECKPOINT.resolve()), "teacher_checkpoint_sha256": sha(L29_CHECKPOINT),
               "manifest": str(FAST_MANIFEST.resolve()), "manifest_sha256": sha(FAST_MANIFEST),
               "cache_manifest_sha256": sha(L28_ROOT / "manifest.json"), "counts": counts,
               "threshold_contract": "one calibration-only L29 balanced-F1 threshold reused for teacher, no-residual control, and L38",
               "screening_gt_used_for_threshold": False, "screening_gt_used_for_model_selection": False,
               "null_gate": "disabled by L38 contract; NULL diagnostic not used for emission",
               "residual_contract": {"bound": 0.05, "observed_max_abs": float(np.max(np.abs(residual))),
                                     "observed_mean_abs": float(np.mean(np.abs(residual))),
                                     "teacher_error_max": float(max(diag["screening_teacher_error"]))},
               "fragment_continuation_diagnostic": {"mean_logit": float(np.mean(cont)), "std_logit": float(np.std(cont)),
                                                     "used_for_final_emission": False},
               "semantic_inputs_excluded": ["pool_id", "source_id", "group_id", "state_key"],
               "token_level_alignment_verified": False,
               "motion_language_decomposition": "not claimed; no verified motion-language mask",
               "strategies": strategy,
               "gate": {"baseline": "l29", "l38_top1_not_drop_gt_0.02": strategy["l38"]["screening"]["top1_frame_recall"] >= strategy["l29"]["screening"]["top1_frame_recall"] - 0.02,
                        "l38_recall_not_drop_gt_0.03": strategy["l38"]["screening"]["recall"] >= strategy["l29"]["screening"]["recall"] - 0.03,
                        "hard_violation_improved": strategy["l38"]["screening"]["hard_violation_rate"] < strategy["l29"]["screening"]["hard_violation_rate"],
                        "precision_not_obviously_worse": strategy["l38"]["screening"]["precision"] >= strategy["l29"]["screening"]["precision"] - 0.02,
                        "residual_within_bound": float(np.max(np.abs(residual))) <= 0.05 + 1e-6}}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"out": str(out), "counts": counts, "gate": payload["gate"],
                      "l29": strategy["l29"]["screening"], "l38": strategy["l38"]["screening"]}, indent=2), flush=True)


if __name__ == "__main__": main()
