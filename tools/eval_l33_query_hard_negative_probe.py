#!/usr/bin/env python3
"""Evaluate L29 and L33 on fixed calibration plus 100 held-out frame units.

Calibration labels are used only to freeze one balanced threshold per model.
Screening labels are consumed only after scoring, for the reported diagnostics.
No TrackEval or full dataset evaluation is performed here.
"""
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
from locatemot.models.l33_query_hard_negative_probe import L33QueryHardNegativeProbe
from tools.audit_l28_identity_bank import BANK_ROOT
from tools.eval_l27_fast_rmot import (base_metrics, make_entries, pool_frame_units,
                                      selection, threshold_metrics)
from tools.summarize_l27_fast_rmot import cache_name, load_caches
from tools.train_l33_query_hard_negative_probe import candidate_visual_features
from tools.train_l28_track_set_decoder import state_at

MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
L27_CACHE = ROOT / "outputs/l27/fast_rmot_validation_retry"
L29_CHECKPOINT = ROOT / "outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"
L33_CHECKPOINT = ROOT / "outputs/l33/train/query_hard_negative_probe_smoke100/checkpoint_query_hard_negative_probe_step100.pt"
TEXT_ROOT = ROOT / "outputs/l26/candidate_bank_v5_crossmodal"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_sequence(video):
    bank = torch.load(BANK_ROOT / f"{video}.pt", map_location="cpu", weights_only=False)["tensors"]
    frame = bank["frame"].numpy().astype(np.int32); track = bank["track_id"].numpy().astype(np.int64)
    groups = defaultdict(list)
    for row, tid in enumerate(track.tolist()): groups[int(tid)].append(row)
    tracks = sorted(groups); order=[]; ptr=[0]
    for tid in tracks:
        order.extend(sorted(groups[tid], key=lambda r: (int(frame[r]), r))); ptr.append(len(order))
    order = np.asarray(order, np.int64)
    pieces = [bank[k].float().reshape(len(frame), -1) for k in
              ("clip", "history_clip", "uidm_h", "geometry", "motion", "lifecycle", "objectness")]
    return {"track_ids": torch.as_tensor(np.asarray(tracks, np.int64)),
            "track_ptr": torch.as_tensor(np.asarray(ptr, np.int64)),
            "obs_features": torch.cat(pieces, 1)[torch.as_tensor(order)].contiguous(),
            "obs_frame": torch.as_tensor(frame[order], dtype=torch.int32),
            "obs_gt_ids": [None] * len(order)}


def valid_track_indices(seq, cutoff):
    ptr, frames = seq["track_ptr"].numpy(), seq["obs_frame"].numpy()
    return [i for i in range(len(ptr)-1)
            if np.any(frames[int(ptr[i]):int(ptr[i+1])] <= int(cutoff))]


def state_at_fast(seq, cutoff, history=8):
    """Equivalent L28 history selection using binary search per track."""
    ptr, frames, feature = seq["track_ptr"].numpy(), seq["obs_frame"].numpy(), seq["obs_features"]
    n, dim = len(ptr) - 1, int(feature.shape[1])
    values = torch.zeros((n, history, dim), dtype=torch.float32)
    mask = torch.zeros((n, history), dtype=torch.bool)
    times = torch.zeros((n, history), dtype=torch.float32)
    selected_gt = [[None] * history for _ in range(n)]; selected_frames = [[-1] * history for _ in range(n)]
    denom = max(1.0, float(cutoff) + 1.0)
    valid = []
    for ti in range(n):
        begin, end = int(ptr[ti]), int(ptr[ti + 1]); end_pos = int(np.searchsorted(frames[begin:end], int(cutoff), side="right"))
        if not end_pos: continue
        valid.append(ti); chosen = np.arange(begin + max(0, end_pos-history), begin + end_pos, dtype=np.int64)
        offset = history - len(chosen); values[ti, offset:] = feature[torch.as_tensor(chosen)].float(); mask[ti, offset:] = True
        times[ti, offset:] = torch.as_tensor(frames[chosen].astype(np.float32) / denom)
        for j, row in enumerate(chosen.tolist(), offset): selected_frames[ti][j] = int(frames[row])
    v = torch.as_tensor(valid, dtype=torch.long)
    return values[v], mask[v], times[v], [selected_gt[i] for i in valid], [selected_frames[i] for i in valid]


def fast_balanced_calibration(data):
    """Frame-balanced threshold sweep with frame indices materialized once."""
    frames = np.asarray(data["frame"]); score = np.asarray(data["score"]); label = np.asarray(data["label"]).astype(bool)
    groups = [np.flatnonzero(frames == f) for f in np.unique(frames)]
    candidates = np.unique(np.quantile(score, np.linspace(.01, .995, 128))).tolist()
    candidates += [float(score.min()), float(score.max())]
    rows=[]
    for t in candidates:
        selected = score >= float(t); frame_f1=[]; fp=[]; frame_recall=[]
        for idx in groups:
            y=label[idx]; s=selected[idx]; tp=int((s&y).sum()); fpi=int((s&~y).sum()); fn=int((~s&y).sum())
            p=tp/max(1,tp+fpi); r=tp/max(1,tp+fn); frame_f1.append(2*p*r/max(1e-12,p+r) if y.any() else 0.0); fp.append(fpi); frame_recall.append(r)
        tp=int((selected&label).sum()); fpi=int((selected&~label).sum()); fn=int((~selected&label).sum())
        rows.append((float(t), {"frame_f1":float(np.mean(frame_f1)),"frame_recall":float(np.mean(frame_recall)),"precision":tp/max(1,tp+fpi),"recall":tp/max(1,tp+fn),"false_positive_candidates_per_frame":float(np.mean(fp))}))
    eligible=[x for x in rows if x[1]["false_positive_candidates_per_frame"] <= 10]
    t,m=max(eligible or rows,key=lambda x:(x[1]["frame_f1"],x[1]["precision"],x[1]["recall"]))
    return {"balanced":{"threshold":t,"calibration_metrics":m},"null_max_threshold":None,"null_max_samples":0,"gap_threshold":None,"gap_samples":0}


def add_hard_violation(data, ranking):
    strict = []
    for frame in np.unique(data["frame"]):
        idx = np.flatnonzero(data["frame"] == frame); y = data["label"][idx].astype(bool)
        if y.any() and (~y).any():
            strict.append(float(data["score"][idx][y].min() - data["score"][idx][~y].max()))
    ranking["hard_violation"] = float(np.mean(np.asarray(strict) < 0)) if strict else None
    ranking["hard_violation_units"] = len(strict)
    return ranking


def current_membership_logits(model, encoded, query_tokens, query_mask):
    """Exact L29 current-membership path without unused track-set attention."""
    obs, mask, _track_base = encoded
    n, length, _ = obs.shape
    # This is the same expression used by L29.forward_encoded after its base
    # call; skipping track logits/stale logits only removes unrelated compute.
    qtok = model.base.text_proj(torch.nan_to_num(query_tokens.float()))
    q = model.base._masked_mean(
        qtok, query_mask.bool().unsqueeze(0) if query_mask.ndim == 1 else query_mask.bool())
    if q.shape[0] != n:
        q = q[:1].expand(n, -1)
    membership = model.base.membership_head(
        torch.cat((obs, q[:, None, :].expand(n, length, -1)), dim=-1)).squeeze(-1)
    latest = mask.long().sum(1).clamp_min(1) - 1
    return membership.gather(1, latest[:, None]).squeeze(1)


def empty_record():
    return {"frame": [], "track_id": [], "box": [], "score": [], "source": [], "label": [], "gt_iou": []}


def append_rows(record, frame, data, idx, scores):
    record["frame"].extend([int(frame)] * len(idx)); record["track_id"].extend(data["track_id"][idx].tolist())
    record["box"].extend(data["box"][idx].tolist()); record["score"].extend(scores.tolist())
    record["source"].extend(data["source"][idx].tolist()); record["label"].extend(data["label"][idx].tolist())
    record["gt_iou"].extend(data["gt_iou"][idx].tolist())


def finalize(record):
    record["frame"] = np.asarray(record["frame"], np.int32); record["track_id"] = np.asarray(record["track_id"], np.int64)
    record["box"] = np.asarray(record["box"], np.float32).reshape(-1, 4); record["score"] = np.asarray(record["score"], np.float32)
    record["source"] = np.asarray(record["source"], np.int8); record["label"] = np.asarray(record["label"], np.uint8); record["gt_iou"] = np.asarray(record["gt_iou"], np.float32)
    return record


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(); out = Path(args.out); out = out if out.is_absolute() else ROOT / out
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    entries = make_entries(); loaded = load_caches(L27_CACHE, entries, ("A_C1_S2000",))["A_C1_S2000"]
    text = torch.load(TEXT_ROOT / "text_tokens.pt", map_location="cpu", weights_only=False)
    hidden, mask = text["token_hidden"].float(), text["attention_mask"].bool(); del text
    screen_units = []
    for entry in entries:
        if entry["split"] != "screening": continue
        data = loaded[(entry["video"], entry["expression"])]
        screen_units.extend((entry["video"], entry["expression"], int(frame)) for frame in np.unique(data["frame"]))
    screen_units.sort(); chosen = {screen_units[int(i)] for i in np.linspace(0, len(screen_units)-1, 100, dtype=int)}
    wanted = {"calibration": defaultdict(set), "screening": defaultdict(set)}
    for entry in entries:
        key=(entry["video"], entry["expression"]); data=loaded[key]
        if entry["split"] == "calibration": wanted["calibration"][key].update(np.unique(data["frame"]).tolist())
        else: wanted["screening"][key].update(f for v,e,f in chosen if (v,e)==key)
    by_video = defaultdict(list)
    for entry in entries: by_video[entry["video"]].append(entry)
    device = torch.device(args.device)
    l29 = L29FrameMembershipSetDecoder().to(device)
    l29.load_state_dict(torch.load(L29_CHECKPOINT, map_location=device, weights_only=False)["model"]); l29.eval()
    l33 = L33QueryHardNegativeProbe(hidden=128).to(device)
    l33.load_state_dict(torch.load(L33_CHECKPOINT, map_location=device, weights_only=False)["model"]); l33.eval()
    records = {"L29_membership": {"calibration": [], "screening": []}, "L33_query_conditioned": {"calibration": [], "screening": []}}
    audit = {"duplicate_keys": 0, "nonfinite": 0, "missing_tracks": 0, "rows": 0}; seen=set()
    for video, video_entries in sorted(by_video.items()):
        seq = build_sequence(video); frames = sorted(set().union(*(wanted[e["split"]][(video,e["expression"])] for e in video_entries)))
        for frame in frames:
            obs, om, ot, _, _ = state_at_fast(seq, frame)
            with torch.inference_mode(): encoded = l29.encode_observations(obs.to(device), om.to(device), ot.to(device))
            valid = valid_track_indices(seq, frame); track_to_idx = {int(seq["track_ids"][i]): j for j,i in enumerate(valid)}
            visual_base = candidate_visual_features(seq, frame)
            with torch.inference_mode():
                current_by_query = {}
                for entry in video_entries:
                    key=(video,entry["expression"]); split=entry["split"]
                    if frame not in wanted[split][key]: continue
                    qh, qm = hidden[int(entry["query_index"])].to(device), mask[int(entry["query_index"])].to(device)
                    current = current_membership_logits(l29, encoded, qh, qm).float().cpu().numpy()
                    data=loaded[key]; idx=np.flatnonzero(data["frame"] == frame); tids=data["track_id"][idx]
                    raw=np.asarray([current[track_to_idx[int(t)]] if int(t) in track_to_idx else -20.0 for t in tids], np.float32)
                    features = np.concatenate((current[:, None], visual_base), axis=1)
                    qscore_all = l33(torch.as_tensor(features, device=device), qh, qm)["relevance_logits"].float().cpu().numpy()
                    qscore = np.asarray([qscore_all[track_to_idx[int(t)]] if int(t) in track_to_idx else -20.0 for t in tids], np.float32)
                    # Candidate rows are aligned by stable track namespace; source
                    # is retained for reporting only and is never a model input.
                    missing = sum(int(t) not in track_to_idx for t in tids); audit["missing_tracks"] += missing
                    for score, name in ((raw, "L29_membership"), (qscore, "L33_query_conditioned")):
                        rec=empty_record(); append_rows(rec, frame, data, idx, score); rec=finalize(rec)
                        records[name][split].append(rec); audit["rows"] += len(rec["label"])
                        audit["nonfinite"] += int(not np.isfinite(score).all())
                    for t in tids.tolist():
                        keyrow=(video,int(entry["query_index"]),int(frame),int(t),int(frame)); audit["duplicate_keys"] += int(keyrow in seen); seen.add(keyrow)
        del seq
    results={}
    for name, parts in records.items():
        cal=pool_frame_units(parts["calibration"]); screen=pool_frame_units(parts["screening"]); calibration=fast_balanced_calibration(cal); threshold=calibration["balanced"]["threshold"]
        selected=selection(screen, threshold, "threshold")
        ranking = add_hard_violation(screen, base_metrics(screen))
        results[name]={"calibration": {"balanced": calibration["balanced"], "null_max_threshold": calibration["null_max_threshold"], "null_max_samples": calibration["null_max_samples"], "gap_threshold": calibration["gap_threshold"], "gap_samples": calibration["gap_samples"]}, "candidate_ranking": ranking, "threshold": threshold, "threshold_metrics": threshold_metrics(screen, selected), "screening_frame_units": len(parts["screening"]), "screening_gt_used_for_threshold": False}
    old = json.loads((ROOT / "outputs/l32/eval/agreement_gate_smoke100.json").read_text())
    payload={"format":"locatemot-l33-query-hard-negative-heldout-v1","manifest":str(MANIFEST.resolve()),"manifest_sha256":sha(MANIFEST),"cache_root":str(L27_CACHE.resolve()),"membership_checkpoint":str(L29_CHECKPOINT.resolve()),"probe_checkpoint":str(L33_CHECKPOINT.resolve()),"query_counts":{"calibration":64,"screening":96},"screening_selection":"fixed sorted 100 frame units from 96 screening queries","calibration_labels_used_for_threshold":True,"screening_gt_used_for_selection":False,"semantic_inputs_excluded":["pool_id","source_id","group_id","state_key","l30_association_score"],"audit":audit,"results":results,"l32_reference":old.get("strategies",old)}
    (out/"heldout_100.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2),flush=True)


if __name__ == "__main__": main()
