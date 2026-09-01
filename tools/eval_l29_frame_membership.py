#!/usr/bin/env python3
"""Candidate-level held-out evaluation for the L29 decoder."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
from tools.audit_l28_identity_bank import BANK_ROOT, load_labels
from tools.eval_l27_fast_rmot import frame_groups, make_entries
from tools.summarize_l27_fast_rmot import load_caches
from tools.train_l26_crossmodal_adapter import V5 as TEXT_ROOT
from tools.train_l28_track_set_decoder import state_at

SCORE_ROOT = ROOT / "outputs/l27/fast_rmot_validation_retry"


def build_cache(video):
    path = BANK_ROOT / f"{video}.pt"; bank = torch.load(path, map_location="cpu", weights_only=False)
    tensors = bank["tensors"]; count = int(tensors["track_id"].numel())
    labels, _ = load_labels(path, count, tensors=tensors)
    ids = tensors["track_id"].long().numpy(); frames = tensors["frame"].long().numpy()
    by_track = defaultdict(list)
    for row, track in enumerate(ids.tolist()): by_track[int(track)].append(row)
    tracks = sorted(by_track); ptr = [0]; order = []
    for track in tracks: order.extend(by_track[track]); ptr.append(ptr[-1] + len(by_track[track]))
    order_t = torch.as_tensor(np.asarray(order, np.int64))
    feature = torch.cat([tensors[n].float().reshape(count, -1) for n in
                         ("clip", "history_clip", "uidm_h", "geometry", "motion", "lifecycle", "objectness")], 1).half()
    return {"track_ids": torch.as_tensor(np.asarray(tracks, np.int64)),
            "track_ptr": torch.as_tensor(np.asarray(ptr, np.int64)),
            "obs_features": feature[order_t].contiguous(),
            "obs_frame": torch.as_tensor(frames[order_t.numpy()], dtype=torch.int32),
            "obs_gt_ids": [None] * len(order)}


def valid_tracks(cache, cutoff):
    ptr = cache["track_ptr"].numpy(); frames = cache["obs_frame"].numpy()
    return [i for i in range(len(ptr) - 1) if np.any(frames[int(ptr[i]):int(ptr[i + 1])] <= cutoff)]


def auc(score, label):
    score = np.asarray(score); label = np.asarray(label, bool); p = score[label]; n = score[~label]
    if not len(p) or not len(n): return None
    order = np.argsort(score, kind="stable"); rank = np.empty(len(order)); rank[order] = np.arange(1, len(order) + 1)
    return float((rank[label].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def evaluate(checkpoint, cap, device, emission):
    entries = make_entries(); text = torch.load(TEXT_ROOT / "text_tokens.pt", map_location="cpu", weights_only=False)
    hidden, mask = text["token_hidden"], text["attention_mask"].bool()
    caches = load_caches(SCORE_ROOT, entries, ("A_C1_S2000",))["A_C1_S2000"]
    model = L29FrameMembershipSetDecoder().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"]); model.eval()
    by_video = defaultdict(list)
    for e in entries:
        if e["split"] == "screening": by_video[str(e["video"])].append(e)
    records = []; used = 0
    for video, es in by_video.items():
        seq = build_cache(video); groups = []; union = set()
        for e in es:
            data = caches[(e["video"], e["expression"])]
            g = {int(f): idx for f, idx in frame_groups(data)}; groups.append((e, data, g)); union.update(g)
        for frame in sorted(union):
            if used >= cap: break
            obs, om, ot, _, _ = state_at(seq, frame)
            with torch.inference_mode(): enc = model.encode_observations(obs.to(device), om.to(device), ot.to(device))
            valid = valid_tracks(seq, frame)
            for e, data, g in groups:
                if frame not in g or used >= cap: continue
                qh = hidden[int(e["query_index"])].to(device); qm = mask[int(e["query_index"])].to(device)
                with torch.inference_mode(): out = model.forward_encoded(enc, enc[1], qh, qm)
                key = "set_membership_logits" if emission == "set" else "current_membership_logits"
                scores = out[key].float().cpu().numpy()
                by_track = {int(seq["track_ids"][ti]): float(scores[i]) for i, ti in enumerate(valid)}
                idx = g[frame]; track = data["track_id"][idx].astype(np.int64)
                records.append({"score": np.asarray([by_track.get(int(t), -20.) for t in track], np.float32),
                                "label": data["label"][idx].astype(bool)})
                used += 1
    flat_s = np.concatenate([r["score"] for r in records]); flat_y = np.concatenate([r["label"] for r in records])
    top1=[]; top5=[]; strict=[]; best=[]; aps=[]
    for r in records:
        order = np.argsort(-r["score"], kind="stable"); y=r["label"]; p=r["score"][y]; n=r["score"][~y]
        if y.any():
            top1.append(float(y[order[:1]].any())); top5.append(float(y[order[:5]].any()))
            ordered=y[order]; pos=np.flatnonzero(ordered); aps.append(float(np.mean([(ordered[:j+1]).mean() for j in pos])))
            if len(n): strict.append(float(p.min()-n.max())); best.append(float(p.max()-n.max()))
    selected = flat_s >= 0; tp=int((selected & flat_y).sum()); fp=int((selected & ~flat_y).sum())
    return {"frame_units":len(records),"candidate_rows":int(len(flat_y)),"positive_rows":int(flat_y.sum()),
            "roc_auc":auc(flat_s,flat_y),"frame_average_precision":float(np.mean(aps)),
            "top1_frame_recall":float(np.mean(top1)),"top5_frame_recall":float(np.mean(top5)),
            "strict_min_positive_margin":float(np.mean(strict)),"best_positive_margin":float(np.mean(best)),
            "hard_violation_rate":float(np.mean(np.asarray(strict)<0)),"zero_threshold_predictions":int(selected.sum()),
            "zero_threshold_precision":tp/max(1,tp+fp),"zero_threshold_recall":tp/max(1,int(flat_y.sum())),
            "predictions_per_positive":float(selected.sum()/max(1,int(flat_y.sum()))),"emission":emission,
            "screening_gt_used_for_selection":False}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--cap",type=int,default=100); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--emission",choices=("current","set"),default="current")
    a=ap.parse_args(); result=evaluate(a.checkpoint,a.cap,torch.device(a.device),a.emission)
    payload={"format":"locatemot-l29-frame-membership-eval-v1","checkpoint":str(Path(a.checkpoint).resolve()),"device":a.device,"cap":a.cap,"metrics":result}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2),flush=True)


if __name__ == "__main__": main()
