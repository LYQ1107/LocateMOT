#!/usr/bin/env python3
"""Held-out candidate diagnostics for the L37 expression-level decoder."""
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
from locatemot.models.l37_expression_track_set import L37ExpressionTrackSet
from tools.audit_l29_emission_contract import build_cache as build_l19_sequence_cache
from tools.eval_l27_fast_rmot import frame_groups, make_entries
from tools.summarize_l27_fast_rmot import load_caches
from tools.train_l26_crossmodal_adapter import V5 as TEXT_ROOT
from tools.train_l28_track_set_decoder import state_at

SCORE_ROOT = ROOT / "outputs/l27/fast_rmot_validation_retry"
L28_ROOT = ROOT / "outputs/l28/track_sequence_bank_final"
L29_CHECKPOINT = ROOT / "outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"
L37_CHECKPOINT = ROOT / "outputs/l37/train/expression_track_set_smoke100/checkpoint_l37_expression_track_set_step100.pt"
L27_SUMMARY = ROOT / "outputs/l27/fast_rmot_validation_formal/summary.json"
FAST_MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_name(entry):
    return f"{entry['video']}_{hashlib.sha1(entry['expression'].encode()).hexdigest()[:12]}.npz"


def valid_indices(cache, cutoff):
    ptr = cache["track_ptr"].numpy(); frames = cache["obs_frame"].numpy()
    return [i for i in range(len(ptr) - 1)
            if np.any(frames[int(ptr[i]):int(ptr[i + 1])] <= int(cutoff))]


def auc(scores, labels):
    scores = np.asarray(scores, np.float64); labels = np.asarray(labels, bool)
    p, n = scores[labels], scores[~labels]
    if not len(p) or not len(n): return None
    order = np.argsort(scores, kind="stable"); ranks = np.empty(len(order), np.float64)
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[labels].sum() - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def choose_threshold(records, fixed=None):
    if fixed is not None:
        return {"threshold": float(fixed), "source": "immutable_l27_calibration_precision_first"}
    values = np.concatenate([x["score"] for x in records if len(x["score"])])
    labels = np.concatenate([x["label"] for x in records if len(x["label"])])
    candidates = np.unique(np.quantile(values, np.linspace(.01, .995, 160)))
    candidates = np.concatenate((candidates, [values.min(), values.max()]))
    best = None
    for t in candidates:
        selected = values >= t
        tp = int((selected & labels).sum()); fp = int((selected & ~labels).sum())
        fn = int((~selected & labels).sum())
        precision = tp / max(1, tp + fp); recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        item = (f1, precision, recall, -float(t), float(t), tp, fp, fn)
        if best is None or item > best:
            best = item
    return {"threshold": best[4], "source": "calibration_only_balanced_candidate_f1",
            "calibration_precision": best[1], "calibration_recall": best[2],
            "calibration_f1": best[0], "tp": best[5], "fp": best[6], "fn": best[7]}


def choose_null_threshold(records):
    rows = [(float(np.asarray(x["null_logit"]).reshape(-1)[0]), not bool(x["label"].any()))
            for x in records if len(x["null_logit"])]
    if not rows:
        return {"threshold": None, "source": "no-null-output"}
    values = np.asarray([x[0] for x in rows], np.float64)
    labels = np.asarray([x[1] for x in rows], bool)
    candidates = np.unique(np.quantile(values, np.linspace(.01, .99, 128)))
    best = None
    for t in candidates:
        pred = values >= t
        tp = int((pred & labels).sum()); fp = int((pred & ~labels).sum())
        fn = int((~pred & labels).sum())
        f1 = 2 * tp / max(1, 2 * tp + fp + fn)
        item = (f1, -float(t), float(t), tp, fp, fn)
        if best is None or item > best: best = item
    return {"threshold": best[2], "source": "calibration_only_null_f1",
            "calibration_f1": best[0], "positive_null_frames": int(labels.sum()),
            "predicted_null_frames": int((values >= best[2]).sum())}


def select(record, threshold, use_null):
    chosen = record["score"] >= float(threshold)
    nt = record.get("null_logit")
    if use_null and nt is not None and len(nt) and float(nt[0]) >= float(use_null):
        chosen = np.zeros_like(chosen, bool)
    return chosen


def metrics(records, threshold, null_threshold=None):
    tp = fp = fn = 0; selected_count = 0; empty = 0; null_accept = 0
    top1=[]; top5=[]; topk=[]; strict=[]; best=[]; average=[]; aps=[]
    multi_pos=[]; multi_hit=[]; fp_per=[]; transitions=defaultdict(list); source_stats={}
    for r in records:
        y = r["label"].astype(bool); s = r["score"]
        chosen = select(r, threshold, null_threshold)
        tp += int((chosen & y).sum()); fp += int((chosen & ~y).sum()); fn += int((~chosen & y).sum())
        selected_count += int(chosen.sum()); empty += int(not chosen.any()); null_accept += int(not y.any() and chosen.any())
        fp_per.append(int((chosen & ~y).sum()))
        transitions[r["query_index"]].append((r["frame"], set(r["track_id"][chosen].tolist())))
        order = np.argsort(-s, kind="stable")
        if y.any():
            top1.append(float(y[order[:1]].any())); top5.append(float(y[order[:5]].any()))
            k = min(len(order), int(y.sum())); topk.append(float(y[order[:k]].sum() / max(1, int(y.sum()))))
            pos, neg = s[y], s[~y]
            if len(neg):
                strict.append(float(pos.min() - neg.max())); best.append(float(pos.max() - neg.max())); average.append(float(pos.mean() - neg.max()))
            ordered = y[order]; positions = np.flatnonzero(ordered)
            if len(positions): aps.append(float(np.mean([ordered[:j + 1].mean() for j in positions])))
            if int(y.sum()) > 1:
                multi_pos.append(1); multi_hit.append(float((chosen & y).sum() / max(1, int(y.sum()))))
        for sid, name in ((0, "main"), (1, "reserve")):
            mask = r["source"] == sid
            if name not in source_stats: source_stats[name] = [0, 0, 0]
            source_stats[name][0] += int((chosen & mask).sum()); source_stats[name][1] += int((y & mask).sum()); source_stats[name][2] += int((chosen & mask & y).sum())
    switches = 0; query_recall={}
    for q, seq in transitions.items():
        seq.sort(); prev=set(); hits=[]
        for _, cur in seq:
            if prev and cur and cur != prev: switches += 1
            prev = cur; hits.append(float(bool(cur)))
        query_recall[str(q)] = float(np.mean(hits)) if hits else 0.0
    def summary(values):
        if not values: return {"count": 0, "mean": None}
        a=np.asarray(values,float); return {"count":len(a),"mean":float(a.mean()),"median":float(np.median(a))}
    source = {k:{"selected":v[0],"positive":v[1],"true_positive":v[2],"precision":v[2]/max(1,v[0]),"recall":v[2]/max(1,v[1])} for k,v in source_stats.items()}
    return {"frame_units":len(records),"candidate_rows":int(sum(len(x["label"]) for x in records)),"positive_rows":int(sum(x["label"].sum() for x in records)),"selected":selected_count,"tp":tp,"fp":fp,"fn":fn,"precision":tp/max(1,tp+fp),"recall":tp/max(1,tp+fn),"f1":2*tp/max(1,2*tp+fp+fn),"top1_frame_recall":float(np.mean(top1)) if top1 else None,"top5_frame_recall":float(np.mean(top5)) if top5 else None,"topk_set_recall_k_equals_gt_positive_count":float(np.mean(topk)) if topk else None,"strict_min_positive_margin":summary(strict),"best_positive_margin":summary(best),"average_positive_margin":summary(average),"hard_violation_rate":float(np.mean(np.asarray(strict)<0)) if strict else None,"frame_average_precision":float(np.mean(aps)) if aps else None,"multi_positive_frame_count":len(multi_pos),"multi_positive_recall":float(np.mean(multi_hit)) if multi_hit else None,"false_positive_candidates_per_frame":float(np.mean(fp_per)) if fp_per else None,"empty_output_rate":empty/max(1,len(records)),"null_frame_false_acceptance":null_accept/max(1,len(records)),"predictions_per_gt_positive":selected_count/max(1,int(sum(x["label"].sum() for x in records))),"source_precision":source,"identity_switches_transition_diagnostic":switches,"query_frame_recall":query_recall}


def build_records(entries, arrays, seq_cache, text_hidden, text_mask, l29, l37, device, screen_cap):
    screen_units=[]
    for e in entries:
        if e["split"] != "screening": continue
        d=arrays[(e["video"],e["expression"])]
        screen_units.extend((str(e["video"]),str(e["expression"]),int(f)) for f,_ in frame_groups(d))
    screen_units.sort(); chosen_screen=set(screen_units[i] for i in np.linspace(0,len(screen_units)-1,min(screen_cap,len(screen_units)),dtype=int))
    needed=[]
    for e in entries:
        d=arrays[(e["video"],e["expression"])]
        for frame,_ in frame_groups(d):
            unit=(str(e["video"]),str(e["expression"]),int(frame))
            if e["split"]=="calibration" or unit in chosen_screen: needed.append((unit,e))
    by_frame=defaultdict(list)
    for unit,e in needed: by_frame[(unit[0],unit[2])].append((unit,e))
    records={"calibration":{"l27":[],"l29":[],"l37":[]},"screening":{"l27":[],"l29":[],"l37":[]}}
    for (video,frame), pairs in sorted(by_frame.items()):
        cache=seq_cache[video]; obs,om,ot,_,_=state_at(cache,frame,history=8)
        valid=valid_indices(cache,frame); track_ids=cache["track_ids"][torch.as_tensor(valid)].numpy().astype(np.int64)
        with torch.inference_mode():
            encoded29=l29.encode_observations(obs.to(device),om.to(device),ot.to(device))
            encoded37=None
            for unit,e in pairs:
                qh=text_hidden[int(e["text_index"])].to(device); qm=text_mask[int(e["text_index"])].to(device)
                out29=l29.forward_encoded(encoded29,encoded29[1],qh,qm)
                out37=l37(obs.to(device),om.to(device),ot.to(device),qh,qm)
                d=arrays[(e["video"],e["expression"])]
                idx=np.flatnonzero(d["frame"]==frame); tracks=d["track_id"][idx].astype(np.int64)
                map29={int(t):float(v) for t,v in zip(track_ids,out29["current_membership_logits"].float().cpu().numpy())}
                map37={int(t):float(v) for t,v in zip(track_ids,out37["current_membership_logits"].float().cpu().numpy())}
                base={"video":video,"expression":str(e["expression"]),"query_index":int(e["query_index"]),"frame":frame,"track_id":tracks,"label":d["label"][idx].astype(bool),"source":d["source"][idx].astype(np.int8)}
                r29={**base,"score":np.asarray([map29.get(int(t),-20.) for t in tracks],np.float32),"null_logit":np.asarray([float(out29["null_logit"].float().cpu())],np.float32)}
                r37={**base,"score":np.asarray([map37.get(int(t),-20.) for t in tracks],np.float32),"null_logit":np.asarray([float(out37["null_logit"].float().cpu())],np.float32)}
                kind="calibration" if e["split"]=="calibration" else "screening"
                records[kind]["l27"].append({**base,"score":d["score"][idx].astype(np.float32),"null_logit":None})
                records[kind]["l29"].append(r29); records[kind]["l37"].append(r37)
    return records, {"screening_units_selected":len(chosen_screen),"screening_units_available":len(screen_units),"calibration_units":len(records["calibration"]["l27"])}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",default=str(L37_CHECKPOINT)); ap.add_argument("--out",default="outputs/l37/eval/candidate_gate_100.json"); ap.add_argument("--cap",type=int,default=100); ap.add_argument("--device",default="cuda:0"); args=ap.parse_args()
    assert Path.cwd().resolve()==ROOT
    entries=make_entries(); arrays=load_caches(SCORE_ROOT,entries,("A_C1_S2000",))["A_C1_S2000"]
    text_manifest=json.loads((TEXT_ROOT/"text_manifest.json").read_text())["expressions"]
    text_index={(str(x["video"]),str(x["expression"])):int(x["query_index"]) for x in text_manifest}
    for e in entries:
        e["text_index"] = text_index[(str(e["video"]),str(e["expression"]))]
    text=torch.load(TEXT_ROOT/"text_tokens.pt",map_location="cpu",weights_only=False); text_hidden=text["token_hidden"].float(); text_mask=text["attention_mask"].bool(); del text
    # L28 is deliberately train-only.  The fixed fast-manifest videos are not
    # present there, so construct the replay-only sequence view from the
    # immutable L19 bank; this performs no backbone forward and carries no GT.
    seq_cache={}
    for v in sorted({str(e["video"]) for e in entries}):
        path=L28_ROOT/f"{v}.pt"
        seq_cache[v] = (torch.load(path,map_location="cpu",weights_only=False)
                        if path.exists() else build_l19_sequence_cache(v))
    device=torch.device(args.device); l29=L29FrameMembershipSetDecoder().to(device); l29.load_state_dict(torch.load(L29_CHECKPOINT,map_location=device,weights_only=False)["model"]); l29.eval(); l37=L37ExpressionTrackSet(hidden=128,history=8).to(device); l37.load_state_dict(torch.load(args.checkpoint,map_location=device,weights_only=False)["model"]); l37.eval()
    records,counts=build_records(entries,arrays,seq_cache,text_hidden,text_mask,l29,l37,device,args.cap)
    formal=json.loads(L27_SUMMARY.read_text()); l27_fixed=formal["candidate_metrics"]["A_C1_S2000"]["calibration"]["precision_first"]["threshold"]
    thresholds={"l27":choose_threshold(records["calibration"]["l27"],l27_fixed),"l29":choose_threshold(records["calibration"]["l29"]),"l37":choose_threshold(records["calibration"]["l37"])}
    nulls={"l27":None,"l29":choose_null_threshold(records["calibration"]["l29"]),"l37":choose_null_threshold(records["calibration"]["l37"])}
    strategy={}
    for name in ("l27","l29","l37"):
        t=thresholds[name]["threshold"]; strategy[name]={"calibration_threshold":thresholds[name],"calibration_null_threshold":nulls[name],"screening_with_null":metrics(records["screening"][name],t, nulls[name]["threshold"] if nulls[name] else None),"screening_without_null":metrics(records["screening"][name],t,None)}
    payload={"format":"locatemot-l37-expression-track-set-candidate-gate-v1","checkpoint":str(Path(args.checkpoint).resolve()),"checkpoint_sha256":sha(Path(args.checkpoint)),"l29_checkpoint":str(L29_CHECKPOINT.resolve()),"l29_checkpoint_sha256":sha(L29_CHECKPOINT),"manifest":str(FAST_MANIFEST.resolve()),"manifest_sha256":sha(FAST_MANIFEST),"cache_root":str(L28_ROOT.resolve()),"cache_manifest_sha256":sha(L28_ROOT/"manifest.json"),"query_counts":{"all":len(entries),"calibration":64,"screening":96},"counts":counts,"screening_gt_used_for_threshold":False,"screening_gt_used_for_model_selection":False,"token_level_alignment_verified":False,"motion_language_decomposition":"not claimed; no verified motion-language mask","strategies":strategy,"comparison_notes":{"l27":"immutable formal precision-first calibration threshold", "l29":"frozen L29 current-membership checkpoint, threshold fitted on calibration", "l37":"new expression-level query-token/persistent-track set checkpoint", "l37_no_null":"same L37 score and calibration threshold, NULL gate disabled"}}
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps({"out":str(out),"counts":counts,"l37_screening":strategy["l37"]["screening_without_null"]},indent=2),flush=True)


if __name__=="__main__": main()
