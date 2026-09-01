#!/usr/bin/env python3
"""L42 candidate diagnostics on the fixed 100-unit held-out slice.

The calibration labels choose only scalar thresholds.  Screening labels are
read after the choices are frozen and are never used for model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
from locatemot.models.l42_current_frame_grounding import L42CurrentFrameGrounding
from tools.audit_l29_emission_contract import build_cache as build_l19_sequence_cache
from tools.eval_l27_fast_rmot import frame_groups, make_entries
from tools.summarize_l27_fast_rmot import load_caches
from tools.train_l26_crossmodal_adapter import REC1, REC2, V5, frame_positive
from tools.train_l28_track_set_decoder import state_at
from tools.train_l42_current_frame_grounding import StreamingCropPatchEncoder, numeric_for, load_bank

L19 = ROOT / "outputs/l19/dual_banks_features/kitti"
L27_SCORE = ROOT / "outputs/l27/fast_rmot_validation_retry"
L29 = ROOT / "outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"
L28 = ROOT / "outputs/l28/track_sequence_bank_final"
FAST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
L27_FORMAL = ROOT / "outputs/l27/fast_rmot_validation_formal/summary.json"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def load_gt(video):
    path = REC1 / f"{video}.pkl"
    if not path.exists(): path = REC2 / f"{video}.pkl"
    rec = pickle.loads(path.read_bytes())
    return {int(x["frame"]): {str(k): np.asarray(v, np.float32) for k, v in x.get("gt_boxes", {}).items()} for x in rec["frames"]}


def load_eval_bank(video):
    b = load_bank(video)
    t = torch.load(L19 / f"{video}.pt", map_location="cpu", weights_only=False)["tensors"]
    b["track"] = t["track_id"].long(); b["pool"] = t["pool_id"].long()
    return b


def valid_indices(cache, cutoff):
    ptr, frames = cache["track_ptr"].numpy(), cache["obs_frame"].numpy()
    return [i for i in range(len(ptr) - 1) if np.any(frames[int(ptr[i]):int(ptr[i + 1])] <= int(cutoff))]


def choose_threshold(records):
    values = np.concatenate([x["score"] for x in records if len(x["score"])])
    labels = np.concatenate([x["label"].astype(bool) for x in records if len(x["label"])])
    best = None
    for threshold in np.unique(np.quantile(values, np.linspace(.01, .995, 160))):
        chosen = values >= threshold; tp=int((chosen&labels).sum()); fp=int((chosen&~labels).sum()); fn=int((~chosen&labels).sum())
        p=tp/max(1,tp+fp); r=tp/max(1,tp+fn); f=2*p*r/max(1e-12,p+r); item=(f,p,r,-float(threshold),float(threshold),tp,fp,fn)
        if best is None or item > best: best=item
    return {"threshold": best[4], "source":"calibration_only_balanced_candidate_f1", "precision":best[1], "recall":best[2], "f1":best[0], "tp":best[5], "fp":best[6], "fn":best[7], "calibration_rows":int(len(labels))}


def summary(vals):
    if not vals: return {"count":0,"mean":None,"median":None}
    x=np.asarray(vals,float); return {"count":int(len(x)),"mean":float(x.mean()),"median":float(np.median(x))}


def metrics(records, threshold):
    tp=fp=fn=selected=empty=null_accept=0; top1=[]; top5=[]; strict=[]; best=[]; avg=[]; multi=[]; multi_hit=[]; fp_frame=[]; trans=defaultdict(list); source={"main":[0,0,0],"reserve":[0,0,0]}
    for r in records:
        y=r["label"].astype(bool); s=r["score"]; chosen=s>=float(threshold); tp+=int((chosen&y).sum()); fp+=int((chosen&~y).sum()); fn+=int((~chosen&y).sum()); selected+=int(chosen.sum()); empty+=int(not chosen.any()); null_accept+=int(not y.any() and chosen.any()); fp_frame.append(int((chosen&~y).sum())); trans[int(r["query_index"])].append((int(r["frame"]),set(r["track_id"][chosen].tolist())))
        order=np.argsort(-s,kind="stable")
        if y.any():
            top1.append(float(y[order[:1]].any())); top5.append(float(y[order[:5]].any())); pos=s[y]; neg=s[~y]
            if len(neg): strict.append(float(pos.min()-neg.max())); best.append(float(pos.max()-neg.max())); avg.append(float(pos.mean()-neg.max()))
            if y.sum()>1: multi.append(1); multi_hit.append(float((chosen&y).sum()/max(1,int(y.sum()))))
        for sid,name in ((0,"main"),(1,"reserve")):
            mask=r["source"]==sid; source[name][0]+=int((chosen&mask).sum()); source[name][1]+=int((y&mask).sum()); source[name][2]+=int((chosen&mask&y).sum())
    switches=0
    for seq in trans.values():
        seq.sort(); prev=set()
        for _,cur in seq:
            if prev and cur and cur!=prev: switches+=1
            prev=cur
    sp={k:{"selected":v[0],"positive":v[1],"true_positive":v[2],"precision":v[2]/max(1,v[0]),"recall":v[2]/max(1,v[1])} for k,v in source.items()}
    return {"frame_units":len(records),"candidate_rows":int(sum(len(x["label"]) for x in records)),"positive_rows":int(sum(x["label"].sum() for x in records)),"selected":selected,"tp":tp,"fp":fp,"fn":fn,"precision":tp/max(1,tp+fp),"recall":tp/max(1,tp+fn),"f1":2*tp/max(1,2*tp+fp+fn),"top1_frame_recall":float(np.mean(top1)) if top1 else None,"top5_frame_recall":float(np.mean(top5)) if top5 else None,"strict_min_positive_margin":summary(strict),"best_positive_margin":summary(best),"average_positive_margin":summary(avg),"hard_violation_rate":float(np.mean(np.asarray(strict)<0)) if strict else None,"multi_positive_frame_count":len(multi),"multi_positive_recall":float(np.mean(multi_hit)) if multi_hit else None,"false_positive_candidates_per_frame":float(np.mean(fp_frame)) if fp_frame else None,"empty_output_rate":empty/max(1,len(records)),"null_frame_false_acceptance":null_accept/max(1,len(records)),"predictions_per_gt_positive":selected/max(1,int(sum(x["label"].sum() for x in records))),"source_precision":sp,"identity_switch_proxy":switches}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--out",default="outputs/l42/eval/candidate_gate_100.json"); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--cap",type=int,default=100); args=ap.parse_args(); assert Path.cwd().resolve()==ROOT
    entries=make_entries(); arrays=load_caches(L27_SCORE,entries,("A_C1_S2000",))["A_C1_S2000"]
    text_manifest=json.loads((V5/"text_manifest.json").read_text())["expressions"]; text_index={(str(x["video"]),str(x["expression"])):int(x["query_index"]) for x in text_manifest}; text=torch.load(V5/"text_tokens.pt",map_location="cpu",weights_only=False); hidden=text["token_hidden"].float(); mask=text["attention_mask"].bool(); del text
    screen=[]
    for e in entries:
        if e["split"]!="screening": continue
        d=arrays[(e["video"],e["expression"])]
        screen.extend((str(e["video"]),str(e["expression"]),int(f)) for f,_ in frame_groups(d))
    screen.sort(); selected_screen={screen[i] for i in np.linspace(0,len(screen)-1,min(args.cap,len(screen)),dtype=int)}
    needed=[]
    for e in entries:
        d=arrays[(e["video"],e["expression"])]
        for f,_ in frame_groups(d):
            unit=(str(e["video"]),str(e["expression"]),int(f))
            if e["split"]=="calibration" or unit in selected_screen: needed.append((unit,e))
    by_frame=defaultdict(list)
    for unit,e in needed: by_frame[(unit[0],unit[2])].append((unit,e))
    videos=sorted({str(e["video"]) for _,e in needed}); banks={v:load_eval_bank(v) for v in videos}; gts={v:load_gt(v) for v in videos}; caches={}
    for v in videos:
        p=L28/f"{v}.pt"; caches[v]=torch.load(p,map_location="cpu",weights_only=False) if p.exists() else build_l19_sequence_cache(v)
    device=torch.device(args.device); l29=L29FrameMembershipSetDecoder().to(device); l29.load_state_dict(torch.load(L29,map_location=device,weights_only=False)["model"]); l29.eval(); model=L42CurrentFrameGrounding(hidden=128,heads=4,layers=2).to(device); model.load_state_dict(torch.load(args.checkpoint,map_location=device,weights_only=False)["model"]); model.eval(); encoder=StreamingCropPatchEncoder(device)
    records={"calibration":{"l27":[],"l29":[],"l42":[],"fallback":[],"teacher_only":[]},"screening":{"l27":[],"l29":[],"l42":[],"fallback":[],"teacher_only":[]}}; raw_stats={"units":0,"missing_bank_rows":0,"missing_teacher_tracks":0,"crop_count":0,"q_conf_max":[]}
    for (video,frame), pairs in sorted(by_frame.items()):
        b=banks[video]; d0=arrays[(pairs[0][1]["video"],pairs[0][1]["expression"])] ; idx0=np.flatnonzero(d0["frame"]==frame); tracks0=d0["track_id"][idx0].astype(np.int64)
        lookup={(int(b["frame"][r]),int(b["track"][r])):r for r in range(len(b["track"]))}; rows=[]
        for t in tracks0:
            r=lookup.get((int(frame),int(t))); rows.append(r if r is not None else -1)
        if any(r<0 for r in rows): raw_stats["missing_bank_rows"]+=sum(r<0 for r in rows); continue
        raw_stats["crop_count"]+=len(rows); patches=encoder.encode(video,b,rows); numeric=numeric_for(b,rows); cm=torch.ones((1,len(rows)),dtype=torch.bool,device=device); obs,om,ot,_,_=state_at(caches[video],frame,history=8); valid=valid_indices(caches[video],frame); track_ids=caches[video]["track_ids"][torch.as_tensor(valid)].numpy().astype(np.int64)
        with torch.inference_mode(): enc=l29.encode_observations(obs.to(device),om.to(device),ot.to(device))
        for unit,e in pairs:
            d=arrays[(e["video"],e["expression"])] ; idx=np.flatnonzero(d["frame"]==frame); tracks=d["track_id"][idx].astype(np.int64); y=d["label"][idx].astype(bool); source=d["source"][idx].astype(np.int8); qidx=int(e["query_index"]); qh=hidden[text_index[(str(e["video"]),str(e["expression"]))].item() if hasattr(text_index[(str(e["video"]),str(e["expression"]))],"item") else text_index[(str(e["video"]),str(e["expression"]))]].to(device); qm=mask[text_index[(str(e["video"]),str(e["expression"]))]].to(device)
            with torch.inference_mode():
                tout=l29.forward_encoded(enc,enc[1],qh,qm); teacher_map={int(t):float(s) for t,s in zip(track_ids,tout["current_membership_logits"].float().cpu().tolist())}; teacher=np.asarray([teacher_map.get(int(t),-20.) for t in tracks],np.float32); raw_stats["missing_teacher_tracks"]+=sum(int(t not in teacher_map) for t in tracks)
                out=model(patches.unsqueeze(0).to(device).float(),qh.unsqueeze(0),numeric.unsqueeze(0).to(device).float(),cm,qm.unsqueeze(0),torch.from_numpy(teacher).to(device).unsqueeze(0)); s=out["s_expr"][0].float().cpu().numpy(); qc=torch.sigmoid(out["q_conf"][0].float()).cpu().numpy(); raw_stats["q_conf_max"].append(float(qc.max()))
            kind="calibration" if e["split"]=="calibration" else "screening"; base={"video":video,"expression":str(e["expression"]),"query_index":qidx,"frame":int(frame),"track_id":tracks,"label":y,"source":source}
            records[kind]["l27"].append({**base,"score":d["score"][idx].astype(np.float32)}); records[kind]["l29"].append({**base,"score":teacher}); records[kind]["teacher_only"].append({**base,"score":teacher}); records[kind]["l42"].append({**base,"score":s}); records[kind]["fallback"].append({**base,"score":s if float(qc.max())>=.5 else teacher,"fallback_used":bool(float(qc.max())<.5)})
            raw_stats["units"]+=1
    del encoder, model, l29, caches, banks, gts
    thresholds={"l27":json.loads(L27_FORMAL.read_text())["candidate_metrics"]["A_C1_S2000"]["calibration"]["precision_first"]["threshold"],"l29":choose_threshold(records["calibration"]["l29"]),"l42":choose_threshold(records["calibration"]["l42"]),"fallback":choose_threshold(records["calibration"]["fallback"]),"teacher_only":choose_threshold(records["calibration"]["teacher_only"])}
    strategy={name:{"calibration_threshold":thresholds[name] if isinstance(thresholds[name],dict) else {"threshold":thresholds[name],"source":"immutable_l27_formal_precision_first"},"screening":metrics(records["screening"][name],thresholds[name]["threshold"] if isinstance(thresholds[name],dict) else thresholds[name]),"calibration":metrics(records["calibration"][name],thresholds[name]["threshold"] if isinstance(thresholds[name],dict) else thresholds[name])} for name in records["calibration"]}
    base=strategy["l29"]["screening"]; gates={}
    for name in ("l42","fallback"):
        x=strategy[name]["screening"]; gates[name]={"top1_not_drop_gt_0.02":x["top1_frame_recall"] is not None and x["top1_frame_recall"]>=base["top1_frame_recall"]-.02,"recall_not_drop_gt_0.03":x["recall"]>=base["recall"]-.03,"hard_violation_improved":x["hard_violation_rate"] is not None and x["hard_violation_rate"]<base["hard_violation_rate"],"precision_not_obviously_worse":x["precision"]>=base["precision"]-.02,"fp_not_obviously_worse":x["false_positive_candidates_per_frame"]<=base["false_positive_candidates_per_frame"]*1.05}
    payload={"format":"locatemot-l42-current-frame-grounding-candidate-gate-v1","checkpoint":str(Path(args.checkpoint).resolve()),"checkpoint_sha256":sha(Path(args.checkpoint)),"teacher_checkpoint":str(L29.resolve()),"teacher_checkpoint_sha256":sha(L29),"manifest":str(FAST.resolve()),"manifest_sha256":sha(FAST),"counts":{"screening_units_selected":len(selected_screen),"screening_units_available":len(screen),"calibration_units":len(records["calibration"])},"raw_replay_stats":raw_stats,"threshold_contract":"each L42 strategy threshold fitted only on calibration; no top-k/NULL gate; fallback quality threshold sigmoid(q_conf)>=0.5 fixed from train objective","screening_gt_used_for_threshold":False,"screening_gt_used_for_model_selection":False,"semantic_inputs_excluded":["source_id","pool_id","group_id","state_key"],"token_level_alignment_verified":False,"motion_language_decomposition":"not claimed; no verified motion-language mask","strategies":strategy,"gates_relative_to_l29":gates,"decision":"pass" if any(all(gates[x].values()) for x in gates) else "fail"}
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps({"out":str(out),"decision":payload["decision"],"gates":gates,"screening":{k:strategy[k]["screening"] for k in strategy}},indent=2),flush=True)


if __name__=="__main__": main()
