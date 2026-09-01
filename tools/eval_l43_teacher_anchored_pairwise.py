#!/usr/bin/env python3
"""L43 fixed-unit candidate gate with a single calibration-only threshold."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT=Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT"); sys.path.insert(0,str(ROOT))
from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
from locatemot.models.l43_teacher_anchored_pairwise import L43TeacherAnchoredPairwiseResidual
from tools.audit_l29_emission_contract import build_cache as build_l19_sequence_cache
from tools.eval_l27_fast_rmot import frame_groups, make_entries
from tools.summarize_l27_fast_rmot import load_caches
from tools.train_l26_crossmodal_adapter import V5
from tools.train_l28_track_set_decoder import state_at
from tools.train_l42_current_frame_grounding import StreamingCropPatchEncoder, load_bank, numeric_for

L19=ROOT/"outputs/l19/dual_banks_features/kitti"; SCORE=ROOT/"outputs/l27/fast_rmot_validation_retry"; L29=ROOT/"outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"; L28=ROOT/"outputs/l28/track_sequence_bank_final"; FAST=ROOT/"outputs/l19/protocol/kitti_fast_eval_manifest.json"


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def valid_indices(cache, cutoff):
    ptr,frames=cache["track_ptr"].numpy(),cache["obs_frame"].numpy()
    return [i for i in range(len(ptr)-1) if np.any(frames[int(ptr[i]):int(ptr[i+1])]<=int(cutoff))]


def threshold(records):
    values=np.concatenate([r["score"] for r in records if len(r["score"])])
    labels=np.concatenate([r["label"].astype(bool) for r in records if len(r["label"])])
    best=None
    for t in np.unique(np.quantile(values,np.linspace(.01,.995,160))):
        chosen=values>=t; tp=int((chosen&labels).sum()); fp=int((chosen&~labels).sum()); fn=int((~chosen&labels).sum()); p=tp/max(1,tp+fp); r=tp/max(1,tp+fn); f=2*p*r/max(1e-12,p+r); key=(f,p,r,-float(t),float(t),tp,fp,fn)
        if best is None or key>best: best=key
    return {"threshold":best[4],"source":"single_L29_calibration_only_balanced_F1","precision":best[1],"recall":best[2],"f1":best[0],"tp":best[5],"fp":best[6],"fn":best[7],"rows":int(len(labels))}


def stats(values):
    if not values:return {"count":0,"mean":None,"median":None,"max":None}
    x=np.asarray(values,float);return {"count":int(len(x)),"mean":float(x.mean()),"median":float(np.median(x)),"max":float(x.max())}


def metrics(records,t):
    tp=fp=fn=selected=empty=null_accept=0; top1=[];top5=[];strict=[];best=[];avg=[];multi=[];multi_recall=[];fpper=[];trans=defaultdict(list);source={"main":[0,0,0],"reserve":[0,0,0]}
    for r in records:
        y=r["label"].astype(bool);s=r["score"];chosen=s>=float(t);tp+=int((chosen&y).sum());fp+=int((chosen&~y).sum());fn+=int((~chosen&y).sum());selected+=int(chosen.sum());empty+=int(not chosen.any());null_accept+=int(not y.any() and chosen.any());fpper.append(int((chosen&~y).sum()));trans[int(r["query_index"])].append((int(r["frame"]),set(r["track_id"][chosen].tolist())));order=np.argsort(-s,kind="stable")
        if y.any():
            top1.append(float(y[order[:1]].any()));top5.append(float(y[order[:5]].any()));p=s[y];n=s[~y]
            if len(n):strict.append(float(p.min()-n.max()));best.append(float(p.max()-n.max()));avg.append(float(p.mean()-n.max()))
            if y.sum()>1:multi.append(1);multi_recall.append(float((chosen&y).sum()/max(1,int(y.sum()))))
        for sid,name in ((0,"main"),(1,"reserve")):
            q=r["source"]==sid;source[name][0]+=int((chosen&q).sum());source[name][1]+=int((y&q).sum());source[name][2]+=int((chosen&q&y).sum())
    switches=0
    for seq in trans.values():
        seq.sort();prev=set()
        for _,cur in seq:
            if prev and cur and cur!=prev:switches+=1
            prev=cur
    sp={k:{"selected":v[0],"positive":v[1],"true_positive":v[2],"precision":v[2]/max(1,v[0]),"recall":v[2]/max(1,v[1])} for k,v in source.items()}
    return {"frame_units":len(records),"candidate_rows":int(sum(len(r["label"]) for r in records)),"positive_rows":int(sum(r["label"].sum() for r in records)),"selected":selected,"tp":tp,"fp":fp,"fn":fn,"precision":tp/max(1,tp+fp),"recall":tp/max(1,tp+fn),"f1":2*tp/max(1,2*tp+fp+fn),"top1_frame_recall":float(np.mean(top1)) if top1 else None,"top5_frame_recall":float(np.mean(top5)) if top5 else None,"strict_min_positive_margin":stats(strict),"best_positive_margin":stats(best),"average_positive_margin":stats(avg),"hard_violation_rate":float(np.mean(np.asarray(strict)<0)) if strict else None,"multi_positive_frame_count":len(multi),"multi_positive_recall":float(np.mean(multi_recall)) if multi_recall else None,"false_positive_candidates_per_frame":float(np.mean(fpper)) if fpper else None,"empty_output_rate":empty/max(1,len(records)),"null_frame_false_acceptance":null_accept/max(1,len(records)),"predictions_per_gt_positive":selected/max(1,int(sum(r["label"].sum() for r in records))),"source_precision":sp,"identity_switch_proxy":switches}


def rank_stats(teacher, student, y):
    p=np.flatnonzero(y);n=np.flatnonzero(~y)
    if not len(p) or not len(n):return {"pairs":0,"teacher_correct":0,"teacher_error":0,"teacher_correct_flips":0,"teacher_errors_corrected":0}
    td=teacher[p,None]-teacher[None,n];sd=student[p,None]-student[None,n];correct=td>0;error=~correct
    return {"pairs":int(td.size),"teacher_correct":int(correct.sum()),"teacher_error":int(error.sum()),"teacher_correct_flips":int((correct&(sd<0)).sum()),"teacher_errors_corrected":int((error&(sd>0)).sum())}


def add_rank(total,item):
    for k,v in item.items():total[k]=total.get(k,0)+int(v)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--checkpoint",required=True);ap.add_argument("--out",default="outputs/l43/eval/candidate_gate_100.json");ap.add_argument("--device",default="cuda:0");ap.add_argument("--cap",type=int,default=100);args=ap.parse_args();assert Path.cwd().resolve()==ROOT
    entries=make_entries();arrays=load_caches(SCORE,entries,("A_C1_S2000",))["A_C1_S2000"];tm=json.loads((V5/"text_manifest.json").read_text())["expressions"];ti={(str(x["video"]),str(x["expression"])):int(x["query_index"]) for x in tm};text=torch.load(V5/"text_tokens.pt",map_location="cpu",weights_only=False);hidden=text["token_hidden"].float();textmask=text["attention_mask"].bool();del text
    available=[]
    for e in entries:
        if e["split"]=="screening":
            d=arrays[(e["video"],e["expression"])]
            available.extend((str(e["video"]),str(e["expression"]),int(f)) for f,_ in frame_groups(d))
    available.sort();chosen_screen={available[i] for i in np.linspace(0,len(available)-1,min(args.cap,len(available)),dtype=int)};needed=[]
    for e in entries:
        d=arrays[(e["video"],e["expression"])]
        for f,_ in frame_groups(d):
            u=(str(e["video"]),str(e["expression"]),int(f))
            if e["split"]=="calibration" or u in chosen_screen:needed.append((u,e))
    by_frame=defaultdict(list)
    for u,e in needed:by_frame[(u[0],u[2])].append((u,e))
    videos=sorted({str(e["video"]) for _,e in needed});banks={v:load_bank(v) for v in videos};caches={}
    for v in videos:
        p=L28/f"{v}.pt";caches[v]=torch.load(p,map_location="cpu",weights_only=False) if p.exists() else build_l19_sequence_cache(v)
    device=torch.device(args.device);teacher_model=L29FrameMembershipSetDecoder().to(device);teacher_model.load_state_dict(torch.load(L29,map_location=device,weights_only=False)["model"]);teacher_model.eval();model=L43TeacherAnchoredPairwiseResidual(hidden=128,heads=4,layers=1).to(device);model.load_state_dict(torch.load(args.checkpoint,map_location=device,weights_only=False)["model"]);model.eval();encoder=StreamingCropPatchEncoder(device)
    records={"calibration":{"teacher":[],"zero_residual":[],"anchored_residual":[]},"screening":{"teacher":[],"zero_residual":[],"anchored_residual":[]}};rank_total={"pairs":0,"teacher_correct":0,"teacher_error":0,"teacher_correct_flips":0,"teacher_errors_corrected":0};residual_values=[];delta_values=[];raw={"expression_frame_units":0,"candidate_crops":0,"missing_bank_rows":0,"missing_teacher_tracks":0,"pair_symmetry_max":0.0}
    for (video,frame),pairs in sorted(by_frame.items()):
        b=banks[video];first=arrays[(pairs[0][1]["video"],pairs[0][1]["expression"])];idx0=np.flatnonzero(first["frame"]==frame);base_tracks=first["track_id"][idx0].astype(np.int64);lookup={(int(b["frame"][r]),int(b["track"][r])):r for r in range(len(b["track"]))};rows=[lookup.get((int(frame),int(t)),-1) for t in base_tracks]
        if any(r<0 for r in rows):raw["missing_bank_rows"]+=sum(r<0 for r in rows);continue
        patches=encoder.encode(video,b,rows);numeric=numeric_for(b,rows);raw["candidate_crops"]+=len(rows);cm=torch.ones(len(rows),dtype=torch.bool,device=device);cache=caches[video];obs,om,ot,_,_=state_at(cache,frame,history=8);valid=valid_indices(cache,frame);ids=cache["track_ids"][torch.as_tensor(valid)].numpy().astype(np.int64)
        with torch.inference_mode():encoded=teacher_model.encode_observations(obs.to(device),om.to(device),ot.to(device))
        for unit,e in pairs:
            d=arrays[(e["video"],e["expression"])] ;idx=np.flatnonzero(d["frame"]==frame);tracks=d["track_id"][idx].astype(np.int64);y=d["label"][idx].astype(bool);source=d["source"][idx].astype(np.int8);qidx=int(e["query_index"]);qrow=ti[(str(e["video"]),str(e["expression"]))];qh=hidden[qrow].to(device);qm=textmask[qrow].to(device)
            order=[int(np.flatnonzero(base_tracks==t)[0]) if np.any(base_tracks==t) else -1 for t in tracks]
            if any(i<0 for i in order):raw["missing_bank_rows"]+=sum(i<0 for i in order);continue
            pp=patches[torch.as_tensor(order)];nnumeric=numeric[torch.as_tensor(order)];
            with torch.inference_mode():
                tout=teacher_model.forward_encoded(encoded,encoded[1],qh,qm);tmap={int(t):float(v) for t,v in zip(ids,tout["current_membership_logits"].float().cpu().tolist())};m=np.asarray([tmap.get(int(t),-20.) for t in tracks],np.float32);raw["missing_teacher_tracks"]+=sum(int(t not in tmap) for t in tracks);out=model(pp.to(device).float(),qh,nnumeric.to(device).float(),torch.from_numpy(m).to(device),cm[torch.as_tensor(order)],qm);student=out["final_score"].float().cpu().numpy();res=out["residual"].float().cpu().numpy();delta=out["delta_score"].float().cpu().numpy();raw["pair_symmetry_max"]=max(raw["pair_symmetry_max"],float(np.max(np.abs(res+res.T))) if res.size else 0.0);residual_values.extend(res[np.triu_indices(len(res),1)].tolist() if len(res)>1 else []);delta_values.extend(delta.tolist());add_rank(rank_total,rank_stats(m,student,y))
            kind="calibration" if e["split"]=="calibration" else "screening";base={"video":video,"expression":str(e["expression"]),"query_index":qidx,"frame":int(frame),"track_id":tracks,"label":y,"source":source};records[kind]["teacher"].append({**base,"score":m});records[kind]["zero_residual"].append({**base,"score":m});records[kind]["anchored_residual"].append({**base,"score":student});raw["expression_frame_units"]+=1
    del encoder,model,teacher_model,banks,caches
    th=threshold(records["calibration"]["teacher"]);t=float(th["threshold"]);strategy={}
    for name in records["calibration"]:strategy[name]={"calibration_threshold":th,"calibration":metrics(records["calibration"][name],t),"screening":metrics(records["screening"][name],t)}
    base=strategy["teacher"]["screening"];pair_total=max(1,rank_total["pairs"]);gate={"top1_delta":strategy["anchored_residual"]["screening"]["top1_frame_recall"]-base["top1_frame_recall"],"recall_delta":strategy["anchored_residual"]["screening"]["recall"]-base["recall"],"hard_violation_delta":strategy["anchored_residual"]["screening"]["hard_violation_rate"]-base["hard_violation_rate"],"teacher_correct_flip_ratio":rank_total["teacher_correct_flips"]/pair_total,"precision_delta":strategy["anchored_residual"]["screening"]["precision"]-base["precision"],"fp_frame_delta":strategy["anchored_residual"]["screening"]["false_positive_candidates_per_frame"]-base["false_positive_candidates_per_frame"],"multi_positive_recall_delta":strategy["anchored_residual"]["screening"]["multi_positive_recall"]-base["multi_positive_recall"]}
    gate.update({"top1_preserved":gate["top1_delta"]>=-.02,"recall_preserved":gate["recall_delta"]>=-.03,"hard_violation_improved_by_0.05":gate["hard_violation_delta"]<=-.05,"teacher_correct_flip_under_1pct":gate["teacher_correct_flip_ratio"]<=.01,"precision_preserved":gate["precision_delta"]>=-.01,"fp_frame_preserved":gate["fp_frame_delta"]<=.10,"multi_positive_preserved":gate["multi_positive_recall_delta"]>=-.03,"residual_bound":max([abs(x) for x in residual_values],default=0.0)<=.05+1e-6});payload={"format":"locatemot-l43-teacher-anchored-pairwise-candidate-gate-v1","stage":"L43-B1","checkpoint":str(Path(args.checkpoint).resolve()),"checkpoint_sha256":sha(Path(args.checkpoint)),"teacher_checkpoint":str(L29.resolve()),"teacher_checkpoint_sha256":sha(L29),"manifest":str(FAST.resolve()),"manifest_sha256":sha(FAST),"counts":{"screening_units_selected":len(chosen_screen),"screening_units_available":len(available),"calibration_expression_frame_units":len(records["calibration"]["teacher"]),"screening_expression_frame_units":len(records["screening"]["teacher"])},"raw_replay":raw,"threshold_contract":"one L29 calibration-only threshold reused unchanged for teacher, zero-residual and anchored residual; no top-k/NULL/post-filter","strategies":strategy,"residual_diagnostics":{"pair_residual_mean":float(np.mean(residual_values)) if residual_values else 0.0,"pair_residual_max_abs":float(np.max(np.abs(residual_values))) if residual_values else 0.0,"candidate_delta_mean":float(np.mean(delta_values)) if delta_values else 0.0,"candidate_delta_max_abs":float(np.max(np.abs(delta_values))) if delta_values else 0.0,"rank_pair_totals":rank_total,"teacher_correct_flip_ratio":rank_total["teacher_correct_flips"]/pair_total,"teacher_error_correction_ratio":rank_total["teacher_errors_corrected"]/max(1,rank_total["teacher_error"])},"gates":gate,"screening_gt_used_for_fit":False,"screening_gt_used_for_threshold":False,"screening_gt_used_for_model_selection":False,"semantic_inputs_excluded":["source_id","pool_id","group_id","state_key"],"token_level_alignment_verified":False,"motion_language_decomposition":"not claimed; no verified motion-language mask","decision":"pass" if all(gate[k] for k in ("top1_preserved","recall_preserved","hard_violation_improved_by_0.05","teacher_correct_flip_under_1pct","precision_preserved","fp_frame_preserved","multi_positive_preserved","residual_bound")) else "fail"}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2)+"\n");print(json.dumps({"out":str(out),"decision":payload["decision"],"gates":gate,"teacher":base,"anchored":strategy["anchored_residual"]["screening"]},indent=2),flush=True)


if __name__=="__main__":main()
