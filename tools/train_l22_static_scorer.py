"""Train/evaluate the bounded L22 static-only candidate scorer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.rmot_candidate_scorer_v2 import L22StaticCandidateScorer  # noqa: E402
from tools.train_rmot_candidate_scorer import auc, average_precision, load_bank, load_metadata, make_refs, scalar_stats  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def feature(ref, bank, rows):
    t = bank["tensors"]; absolute = torch.as_tensor(ref["begin"] + np.asarray(rows), dtype=torch.long)
    count = len(absolute); q = torch.as_tensor(ref["spec"], dtype=torch.float32).reshape(1, -1).expand(count, -1)
    return {"query": q, "crop_tight": t["crop_tight"].float().index_select(0, absolute),
            "crop_context": t["crop_context_1p5"].float().index_select(0, absolute),
            "geometry": t["geometry_v2"].float().index_select(0, absolute),
            "motion": t["motion_v2"].float().index_select(0, absolute),
            "objectness": t["objectness"].float().index_select(0, absolute).reshape(count, 1)}


def score(model, value, device):
    return model(*(value[k].to(device) for k in ("query", "crop_tight", "crop_context", "geometry", "motion", "objectness")))


def online_hard(model, ref, bank, pre, device):
    training = model.training; model.eval()
    with torch.no_grad(): out = score(model, feature(ref, bank, pre), device).cpu().numpy()
    if training: model.train()
    return out


def choose(ref, bank, rng, model, device, prefilter=96, topk=24):
    pos = np.flatnonzero(ref["positive"]); neg = np.flatnonzero(~ref["positive"])
    if len(pos) > 8: pos = np.asarray(rng.sample(pos.tolist(), 8), dtype=np.int64)
    obj = bank["tensors"]["objectness"][ref["begin"]:ref["end"]].float().numpy().reshape(-1)
    pre = neg[np.argsort(-obj[neg], kind="stable")[:min(prefilter, len(neg))]]
    if len(pre):
        hs = online_hard(model, ref, bank, pre, device); hard = pre[np.argsort(-hs, kind="stable")[:min(topk, len(pre))]]
    else: hard = np.zeros(0, dtype=np.int64)
    remaining = np.setdiff1d(neg, hard, assume_unique=False); easy_count = min(16, len(remaining))
    easy = np.asarray(rng.sample(remaining.tolist(), easy_count), dtype=np.int64) if easy_count else np.zeros(0, dtype=np.int64)
    rows = np.concatenate((pos, hard, easy)); mask = np.zeros(len(rows), dtype=bool); mask[len(pos):len(pos)+len(hard)] = True
    return rows, mask


def batch(model, refs, banks, rng, device, prefilter, topk):
    pieces = {k: [] for k in ("query", "crop_tight", "crop_context", "geometry", "motion", "objectness", "target", "hard")}; groups=[]; offset=0
    for ref in refs:
        rows, hard = choose(ref, banks[ref["video"]], rng, model, device, prefilter, topk); val=feature(ref,banks[ref["video"]],rows)
        for k in ("query", "crop_tight", "crop_context", "geometry", "motion", "objectness"): pieces[k].append(val[k])
        pieces["target"].append(torch.as_tensor(ref["positive"][rows], dtype=torch.float32)); pieces["hard"].append(torch.as_tensor(hard)); groups.append((offset,offset+len(rows),bool(ref["null"]))); offset+=len(rows)
    return {k: torch.cat(v).to(device) for k,v in pieces.items()}, groups


def train_step(model, optimizer, refs, banks, rng, device, args):
    values, groups = batch(model, refs, banks, rng, device, args.hard_prefilter, args.hard_topk)
    logits = score(model, values, device); target=values["target"]; hard_mask=values["hard"].bool(); pos_terms=[]; hard_terms=[]; easy_terms=[]; pair_terms=[]; list_terms=[]
    for a,b,_null in groups:
        l,y,h=logits[a:b],target[a:b],hard_mask[a:b]; pos=l[y>.5]; neg=l[y<=.5]; hn=l[(y<=.5)&h]; en=l[(y<=.5)&~h]
        if len(pos): pos_terms.append(nn.functional.binary_cross_entropy_with_logits(pos,torch.ones_like(pos)))
        if len(hn): hard_terms.append(nn.functional.binary_cross_entropy_with_logits(hn,torch.zeros_like(hn)))
        if len(en): easy_terms.append(nn.functional.binary_cross_entropy_with_logits(en,torch.zeros_like(en)))
        pn=hn[torch.topk(hn,min(args.pairwise_topk,len(hn)),largest=True).indices] if len(hn) else neg
        if len(pos) and len(pn):
            pair_terms.append(nn.functional.softplus(args.pair_margin-(pos[:,None]-pn[None,:])).mean())
            list_terms.append(torch.logsumexp(torch.cat((pos,pn)),0)-torch.logsumexp(pos,0))
    zero=logits.sum()*0.; pb=torch.stack(pos_terms).mean() if pos_terms else zero; hb=torch.stack(hard_terms).mean() if hard_terms else zero; eb=torch.stack(easy_terms).mean() if easy_terms else zero; pair=torch.stack(pair_terms).mean() if pair_terms else zero; listwise=torch.stack(list_terms).mean() if list_terms else zero
    total=pb+hb+.1*eb+args.pair_weight*pair+args.listwise_weight*listwise
    optimizer.zero_grad(set_to_none=True); total.backward(); grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)); optimizer.step()
    return {"total":float(total.detach().cpu()),"positive_bce":float(pb.detach().cpu()),"hard_negative_bce":float(hb.detach().cpu()),"easy_negative_bce":float(eb.detach().cpu()),"pairwise_margin":float(pair.detach().cpu()),"listwise_ranking":float(listwise.detach().cpu()),"sampled_candidates":int(len(target)),"sampled_positive":int(target.sum().item()),"sampled_hard":int(hard_mask.sum().item()),"null_buckets":int(sum(g[2] for g in groups)),"grad_norm":grad}


def evaluate(model, refs, bank, device, prefilter=96, topk=24):
    scores=[]; labels=[]; margins=[]; online_margins=[]; violations=[]; online_violations=[]; null_max=[]; top1=top5=pf=zero_pred=zero_pos=0; source={0:[0,0,0,0],1:[0,0,0,0]}; model.eval()
    for ref in refs:
        rows=np.arange(ref["end"]-ref["begin"],dtype=np.int64); out=score(model,feature(ref,bank,rows),device).detach().cpu().numpy(); y=ref["positive"].astype(bool); scores.append(out);labels.append(y);zero_pred+=int((out>=0).sum());zero_pos+=int(((out>=0)&y).sum());pos=np.flatnonzero(y);neg=np.flatnonzero(~y)
        if len(pos) and len(neg):
            pf+=1; order=np.argsort(-out,kind="stable");top1+=int(y[order[:1]].any());top5+=int(y[order[:5]].any());m=float(out[pos].min()-out[neg].max());margins.append(m);violations.append(m<0);obj=bank["tensors"]["objectness"][ref["begin"]:ref["end"]].float().numpy().reshape(-1);pre=neg[np.argsort(-obj[neg],kind="stable")[:min(prefilter,len(neg))]];hs=pre[np.argsort(-out[pre],kind="stable")[:min(topk,len(pre))]] if len(pre) else pre;om=float(out[pos].min()-out[hs].max()) if len(hs) else m;online_margins.append(om);online_violations.append(om<0)
            pool=bank["tensors"]["pool_id"][ref["begin"]:ref["end"]].numpy();
            for sid in (0,1):
                sr=np.flatnonzero(pool==sid)
                if len(sr): so=sr[np.argsort(-out[sr],kind="stable")];source[sid][0]+=1;source[sid][1]+=int(y[so[:1]].any());source[sid][2]+=min(5,len(so));source[sid][3]+=int(y[so[:5]].sum())
        elif ref["null"]: null_max.append(float(out.max()) if len(out) else 0.)
    s=np.concatenate(scores);y=np.concatenate(labels)
    return {"candidate_count":int(len(y)),"positive_count":int(y.sum()),"roc_auc":auc(s,y),"pr_auc":average_precision(s,y),"positive_score":scalar_stats(s[y]),"negative_score":scalar_stats(s[~y]),"positive_model_hard_margin":scalar_stats(margins),"model_hard_violation_rate":float(np.mean(violations)) if violations else None,"positive_online_hard_margin":scalar_stats(online_margins),"online_hard_violation_rate":float(np.mean(online_violations)) if online_violations else None,"positive_frame_count":pf,"top1_frame_recall":top1/max(1,pf),"top5_frame_recall":top5/max(1,pf),"source_internal_precision":{"main":{"top1":source[0][1]/max(1,source[0][0]),"top5":source[0][3]/max(1,source[0][2]),"selected_frames":source[0][0]},"reserve":{"top1":source[1][1]/max(1,source[1][0]),"top5":source[1][3]/max(1,source[1][2]),"selected_frames":source[1][0]}},"null_highest_candidate_score":scalar_stats(null_max),"zero_threshold":{"predictions":zero_pred,"positive":zero_pos,"predictions_per_positive":zero_pred/max(1,int(y.sum()))}}


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--manifest",default="outputs/l19/protocol/kitti_fast_eval_manifest.json");ap.add_argument("--bank-root",default="outputs/l22/candidate_bank_v2");ap.add_argument("--out-root",required=True);ap.add_argument("--steps",type=int,default=50);ap.add_argument("--seed",type=int,default=17);ap.add_argument("--device",default="cuda:0");ap.add_argument("--batch-frames",type=int,default=8);ap.add_argument("--hard-prefilter",type=int,default=96);ap.add_argument("--hard-topk",type=int,default=24);ap.add_argument("--pairwise-topk",type=int,default=12);ap.add_argument("--pair-weight",type=float,default=1.0);ap.add_argument("--listwise-weight",type=float,default=.5);ap.add_argument("--pair-margin",type=float,default=.5);ap.add_argument("--grad-clip",type=float,default=5.);args=ap.parse_args();
    manifest=Path(args.manifest);bank_root=Path(args.bank_root);out=Path(args.out_root)
    if not manifest.is_absolute():manifest=ROOT/manifest
    if not bank_root.is_absolute():bank_root=ROOT/bank_root
    if not out.is_absolute():out=ROOT/out
    if out.exists():raise FileExistsError(out)
    data=json.loads(manifest.read_text());rows=sorted(data["queries"],key=lambda x:int(x["query_index"]));
    if len(rows)!=160 or sum(r["split"]=="calibration" for r in rows)!=64 or sum(r["split"]=="screening" for r in rows)!=96:raise ValueError("fixed manifest must be 160=64+96")
    meta=load_metadata();videos=sorted({str(r["video"]) for r in rows});banks={v:load_bank(bank_root/"kitti"/f"{v}.pt") for v in videos};refs=make_refs(rows,meta,banks);train=[r for r in refs if r["split"]=="calibration"];val=[r for r in refs if r["split"]=="screening"];device=torch.device(args.device)
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    torch.manual_seed(args.seed);np.random.seed(args.seed);rng=random.Random(args.seed);model=L22StaticCandidateScorer().to(device);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.steps,eta_min=2e-5);initial={v:evaluate(model,[r for r in val if r["video"]==v],banks[v],device,args.hard_prefilter,args.hard_topk) for v in videos};
    def merge(parts):
        outp={k:sum(p[k] for p in parts) for k in ("candidate_count","positive_count","positive_frame_count")};
        for k in ("roc_auc","pr_auc","top1_frame_recall","top5_frame_recall","model_hard_violation_rate","online_hard_violation_rate"):outp[k]=float(np.mean([p[k] for p in parts if p[k] is not None]))
        for k in ("positive_model_hard_margin","positive_online_hard_margin","null_highest_candidate_score"):outp[k]={"count":sum(p[k].get("count",0) for p in parts),"mean":float(np.average([p[k].get("mean",0.) for p in parts],weights=[max(1,p[k].get("count",0)) for p in parts]))}
        outp["source_internal_precision"]={s:{k:float(np.mean([p["source_internal_precision"][s][k] for p in parts])) for k in ("top1","top5")} for s in ("main","reserve")};outp["zero_threshold"]={k:sum(p["zero_threshold"][k] for p in parts) for k in ("predictions","positive")};outp["zero_threshold"]["predictions_per_positive"]=outp["zero_threshold"]["predictions"]/max(1,outp["positive_count"]);return outp
    loss=[];started=time.time();model.train()
    for step in range(1,args.steps+1):
        sampled=rng.sample(train,min(args.batch_frames,len(train)));row=train_step(model,opt,sampled,banks,rng,device,args);scheduler.step();row["step"]=step;loss.append(row)
        if step==1 or step==args.steps:print(json.dumps(row,sort_keys=True),flush=True)
    train_metrics=merge([evaluate(model,[r for r in train if r["video"]==v],banks[v],device,args.hard_prefilter,args.hard_topk) for v in videos]);val_metrics=merge([evaluate(model,[r for r in val if r["video"]==v],banks[v],device,args.hard_prefilter,args.hard_topk) for v in videos]);out.mkdir(parents=True,exist_ok=False);checkpoint=out/f"checkpoint_step{args.steps}.pt";torch.save({"format":"locatemot-l22-static-scorer-v2","step":args.steps,"seed":args.seed,"manifest_sha256":sha(manifest),"bank_sha256":{v:banks[v]["bank_sha256"] for v in videos},"config":vars(args),"model":model.state_dict(),"optimizer":opt.state_dict()},checkpoint)
    gate={"auc_gt_065":bool(val_metrics["roc_auc"]>.65),"pr_auc_gate":bool(val_metrics["pr_auc"]>=.3667),"top1_gt_060":bool(val_metrics["top1_frame_recall"]>.60),"hard_margin_gt_previous":bool(val_metrics["positive_online_hard_margin"]["mean"]> -1.075),"hard_violation_below_previous":bool(val_metrics["online_hard_violation_rate"]<.8473),"zero_threshold_predictions_per_positive_lt3":bool(val_metrics["zero_threshold"]["predictions_per_positive"]<3),"all_candidates_accepted":False};gate["passed_250"]=args.steps>=250 and all(gate.values())
    payload={"format":"locatemot-l22-static-scorer-training-v1","provenance":{"project_root":str(ROOT),"manifest":str(manifest),"manifest_sha256":sha(manifest),"query_count":160,"calibration_queries":64,"screening_queries":96,"bank_root":str(bank_root),"bank_sha256":{v:banks[v]["bank_sha256"] for v in videos},"official_eval_used":False,"trackeval_used":False,"tracker_modified":False,"old_checkpoint_modified":False,"grouping":False,"membership":False,"source_acceptance":False,"null_scalar_subtraction":False,"temporal_gru":False,"source_in_score":False,"motion_query_status":"static-only; no motion-specific query embedding; real causal bank motion feature only"},"config":vars(args),"initial_validation":merge(list(initial.values())),"train":train_metrics,"validation":val_metrics,"loss":{k:float(np.mean([r[k] for r in loss])) for k in ("total","positive_bce","hard_negative_bce","easy_negative_bce","pairwise_margin","listwise_ranking","grad_norm")},"sampling":{k:int(sum(r[k] for r in loss)) for k in ("sampled_candidates","sampled_positive","sampled_hard","null_buckets")},"gate":gate,"checkpoint":str(checkpoint),"elapsed_sec":time.time()-started}
    (out/f"metrics_step{args.steps}.json").write_text(json.dumps(payload,indent=2)+"\n");(out/"README.md").write_text("# L22 static-only v2 scorer\n\nNo grouping, source acceptance, NULL scalar, GRU, tracker or TrackEval.\n")
    print(json.dumps({"output":str(out),"gate":gate,"validation":{k:val_metrics[k] for k in ("roc_auc","pr_auc","top1_frame_recall","positive_online_hard_margin","online_hard_violation_rate")}},indent=2),flush=True)


if __name__=="__main__":main()
