#!/usr/bin/env python3
"""Train/evaluate the L26 DINOv2-to-word-token cross-modal adapter.

Only train-split expressions are used for optimization.  The fixed 96-query
screening set is loaded only for held-out reporting and never for fitting,
temperature selection, or threshold selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l26_crossmodal_adapter import L26BoundedResidual, L26CrossModalAdapter
from tools.train_rmot_candidate_scorer import auc, average_precision, scalar_stats

V5 = ROOT / "outputs/l26/candidate_bank_v5_crossmodal"
OLD = ROOT / "outputs/l19/dual_banks_features/kitti"
EXP = (ROOT / "outputs/l11/data/rmot_kitti/expressions.json", ROOT / "outputs/l16/data/kitti_missing/records/expressions.json")
SPLIT = ROOT / "outputs/l16/data/protocol/split_manifest.json"
FAST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
REC1 = ROOT / "outputs/l11/data/rmot_kitti"
REC2 = ROOT / "outputs/l16/data/kitti_missing/records"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_expressions():
    out = {}
    for path in EXP:
        for video, rows in json.loads(path.read_text()).items():
            for row in rows:
                out[(str(video), str(row["expression"]))] = {"video": str(video), **row}
    return [out[k] for k in sorted(out)]


def gt_path(video):
    for root in (REC1, REC2):
        p = root / f"{video}.pkl"
        if p.exists(): return p
    raise FileNotFoundError(video)


def load_gt(video):
    rec = pickle.loads(gt_path(video).read_bytes())
    return {int(x["frame"]): {str(k): np.asarray(v, np.float32) for k, v in x.get("gt_boxes", {}).items()} for x in rec["frames"]}


def frame_positive(boxes, target_ids, gt):
    targets = [gt.get(str(i)) for i in target_ids]
    targets = [x for x in targets if x is not None]
    if not targets: return np.zeros(len(boxes), bool), 0
    g = np.stack(targets); b = boxes.astype(np.float32)
    l = np.maximum(b[:, None, 0], g[None, :, 0]); t = np.maximum(b[:, None, 1], g[None, :, 1])
    r = np.minimum(b[:, None, 2], g[None, :, 2]); d = np.minimum(b[:, None, 3], g[None, :, 3])
    inter = np.maximum(0, r-l) * np.maximum(0, d-t)
    ba = np.maximum(0, b[:, 2]-b[:, 0]) * np.maximum(0, b[:, 3]-b[:, 1])
    ga = np.maximum(0, g[:, 2]-g[:, 0]) * np.maximum(0, g[:, 3]-g[:, 1])
    iou = inter / np.maximum(1e-6, ba[:, None] + ga[None, :] - inter)
    return iou.max(1) >= .5, int(np.count_nonzero(iou.max(0) >= .5))


def load_bank(video):
    d = torch.load(V5 / "kitti" / f"{video}.pt", map_location="cpu", weights_only=False)
    t = d["tensors"]
    return {"roi": t["dino_roi_tokens_v5"], "coords": t["roi_coords_v5"], "objectness": t["objectness"].float(),
            "pool": t["pool_id"].long(), "box": t["box"].float(), "frame_ids": t["frame_ids"].long(),
            "ptr": t["frame_ptr"].long(), "frames": t["frame"].long()}


def make_queries(expressions, split_of, banks, gts, text_index, only_keys=None):
    out = []
    for row in expressions:
        key = (row["video"], str(row["expression"]))
        if only_keys is not None and key not in only_keys: continue
        video = row["video"]
        if video not in banks: continue
        b = banks[video]; gt = gts[video]
        target = {int(k): {str(x) for x in v} for k, v in row.get("label", {}).items()}
        positive_frames = []; positive_counts = []; covered_ids = 0; target_count = 0
        for fi, frame in enumerate(b["frame_ids"].tolist()):
            ids = target.get(int(frame), set())
            p, covered = frame_positive(b["box"][int(b["ptr"][fi]):int(b["ptr"][fi+1])].numpy(), ids, gt.get(int(frame), {}))
            if p.any(): positive_frames.append(fi)
            positive_counts.append(int(p.sum())); covered_ids += covered; target_count += len(ids)
        out.append({"query_index": int(text_index[key]), "video": video, "expression": row["expression"], "sentence": row.get("sentence", row["expression"]),
                    "split": split_of[video], "target": target, "positive_frames": positive_frames, "positive_counts": positive_counts,
                    "covered_ids": covered_ids, "target_ids": target_count, "text_index": int(text_index[key])})
    return out


def frame_ref(q, fi, bank, gt):
    begin, end = int(bank["ptr"][fi]), int(bank["ptr"][fi+1]); frame = int(bank["frame_ids"][fi]); ids = q["target"].get(frame, set())
    positive, covered = frame_positive(bank["box"][begin:end].numpy(), ids, gt.get(frame, {}))
    return {"begin": begin, "end": end, "frame": frame, "positive": positive, "covered": covered}


def choose_hard(y, obj, scores):
    neg = np.flatnonzero(~y)
    if not len(neg): return np.empty(0, int), np.empty(0, int), np.empty(0, int)
    pre = neg[np.argsort(-obj[neg], kind="stable")[:min(96, len(neg))]]
    hard = pre[np.argsort(-scores[pre], kind="stable")[:min(24, len(pre))]]
    easy = np.setdiff1d(neg, hard, assume_unique=False)[:24]
    return hard, easy, pre


def score(model, qhidden, qmask, bank, ref, device):
    sl = slice(ref["begin"], ref["end"])
    out = model(qhidden.to(device), qmask.to(device), bank["roi"][sl].to(device), bank["coords"][sl].to(device))
    return out


def frame_loss(model, q, bank, gt, text_hidden, text_mask, device):
    rng = random.randrange(len(q["positive_frames"]))
    ref = frame_ref(q, q["positive_frames"][rng], bank, gt)
    qh = text_hidden[q["text_index"]]; qm = text_mask[q["text_index"]]
    with torch.inference_mode(): prelim = score(model, qh, qm, bank, ref, device)["score"].cpu().numpy()
    y = ref["positive"]; obj = bank["objectness"][ref["begin"]:ref["end"]].numpy(); hard, easy, _ = choose_hard(y, obj, prelim)
    out = score(model, qh, qm, bank, ref, device); s = out["score"]; z = s.new_zeros(())
    p = s[torch.as_tensor(np.flatnonzero(y), device=device)]; h = s[torch.as_tensor(hard, device=device)]; e = s[torch.as_tensor(easy, device=device)]
    pb = F.binary_cross_entropy_with_logits(p, torch.ones_like(p)) if len(p) else z
    hb = F.binary_cross_entropy_with_logits(h, torch.zeros_like(h)) if len(h) else z
    eb = F.binary_cross_entropy_with_logits(e, torch.zeros_like(e)) if len(e) else z
    pair = F.softplus(.2 + h[None, :] - p[:, None]).mean() if len(p) and len(h) else z
    listwise = torch.logsumexp(s, 0) - torch.logsumexp(p, 0) if len(p) else z
    violation = F.softplus(h[None, :] - p[:, None]).mean() if len(p) and len(h) else z
    entropy_reg = F.relu(.25 - out["attention_entropy"])
    total = pb + hb + .1*eb + pair + .5*listwise + .2*violation + .05*entropy_reg
    return total, {"total": float(total.detach()), "positive_bce": float(pb.detach()), "hard_bce": float(hb.detach()), "easy_bce": float(eb.detach()),
                   "pairwise": float(pair.detach()), "listwise": float(listwise.detach()), "violation": float(violation.detach()),
                   "entropy_reg": float(entropy_reg.detach()), "attention_entropy": float(out["attention_entropy"].detach()),
                   "positive_count": int(y.sum()), "hard_count": len(hard), "easy_count": len(easy)}


def select_eval_refs(queries, banks, gts, cap, seed):
    all_refs = [(q, fi) for q in queries for fi in range(len(banks[q["video"]]["frame_ids"]))]
    rng = random.Random(seed)
    if len(all_refs) <= cap: return all_refs
    # Keep every query represented, then fill a deterministic global sample.
    keep = []
    for q in queries:
        choices = list(range(len(banks[q["video"]]["frame_ids"])))
        keep.append((q, rng.choice(choices)))
    rest = [x for x in all_refs if x not in keep]; rng.shuffle(rest)
    return keep + rest[:max(0, cap-len(keep))]


def evaluate(model, queries, banks, gts, text_hidden, text_mask, device, cap, seed):
    model.eval(); refs = select_eval_refs(queries, banks, gts, cap, seed)
    scores=[]; labels=[]; margins=[]; full_margins=[]; violations=[]; ent=[]; null=[]; zero=pos_zero=0; top1=top5=multi=multi1=multi5=0; source={0:[0,0,0],1:[0,0,0]}; covered=targets=0
    for q, fi in refs:
        b=banks[q["video"]]; ref=frame_ref(q,fi,b,gts[q["video"]]); sl=slice(ref["begin"],ref["end"])
        with torch.inference_mode(): out=score(model,text_hidden[q["text_index"]],text_mask[q["text_index"]],b,ref,device); s=out["score"].cpu().numpy(); en=float(out["attention_entropy"].cpu())
        y=ref["positive"]; obj=b["objectness"][sl].numpy(); hard,easy,pre=choose_hard(y,obj,s); neg=np.flatnonzero(~y); full=neg[np.argsort(-s[neg],kind="stable")[:min(24,len(neg))]] if len(neg) else neg
        scores.append(s); labels.append(y); ent.append(en); zero+=int((s>=0).sum());pos_zero+=int(((s>=0)&y).sum()); covered+=ref["covered"];targets+=len(q["target"].get(ref["frame"],set()))
        if not y.any(): null.append(float(s.max()) if len(s) else 0); continue
        order=np.argsort(-s); top1+=int(y[order[:1]].any());top5+=int(y[order[:5]].any());
        if y.sum()>1: multi+=1;multi1+=int(y[order[:1]].any());multi5+=int(y[order[:5]].any())
        if len(hard): margins.append(float(s[y].min()-s[hard].max())); violations.append(margins[-1]<0); full_margins.append(float(s[y].min()-s[full].max()) if len(full) else margins[-1])
        for sid in (0,1):
            idx=np.flatnonzero(b["pool"][sl].numpy()==sid)
            if len(idx): so=idx[np.argsort(-s[idx])];source[sid][0]+=1;source[sid][1]+=int(y[so[:1]].any());source[sid][2]+=int(y[so[:5]].sum())
    flat_s=np.concatenate(scores) if scores else np.zeros(0); flat_y=np.concatenate(labels) if labels else np.zeros(0,bool)
    return {"frame_units":len(refs),"candidate_count":int(flat_y.size),"positive_count":int(flat_y.sum()),"positive_frame_count":int(sum(x.any() for x in labels)),"null_frame_count":len(null),
            "roc_auc":auc(flat_s,flat_y),"pr_auc":average_precision(flat_s,flat_y),"top1_frame_recall":top1/max(1,len(refs)-len(null)),"top5_frame_recall":top5/max(1,len(refs)-len(null)),
            "multi_positive_frame_count":multi,"multi_positive_top1_recall":multi1/max(1,multi),"multi_positive_top5_recall":multi5/max(1,multi),
            "positive_min_model_hard_margin":scalar_stats(margins),"full_frame_model_hard_margin":scalar_stats(full_margins),"hard_violation_rate":float(np.mean(violations)) if violations else None,
            "attention_entropy":scalar_stats(ent),"source_internal_precision":{"main":{"top1":source[0][1]/max(1,source[0][0]),"top5":source[0][2]/max(1,source[0][0]*5),"frames":source[0][0]},"reserve":{"top1":source[1][1]/max(1,source[1][0]),"top5":source[1][2]/max(1,source[1][0]*5),"frames":source[1][0]}},
            "null_highest_candidate_score":scalar_stats(null),"zero_threshold":{"predictions":zero,"positive":pos_zero,"predictions_per_positive":zero/max(1,int(flat_y.sum()))},"coverage":covered/max(1,targets)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out-root",required=True); ap.add_argument("--steps",type=int,default=100); ap.add_argument("--resume",default=""); ap.add_argument("--device",default="cuda:0"); ap.add_argument("--seed",type=int,default=17); ap.add_argument("--train-frame-cap",type=int,default=0); ap.add_argument("--eval-frame-cap",type=int,default=6000); ap.add_argument("--variant",choices=["token_region","projection","attribute_mask","bounded_residual"],default="token_region"); ap.add_argument("--base-checkpoint",default=""); args=ap.parse_args()
    out=Path(args.out_root); out=out if out.is_absolute() else ROOT/out
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    manifest=json.loads(FAST.read_text()); split=json.loads(SPLIT.read_text())["kitti_v2"]; split_of={v:"train" for v in split["train"]};split_of.update({v:"train_val" for v in split["train_val"]});split_of.update({v:"official_eval" for v in split["official_eval"]})
    expressions=load_expressions(); text_manifest=json.loads((V5/"text_manifest.json").read_text()); text_index={(x["video"],x["expression"]):int(x["query_index"]) for x in text_manifest["expressions"]}; text=torch.load(V5/"text_tokens.pt",map_location="cpu",weights_only=False); text_hidden=text["token_hidden"].float(); text_mask=text["attention_mask"].bool()
    videos=sorted(set(split["train"] + [str(x["video"]) for x in manifest["queries"]])); banks={v:load_bank(v) for v in videos};gts={v:load_gt(v) for v in videos}
    train_keys={(x["video"],str(x["expression"])) for x in expressions if split_of[x["video"]]=="train"}; screen_keys={(str(x["video"]),str(x["expression"])) for x in manifest["queries"] if str(x.get("split"))=="screening"}
    train_q=make_queries(expressions,split_of,banks,gts,text_index,train_keys); screen_q=make_queries(expressions,split_of,banks,gts,text_index,screen_keys)
    if len(train_q)!=7757 or len(screen_q)!=96: raise AssertionError(f"query isolation train={len(train_q)} screen={len(screen_q)}")
    device=torch.device(args.device)
    if args.variant == "bounded_residual":
        if not args.base_checkpoint: raise ValueError("bounded_residual requires --base-checkpoint")
        base=L26CrossModalAdapter(variant="token_region").to(device);base.load_state_dict(torch.load(args.base_checkpoint,map_location=device,weights_only=False)["model"]);model=L26BoundedResidual(base).to(device)
    else:
        model=L26CrossModalAdapter(variant=args.variant).to(device)
    resumed_from=None
    if args.resume:
        ck=torch.load(args.resume,map_location=device,weights_only=False);model.load_state_dict(ck["model"]);resumed_from=str(Path(args.resume).resolve())
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4); train_rows=[];grads=[];start=time.time(); trainable=[q for q in train_q if q["positive_frames"]]
    model.train(); rng=random.Random(args.seed)
    for step in range(args.steps):
        chosen=[trainable[rng.randrange(len(trainable))] for _ in range(4)]; losses=[]; parts=[]
        for q in chosen:
            l,p=frame_loss(model,q,banks[q["video"]],gts[q["video"]],text_hidden,text_mask,device);losses.append(l);parts.append(p)
        loss=torch.stack(losses).mean();opt.zero_grad(set_to_none=True);loss.backward();grads.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)));opt.step();train_rows.append({k:float(np.mean([x[k] for x in parts])) for k in parts[0] if k in ("total","positive_bce","hard_bce","easy_bce","pairwise","listwise","violation","entropy_reg","attention_entropy")})
    train_eval=evaluate(model,train_q,banks,gts,text_hidden,text_mask,device,min(args.eval_frame_cap,3000),args.seed+11);screen_eval=evaluate(model,screen_q,banks,gts,text_hidden,text_mask,device,args.eval_frame_cap,args.seed+29)
    report={"format":"locatemot-l26-crossmodal-adapter-v1","stage":args.variant,"manifest":str(FAST),"manifest_sha256":sha(FAST),"v5_root":str(V5),"dino_checkpoint_sha256":json.loads((V5/"build_summary.json").read_text())["dino_checkpoint_sha256"],"roberta_checkpoint_sha256":json.loads((V5/"build_summary.json").read_text())["roberta_checkpoint_sha256"],"device":str(device),"seed":args.seed,"steps":args.steps,"variant":args.variant,"train_query_count":len(train_q),"screening_query_count":len(screen_q),"train_frame_cap":min(args.eval_frame_cap,3000),"screening_frame_cap":args.eval_frame_cap,"screening_gt_used_for_fit":False,"excluded_semantic_inputs":["pool_id","source","group","state","tracker","old_checkpoint"],"motion_language_decomposition":"not claimed; no verified motion token mask","train":{"steps":args.steps,"positive_query_count":len(trainable),"elapsed_sec":time.time()-start,"loss":{k:scalar_stats([r[k] for r in train_rows]) for k in ("total","positive_bce","hard_bce","easy_bce","pairwise","listwise","violation","entropy_reg","attention_entropy")},"gradient_norm":scalar_stats(grads)},"train_metrics":train_eval,"screening_metrics":screen_eval,"resumed_from":resumed_from,"base_checkpoint":str(Path(args.base_checkpoint).resolve()) if args.base_checkpoint else None}
    ck=out/f"checkpoint_{args.variant.lower()}_step{args.steps}.pt";torch.save({"model":model.state_dict(),"manifest_sha256":report["manifest_sha256"],"v5_root":str(V5),"steps":args.steps,"variant":args.variant},ck);report["checkpoint"]=str(ck);reload=(L26BoundedResidual(L26CrossModalAdapter(variant="token_region")).to(device) if args.variant=="bounded_residual" else L26CrossModalAdapter(variant=args.variant).to(device));reload.load_state_dict(torch.load(ck,map_location=device,weights_only=False)["model"]);report["checkpoint_reload"]=True;report["elapsed_sec"]=time.time()-start;(out/f"metrics_{args.variant.lower()}_step{args.steps}.json").write_text(json.dumps(report,indent=2)+"\n");(out/"README.md").write_text(f"# L26 {args.variant} cross-modal adapter\n\nFull Refer-KITTI train expressions fit; fixed screening is held-out.\n")
    print(json.dumps({"out":str(out),"screening":{k:screen_eval.get(k) for k in ("roc_auc","pr_auc","top1_frame_recall","positive_min_model_hard_margin","hard_violation_rate","zero_threshold")}},indent=2),flush=True)


if __name__=="__main__":main()
