#!/usr/bin/env python3
"""Held-out language-free fragment association evaluation."""
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
from tools.audit_l28_identity_bank import BANK_ROOT, load_labels


def feature(a, b):
    out=[]
    for s in (slice(0,512), slice(512,1024), slice(1024,1408)):
        x=a[:,s]; y=b[:,s]
        out.append((x*y).sum(1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(y,axis=1)).clip(1e-6))
    out.extend((np.abs(a[:,s]-b[:,s]) for s in (slice(1408,1415),slice(1415,1423),slice(1423,1431))))
    return np.concatenate([x.reshape(len(a),-1) for x in out],1).astype(np.float32)


def auc(scores, labels):
    order=np.argsort(scores,kind="stable"); y=np.asarray(labels)[order]; neg=0; wins=0
    for x in y:
        if x: wins += neg
        else: neg += 1
    return float(wins/max(1,int(y.sum())*int((~y.astype(bool)).sum())))


def make_video(video, limit, rng):
    path=BANK_ROOT/f"{video}.pt"; bank=torch.load(path,map_location="cpu",weights_only=False); t=bank["tensors"]
    n=int(t["track_id"].numel()); labels,_=load_labels(path,n,tensors=t)
    f=np.asarray(t["frame"].numpy()); tracks=np.asarray(t["track_id"].numpy()); source=np.asarray(t["pool_id"].numpy())
    x=np.concatenate([t[k].float().numpy().reshape(n,-1) for k in ("clip","history_clip","uidm_h","geometry","motion","lifecycle","objectness")],1)
    by=defaultdict(list)
    for row,tr in enumerate(tracks.tolist()): by[int(tr)].append(row)
    left=[]; right=[]; y=[]; src=[]
    for rows in by.values():
        rows=sorted(rows,key=lambda r:(int(f[r]),r))
        previous=None
        for row in rows:
            if labels[row] is None: continue
            if previous is not None and labels[previous] is not None:
                left.append(previous); right.append(row); y.append(float(labels[previous]==labels[row])); src.append(int(source[row]))
            previous=row
    if len(y)>limit:
        take=rng.choice(len(y),limit,replace=False); left=np.asarray(left)[take]; right=np.asarray(right)[take]; y=np.asarray(y)[take]; src=np.asarray(src)[take]
    else: left=np.asarray(left); right=np.asarray(right); y=np.asarray(y); src=np.asarray(src)
    return feature(x[left],x[right]),y.astype(bool),src


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--out",required=True); ap.add_argument("--max-pairs",type=int,default=100000); ap.add_argument("--seed",type=int,default=20260829)
    a=ap.parse_args(); rng=np.random.default_rng(a.seed); state=torch.load(a.checkpoint,map_location="cpu",weights_only=False); w=state["model"]["linear.weight"].numpy().reshape(-1); b=float(state["model"]["linear.bias"].item())
    xs=[]; ys=[]; sources=[]
    for video in ("0005","0011","0013","0019"):
        x,y,s=make_video(video,a.max_pairs//4,rng); xs.append(x); ys.append(y); sources.append(s)
    x=np.concatenate(xs); y=np.concatenate(ys); s=np.concatenate(sources); score=x@w+b; positive=score[y]; negative=score[~y]
    source_stats={}
    for name,flag in (("main",s==0),("reserve",s!=0)):
        source_stats[name]={"pairs":int(flag.sum()),"positive_rate":float(y[flag].mean()) if flag.any() else None,"auc":auc(score[flag],y[flag]) if flag.any() and y[flag].any() and (~y[flag]).any() else None}
    payload={"format":"locatemot-l30-fragment-association-heldout-v1","checkpoint":str(Path(a.checkpoint).resolve()),"videos":["0005","0011","0013","0019"],"screening_pairs":len(y),"positive_pairs":int(y.sum()),"negative_pairs":int((~y).sum()),"pair_auc":auc(score,y),"positive_score_mean":float(positive.mean()),"negative_score_mean":float(negative.mean()),"same_gt_pair_score_gt0":float((positive>0).mean()),"different_gt_pair_score_gt0":float((negative>0).mean()),"source_stats":source_stats,"screening_gt_used_for_fit":False,"semantic_inputs_excluded":["query","pool_id","source_id","group_id","state_key"]}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__": main()
