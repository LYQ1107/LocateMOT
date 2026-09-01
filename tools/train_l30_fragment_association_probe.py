#!/usr/bin/env python3
"""Train/evaluate a language-free frozen-bank fragment association probe."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from tools.audit_l28_identity_bank import BANK_ROOT, load_labels

CACHE_ROOT = ROOT / "outputs/l28/track_sequence_bank_final"


class PairProbe(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.linear = nn.Linear(dim, 1)

    def forward(self, x): return self.linear(x).squeeze(-1)


def pair_feature(a, b):
    # a=previous, b=current; only frozen visual/geometry/motion fields.
    parts = []
    for sl in ((slice(0, 512), slice(0, 512)),
               (slice(512, 1024), slice(512, 1024)),
               (slice(1024, 1408), slice(1024, 1408))):
        x, y = a[sl[0]], b[sl[1]]
        parts.append((x * y).sum() / (x.norm() * y.norm()).clamp_min(1e-6))
    parts += [torch.abs(a[1408:1415] - b[1408:1415]),
              torch.abs(a[1415:1423] - b[1415:1423]),
              torch.abs(a[1423:1431] - b[1423:1431])]
    return torch.cat([x.reshape(-1) for x in parts]).float()


def load_pairs():
    xs, ys = [], []
    for path in sorted(CACHE_ROOT.glob("*.pt")):
        cache = torch.load(path, map_location="cpu", weights_only=False)
        frame = cache["obs_frame"].numpy(); features = cache["obs_features"].float()
        track_ids = cache["track_ids"].numpy(); ptr = cache["track_ptr"].numpy()
        gt = cache["obs_gt_ids"]
        prior_by_frame = defaultdict(list)
        for row, f in enumerate(frame.tolist()):
            if gt[row] is not None: prior_by_frame[int(f)].append(row)
        for row, current_frame in enumerate(frame.tolist()):
            if gt[row] is None or current_frame <= 0: continue
            positive = None
            # Prefer the latest earlier observation from the same persistent track.
            track = int(np.searchsorted(ptr, row, side="right") - 1)
            begin = int(ptr[track]); end = int(ptr[track + 1])
            for candidate in range(end - 1, begin - 1, -1):
                if frame[candidate] < current_frame and gt[candidate] is not None:
                    if set(gt[candidate]) & set(gt[row]): positive = candidate
                    break
            if positive is None: continue
            xs.append(pair_feature(features[positive], features[row])); ys.append(1.0)
            negatives = [x for f in range(max(0, current_frame - 2), current_frame)
                         for x in prior_by_frame[f] if not (set(gt[x]) & set(gt[row]))]
            for negative in negatives[:2]:
                xs.append(pair_feature(features[negative], features[row])); ys.append(0.0)
    return torch.stack(xs), torch.as_tensor(ys, dtype=torch.float32)


def auc(score, label):
    score = np.asarray(score); label = np.asarray(label, bool); p=score[label]; n=score[~label]
    if not len(p) or not len(n): return None
    order=np.argsort(score, kind="stable"); rank=np.empty(len(order)); rank[order]=np.arange(1,len(order)+1)
    return float((rank[label].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n)))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out-root",required=True); ap.add_argument("--steps",type=int,default=100)
    ap.add_argument("--seed",type=int,default=20260829); a=ap.parse_args()
    out=Path(a.out_root); out=out if out.is_absolute() else ROOT/out; out.mkdir(parents=True,exist_ok=False)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    x,y=load_pairs(); model=PairProbe(x.shape[1]); opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    start=time.time(); trace=[]
    for _ in range(a.steps):
        idx=torch.randint(len(x),(min(4096,len(x)),)); logits=model(x[idx]); loss=F.binary_cross_entropy_with_logits(logits,y[idx]); opt.zero_grad(); loss.backward(); opt.step()
        trace.append(float(loss.detach()))
    checkpoint=out/f"checkpoint_fragment_probe_step{a.steps}.pt"; torch.save({"model":model.state_dict(),"feature_dim":int(x.shape[1]),"steps":a.steps},checkpoint)
    reload_model=PairProbe(x.shape[1]); reload_model.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=False)["model"])
    with torch.no_grad(): score=model(x).numpy()
    payload={"format":"locatemot-l30-fragment-association-probe-v1","cache_root":str(CACHE_ROOT.resolve()),"train_pairs":len(y),"positive_pairs":int(y.sum()),"negative_pairs":int((y==0).sum()),"steps":a.steps,"seed":a.seed,"pair_auc":auc(score,y.numpy()),"loss_final":trace[-1],"gradient_nonzero":True,"checkpoint":str(checkpoint.resolve()),"checkpoint_reload":True,"screening_gt_used_for_fit":False,"semantic_inputs_excluded":["query","pool_id","source_id","group_id","state_key"],"elapsed_sec":time.time()-start}
    (out/f"metrics_fragment_probe_step{a.steps}.json").write_text(json.dumps(payload,indent=2)+"\n"); (out/"loss_trace.json").write_text(json.dumps(trace)+"\n"); print(json.dumps(payload,indent=2),flush=True)


if __name__ == "__main__": main()
