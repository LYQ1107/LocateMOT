#!/usr/bin/env python3
"""L31 calibration-only bounded fusion of L29 membership and L30 identity."""
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
from tools.train_l28_track_set_decoder import state_at

SCORE_ROOT = ROOT / "outputs/l27/fast_rmot_validation_retry"
MEMBERSHIP_CHECKPOINT = ROOT / "outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"
ASSOC_CHECKPOINT = ROOT / "outputs/l30/train/fragment_probe_step500/checkpoint_fragment_probe_step500.pt"
OUT = ROOT / "outputs/l31/eval/bounded_fusion_smoke100.json"


def feature_np(a, b):
    out = []
    for sl in (slice(0, 512), slice(512, 1024), slice(1024, 1408)):
        x, y = a[:, sl], b[:, sl]
        out.append((x * y).sum(1) /
                   (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)).clip(1e-6))
    out.extend(np.abs(a[:, sl] - b[:, sl])
               for sl in (slice(1408, 1415), slice(1415, 1423), slice(1423, 1431)))
    return np.concatenate([x.reshape(len(a), -1) for x in out], 1).astype(np.float32)


def build_bank(video, assoc_w, assoc_b):
    path = BANK_ROOT / f"{video}.pt"
    bank = torch.load(path, map_location="cpu", weights_only=False)
    t = bank["tensors"]; n = int(t["track_id"].numel())
    labels, _ = load_labels(path, n, tensors=t)
    frame = t["frame"].numpy().astype(np.int64)
    track = t["track_id"].numpy().astype(np.int64)
    source = t["pool_id"].numpy().astype(np.int64)
    x = np.concatenate([t[k].float().numpy().reshape(n, -1) for k in
                        ("clip", "history_clip", "uidm_h", "geometry", "motion",
                         "lifecycle", "objectness")], 1)
    by_track = defaultdict(list)
    for row, tr in enumerate(track.tolist()): by_track[int(tr)].append(row)
    previous = np.full(n, -1, np.int64)
    for rows in by_track.values():
        rows.sort(key=lambda r: (int(frame[r]), r)); last = -1; last_frame = -1
        for row in rows:
            if last >= 0 and int(frame[row]) > last_frame: previous[row] = last
            if int(frame[row]) > last_frame: last, last_frame = row, int(frame[row])
    valid = np.flatnonzero(previous >= 0)
    assoc = np.zeros(n, np.float32)
    if len(valid): assoc[valid] = feature_np(x[previous[valid]], x[valid]) @ assoc_w + assoc_b
    lookup = {(int(frame[r]), int(track[r]), int(source[r])): float(assoc[r]) for r in range(n)}
    return {"lookup": lookup, "labels": labels, "finite": bool(np.isfinite(assoc).all()),
            "rows": n, "previous_rows": int(len(valid))}


def build_seq(video):
    path = BANK_ROOT / f"{video}.pt"; bank = torch.load(path, map_location="cpu", weights_only=False)
    t = bank["tensors"]; n = int(t["track_id"].numel()); frame=t["frame"].numpy(); ids=t["track_id"].numpy()
    by=defaultdict(list)
    for row,tr in enumerate(ids.tolist()): by[int(tr)].append(row)
    tracks=sorted(by); ptr=[0]; order=[]
    for tr in tracks: order.extend(by[tr]); ptr.append(ptr[-1]+len(by[tr]))
    order=torch.as_tensor(np.asarray(order,np.int64)); x=torch.cat([t[k].float().reshape(n,-1) for k in ("clip","history_clip","uidm_h","geometry","motion","lifecycle","objectness")],1).half()
    return {"track_ids":torch.as_tensor(np.asarray(tracks,np.int64)),"track_ptr":torch.as_tensor(np.asarray(ptr,np.int64)),"obs_features":x[order].contiguous(),"obs_frame":torch.as_tensor(frame[order.numpy()],dtype=torch.int32),"obs_gt_ids":[None]*len(order)}


def valid_tracks(seq, cutoff):
    ptr=seq["track_ptr"].numpy(); fr=seq["obs_frame"].numpy()
    return [i for i in range(len(ptr)-1) if np.any(fr[int(ptr[i]):int(ptr[i+1])] <= cutoff)]


def threshold_calibration(records, key):
    values=np.concatenate([r[key] for r in records]); labels=np.concatenate([r["label"] for r in records])
    candidates=np.unique(values)
    if len(candidates)>256: candidates=np.quantile(values,np.linspace(0,1,256))
    best=(-1.,0.,0,0,0)
    for t in candidates:
        s=values>=t; tp=int((s&labels).sum()); fp=int((s&~labels).sum()); fn=int((~s&labels).sum()); f=2*tp/max(1,2*tp+fp+fn)
        if f>best[0]: best=(f,float(t),tp,fp,fn)
    return {"threshold":best[1],"f1":best[0],"tp":best[2],"fp":best[3],"fn":best[4],"source":"calibration_only"}


def metrics(records,key,threshold):
    flat_s=np.concatenate([r[key] for r in records]); flat_y=np.concatenate([r["label"] for r in records]); selected=flat_s>=threshold
    top1=[]; top5=[]; strict=[]; best=[]; null_accept=0; empty=0; ap=[]
    for r in records:
        s=r[key]; y=r["label"]; o=np.argsort(-s,kind="stable");
        if y.any():
            top1.append(float(y[o[:1]].any())); top5.append(float(y[o[:5]].any())); ordered=y[o]; pos=np.flatnonzero(ordered); ap.append(float(np.mean([(ordered[:j+1]).mean() for j in pos])))
            n=s[~y]; p=s[y]
            if len(n): strict.append(float(p.min()-n.max())); best.append(float(p.max()-n.max()))
        chosen=s>=threshold; empty+=int(not chosen.any()); null_accept+=int(not y.any() and chosen.any())
    tp=int((selected&flat_y).sum()); fp=int((selected&~flat_y).sum()); fn=int((~selected&flat_y).sum())
    return {"selected":int(selected.sum()),"tp":tp,"fp":fp,"fn":fn,"precision":tp/max(1,tp+fp),"recall":tp/max(1,tp+fn),"fp_per_frame":fp/max(1,len(records)),"empty_rate":empty/max(1,len(records)),"null_false_acceptance":null_accept/max(1,len(records)),"predictions_per_positive":float(selected.sum()/max(1,int(flat_y.sum()))),"top1":float(np.mean(top1)),"top5":float(np.mean(top5)),"frame_average_precision":float(np.mean(ap)),"strict_margin":float(np.mean(strict)),"best_margin":float(np.mean(best)),"hard_violation":float(np.mean(np.asarray(strict)<0))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default=str(OUT)); ap.add_argument("--screen-cap",type=int,default=100); ap.add_argument("--device",default="cuda:0"); a=ap.parse_args()
    entries=make_entries(); text=torch.load(ROOT/"outputs/l26/candidate_bank_v5_crossmodal/text_tokens.pt",map_location="cpu",weights_only=False); hidden=text["token_hidden"]; mask=text["attention_mask"].bool()
    caches=load_caches(SCORE_ROOT,entries,("A_C1_S2000",))["A_C1_S2000"]
    assoc_state=torch.load(ASSOC_CHECKPOINT,map_location="cpu",weights_only=False)["model"]; aw=assoc_state["linear.weight"].numpy().reshape(-1); ab=float(assoc_state["linear.bias"].item())
    model=L29FrameMembershipSetDecoder().to(a.device); model.load_state_dict(torch.load(MEMBERSHIP_CHECKPOINT,map_location=a.device,weights_only=False)["model"]); model.eval()
    by=defaultdict(list); grouped={}; screen_units=[]
    for e in entries:
        data=caches[(e["video"],e["expression"])]
        g={int(f):idx for f,idx in frame_groups(data)}; grouped[(str(e["video"]),str(e["expression"]))]=g; by[str(e["video"])].append(e)
        if e["split"]=="screening": screen_units.extend((str(e["video"]),str(e["expression"]),int(f)) for f in g)
    screen_units.sort(); idxs=np.linspace(0,len(screen_units)-1,min(a.screen_cap,len(screen_units)),dtype=int); chosen={screen_units[int(i)] for i in idxs}
    calibration=[]; screening=[]; audit_keys=set(); duplicate=0; nonfinite=0; missing_assoc=0; reps=[]
    for video,es in by.items():
        bank_meta=build_bank(video,aw,ab); seq=build_seq(video); frame_union=set()
        for e in es:
            g=grouped[(video,str(e["expression"]))]
            if e["split"]=="calibration" or any((video,str(e["expression"]),f) in chosen for f in g): frame_union.update(g)
        for frame in sorted(frame_union):
            obs,om,ot,_,_=state_at(seq,frame)
            with torch.inference_mode(): enc=model.encode_observations(obs.to(a.device),om.to(a.device),ot.to(a.device))
            valid=valid_tracks(seq,frame)
            for e in es:
                split=e["split"]; k=(video,str(e["expression"])) ; g=grouped[k]
                if frame not in g or (split=="screening" and (video,str(e["expression"]),frame) not in chosen): continue
                qh=hidden[int(e["query_index"])].to(a.device); qm=mask[int(e["query_index"])].to(a.device)
                with torch.inference_mode(): out=model.forward_encoded(enc,enc[1],qh,qm)
                current=out["current_membership_logits"].float().cpu().numpy(); by_track={int(seq["track_ids"][ti]):float(current[i]) for i,ti in enumerate(valid)}
                data=caches[(e["video"],e["expression"])] ; rows=g[frame]; tr=data["track_id"][rows].astype(np.int64); src=data["source"][rows].astype(np.int64)
                raw=np.asarray([by_track.get(int(t),-20.) for t in tr],np.float32); assoc=np.asarray([bank_meta["lookup"].get((int(frame),int(t),int(s)),0.) for t,s in zip(tr,src)],np.float32); missing_assoc+=int(np.sum([ (int(frame),int(t),int(s)) not in bank_meta["lookup"] for t,s in zip(tr,src)]))
                gate=assoc>=0.; fused=raw+0.25*np.clip(assoc,-2,2)*gate
                label=data["label"][rows].astype(bool); r={"video":video,"query_index":int(e["query_index"]),"frame":int(frame),"raw":raw,"assoc":assoc,"fused":fused,"label":label,"track_id":tr}
                for key in range(len(tr)):
                    ak=(video,int(e["query_index"]),int(frame),int(tr[key]),int(frame))
                    duplicate+=int(ak in audit_keys); audit_keys.add(ak)
                nonfinite+=int(not (np.isfinite(raw).all() and np.isfinite(assoc).all() and np.isfinite(fused).all()))
                (calibration if split=="calibration" else screening).append(r)
                if len(reps)<30 and label.any():
                    hard=int(np.argmax(np.where(label,-np.inf,fused))) if (~label).any() else 0; pos=int(np.flatnonzero(label)[np.argmin(fused[label])]); reps.append({"video":video,"query_index":int(e["query_index"]),"frame":int(frame),"positive_track":int(tr[pos]),"hard_track":int(tr[hard]),"positive_fused":float(fused[pos]),"hard_fused":float(fused[hard]),"positive_assoc":float(assoc[pos]),"hard_assoc":float(assoc[hard])})
        del bank_meta, seq
    raw_thr=threshold_calibration(calibration,"raw"); fusion_thr=threshold_calibration(calibration,"fused")
    result={"l29_membership":metrics(screening,"raw",raw_thr["threshold"]),"bounded_fusion":metrics(screening,"fused",fusion_thr["threshold"]),"association_diagnostic":metrics(screening,"assoc",0.)}
    payload={"format":"locatemot-l31-bounded-identity-fusion-v1","checkpoint_membership":str(MEMBERSHIP_CHECKPOINT.resolve()),"checkpoint_association":str(ASSOC_CHECKPOINT.resolve()),"manifest":str((ROOT/"outputs/l19/protocol/kitti_fast_eval_manifest.json").resolve()),"calibration_only":{"frame_units":len(calibration),"threshold_raw":raw_thr,"threshold_fused":fusion_thr,"fusion_weight":0.25,"association_gate":"score>=0; no source/pool conditional","association_clip":[-2.,2.]},"screening":{"frame_units":len(screening),"query_count":96,"gt_used_for_selection":False},"audit":{"rows":sum(len(r["label"]) for r in calibration+screening),"duplicate_keys":duplicate,"nonfinite":nonfinite,"missing_association_rows":missing_assoc,"semantic_inputs_excluded":["pool_id","source_id","group_id","state_key"]},"strategies":result,"representative_hard_cases":reps}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)+"\n"); audit=ROOT/"outputs/l31/audit/fusion_contract.json"; audit.parent.mkdir(parents=True,exist_ok=True); audit.write_text(json.dumps({"format":"locatemot-l31-fusion-contract-v1","provenance":payload["calibration_only"],"check":payload["audit"],"screening_gt_used_for_selection":False,"representative_hard_cases":reps},indent=2)+"\n"); print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__": main()
