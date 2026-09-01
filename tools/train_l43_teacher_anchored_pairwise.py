#!/usr/bin/env python3
"""Train-only L43 teacher-anchored pairwise residual smoke."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l43_teacher_anchored_pairwise import L43TeacherAnchoredPairwiseResidual
from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
from tools.train_l26_crossmodal_adapter import FAST, V5
from tools.train_l28_track_set_decoder import state_at
from tools.train_l42_current_frame_grounding import (StreamingCropPatchEncoder, load_bank,
                                                      make_units, numeric_for)

# Importing the L42 helpers is intentional: it reuses the already audited
# train-only query/box/crop contract and does not read or write its artifacts.
from tools.train_l42_current_frame_grounding import load_queries

L29 = ROOT / "outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"
L28 = ROOT / "outputs/l28/track_sequence_bank_final"
TRAIN_VIDEOS = ("0000", "0001", "0002", "0003", "0006", "0007", "0008", "0009", "0010", "0012", "0014", "0015", "0016", "0017", "0020")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def balanced_bce(s, y):
    if not len(s): return s.new_zeros(())
    p, n = y.bool(), ~y.bool(); parts=[]
    if p.any(): parts.append(F.binary_cross_entropy_with_logits(s[p], y[p].float()))
    if n.any(): parts.append(F.binary_cross_entropy_with_logits(s[n], y[n].float()))
    return torch.stack(parts).mean()


def pair_stats(final, teacher, y, hard):
    pos=torch.nonzero(y).flatten(); neg=hard
    if not len(pos) or not len(neg): return {"pairs":0,"teacher_correct":0,"teacher_error":0,"teacher_correct_flips":0,"teacher_errors_corrected":0}
    td=teacher[pos,None]-teacher[None,neg]; sd=final[pos,None]-final[None,neg]; correct=td>0; error=~correct
    return {"pairs":int(td.numel()),"teacher_correct":int(correct.sum()),"teacher_error":int(error.sum()),"teacher_correct_flips":int((correct&(sd<0)).sum()),"teacher_errors_corrected":int((error&(sd>0)).sum())}


def teacher_for_l43(l29, cache, q, frame, bank, rows, hidden, text_mask, device):
    obs, om, ot, _, _ = state_at(cache, int(frame), history=8)
    ptr, frames = cache["track_ptr"].numpy(), cache["obs_frame"].numpy()
    valid = [i for i in range(len(ptr) - 1)
             if np.any(frames[int(ptr[i]):int(ptr[i + 1])] <= int(frame))]
    with torch.inference_mode():
        encoded = l29.encode_observations(obs.to(device), om.to(device), ot.to(device))
        out = l29.forward_encoded(encoded, encoded[1], hidden[q["text_index"]].to(device), text_mask[q["text_index"]].to(device))
    ids = cache["track_ids"][torch.as_tensor(valid)].tolist()
    values = {int(t): float(s) for t, s in zip(ids, out["current_membership_logits"].float().cpu().tolist())}
    return torch.tensor([values.get(int(bank["track"][r]), -20.0) for r in rows], dtype=torch.float32)


def unit_loss(model, unit, hidden, text_mask, device):
    patches=unit["patch"].to(device).float(); numeric=unit["numeric"].to(device).float(); teacher=unit["teacher"].to(device).float(); n=patches.shape[0]
    y=torch.as_tensor(unit["y"],dtype=torch.bool,device=device); cm=torch.ones(n,dtype=torch.bool,device=device); qh=hidden[unit["query"]["text_index"]].to(device); qm=text_mask[unit["query"]["text_index"]].to(device)
    out=model(patches,qh,numeric,teacher,cm,qm); final=out["final_score"]; final.retain_grad(); pos=torch.nonzero(y).flatten(); neg=torch.nonzero(~y).flatten(); pre=neg[torch.argsort(unit["objectness"].to(device)[neg],descending=True)[:min(24,len(neg))]] if len(neg) else neg
    # Pair computation itself sees the complete set.  The loss focuses on the
    # deterministic teacher/objectness hard subset for bounded smoke cost.
    hard=pre
    zero=final.new_zeros(()); stats=pair_stats(final,teacher,y,hard)
    if len(pos) and len(hard):
        td=teacher[pos,None]-teacher[None,hard]; sd=final[pos,None]-final[None,hard]; correct=td>0; error=~correct
        gt_pair=F.softplus(.1-sd).mean(); preserve=F.relu(-sd[correct]).mean() if correct.any() else zero; correction=F.softplus(.1-sd[error]).mean() if error.any() else zero
    else: gt_pair=preserve=correction=zero
    listwise=torch.logsumexp(final,0)-torch.logsumexp(final[pos],0) if len(pos) else zero
    all_pos=F.binary_cross_entropy_with_logits(final[pos],torch.ones_like(final[pos])) if len(pos) else zero
    inactive=out["delta_score"].pow(2).mean() if not len(pos) else zero
    residual_l2=out["residual"][out["pair_valid"]].pow(2).mean() if out["pair_valid"].any() else zero
    zero_mean=out["delta_score"].mean().pow(2)
    bce=.1*balanced_bce(final,y)
    total=gt_pair+1.0*preserve+.5*correction+.5*listwise+.5*all_pos+inactive+bce+.1*residual_l2+.1*zero_mean
    part={"total":float(total.detach()),"gt_pairwise":float(gt_pair.detach()),"teacher_order_preservation":float(preserve.detach()),"teacher_error_correction":float(correction.detach()),"multi_positive_listwise":float(listwise.detach()),"all_positive":float(all_pos.detach()),"inactive_aggregate":float(inactive.detach()),"residual_l2":float(residual_l2.detach()),"zero_mean":float(zero_mean.detach()),"balanced_bce_aux":float(bce.detach()),"residual_max_abs":float(out["residual"].detach().abs().max()),"delta_mean":float(out["delta_score"].detach().mean()),"delta_max_abs":float(out["delta_score"].detach().abs().max()),**stats,"positive_count":int(y.sum()),"negative_count":int((~y).sum())}
    return total,part,final,y,hard,out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out-root",required=True); ap.add_argument("--steps",type=int,default=100); ap.add_argument("--seed",type=int,default=20260829); ap.add_argument("--device",default="cuda:0"); args=ap.parse_args(); assert Path.cwd().resolve()==ROOT
    out=Path(args.out_root); out=out if out.is_absolute() else ROOT/out
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); torch.manual_seed(args.seed); np.random.seed(args.seed)
    device=torch.device(args.device); queries=load_queries(); banks={v:load_bank(v) for v in TRAIN_VIDEOS}; text=torch.load(V5/"text_tokens.pt",map_location="cpu",weights_only=False); hidden=text["token_hidden"].float(); text_mask=text["attention_mask"].bool(); del text; caches={v:torch.load(L28/f"{v}.pt",map_location="cpu",weights_only=False) for v in TRAIN_VIDEOS}
    l29=L29FrameMembershipSetDecoder().to(device); l29.load_state_dict(torch.load(L29,map_location=device,weights_only=False)["model"]); l29.eval(); metas=make_units(queries,banks,32); enc=StreamingCropPatchEncoder(device); units=[]
    for q,fi,y in metas:
        b=banks[q["video"]]; begin,end=int(b["ptr"][fi]),int(b["ptr"][fi+1]); rows=list(range(begin,end)); frame=int(b["frame_ids"][fi]); units.append({"query":q,"frame":frame,"y":y,"objectness":b["objectness"][rows].cpu(),"numeric":numeric_for(b,rows).cpu(),"patch":enc.encode(q["video"],b,rows),"teacher":teacher_for_l43(l29,caches[q["video"]],q,frame,b,rows,hidden,text_mask,device)})
    del enc,l29,caches,banks
    model=L43TeacherAnchoredPairwiseResidual(hidden=128,heads=4,layers=1).to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4); rng=np.random.default_rng(args.seed); trace=[]; grads=[]; pgr=[]; hgr=[]; flips=[]; corrections=[]; start=time.time(); model.train()
    for _ in range(args.steps):
        u=units[int(rng.integers(len(units)))]; loss,part,final,y,hard,_=unit_loss(model,u,hidden,text_mask,device); opt.zero_grad(set_to_none=True); loss.backward(); grads.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.0))); g=final.grad.detach().abs(); pi=torch.nonzero(y).flatten(); pgr.append(float((g[pi]>1e-10).float().mean()) if len(pi) else np.nan); hgr.append(float((g[hard]>1e-10).float().mean()) if len(hard) else np.nan); flips.append(part["teacher_correct_flips"]); corrections.append(part["teacher_errors_corrected"]); opt.step(); trace.append({k:part[k] for k in ("total","gt_pairwise","teacher_order_preservation","teacher_error_correction","multi_positive_listwise","all_positive","inactive_aggregate","residual_l2","zero_mean","balanced_bce_aux","residual_max_abs","delta_mean","delta_max_abs")})
    ck=out/f"checkpoint_l43_teacher_anchored_step{args.steps}.pt"; payload={"format":"locatemot-l43-teacher-anchored-pairwise-v1","stage":"train-only-smoke","seed":args.seed,"steps":args.steps,"device":str(device),"train_video_count":len(TRAIN_VIDEOS),"train_query_count":len(queries),"sampled_unit_count":len(units),"sample_categories":{"multi_positive":sum(int(x[2].sum()>1) for x in metas),"single_positive":sum(int(x[2].sum()==1) for x in metas),"inactive":sum(int(not x[2].any()) for x in metas)},"screening_gt_used_for_fit":False,"semantic_inputs_excluded":["source_id","pool_id","group_id","state_key"],"token_level_alignment_verified":False,"motion_language_decomposition":"not claimed; no verified motion-language mask","model_config":model.config,"teacher":{"checkpoint":str(L29.resolve()),"sha256":sha(L29),"final_score":"teacher + mean pair residual","residual_bound":0.05,"teacher_control":"s_i=m_i"},"pair_contract":{"antisymmetric":True,"residual_formula":"0.025*(tanh(g_ij)-tanh(g_ji))","hard_subset":"objectness top-24 for loss; complete current set used by forward","teacher_correct_preservation_weight":1.0,"teacher_error_correction_weight":0.5,"multi_positive_all_positive":True,"inactive_only_aggregate":True},"loss_mean":{k:float(np.nanmean([x[k] for x in trace])) for k in trace[0]},"gradient_norm":{"mean":float(np.mean(grads)),"max":float(np.max(grads)),"nonzero_steps":int(np.count_nonzero(np.asarray(grads)>0))},"gradient_audit":{"positive_nonzero_fraction_on_positive_units":float(np.nanmean(pgr)),"hard_nonzero_fraction":float(np.nanmean(hgr)),"positive_unit_steps":int(np.count_nonzero(np.isfinite(pgr))),"multi_positive_loss_path":True},"rank_diagnostics":{"teacher_correct_flips_in_smoke":int(sum(flips)),"teacher_error_corrections_in_smoke":int(sum(corrections))},"elapsed_sec":time.time()-start}
    torch.save({"model":model.state_dict(),"config":payload},ck); payload["checkpoint"]=str(ck.resolve()); reload=L43TeacherAnchoredPairwiseResidual(**model.config); reload.load_state_dict(torch.load(ck,map_location="cpu",weights_only=False)["model"]); payload["checkpoint_reload"]=True; (out/f"metrics_l43_smoke{args.steps}.json").write_text(json.dumps(payload,indent=2)+"\n"); (out/"loss_trace.json").write_text(json.dumps(trace,indent=2)+"\n"); (out/"config.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps(payload,indent=2),flush=True)


if __name__=="__main__": main()
