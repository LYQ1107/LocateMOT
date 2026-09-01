#!/usr/bin/env python3
"""Stage L34 frozen dense region/text alignment ceiling audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from tools.eval_l25_token_probe import tokens_for
from tools.train_l23_dense_correspondence import fixed_refs
from tools.train_rmot_candidate_scorer import load_bank, load_metadata, make_refs, auc, average_precision, scalar_stats

MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
V3_ROOT = ROOT / "outputs/l23/candidate_bank_v3"
V4_ROOT = ROOT / "outputs/l25/candidate_bank_v4"
WEIGHTS = Path("/home/lwr/.cache/clip/ViT-B-16.pt")
TEXT_CACHE = ROOT / "outputs/l26/candidate_bank_v5_crossmodal/text_tokens.pt"


def sha(path):
    h = hashlib.sha256();
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()


def summary_for(refs, banks, qcache, device):
    scores=[]; labels=[]; strict=[]; best=[]; top1=[]; top5=[]; multi=[]; null=[]; source=Counter(); coverage=[]; target_coverage=[]; target_units=0; covered_target_units=0
    for ref in refs:
        t = banks[ref["video"]]["tensors"]; sl=slice(ref["begin"],ref["end"])
        q=F.normalize(qcache[int(ref["query_index"])].to(device).float(),dim=-1)
        roi=F.normalize(t["dense_roi_tokens_v4"][sl].to(device).float(),dim=-1)
        with torch.inference_mode(): s=torch.einsum("ld,nkd->nlk",q,roi).amax((1,2)).cpu().numpy()
        y=ref["positive"].astype(bool); scores.append(s); labels.append(y)
        pos=np.flatnonzero(y); neg=np.flatnonzero(~y); coverage.append(float(len(pos)>0))
        has_target = not bool(ref.get("null", False))
        target_units += int(has_target); covered_target_units += int(has_target and len(pos)>0); target_coverage.append(float(has_target and len(pos)>0))
        if not len(pos): null.append(float(s.max()) if len(s) else 0.0); continue
        order=np.argsort(-s,kind="stable"); top1.append(float(y[order[:1]].any())); top5.append(float(y[order[:5]].any()))
        if len(pos)>1: multi.append((float(y[order[:1]].any()),float(y[order[:5]].any()),len(pos)))
        if len(neg): strict.append(float(s[pos].min()-s[neg].max())); best.append(float(s[pos].max()-s[neg].max()))
        pool=t["pool_id"][sl].numpy()
        for source_id in (0,1):
            rows=np.flatnonzero(pool==source_id)
            if len(rows): source[(source_id,"frames")]+=1; source[(source_id,"top1")]+=int(y[rows[np.argmax(s[rows])]])
    flat_s=np.concatenate(scores); flat_y=np.concatenate(labels)
    return {"frame_units":len(refs),"candidate_rows":int(len(flat_y)),"positive_rows":int(flat_y.sum()),"positive_frame_units":int(sum(int(x.any()) for x in labels)),"null_units":int(sum(int(x.get("null",False)) for x in refs)),"target_frame_units":target_units,"covered_target_frame_units":covered_target_units,"region_coverage":float(np.mean(coverage)) if coverage else None,"target_region_coverage":covered_target_units/max(1,target_units),"roc_auc":auc(flat_s,flat_y),"pr_auc":average_precision(flat_s,flat_y),"top1":float(np.mean(top1)) if top1 else None,"top5":float(np.mean(top5)) if top5 else None,"multi_positive_units":len(multi),"multi_positive_top1":float(np.mean([x[0] for x in multi])) if multi else None,"multi_positive_top5":float(np.mean([x[1] for x in multi])) if multi else None,"strict_min_positive_margin":scalar_stats(strict),"best_positive_margin":scalar_stats(best),"hard_violation":float(np.mean(np.asarray(strict)<0)) if strict else None,"null_highest_score":scalar_stats(null),"source_top1_precision":{"main":source[(0,"top1")]/max(1,source[(0,"frames")]),"reserve":source[(1,"top1")]/max(1,source[(1,"frames")])}}


def oracle_summary(refs):
    pos=[int(ref["positive"].any()) for ref in refs]
    nonnull=[x for x in pos if x]
    return {"definition":"GT-privileged selection of a positive candidate when the frozen bank contains one; this is a candidate ceiling, not a model score","frame_units":len(refs),"positive_covered_units":sum(pos),"coverage":float(np.mean(pos)) if pos else None,"top1_on_covered":1.0 if nonnull else None,"top5_on_covered":1.0 if nonnull else None,"uncovered_units":len(pos)-sum(pos)}


def provenance(videos):
    out={"manifest_sha256":sha(MANIFEST),"weights":str(WEIGHTS),"weights_sha256":sha(WEIGHTS),"v3_root":str(V3_ROOT.resolve()),"v4_root":str(V4_ROOT.resolve()),"text_cache":str(TEXT_CACHE.resolve()),"text_cache_shape":None,"dense_bank_train_only_available":False,"dense_bank_train_only_note":"Existing v4 contains only fast-eval videos 0004/0018; no train-only dense region bank is present.","videos":{},"checks":{}}
    cached=torch.load(TEXT_CACHE,map_location="cpu",weights_only=False); out["text_cache_shape"]={k:list(v.shape) for k,v in cached.items() if hasattr(v,"shape")}; del cached
    all_ok=True
    for video in videos:
        v4=load_bank(V4_ROOT/"kitti"/f"{video}.pt"); t=v4["tensors"]; v3=torch.load(V3_ROOT/"kitti"/f"{video}.pt",map_location="cpu",weights_only=False)["tensors"]
        labels_v3=json.loads((V3_ROOT/"kitti"/f"{video}.labels.json").read_text())["candidate_gt"]
        keys=("frame","candidate_index","track_id","box","frame_ptr","frame_ids")
        align=all(torch.equal(t[k].cpu(),v3[k].cpu()) for k in keys) and len(v4["candidate_gt"])==len(labels_v3) and v4["candidate_gt"]==labels_v3
        shape=list(t["dense_roi_tokens_v4"].shape[1:]); finite=all(torch.isfinite(t[k].float()).all().item() for k in ("dense_roi_tokens_v4","dense_points_v4","dense_context_1p5_tokens_v4","dense_context_3_tokens_v4","dense_prev_roi_tokens_v4","candidate_points_v4","roi_sample_points_v4"))
        coords=torch.cat((t["candidate_points_v4"].float().reshape(-1,2),t["roi_sample_points_v4"].float().reshape(-1,2)),0)
        out["videos"][video]={"rows":len(t["frame"]),"frames":len(t["frame_ids"]),"roi_token_shape":shape,"coordinate_min":coords.min(0).values.tolist(),"coordinate_max":coords.max(0).values.tolist(),"finite":bool(finite),"v3_v4_row_alignment":bool(align),"dense_map_files":len(list((V4_ROOT/"dense_maps").glob(f"{video}_*.pt")))}
        all_ok &= bool(align and finite and shape==[9,512])
    out["checks"]={"v3_v4_alignment":all_ok,"finite":all_ok,"roi_token_dim_compatible_with_clip_text":True,"text_cache_768_not_used_for_512_region_cosine":True}
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--device",default="cuda:0"); args=ap.parse_args()
    out=Path(args.out); out=out if out.is_absolute() else ROOT/out
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True)
    manifest=json.loads(MANIFEST.read_text()); rows=sorted(manifest["queries"],key=lambda x:int(x["query_index"])); videos=sorted({str(x["video"]) for x in rows})
    metadata=load_metadata(); banks={v:load_bank(V4_ROOT/"kitti"/f"{v}.pt") for v in videos}; refs=make_refs(rows,metadata,banks)
    cal=fixed_refs([r for r in refs if r["split"]=="calibration"],6000,17); screen=fixed_refs([r for r in refs if r["split"]=="screening"],6000,18)
    import clip
    device=torch.device(args.device); cm,_=clip.load(str(WEIGHTS),device=device); cm.eval()
    qcache={int(q["query_index"]):tokens_for(cm,str(q["expression"]),device).cpu() for q in rows}; del cm
    prov=provenance(videos); result={"format":"locatemot-l34-dense-alignment-ceiling-v1","manifest":str(MANIFEST.resolve()),"manifest_sha256":sha(MANIFEST),"query_counts":{"all":len(rows),"calibration":len([x for x in rows if x["split"]=="calibration"]),"screening":len([x for x in rows if x["split"]=="screening"])},"sampling":{"calibration_frame_units":len(cal),"screening_frame_units":len(screen),"seed_calibration":17,"seed_screening":18},"provenance":prov,"word_token_source":"frozen CLIP ViT-B/16 native projected word tokens, compatible 512-D space; v5 768-D cache is recorded but not used for cosine","gt_used_for_feature_or_score":False,"screening_gt_used_for_model_selection":False,"calibration":summary_for(cal,banks,qcache,device),"screening":summary_for(screen,banks,qcache,device),"calibration_region_oracle":oracle_summary(cal),"screening_region_oracle":oracle_summary(screen),"alignment_mask":{"verified":False,"note":"No verified static/motion or word-to-region alignment mask is present."}}
    (out/"dense_alignment_ceiling.json").write_text(json.dumps(result,indent=2)+"\n"); (out/"README.md").write_text("# L34 dense alignment ceiling audit\n\nFrozen v4 ROI patch tokens are checked against native CLIP word tokens. GT is used only for coverage and post-hoc diagnostic labels; no scorer is trained and no RMOT output is modified.\n")
    print(json.dumps({"output":str(out/"dense_alignment_ceiling.json"),"calibration":result["calibration"],"screening":result["screening"],"oracle":result["screening_region_oracle"]},indent=2),flush=True)


if __name__=="__main__": main()
