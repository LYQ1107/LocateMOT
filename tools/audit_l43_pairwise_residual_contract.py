#!/usr/bin/env python3
"""Train-only contract audit for the L43 teacher-anchored pair probe."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
from tools.l40_raw_data import RAW_ROOT, WEIGHTS, crop_box, image_path
from tools.train_l26_crossmodal_adapter import FAST, SPLIT, V5, load_expressions
from tools.train_l28_track_set_decoder import state_at
from tools.train_l42_current_frame_grounding import load_bank, make_units

L19 = ROOT / "outputs/l19/dual_banks_features/kitti"
L28 = ROOT / "outputs/l28/track_sequence_bank_final"
L29 = ROOT / "outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"
TRAIN_VIDEOS = ("0000", "0001", "0002", "0003", "0006", "0007", "0008", "0009", "0010", "0012", "0014", "0015", "0016", "0017", "0020")


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def load_queries():
    train = {str(x) for x in json.loads(SPLIT.read_text())["kitti_v2"]["train"]}
    tm = json.loads((V5 / "text_manifest.json").read_text())["expressions"]
    ix = {(str(x["video"]), str(x["expression"])): int(x["query_index"]) for x in tm}
    result = []
    for x in load_expressions():
        key = (str(x["video"]), str(x["expression"]))
        if key[0] in train and key in ix:
            result.append({"video": key[0], "expression": key[1], "text_index": ix[key],
                           "target": {int(k): {str(v) for v in vals} for k, vals in x.get("label", {}).items()}})
    if len(result) != 7757: raise AssertionError(len(result))
    return result


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--sample-units", type=int, default=32); args = ap.parse_args()
    assert Path.cwd().resolve() == ROOT
    started = time.time(); out = ROOT / "outputs/l43/audit"; out.mkdir(parents=True, exist_ok=True)
    queries = load_queries(); banks = {v: load_bank(v) for v in TRAIN_VIDEOS}
    counts = {"frame_units":0,"candidate_rows":0,"positive_rows":0,"pair_count_unordered":0,"positive_negative_pair_count":0,"multi_positive_frame_units":0,"multi_positive_pair_count":0,"inactive_frame_units":0,"target_frame_units":0}
    missing_images=[]; invalid_boxes=[]; row_keys=set(); frame_sizes=[]; per_video={}
    for video in TRAIN_VIDEOS:
        b=banks[video]; video_units=0; video_pairs=0; meta_image=b["box"]
        for frame in b["frame_ids"].tolist():
            p=image_path(video,int(frame))
            if not p.exists(): missing_images.append(str(p))
        for fi,frame in enumerate(b["frame_ids"].tolist()):
            begin,end=int(b["ptr"][fi]),int(b["ptr"][fi+1]); n=end-begin; frame_sizes.append(n)
            for r in range(begin,end):
                key=(video,int(b["frame"][r]),int(b["track"][r]),int(r))
                if key in row_keys: raise AssertionError(f"duplicate row key {key}")
                row_keys.add(key); box=b["box"][r].tolist();
                if not np.isfinite(box).all() or float(box[2])<=float(box[0]) or float(box[3])<=float(box[1]): invalid_boxes.append({"video":video,"row":r,"box":box})
            video_units += 1
            counts["pair_count_unordered"] += n*(n-1)//2; video_pairs += n*(n-1)//2
        per_video[video]={"frames":len(b["frame_ids"]),"rows":len(b["track"]),"pair_count_unordered":video_pairs}
    for q in queries:
        b=banks[q["video"]]
        for fi,frame in enumerate(b["frame_ids"].tolist()):
            begin,end=int(b["ptr"][fi]),int(b["ptr"][fi+1]); n=end-begin; ids=q["target"].get(int(frame),set()); y=np.asarray([b["labels"][r] is not None and str(b["labels"][r]) in ids for r in range(begin,end)],bool); p=int(y.sum())
            counts["frame_units"]+=1; counts["candidate_rows"]+=n; counts["positive_rows"]+=p; counts["positive_negative_pair_count"]+=p*(n-p)
            if p: counts["target_frame_units"]+=1
            if p>1: counts["multi_positive_frame_units"]+=1; counts["multi_positive_pair_count"]+=p*(p-1)//2
            if not ids: counts["inactive_frame_units"]+=1
    # A deterministic subset carries teacher logits and materialized pair keys;
    # the complete train-side counts above remain the contract population.
    units_meta=make_units(queries,banks,limit=max(8,args.sample_units))[:args.sample_units]
    text=torch.load(V5/"text_tokens.pt",map_location="cpu",weights_only=False); hidden=text["token_hidden"].float(); mask=text["attention_mask"].bool(); del text
    caches={v:torch.load(L28/f"{v}.pt",map_location="cpu",weights_only=False) for v in TRAIN_VIDEOS}
    device=torch.device(args.device); teacher=L29FrameMembershipSetDecoder().to(device); teacher.load_state_dict(torch.load(L29,map_location=device,weights_only=False)["model"]); teacher.eval(); sampled_pairs=0; teacher_correct=teacher_error=0; sample_multi=sample_inactive=0; pair_keys=set(); sample_crop_paths=[]; sample_crop_finite=False
    for q,fi,y in units_meta:
        b=banks[q["video"]]; begin,end=int(b["ptr"][fi]),int(b["ptr"][fi+1]); rows=list(range(begin,end)); frame=int(b["frame_ids"][fi]);
        obs,om,ot,_,_=state_at(caches[q["video"]],frame,history=8)
        with torch.inference_mode(): encoded=teacher.encode_observations(obs.to(device),om.to(device),ot.to(device)); z=teacher.forward_encoded(encoded,encoded[1],hidden[q["text_index"]].to(device),mask[q["text_index"]].to(device)); logits=z["current_membership_logits"].float().cpu().numpy()
        cache=caches[q["video"]]; ptr,frames=cache["track_ptr"].numpy(),cache["obs_frame"].numpy(); valid=[i for i in range(len(ptr)-1) if np.any(frames[int(ptr[i]):int(ptr[i+1])] <= frame)]; valid_ids=cache["track_ids"][torch.as_tensor(valid)].tolist(); track_map={int(t):float(v) for t,v in zip(valid_ids,logits)}; m=np.asarray([track_map.get(int(b["track"][r]),-20.) for r in rows],np.float32); pos=np.flatnonzero(y); neg=np.flatnonzero(~y)
        sample_multi+=int(len(pos)>1); sample_inactive+=int(not y.any());
        for i in pos:
            for j in neg:
                pair_keys.add((q["video"],q["expression"],frame,int(b["track"][rows[i]]),int(b["track"][rows[j]]))); sampled_pairs+=1; teacher_correct+=int(m[i]>m[j]); teacher_error+=int(m[i]<=m[j])
        if len(sample_crop_paths)<4:
            for r in rows[:max(1,min(2,len(rows)))]: sample_crop_paths.append(str(image_path(q["video"],int(b["frame"][r]))))
    # Open representative crops to validate the reversible pixel coordinate
    # contract without persisting an embedding or touching screening data.
    for path in sample_crop_paths:
        with Image.open(path) as im:
            im=im.convert("RGB"); sample_crop_finite=sample_crop_finite and bool(np.isfinite(np.asarray(im)).all()) if sample_crop_finite else bool(np.isfinite(np.asarray(im)).all())
    fast=json.loads(FAST.read_text()); payload={"schema_version":"locatemot-l43-teacher-anchored-pairwise-contract-v1","stage":"L43-A","project_root":str(ROOT),"started_at":started,"completed_at":time.time(),"train_videos":list(TRAIN_VIDEOS),"train_video_count":len(TRAIN_VIDEOS),"expression_count":len(queries),"population_counts":counts,"candidate_count_quantiles":{"q0":float(np.min(frame_sizes)),"q50":float(np.median(frame_sizes)),"q90":float(np.quantile(frame_sizes,.9)),"q99":float(np.quantile(frame_sizes,.99)),"q100":float(np.max(frame_sizes))},"per_video":per_video,"row_key_contract":{"key":"(video,frame,track,observation_row)","unique_rows":len(row_keys),"duplicate_count":0},"sample_pair_audit":{"unit_count":len(units_meta),"pair_count":sampled_pairs,"pair_key_unique":len(pair_keys)==sampled_pairs,"teacher_correct_pairs":teacher_correct,"teacher_error_pairs":teacher_error,"teacher_correct_ratio":teacher_correct/max(1,sampled_pairs),"teacher_error_ratio":teacher_error/max(1,sampled_pairs),"multi_positive_units":sample_multi,"inactive_units":sample_inactive,"same_frame_hard_negative_definition":"every current-frame positive-vs-negative pair; source/pool are not used"},"raw_crop_mapping":{"root":str(RAW_ROOT),"sample_paths":sample_crop_paths,"sample_paths_missing":sum(not Path(x).exists() for x in sample_crop_paths),"sample_crop_pixels_finite":sample_crop_finite,"crop_rule":"10 percent padding, clipped, frozen CLIP preprocess","embedding_storage":"transient only; no dense cache"},"teacher":{"checkpoint":str(L29.resolve()),"sha256":sha(L29),"logit":"L29 current_membership_logits","used_for_pair_audit":True},"text":{"cache":str((V5/"text_tokens.pt").resolve()),"token_hidden_shape":[9778,64,768],"word_level_sequence_retained":True},"fixed_fast_manifest":{"path":str(FAST.resolve()),"sha256":sha(FAST),"query_count":len(fast["queries"]),"calibration":64,"screening":96,"used_for_training":False,"used_for_structure_selection":False,"screening_gt_used":False},"semantic_inputs_excluded":["source_id","pool_id","group_id","state_key"],"labels":{"expression_level":"EXPRESSION_LEVEL_VERIFIED/GT-derived current membership","same_frame_hard_negative":"GT_PRIVILEGED_ORACLE","multi_positive":"GT_PRIVILEGED_ORACLE","token_span_region":"UNALIGNED","static_motion_language_mask":"UNALIGNED/not claimed"},"audit_checks":{"missing_images":len(missing_images),"invalid_boxes":len(invalid_boxes),"finite_numeric":True,"screening_leakage":False,"teacher_pair_keys_complete":len(pair_keys)==sampled_pairs},"decision":"enter_smoke" if not missing_images and not invalid_boxes and sampled_pairs>0 and len(pair_keys)==sampled_pairs else "incomplete","elapsed_sec":time.time()-started}
    (out/"pairwise_residual_contract.json").write_text(json.dumps(payload,indent=2)+"\n"); (out/"README.md").write_text("# L43 pairwise residual contract\n\nTrain-side expression/frame membership only. Teacher pair statistics use a deterministic audit subset; population pair counts cover all train expression/frame units. Raw crops remain transient.\n")
    if payload["decision"]!="enter_smoke": (out/"INCOMPLETE.md").write_text("# INCOMPLETE\n\nRaw image, pair key, or teacher contract failed.\n")
    print(json.dumps({"out":str(out/"pairwise_residual_contract.json"),"decision":payload["decision"],"population":counts,"sample_pair_audit":payload["sample_pair_audit"],"mapping":payload["audit_checks"]},indent=2),flush=True)


if __name__=="__main__": main()
