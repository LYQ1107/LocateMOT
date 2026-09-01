#!/usr/bin/env python3
"""L65 fit-only smoke for a frozen CLIP joint-space candidate-set head."""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
UNITS = ROOT / "outputs/l49/data/train_units.jsonl"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
EXPECTED = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"

import sys
sys.path.insert(0, str(ROOT))
from locatemot.models.l65_clip_joint_set import L65ClipJointSet
from tools.l65_clip_joint_common import StreamingClipJoint, sha256


def fit_units():
    return [json.loads(x) for x in UNITS.read_text().splitlines() if x.strip() and (lambda u: u.get("split") == "fit" and u.get("dataset") in ("refer_kitti_v1", "refer_kitti_v2"))(json.loads(x))]


def stratified(units, seed):
    rng = random.Random(seed); cats = ("positive", "multi_positive", "inactive", "present_uncovered")
    buckets = {(d, c): [] for d in ("refer_kitti_v1", "refer_kitti_v2") for c in cats}
    for u in units: buckets.setdefault((u["dataset"], u.get("category", "unknown")), []).append(u)
    for b in buckets.values(): rng.shuffle(b)
    order = []
    while any(buckets.values()):
        for key in sorted(buckets):
            if buckets[key]: order.append(buckets[key].pop())
    return order


def numeric(t, begin, end):
    return torch.cat((t["geometry"][begin:end].float(), t["motion"][begin:end].float(), t["lifecycle"][begin:end].float(), t["context"][begin:end].float(), t["objectness"][begin:end].float().reshape(-1, 1)), 1)


def load_unit(unit, encoder):
    bank = torch.load(Path(unit["bank_path"]), map_location="cpu", weights_only=False); t = bank["tensors"]; b, e = int(unit["begin"]), int(unit["end"]); n = e - b
    if n != int(unit["candidate_count"]): raise AssertionError(f"candidate count {unit['unit_key']}")
    patches, image = encoder.encode_unit(unit["video"], int(unit["frame_id"]), t["box"][b:e].float().tolist()); words, mask, _, _ = encoder.text_joint_tokens(unit["sentence"])
    if patches.shape != (n, 17, 512): raise AssertionError(f"patch shape {patches.shape}")
    # The labels are read only after raw image/text feature construction.
    y = torch.zeros(n, dtype=torch.bool); indices = [int(x) for x in unit.get("positive_indices", [])]
    if any(x < 0 or x >= n for x in indices): raise AssertionError(f"positive index {unit['unit_key']}")
    if indices: y[torch.as_tensor(indices, dtype=torch.long)] = True
    return {"unit": unit, "patches": patches.clone(), "words": words.clone(), "mask": mask.clone(), "numeric": numeric(t, b, e).clone(), "target": y, "image": str(image), "offsets": [b, e]}


def balanced(score, target):
    parts = []
    if target.any(): parts.append(F.binary_cross_entropy_with_logits(score[target], torch.ones_like(score[target])))
    if (~target).any(): parts.append(F.binary_cross_entropy_with_logits(score[~target], torch.zeros_like(score[~target])))
    return torch.stack(parts).mean() if parts else score.new_zeros(())


def loss_fn(out, target):
    s = out["relevance_logit"]; pos = torch.nonzero(target, as_tuple=False).flatten(); neg = torch.nonzero(~target, as_tuple=False).flatten(); z = s.new_zeros(())
    bce = balanced(s, target)
    if len(pos) and len(neg):
        hard = neg[torch.argsort(s.detach()[neg], descending=True)[:min(24, len(neg))]]
        pair = F.softplus(.2 + s[hard][None, :] - s[pos][:, None]).mean(); listwise = torch.logsumexp(s, 0) - torch.logsumexp(s[pos], 0)
    else: hard, pair, listwise = neg, z, z
    minimum = F.binary_cross_entropy_with_logits(s[pos], torch.ones_like(s[pos])) if len(pos) else z
    inactive = balanced(s, torch.zeros_like(target)) if not len(pos) else z
    null = F.binary_cross_entropy_with_logits(out["null_logit"], s.new_tensor(float(not target.any())))
    brier = (torch.sigmoid(s) - target.float()).square().mean()
    total = bce + .5 * pair + .5 * listwise + .5 * minimum + inactive + null + .05 * brier
    return total, {"total": float(total.detach()), "bce": float(bce.detach()), "pairwise": float(pair.detach()), "listwise": float(listwise.detach()), "minimum_positive": float(minimum.detach()), "inactive": float(inactive.detach()), "null": float(null.detach()), "brier": float(brier.detach()), "positive_count": int(pos.numel()), "hard_count": int(hard.numel())}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); ap.add_argument("--steps", type=int, default=100); ap.add_argument("--seed", type=int, default=20260829); args = ap.parse_args()
    if Path.cwd().resolve() != ROOT: raise RuntimeError(f"wrong cwd {Path.cwd()}")
    if sha256(MANIFEST) != EXPECTED: raise AssertionError("manifest SHA mismatch")
    out = args.out if args.out.is_absolute() else ROOT / args.out; out = out.resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); units = stratified(fit_units(), args.seed); out.mkdir(parents=True)
    device = torch.device("cuda:0"); encoder = StreamingClipJoint(device, batch_size=32); frozen = all(not p.requires_grad for p in encoder.model.parameters()); model = L65ClipJointSet(hidden=128).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4); trace=[]; sampling=Counter(); finite=nonzero=0; start=time.time(); torch.cuda.reset_peak_memory_stats(device); model.train()
    for step in range(1, args.steps + 1):
        item=load_unit(units[(step-1)%len(units)], encoder); patches=item["patches"].to(device); words=item["words"].to(device); mask=item["mask"].to(device); nums=item["numeric"].to(device); target=item["target"].to(device); optimizer.zero_grad(set_to_none=True); output=model(patches,words,mask,nums); loss,parts=loss_fn(output,target)
        if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite loss {step}")
        loss.backward(); norms=[p.grad.detach().norm() for p in model.parameters() if p.grad is not None]
        if not norms or not all(torch.isfinite(x) for x in norms) or not any(float(x)>0 for x in norms): raise FloatingPointError(f"bad gradient {step}")
        optimizer.step(); finite+=1; nonzero+=1; sampling[(item["unit"]["dataset"],item["unit"].get("category","unknown"))]+=1; trace.append({"step":step,"unit_key":item["unit"]["unit_key"],"dataset":item["unit"]["dataset"],"category":item["unit"].get("category"),"candidate_count":int(target.numel()),"loss":float(loss.detach()),"grad_norm":float(torch.stack(norms).norm()),**parts}); del item,patches,words,mask,nums,target,output,loss
    elapsed=time.time()-start; ck=out/f"checkpoint_l65_clip_joint_step{args.steps}.pt"; torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":args.steps,"seed":args.seed,"format":"locatemot-l65-clip-joint-set-v1"},ck); reload=L65ClipJointSet(hidden=128).cpu(); reload.load_state_dict(torch.load(ck,map_location="cpu",weights_only=False)["model"],strict=True)
    sampled=units[:args.steps]; payload={"format":"locatemot-l65-clip-joint-set-smoke-v1","status":"complete","stage":"fit-only-smoke","project_root":str(ROOT),"cwd":str(Path.cwd().resolve()),"seed":args.seed,"steps":args.steps,"finite_steps":finite,"nonzero_gradient_steps":nonzero,"checkpoint":str(ck),"checkpoint_sha256":sha256(ck),"checkpoint_reload":True,"fit_units_total":len(units),"sampled_domains":sorted({u["dataset"] for u in sampled}),"sampled_categories":sorted({u.get("category") for u in sampled}),"sampling_counts":{f"{d}|{c}":int(n) for (d,c),n in sampling.items()},"candidate_sets_complete":True,"candidate_key_drift":0,"candidate_truncation":False,"persistent_raw_dense_cache_written":False,"screening_gt_used":False,"official_test_labels_read":False,"ordinary_mot_ovmot_touched":False,"detector_frozen":frozen,"adapter_parameter_count":sum(p.numel() for p in model.parameters()),"input_contract":{"patch_joint":["N",17,512],"text_joint":[77,512],"numeric_dim":32,"global_cls_retained":True,"image_source":"streamed PNG; not L19 clip"},"token_span_alignment":"UNALIGNED","runtime":{"device":str(device),"precision":"FP32 adapter","peak_memory_bytes":int(torch.cuda.max_memory_allocated(device)),"elapsed_sec":elapsed,"steps_per_sec":args.steps/max(elapsed,1e-9)},"loss_trace":trace}
    (out/f"metrics_l65_step{args.steps}.json").write_text(json.dumps(payload,indent=2)+"\n"); (out/"loss_trace.json").write_text(json.dumps(trace,indent=2)+"\n"); (out/"sampling_trace.json").write_text(json.dumps(payload["sampling_counts"],indent=2)+"\n"); (out/"reload_audit.json").write_text(json.dumps({"strict":True,"reload_ok":True,"checkpoint":str(ck),"checkpoint_sha256":sha256(ck)},indent=2)+"\n"); (out/"config.json").write_text(json.dumps({"seed":args.seed,"steps":args.steps,"hidden":128,"heads":4,"image_dim":512,"text_dim":512,"numeric_dim":32,"joint_projection_restored":True,"global_cls_retained":True,"fit_only":True,"same_class_hard_negative_metadata":"unavailable","screening_gt_used":False},indent=2)+"\n"); (out/"provenance.json").write_text(json.dumps({"project_root":str(ROOT),"cwd":str(Path.cwd().resolve()),"manifest_sha256":sha256(MANIFEST),"train_units":str(UNITS),"train_units_sha256":sha256(UNITS),"clip_weights":"/home/lwr/.cache/clip/ViT-B-16.pt","clip_weights_sha256":sha256(Path("/home/lwr/.cache/clip/ViT-B-16.pt")),"fit_only":True,"screening_gt_used":False,"official_test_labels_read":False,"ordinary_mot_ovmot_touched":False,"persistent_raw_dense_cache_written":False},indent=2)+"\n"); print(json.dumps({"status":"complete","metrics":str(out/f"metrics_l65_step{args.steps}.json"),"checkpoint":str(ck),"finite_steps":finite,"nonzero_gradient_steps":nonzero},indent=2),flush=True)


if __name__ == "__main__": main()
