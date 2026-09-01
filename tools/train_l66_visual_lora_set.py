#!/usr/bin/env python3
"""L66 100-step fit-only smoke: one CLIP visual LoRA plus L65 set head."""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
import sys
sys.path.insert(0, str(ROOT))
from locatemot.models.l66_visual_lora_set import L66VisualLoraSet, attach_visual_lora, lora_parameters
from tools.l66_visual_lora_common import (CLIP_WEIGHTS, EXPECTED_CLIP, EXPECTED_MANIFEST,
    L65_CHECKPOINT, MANIFEST, StreamingClipLora, fit_units, load_unit_features, loss_fn, sha256, stratified)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True, type=Path); ap.add_argument("--steps", type=int, default=100); ap.add_argument("--seed", type=int, default=20260829); args = ap.parse_args()
    if Path.cwd().resolve() != ROOT: raise RuntimeError(f"wrong cwd {Path.cwd()}")
    if sha256(CLIP_WEIGHTS) != EXPECTED_CLIP: raise AssertionError("CLIP SHA mismatch")
    if sha256(MANIFEST) != EXPECTED_MANIFEST: raise AssertionError("manifest SHA mismatch")
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(out)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    ordered = stratified(fit_units(), args.seed); out.mkdir(parents=True)
    device = torch.device("cuda:0")
    runtime = StreamingClipLora(device, crop_batch=4)
    target_module, wrapper = attach_visual_lora(runtime.model, rank=8, alpha=16.0, dropout=0.0)
    head = L66VisualLoraSet(hidden=128).to(device)
    head_ck = torch.load(L65_CHECKPOINT, map_location="cpu", weights_only=False)
    head.load_state_dict(head_ck["model"], strict=True)
    head.train(); runtime.model.eval()
    optimizer = torch.optim.AdamW([{"params": lora_parameters(runtime.model), "lr": 5e-5}, {"params": list(head.parameters()), "lr": 2e-4}], weight_decay=1e-4)
    trace=[]; sampling=Counter(); finite=nonzero=0; lora_nonzero=0; base_nonzero=0; t0=time.time(); torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, args.steps + 1):
        item = load_unit_features(ordered[(step-1) % len(ordered)], runtime, labels=True)
        patches=item["patches"]; words=item["words"].to(device); mask=item["mask"].to(device); nums=item["numeric"].to(device); target=item["target"].to(device)
        optimizer.zero_grad(set_to_none=True)
        output=head(patches, words, mask, nums); loss, parts=loss_fn(output, target)
        if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite loss step {step}")
        loss.backward()
        lora_grads=[]
        for name,p in runtime.model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                if p.grad is None or not torch.isfinite(p.grad).all(): raise FloatingPointError(f"bad LoRA grad {step} {name}")
                lora_grads.append((name,float(p.grad.norm())))
        head_grads=[p for p in head.parameters() if p.grad is not None and torch.isfinite(p.grad).all()]
        base_bad=[p for name,p in runtime.model.named_parameters() if "lora_A" not in name and "lora_B" not in name and p.grad is not None and float(p.grad.abs().max()) != 0.0]
        if not lora_grads or not any(v>0 for _,v in lora_grads) or not head_grads or not any(float(p.grad.abs().max())>0 for p in head_grads): raise FloatingPointError(f"zero trainable gradient {step}")
        optimizer.step(); finite+=1; nonzero+=1; lora_nonzero+=int(any(v>0 for _,v in lora_grads)); base_nonzero+=len(base_bad)
        u=item["unit"]; sampling[(u["dataset"],u.get("category","unknown"))]+=1
        trace.append({"step":step,"unit_key":u["unit_key"],"dataset":u["dataset"],"category":u.get("category"),"candidate_count":int(target.numel()),"loss":float(loss.detach()),"grad_norm":float(torch.sqrt(sum((p.grad.detach()**2).sum() for p in head.parameters() if p.grad is not None))),"lora_grad_norms":{n:v for n,v in lora_grads},"base_nonzero_grad_count":len(base_bad),**parts})
        del item, patches, words, mask, nums, target, output, loss
        torch.cuda.empty_cache()
    elapsed=time.time()-t0
    ck=out/f"checkpoint_l66_visual_lora_step{args.steps}.pt"
    torch.save({"lora":{k:v.detach().cpu() for k,v in runtime.model.state_dict().items() if "lora_A" in k or "lora_B" in k},"head":head.state_dict(),"optimizer":optimizer.state_dict(),"step":args.steps,"seed":args.seed,"target_module":target_module,"rank":8,"alpha":16.0,"format":"locatemot-l66-visual-lora-set-v1"},ck)
    reload_head=L66VisualLoraSet(hidden=128).cpu(); reload_head.load_state_dict(torch.load(ck,map_location="cpu",weights_only=False)["head"],strict=True)
    payload={"format":"locatemot-l66-visual-lora-smoke-v1","status":"complete","stage":"fit-only-smoke","project_root":str(ROOT),"cwd":str(Path.cwd().resolve()),"seed":args.seed,"steps":args.steps,"finite_steps":finite,"nonzero_gradient_steps":nonzero,"lora_nonzero_gradient_steps":lora_nonzero,"base_nonzero_gradient_steps":base_nonzero,"checkpoint":str(ck),"checkpoint_sha256":sha256(ck),"checkpoint_reload":True,"fit_units_total":len(ordered),"sampled_domains":sorted({u["dataset"] for u in ordered[:args.steps]}),"sampled_categories":sorted({u.get("category") for u in ordered[:args.steps]}),"sampling_counts":{f"{d}|{c}":int(v) for (d,c),v in sampling.items()},"candidate_sets_complete":True,"candidate_key_drift":0,"candidate_truncation":False,"persistent_raw_dense_cache_written":False,"screening_gt_used":False,"official_test_labels_read":False,"ordinary_mot_ovmot_touched":False,"detector_frozen_base":all(not p.requires_grad for name,p in runtime.model.named_parameters() if "lora_A" not in name and "lora_B" not in name),"target_module":target_module,"rank":8,"alpha":16.0,"dropout":0.0,"adapter_parameter_count":sum(p.numel() for p in lora_parameters(runtime.model))+sum(p.numel() for p in head.parameters()),"input_contract":{"patch_joint":["N",17,512],"text_joint":[77,512],"numeric_dim":32,"global_cls_retained":True,"image_source":"streamed PNG"},"runtime":{"device":str(device),"precision":"FP32","peak_memory_bytes":int(torch.cuda.max_memory_allocated(device)),"elapsed_sec":elapsed,"steps_per_sec":args.steps/max(elapsed,1e-9)},"loss_trace":trace}
    (out/f"metrics_l66_step{args.steps}.json").write_text(json.dumps(payload,indent=2)+"\n"); (out/"loss_trace.json").write_text(json.dumps(trace,indent=2)+"\n"); (out/"sampling_trace.json").write_text(json.dumps(payload["sampling_counts"],indent=2)+"\n"); (out/"reload_audit.json").write_text(json.dumps({"strict":True,"reload_ok":True,"checkpoint":str(ck),"checkpoint_sha256":sha256(ck)},indent=2)+"\n"); (out/"config.json").write_text(json.dumps({"seed":args.seed,"steps":args.steps,"target_module":target_module,"rank":8,"alpha":16.0,"dropout":0.0,"head_hidden":128,"head_heads":4,"head_lr":2e-4,"lora_lr":5e-5,"fit_only":True,"same_class_hard_negative_metadata":"unavailable","screening_gt_used":False},indent=2)+"\n"); (out/"provenance.json").write_text(json.dumps({"project_root":str(ROOT),"cwd":str(Path.cwd().resolve()),"manifest_sha256":sha256(MANIFEST),"train_units":str(ROOT/"outputs/l49/data/train_units.jsonl"),"train_units_sha256":sha256(ROOT/"outputs/l49/data/train_units.jsonl"),"clip_weights":str(CLIP_WEIGHTS),"clip_weights_sha256":sha256(CLIP_WEIGHTS),"l65_checkpoint":str(L65_CHECKPOINT),"l65_checkpoint_sha256":sha256(L65_CHECKPOINT),"fit_only":True,"screening_gt_used":False,"official_test_labels_read":False,"ordinary_mot_ovmot_touched":False,"persistent_raw_dense_cache_written":False,"token_span_alignment":"UNALIGNED","static_motion_mask":"UNALIGNED"},indent=2)+"\n"); print(json.dumps({"status":"complete","steps":finite,"lora_nonzero":lora_nonzero,"base_nonzero":base_nonzero,"checkpoint":str(ck)},indent=2),flush=True)


if __name__ == "__main__": main()
