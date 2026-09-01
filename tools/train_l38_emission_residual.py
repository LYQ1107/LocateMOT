#!/usr/bin/env python3
"""Train-only smoke for the bounded L29-emission-preserving residual."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l29_frame_membership_set_decoder import L29FrameMembershipSetDecoder
from locatemot.models.l38_bounded_emission_residual import L38BoundedEmissionResidual
from tools.train_l28_track_set_decoder import build_queries, state_at

L29_CHECKPOINT = ROOT / "outputs/l29/train/frame_membership_step1000/checkpoint_frame_membership_step1000.pt"
L28_ROOT = ROOT / "outputs/l28/track_sequence_bank_final"
TEXT_ROOT = ROOT / "outputs/l26/candidate_bank_v5_crossmodal"
AUDIT = ROOT / "outputs/l38/audit/emission_contract.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def balanced_bce(logits, target):
    if not logits.numel():
        return logits.new_zeros(())
    target = target.float(); pos = target.bool(); parts = []
    if pos.any(): parts.append(F.binary_cross_entropy_with_logits(logits[pos], target[pos]))
    if (~pos).any(): parts.append(F.binary_cross_entropy_with_logits(logits[~pos], target[~pos]))
    return torch.stack(parts).mean() if parts else logits.new_zeros(())


def targets(selected_gt, selected_frames, query, cutoff):
    n = len(selected_gt); length = len(selected_gt[0]) if n else 0
    frame_targets = query["target"]
    current = torch.zeros(n, dtype=torch.bool)
    sequence = torch.zeros(n, dtype=torch.bool)
    members = torch.zeros((n, length), dtype=torch.bool)
    latest_frame = torch.full((n,), -1, dtype=torch.int64)
    for i in range(n):
        valid = [j for j, f in enumerate(selected_frames[i]) if f >= 0]
        if valid: latest_frame[i] = int(selected_frames[i][valid[-1]])
        for j in valid:
            gt = selected_gt[i][j]; frame = int(selected_frames[i][j])
            hit = gt in frame_targets.get(frame, set())
            members[i, j] = hit; sequence[i] |= hit
        if latest_frame[i] == int(cutoff):
            j = valid[-1]
            current[i] = selected_gt[i][j] in frame_targets.get(int(cutoff), set())
    return current, sequence, members, latest_frame


def one_loss(residual, teacher, query, cache, text_hidden, text_mask, teacher_model,
             device, rng, use_amp):
    cutoff = int(rng.choice(cache["obs_frame"].numpy()))
    obs, om, ot, selected_gt, selected_frames = state_at(cache, cutoff, history=8)
    current_y, sequence_y, member_y, latest_frame = targets(
        selected_gt, selected_frames, query, cutoff)
    obs, om, ot = obs.to(device), om.to(device), ot.to(device)
    current_y, sequence_y, member_y = current_y.to(device), sequence_y.to(device), member_y.to(device)
    qh = text_hidden[query["text_index"]].to(device); qm = text_mask[query["text_index"]].to(device)
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else contextlib.nullcontext()
    with torch.no_grad(), amp:
        encoded = teacher_model.encode_observations(obs, om, ot)
        teacher_out = teacher_model.forward_encoded(encoded, encoded[1], qh, qm)
        teacher_logits = teacher_out["current_membership_logits"].float()
    with amp:
        out = residual(obs, om, ot, qh, qm, teacher_logits)
    active = torch.ones_like(current_y, dtype=torch.bool)
    pos = torch.nonzero(current_y, as_tuple=False).flatten()
    neg = torch.nonzero(~current_y, as_tuple=False).flatten()
    hard = neg[torch.argsort(teacher_logits[neg], descending=True)[:min(24, len(neg))]] if len(neg) else neg
    z = out["final_score"].new_zeros(())
    final = out["final_score"]
    distill = F.smooth_l1_loss(final, teacher_logits.detach())
    membership = balanced_bce(final[active], current_y[active])
    if len(pos) and len(hard):
        pair = F.softplus(0.25 + final[hard][None, :] - final[pos][:, None]).mean()
    else: pair = z
    if len(pos):
        # All positives are retained in both terms; no max-positive reduction.
        set_loss = (torch.logsumexp(final[active], 0) - torch.logsumexp(final[pos], 0)
                    - F.logsigmoid(final[pos]).mean())
    else: set_loss = z
    continuation_target = (sequence_y & ~current_y).float()
    continuation = F.binary_cross_entropy_with_logits(out["continuation_logit"], continuation_target)
    hist_mask = om[:, 1:] & om[:, :-1]
    temporal = (out["residual_history"][:, 1:] - out["residual_history"][:, :-1]).abs()[hist_mask].mean() if hist_mask.any() else z
    total = distill + 0.25 * membership + 0.20 * pair + 0.15 * set_loss + 0.10 * continuation + 0.05 * temporal
    return total, {
        "total": float(total.detach()), "distillation": float(distill.detach()),
        "membership_bce": float(membership.detach()), "pairwise": float(pair.detach()),
        "multi_positive_set": float(set_loss.detach()), "continuation": float(continuation.detach()),
        "temporal": float(temporal.detach()), "residual_abs_max": float(out["residual_score"].detach().abs().max()),
        "residual_abs_mean": float(out["residual_score"].detach().abs().mean()),
        "current_candidates": int(active.sum()), "current_positives": int(current_y.sum()),
        "hard_negatives": int(len(hard)), "multi_positive_frame": int(current_y.sum() > 1),
        "continuation_positive": int(continuation_target.sum()), "cutoff": cutoff,
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-root", required=True)
    ap.add_argument("--steps", type=int, default=100); ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--device", default="cuda:0"); args = ap.parse_args()
    assert Path.cwd().resolve() == ROOT
    out = Path(args.out_root); out = out if out.is_absolute() else ROOT / out; out.mkdir(parents=True, exist_ok=False)
    audit = json.loads(AUDIT.read_text())
    if not audit["contract_checks"]["teacher_finite"]: raise RuntimeError("L38 teacher audit failed")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    queries = build_queries(); text = torch.load(TEXT_ROOT / "text_tokens.pt", map_location="cpu", weights_only=False)
    text_hidden = text["token_hidden"].float(); text_mask = text["attention_mask"].bool(); del text
    cache_by_video = {v: torch.load(L28_ROOT / f"{v}.pt", map_location="cpu", weights_only=False)
                      for v in sorted({q["video"] for q in queries})}
    device = torch.device(args.device)
    teacher = L29FrameMembershipSetDecoder().to(device)
    teacher.load_state_dict(torch.load(L29_CHECKPOINT, map_location=device, weights_only=False)["model"]); teacher.eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    residual = L38BoundedEmissionResidual(hidden=96, history=8).to(device)
    optimizer = torch.optim.AdamW(residual.parameters(), lr=2e-4, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed); trace=[]; grads=[]; start=time.time(); residual.train(); use_amp=device.type == "cuda"
    for step in range(args.steps):
        q = queries[int(rng.integers(len(queries)))]
        loss, part = one_loss(residual, teacher, q, cache_by_video[q["video"]], text_hidden, text_mask, teacher, device, rng, use_amp)
        if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite loss step {step+1}")
        optimizer.zero_grad(set_to_none=True); loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(residual.parameters(), 5.0))
        if not np.isfinite(grad): raise FloatingPointError(f"nonfinite gradient step {step+1}")
        optimizer.step(); part["step"]=step+1; trace.append(part); grads.append(grad)
    checkpoint=out/f"checkpoint_l38_emission_residual_step{args.steps}.pt"
    payload={"format":"locatemot-l38-bounded-emission-residual-v1","stage":"train-only-smoke","seed":args.seed,"steps":args.steps,"device":str(device),"hidden":96,"history_length":8,"residual_bound":0.05,"train_query_count":len(queries),"train_video_count":len(cache_by_video),"teacher_checkpoint":str(L29_CHECKPOINT.resolve()),"teacher_checkpoint_sha256":sha(L29_CHECKPOINT),"cache_manifest_sha256":sha(L28_ROOT/"manifest.json"),"expression_audit":str((ROOT/"outputs/l37/audit/expression_supervision_manifest.json").resolve()),"screening_gt_used_for_fit":False,"semantic_inputs_excluded":["pool_id","source_id","group_id","state_key"],"token_level_alignment_verified":False,"motion_language_decomposition":"not claimed; no verified motion-language mask","loss_mean":{k:float(np.mean([x[k] for x in trace])) for k in ("total","distillation","membership_bce","pairwise","multi_positive_set","continuation","temporal")},"sampling_totals":{k:int(sum(x[k] for x in trace)) for k in ("current_candidates","current_positives","hard_negatives","multi_positive_frame","continuation_positive")},"residual_bound_observed":{"max":float(max(x["residual_abs_max"] for x in trace)),"mean_abs":float(np.mean([x["residual_abs_mean"] for x in trace]))},"gradient_norm":{"mean":float(np.mean(grads)),"max":float(np.max(grads)),"nonzero_steps":int(np.count_nonzero(np.asarray(grads)>0))},"elapsed_sec":time.time()-start}
    torch.save({"model":residual.state_dict(),"config":payload},checkpoint)
    reload_model=L38BoundedEmissionResidual(hidden=96,history=8).to(device); reload_model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)["model"]); reload_model.eval()
    payload["checkpoint"]=str(checkpoint.resolve()); payload["checkpoint_reload"]=True
    (out/f"metrics_l38_smoke{args.steps}.json").write_text(json.dumps(payload,indent=2)+"\n"); (out/"loss_trace.json").write_text(json.dumps(trace,indent=2)+"\n"); print(json.dumps(payload,indent=2),flush=True)


if __name__ == "__main__": main()
