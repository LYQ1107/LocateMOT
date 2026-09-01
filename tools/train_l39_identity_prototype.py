#!/usr/bin/env python3
"""Train-only supervised temporal prototype smoke for L39."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l39_identity_prototype import L39IdentityPrototype

CACHE_ROOT = ROOT / "outputs/l28/track_sequence_bank_final"
AUDIT = ROOT / "outputs/l39/audit/identity_probe_contract.json"
TRAIN_VIDEOS = ["0000", "0001", "0002", "0003", "0006", "0007", "0008", "0009", "0010", "0012", "0014", "0020"]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def gt_set(value):
    if value is None: return set()
    if isinstance(value, (list, tuple, set)): return {str(x) for x in value}
    return {str(value)}


def load_fragments(videos):
    fragments = []
    for video in videos:
        cache = torch.load(CACHE_ROOT / f"{video}.pt", map_location="cpu", weights_only=False)
        ptr = cache["track_ptr"].tolist(); frames = cache["obs_frame"].numpy(); source = cache["obs_source"].numpy(); labels = cache["obs_gt_ids"]
        for ti, track in enumerate(cache["track_ids"].tolist()):
            begin, end = int(ptr[ti]), int(ptr[ti + 1]); indices = list(range(begin, end))
            gids = set().union(*(gt_set(x) for x in labels[begin:end]))
            if not gids: continue
            indices = indices[-8:]
            feat = cache["obs_features"][torch.as_tensor(indices)].float()
            fs = frames[indices].astype(np.float32); denom = max(1.0, float(frames.max() + 1))
            fragments.append({"video": video, "track_id": int(track), "features": feat,
                              "times": torch.as_tensor(fs / denom), "gids": gids,
                              "frames": set(int(x) for x in frames[indices]),
                              "source": int(np.round(source[begin:end].mean()))})
    return fragments


def build_pairs(fragments, max_pos_per_gt=64, max_neg_per_fragment=8):
    gt_to_frag = defaultdict(list); frame_to_frag = defaultdict(list)
    for i, f in enumerate(fragments):
        for g in f["gids"]: gt_to_frag[(f["video"], g)].append(i)
        for frame in f["frames"]: frame_to_frag[(f["video"], frame)].append(i)
    pairs = []; seen = set()
    for key, ids in sorted(gt_to_frag.items()):
        ids = sorted(set(ids)); count = 0
        for ai, a in enumerate(ids):
            for b in ids[ai + 1:]:
                if fragments[a]["track_id"] == fragments[b]["track_id"]: continue
                pair = (a, b, 1, "same_gt_fragment")
                if pair not in seen: pairs.append(pair); seen.add(pair); count += 1
                if count >= max_pos_per_gt: break
            if count >= max_pos_per_gt: break
    for a, fa in enumerate(fragments):
        candidates = []
        for frame in fa["frames"]:
            candidates.extend(frame_to_frag[(fa["video"], frame)])
        for b in sorted(set(candidates)):
            fb = fragments[b]
            if a == b or fa["gids"] & fb["gids"] or fa["track_id"] == fb["track_id"]: continue
            pair = (a, b, 0, "same_frame_different_gt_hard")
            if pair not in seen: pairs.append(pair); seen.add(pair)
            if sum(x[0] == a and x[2] == 0 for x in pairs) >= max_neg_per_fragment: break
    if not any(x[2] for x in pairs) or not any(not x[2] for x in pairs):
        raise RuntimeError("identity pair construction lacks positive or hard-negative pairs")
    return pairs


def batch_tensor(fragments, ids, device):
    batch = len(ids); length = 8; dim = int(fragments[ids[0]]["features"].shape[-1])
    values = torch.zeros((batch, length, dim), dtype=torch.float32)
    times = torch.zeros((batch, length), dtype=torch.float32)
    mask = torch.zeros((batch, length), dtype=torch.bool)
    for row, index in enumerate(ids):
        n = min(length, int(fragments[index]["features"].shape[0]))
        values[row, -n:] = fragments[index]["features"][-n:]
        times[row, -n:] = fragments[index]["times"][-n:]
        mask[row, -n:] = True
    values = values.to(device); times = times.to(device); mask = mask.to(device)
    return values, mask, times


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-root", required=True); ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260829); ap.add_argument("--device", default="cuda:0"); args = ap.parse_args()
    assert Path.cwd().resolve() == ROOT
    out = Path(args.out_root); out = out if out.is_absolute() else ROOT / out; out.mkdir(parents=True, exist_ok=False)
    audit = json.loads(AUDIT.read_text())
    if not audit["decision"]["identity_supervision_available"]: raise RuntimeError("identity audit did not pass")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    fragments = load_fragments(TRAIN_VIDEOS); pairs = build_pairs(fragments)
    device = torch.device(args.device); model = L39IdentityPrototype(hidden=96, history=8, prototype_dim=96).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4); rng = np.random.default_rng(args.seed)
    trace=[]; grads=[]; proto_norms=[]; start=time.time(); model.train(); amp = device.type == "cuda"
    for step in range(args.steps):
        take = rng.integers(len(pairs), size=min(64, len(pairs))); chosen = [pairs[int(i)] for i in take]
        left = [x[0] for x in chosen]; right = [x[1] for x in chosen]; y = torch.as_tensor([x[2] for x in chosen], dtype=torch.float32, device=device)
        a, am, at = batch_tensor(fragments, left, device); b, bm, bt = batch_tensor(fragments, right, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
            za = model(a, am, at)["prototype"]; zb = model(b, bm, bt)["prototype"]
            score = (za * zb).sum(-1)
            pair_loss = F.binary_cross_entropy_with_logits(score / 0.1, y)
            pos = score[y.bool()]; neg = score[~y.bool()]
            margin = F.softplus(0.2 + neg[None, :] - pos[:, None]).mean() if len(pos) and len(neg) else score.new_zeros(())
            norm_loss = (za.norm(dim=-1).sub(1).abs().mean() + zb.norm(dim=-1).sub(1).abs().mean())
            loss = pair_loss + 0.5 * margin + 0.05 * norm_loss
        if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite loss at step {step+1}")
        opt.zero_grad(set_to_none=True); loss.backward(); grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        if not np.isfinite(grad): raise FloatingPointError(f"nonfinite gradient at step {step+1}")
        opt.step(); grads.append(grad); proto_norms.extend(za.detach().float().norm(dim=-1).cpu().tolist())
        trace.append({"step":step+1,"total":float(loss.detach()),"pairwise_supervised_contrastive":float(pair_loss.detach()),"hard_margin":float(margin.detach()),"prototype_norm_loss":float(norm_loss.detach()),"positive_pairs":int(y.sum()),"negative_hard_pairs":int((~y.bool()).sum())})
    checkpoint=out/f"checkpoint_l39_identity_prototype_step{args.steps}.pt"
    payload={"format":"locatemot-l39-identity-prototype-v1","stage":"train-only-identity-smoke","seed":args.seed,"steps":args.steps,"device":str(device),"train_videos":TRAIN_VIDEOS,"train_fragment_count":len(fragments),"pair_count":len(pairs),"positive_pairs":int(sum(x[2] for x in pairs)),"hard_negative_pairs":int(sum(not x[2] for x in pairs)),"teacher_or_emission_used":False,"screening_gt_used_for_fit":False,"identity_labels":"GT_PRIVILEGED_ORACLE from train cache","semantic_inputs_excluded":["expression","source_id","pool_id","group_id","state_key"],"loss_mean":{k:float(np.mean([x[k] for x in trace])) for k in ("total","pairwise_supervised_contrastive","hard_margin","prototype_norm_loss")},"prototype_norm":{"mean":float(np.mean(proto_norms)),"max":float(np.max(proto_norms))},"gradient_norm":{"mean":float(np.mean(grads)),"max":float(np.max(grads)),"nonzero_steps":int(np.count_nonzero(np.asarray(grads)>0))},"cache_manifest_sha256":sha(CACHE_ROOT/"manifest.json"),"audit_sha256":sha(AUDIT),"elapsed_sec":time.time()-start}
    torch.save({"model":model.state_dict(),"config":payload},checkpoint)
    reload_model=L39IdentityPrototype(hidden=96,history=8,prototype_dim=96).to(device); reload_model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=False)["model"]); reload_model.eval()
    payload["checkpoint"]=str(checkpoint.resolve()); payload["checkpoint_reload"]=True
    (out/f"metrics_l39_smoke{args.steps}.json").write_text(json.dumps(payload,indent=2)+"\n"); (out/"loss_trace.json").write_text(json.dumps(trace,indent=2)+"\n"); print(json.dumps(payload,indent=2),flush=True)


if __name__ == "__main__": main()
