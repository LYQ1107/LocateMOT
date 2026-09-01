#!/usr/bin/env python3
"""Train-only 100-step L41 bidirectional raw-image relation smoke."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.models.l41_raw_relational_identity import L41RawRelationalIdentity
from tools.l41_raw_data import (FIT_VIDEOS, WEIGHTS, StreamingClipPatchEncoder, load_fragments,
                                make_pairs, pad_patches, relation_features, sha256)

AUDIT = ROOT / "outputs/l41/audit/relational_identity_contract.json"


def choose_pairs(pairs, each=256):
    pos = [x for x in pairs if x["label"] and x["kind"] == "same_gt_fragment"]
    hard = [x for x in pairs if not x["label"] and x["kind"] == "same_frame_different_gt_hard"]
    inactive = [x for x in pairs if not x["label"] and x["kind"] == "inactive"]
    if not pos or not hard: raise RuntimeError(f"missing positive/hard pairs {len(pos)}/{len(hard)}")
    return pos[:each] + hard[:each] + inactive[:each // 4]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out-root", required=True); ap.add_argument("--steps", type=int, default=100); ap.add_argument("--seed", type=int, default=20260829); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--crop-batch", type=int, default=32); args = ap.parse_args()
    assert Path.cwd().resolve() == ROOT
    out = Path(args.out_root); out = out if out.is_absolute() else ROOT / out; out.mkdir(parents=True, exist_ok=False)
    audit = json.loads(AUDIT.read_text())
    if not audit["decision"]["enter_l41_smoke"]: raise RuntimeError("L41 relation audit failed")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    started = time.time(); fragments, alignment = load_fragments(FIT_VIDEOS); pairs = make_pairs(fragments); selected = choose_pairs(pairs)
    used = sorted({x for p in selected for x in (p["a"], p["b"])})
    encoder = StreamingClipPatchEncoder(device=device, weights=WEIGHTS, batch_size=args.crop_batch); patch_list = encoder.encode(fragments, used); patch_map = {i: x for i, x in zip(used, patch_list)}; del patch_list, encoder
    model = L41RawRelationalIdentity(hidden=96, history=8).to(device); opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4); rng = np.random.default_rng(args.seed)
    pos_pool = [x for x in selected if x["label"]]; neg_pool = [x for x in selected if not x["label"]]; trace=[]; grads=[]; attn=[]
    model.train()
    for step in range(args.steps):
        pos = [pos_pool[int(i)] for i in rng.integers(len(pos_pool), size=min(32, len(pos_pool)))]; neg = [neg_pool[int(i)] for i in rng.integers(len(neg_pool), size=min(32, len(neg_pool)))]; batch = pos + neg
        left_ids = [x["a"] for x in batch]; right_ids = [x["b"] for x in batch]; left, lm = pad_patches(fragments, left_ids, patch_map, device); right, rm = pad_patches(fragments, right_ids, patch_map, device)
        rel = torch.stack([relation_features(fragments[x["a"]], fragments[x["b"]]) for x in batch]).to(device); y = torch.tensor([1.0] * len(pos) + [0.0] * len(neg), device=device)
        result = model(left, right, rel, lm, rm); logits = result["logit"]; bce = F.binary_cross_entropy_with_logits(logits, y); ps, ns = logits[:len(pos)], logits[len(pos):]; pairwise = F.softplus(0.2 + ns.unsqueeze(0) - ps.unsqueeze(1)).mean(); loss = bce + 0.5 * pairwise
        if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite loss at step {step + 1}")
        opt.zero_grad(set_to_none=True); loss.backward(); grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        if not np.isfinite(grad) or grad <= 0: raise FloatingPointError(f"invalid gradient at step {step + 1}: {grad}")
        opt.step(); grads.append(grad); attn.append(float(result["attention_norm"].detach())); trace.append({"step": step + 1, "total": float(loss.detach()), "pair_bce": float(bce.detach()), "hard_pairwise": float(pairwise.detach()), "positive_pairs": len(pos), "hard_negative_pairs": len(neg)})
    checkpoint = out / f"checkpoint_l41_raw_relational_identity_step{args.steps}.pt"; payload = {"format": "locatemot-l41-streaming-raw-relational-identity-v1", "stage": "L41", "seed": args.seed, "steps": args.steps, "device": str(device), "fit_videos": list(FIT_VIDEOS), "fragment_count": len(fragments), "all_pair_count": len(pairs), "selected_pair_count": len(selected), "selected_fragment_count": len(used), "positive_pairs": len(pos_pool), "hard_negative_pairs": sum(x["kind"] == "same_frame_different_gt_hard" for x in selected), "inactive_pairs": sum(x["kind"] == "inactive" for x in selected), "raw_embeddings_persisted": False, "backbone": "frozen OpenAI CLIP ViT-B/16 spatial patch tokens reduced to 2x2 cells", "weights": str(WEIGHTS), "weights_sha256": sha256(WEIGHTS), "loss_mean": {k: float(np.mean([x[k] for x in trace])) for k in ("total", "pair_bce", "hard_pairwise")}, "loss_final": trace[-1], "gradient_norm": {"mean": float(np.mean(grads)), "max": float(np.max(grads)), "nonzero_steps": len([x for x in grads if x > 0])}, "attention_norm": {"mean": float(np.mean(attn)), "max": float(np.max(attn))}, "checkpoint": str(checkpoint.resolve()), "checkpoint_reload": False, "audit_sha256": sha256(AUDIT), "semantic_inputs_excluded": ["expression", "source_id", "pool_id", "group_id", "state_key"], "screening_gt_used_for_fit": False, "labels": "GT_PRIVILEGED_ORACLE from train-only L28 cache", "elapsed_sec": time.time() - started, "alignment_count": len(alignment)}
    torch.save({"model": model.state_dict(), "config": payload}, checkpoint); reload_model = L41RawRelationalIdentity(hidden=96, history=8).to(device); reload_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"]); reload_model.eval(); payload["checkpoint_reload"] = True
    (out / f"metrics_l41_smoke{args.steps}.json").write_text(json.dumps(payload, indent=2) + "\n"); (out / "loss_trace.json").write_text(json.dumps(trace, indent=2) + "\n"); print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
