#!/usr/bin/env python3
"""L40 train-only raw-image identity prototype smoke.

The CLIP encoder is frozen.  Crop pixels are streamed once into an in-memory
smoke set, then released; no raw-image embedding cache is written to disk.
"""
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
from locatemot.models.l40_raw_image_identity import L40RawImageIdentity
from tools.l40_raw_data import FIT_VIDEOS, WEIGHTS, StreamingClipEncoder, load_fragments, make_pairs, sha256

AUDIT = ROOT / "outputs/l40/audit/raw_image_identity_contract.json"


def pad_embeddings(fragments, ids, embeddings, device):
    n = len(ids); h = 8; image_dim = int(embeddings[ids[0]].shape[-1])
    images = torch.zeros((n, h, image_dim), dtype=torch.float32)
    numeric = torch.zeros((n, h, 24), dtype=torch.float32)
    times = torch.zeros((n, h), dtype=torch.float32)
    mask = torch.zeros((n, h), dtype=torch.bool)
    for j, i in enumerate(ids):
        f = fragments[int(i)]; count = min(h, len(f["obs"])); start = h - count
        images[j, start:] = embeddings[int(i)][-count:]
        numeric[j, start:] = torch.stack([x["numeric"] for x in f["obs"][-count:]])
        denom = max(1.0, float(max(f["frames"]) + 1))
        times[j, start:] = torch.tensor([x["frame"] / denom for x in f["obs"][-count:]])
        mask[j, start:] = True
    return images.to(device), numeric.to(device), mask.to(device), times.to(device)


def select_pairs(pairs, max_each=256):
    positives = [x for x in pairs if x["label"] and x["kind"] == "same_gt_fragment"]
    hard = [x for x in pairs if not x["label"] and x["kind"] == "same_frame_different_gt_hard"]
    inactive = [x for x in pairs if not x["label"] and x["kind"] == "inactive"]
    if not positives or not hard:
        raise RuntimeError(f"missing positive/hard pairs: {len(positives)}/{len(hard)}")
    return positives[:max_each] + hard[:max_each] + inactive[:max_each // 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True); ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260829); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--crop-batch", type=int, default=32); args = ap.parse_args()
    assert Path.cwd().resolve() == ROOT
    out = Path(args.out_root); out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=False)
    audit = json.loads(AUDIT.read_text())
    if not audit["decision"]["enter_l40_smoke"]:
        raise RuntimeError("L40 raw-image contract did not pass")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    start = time.time()
    fragments, alignment = load_fragments(FIT_VIDEOS)
    all_pairs = make_pairs(fragments)
    selected = select_pairs(all_pairs)
    used_ids = sorted({x for p in selected for x in (p["a"], p["b"])})
    streamer = StreamingClipEncoder(device=device, weights=WEIGHTS, batch_size=args.crop_batch)
    # Only selected fragment embeddings are retained in RAM for this smoke.
    encoded_list = streamer.encode(fragments, used_ids)
    embeddings = {i: z for i, z in zip(used_ids, encoded_list)}
    del encoded_list, streamer
    if device.type == "cuda": torch.cuda.empty_cache()
    model = L40RawImageIdentity(hidden=96, history=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed); trace=[]; gradients=[]; norms=[]; crop_count=sum(len(embeddings[i]) for i in used_ids)
    pos_pool = [p for p in selected if p["label"]]; neg_pool = [p for p in selected if not p["label"]]
    model.train()
    for step in range(args.steps):
        pos = [pos_pool[int(x)] for x in rng.integers(len(pos_pool), size=min(32, len(pos_pool)))]
        neg = [neg_pool[int(x)] for x in rng.integers(len(neg_pool), size=min(32, len(neg_pool)))]
        batch = pos + neg; ids_a = [p["a"] for p in batch]; ids_b = [p["b"] for p in batch]
        a = pad_embeddings(fragments, ids_a, embeddings, device); b = pad_embeddings(fragments, ids_b, embeddings, device)
        za = model(*a)["prototype"]; zb = model(*b)["prototype"]; scores = (za * zb).sum(-1)
        y = torch.tensor([1.0] * len(pos) + [0.0] * len(neg), device=device)
        pair_bce = F.binary_cross_entropy_with_logits(scores / 0.1, y)
        pos_s, neg_s = scores[:len(pos)], scores[len(pos):]
        pairwise = F.softplus(0.2 + neg_s.unsqueeze(0) - pos_s.unsqueeze(1)).mean()
        norm_loss = (za.norm(dim=-1).sub(1).abs().mean() + zb.norm(dim=-1).sub(1).abs().mean())
        loss = pair_bce + 0.5 * pairwise + 0.05 * norm_loss
        if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite loss at step {step + 1}")
        opt.zero_grad(set_to_none=True); loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0))
        if not np.isfinite(grad) or grad <= 0: raise FloatingPointError(f"invalid gradient at step {step + 1}: {grad}")
        opt.step(); gradients.append(grad); norms.extend(za.detach().float().norm(dim=-1).cpu().tolist())
        trace.append({"step": step + 1, "total": float(loss.detach()), "pair_bce": float(pair_bce.detach()), "hard_pairwise": float(pairwise.detach()), "prototype_norm": float(norm_loss.detach()), "positive_pairs": len(pos), "hard_negative_pairs": len(neg)})
    checkpoint = out / f"checkpoint_l40_raw_image_identity_step{args.steps}.pt"
    payload = {"format": "locatemot-l40-streaming-raw-image-identity-v1", "stage": "L40", "seed": args.seed, "steps": args.steps, "device": str(device), "fit_videos": list(FIT_VIDEOS), "fragment_count": len(fragments), "all_pair_count": len(all_pairs), "selected_pair_count": len(selected), "selected_fragment_count": len(used_ids), "positive_pairs": len(pos_pool), "hard_negative_pairs": len(neg_pool), "inactive_pairs": sum(x["kind"] == "inactive" for x in selected), "crop_count": crop_count, "raw_embeddings_persisted": False, "backbone": "frozen OpenAI CLIP ViT-B/16", "weights": str(WEIGHTS), "weights_sha256": sha256(WEIGHTS), "semantic_inputs_excluded": ["expression", "source_id", "pool_id", "group_id", "state_key"], "labels": "GT_PRIVILEGED_ORACLE from train-only L28 cache", "screening_gt_used_for_fit": False, "loss_mean": {k: float(np.mean([x[k] for x in trace])) for k in ("total", "pair_bce", "hard_pairwise", "prototype_norm")}, "loss_final": trace[-1], "gradient_norm": {"mean": float(np.mean(gradients)), "max": float(np.max(gradients)), "nonzero_steps": len([x for x in gradients if x > 0])}, "prototype_norm_stats": {"mean": float(np.mean(norms)), "max": float(np.max(norms))}, "checkpoint": str(checkpoint.resolve()), "audit_sha256": sha256(AUDIT), "elapsed_sec": time.time() - start, "alignment_count": len(alignment)}
    torch.save({"model": model.state_dict(), "config": payload}, checkpoint)
    reload_model = L40RawImageIdentity(hidden=96, history=8).to(device); reload_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model"]); reload_model.eval()
    payload["checkpoint_reload"] = True
    (out / f"metrics_l40_smoke{args.steps}.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "loss_trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
