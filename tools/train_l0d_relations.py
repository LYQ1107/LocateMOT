#!/usr/bin/env python
"""Stage L0-D: train B5 RelationPairwise variants and B6 Relation Track Decoder.

Sampling: weighted by target-count bucket (towards 25/45/30) and prediction-side
hard-competition flag (2x). Checkpoint selection only on calibration.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locatemot.data.pair_dataset import PairDataset, PrecomputedPairSet  # noqa: E402
from locatemot.evaluation.assignment import assign_tracks_to_candidates  # noqa: E402
from locatemot.models.track_decoder.losses import pair_losses  # noqa: E402
from locatemot.models.track_decoder.relation_pairwise import RelationPairwiseModel  # noqa: E402
from locatemot.models.track_decoder.relation_track_decoder import RelationTrackDecoderModel  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT = "/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L0C/cache"
MANIFEST = os.path.join(ROOT, "outputs/l0_c/pair_manifest.jsonl")
OUT = os.path.join(ROOT, "outputs/l0_d")


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def hard_flag_from_tensors(t):
    """Prediction-side hard flag from precomputed tensors (same frozen definition)."""
    rb, cb = t["ref_boxes"].float(), t["cur_boxes"].float()
    iou = torch.zeros((rb.shape[0], cb.shape[0]))
    M, N = rb.shape[0], cb.shape[0]
    for i in range(M):
        for j in range(N):
            iou[i, j] = _iou(rb[i].tolist(), cb[j].tolist())
    cos = torch.nn.functional.cosine_similarity(
        t["ref_pbd_be"].float().unsqueeze(1), t["cur_pbd_be"].float().unsqueeze(0), dim=-1)
    if M >= 5 or N >= 6:
        return True
    for i in range(M):
        ious = iou[i]
        coses = cos[i]
        if N >= 2:
            order = torch.argsort(ious, descending=True)
            if ious[order[0]] - ious[order[1]] < 0.10:
                return True
            order_c = torch.argsort(coses, descending=True)
            if coses[order_c[0]] - coses[order_c[1]] < 0.05:
                return True
        if (ious >= 0.30).sum() >= 3:
            return True
    for j in range(N):
        if (iou[:, j] >= 0.30).sum() >= 2:
            return True
    return False


def build_sampler_weights(records):
    weights = np.zeros(len(records), dtype=np.float32)
    actual = defaultdict(int)
    for r in records:
        m = r["reference_target_count"]
        actual["1" if m == 1 else ("2-4" if m <= 4 else "5-8")] += 1
    desired = {"1": 0.25, "2-4": 0.45, "5-8": 0.30}
    bucket_w = {k: desired[k] / (actual[k] / len(records)) for k in desired}
    bucket_w["5-8"] = min(bucket_w["5-8"], 6.0)
    for i, r in enumerate(records):
        m = r["reference_target_count"]
        b = "1" if m == 1 else ("2-4" if m <= 4 else "5-8")
        weights[i] = bucket_w[b]
    return weights, bucket_w, dict(actual)


def make_model(variant: str):
    if variant == "b5a":
        return RelationPairwiseModel(use_pbd_base=False, use_region_geom=False, residual=True)
    if variant == "b5b":
        return RelationPairwiseModel(use_pbd_base=True, use_region_geom=False, residual=True)
    if variant == "b5c":
        return RelationPairwiseModel(use_pbd_base=True, use_region_geom=True, residual=True)
    if variant == "b6":
        return RelationTrackDecoderModel(use_pbd_base=True, use_region_geom=True, residual=True)
    if variant == "b6_nores":
        return RelationTrackDecoderModel(use_pbd_base=True, use_region_geom=True, residual=False)
    raise ValueError(variant)


def clean_eval(model, calib_set, device, batch_size=32, nm_sigmoid=False):
    model.eval()
    total = corr = 0
    with torch.no_grad():
        for s in range(0, len(calib_set), batch_size):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in calib_set.batch(torch.arange(s, min(s + batch_size, len(calib_set)))).items()}
            pred = model(batch)
            match = pred["match_logits"]
            nm = pred["no_match_logits"]
            for b in range(match.shape[0]):
                M = int(batch["ref_mask"][b].sum())
                N = int(batch["cur_mask"][b].sum())
                if N == 0 or M == 0:
                    continue
                assign = assign_tracks_to_candidates(
                    match[b, :M, :N].cpu().numpy(), nm[b, :M].cpu().numpy())
                for i in range(M):
                    if bool(batch["candidate_missing"][b, i]) or not bool(batch["ref_mask"][b, i]):
                        continue
                    total += 1
                    tgt = int(batch["match_targets"][b, i])
                    got = next((tag for ti, tag in assign if ti == i), None)
                    if tgt >= 0:
                        corr += int(got == f"candidate:{tgt}")
                    else:
                        corr += int(got == "NO_MATCH")
    return corr / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["b5a", "b5b", "b5c", "b6", "b6_nores"], required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--precomputed", default=os.path.join(OUT, "precomputed/train_full.pt"))
    ap.add_argument("--precomputed-calib", default=os.path.join(OUT, "precomputed/calib_full.pt"))
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = [json.loads(l) for l in open(MANIFEST)]
    train_recs = [r for r in records if r["split"] == "train"]
    calib_recs = [r for r in records if r["split"] == "calibration"]

    def get_set(recs, path):
        if os.path.exists(path):
            return PrecomputedPairSet(None, cache_path=path)
        ds = PairDataset(recs, CACHE_ROOT, seed=20260806, cache_items=False)
        return PrecomputedPairSet(ds, cache_path=path)

    train_set = get_set(train_recs, args.precomputed)
    calib_set = get_set(calib_recs, args.precomputed_calib)
    print(f"[train] train={len(train_set)} calib={len(calib_set)} device={device}", flush=True)

    # per-sample hard flags from precomputed tensors
    hard_flags = []
    for i in range(len(train_set)):
        t = {k: train_set._tensors[k][i] for k in ("ref_boxes", "cur_boxes", "ref_pbd_be", "cur_pbd_be")}
        M = int(train_set._tensors["ref_mask"][i].sum())
        N = int(train_set._tensors["cur_mask"][i].sum())
        if M == 0 or N == 0:
            hard_flags.append(False)
            continue
        hard_flags.append(hard_flag_from_tensors(
            {"ref_boxes": t["ref_boxes"][:M], "cur_boxes": t["cur_boxes"][:N],
             "ref_pbd_be": t["ref_pbd_be"][:M], "cur_pbd_be": t["cur_pbd_be"][:N]}))
    hard_count = sum(hard_flags)
    weights, bucket_w, actual = build_sampler_weights(train_recs)
    for i, hf in enumerate(hard_flags):
        if hf:
            weights[i] *= 2.0
    print(f"[train] hard={hard_count}/{len(train_set)} bucket_weights={bucket_w} actual={actual}", flush=True)
    with open(os.path.join(OUT, "l0d_sampling_weights.json"), "w") as f:
        json.dump({"hard_count": hard_count, "total": len(train_set),
                   "bucket_weights": bucket_w, "actual": actual}, f, ensure_ascii=False, indent=2)

    model = make_model(args.model)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    num_samples = len(train_set) * 2
    sampler = WeightedRandomSampler(torch.from_numpy(weights).double(), num_samples=num_samples, replacement=True)
    loader = DataLoader(range(len(train_set)), batch_size=args.batch_size, sampler=sampler, drop_last=True)
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup = int(total_steps * 0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / max(1, warmup)) * 0.5 * (1 + math.cos(math.pi * s / total_steps)))

    out_dir = args.out or os.path.join(OUT, f"checkpoints/{args.model}")
    os.makedirs(out_dir, exist_ok=True)
    best_score, best_step, bad_evals = -1e9, -1, 0
    curves = []
    step = 0
    for epoch in range(args.epochs):
        model.train()
        for idxs in loader:
            idxs = torch.as_tensor(idxs, dtype=torch.long)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in train_set.batch(idxs).items()}
            opt.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                pred = model(batch)
                losses = pair_losses(pred, batch, {
                    "assignment": 1.0, "no_match": 2.0, "contrastive": 0.25,
                    "geometry": 0.1, "calibration": 0.1,
                })
                loss = losses["loss"]
            if not torch.isfinite(loss):
                print(f"[train] step {step} non-finite loss; skip", flush=True)
                opt.zero_grad(set_to_none=True)
                step += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % args.eval_every == 0:
                score = clean_eval(model, calib_set, device)
                curves.append({"step": step, "epoch": epoch,
                               "calib_cond": round(score, 4)})
                if score > best_score:
                    best_score, best_step, bad_evals = score, step, 0
                    torch.save({
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "scheduler": sched.state_dict(),
                        "step": step,
                        "seed": 20260806,
                    }, os.path.join(out_dir, "best.pt"))
                else:
                    bad_evals += 1
                print(f"[train] step {step} calib_cond={score:.4f} best={best_score:.4f} "
                      f"bad={bad_evals} alpha={float(model.alpha):.4f}", flush=True)
                model.train()
        torch.save({
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "scheduler": sched.state_dict(), "step": step, "seed": 20260806,
        }, os.path.join(out_dir, "latest.pt"))
        if bad_evals >= args.patience:
            print(f"[train] early stop at epoch {epoch} (bad_evals={bad_evals})", flush=True)
            break
    with open(os.path.join(out_dir, "training_curves.csv"), "w") as f:
        if curves:
            f.write("step,epoch,calib_cond\n")
            for c in curves:
                f.write(f"{c['step']},{c['epoch']},{c['calib_cond']}\n")
    with open(os.path.join(out_dir, "train_manifest.json"), "w") as f:
        json.dump({"model": args.model, "best_step": best_step, "best_calib_cond": best_score,
                   "final_step": step, "seed": 20260806, "epochs_run": epoch + 1,
                   "batch_size": args.batch_size, "lr": args.lr}, f, ensure_ascii=False, indent=2)
    print(json.dumps({"best_step": best_step, "best_calib_cond": best_score}, indent=2))


if __name__ == "__main__":
    main()
