#!/usr/bin/env python
"""Stage L0-C: train B3 Pairwise MLP on cached features."""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locatemot.data.pair_dataset import PairDataset, PrecomputedPairSet  # noqa: E402
from locatemot.models.track_decoder.model import PairwiseModel  # noqa: E402
from locatemot.models.track_decoder.trainer import train_loop  # noqa: E402


def loss_fn(pred, batch, weights=None):
    weights = weights or {"assignment": 1.0, "no_match": 0.5, "contrastive": 0.25,
                          "geometry": 0.1, "calibration": 0.1}
    B, M, N = batch["labels"].shape
    E = B * M * N
    match = pred["match_logits"].reshape(-1)
    nm = pred["no_match_logits"].reshape(-1)
    labels = batch["labels"].reshape(-1)
    valid = (
        batch["ref_mask"].unsqueeze(2)
        & batch["cur_mask"].unsqueeze(1)
        & (~batch["candidate_missing"].unsqueeze(2))
    ).reshape(-1)
    assignment = F.binary_cross_entropy_with_logits(match[valid], labels[valid])

    nm_t = batch["no_match_targets"].unsqueeze(2).expand(B, M, N).reshape(-1)
    no_match = F.binary_cross_entropy_with_logits(nm[valid], nm_t[valid])

    rf = pred["ref_feats"]
    cf = pred["cur_feats"]
    D = rf.shape[-1]
    cos = F.cosine_similarity(
        rf.unsqueeze(2).expand(B, M, N, D).reshape(E, D),
        cf.unsqueeze(1).expand(B, M, N, D).reshape(E, D),
        dim=-1,
    )
    pos = valid & (labels == 1)
    neg = valid & (labels == 0)
    contrastive = (1 - cos[pos]).mean() + torch.clamp(cos[neg] - 0.3, min=0).mean()

    geom_target = torch.zeros(B, M, N, device=match.device)
    tgt = batch["match_targets"]
    for b in range(B):
        for i in range(M):
            if int(tgt[b, i]) >= 0:
                geom_target[b, i, int(tgt[b, i])] = batch["gt_iou"][b, i]
    geometry = F.mse_loss(
        torch.sigmoid(pred["iou_logit"].reshape(-1)[pos]),
        geom_target.reshape(-1)[pos],
    )

    top_logit, _ = match.reshape(B, M, N).max(dim=2)
    calib_t = (tgt >= 0).float()
    calib_mask = (~batch["candidate_missing"]) & batch["ref_mask"]
    calibration = F.binary_cross_entropy_with_logits(top_logit[calib_mask], calib_t[calib_mask])

    losses = {
        "assignment": assignment, "no_match": no_match, "contrastive": contrastive,
        "geometry": geometry, "calibration": calibration,
    }
    losses["loss"] = sum(weights[k] * v for k, v in losses.items())
    return losses


def eval_fn(model, loader, device):
    model.eval()
    total = corr = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            pred = model(batch)
            B, M, N = batch["labels"].shape
            match = pred["match_logits"].reshape(B, M, N)
            nm = pred["no_match_logits"].reshape(B, M, N).mean(dim=2)
            for b in range(B):
                for i in range(M):
                    if bool(batch["candidate_missing"][b, i]):
                        continue
                    total += 1
                    if int(batch["match_targets"][b, i]) >= 0:
                        corr += int(match[b, i].argmax().item() == int(batch["match_targets"][b, i]))
                    else:
                        corr += int((nm[b, i] > 0) == (float(batch["no_match_targets"][b, i]) > 0))
    acc = corr / max(1, total)
    return acc, {"calib_acc": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-root", default="/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L0C/cache")
    ap.add_argument("--manifest", default="outputs/l0_c/pair_manifest.jsonl")
    ap.add_argument("--out", default="outputs/l0_c/checkpoints/b3")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    records = [json.loads(l) for l in open(args.manifest)]
    train_recs = [r for r in records if r["split"] == "train"]
    calib_recs = [r for r in records if r["split"] == "calibration"]
    print(f"train pairs: {len(train_recs)}, calib pairs: {len(calib_recs)}")
    train_ds = PairDataset(train_recs, args.cache_root, seed=20260806)
    calib_ds = PairDataset(calib_recs, args.cache_root, seed=20260806)
    os.makedirs(os.path.join(args.out, "..", "..", "precomputed"), exist_ok=True)
    pre_root = os.path.normpath(os.path.join(args.out, "..", "..", "precomputed"))
    train_ds = PrecomputedPairSet(train_ds, cache_path=os.path.join(pre_root, "train.pt"))
    calib_ds = PrecomputedPairSet(calib_ds, cache_path=os.path.join(pre_root, "calibration.pt"))
    model = PairwiseModel()
    cfg = {"model": "b3_pairwise_mlp", "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size}
    train_loop(
        model, train_ds, calib_ds, out_dir=args.out, lr=args.lr,
        max_epochs=args.epochs, batch_size=args.batch_size, cfg=cfg,
        loss_fn=loss_fn, eval_fn=eval_fn, device="cuda",
    )


if __name__ == "__main__":
    main()
