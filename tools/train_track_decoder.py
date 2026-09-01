#!/usr/bin/env python
"""Stage L0-C: train B4 Persistent Track Decoder."""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locatemot.data.collate import collate_track_batch  # noqa: E402
from locatemot.data.pair_dataset import PairDataset, PrecomputedPairSet  # noqa: E402
from locatemot.models.track_decoder.losses import pair_losses  # noqa: E402
from locatemot.models.track_decoder.model import TrackDecoderModel  # noqa: E402
from locatemot.models.track_decoder.trainer import train_loop  # noqa: E402


def eval_fn(model, loader, device):
    model.eval()
    total = corr = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            pred = model(batch)
            match = pred["match_logits"]
            nm = pred["no_match_logits"]
            for b in range(match.shape[0]):
                for i in range(match.shape[1]):
                    if bool(batch["candidate_missing"][b, i]) or not bool(batch["ref_mask"][b, i]):
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
    ap.add_argument("--out", default="outputs/l0_c/checkpoints/b4")
    ap.add_argument("--query-direction", default="reference_query", choices=["reference_query", "current_query"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--precomputed-train", default="")
    ap.add_argument("--precomputed-calib", default="")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    records = [json.loads(l) for l in open(args.manifest)]
    pre_root = os.path.normpath(os.path.join(args.out, "..", "..", "precomputed"))
    os.makedirs(pre_root, exist_ok=True)
    if args.precomputed_train:
        train_ds = PrecomputedPairSet(None, cache_path=args.precomputed_train)
    else:
        train_recs = [r for r in records if r["split"] == "train"]
        train_ds = PrecomputedPairSet(PairDataset(train_recs, args.cache_root, seed=20260806),
                                      cache_path=os.path.join(pre_root, "train.pt"))
    if args.precomputed_calib:
        calib_ds = PrecomputedPairSet(None, cache_path=args.precomputed_calib)
    else:
        calib_recs = [r for r in records if r["split"] == "calibration"]
        calib_ds = PrecomputedPairSet(PairDataset(calib_recs, args.cache_root, seed=20260806),
                                      cache_path=os.path.join(pre_root, "calibration.pt"))
    print(f"train pairs: {len(train_ds)}, calib pairs: {len(calib_ds)}")
    model = TrackDecoderModel(query_direction=args.query_direction)
    cfg = {
        "model": "b4_track_decoder", "query_direction": args.query_direction,
        "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
    }
    train_loop(
        model, train_ds, calib_ds, out_dir=args.out, lr=args.lr,
        max_epochs=args.epochs, batch_size=args.batch_size, cfg=cfg,
        loss_fn=lambda p, b: pair_losses(p, b, {
            "assignment": 1.0, "no_match": 0.5, "contrastive": 0.25,
            "geometry": 0.1, "calibration": 0.1,
        }),
        eval_fn=eval_fn, device="cuda",
    )


if __name__ == "__main__":
    main()
