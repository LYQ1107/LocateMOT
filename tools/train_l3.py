"""Stage L3: train shared dense (U0) or regime/spec-conditioned (U1) core.

Usage:
  python tools/train_l3.py --model u1 --gpu 8 \
    --data outputs/l1_d/data/dancetrack_calibration_k.pkl \
           outputs/l1_d/data/bdd100k_train_k.pkl \
           outputs/l1_d/data/mot17_train_k.pkl \
           outputs/l1_d/data/mot20_train_k.pkl \
    --out outputs/l3/checkpoints/u1 --epochs 30
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

from locatemot.models.l1d_association import L1DAssociator, l1d_loss  # noqa: E402
from locatemot.models.l3_unified import L3Associator  # noqa: E402


def load_data(paths):
    samples = []
    for p in paths:
        with open(p, "rb") as f:
            samples.extend(pickle.load(f))
    return samples


class L3Dataset(Dataset):
    def __init__(self, samples, seed=20260806):
        self.samples = samples
        self.rng = np.random.RandomState(seed)
        wrong = []
        for s in samples:
            labs = s["row_label"] >= 0
            if labs.any():
                wrong.append(float((~s["base_correct"][labs]).mean()))
            else:
                wrong.append(0.0)
        w = np.asarray(wrong, np.float64) + 0.05
        self.hard_probs = w / w.sum()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.rng.rand() < 0.5:
            idx = int(self.rng.choice(len(self.samples), p=self.hard_probs))
        return self.samples[idx]


def collate(batch, spec=True):
    T = max(int(s["track_feats"].shape[0]) for s in batch)
    N = max(int(s["cand_feats"].shape[0]) for s in batch)
    B = len(batch)
    Fp = batch[0]["pair_feats"].shape[-1]
    Ft = batch[0]["track_feats"].shape[-1]
    Fc = batch[0]["cand_feats"].shape[-1]
    out = {
        "pair_feats": np.zeros((B, T, N, Fp), np.float32),
        "track_feats": np.zeros((B, T, Ft), np.float32),
        "cand_feats": np.zeros((B, N, Fc), np.float32),
        "base": np.zeros((B, T, N), np.float32),
        "row_label": np.full((B, T), -1, np.int64),
        "col_label": np.full((B, N), -1, np.int64),
        "base_correct": np.zeros((B, T), bool),
        "trk_mask": np.zeros((B, T), bool),
        "cand_mask": np.zeros((B, N), bool),
        "spec": np.zeros(B, np.int64),
    }
    for b, s in enumerate(batch):
        t = s["track_feats"].shape[0]
        n = s["cand_feats"].shape[0]
        out["pair_feats"][b, :t, :n] = s["pair_feats"]
        out["track_feats"][b, :t] = s["track_feats"]
        out["cand_feats"][b, :n] = s["cand_feats"]
        out["base"][b, :t, :n] = s["base"]
        out["row_label"][b, :t] = s["row_label"]
        out["col_label"][b, :n] = s["col_label"]
        out["base_correct"][b, :t] = s["base_correct"]
        out["trk_mask"][b, :t] = True
        out["cand_mask"][b, :n] = True
        if spec:
            out["spec"][b] = int(s.get("spec_idx", 0))
    return {k: torch.from_numpy(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["u0", "u1"], default="u1")
    ap.add_argument("--gpu", type=int, default=8)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--spec", action="store_true")
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--z-dim", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260806)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[l3] model={args.model} device={device}", flush=True)

    samples = load_data(args.data)
    ds = L3Dataset(samples, seed=args.seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        collate_fn=lambda b: collate(b, spec=args.spec),
        num_workers=4, persistent_workers=True, drop_last=True)
    print(f"[l3] samples={len(samples)} steps/epoch={len(loader)}", flush=True)

    if args.model == "u0":
        model = L1DAssociator().to(device)
    else:
        model = L3Associator(
            d_model=args.d_model, n_layers=args.n_layers,
            z_dim=args.z_dim, use_spec=args.spec).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[l3] trainable params={n_params/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr,
        total_steps=args.epochs * len(loader),
        pct_start=args.warmup / max(1, args.epochs * len(loader)),
        anneal_strategy="cos")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(args)
    cfg["n_params"] = n_params
    cfg["n_samples"] = len(samples)
    with open(out_dir / "train_config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        ep_loss = 0.0
        ep_n = 0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            pred = model(batch)
            losses = l1d_loss(batch, pred)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            ep_loss += float(losses["loss"])
            ep_n += 1
            if step % 100 == 0:
                print(
                    f"[l3] step={step} loss={float(losses['loss']):.4f} "
                    f"row={float(losses['row_ce']):.4f} "
                    f"n_row={int(losses['n_row'])} "
                    f"time={time.time()-t0:.1f}s", flush=True)
            if step % args.save_every == 0:
                torch.save({"model": model.state_dict(), "step": step,
                            "args": vars(args)},
                           out_dir / f"step_{step}.pt")
        print(f"[l3] epoch={epoch+1} mean_loss={ep_loss/max(1,ep_n):.4f} "
              f"elapsed={time.time()-t0:.1f}s", flush=True)
        torch.save({"model": model.state_dict(), "step": step,
                    "epoch": epoch + 1, "args": vars(args)},
                   out_dir / "latest.pt")
    torch.save({"model": model.state_dict(), "step": step,
                "epoch": args.epochs, "args": vars(args)},
               out_dir / "final.pt")
    print("[l3] done", flush=True)


if __name__ == "__main__":
    main()
