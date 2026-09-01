"""Stage L4: specification-equivariant training on paired views.

Each training item is a pair (full view, restricted view) of the same video
frame with the same common objects but different candidate subsets.  The
shared identity core is optimized for:
  - local assignment CE in each view (preserves U0 tracking quality);
  - permutation-invariant assignment consistency on common tracks/candidates;
  - track-state consistency for the same privileged identity across views.

Variants (--tag):
  a2 = spec-conditioned, no consistency (naive)
  a3 = + assignment consistency
  a4 = + state consistency
  a5 = full (assignment + state)

Usage:
  python tools/train_l4.py --tag a5 \
      --data outputs/l4/data/*.pkl \
      --out outputs/l4/checkpoints/a5 \
      --gpu 9 --epochs 30 --batch 32 --w-assign 1.0 --w-state 0.1
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

from locatemot.models.l1d_association import l1d_loss  # noqa: E402
from locatemot.models.l4_spec_eq import L4SpecEqAssociator  # noqa: E402


def load_pairs(paths):
    pairs = []
    for p in paths:
        with open(p, "rb") as f:
            pairs.extend(pickle.load(f))
    return pairs


class L4PairDataset(Dataset):
    def __init__(self, pairs, seed=20260806):
        self.pairs = pairs
        self.rng = np.random.RandomState(seed)
        weights = []
        for s in pairs:
            fw = float((~s["full"]["base_correct"]).mean()) if len(
                s["full"]["base_correct"]) else 0.0
            rw = float((~s["rest"]["base_correct"]).mean()) if len(
                s["rest"]["base_correct"]) else 0.0
            weights.append(1.0 + 2.0 * max(fw, rw))
        w = np.asarray(weights, np.float64)
        self.probs = w / w.sum()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        if self.rng.rand() < 0.6:
            idx = int(self.rng.choice(len(self.pairs), p=self.probs))
        return self.pairs[idx]


def pack_view(views, spec_idx, max_t, max_n):
    B = len(views)
    if np.ndim(spec_idx) == 0:
        spec_idx = np.full(B, int(spec_idx), np.int64)
    else:
        spec_idx = np.asarray(spec_idx, np.int64).reshape(B)
    Fp = views[0]["pair_feats"].shape[-1]
    Ft = views[0]["track_feats"].shape[-1]
    Fc = views[0]["cand_feats"].shape[-1]
    out = {
        "pair_feats": np.zeros((B, max_t, max_n, Fp), np.float32),
        "track_feats": np.zeros((B, max_t, Ft), np.float32),
        "cand_feats": np.zeros((B, max_n, Fc), np.float32),
        "base": np.zeros((B, max_t, max_n), np.float32),
        "row_label": np.full((B, max_t), -1, np.int64),
        "col_label": np.full((B, max_n), -1, np.int64),
        "base_correct": np.zeros((B, max_t), bool),
        "trk_mask": np.zeros((B, max_t), bool),
        "cand_mask": np.zeros((B, max_n), bool),
        "spec": spec_idx,
    }
    for b, v in enumerate(views):
        t = v["track_feats"].shape[0]
        n = v["cand_feats"].shape[0]
        out["pair_feats"][b, :t, :n] = v["pair_feats"]
        out["track_feats"][b, :t] = v["track_feats"]
        out["cand_feats"][b, :n] = v["cand_feats"]
        out["base"][b, :t, :n] = v["base"]
        out["row_label"][b, :t] = v["row_label"]
        out["col_label"][b, :n] = v["col_label"]
        out["base_correct"][b, :t] = v["base_correct"]
        out["trk_mask"][b, :t] = True
        out["cand_mask"][b, :n] = True
    return {k: torch.from_numpy(v) for k, v in out.items()}


def collate(batch):
    max_tf = max(s["full"]["track_feats"].shape[0] for s in batch)
    max_nf = max(s["full"]["cand_feats"].shape[0] for s in batch)
    max_tr = max(s["rest"]["track_feats"].shape[0] for s in batch)
    max_nr = max(s["rest"]["cand_feats"].shape[0] for s in batch)
    full = pack_view([s["full"] for s in batch], 0, max_tf, max_nf)
    rest = pack_view([s["rest"] for s in batch],
                     np.asarray([s["spec_idx"] for s in batch], np.int64),
                     max_tr, max_nr)
    return {
        "full": full,
        "rest": rest,
        "common_cand": [s["common_cand"] for s in batch],
        "common_track": [s["common_track"] for s in batch],
        "spec_idx": np.asarray([s["spec_idx"] for s in batch], np.int64),
    }


def partition_consistency(final_full, final_rest, batch):
    """Permutation-invariant partition consistency on common candidates.

    For each paired sample, column-softmax over each view's own tracks gives
    P_full [L,Tf] and P_rest [L,Tr] for the L common candidates.  The
    within-view co-assignment matrices S_full = P_full @ P_full.T and
    S_rest = P_rest @ P_rest.T are invariant to track relabeling; the loss
    is their MSE.  This directly optimizes the diagnostic we evaluate
    (pairwise co-identity agreement), without relying on birth-GT track
    alignment.
    """
    losses = []
    for b in range(final_full.shape[0]):
        ccd = batch["common_cand"][b]
        if ccd is None or len(ccd) < 2:
            continue
        ccd = np.asarray(ccd, np.int64)
        pf = torch.softmax(final_full[b][:, ccd[:, 0]].T, dim=-1)  # [L,Tf]
        pr = torch.softmax(final_rest[b][:, ccd[:, 1]].T, dim=-1)  # [L,Tr]
        sf = pf @ pf.T
        sr = pr @ pr.T
        losses.append(((sf - sr) ** 2).mean())
    return torch.stack(losses).mean() if losses else final_full.new_zeros(())


def consistency_losses(final_full, final_rest, trk_full, trk_rest, batch):
    row_kls = []
    col_kls = []
    state_vals = []
    for b in range(final_full.shape[0]):
        ccd = batch["common_cand"][b]
        cct = batch["common_track"][b]
        if ccd is None or len(ccd) == 0 or cct is None or len(cct) == 0:
            continue
        ccd = np.asarray(ccd, np.int64)
        cct = np.asarray(cct, np.int64)
        full_c = final_full[b][:, ccd[:, 0]]  # [Tf,L]
        rest_c = final_rest[b][:, ccd[:, 1]]  # [Tr,L]
        for fi, ri in cct:
            pf = F.softmax(full_c[int(fi)], dim=-1)
            pr = F.softmax(rest_c[int(ri)], dim=-1)
            row_kls.append((pf * (pf.log() - pr.log())).sum()
                           + (pr * (pr.log() - pf.log())).sum())
        full_t = final_full[b][cct[:, 0]][:, ccd[:, 0]]  # [K,L]
        rest_t = final_rest[b][cct[:, 1]][:, ccd[:, 1]]  # [K,L]
        for l in range(ccd.shape[0]):
            pf = F.softmax(full_t[:, l], dim=-1)
            pr = F.softmax(rest_t[:, l], dim=-1)
            col_kls.append((pf * (pf.log() - pr.log())).sum()
                           + (pr * (pr.log() - pf.log())).sum())
        for fi, ri in cct:
            a = F.normalize(trk_full[b, int(fi)], dim=-1)
            bvec = F.normalize(trk_rest[b, int(ri)], dim=-1)
            state_vals.append(1.0 - (a * bvec).sum())
    row_kl = torch.stack(row_kls).mean() if row_kls else final_full.new_zeros(())
    col_kl = torch.stack(col_kls).mean() if col_kls else final_full.new_zeros(())
    state = torch.stack(state_vals).mean() if state_vals else final_full.new_zeros(())
    part = partition_consistency(final_full, final_rest, batch)
    return part, row_kl, col_kl, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    choices=["a2", "a3", "a4", "a5", "a5p"])
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=9)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--w-assign", type=float, default=1.0)
    ap.add_argument("--w-state", type=float, default=0.1)
    ap.add_argument("--init-u0", default="outputs/l3/checkpoints/u0/final.pt")
    ap.add_argument("--seed", type=int, default=20260806)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[l4] tag={args.tag} device={device}", flush=True)

    pairs = load_pairs(args.data)
    ds = L4PairDataset(pairs, seed=args.seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True,
        collate_fn=collate, num_workers=4, persistent_workers=True,
        drop_last=True)
    print(f"[l4] pairs={len(pairs)} steps/epoch={len(loader)}", flush=True)

    model = L4SpecEqAssociator(n_spec=3, d_spec=16).to(device)
    if args.init_u0:
        ck = torch.load(args.init_u0, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        print("[l4] initialized from U0 checkpoint", flush=True)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[l4] trainable params={n_params/1e6:.3f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(loader),
        pct_start=args.warmup / max(1, args.epochs * len(loader)),
        anneal_strategy="cos")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(args)
    cfg["n_params"] = n_params
    cfg["n_pairs"] = len(pairs)
    with open(out_dir / "train_config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    step = 0
    t0 = time.time()
    for epoch in range(args.epochs):
        ep_loss = 0.0
        ep_n = 0
        for batch in loader:
            full = {k: v.to(device) for k, v in batch["full"].items()}
            rest = {k: v.to(device) for k, v in batch["rest"].items()}
            opt.zero_grad(set_to_none=True)
            pred_full = model(full)
            pred_rest = model(rest)
            lf = l1d_loss(full, pred_full)
            lr = l1d_loss(rest, pred_rest)
            part, row_kl, col_kl, state = consistency_losses(
                pred_full["final"], pred_rest["final"],
                pred_full["trk_tok"], pred_rest["trk_tok"], batch)
            assign_used = args.tag in ("a3", "a5", "a5p")
            state_used = args.tag in ("a4", "a5", "a5p")
            w_a = args.w_assign if assign_used else 0.0
            w_s = args.w_state if state_used else 0.0
            loss = lf["loss"] + lr["loss"] + w_a * part + w_s * state
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            ep_loss += float(loss)
            ep_n += 1
            if step % 100 == 0:
                print(
                    f"[l4:{args.tag}] step={step} loss={float(loss):.4f} "
                    f"lf={float(lf['loss']):.4f} lr={float(lr['loss']):.4f} "
                    f"part={float(part):.4f} rowkl={float(row_kl):.4f} "
                    f"colkl={float(col_kl):.4f} "
                    f"state={float(state):.4f} time={time.time()-t0:.1f}s",
                    flush=True)
        print(f"[l4:{args.tag}] epoch={epoch+1} mean_loss={ep_loss/max(1,ep_n):.4f} "
              f"elapsed={time.time()-t0:.1f}s", flush=True)
        torch.save({"model": model.state_dict(), "step": step,
                    "epoch": epoch + 1, "args": vars(args)},
                   out_dir / "latest.pt")
    torch.save({"model": model.state_dict(), "step": step,
                "epoch": args.epochs, "args": vars(args)},
               out_dir / "final.pt")
    print("[l4] done", flush=True)


if __name__ == "__main__":
    main()
