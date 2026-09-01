"""Stage L3: analyze U1 regime token behavior (dataset shortcut / variation).

Loads the U1 checkpoint and computes z_regime for held-out samples from each
domain, then reports:
  - z_regime mean/std per domain (dataset separation -> shortcut risk);
  - correlation of z with prediction-side density (intra-domain variation);
  - classifier accuracy of predicting domain from z (shortcut audit).

Usage:
  python tools/analyze_l3_routing.py --ckpt outputs/l3/checkpoints/u1/final.pt \
      --out outputs/l3/routing_audit.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from locatemot.models.l3_unified import L3Associator, regime_features_from_batch  # noqa: E402


def load_samples(paths):
    samples = []
    for p in paths:
        with open(p, "rb") as f:
            samples.extend(pickle.load(f))
    return samples


def collate_mini(batch):
    T = max(int(s["track_feats"].shape[0]) for s in batch)
    N = max(int(s["cand_feats"].shape[0]) for s in batch)
    B = len(batch)
    out = {
        "pair_feats": np.zeros((B, T, N, 19), np.float32),
        "track_feats": np.zeros((B, T, 16), np.float32),
        "cand_feats": np.zeros((B, N, 12), np.float32),
        "base": np.zeros((B, T, N), np.float32),
        "trk_mask": np.zeros((B, T), bool),
        "cand_mask": np.zeros((B, N), bool),
    }
    for b, s in enumerate(batch):
        t, n = s["track_feats"].shape[0], s["cand_feats"].shape[0]
        out["pair_feats"][b, :t, :n] = s["pair_feats"]
        out["track_feats"][b, :t] = s["track_feats"]
        out["cand_feats"][b, :n] = s["cand_feats"]
        out["base"][b, :t, :n] = s["base"]
        out["trk_mask"][b, :t] = True
        out["cand_mask"][b, :n] = True
    return {k: torch.from_numpy(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="outputs/l3/routing_audit.json")
    ap.add_argument("--data", nargs="+", default=[
        "outputs/l1_d/data/dancetrack_calibration_k.pkl",
        "outputs/l1_d/data/bdd100k_train_k.pkl",
        "outputs/l1_d/data/mot17_train_k.pkl",
        "outputs/l1_d/data/mot20_train_k.pkl",
    ])
    ap.add_argument("--gpu", type=int, default=8)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = L3Associator(use_spec=False)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    doms = ["dancetrack", "bdd", "mot17", "mot20"]
    all_z = {d: [] for d in doms}
    all_rho = {d: [] for d in doms}
    rng = np.random.default_rng(0)
    for d, p in zip(doms, args.data):
        samples = load_samples([p])
        idx = rng.choice(len(samples), size=min(2000, len(samples)),
                         replace=False)
        with torch.no_grad():
            for i in range(0, len(idx), 64):
                batch = collate_mini([samples[j] for j in idx[i:i+64]])
                batch = {k: v.to(device) for k, v in batch.items()}
                z = model.regime_enc(batch).cpu().numpy()
                rf = regime_features_from_batch(batch).cpu().numpy()
                for k in range(z.shape[0]):
                    all_z[d].append(z[k])
                    all_rho[d].append(rf[k, 1])  # log1p(n_cand)
    means = {d: np.mean(np.stack(all_z[d]), 0) for d in doms}
    # dataset separation: mean pairwise centroid distance vs intra spread
    centroids = np.stack([means[d] for d in doms])
    pair_dist = np.linalg.norm(
        centroids[:, None] - centroids[None, :], axis=-1)
    intra = {d: float(np.std(np.stack(all_z[d]), 0).mean()) for d in doms}
    # correlation of z components with density
    corr = {}
    for d in doms:
        Z = np.stack(all_z[d])
        rho = np.asarray(all_rho[d])
        c = np.corrcoef(Z.T, rho)[:Z.shape[1], -1]
        corr[d] = float(np.max(np.abs(c)))
    # simple domain classifier on z (logistic regression head)
    from sklearn.linear_model import LogisticRegression
    X = np.concatenate([np.stack(all_z[d]) for d in doms])
    y = np.concatenate([np.full(len(all_z[d]), i) for i, d in enumerate(doms)])
    rperm = rng.permutation(len(y))
    X, y = X[rperm], y[rperm]
    split = int(0.8 * len(y))
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X[:split], y[:split])
    acc = clf.score(X[split:], y[split:])
    out = {
        "n_per_domain": {d: len(all_z[d]) for d in doms},
        "z_mean_norm": {d: float(np.linalg.norm(means[d])) for d in doms},
        "centroid_pair_dist": pair_dist.tolist(),
        "intra_std_mean": intra,
        "max_abs_corr_with_density": corr,
        "domain_classifier_acc": float(acc),
        "random_acc": 0.25,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
