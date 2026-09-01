"""Stage L2: Trajectory Utility Model (TUM) training / evaluation.

Inputs (causal): track history features, candidate features, pair features,
and an action (component assignment) indicator. Outputs: predicted future
trajectory utility per horizon H.

Supervision (privileged): oracle counterfactual rollout utilities
(windowed AssA from TrackEval formulas) saved by tools/run_l2_oracle.py.

Usage:
  python tools/train_l2_tum.py \
      --events outputs/l2/oracle/events_dancetrack_val.pkl \
             outputs/l2/oracle/events_bdd100k_train.pkl \
      --out outputs/l2/tum --epochs 60 --d-model 256 --n-layers 4
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from locatemot.models.l1d_association import (  # noqa: E402
    CAND_FEATURES,
    PAIR_FEATURES,
    TRACK_FEATURES,
    compute_affinity_features,
)


def event_features(ev):
    """Recompute causal features from an oracle event (no future info)."""
    snaps = ev["track_snaps"]
    cands = ev["cands"]
    T, N = len(snaps), len(cands)
    if T == 0 or N == 0:
        return None
    tb = np.stack([s["last_box"] for s in snaps])
    pb = np.stack([s["prev_box"] for s in snaps])
    rb = np.stack([s["ref_pbd"] for s in snaps])
    ab = np.stack([s["anchor_pbd"] for s in snaps])
    cb = np.stack([c["box"] for c in cands])
    cp = np.stack([c["pbd"] for c in cands])
    cg = np.asarray([c["gen"] for c in cands], np.float32)
    gaps = np.asarray([max(1, ev["frame"] - s["last_frame"]) for s in snaps],
                      np.float32)
    ages = np.asarray([s["age"] for s in snaps], np.float32)
    hits = np.asarray([s["hits"] for s in snaps], np.float32)
    feats = compute_affinity_features(
        tb, cb, rb, ab, cp, cg, gaps, ages, hits, pb,
        (0.4, 0.2, 0.4), ev["image_size"])
    return {
        "pair": feats["pair_feats"],
        "track": feats["track_feats"],
        "cand": feats["cand_feats"],
        "base": feats["base"],
    }


def action_matrix(act, T, N):
    m = np.zeros((T, N), np.float32)
    for ti, ci in act:
        if ti < T and ci < N:
            m[ti, ci] = 1.0
    return m


class TUM(nn.Module):
    """Action-conditional set-level trajectory utility model."""

    def __init__(self, d_model=256, n_layers=4, n_heads=8, ffn=1024,
                 horizons=(4, 8, 16, 32), dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.horizons = horizons
        self.track_proj = nn.Sequential(
            nn.Linear(len(TRACK_FEATURES), d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Dropout(dropout))
        self.cand_proj = nn.Sequential(
            nn.Linear(len(CAND_FEATURES), d_model), nn.LayerNorm(d_model),
            nn.GELU(), nn.Dropout(dropout))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ffn, dropout, batch_first=True,
            norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        # pair head: track emb + cand emb + pair feats + base + action
        pair_in = 2 * d_model + len(PAIR_FEATURES) + 1 + 1
        self.pair_head = nn.Sequential(
            nn.Linear(pair_in, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1))
        # per-horizon utility from pooled pair logits
        self.heads = nn.ModuleDict({
            f"H{h}": nn.Sequential(nn.Linear(1, 64), nn.GELU(),
                                   nn.Linear(64, 1))
            for h in horizons})
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch):
        """batch keys: pair [B,T,N,Fp], track [B,T,Ft], cand [B,N,Fc],
        base [B,T,N], action [B,T,N], trk_mask, cand_mask."""
        pair = torch.nan_to_num(batch["pair"].float())
        trk = torch.nan_to_num(batch["track"].float())
        cand = torch.nan_to_num(batch["cand"].float())
        base = torch.nan_to_num(batch["base"].float())
        act = batch["action"].float()
        B, T, N, _ = pair.shape
        trk_tok = self.track_proj(trk)
        cand_tok = self.cand_proj(cand)
        tokens = torch.cat([cand_tok, trk_tok], dim=1)
        mask = torch.cat([batch["cand_mask"], batch["trk_mask"]], dim=1)
        out = self.encoder(tokens, src_key_padding_mask=~mask)
        cand_out = out[:, :N]
        trk_out = out[:, N:]
        t_exp = trk_out.unsqueeze(2).expand(B, T, N, self.d_model)
        c_exp = cand_out.unsqueeze(1).expand(B, T, N, self.d_model)
        pair_in = torch.cat(
            [t_exp, c_exp, pair, base.unsqueeze(-1), act.unsqueeze(-1)], dim=-1)
        logits = self.pair_head(pair_in).squeeze(-1)  # [B,T,N]
        # weighted pool: only action edges contribute
        act_sum = act.sum(dim=(1, 2)).clamp(min=1e-6)
        pooled = (logits * act).sum(dim=(1, 2)) / act_sum  # [B]
        out_h = {}
        for h in self.horizons:
            out_h[f"H{h}"] = self.heads[f"H{h}"](pooled.unsqueeze(-1)).squeeze(-1)
        return out_h, pooled


class L2EventDataset(torch.utils.data.Dataset):
    def __init__(self, events, horizons, seed=20260806):
        self.samples = []
        rng = np.random.default_rng(seed)
        feat_cache = [event_features(ev) for ev in events]
        for ei, ev in enumerate(events):
            feats = feat_cache[ei]
            if feats is None:
                continue
            T, N = feats["pair"].shape[:2]
            for ai, act in enumerate(ev["actions"]):
                am = action_matrix(act, T, N)
                utils = ev["action_utils"][str(ai)]["utils"]
                target = {}
                ok = True
                for h in horizons:
                    w = utils.get(f"H{h}")
                    if w is None:
                        ok = False
                        break
                    target[f"H{h}"] = float(w["assa"])
                if not ok:
                    continue
                self.samples.append({
                    "pair": feats["pair"].astype(np.float32),
                    "track": feats["track"].astype(np.float32),
                    "cand": feats["cand"].astype(np.float32),
                    "base": feats["base"].astype(np.float32),
                    "action": am,
                    "target": target,
                    "event": ei,
                })
        self.horizons = list(horizons)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        return {
            "pair": torch.as_tensor(s["pair"]),
            "track": torch.as_tensor(s["track"]),
            "cand": torch.as_tensor(s["cand"]),
            "base": torch.as_tensor(s["base"]),
            "action": torch.as_tensor(s["action"]),
            "target": {h: torch.as_tensor(s["target"][h]) for h in self.horizons},
        }


def collate(batch):
    max_t = max(b["pair"].shape[0] for b in batch)
    max_n = max(b["pair"].shape[1] for b in batch)
    B = len(batch)
    Ft = batch[0]["track"].shape[1]
    Fc = batch[0]["cand"].shape[1]
    Fp = batch[0]["pair"].shape[2]
    out = {
        "pair": torch.zeros(B, max_t, max_n, Fp),
        "track": torch.zeros(B, max_t, Ft),
        "cand": torch.zeros(B, max_n, Fc),
        "base": torch.zeros(B, max_t, max_n),
        "action": torch.zeros(B, max_t, max_n),
    }
    for bi, b in enumerate(batch):
        t, n = b["pair"].shape[0], b["pair"].shape[1]
        out["pair"][bi, :t, :n] = b["pair"]
        out["track"][bi, :t] = b["track"]
        out["cand"][bi, :n] = b["cand"]
        out["base"][bi, :t, :n] = b["base"]
        out["action"][bi, :t, :n] = b["action"]
    out["trk_mask"] = torch.zeros(len(batch), max_t, dtype=torch.bool)
    out["cand_mask"] = torch.zeros(len(batch), max_n, dtype=torch.bool)
    for bi, b in enumerate(batch):
        out["trk_mask"][bi, : b["track"].shape[0]] = True
        out["cand_mask"][bi, : b["cand"].shape[0]] = True
    out["target"] = {
        h: torch.stack([b["target"][h] for b in batch])
        for h in batch[0]["target"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", nargs="+", required=True)
    ap.add_argument("--out", default="outputs/l2/tum")
    ap.add_argument("--horizons", default="4,8,16,32")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--ffn", type=int, default=1024)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--eval-videos", type=int, default=10,
                    help="hold out last N videos per domain for eval")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizons = [int(x) for x in args.horizons.split(",")]

    events = []
    for p in args.events:
        with open(p, "rb") as f:
            events.extend(pickle.load(f))
    # hold out last eval_videos per domain (sorted video ids)
    by_domain = {}
    for ev in events:
        by_domain.setdefault(ev["domain"], []).append(ev)
    tr_events, ev_events = [], []
    for dom, evs in by_domain.items():
        vids = sorted({e["video"] for e in evs})
        hold = set(vids[-args.eval_videos:] if args.eval_videos else [])
        tr_events += [e for e in evs if e["video"] not in hold]
        ev_events += [e for e in evs if e["video"] in hold]
    print(f"train events={len(tr_events)} eval events={len(ev_events)}")

    ds = L2EventDataset(tr_events, horizons, args.seed)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
        num_workers=2, persistent_workers=True)
    model = TUM(args.d_model, args.n_layers, args.n_heads, args.ffn,
                horizons).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TUM params={n_params/1e6:.2f}M")
    os.makedirs(args.out, exist_ok=True)

    def evaluate(eval_events):
        model.eval()
        ev_ds = L2EventDataset(eval_events, horizons, args.seed)
        if len(ev_ds) == 0:
            return None
        ev_dl = torch.utils.data.DataLoader(
            ev_ds, batch_size=args.batch_size, collate_fn=collate)
        preds = {h: [] for h in horizons}
        with torch.no_grad():
            for batch in ev_dl:
                b = {k: v.to(device) for k, v in batch.items() if k != "target"}
                out_h, _ = model(b)
                for h in horizons:
                    preds[h].append(out_h[f"H{h}"].cpu().numpy())
        # group by event: dataset order is per-event contiguous
        result = {}
        for h in horizons:
            pred = np.concatenate(preds[h])
            # reconstruct event-level grouping
            # simpler: use dataset's event ids
            event_ids = np.asarray([s["event"] for s in ev_ds.samples])
            uniq = np.unique(event_ids)
            top1 = auc_sum = auc_total = ndcg_sum = regret_sum = 0
            n_ev = len(uniq)
            for u in uniq:
                m = event_ids == u
                p = pred[m]
                acts = eval_events[u]["actions"]
                utils = np.asarray([
                    eval_events[u]["action_utils"][str(i)]["utils"][f"H{h}"]["assa"]
                    for i in range(len(acts))])
                best = int(np.argmax(utils))
                top1 += int(np.argmax(p) == best)
                regret_sum += utils[best] - utils[int(np.argmax(p))]
                # pairwise AUC
                for i in range(len(p)):
                    for j in range(i + 1, len(p)):
                        if utils[i] == utils[j]:
                            continue
                        auc_total += 1
                        auc_sum += int((p[i] - p[j]) * (utils[i] - utils[j]) > 0)
                # NDCG
                order = np.argsort(-p)
                dcg = sum((2 ** utils[order[k]] - 1) / np.log2(k + 2)
                          for k in range(len(order)))
                idcg = sum((2 ** utils[k] - 1) / np.log2(k + 2)
                           for k in np.argsort(-utils))
                ndcg_sum += dcg / max(1e-9, idcg)
            result[f"H{h}"] = {
                "top1": top1 / max(1, n_ev),
                "n_events": n_ev,
                "auc": auc_sum,
                "auc_total": auc_total,
                "auc_frac": auc_sum / max(1, auc_total),
                "ndcg": ndcg_sum / max(1, n_ev),
                "mean_regret": regret_sum / max(1, n_ev),
            }
        return result

    for ep in range(args.epochs):
        model.train()
        losses = []
        t0 = time.time()
        for batch in dl:
            b = {k: v.to(device) for k, v in batch.items() if k != "target"}
            out_h, _ = model(b)
            loss = torch.zeros((), device=device)
            for h in horizons:
                # listwise CE over actions within each sample (batch dim)
                logits = out_h[f"H{h}"]
                tgt = batch["target"][h].to(device)
                loss = loss + F.mse_loss(logits, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        if (ep + 1) % 10 == 0 or ep == 0:
            ev_res = evaluate(ev_events)
            print(f"epoch {ep+1} loss={np.mean(losses):.5f} "
                  f"eval={json.dumps(ev_res, default=str)} "
                  f"time={time.time()-t0:.1f}s", flush=True)
            torch.save({"model": model.state_dict(), "epoch": ep,
                        "args": vars(args)},
                       os.path.join(args.out, "checkpoint.pt"))
    ev_res = evaluate(ev_events)
    with open(os.path.join(args.out, "eval.json"), "w") as f:
        json.dump(ev_res, f, indent=2, default=str)
    torch.save({"model": model.state_dict(), "epoch": args.epochs,
                "args": vars(args)},
               os.path.join(args.out, "final.pt"))
    print(json.dumps(ev_res, indent=2, default=str))


if __name__ == "__main__":
    main()
