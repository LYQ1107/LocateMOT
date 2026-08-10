"""Stage L5 Route A: train GT-anchored temporal identity transformer.

Usage:
  python tools/train_l5_route_a.py \
      --data outputs/l5/clips/small_bdd_train.pkl \
             outputs/l5/clips/small_dance_train.pkl \
      --val-data outputs/l5/clips/small_bdd_val.pkl \
                outputs/l5/clips/small_dance_val.pkl \
      --out outputs/l5/checkpoints/route_a_small --model small \
      --epochs 120 --gpu 6
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

from locatemot.models.l5_route_a import (  # noqa: E402
    L5TemporalAssociator,
    l5_association_loss,
    l5_relation_loss,
)

SPEC_TO_IDX = {"ALL": 0}


def spec_idx(spec):
    if spec in SPEC_TO_IDX:
        return SPEC_TO_IDX[spec]
    if spec.startswith("cat:"):
        SPEC_TO_IDX[spec] = 1
    else:
        SPEC_TO_IDX[spec] = 2
    return SPEC_TO_IDX[spec]


def load_clips(paths):
    out = []
    for p in paths:
        with open(p, "rb") as f:
            data = pickle.load(f)
        for vid, rec in data["videos"].items():
            for spec, views in rec["views"].items():
                for s in views["gt"]:
                    s = dict(s)
                    s.update({"video": vid, "spec": spec,
                              "dataset": data["domain"],
                              "image_size": rec["image_size"]})
                    out.append(s)
    return out


def _box_norm(box, image_size):
    w, h = float(image_size[0]), float(image_size[1])
    b = np.asarray(box, np.float64)
    return np.asarray([b[0] / w, b[1] / h, b[2] / w, b[3] / h], np.float32)


def build_obs_tensor(sample, K, device):
    """obs arrays for one sample: [T,K,2048], [T,K,9], [T,K]."""
    image_size = sample["image_size"]
    hist = sample["track_hist"]
    T = len(hist)
    pbd = np.zeros((T, K, 2048), np.float32)
    feat = np.zeros((T, K, 9), np.float32)
    mask = np.zeros((T, K), bool)
    for i, obs in enumerate(hist):
        obs = obs[-K:]
        start = K - len(obs)
        prev_center = None
        for j, (abs_frame, fp, ci, box, p, gt, gen, log_ncand) in enumerate(obs):
            k = start + j
            mask[i, k] = True
            pbd[i, k] = np.asarray(p, np.float32).reshape(2048)
            bn = _box_norm(box, image_size)
            feat[i, k, 0:4] = bn
            center = np.asarray([(box[0] + box[2]) / 2.0,
                                 (box[1] + box[3]) / 2.0], np.float64)
            if prev_center is not None:
                feat[i, k, 4:6] = (center - prev_center) / (
                    float(image_size[0]), float(image_size[1]))
            prev_center = center
            feat[i, k, 6] = float(gen)
            feat[i, k, 7] = float(log_ncand)
            # gap between this obs and the *current frame* (stored in sample)
            feat[i, k, 8] = float(sample["frame_id"] - int(abs_frame))
    return (torch.as_tensor(pbd, dtype=torch.float32, device=device),
            torch.as_tensor(feat, dtype=torch.float32, device=device),
            torch.as_tensor(mask, dtype=torch.bool, device=device))


def collate(batch):
    # batch: list of groups; each group is a list of samples with the same
    # (video, frame_id); flatten and remember group boundaries.
    flat = []
    group_id = []
    for gi, group in enumerate(batch):
        for s in group:
            flat.append(s)
            group_id.append(gi)
    batch = flat
    B = len(batch)
    T = max(len(s["track_hist"]) for s in batch)
    N = max(int(s["cand_feats"].shape[0]) for s in batch)
    K = max(min(16, max(len(h) for h in s["track_hist"])) for s in batch)
    K = max(1, K)
    Fp = batch[0]["pair_feats"].shape[-1]
    Ft = batch[0]["track_feats"].shape[-1]
    Fc = batch[0]["cand_feats"].shape[-1]
    out = {
        "obs_pbd": np.zeros((B, T, K, 2048), np.float32),
        "obs_feat": np.zeros((B, T, K, 9), np.float32),
        "obs_mask": np.zeros((B, T, K), bool),
        "cand_pbd": np.zeros((B, N, 2048), np.float32),
        "cand_feat": np.zeros((B, N, Fc), np.float32),
        "pair_feats": np.zeros((B, T, N, Fp), np.float32),
        "track_feats": np.zeros((B, T, Ft), np.float32),
        "base": np.zeros((B, T, N), np.float32),
        "row_label": np.full((B, T), -1, np.int64),
        "col_label": np.full((B, N), -1, np.int64),
        "base_correct": np.zeros((B, T), bool),
        "trk_mask": np.zeros((B, T), bool),
        "cand_mask": np.zeros((B, N), bool),
        "track_gt": np.full((B, T), -1, np.int64),
        "track_gt_str": [None] * B,
        "keep": [None] * B,
        "cand_gt_str": [None] * B,
        "spec": np.zeros(B, np.int64),
        "group_id": np.asarray(group_id, np.int64),
        "meta": [{"video": s["video"], "frame_id": s["frame_id"],
                  "spec": s["spec"], "dataset": s["dataset"]} for s in batch],
    }
    # GT id -> small int per sample
    for b, s in enumerate(batch):
        t = len(s["track_hist"])
        n = int(s["cand_feats"].shape[0])
        out["pair_feats"][b, :t, :n] = s["pair_feats"][:t, :n]
        out["track_feats"][b, :t] = s["track_feats"][:t]
        out["cand_feat"][b, :n] = s["cand_feats"][:n]
        out["base"][b, :t, :n] = s["base"][:t, :n]
        out["row_label"][b, :t] = s["row_label"][:t]
        out["col_label"][b, :n] = s["col_label"][:n]
        out["base_correct"][b, :t] = s["base_correct"][:t]
        out["trk_mask"][b, :t] = True
        out["cand_mask"][b, :n] = True
        out["spec"][b] = spec_idx(s["spec"])
        # candidate pbd from the video candidate table
        for j in range(n):
            ci = int(s["keep"][j])
            out["cand_pbd"][b, j] = s["_cand_pbd"][j]
        # obs tensors
        pb, ft, mk = build_obs_tensor(s, K, "cpu")
        out["obs_pbd"][b, :t] = pb.numpy()
        out["obs_feat"][b, :t] = ft.numpy()
        out["obs_mask"][b, :t] = mk.numpy()
        # track gt int ids
        gid_map = {}
        sup_gt = s.get("track_cur_gt", s["track_dom_gt"])
        for i, g in enumerate(sup_gt):
            if g is None:
                continue
            if g not in gid_map:
                gid_map[g] = len(gid_map)
            out["track_gt"][b, i] = gid_map[g]
        out["track_gt_str"][b] = list(sup_gt)
        out["keep"][b] = [int(i) for i in s["keep"]]
        out["cand_gt_str"][b] = list(s["_cand_gt"])
    return {k: torch.from_numpy(v)
            if k not in ("meta", "track_gt_str", "keep", "cand_gt_str") else v
            for k, v in out.items()}


class L5ClipDataset(Dataset):
    MAX_T = 48

    def __init__(self, groups, seed=20260806):
        self.groups = groups
        self.rng = np.random.RandomState(seed)
        w = []
        for g in groups:
            wrong = np.mean([float((s["base_correct"] == False).mean())
                             for s in g])
            w.append(max(0.05, 1.0 + 3.0 * wrong))
        w = np.asarray(w, np.float64)
        self.probs = w / w.sum()

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        if self.rng.rand() < 0.6:
            idx = int(self.rng.choice(len(self.groups), p=self.probs))
        return [self._cap(s) for s in self.groups[idx]]

    @staticmethod
    def _cap(s):
        T = len(s["track_hist"])
        if T <= L5ClipDataset.MAX_T:
            return s
        order = sorted(range(T), key=lambda i: (int(s["row_label"][i] < 0), i))
        keep = order[:L5ClipDataset.MAX_T]
        ns = dict(s)
        ns["track_hist"] = [s["track_hist"][i] for i in keep]
        ns["track_dom_gt"] = [s["track_dom_gt"][i] for i in keep]
        if "track_cur_gt" in s:
            ns["track_cur_gt"] = [s["track_cur_gt"][i] for i in keep]
        ns["row_label"] = s["row_label"][keep]
        ns["col_label"] = s["col_label"].copy()
        inv = {old: new for new, old in enumerate(keep)}
        for j in range(len(ns["col_label"])):
            old = int(s["col_label"][j])
            ns["col_label"][j] = inv.get(old, -1)
        ns["base"] = s["base"][keep]
        ns["pair_feats"] = s["pair_feats"][keep]
        ns["track_feats"] = s["track_feats"][keep]
        ns["base_correct"] = s["base_correct"][keep]
        return ns


def prepare_samples(samples, clips):
    """Attach per-sample candidate pbd arrays (from video candidate table)."""
    for s in samples:
        vid = s["video"]
        rec = clips[vid]
        fr = rec["cands"][s["frame"]]
        keep = s["keep"]
        s["_cand_pbd"] = [np.asarray(fr["pbd"][i], np.float32) for i in keep]
        s["_cand_gt"] = [fr["gt"][int(i)] for i in keep]
    return samples


def group_cross_spec(batch):
    """Return dict key -> list of (b, pred) for cross-spec consistency."""
    groups = defaultdict(list)
    for b, m in enumerate(batch["meta"]):
        groups[int(batch["group_id"][b])].append(b)
    return groups


def cross_spec_loss(model, batch, pred):
    """Assignment-level cross-spec consistency on common candidates.

    For every (video, frame, source) group with >= 2 views, build per-view
    softmax distributions over *common GT tracks* for each *common candidate*
    and minimise symmetric KL.  Both views are separately GT-supervised, so
    this never forces a view to imitate the other view's errors; it only
    removes ambiguity drift on evidence both views share.
    """
    groups = group_cross_spec(batch)
    losses = []
    n_common = 0
    for key, idxs in groups.items():
        if len(idxs) < 2:
            continue
        views = []
        for b in idxs:
            t = int(batch["trk_mask"][b].sum())
            views.append({
                "final": pred["final"][b, :t],
                "gts": np.asarray(batch["track_gt_str"][b])[:t].tolist(),
                "keep": batch["keep"][b],
            })
        for ai in range(len(views)):
            for bi in range(ai + 1, len(views)):
                va, vb = views[ai], views[bi]
                common_gt = set(va["gts"]) & set(vb["gts"])
                common_gt.discard(None)
                common_gt.discard(-1)
                common_gt.discard("None")
                common_cand = set(va["keep"]) & set(vb["keep"])
                if len(common_gt) < 2 or len(common_cand) < 1:
                    continue
                mats = []
                ok = True
                for v in (va, vb):
                    row_of = {}
                    for i, gid in enumerate(v["gts"]):
                        if gid in common_gt and gid not in row_of:
                            row_of[gid] = i
                    cand_loc = {full: j for j, full in enumerate(v["keep"])}
                    rows = [row_of[gid] for gid in sorted(common_gt)]
                    cols = [cand_loc[ci] for ci in sorted(common_cand)
                            if ci in cand_loc]
                    if not cols or len(rows) < 2:
                        ok = False
                        break
                    M = v["final"][rows][:, cols]  # [G, C]
                    mats.append(torch.log_softmax(M, dim=0))
                if not ok:
                    continue
                p0, p1 = mats[0], mats[1]
                kl = (p0.exp() * (p0 - p1)).sum(dim=0).mean() + \
                     (p1.exp() * (p1 - p0)).sum(dim=0).mean()
                losses.append(kl / 2.0)
                n_common += len(common_cand)
    if not losses:
        return None, 0
    return torch.stack(losses).mean(), n_common


def decode_assignments(final, threshold=0.25):
    """Hungarian 1-1 with NO_MATCH under threshold (L1D inference rule)."""
    from locatemot.tracking.association import hungarian_max
    final = final.cpu().numpy()
    final = np.nan_to_num(final, nan=-1e9, posinf=1e9, neginf=-1e9)
    T, N = final.shape
    out = np.full(T, -1, np.int64)
    if T > 0 and N > 0:
        for r, c in hungarian_max(final, threshold):
            out[r] = c
    return out


def eval_split(model, samples, device, max_samples=800):
    model.eval()
    rows = []
    rng = np.random.RandomState(0)
    idxs = rng.choice(len(samples), size=min(max_samples, len(samples)),
                      replace=False) if len(samples) > max_samples \
        else np.arange(len(samples))
    n_row = 0
    n_acc = 0
    with torch.no_grad():
        for i in range(0, len(idxs), 64):
            sub = [samples[int(j)] for j in idxs[i:i + 64]]
            batch = collate([[s] for s in sub])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            pred = model(batch)
            final = pred["final"]
            valid = batch["row_label"] >= 0
            if valid.any():
                fm = final.clone()
                B, T, N = fm.shape
                fm = fm.masked_fill(
                    ~batch["cand_mask"].unsqueeze(1).expand(B, T, N),
                    float("-inf"))
                acc = (fm[valid].argmax(-1) == batch["row_label"][valid])
                n_acc += int(acc.sum())
                n_row += int(valid.sum())
            for b, m in enumerate(batch["meta"]):
                t = int(batch["trk_mask"][b].sum())
                n = int(batch["cand_mask"][b].sum())
                assign = decode_assignments(final[b, :t, :n], 0.25)
                rows.append({
                    "video": m["video"], "frame_id": m["frame_id"],
                    "spec": m["spec"], "dataset": m["dataset"],
                    "assign": assign,
                    "track_gt": batch["track_gt"][b, :t].tolist(),
                    "cand_gt": [None],  # filled below by caller? keep light
                })
    return {"row_acc": n_acc / max(1, n_row), "n_row": n_row, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--val-data", nargs="+", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="base",
                    choices=["small", "base", "large"])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--rel-weight", type=float, default=0.0)
    ap.add_argument("--spec-weight", type=float, default=10.0)
    ap.add_argument("--pres-weight", type=float, default=0.0)
    ap.add_argument("--delta-scale", type=float, default=1.0)
    ap.add_argument("--source", default="mixed",
                    choices=["gt", "u0", "mixed"])
    ap.add_argument("--max-groups-per-video", type=int, default=700)
    ap.add_argument("--seed", type=int, default=20260806)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[l5a] device={device}", flush=True)

    clips = {}
    flat_samples = []
    sources = ["gt", "u0"] if args.source == "mixed" else [args.source]
    for p in args.data:
        with open(p, "rb") as f:
            d = pickle.load(f)
        clips.update(d["videos"])
        for vid, rec in d["videos"].items():
            for spec, views in rec["views"].items():
                for source in sources:
                    for s in views.get(source, []):
                        s = dict(s)
                        s.update({"video": vid, "spec": spec,
                                  "dataset": d["domain"],
                                  "image_size": rec["image_size"],
                                  "source": source})
                        flat_samples.append(s)
    flat_samples = prepare_samples(flat_samples, clips)
    groups = defaultdict(list)
    for s in flat_samples:
        groups[(s["dataset"], s["video"], int(s["frame_id"]),
                s.get("source", "gt"))].append(s)
    if args.max_groups_per_video > 0:
        per_video = defaultdict(list)
        for key, g in groups.items():
            per_video[key[1]].append((int(key[2]), g))
        capped = []
        for vid, items in per_video.items():
            items.sort(key=lambda x: x[0])
            step = max(1, len(items) // args.max_groups_per_video)
            capped.extend(g for _, g in items[::step])
        samples = capped
    else:
        samples = list(groups.values())
    val_samples = []
    val_clips = {}
    for p in args.val_data:
        with open(p, "rb") as f:
            d = pickle.load(f)
        val_clips.update(d["videos"])
        for vid, rec in d["videos"].items():
            for spec, views in rec["views"].items():
                for source in sources:
                    for s in views.get(source, []):
                        s = dict(s)
                        s.update({"video": vid, "spec": spec,
                                  "dataset": d["domain"],
                                  "image_size": rec["image_size"],
                                  "source": source})
                        val_samples.append(s)
    val_samples = prepare_samples(val_samples, val_clips)
    print(f"[l5a] train_groups={len(samples)} train_samples="
          f"{len(flat_samples)} val_samples={len(val_samples)}", flush=True)

    sizes = {
        "small": dict(d_model=128, temporal_layers=2, set_layers=2,
                      n_heads=4, ffn_dim=512),
        "base": dict(d_model=256, temporal_layers=4, set_layers=4,
                     n_heads=8, ffn_dim=1024),
        "large": dict(d_model=384, temporal_layers=6, set_layers=6,
                      n_heads=8, ffn_dim=1536),
    }
    model = L5TemporalAssociator(**sizes[args.model],
                                 delta_scale=args.delta_scale).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[l5a] model={args.model} params={n_params/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ds = L5ClipDataset(samples, seed=args.seed)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
        num_workers=4, persistent_workers=True, drop_last=False)
    total_steps = args.epochs * max(1, len(loader))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps, pct_start=0.05,
        anneal_strategy="cos")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = vars(args)
    cfg["n_params"] = n_params
    cfg["n_samples"] = len(flat_samples)
    with open(out_dir / "train_config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    curve = []
    step = 0
    t0 = time.time()
    save_epochs = {1, 5, 10, 20, 40, 80, 120}
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        ep_assoc = 0.0
        ep_rel = 0.0
        ep_spec = 0.0
        ep_n = 0
        ep_acc = 0
        ep_nrow = 0
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            opt.zero_grad(set_to_none=True)
            pred = model(batch)
            a = l5_association_loss(batch, pred, pres_weight=args.pres_weight)
            r, rn = l5_relation_loss(batch, pred)
            c, cn = cross_spec_loss(model, batch, pred)
            loss = a["loss"] + args.rel_weight * r
            if c is not None:
                loss = loss + args.spec_weight * c
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            ep_loss += float(loss)
            ep_assoc += float(a["loss"])
            ep_rel += float(r)
            if c is not None:
                ep_spec += float(c)
            ep_n += 1
            valid = batch["row_label"] >= 0
            if valid.any():
                fm = pred["final"].clone()
                B, T, N = fm.shape
                fm = fm.masked_fill(
                    ~batch["cand_mask"].unsqueeze(1).expand(B, T, N),
                    float("-inf"))
                ep_acc += int((fm[valid].argmax(-1)
                               == batch["row_label"][valid]).sum())
                ep_nrow += int(valid.sum())
            if step % 20 == 0:
                print(f"[l5a] ep={epoch} step={step} loss={float(loss):.4f} "
                      f"assoc={float(a['loss']):.4f} rel={float(r):.4f} "
                      f"spec={float(c) if c is not None else 0:.4f} "
                      f"rowacc={ep_acc/max(1,ep_nrow):.3f} "
                      f"lr={sched.get_last_lr()[0]:.2e} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
        val = eval_split(model, val_samples, device) if val_samples else None
        row = {
            "epoch": epoch, "loss": ep_loss / max(1, ep_n),
            "assoc": ep_assoc / max(1, ep_n),
            "rel": ep_rel / max(1, ep_n),
            "spec": ep_spec / max(1, ep_n),
            "train_row_acc": ep_acc / max(1, ep_nrow),
            "val_row_acc": val["row_acc"] if val else None,
        }
        curve.append(row)
        print(f"[l5a] epoch={epoch} " +
              " ".join(f"{k}={v:.4f}" if isinstance(v, float) else
                       f"{k}={v}" for k, v in row.items()), flush=True)
        if epoch in save_epochs or epoch == args.epochs:
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "cfg": cfg, "curve": curve},
                       out_dir / f"epoch{epoch}.pt")
            with open(out_dir / "latest.pt", "wb") as f:
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "cfg": cfg, "curve": curve}, f)
        with open(out_dir / "learning_curve.json", "w") as f:
            json.dump(curve, f, indent=2)
    torch.save({"model": model.state_dict(), "epoch": args.epochs,
                "cfg": cfg, "curve": curve}, out_dir / "final.pt")
    print(f"[l5a] done seconds={time.time()-t0:.1f}", flush=True)


if __name__ == "__main__":
    main()
