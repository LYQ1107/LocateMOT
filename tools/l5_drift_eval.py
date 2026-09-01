"""Stage L5: cross-spec identity drift evaluation on clip data.

Computes, for every (video, frame) with >= 2 views:
  - candidate-level assignment (track -> dominant GT) per view;
  - common-candidate drift: fraction of candidates present in both views
    whose assigned track GT differs;
  - P0/P1 event classification (Type 1..5).

Scorers:
  --scorer base   : frozen L1DK base affinity (argmax / Hungarian 0.25).
  --scorer model  : L5 Route A model checkpoint (final affinity).

Usage:
  python tools/l5_drift_eval.py --clips outputs/l5/clips/small_bdd_val.pkl \
      --out outputs/l5/drift_base.json --scorer base
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from locatemot.models.l5_route_a import L5TemporalAssociator  # noqa: E402
from locatemot.tracking.association import hungarian_max  # noqa: E402


def decode(final, threshold=0.25):
    final = np.nan_to_num(final, nan=-1e9, posinf=1e9, neginf=-1e9)
    T, N = final.shape
    out = np.full(T, -1, np.int64)
    if T > 0 and N > 0:
        for r, c in hungarian_max(final, threshold):
            out[r] = c
    return out


def load_samples(clip_paths, source="u0"):
    out = []
    for p in clip_paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        for vid, rec in d["videos"].items():
            for spec, views in rec["views"].items():
                for s in views.get(source, []):
                    s = dict(s)
                    s.update({"video": vid, "spec": spec,
                              "dataset": d["domain"],
                              "image_size": rec["image_size"]})
                    # candidate gt: look up from cand table
                    fr = rec["cands"][s["frame"]]
                    cand_gt = [fr["gt"][int(i)] for i in s["keep"]]
                    s["_cand_gt"] = cand_gt
                    s["_cand_idx"] = s["keep"]
                    s["_cand_pbd"] = [
                        np.asarray(fr["pbd"][int(i)], np.float32)
                        for i in s["keep"]]
                    out.append(s)
    return out


def model_forward(model, samples, device):
    """Return per-sample decoded assignments using the model."""
    from tools.train_l5_route_a import collate
    model.eval()
    assigns = {}
    with torch.no_grad():
        for i in range(0, len(samples), 64):
            sub = samples[i:i + 64]
            batch = collate([[s] for s in sub])
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            pred = model(batch)
            for b, s in enumerate(sub):
                t = int(batch["trk_mask"][b].sum())
                n = int(batch["cand_mask"][b].sum())
                assigns[id(s)] = decode(
                    pred["final"][b, :t, :n].cpu().numpy(), 0.25)
    return assigns


def eval_drift(samples, assigns=None, scorer="base"):
    """Primary: video-level globally-aligned track-ID disagreement.

    For each (video, spec pair), collect (tidA, tidB) for common candidates
    across all frames, align track IDs with one global Hungarian on
    co-occurrence counts, and report the disagreement rate.  This is the
    L4-consistent persistent-identity drift metric.
    """
    groups = defaultdict(list)
    for s in samples:
        groups[(s["dataset"], s["video"], int(s["frame_id"]))].append(s)
    vids = defaultdict(list)
    for key, ss in groups.items():
        all_view = next((s for s in ss if s["spec"] == "ALL"), None)
        if all_view is None:
            continue
        for s in ss:
            if s["spec"] == "ALL":
                continue
            vids[(key[0], key[1], s["spec"])].append((all_view, s))
    per_domain = defaultdict(lambda: {"common": 0, "drift": 0})
    total = 0
    drift = 0
    type_counts = defaultdict(int)
    for key, pairs in vids.items():
        all_pairs = []
        for all_view, s in pairs:
            a0 = decode(all_view["base"]) if scorer == "base" \
                else assigns[id(all_view)]
            a1 = decode(s["base"]) if scorer == "base" else assigns[id(s)]
            tid0 = {}
            for r, c in enumerate(a0):
                if c >= 0:
                    tid0[int(all_view["keep"][c])] = int(all_view["track_tid"][r])
            tid1 = {}
            for r, c in enumerate(a1):
                if c >= 0:
                    tid1[int(s["keep"][c])] = int(s["track_tid"][r])
            for ci in sorted(set(tid0) & set(tid1)):
                all_pairs.append((tid0[ci], tid1[ci]))
            # secondary: cur_gt event classification
            sup0 = all_view.get("track_cur_gt", all_view["track_dom_gt"])
            sup1 = s.get("track_cur_gt", s["track_dom_gt"])
            g0map = {int(all_view["keep"][c]): sup0[r]
                     for r, c in enumerate(a0) if c >= 0}
            g1map = {int(s["keep"][c]): sup1[r]
                     for r, c in enumerate(a1) if c >= 0}
            gt_by_idx = {int(all_view["keep"][j]): g
                         for j, g in enumerate(all_view["_cand_gt"])}
            for ci in sorted(set(g0map) & set(g1map)):
                g0 = g0map[ci]
                g1 = g1map[ci]
                gt = gt_by_idx.get(ci)
                if g0 == gt and g1 == gt:
                    typ = 1
                elif g0 != gt and g1 == gt:
                    typ = 2
                elif g0 == gt and g1 != gt:
                    typ = 3
                elif g0 != gt and g1 != gt and g0 == g1:
                    typ = 4
                else:
                    typ = 5
                type_counts[typ] += 1
        if not all_pairs:
            continue
        ids_a = sorted({p[0] for p in all_pairs})
        ids_b = sorted({p[1] for p in all_pairs})
        ia = {v: i for i, v in enumerate(ids_a)}
        ib = {v: i for i, v in enumerate(ids_b)}
        cnt = np.zeros((len(ids_a), len(ids_b)))
        for x, y in all_pairs:
            cnt[ia[x], ib[y]] += 1
        rows, cols = linear_sum_assignment(-cnt)
        mapping = {ids_a[r]: ids_b[c] for r, c in zip(rows, cols)}
        dis = sum(1 for x, y in all_pairs if mapping.get(x, x) != y)
        total += len(all_pairs)
        drift += dis
        per_domain[key[0]]["common"] += len(all_pairs)
        per_domain[key[0]]["drift"] += dis
    return {
        "common_candidates": total,
        "drift_candidates": drift,
        "drift_rate": drift / max(1, total),
        "type_counts": dict(type_counts),
        "per_domain": {k: {"common": v["common"], "drift": v["drift"],
                           "drift_rate": v["drift"] / max(1, v["common"])}
                       for k, v in per_domain.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scorer", default="base", choices=["base", "model"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--gpu", type=int, default=6)
    ap.add_argument("--source", default="u0", choices=["gt", "u0"])
    args = ap.parse_args()
    samples = load_samples(args.clips, source=args.source)
    assigns = None
    device = None
    if args.scorer == "model":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(args.ckpt, map_location=device)
        sizes = {
            "small": dict(d_model=128, temporal_layers=2, set_layers=2,
                          n_heads=4, ffn_dim=512),
            "base": dict(d_model=256, temporal_layers=4, set_layers=4,
                         n_heads=8, ffn_dim=1024),
            "large": dict(d_model=384, temporal_layers=6, set_layers=6,
                          n_heads=8, ffn_dim=1536),
        }
        model = L5TemporalAssociator(
            **sizes[ck["cfg"]["model"]],
            delta_scale=ck["cfg"].get("delta_scale", 0.6)).to(device)
        model.load_state_dict(ck["model"])
        assigns = model_forward(model, samples, device)
    res = eval_drift(samples, assigns, scorer=args.scorer)
    res["n_samples"] = len(samples)
    res["scorer"] = args.scorer
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
