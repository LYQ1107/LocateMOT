"""Stage L5 Route A: online drift + TrackEval with the model's own rollouts.

Runs OnlineTracker(variant=L5) on ALL and restricted views, then computes
the video-level globally-aligned track-ID disagreement (same metric as
`l5_drift_eval.py` but using the model's own track chains instead of frozen
U0 histories).

Usage:
  python tools/l5_online_eval.py \
      --ckpt outputs/l5/checkpoints/route_a_small/epoch10.pt \
      --manifest outputs/l1_c/fixed_candidate_manifest/bdd100k_train.jsonl \
      --domain bdd100k --gpu 3 --out outputs/l5/online_drift_bdd.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from locatemot.models.l5_route_a import L5TemporalAssociator  # noqa: E402
from locatemot.models.l5_route_b import L5IdentityPredictor  # noqa: E402
from locatemot.models.l1d_association import L1DAssociator  # noqa: E402
from locatemot.models.l6_uidm import UIDM  # noqa: E402
from locatemot.tracking.online_tracker import OnlineTracker  # noqa: E402
from tools.eval_l3 import build_candidates  # noqa: E402
from tools.l4_restriction_audit import spec_mask  # noqa: E402


SIZES = {
    "small": dict(d_model=128, temporal_layers=2, set_layers=2,
                  n_heads=4, ffn_dim=512),
    "base": dict(d_model=256, temporal_layers=4, set_layers=4,
                 n_heads=8, ffn_dim=1024),
    "large": dict(d_model=384, temporal_layers=6, set_layers=6,
                  n_heads=8, ffn_dim=1536),
}
SIZES_UIDM = {
    "small": dict(d_model=192, n_layers=3, n_heads=4, ffn_dim=768),
    "base": dict(d_model=320, n_layers=4, n_heads=8, ffn_dim=1280),
    "large": dict(d_model=384, n_layers=6, n_heads=8, ffn_dim=1536),
}


def run_view(model, entries, spec, device):
    if getattr(model, "memory", None) is not None and hasattr(model, "d_model"):
        tracker = OnlineTracker(variant="UIDM", uidm=model,
                                device=str(device),
                                output_all_candidates=True)
    elif getattr(model, "slot_head", None) is not None:
        tracker = OnlineTracker(variant="L5B", l5b=model, device=str(device),
                                output_all_candidates=True)
    elif isinstance(model, L5TemporalAssociator):
        tracker = OnlineTracker(variant="L5", l5=model, device=str(device),
                                output_all_candidates=True)
    else:
        tracker = OnlineTracker(variant="L1D", l1d=model, device=str(device),
                                output_all_candidates=True)
    tracker.l1d_weights = (0.4, 0.2, 0.4)
    tracker.l1d_threshold = 0.25
    rows = []
    for e in entries:
        cands, image_size = build_candidates(e)
        tracker.image_size = image_size
        keep, _gt, _cg = spec_mask(e, spec)
        restricted = [cands[i] for i in keep]
        outputs = tracker.process_frame(int(e["frame"]), restricted)
        for o, ci in zip(outputs, keep):
            rows.append((int(e["frame"]), int(ci), int(o["track_id"])))
    return rows


def compute_drift(all_rows, rest_rows):
    all_map = {(f, c): t for f, c, t in all_rows}
    rest_map = {(f, c): t for f, c, t in rest_rows}
    common = sorted(set(all_map) & set(rest_map))
    pairs = [(all_map[k], rest_map[k]) for k in common]
    if not pairs:
        return 0.0, 0, 0
    ids_a = sorted({p[0] for p in pairs})
    ids_b = sorted({p[1] for p in pairs})
    ia = {v: i for i, v in enumerate(ids_a)}
    ib = {v: i for i, v in enumerate(ids_b)}
    cnt = np.zeros((len(ids_a), len(ids_b)))
    for x, y in pairs:
        cnt[ia[x], ib[y]] += 1
    rows, cols = linear_sum_assignment(-cnt)
    mapping = {ids_a[r]: ids_b[c] for r, c in zip(rows, cols)}
    dis = sum(1 for x, y in pairs if mapping.get(x, x) != y)
    return dis / max(1, len(pairs)), dis, len(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--domain", required=True)
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--model-type", default="l5",
                    choices=["l5", "u0", "l5b", "uidm"])
    ap.add_argument("--videos", nargs="*", default=None)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if args.model_type == "uidm":
        model = UIDM(
            **SIZES_UIDM[ck["cfg"].get("model", "base")],
            no_interaction=ck["cfg"].get("no_interaction", False))
        model.load_state_dict(ck["model"])
    elif args.model_type == "u0":
        state = ck["model"] if "model" in ck else ck
        model = L1DAssociator()
        model.load_state_dict(state)
    elif args.model_type == "l5b":
        model = L5IdentityPredictor(
            **SIZES[ck["cfg"]["model"]],
            max_slots=ck["cfg"].get("max_slots", 128))
        model.load_state_dict(ck["model"])
    else:
        model = L5TemporalAssociator(
            **SIZES[ck["cfg"]["model"]],
            delta_scale=ck["cfg"].get("delta_scale", 0.6))
        model.load_state_dict(ck["model"])
    model = model.to(device)
    model.eval()
    by_video = defaultdict(list)
    with open(args.manifest) as f:
        for line in f:
            e = json.loads(line)
            by_video[e["video_id"]].append(e)
    if args.videos:
        by_video = {v: by_video[v] for v in args.videos if v in by_video}
    if args.max_videos:
        by_video = dict(list(by_video.items())[:args.max_videos])
    results = {}
    for vid, entries in by_video.items():
        entries.sort(key=lambda e: e["frame"])
        all_rows = run_view(model, entries, "ALL", device)
        specs = []
        if args.domain == "bdd100k":
            cats = set()
            for e in entries:
                cats.update(e.get("gt_categories", {}).values())
            specs = [f"cat:{c}" for c in ("car", "truck", "bus", "pedestrian")
                     if c in cats]
        else:
            from collections import Counter
            gt_len = Counter()
            for e in entries:
                gt_len.update(e.get("gt_boxes", {}).keys())
            specs = [f"inst:{g}" for g, _ in
                     sorted(gt_len.items(), key=lambda x: -x[1])[:2]]
        vres = {}
        for spec in specs:
            rest_rows = run_view(model, entries, spec, device)
            rate, dis, n = compute_drift(all_rows, rest_rows)
            vres[spec] = {"drift_rate": rate, "drift": dis, "common": n}
            print(f"[online] {vid} {spec} drift={rate:.4f} "
                  f"({dis}/{n})", flush=True)
        results[vid] = vres
    total_n = sum(v[s]["common"] for v in results.values() for s in v)
    total_d = sum(v[s]["drift"] for v in results.values() for s in v)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out = {
        "ckpt": args.ckpt,
        "per_video": results,
        "total_common": total_n,
        "total_drift": total_d,
        "drift_rate": total_d / max(1, total_n),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "per_video"},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
