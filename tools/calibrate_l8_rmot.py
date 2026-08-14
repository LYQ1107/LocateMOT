"""Stage L8: calibrate the RMOT relevance threshold on Refer-Dance train.

Aggregates per-candidate relevance logits and expression target labels over
the RMOT training split, then picks the logit threshold that maximizes
candidate-level F1.  The chosen threshold is saved for the official
Refer-Dance evaluation (post-hoc scalar calibration; protocol-comparable to
iKUN's similarity-calibration step).

Usage:
  python tools/calibrate_l8_rmot.py --ckpt outputs/l8/checkpoints/.../latest.pt
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

from locatemot.models.l8_unified import L8UnifiedUIDM  # noqa: E402

SIZES = {
    "small": dict(d_model=192, n_layers=3, n_heads=4, ffn_dim=768),
    "base": dict(d_model=320, n_layers=4, n_heads=8, ffn_dim=1280),
    "large": dict(d_model=384, n_layers=6, n_heads=8, ffn_dim=1536),
}
RMOT = ROOT / "outputs" / "l8" / "data" / "rmot_train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="outputs/l8/calib/threshold.json")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-clips", type=int, default=0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    model = L8UnifiedUIDM(**SIZES[cfg.get("model", "base")],
                          mode=cfg.get("mode", "unified")).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    adapter = model.adapter
    meta = json.loads((RMOT / "expressions.json").read_text())
    logits, labels = [], []
    clips = [(v, e) for v, es in meta.items() for e in es]
    if args.max_clips:
        clips = clips[:args.max_clips]
    for vi, (vid, e) in enumerate(clips):
        rec = pickle.load(open(RMOT / f"{vid}.pkl", "rb"))
        spec = torch.as_tensor(e["spec"], device=device)
        ids = e["label"]
        for fr in rec["frames"]:
            n = len(fr["boxes"])
            if n == 0:
                continue
            pbd = torch.as_tensor(np.asarray(fr["pbd"], np.float32),
                                  device=device)
            clip = torch.as_tensor(np.asarray(fr["clip"], np.float32),
                                   device=device)
            with torch.no_grad():
                rel = adapter(pbd, clip, spec[None].expand(n, -1))[1].cpu()
            tgt = {str(x) for x in ids.get(str(fr["frame"]), [])}
            y = np.zeros(n, np.float32)
            for j, gid in enumerate(fr["cand_gt"]):
                if gid is not None and gid in tgt:
                    y[j] = 1.0
            logits.append(rel.numpy())
            labels.append(y)
        if (vi + 1) % 25 == 0:
            print(f"[l8calib] {vi+1}/{len(clips)}", flush=True)
    L = np.concatenate(logits)
    Y = np.concatenate(labels)
    print(f"[l8calib] n={len(L)} pos={Y.sum():.0f} "
          f"neg={(1-Y).sum():.0f}", flush=True)
    best = None
    rows = []
    for t in np.arange(-2.0, 2.01, 0.05):
        p = (L > t)
        tp = (p & (Y == 1)).sum()
        fp = (p & (Y == 0)).sum()
        fn = (~p & (Y == 1)).sum()
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        rows.append({"t": round(float(t), 2), "prec": float(prec),
                     "rec": float(rec), "f1": float(f1)})
        if best is None or f1 > best["f1"]:
            best = rows[-1]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "threshold": best["t"], "f1": best["f1"],
        "precision": best["prec"], "recall": best["rec"],
        "curve": rows[:50],
    }, indent=1))
    print(f"[l8calib] best threshold={best['t']} f1={best['f1']:.4f} "
          f"p={best['prec']:.4f} r={best['rec']:.4f} -> {out_path}",
          flush=True)


if __name__ == "__main__":
    main()
