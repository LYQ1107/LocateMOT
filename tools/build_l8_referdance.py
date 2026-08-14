"""Stage L8: build Refer-Dance RMOT training/eval caches.

Training cache: one pkl per train video with aligned PBD (L6) + CLIP (L7)
candidate tokens, plus expressions.json containing per-expression spec
embeddings and per-frame target object ids.  Frames are NOT duplicated per
expression.

Eval metadata: expressions.json for the 25 val sequences (425 queries) with
frozen CLIP text embeddings, used by the RMOT evaluator.

Usage:
  python tools/build_l8_referdance.py
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "refer_dance"
L6 = ROOT / "outputs" / "l6" / "data" / "dancetrack_train"
L7 = ROOT / "outputs" / "l7" / "data" / "clip_closed" / "dancetrack_train"
L6_CAL = ROOT / "outputs" / "l6" / "data" / "dancetrack_calibration"
L7_CAL = ROOT / "outputs" / "l7" / "data" / "clip_closed" / "dancetrack_calibration"
OUT = ROOT / "outputs" / "l8" / "data"


def load_expressions(videos):
    out = {}
    for vid in videos:
        d = DATA / "expression" / vid
        if not d.is_dir():
            continue
        exps = []
        for p in sorted(d.glob("*.json")):
            obj = json.loads(p.read_text())
            exps.append({
                "expression": p.stem,
                "sentence": obj.get("sentence", p.stem),
                "label": {str(k): [str(x) for x in v]
                          for k, v in obj.get("label", {}).items()},
            })
        out[vid] = exps
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rmot_train").mkdir(parents=True, exist_ok=True)
    (OUT / "rmot_eval").mkdir(parents=True, exist_ok=True)
    import clip
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cm, _ = clip.load("ViT-B/32", device=device)
    cm.eval()

    seqmap = [l.strip() for l in (DATA / "seqmap.txt").read_text().splitlines()
              if l.strip()]
    val_videos = sorted({l.split("+", 1)[0] for l in seqmap})
    all_videos = sorted(p.name for p in (DATA / "expression").iterdir()
                        if p.is_dir())
    train_videos = [v for v in all_videos if v not in set(val_videos)]
    print(f"[l8data] train videos={len(train_videos)} "
          f"val videos={len(val_videos)}", flush=True)

    all_exps = load_expressions(train_videos + val_videos)

    def embed(texts):
        toks = clip.tokenize(texts).to(device)
        with torch.no_grad():
            feats = cm.encode_text(toks).float().cpu().numpy()
        feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        return feats

    # ---- training cache ----
    exp_meta = {}
    n_frames = 0
    for vid in sorted(train_videos):
        l6p = L6 / f"{vid}.pkl"
        l7p = L7 / f"{vid}.pkl"
        if not l6p.exists() or not l7p.exists():
            l6p = L6_CAL / f"{vid}.pkl"
            l7p = L7_CAL / f"{vid}.pkl"
        if not l6p.exists() or not l7p.exists():
            print(f"[l8data] skip {vid} missing cache", flush=True)
            continue
        r6 = pickle.load(open(l6p, "rb"))
        r7 = pickle.load(open(l7p, "rb"))
        assert r6["video_id"] == r7["video_id"]
        f6 = {fr["frame"]: fr for fr in r6["frames"]}
        f7 = {fr["frame"]: fr for fr in r7["frames"]}
        assert f6.keys() == f7.keys()
        frames = []
        for fid in sorted(f6):
            a, b = f6[fid], f7[fid]
            assert len(a["boxes"]) == len(b["boxes"]) == len(a["pbd"]) \
                == len(b["clip"])
            frames.append({
                "frame": fid,
                "boxes": a["boxes"],
                "pbd": np.asarray(a["pbd"], np.float16),
                "clip": np.asarray(b["clip"], np.float16),
                "gen": a["gen"],
                "cand_gt": a["cand_gt"],
                "gt_boxes": a["gt_boxes"],
            })
        n_frames += len(frames)
        rec = {"video_id": vid, "image_size": r6["image_size"],
               "frames": frames}
        with open(OUT / "rmot_train" / f"{vid}.pkl", "wb") as f:
            pickle.dump(rec, f)
        # expression metadata (only expressions with targets + 2 negatives)
        exps = [e for e in all_exps.get(vid, []) if e["label"]]
        negs = [e for e in all_exps.get(vid, []) if not e["label"]]
        rng = np.random.RandomState(20260806)
        if negs:
            rng.shuffle(negs)
            exps = exps + negs[:2]
        sentences = [e["sentence"] for e in exps]
        specs = embed(sentences)
        exp_meta[vid] = [
            {**e, "spec": specs[i].astype(np.float32).tolist()}
            for i, e in enumerate(exps)
        ]
        print(f"[l8data] {vid} frames={len(frames)} exps={len(exps)}",
              flush=True)
    with open(OUT / "rmot_train" / "expressions.json", "w") as f:
        json.dump(exp_meta, f, indent=1)
    print(f"[l8data] train cache done: videos={len(exp_meta)} "
          f"frames={n_frames} exps={sum(len(v) for v in exp_meta.values())}",
          flush=True)

    # ---- eval metadata (val queries) ----
    ev = {}
    seqmap = [l.strip() for l in (DATA / "seqmap.txt").read_text().splitlines()
              if l.strip()]
    for line in seqmap:
        vid, expr = line.split("+", 1)
        exps = all_exps.get(vid, [])
        hit = next((e for e in exps if e["expression"] == expr), None)
        if hit is None:
            hit = {"expression": expr, "sentence": expr.replace("-", " "),
                   "label": {}}
        ev.setdefault(vid, []).append(hit)
    ev_specs = {}
    for vid, exps in ev.items():
        sentences = [e["sentence"] for e in exps]
        specs = embed(sentences)
        ev_specs[vid] = [
            {**e, "spec": specs[i].astype(np.float32).tolist()}
            for i, e in enumerate(exps)
        ]
    with open(OUT / "rmot_eval" / "expressions.json", "w") as f:
        json.dump(ev_specs, f, indent=1)
    print(f"[l8data] eval metadata done: videos={len(ev_specs)} "
          f"queries={sum(len(v) for v in ev_specs.values())}", flush=True)


if __name__ == "__main__":
    main()
