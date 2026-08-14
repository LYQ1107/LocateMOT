"""Stage L8: RMOT evaluation with the shared UIDM + Unified Observation.

For each (video, expression) query, run the same UIDM online tracker with
the expression spec embedding, keep candidates whose relevance logit passes
the threshold, write TrackEval RMOT `predict.txt`, and run the patched
official RMOT TrackEval runner (HOTA threshold 0.5).

Usage:
  python tools/eval_l8_rmot.py --ckpt outputs/l8/checkpoints/smoke/latest.pt \
      --out outputs/l8/trackeval/smoke --gpu 0 --gt-only
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(ROOT))

from locatemot.models.l6_uidm import UIDM  # noqa: E402
from locatemot.models.l8_unified import L8UnifiedUIDM  # noqa: E402
from locatemot.tracking.online_tracker import OnlineTracker  # noqa: E402
from tools.eval_l3 import build_candidates  # noqa: E402

PY = "/home/lwr/anaconda3/envs/locatemot/bin/python"
MANIFEST = ROOT / "outputs" / "l1_c" / "fixed_candidate_manifest" \
    / "dancetrack_val.jsonl"
CLIP_DIR = ROOT / "outputs" / "l7" / "data" / "clip_eval" / "dancetrack_val"
GT_TEMPLATE = ROOT / "data" / "refer_dance" / "gt_template"
SEQMAP = ROOT / "data" / "refer_dance" / "seqmap.txt"
EVAL_RUN = (ROOT / "references" / "l8" / "TrackEval_rmot" / "scripts"
            / "run_mot_challenge.py")
SIZES = {
    "small": dict(d_model=192, n_layers=3, n_heads=4, ffn_dim=768),
    "base": dict(d_model=320, n_layers=4, n_heads=8, ffn_dim=1280),
    "large": dict(d_model=384, n_layers=6, n_heads=8, ffn_dim=1536),
}


def load_manifest():
    by_video = {}
    with open(MANIFEST) as f:
        for line in f:
            e = json.loads(line)
            by_video.setdefault(e["video_id"], []).append(e)
    for v in by_video:
        by_video[v].sort(key=lambda x: int(x["frame"]))
    return by_video


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--gt-only", action="store_true", default=True)
    ap.add_argument("--max-videos", type=int, default=0)
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    size = cfg.get("model", "base")
    mode = cfg.get("mode", "unified")
    model = L8UnifiedUIDM(
        **SIZES[size],
        no_interaction=cfg.get("no_interaction", False),
        use_cue_rel=cfg.get("use_cue_rel", False),
        mode=mode).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    core = model.uidm
    adapter = model.adapter
    print(f"[l8rmot] size={size} mode={mode} "
          f"trainable={sum(p.numel() for p in model.parameters())/1e6:.2f}M",
          flush=True)

    exp_meta = json.loads(
        (ROOT / "outputs" / "l8" / "data" / "rmot_eval" /
         "expressions.json").read_text())
    gt_queries = set()
    if args.gt_only:
        for root, _dirs, files in os.walk(GT_TEMPLATE):
            if "gt.txt" in files and os.path.getsize(
                    os.path.join(root, "gt.txt")) > 0:
                rel = os.path.relpath(root, GT_TEMPLATE)
                gt_queries.add(rel.replace(os.sep, "/"))
    queries = []
    for line in SEQMAP.read_text().splitlines():
        if not line.strip():
            continue
        vid, expr = line.strip().split("+", 1)
        if args.gt_only and f"{vid}/{expr}" not in gt_queries:
            continue
        meta = next((e for e in exp_meta.get(vid, [])
                     if e["expression"] == expr), None)
        if meta is None:
            continue
        queries.append((vid, expr, np.asarray(meta["spec"], np.float32)))
    print(f"[l8rmot] queries={len(queries)}", flush=True)

    by_video = load_manifest()
    vids = [q[0] for q in queries]
    if args.max_videos:
        vids = vids[:args.max_videos]
    out_root = Path(args.out)
    res_root = out_root / "uidm8"
    t0 = time.time()
    for vi, vid in enumerate(sorted(set(vids))):
        entries = by_video[vid]
        clip_rec = pickle.load(open(CLIP_DIR / f"{vid}.pkl", "rb"))
        clip_frames = {fr["frame"]: fr for fr in clip_rec["frames"]}
        for e in entries:
            assert int(e["frame"]) in clip_frames
        vid_queries = [q for q in queries if q[0] == vid]
        for qi, (_, expr, spec) in enumerate(vid_queries):
            tracker = OnlineTracker(
                variant="UIDM", uidm=core, device=str(device),
                output_all_candidates=True,
                uidm_adapter=adapter, uidm_spec=spec)
            tracker.uidm_new_margin = 0.0
            tracker.l1d_weights = (0.4, 0.2, 0.4)
            tracker.l1d_threshold = 0.25
            rows = []
            for entry in entries:
                frame = int(entry["frame"])
                cands, image_size = build_candidates(entry)
                cfr = clip_frames[frame]
                assert len(cands) == len(cfr["boxes"])
                for j, c in enumerate(cands):
                    c["features"]["clip"] = np.asarray(
                        cfr["clip"][j], np.float32)
                tracker.image_size = image_size
                if len(cands) == 0:
                    continue
                # per-candidate relevance (frozen adapter at eval time)
                pbd = np.stack([
                    c["features"].get("pbd", np.zeros(2048, np.float32))
                    for c in cands]).astype(np.float32)
                clip = np.stack([c["features"]["clip"] for c in cands])
                with torch.no_grad():
                    rel = adapter(
                        torch.as_tensor(pbd, device=device),
                        torch.as_tensor(clip, device=device),
                        torch.as_tensor(spec[None], device=device))[1]
                    rel = rel.cpu().numpy()
                outputs = tracker.process_frame(frame, cands)
                assert len(outputs) == len(cands)
                for o, r in zip(outputs, rel):
                    if float(r) <= args.threshold:
                        continue
                    x1, y1, x2, y2 = o["box"]
                    rows.append([frame, o["track_id"], x1, y1,
                                 x2 - x1, y2 - y1,
                                 float(o.get("score", 1.0)), -1, -1, -1])
            exp_dir = res_root / vid / expr
            exp_dir.mkdir(parents=True, exist_ok=True)
            gt_src = GT_TEMPLATE / vid / expr / "gt.txt"
            gt_dst = exp_dir / "gt.txt"
            if gt_src.exists() and not gt_dst.exists():
                gt_dst.symlink_to(gt_src)
            with open(exp_dir / "predict.txt", "w") as f:
                for r in rows:
                    f.write(",".join(
                        f"{v:.3f}" if isinstance(v, float) else str(v)
                        for v in r) + "\n")
        print(f"[l8rmot] {vid} {vi+1}/{len(set(vids))} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
    print("[l8rmot] predictions written", flush=True)

    eval_vids = set(vids)
    seqmap40 = out_root / "seqmap_gt.txt"
    with open(seqmap40, "w") as f:
        for vid, expr, _ in queries:
            if vid not in eval_vids:
                continue
            f.write(f"{vid}+{expr}\n")
    res_root = res_root.resolve()
    seqmap40 = seqmap40.resolve()
    env = dict(os.environ)
    env["RMOT_IMG_ROOT"] = str(
        ROOT / "data" / "refer_dance" / "DanceTrack" / "training" /
        "image_02")
    cmd = [
        PY, str(EVAL_RUN),
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--SEQMAP_FILE", str(seqmap40),
        "--SKIP_SPLIT_FOL", "True",
        "--GT_FOLDER", str(res_root),
        "--TRACKERS_FOLDER", str(res_root),
        "--TRACKERS_TO_EVAL", str(res_root),
        "--GT_LOC_FORMAT",
        "{gt_folder}{video_id}/{expression_id}/gt.txt",
        "--USE_PARALLEL", "False",
        "--PRINT_ONLY_COMBINED", "False",
        "--PLOT_CURVES", "False",
    ]
    log = out_root / "trackeval.log"
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=str(EVAL_RUN.parent), env=env, stdout=f,
                       stderr=subprocess.STDOUT, check=True)
    print(f"[l8rmot] eval done, log={log}", flush=True)


if __name__ == "__main__":
    main()
