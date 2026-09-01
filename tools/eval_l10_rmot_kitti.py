"""Stage L10: Refer-KITTI-V2 RMOT evaluation with the shared UIDM.

For each official (sequence, expression) query, runs the same UIDM
online tracker with DLA candidates + CLIP + crop-PBD, writes TrackEval
`predict.txt`, and runs the patched official RMOT TrackEval runner.

Usage:
  python tools/eval_l10_rmot_kitti.py --ckpt ... --out ... --gpu 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.data.token_cache import cache_key, read_frame_cache  # noqa: E402
from locatemot.models.l8_unified import L8UnifiedUIDM, load_l8_state  # noqa: E402
from locatemot.tracking.online_tracker import OnlineTracker  # noqa: E402

PY = "/home/lwr/anaconda3/envs/locatemot/bin/python"
DATA_DIR = ROOT / "outputs" / "l10" / "data" / "rmot_kitti"
PBD_CACHE = ROOT / "outputs" / "l10" / "cache" / "kitti_pbd"
GT_TEMPLATE = DATA_DIR / "gt_template"
SEQMAP = (Path("/data1/LWR/vranlee/SERVER_ONLY/avis/"
               "LocateMOT_reference_repos") / "temp_rmot" /
          "datasets" / "data_path" / "seqmap.txt")
IMG_ROOT = ROOT / "data" / "kitti_tracking_training" / "image_02"
EVAL_RUN = (ROOT / "references" / "l8" / "TrackEval_rmot" / "scripts"
            / "run_mot_challenge.py")
SIZES = {
    "small": dict(d_model=192, n_layers=3, n_heads=4, ffn_dim=768),
    "base": dict(d_model=320, n_layers=4, n_heads=8, ffn_dim=1280),
    "large": dict(d_model=384, n_layers=6, n_heads=8, ffn_dim=1536),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.0)
    ap.add_argument("--threshold-file", default=None)
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--predict-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    if args.threshold_file:
        calib = json.loads(Path(args.threshold_file).read_text())
        args.threshold = float(calib["threshold"])
        print(f"[l10rk] calibrated threshold={args.threshold}", flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    core = adapter = None
    if not args.eval_only:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = ck.get("cfg", {})
        size = cfg.get("model", "base")
        mode = cfg.get("mode", "unified")
        model = L8UnifiedUIDM(
            **SIZES[size],
            no_interaction=cfg.get("no_interaction", False),
            use_cue_rel=cfg.get("use_cue_rel", False),
            mode=mode,
            sem_in_core=cfg.get("sem_in_core", True),
            cond_gated=cfg.get("cond_gated", False)).to(device)
        load_l8_state(model, ck["model"])
        model.eval()
        core = model.uidm
        adapter = model.adapter
        print(f"[l10rk] size={size} mode={mode}", flush=True)

    exp_meta = json.loads((DATA_DIR / "expressions.json").read_text())
    queries = []
    for line in SEQMAP.read_text().splitlines():
        if not line.strip():
            continue
        seq, expr = line.strip().split("+", 1)
        meta = next((e for e in exp_meta.get(seq, [])
                     if e["expression"] == expr), None)
        if meta is None:
            continue
        queries.append((seq, expr, np.asarray(meta["spec"], np.float32)))
    if args.num_shards > 1:
        queries = [q for q in queries
                   if int(hashlib.md5((q[0] + "+" + q[1]).encode()).hexdigest(), 16)
                   % args.num_shards == args.shard]
    print(f"[l10rk] queries={len(queries)}", flush=True)

    seqs = sorted(set(q[0] for q in queries))
    if args.max_seqs:
        seqs = seqs[:args.max_seqs]
    out_root = Path(args.out)
    res_root = out_root / "uidm10"
    t0 = time.time()
    for si, seq in enumerate(seqs):
        rec = pickle.load(open(DATA_DIR / f"{seq}.pkl", "rb"))
        frames = {fr["frame"]: fr for fr in rec["frames"]}
        seq_queries = [q for q in queries if q[0] == seq]
        for _, expr, spec in seq_queries:
            exp_dir = res_root / seq / expr
            exp_dir.mkdir(parents=True, exist_ok=True)
            gt_src = GT_TEMPLATE / seq / expr / "gt.txt"
            gt_dst = exp_dir / "gt.txt"
            if gt_src.exists() and not gt_dst.exists():
                gt_dst.symlink_to(gt_src)
            if args.eval_only:
                continue
            if (exp_dir / "predict.txt").exists():
                continue
            tracker = OnlineTracker(
                variant="UIDM", uidm=core, device=str(device),
                output_all_candidates=True,
                uidm_adapter=adapter, uidm_spec=spec)
            tracker.uidm_sem_in_core = model.sem_in_core
            tracker.uidm_new_margin = 0.0
            tracker.l1d_weights = (0.4, 0.2, 0.4)
            tracker.l1d_threshold = 0.25
            rows = []
            for frame in sorted(frames):
                fr = frames[frame]
                n = len(fr["boxes"])
                if n == 0:
                    continue
                pbd = np.zeros((n, 2048), np.float32)
                d = read_frame_cache(str(PBD_CACHE),
                                     cache_key("kitti", seq, frame,
                                               "pbd_full"))
                if d is not None:
                    p = np.asarray(d["features"]["pbd_box_end_last"],
                                   np.float32)
                    if len(p) == n:
                        pbd = p
                clip = np.asarray(fr["clip"], np.float32)
                cands = []
                for j in range(n):
                    f = {
                        "pbd": np.zeros(2048, np.float32),
                        "pbd_be": pbd[j],
                        "region": np.zeros(4608, np.float32),
                        "geom": np.zeros(5, np.float32),
                        "gen": float(fr["gen"][j]),
                    }
                    cands.append({"box": fr["boxes"][j],
                                  "features": f, "index": j})
                tracker.image_size = rec["image_size"]
                with torch.no_grad():
                    pbd_eff = pbd
                    clip_eff = clip
                    spec_eff = np.broadcast_to(spec, (n, 512))
                    rel = adapter(
                        torch.as_tensor(pbd_eff, device=device),
                        torch.as_tensor(clip_eff, device=device),
                        torch.as_tensor(spec_eff, device=device))[1]
                    rel = rel.cpu().numpy()
                outputs = tracker.process_frame(frame, cands)
                assert len(outputs) == len(cands)
                for o, r in zip(outputs, rel):
                    if float(r) <= args.threshold:
                        continue
                    x1, y1, x2, y2 = o["box"]
                    rows.append([frame + 1, o["track_id"], x1, y1,
                                 x2 - x1, y2 - y1,
                                 float(o.get("score", 1.0)), -1, -1, -1])
            with open(exp_dir / "predict.txt", "w") as f:
                for r in rows:
                    f.write(",".join(
                        f"{v:.3f}" if isinstance(v, float) else str(v)
                        for v in r) + "\n")
        print(f"[l10rk] {seq} {si+1}/{len(seqs)} "
              f"elapsed={time.time()-t0:.0f}s", flush=True)
    if args.predict_only:
        print("[l10rk] predictions written (predict-only)", flush=True)
        return

    eval_seqs = set(seqs)
    seqmap_out = out_root / "seqmap_kitti.txt"
    with open(seqmap_out, "w") as f:
        for seq, expr, _ in queries:
            if seq in eval_seqs:
                f.write(f"{seq}+{expr}\n")
    res_root = res_root.resolve()
    env = dict(os.environ)
    env["RMOT_IMG_ROOT"] = str(IMG_ROOT.resolve())
    cmd = [
        PY, str(EVAL_RUN),
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--SEQMAP_FILE", str(seqmap_out.resolve()),
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
    log = out_root / "trackeval_kitti.log"
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=str(EVAL_RUN.parent), env=env, stdout=f,
                       stderr=subprocess.STDOUT, check=True)
    print(f"[l10rk] eval done, log={log}", flush=True)


if __name__ == "__main__":
    main()
