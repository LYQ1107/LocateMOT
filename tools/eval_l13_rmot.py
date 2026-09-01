"""Stage L13 RMOT front-end ablations with the unchanged shared UIDM.

The front-end chooses a bounded candidate subset, but every selected box is
then processed by the existing OnlineTracker/UIDM implementation.  The
trajectory mode is a causal IoU-linked proposal memory: it smooths the
language score over a candidate trajectory without replacing UIDM identity
states, set interaction, NEW/NO-MATCH transitions, or lifecycle handling.

Datasets:
  --dataset kitti_v1       official RMOT v1 seqmap (150 queries)
  --dataset kitti_v2       official TempRMOT seqmap (862 queries)
  --dataset dance          Refer-Dance non-empty official queries (40)

Front-ends:
  clip       per-frame CLIP crop/text ranking (L11-style baseline)
  open_vocab precomputed language/open-vocabulary detector box score
  trajectory causal trajectory-language score + UIDM relevance for output
  ours       causal trajectory score fused with UIDM relevance ranking

Sharded prediction runs can write to one output directory concurrently:
  python tools/eval_l13_rmot.py ... --shard 0 --num-shards 4 --predict-only
  ... repeat shards ...
  python tools/eval_l13_rmot.py ... --eval-only
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
from tools.eval_l3 import build_candidates  # noqa: E402

PY = "/home/lwr/anaconda3/envs/locatemot/bin/python"
SIZES = {
    "small": dict(d_model=192, n_layers=3, n_heads=4, ffn_dim=768),
    "base": dict(d_model=320, n_layers=4, n_heads=8, ffn_dim=1280),
    "large": dict(d_model=384, n_layers=6, n_heads=8, ffn_dim=1536),
}
KITTI_DATA = ROOT / "outputs" / "l11" / "data" / "rmot_kitti"
KITTI_PBD = ROOT / "outputs" / "l10" / "cache" / "kitti_pbd"
V1_DATA = ROOT / "outputs" / "l13" / "data" / "refer_kitti_v1"
V2_GT = ROOT / "outputs" / "l10" / "data" / "rmot_kitti" / "gt_template"
V1_GT = V1_DATA / "gt_template"
V2_SEQMAP = (Path("/data1/LWR/vranlee/SERVER_ONLY/avis/") /
             "LocateMOT_reference_repos" / "temp_rmot" /
             "datasets" / "data_path" / "seqmap.txt")
V1_SEQMAP = (Path("/data1/LWR/vranlee/SERVER_ONLY/avis/") /
             "LocateMOT_reference_repos" / "rmot_official" /
             "datasets" / "data_path" / "seqmap.txt")
DANCE_MANIFEST = ROOT / "outputs" / "l1_c" / "fixed_candidate_manifest" / \
    "dancetrack_val.jsonl"
DANCE_CLIP = ROOT / "outputs" / "l7" / "data" / "clip_eval" / \
    "dancetrack_val"
DANCE_META = ROOT / "outputs" / "l8" / "data" / "rmot_eval" / \
    "expressions.json"
DANCE_GT = ROOT / "data" / "refer_dance" / "gt_template"
DANCE_SEQMAP = ROOT / "data" / "refer_dance" / "seqmap.txt"
EVAL_RUN = ROOT / "references" / "l8" / "TrackEval_rmot" / "scripts" / \
    "run_mot_challenge.py"


def norm_rows(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def rank01(x):
    """Stable within-frame rank, where larger values are better."""
    x = np.asarray(x, np.float32)
    if len(x) <= 1:
        return np.ones(len(x), np.float32)
    order = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    return order.astype(np.float32) / float(len(x) - 1)


def box_iou(a, b):
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2]) - float(a[0])) * \
        max(0.0, float(a[3]) - float(a[1]))
    bb = max(0.0, float(b[2]) - float(b[0])) * \
        max(0.0, float(b[3]) - float(b[1]))
    return inter / max(1e-9, aa + bb - inter)


def causal_tracklets(frames, iou_threshold=0.30, max_gap=1):
    """Assign causal proposal IDs using only previous-frame boxes."""
    active = []
    next_id = 0
    all_ids = []
    for fr in frames:
        boxes = np.asarray(fr["boxes"], np.float32)
        n = len(boxes)
        assigned = [None] * n
        used = set()
        pairs = []
        for j, box in enumerate(boxes):
            for k, (_tid, last_box, last_frame) in enumerate(active):
                if int(fr["frame"]) - int(last_frame) <= max_gap:
                    value = box_iou(box, last_box)
                    if value >= iou_threshold:
                        pairs.append((value, j, k))
        for value, j, k in sorted(pairs, reverse=True):
            if assigned[j] is None and k not in used:
                assigned[j] = active[k][0]
                used.add(k)
        for j in range(n):
            if assigned[j] is None:
                assigned[j] = next_id
                next_id += 1
        active = [(assigned[j], boxes[j], int(fr["frame"]))
                  for j in range(n)]
        all_ids.append(np.asarray(assigned, np.int64))
    return all_ids


def load_queries(dataset):
    if dataset == "kitti_v1":
        meta_path, seqmap, gt_root = V1_DATA / "expressions.json", V1_SEQMAP, V1_GT
        keep_nonempty = False
    elif dataset == "kitti_v2":
        meta_path, seqmap, gt_root = KITTI_DATA / "expressions.json", V2_SEQMAP, V2_GT
        keep_nonempty = False
    else:
        meta_path, seqmap, gt_root = DANCE_META, DANCE_SEQMAP, DANCE_GT
        keep_nonempty = True
    meta = json.loads(meta_path.read_text())
    queries = []
    for line in seqmap.read_text().splitlines():
        if not line.strip():
            continue
        seq, expr = line.strip().split("+", 1)
        entries = meta.get(seq, [])
        entry = next((e for e in entries if e["expression"] == expr), None)
        if entry is None:
            continue
        gt = gt_root / seq / expr / "gt.txt"
        if keep_nonempty and (not gt.exists() or gt.stat().st_size == 0):
            continue
        queries.append((seq, expr, np.asarray(entry["spec"], np.float32)))
    return queries, gt_root, seqmap


def load_dance_entries():
    by_video = {}
    with DANCE_MANIFEST.open() as f:
        for line in f:
            entry = json.loads(line)
            by_video.setdefault(entry["video_id"], []).append(entry)
    for video in by_video:
        by_video[video].sort(key=lambda x: int(x["frame"]))
    return by_video


def make_kitti_frames(seq):
    rec = pickle.load(open(KITTI_DATA / f"{seq}.pkl", "rb"))
    frames = []
    for raw in rec["frames"]:
        boxes = np.asarray(raw["boxes"], np.float32)
        n = len(boxes)
        pbd = np.zeros((n, 2048), np.float32)
        cached = read_frame_cache(
            str(KITTI_PBD), cache_key("kitti", seq, int(raw["frame"]), "pbd_full"))
        if cached is not None:
            values = np.asarray(cached["features"]["pbd_box_end_last"], np.float32)
            orig_idx = np.asarray(raw.get("orig_idx", np.arange(n)), np.int64)
            if len(values) and len(orig_idx) and int(orig_idx.max()) < len(values):
                pbd = values[orig_idx]
        frames.append({
            "frame": int(raw["frame"]),
            "boxes": boxes,
            "gen": np.asarray(raw.get("gen", np.zeros(n)), np.float32),
            "clip": np.asarray(raw.get("clip", np.zeros((n, 512))), np.float32),
            "pbd": pbd,
            "cand_gt": list(raw.get("cand_gt", [None] * n)),
        })
    return frames, rec["image_size"]


def make_dance_frames(video, by_video):
    entries = by_video[video]
    clip_rec = pickle.load(open(DANCE_CLIP / f"{video}.pkl", "rb"))
    clip_frames = {int(fr["frame"]): fr for fr in clip_rec["frames"]}
    frames = []
    image_size = entries[0].get("image_size", [1920, 1080])
    for entry in entries:
        frame = int(entry["frame"])
        cands, image_size = build_candidates(entry)
        clip = np.asarray(clip_frames[frame]["clip"], np.float32)
        if len(cands) != len(clip):
            raise RuntimeError(f"{video} frame {frame}: candidate/CLIP mismatch")
        pbd = np.stack([
            np.asarray(c["features"].get("pbd_be", np.zeros(2048)), np.float32)
            for c in cands], axis=0) if cands else np.zeros((0, 2048), np.float32)
        gen = np.asarray([
            float(c["features"].get("gen", 0.0)) for c in cands
        ], np.float32)
        frames.append({
            "frame": frame,
            "boxes": np.asarray([c["box"] for c in cands], np.float32)
            if cands else np.zeros((0, 4), np.float32),
            "gen": gen,
            "clip": clip,
            "pbd": pbd,
            "cands": cands,
        })
    return frames, image_size


def load_dino_cache(path):
    if not path:
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def dino_scores(cache, seq, frame, boxes):
    if cache is None:
        return np.zeros(len(boxes), np.float32)
    entry = cache.get(seq, {}).get(int(frame))
    if entry is None:
        return np.zeros(len(boxes), np.float32)
    det_boxes = np.asarray(entry.get("boxes", []), np.float32)
    det_scores = np.asarray(entry.get("scores", []), np.float32)
    if not len(det_boxes):
        return np.zeros(len(boxes), np.float32)
    out = np.zeros(len(boxes), np.float32)
    for i, box in enumerate(boxes):
        out[i] = max((float(s) * box_iou(box, db)
                      for db, s in zip(det_boxes, det_scores)), default=0.0)
    return out


def adapter_relevance(adapter, pbd, clip, spec, device):
    n = len(clip)
    if not n:
        return np.zeros(0, np.float32)
    with torch.no_grad():
        _, rel = adapter(
            torch.as_tensor(pbd, device=device),
            torch.as_tensor(clip, device=device),
            torch.as_tensor(np.broadcast_to(spec, (n, 512)).copy(),
                            device=device))
    return rel.detach().cpu().numpy().astype(np.float32)


def select_indices(frontend, cos, traj, rel, dino, topk):
    n = len(cos)
    if n == 0:
        return np.zeros(0, np.int64)
    if topk <= 0 or n <= topk:
        return np.arange(n, dtype=np.int64)
    if frontend == "clip":
        score = rank01(cos)
    elif frontend == "open_vocab":
        score = rank01(dino)
    elif frontend == "trajectory":
        score = rank01(traj)
    elif frontend == "spec_adapter":
        score = rank01(rel if rel is not None else cos)
    elif frontend in {"trajectory_spec", "trajectory+spec"}:
        score = rank01(traj)
    elif frontend in {"ours", "full"}:
        score = (0.45 * rank01(traj) + 0.35 * rank01(rel) +
                 0.20 * rank01(cos))
    else:
        raise ValueError(frontend)
    return np.argsort(-score, kind="stable")[:topk]


def build_tracker(model, spec, device):
    tracker = OnlineTracker(
        variant="UIDM", uidm=model.uidm, device=str(device),
        output_all_candidates=True, uidm_adapter=model.adapter,
        uidm_spec=spec)
    tracker.uidm_sem_in_core = model.sem_in_core
    tracker.uidm_new_margin = 0.0
    tracker.l1d_weights = (0.4, 0.2, 0.4)
    tracker.l1d_threshold = 0.25
    return tracker


def run_query(model, frames, image_size, seq, spec, args, device, dino,
              tids):
    tracker = build_tracker(model, spec, device)
    tracker.image_size = image_size
    ema = {}
    rows = []
    for frame_index, fr in enumerate(frames):
        track_ids = tids[frame_index] if tids is not None else None
        n = len(fr["boxes"])
        if n == 0:
            continue
        clip_norm = norm_rows(fr["clip"])
        spec_norm = np.asarray(spec, np.float32)
        spec_norm = spec_norm / max(1e-9, float(np.linalg.norm(spec_norm)))
        cos = clip_norm @ spec_norm
        traj = np.zeros(n, np.float32)
        if tids is None:
            traj = cos.copy()
        else:
            for j, tid in enumerate(track_ids):
                old = ema.get(int(tid), float(cos[j]))
                traj[j] = args.trajectory_blend * float(cos[j]) + \
                    (1.0 - args.trajectory_blend) * old
                ema[int(tid)] = args.trajectory_decay * old + \
                    (1.0 - args.trajectory_decay) * float(cos[j])

        # Ours uses the learned relevance head as a WHAT score before UIDM
        # receives the selected candidates.  The other modes only calculate
        # it for the selected candidates, matching the L11 execution path.
        rel_all = None
        if args.frontend in {"spec_adapter", "ours", "full"}:
            rel_all = adapter_relevance(
                model.adapter, fr["pbd"], fr["clip"], spec_norm, device)
        dino_frame = dino_scores(dino, seq, fr["frame"], fr["boxes"])
        keep = select_indices(args.frontend, cos, traj, rel_all, dino_frame,
                              args.topk)
        if len(keep) == 0:
            continue
        cands = []
        for j in keep:
            features = {
                "pbd": np.asarray(fr["pbd"][j], np.float32),
                "pbd_be": np.asarray(fr["pbd"][j], np.float32),
                "region": np.zeros(4608, np.float32),
                "geom": np.zeros(5, np.float32),
                "gen": float(fr["gen"][j]),
                "clip": np.asarray(fr["clip"][j], np.float32),
            }
            cands.append({"box": fr["boxes"][j], "features": features,
                          "index": int(j)})
        tracker.image_size = image_size
        outputs = tracker.process_frame(int(fr["frame"]), cands)
        if len(outputs) != len(cands):
            raise RuntimeError("OnlineTracker output/candidate mismatch")
        if rel_all is None:
            rel = adapter_relevance(
                model.adapter, fr["pbd"][keep], fr["clip"][keep],
                spec_norm, device)
        else:
            rel = rel_all[keep]
        for out, score in zip(outputs, rel):
            if float(score) <= args.rel_threshold:
                continue
            x1, y1, x2, y2 = out["box"]
            frame_no = int(fr["frame"]) + (0 if args.dataset == "dance" else 1)
            rows.append([frame_no, out["track_id"], x1, y1, x2 - x1,
                         y2 - y1, float(out.get("score", 1.0)), -1, -1, -1])
    return rows


def write_prediction(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(",".join(f"{x:.3f}" if isinstance(x, float) else str(x)
                             for x in row) + "\n")


def run_eval(dataset, out_root, gt_root, seqmap, seqs):
    res_root = (out_root / "uidm13").resolve()
    seqmap_out = out_root / "seqmap_l13.txt"
    with seqmap_out.open("w") as f:
        for line in seqmap.read_text().splitlines():
            if not line.strip():
                continue
            seq, expr = line.split("+", 1)
            if seq in seqs:
                # Refer-Dance's public seqmap lists all 425 expressions, but
                # the official RMOT evaluation is defined on the 40 queries
                # with non-empty GT.  Kitti evaluates every seqmap entry.
                if dataset == "dance":
                    gt = gt_root / seq / expr / "gt.txt"
                    if not gt.exists() or gt.stat().st_size == 0:
                        continue
                f.write(line.strip() + "\n")
    env = dict(os.environ)
    env["RMOT_IMG_ROOT"] = str(
        (ROOT / "data" / "refer_dance" / "DanceTrack" / "training" /
         "image_02") if dataset == "dance" else
        (ROOT / "data" / "kitti_tracking_training" / "image_02"))
    cmd = [PY, str(EVAL_RUN), "--METRICS", "HOTA", "CLEAR", "Identity",
           "--SEQMAP_FILE", str(seqmap_out.resolve()),
           "--SKIP_SPLIT_FOL", "True", "--GT_FOLDER", str(res_root),
           "--TRACKERS_FOLDER", str(res_root),
           "--TRACKERS_TO_EVAL", str(res_root),
           "--GT_LOC_FORMAT", "{gt_folder}{video_id}/{expression_id}/gt.txt",
           "--USE_PARALLEL", "False", "--PRINT_ONLY_COMBINED", "False",
           "--PLOT_CURVES", "False"]
    log = out_root / "trackeval_l13.log"
    with log.open("w") as f:
        subprocess.run(cmd, cwd=str(EVAL_RUN.parent), env=env, stdout=f,
                       stderr=subprocess.STDOUT, check=True)
    print(f"[l13] eval done: {log}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["kitti_v1", "kitti_v2", "dance"],
                    required=True)
    ap.add_argument("--frontend", choices=[
        "clip", "open_vocab", "trajectory", "spec_adapter",
        "trajectory_spec", "trajectory+spec", "ours", "full"],
                    default="clip")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--topk", type=int, default=0,
                    help="per-frame candidate cap; 0 keeps all")
    ap.add_argument("--rel-threshold", type=float, default=0.0)
    ap.add_argument("--trajectory-iou", type=float, default=0.30)
    ap.add_argument("--trajectory-max-gap", type=int, default=1)
    ap.add_argument("--trajectory-decay", type=float, default=0.80)
    ap.add_argument("--trajectory-blend", type=float, default=0.35)
    ap.add_argument("--dino-cache", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-seqs", type=int, default=0)
    ap.add_argument("--predict-only", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    queries, gt_root, seqmap = load_queries(args.dataset)
    seqs = sorted({q[0] for q in queries})
    if args.max_seqs:
        seqs = seqs[:args.max_seqs]
        queries = [q for q in queries if q[0] in set(seqs)]
    if args.num_shards > 1:
        queries = [q for q in queries if int(hashlib.md5(
            (q[0] + "+" + q[1]).encode()).hexdigest(), 16) % args.num_shards
                   == args.shard]
    print(f"[l13] dataset={args.dataset} frontend={args.frontend} "
          f"queries={len(queries)} topk={args.topk} device={device}", flush=True)

    # Eval-only is deliberately cheap: sharded prediction processes create
    # the GT links; this pass only writes the full seqmap and invokes TrackEval.
    if args.eval_only:
        for seq, expr, _ in load_queries(args.dataset)[0]:
            if seq not in set(seqs):
                continue
            d = out_root / "uidm13" / seq / expr
            d.mkdir(parents=True, exist_ok=True)
            src = gt_root / seq / expr / "gt.txt"
            dst = d / "gt.txt"
            if src.exists() and not dst.exists():
                dst.symlink_to(src)
        run_eval(args.dataset, out_root, gt_root, seqmap, set(seqs))
        return

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    model = L8UnifiedUIDM(
        **SIZES[cfg.get("model", "base")],
        no_interaction=cfg.get("no_interaction", False),
        use_cue_rel=cfg.get("use_cue_rel", False),
        mode=cfg.get("mode", "unified"),
        sem_in_core=cfg.get("sem_in_core", True),
        cond_gated=cfg.get("cond_gated", False),
        spec_conditioned=cfg.get("spec_conditioned", False),
        trajectory_memory=cfg.get("trajectory_memory", True)).to(device)
    load_l8_state(model, ck["model"])
    model.eval()
    dino = load_dino_cache(args.dino_cache)
    dance_entries = load_dance_entries() if args.dataset == "dance" else None
    q_by_seq = {}
    for q in queries:
        q_by_seq.setdefault(q[0], []).append(q)
    t0 = time.time()
    for si, seq in enumerate(sorted(q_by_seq)):
        if args.dataset == "dance":
            frames, image_size = make_dance_frames(seq, dance_entries)
        else:
            frames, image_size = make_kitti_frames(seq)
        tids = causal_tracklets(frames, args.trajectory_iou,
                                args.trajectory_max_gap) \
            if args.frontend in {"trajectory", "trajectory_spec",
                                 "trajectory+spec", "ours", "full"} else None
        for _, expr, spec in q_by_seq[seq]:
            out_dir = out_root / "uidm13" / seq / expr
            out_dir.mkdir(parents=True, exist_ok=True)
            gt_src = gt_root / seq / expr / "gt.txt"
            gt_dst = out_dir / "gt.txt"
            if gt_src.exists() and not gt_dst.exists():
                gt_dst.symlink_to(gt_src)
            pred = out_dir / "predict.txt"
            if pred.exists():
                continue
            rows = run_query(model, frames, image_size, seq, spec, args,
                             device, dino, tids)
            write_prediction(pred, rows)
        print(f"[l13] {seq} {si + 1}/{len(q_by_seq)} "
              f"elapsed={time.time() - t0:.0f}s", flush=True)
    if args.predict_only:
        return
    run_eval(args.dataset, out_root, gt_root, seqmap, set(seqs))


if __name__ == "__main__":
    main()
