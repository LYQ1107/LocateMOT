"""Evaluate the L15 query-conditioned observation frontend.

The script deliberately reuses the L13 frame builders and L11 shared UIDM.
Only proposal ranking is changed: a frozen crop/text fusion head is followed
by a causal proposal-tracklet EMA, then the selected boxes go through the
same OnlineTracker and UIDM lifecycle.  No evaluation labels are read during
prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l15_observation import L15ObservationHead  # noqa: E402
from locatemot.models.l8_unified import L8UnifiedUIDM, load_l8_state  # noqa: E402
from tools.eval_l13_rmot import (  # noqa: E402
    DANCE_META,
    DANCE_SEQMAP,
    EVAL_RUN,
    KITTI_DATA,
    SIZES,
    V1_DATA,
    V1_SEQMAP,
    V2_SEQMAP,
    V2_GT,
    V1_GT,
    build_tracker,
    causal_tracklets,
    load_dance_entries,
    load_queries,
    make_dance_frames,
    make_kitti_frames,
    rank01,
    adapter_relevance,
)

PY = "/home/lwr/anaconda3/envs/locatemot/bin/python"


def geometry(boxes, image_size):
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    w, h = [max(1.0, float(x)) for x in image_size]
    if len(boxes):
        x1, y1, x2, y2 = boxes.T
    else:
        x1 = y1 = x2 = y2 = np.zeros(0, np.float32)
    bw = np.maximum(0.0, x2 - x1)
    bh = np.maximum(0.0, y2 - y1)
    nw, nh = bw / w, bh / h
    return np.stack(((x1 + x2) * 0.5 / w, (y1 + y2) * 0.5 / h,
                     nw, nh, nw * nh,
                     np.clip(bw / np.maximum(bh, 1.0), 0.0, 20.0) / 20.0,
                     y2 / h), axis=1).astype(np.float32)


def load_shared(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
    missing, unexpected = load_l8_state(model, ck["model"])
    if missing or unexpected:
        raise RuntimeError(f"shared checkpoint mismatch: {missing} {unexpected}")
    model.eval()
    return model


def load_head(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck.get("cfg", {})
    head = L15ObservationHead(hidden=int(cfg.get("hidden", 384))).to(device)
    missing, unexpected = head.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"observation checkpoint mismatch: {missing} {unexpected}")
    head.eval()
    return head


def head_scores_many(head, frames, spec, image_size, device, batch_size):
    """Compute all proposal scores for one query in a few GPU batches."""
    clips, geoms, gens, spans = [], [], [], []
    offset = 0
    for frame in frames:
        n = len(frame["boxes"])
        spans.append((offset, offset + n))
        offset += n
        if not n:
            continue
        clip = np.asarray(frame.get("clip", np.zeros((n, 512))), np.float32)
        if clip.shape != (n, 512):
            clip = np.zeros((n, 512), np.float32)
        gen = np.asarray(frame.get("gen", np.zeros(n)), np.float32).reshape(-1)
        if len(gen) != n:
            gen = np.zeros(n, np.float32)
        clips.append(np.nan_to_num(clip))
        geoms.append(geometry(frame["boxes"], image_size))
        gens.append(np.nan_to_num(gen))
    if not clips:
        return [np.zeros(0, np.float32) for _ in frames]
    all_clip = np.concatenate(clips, axis=0)
    all_geom = np.concatenate(geoms, axis=0)
    all_gen = np.concatenate(gens, axis=0)
    all_scores = []
    spec_t = torch.from_numpy(np.asarray(spec, np.float32)).to(device)
    with torch.no_grad():
        for start in range(0, len(all_clip), int(batch_size)):
            end = min(len(all_clip), start + int(batch_size))
            values = head(
                torch.from_numpy(all_clip[start:end]).to(device), spec_t,
                torch.from_numpy(all_geom[start:end]).to(device),
                torch.from_numpy(all_gen[start:end]).to(device))
            all_scores.append(values.float().cpu().numpy().astype(np.float32))
    flat = np.concatenate(all_scores, axis=0)
    return [flat[start:end] for start, end in spans]


def run_query(shared, head, frames, image_size, spec, args, device, tids):
    tracker = build_tracker(shared, spec, device)
    tracker.image_size = image_size
    ema = {}
    rows = []
    scores_by_frame = head_scores_many(
        head, frames, spec, image_size, device, args.head_batch)
    for frame_index, frame in enumerate(frames):
        n = len(frame["boxes"])
        if not n:
            continue
        current = scores_by_frame[frame_index]
        causal = current.copy()
        if tids is not None:
            for j, tid in enumerate(tids[frame_index]):
                old = ema.get(int(tid), float(current[j]))
                causal[j] = args.current_blend * float(current[j]) + \
                    (1.0 - args.current_blend) * old
                ema[int(tid)] = args.temporal_decay * old + \
                    (1.0 - args.temporal_decay) * float(current[j])
        score = 0.70 * rank01(causal) + 0.30 * rank01(current)
        if args.topk > 0 and n > args.topk:
            keep = np.argsort(-score, kind="stable")[:args.topk]
        else:
            keep = np.arange(n, dtype=np.int64)
        cands = []
        for j in keep:
            pbd = np.asarray(frame.get("pbd", np.zeros((n, 2048)))[j],
                             np.float32)
            clip = np.asarray(frame.get("clip", np.zeros((n, 512)))[j],
                              np.float32)
            cands.append({
                "box": np.asarray(frame["boxes"][j], np.float32),
                "features": {
                    "pbd": pbd,
                    "pbd_be": pbd,
                    "region": np.zeros(4608, np.float32),
                    "geom": np.zeros(5, np.float32),
                    "gen": float(np.asarray(frame.get("gen", np.zeros(n)))[j]),
                    "clip": clip,
                },
                "index": int(j),
            })
        outputs = tracker.process_frame(int(frame["frame"]), cands)
        if len(outputs) != len(cands):
            raise RuntimeError("OnlineTracker output/candidate mismatch")
        selected_pbd = np.asarray([x["features"]["pbd"] for x in cands],
                                  np.float32)
        selected_clip = np.asarray([x["features"]["clip"] for x in cands],
                                    np.float32)
        spec_norm = np.asarray(spec, np.float32)
        spec_norm = spec_norm / max(1e-9, float(np.linalg.norm(spec_norm)))
        relevance = adapter_relevance(
            shared.adapter, selected_pbd, selected_clip, spec_norm, device)
        for out, rel in zip(outputs, relevance):
            if float(rel) <= args.rel_threshold:
                continue
            x1, y1, x2, y2 = [float(x) for x in out["box"]]
            frame_no = int(frame["frame"]) + \
                (0 if args.dataset == "dance" else 1)
            rows.append([frame_no, int(out["track_id"]), x1, y1,
                         x2 - x1, y2 - y1, float(out.get("score", 1.0)),
                         -1, -1, -1])
    return rows


def run_eval(dataset, out_root, gt_root, seqmap, seqs):
    res_root = (out_root / "uidm15").resolve()
    seqmap_out = out_root / "seqmap_l15.txt"
    with seqmap_out.open("w") as f:
        for line in seqmap.read_text().splitlines():
            if not line.strip():
                continue
            seq, expr = line.strip().split("+", 1)
            if seq not in seqs:
                continue
            if dataset == "dance":
                gt = gt_root / seq / expr / "gt.txt"
                if not gt.exists() or gt.stat().st_size == 0:
                    continue
            f.write(line.strip() + "\n")
    env = dict(os.environ)
    env["RMOT_IMG_ROOT"] = str(
        ROOT / ("data/refer_dance/DanceTrack/training/image_02"
                if dataset == "dance" else
                "data/kitti_tracking_training/image_02"))
    cmd = [PY, str(EVAL_RUN), "--METRICS", "HOTA", "CLEAR", "Identity",
           "--SEQMAP_FILE", str(seqmap_out.resolve()),
           "--SKIP_SPLIT_FOL", "True", "--GT_FOLDER", str(res_root),
           "--TRACKERS_FOLDER", str(res_root), "--TRACKERS_TO_EVAL",
           str(res_root), "--GT_LOC_FORMAT",
           "{gt_folder}{video_id}/{expression_id}/gt.txt",
           "--USE_PARALLEL", "False", "--PRINT_ONLY_COMBINED", "False",
           "--PLOT_CURVES", "False"]
    log = out_root / "trackeval_l15.log"
    with log.open("w") as f:
        subprocess.run(cmd, cwd=str(EVAL_RUN.parent), env=env, stdout=f,
                       stderr=subprocess.STDOUT, check=True)
    print(f"[l15] eval done: {log}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["kitti_v1", "kitti_v2", "dance"],
                    required=True)
    ap.add_argument("--shared-ckpt", required=True)
    ap.add_argument("--observation-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--current-blend", type=float, default=0.35)
    ap.add_argument("--temporal-decay", type=float, default=0.80)
    ap.add_argument("--head-batch", type=int, default=4096)
    ap.add_argument("--rel-threshold", type=float, default=0.0)
    ap.add_argument("--trajectory-iou", type=float, default=0.30)
    ap.add_argument("--trajectory-max-gap", type=int, default=1)
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
    all_seqs = sorted({q[0] for q in queries})
    seqs = all_seqs[:args.max_seqs] if args.max_seqs else all_seqs
    qset = set(seqs)
    queries = [q for q in queries if q[0] in qset]
    if args.num_shards > 1:
        queries = [q for q in queries if int(hashlib.md5(
            (q[0] + "+" + q[1]).encode()).hexdigest(), 16) % args.num_shards
                   == args.shard]
    print(f"[l15] dataset={args.dataset} queries={len(queries)} "
          f"topk={args.topk} shard={args.shard}/{args.num_shards} "
          f"device={device}", flush=True)

    if args.eval_only:
        for seq, expr, _ in load_queries(args.dataset)[0]:
            if seq not in qset:
                continue
            dst = out_root / "uidm15" / seq / expr / "gt.txt"
            src = gt_root / seq / expr / "gt.txt"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists() and not dst.exists():
                dst.symlink_to(src)
        run_eval(args.dataset, out_root, gt_root, seqmap, qset)
        return

    shared = load_shared(args.shared_ckpt, device)
    head = load_head(args.observation_ckpt, device)
    dance_entries = load_dance_entries() if args.dataset == "dance" else None
    grouped = {}
    for seq, expr, spec in queries:
        grouped.setdefault(seq, []).append((expr, spec))
    t0 = time.time()
    for si, seq in enumerate(sorted(grouped)):
        if args.dataset == "dance":
            frames, size = make_dance_frames(seq, dance_entries)
        else:
            frames, size = make_kitti_frames(seq)
        tids = causal_tracklets(frames, args.trajectory_iou,
                                args.trajectory_max_gap)
        for expr, spec in grouped[seq]:
            out_dir = out_root / "uidm15" / seq / expr
            out_dir.mkdir(parents=True, exist_ok=True)
            gt_src = gt_root / seq / expr / "gt.txt"
            gt_dst = out_dir / "gt.txt"
            if gt_src.exists() and not gt_dst.exists():
                gt_dst.symlink_to(gt_src)
            pred = out_dir / "predict.txt"
            if pred.exists():
                continue
            rows = run_query(shared, head, frames, size, spec, args, device,
                             tids)
            with pred.open("w") as f:
                for row in rows:
                    f.write(",".join(
                        f"{x:.3f}" if isinstance(x, float) else str(x)
                        for x in row) + "\n")
        print(f"[l15] {seq} {si + 1}/{len(grouped)} "
              f"elapsed={time.time() - t0:.0f}s", flush=True)
    if not args.predict_only:
        run_eval(args.dataset, out_root, gt_root, seqmap, qset)


if __name__ == "__main__":
    main()
