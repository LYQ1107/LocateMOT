"""Formal online track-first RMOT evaluation for Stage L16."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.models.l16_track_selector import (  # noqa: E402
    L16TrackSelector, expression_family_vector,
)
from locatemot.rmot.track_bank import load_track_bank  # noqa: E402
from tools.eval_l13_rmot import (  # noqa: E402
    DANCE_GT, DANCE_SEQMAP, EVAL_RUN, PY, V1_GT, V1_SEQMAP, V2_GT,
    V2_SEQMAP, load_queries,
)

FEATURE_NAMES = (
    "clip", "history_clip", "pbd", "uidm_h", "geometry", "motion",
    "context", "lifecycle", "objectness",
)


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["cfg"]
    model = L16TrackSelector(cfg["hidden"], cfg["heads"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


def evaluate_query(model, bank, expression, spec, device, threshold,
                   no_belief=False, no_cross_track=False, no_motion=False,
                   stateless=False, max_per_frame=0):
    tensors = bank["tensors"]
    query = torch.as_tensor(np.asarray(spec, np.float32), device=device)
    family = expression_family_vector(expression).to(device)
    state = {}
    rows = []
    probabilities = []
    started = time.time()
    with torch.no_grad():
        for frame_index, frame_id in enumerate(tensors["frame_ids"].tolist()):
            start = int(tensors["frame_ptr"][frame_index])
            end = int(tensors["frame_ptr"][frame_index + 1])
            if end <= start:
                continue
            features = {name: tensors[name][start:end].to(
                device, non_blocking=True) for name in FEATURE_NAMES}
            track_ids = tensors["track_id"][start:end].to(device)
            if stateless:
                state = {}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                output = model(
                    features, query, family, track_ids, state,
                    use_belief=not no_belief,
                    use_cross_track=not no_cross_track,
                    use_motion=not no_motion)
            state = output["state"]
            probability = torch.sigmoid(output["logits"].float())
            keep = torch.nonzero(probability >= threshold).flatten()
            if max_per_frame > 0 and len(keep) > max_per_frame:
                order = torch.argsort(probability[keep], descending=True)
                keep = keep[order[:max_per_frame]]
            probabilities.append(probability.cpu())
            for local_index in keep.cpu().tolist():
                index = start + local_index
                x1, y1, x2, y2 = [float(value) for value in tensors["box"][index]]
                frame_number = int(frame_id) if bank["metadata"]["dataset"] == \
                    "dance_eval" else int(frame_id) + 1
                rows.append([
                    frame_number, int(tensors["track_id"][index]),
                    x1, y1, x2 - x1, y2 - y1,
                    float(probability[local_index]), -1, -1, -1,
                ])
    flat = torch.cat(probabilities) if probabilities else torch.zeros(0)
    return rows, {
        "seconds": time.time() - started,
        "observations": int(len(flat)), "selected": len(rows),
        "mean_probability": float(flat.mean()) if len(flat) else None,
        "max_probability": float(flat.max()) if len(flat) else None,
    }


def write_prediction(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(",".join(
                f"{value:.6f}" if isinstance(value, float) else str(value)
                for value in row) + "\n")


def protocol(dataset):
    if dataset == "kitti_v1":
        return V1_GT, V1_SEQMAP
    if dataset == "kitti_v2":
        return V2_GT, V2_SEQMAP
    return DANCE_GT, DANCE_SEQMAP


def run_eval(dataset, out_root, seqs):
    gt_root, seqmap = protocol(dataset)
    result_root = (out_root / "uidm16").resolve()
    seqmap_out = out_root / "seqmap_l16.txt"
    with seqmap_out.open("w") as handle:
        for line in seqmap.read_text().splitlines():
            if not line.strip():
                continue
            sequence, expression = line.strip().split("+", 1)
            if sequence not in seqs:
                continue
            if dataset == "dance":
                gt = gt_root / sequence / expression / "gt.txt"
                if not gt.exists() or gt.stat().st_size == 0:
                    continue
            handle.write(line.strip() + "\n")
    env = dict(os.environ)
    env["RMOT_IMG_ROOT"] = str(
        ROOT / ("data/refer_dance/DanceTrack/training/image_02"
                if dataset == "dance" else
                "data/kitti_tracking_training/image_02"))
    command = [
        PY, str(EVAL_RUN), "--METRICS", "HOTA", "CLEAR", "Identity",
        "--SEQMAP_FILE", str(seqmap_out.resolve()), "--SKIP_SPLIT_FOL", "True",
        "--GT_FOLDER", str(result_root), "--TRACKERS_FOLDER", str(result_root),
        "--TRACKERS_TO_EVAL", str(result_root), "--GT_LOC_FORMAT",
        "{gt_folder}{video_id}/{expression_id}/gt.txt",
        "--USE_PARALLEL", "False", "--PRINT_ONLY_COMBINED", "False",
        "--PLOT_CURVES", "False",
    ]
    log = out_root / "trackeval_l16.log"
    with log.open("w") as handle:
        subprocess.run(command, cwd=str(EVAL_RUN.parent), env=env,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    print(f"[l16-eval] TrackEval complete: {log}", flush=True)


def ensure_gt_links(dataset, out_root, queries, sequences):
    gt_root, _ = protocol(dataset)
    for sequence, expression, _spec in queries:
        if sequence not in sequences:
            continue
        source = gt_root / sequence / expression / "gt.txt"
        destination = out_root / "uidm16" / sequence / expression / "gt.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.exists() and not destination.exists():
            try:
                destination.symlink_to(source)
            except FileExistsError:
                # Multiple query shards may create the shared GT link set.
                # A matching existing link is the expected race outcome.
                if not destination.is_symlink() or destination.resolve() != source.resolve():
                    raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["kitti_v1", "kitti_v2", "dance"],
                        required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bank-root", default="outputs/l16/track_banks_dedup")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--max-per-frame", type=int, default=0)
    parser.add_argument("--no-belief", action="store_true")
    parser.add_argument("--no-cross-track", action="store_true")
    parser.add_argument("--no-motion", action="store_true")
    parser.add_argument("--stateless", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--max-seqs", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--predict-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    all_queries, _, _ = load_queries(args.dataset)
    sequences = sorted({query[0] for query in all_queries})
    if args.max_seqs:
        sequences = sequences[:args.max_seqs]
    sequence_set = set(sequences)
    queries = [query for query in all_queries if query[0] in sequence_set]
    if args.max_queries:
        queries = queries[:args.max_queries]
    ensure_gt_links(args.dataset, out_root, all_queries, sequence_set)
    if args.eval_only:
        run_eval(args.dataset, out_root, sequence_set)
        return
    queries = [query for query in queries if int(hashlib.md5(
        (query[0] + "+" + query[1]).encode()).hexdigest(), 16)
               % args.num_shards == args.shard]
    model, checkpoint = load_model(args.checkpoint, device)
    threshold = float(args.threshold if args.threshold is not None else
                      checkpoint["calibration"]["threshold"])
    grouped = {}
    for query in queries:
        grouped.setdefault(query[0], []).append(query)
    rows = []
    bank_dataset = "dance_eval" if args.dataset == "dance" else "kitti"
    started = time.time()
    for video_index, (video, video_queries) in enumerate(sorted(grouped.items())):
        bank_path = ROOT / args.bank_root / bank_dataset / f"{video}.pt"
        bank = load_track_bank(bank_path)
        for _sequence, expression, spec in video_queries:
            prediction = out_root / "uidm16" / video / expression / "predict.txt"
            if prediction.exists():
                continue
            result, timing = evaluate_query(
                model, bank, expression, spec, device, threshold,
                no_belief=args.no_belief,
                no_cross_track=args.no_cross_track,
                no_motion=args.no_motion, stateless=args.stateless,
                max_per_frame=args.max_per_frame)
            write_prediction(prediction, result)
            rows.append({"video": video, "expression": expression, **timing})
        print(f"[l16-eval] {args.dataset} shard={args.shard} "
              f"video={video} {video_index + 1}/{len(grouped)} "
              f"queries={len(video_queries)} elapsed={time.time()-started:.1f}",
              flush=True)
        del bank
    manifest = out_root / f"prediction_manifest_shard{args.shard:02d}.json"
    manifest.write_text(json.dumps({
        "dataset": args.dataset, "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(
            Path(args.checkpoint).read_bytes()).hexdigest(),
        "threshold": threshold, "max_per_frame": args.max_per_frame,
        "no_belief": args.no_belief, "no_cross_track": args.no_cross_track,
        "no_motion": args.no_motion, "stateless": args.stateless,
        "shard": args.shard, "num_shards": args.num_shards,
        "queries": rows, "wall_seconds": time.time() - started,
    }, indent=2) + "\n")
    if not args.predict_only:
        run_eval(args.dataset, out_root, sequence_set)


if __name__ == "__main__":
    main()
