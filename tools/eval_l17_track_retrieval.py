"""Evaluate the controlled iKUN port or L17 on frozen L16 track banks."""
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

from locatemot.models.ikun_bank_port import (  # noqa: E402
    IKunBankPort, pseudo_frequency_offset,
)
from locatemot.models.ikun_rn50_port import IKunRN50BankPort  # noqa: E402
from locatemot.models.l16_track_selector import expression_family_vector  # noqa: E402
from locatemot.models.l17_track_retriever import L17TrackSetRetriever  # noqa: E402
from locatemot.rmot.ikun_cache import RN50FeatureStore  # noqa: E402
from locatemot.rmot.track_bank import load_track_bank  # noqa: E402
from tools.eval_l13_rmot import (  # noqa: E402
    DANCE_GT, DANCE_SEQMAP, EVAL_RUN, PY, V1_GT, V1_SEQMAP, V2_GT,
    V2_SEQMAP, load_queries,
)
from tools.train_l16_track_selector import FEATURE_NAMES  # noqa: E402


def load_model(path: str, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["cfg"]
    name = checkpoint["model_name"]
    if name == "ikun":
        model = IKunBankPort(cfg["hidden"], cfg["heads"])
    elif name == "ikun_rn50":
        model = IKunRN50BankPort(cfg["hidden"], cfg["heads"])
    elif name == "l17":
        model = L17TrackSetRetriever(
            cfg["hidden"], holistic_query=cfg.get("holistic_query", False))
    else:
        raise ValueError(f"unsupported checkpoint model_name={name!r}")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def evaluate_query(model, checkpoint, bank, expression, spec, device,
                   threshold_logit, hysteresis_gap=0.0,
                   no_calibration=False, no_null=False,
                   no_identity=False, stateless=False,
                   max_per_frame=0, feature_store=None,
                   use_global=True, use_kum=True):
    tensors = bank["tensors"]
    model_name = checkpoint["model_name"]
    query = torch.as_tensor(np.asarray(spec, np.float32), device=device)
    family = expression_family_vector(expression).to(device)
    offset = 0.0
    if model_name in ("ikun", "ikun_rn50") and not no_calibration:
        offset = pseudo_frequency_offset(
            query, checkpoint.get("ikun_frequency_table"))
    state = {}
    selected_state: dict[int, bool] = {}
    rows = []
    probabilities = []
    started = time.time()
    for frame_index, frame_id in enumerate(tensors["frame_ids"].tolist()):
        start = int(tensors["frame_ptr"][frame_index])
        end = int(tensors["frame_ptr"][frame_index + 1])
        if end <= start:
            continue
        features = {
            name: tensors[name][start:end].to(device, non_blocking=True)
            for name in FEATURE_NAMES
        }
        if model_name == "ikun_rn50":
            features.update(feature_store.frame_features(
                bank["metadata"]["dataset"], bank["metadata"]["video_id"],
                frame_index, start, end, device))
        track_ids = tensors["track_id"][start:end].to(device)
        if stateless:
            state = {}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            if model_name == "ikun":
                output = model(features, query, track_ids, state)
            elif model_name == "ikun_rn50":
                output = model(features, query, track_ids, state,
                               use_global=use_global, use_kum=use_kum)
            else:
                output = model(
                    features, query, family, track_ids, state,
                    use_null=not no_null, use_identity=not no_identity)
        state = output["state"]
        logits = output["logits"].float() + float(offset)
        probability = torch.sigmoid(logits)
        probabilities.append(probability.cpu())
        raw_ids = track_ids.detach().cpu().tolist()
        keep_values = []
        for local_index, raw_track_id in enumerate(raw_ids):
            track_id = int(raw_track_id)
            was_selected = selected_state.get(track_id, False)
            boundary = threshold_logit - hysteresis_gap \
                if was_selected else threshold_logit
            selected = bool(float(logits[local_index]) >= boundary)
            selected_state[track_id] = selected
            if selected:
                keep_values.append(local_index)
        keep = torch.as_tensor(keep_values, dtype=torch.long, device=device)
        if max_per_frame > 0 and len(keep) > max_per_frame:
            order = torch.argsort(probability[keep], descending=True)
            keep = keep[order[:max_per_frame]]
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
        "pseudo_frequency_offset": float(offset),
    }


def write_prediction(path: Path, rows: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(",".join(
                f"{value:.6f}" if isinstance(value, float) else str(value)
                for value in row) + "\n")


def dataset_protocol(dataset: str):
    if dataset == "kitti_v1":
        return V1_GT, V1_SEQMAP
    if dataset == "kitti_v2":
        return V2_GT, V2_SEQMAP
    return DANCE_GT, DANCE_SEQMAP


def ensure_gt_links(dataset: str, out_root: Path, queries: list,
                    sequences: set[str]):
    gt_root, _ = dataset_protocol(dataset)
    for sequence, expression, _spec in queries:
        if sequence not in sequences:
            continue
        source = gt_root / sequence / expression / "gt.txt"
        destination = out_root / "uidm17" / sequence / expression / "gt.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.exists() and not destination.exists():
            try:
                destination.symlink_to(source)
            except FileExistsError:
                if (not destination.is_symlink() or
                        destination.resolve() != source.resolve()):
                    raise


def run_eval(dataset: str, out_root: Path, sequences: set[str]):
    gt_root, seqmap = dataset_protocol(dataset)
    result_root = (out_root / "uidm17").resolve()
    seqmap_out = out_root / "seqmap_l17.txt"
    with seqmap_out.open("w") as handle:
        for line in seqmap.read_text().splitlines():
            if not line.strip():
                continue
            sequence, expression = line.strip().split("+", 1)
            if sequence not in sequences:
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
    log = out_root / "trackeval_l17.log"
    with log.open("w") as handle:
        subprocess.run(command, cwd=str(EVAL_RUN.parent), env=env,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    print(f"[l17-eval] TrackEval complete: {log}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["kitti_v1", "kitti_v2", "dance"],
                        required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bank-root", default="outputs/l16/track_banks_dedup")
    parser.add_argument("--feature-cache-root",
                        default="outputs/l17/ikun_rn50_cache")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--threshold-logit", type=float, default=None)
    parser.add_argument("--hysteresis-gap", type=float, default=0.0)
    parser.add_argument("--max-per-frame", type=int, default=0)
    parser.add_argument("--no-calibration", action="store_true")
    parser.add_argument("--no-null", action="store_true")
    parser.add_argument("--no-identity", action="store_true")
    parser.add_argument("--no-global", action="store_true")
    parser.add_argument("--no-kum", action="store_true")
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
    feature_store = RN50FeatureStore(args.feature_cache_root, cache_size=2) \
        if checkpoint["model_name"] == "ikun_rn50" else None
    threshold_logit = float(
        args.threshold_logit if args.threshold_logit is not None else
        checkpoint["calibration"]["threshold_logit"])
    cfg = checkpoint.get("cfg", {})
    effective_no_null = args.no_null or cfg.get("no_null", False)
    effective_stateless = args.stateless or cfg.get("stateless", False)
    effective_no_identity = args.no_identity or cfg.get("stateless", False)
    grouped = {}
    for query_item in queries:
        grouped.setdefault(query_item[0], []).append(query_item)
    rows = []
    bank_dataset = "dance_eval" if args.dataset == "dance" else "kitti"
    started = time.time()
    for video_index, (video, video_queries) in enumerate(sorted(grouped.items())):
        bank_path = ROOT / args.bank_root / bank_dataset / f"{video}.pt"
        bank = load_track_bank(bank_path)
        for _sequence, expression, spec in video_queries:
            prediction = out_root / "uidm17" / video / expression / "predict.txt"
            if prediction.exists():
                continue
            result, timing = evaluate_query(
                model, checkpoint, bank, expression, spec, device,
                threshold_logit, hysteresis_gap=args.hysteresis_gap,
                no_calibration=args.no_calibration, no_null=effective_no_null,
                no_identity=effective_no_identity, stateless=effective_stateless,
                max_per_frame=args.max_per_frame,
                feature_store=feature_store,
                use_global=not args.no_global, use_kum=not args.no_kum)
            write_prediction(prediction, result)
            rows.append({"video": video, "expression": expression, **timing})
        print(f"[l17-eval] {args.dataset} shard={args.shard} "
              f"video={video} {video_index + 1}/{len(grouped)} "
              f"queries={len(video_queries)} elapsed={time.time()-started:.1f}",
              flush=True)
        del bank
    manifest = out_root / f"prediction_manifest_shard{args.shard:02d}.json"
    manifest.write_text(json.dumps({
        "dataset": args.dataset, "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(
            Path(args.checkpoint).read_bytes()).hexdigest(),
        "model_name": checkpoint["model_name"],
        "threshold_logit": threshold_logit,
        "hysteresis_gap": args.hysteresis_gap,
        "max_per_frame": args.max_per_frame,
        "no_calibration": args.no_calibration, "no_null": effective_no_null,
        "no_identity": effective_no_identity, "stateless": effective_stateless,
        "no_global": args.no_global, "no_kum": args.no_kum,
        "feature_cache_root": args.feature_cache_root,
        "shard": args.shard, "num_shards": args.num_shards,
        "queries": rows, "wall_seconds": time.time() - started,
    }, indent=2) + "\n")
    if not args.predict_only:
        run_eval(args.dataset, out_root, sequence_set)


if __name__ == "__main__":
    main()
