"""Cache one causal all-object L11 UIDM pass per video for Stage L16."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.data.token_cache import cache_key, read_frame_cache  # noqa: E402
from locatemot.models.l8_unified import L8UnifiedUIDM, load_l8_state  # noqa: E402
from locatemot.rmot.track_bank import build_track_bank, save_track_bank  # noqa: E402
from tools.eval_l13_rmot import (  # noqa: E402
    SIZES, load_dance_entries, make_dance_frames,
)

CHECKPOINT = ROOT / "outputs/l11/checkpoints/uidm_l11_main/step11000.pt"
CHECKPOINT_SHA256 = "f6529743630e335b947118bf860cc2215a95315a4c551351e6866ea9e1076343"
KITTI_OLD = ROOT / "outputs/l11/data/rmot_kitti"
KITTI_NEW = ROOT / "outputs/l16/data/kitti_missing/records"
KITTI_PBD_OLD = ROOT / "outputs/l10/cache/kitti_pbd"
KITTI_PBD_NEW = ROOT / "outputs/l16/data/kitti_missing/pbd"
DANCE_TRAIN = ROOT / "outputs/l8/data/rmot_train"
PROTOCOL = ROOT / "outputs/l16/data/protocol/split_manifest.json"
GENERIC_TEXT = "all objects in the scene"


def load_shared(device):
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("cfg", {})
    model = L8UnifiedUIDM(
        **SIZES[cfg.get("model", "base")],
        no_interaction=cfg.get("no_interaction", False),
        use_cue_rel=cfg.get("use_cue_rel", False),
        mode=cfg.get("mode", "unified"),
        sem_in_core=cfg.get("sem_in_core", True),
        cond_gated=cfg.get("cond_gated", False),
        spec_conditioned=cfg.get("spec_conditioned", False),
        trajectory_memory=cfg.get("trajectory_memory", True)).to(device)
    missing, unexpected = load_l8_state(model, checkpoint["model"])
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: {missing} {unexpected}")
    return model.eval()


def generic_embedding(device):
    import clip
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    with torch.no_grad():
        value = model.encode_text(clip.tokenize([GENERIC_TEXT]).to(device)).float()
        value = value / value.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    result = value[0].cpu().numpy().astype(np.float32)
    del model
    torch.cuda.empty_cache()
    return result


def kitti_frames(video):
    path = KITTI_NEW / f"{video}.pkl"
    if not path.exists():
        path = KITTI_OLD / f"{video}.pkl"
    record = pickle.load(path.open("rb"))
    frames = []
    for raw in record["frames"]:
        boxes = np.asarray(raw["boxes"], np.float32)
        n = len(boxes)
        pbd = np.asarray(raw.get("pbd", np.zeros((n, 2048))), np.float32)
        root = KITTI_PBD_NEW if video in {"0016", "0017", "0018", "0020"} \
            else KITTI_PBD_OLD
        cached = read_frame_cache(
            str(root), cache_key("kitti", video, int(raw["frame"]), "pbd_full"))
        if cached is not None:
            values = np.asarray(cached["features"]["pbd_box_end_last"], np.float32)
            original = np.asarray(raw.get("orig_idx", np.arange(n)), np.int64)
            if len(values) and len(original) and int(original.max()) < len(values):
                pbd = values[original]
        if pbd.shape != (n, 2048):
            pbd = np.zeros((n, 2048), np.float32)
        frames.append({
            "frame": int(raw["frame"]), "boxes": boxes,
            "gen": np.asarray(raw.get("gen", np.zeros(n)), np.float32),
            "clip": np.asarray(raw.get("clip", np.zeros((n, 512))), np.float32),
            "pbd": pbd, "cand_gt": list(raw.get("cand_gt", [None] * n)),
            "gt_boxes": raw.get("gt_boxes", {}),
        })
    return frames, record["image_size"]


def dance_train_frames(video):
    record = pickle.load((DANCE_TRAIN / f"{video}.pkl").open("rb"))
    return record["frames"], record["image_size"]


def dance_eval_frames(video, entries):
    frames, image_size = make_dance_frames(video, entries)
    raw_by_frame = {int(entry["frame"]): entry for entry in entries[video]}
    for frame in frames:
        raw = raw_by_frame[int(frame["frame"])]
        candidate_gt = [None] * len(frame["boxes"])
        for gt_id, match in raw.get("matched", {}).items():
            index = int(match["candidate"])
            if 0 <= index < len(candidate_gt):
                candidate_gt[index] = str(gt_id)
        frame["cand_gt"] = candidate_gt
        frame["gt_boxes"] = raw.get("gt_boxes", {})
    return frames, image_size


def query_counts():
    counts = {}
    sources = [KITTI_OLD / "expressions.json", KITTI_NEW / "expressions.json",
               ROOT / "outputs/l16/data/protocol/refer_dance_expressions.json"]
    for path in sources:
        if not path.exists():
            continue
        for video, values in json.loads(path.read_text()).items():
            counts[video] = max(counts.get(video, 0), len(values))
    return counts


def deduplicate_exact(frames):
    """Retain one proposal for each bit-identical box, without a tuned NMS."""
    result = []
    for raw in frames:
        frame = dict(raw)
        boxes = np.asarray(frame.get("boxes", []), np.float32).reshape(-1, 4)
        groups = {}
        keep = []
        for index, box in enumerate(boxes):
            key = box.tobytes()
            if key not in groups:
                groups[key] = len(keep)
                keep.append(index)
        keep_array = np.asarray(keep, np.int64)
        frame["_source_count"] = len(boxes)
        frame["source_index"] = keep_array.astype(np.int32)
        for name in ("boxes", "gen", "clip", "pbd"):
            if name in frame:
                frame[name] = np.asarray(frame[name])[keep_array]
        raw_gt = list(frame.get("cand_gt", [None] * len(boxes)))
        merged_gt = [raw_gt[index] if index < len(raw_gt) else None
                     for index in keep]
        # Candidate retention is GT-independent.  For training supervision
        # only, merge a non-empty label carried by another exact duplicate.
        for index, box in enumerate(boxes):
            representative = groups[box.tobytes()]
            value = raw_gt[index] if index < len(raw_gt) else None
            if merged_gt[representative] is None and value is not None:
                merged_gt[representative] = value
        frame["cand_gt"] = merged_gt
        result.append(frame)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["kitti", "dance_train", "dance_eval"],
                        required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--out", default="outputs/l16/track_banks")
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--deduplicate-exact", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise RuntimeError("L11 checkpoint hash changed")
    protocol = json.loads(PROTOCOL.read_text())
    splits = {}
    if args.dataset == "kitti":
        for split in ("train", "train_val", "official_eval"):
            splits.update({video: split for video in protocol["kitti_v2"][split]})
    elif args.dataset == "dance_train":
        for split in ("train", "train_val"):
            splits.update({video: split for video in protocol["refer_dance"][split]})
    else:
        splits.update({video: "official_eval"
                       for video in protocol["refer_dance"]["official_eval"]})
    videos = sorted(args.videos if args.videos else splits)
    videos = [video for video in videos if int(hashlib.md5(video.encode()).hexdigest(), 16)
              % args.num_shards == args.shard]
    output_root = (ROOT / args.out / args.dataset).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[l16-bank] dataset={args.dataset} shard={args.shard}/{args.num_shards} "
          f"videos={len(videos)} gpu={args.gpu}", flush=True)
    if not videos:
        return
    spec = generic_embedding(device)
    model = load_shared(device)
    dance_entries = load_dance_entries() if args.dataset == "dance_eval" else None
    counts = query_counts()
    worker_rows = []
    started = time.time()
    for index, video in enumerate(videos):
        output = output_root / f"{video}.pt"
        if output.with_suffix(".complete").exists() and not args.overwrite:
            print(f"[l16-bank] skip complete {video}", flush=True)
            continue
        if args.dataset == "kitti":
            frames, image_size = kitti_frames(video)
        elif args.dataset == "dance_train":
            frames, image_size = dance_train_frames(video)
        else:
            frames, image_size = dance_eval_frames(video, dance_entries)
        if args.deduplicate_exact:
            frames = deduplicate_exact(frames)
        t0 = time.time()
        bank, audit, labels = build_track_bank(
            model, frames, image_size, spec, args.dataset, video,
            splits[video], CHECKPOINT_SHA256, counts.get(video, 0))
        save_track_bank(bank, audit, labels, output,
                        save_supervision=splits[video] != "official_eval")
        worker_rows.append(audit)
        print(f"[l16-bank] {video} {index + 1}/{len(videos)} "
              f"frames={audit['frames']} obs={audit['observations']} "
              f"tracks={audit['unique_tracks']} seconds={time.time()-t0:.1f}",
              flush=True)
    worker = output_root / f"worker_{args.shard:02d}.json"
    worker.write_text(json.dumps({
        "dataset": args.dataset, "shard": args.shard,
        "num_shards": args.num_shards, "videos": worker_rows,
        "deduplicate_exact": args.deduplicate_exact,
        "seconds": time.time() - started,
    }, indent=2) + "\n")
    print(f"[l16-bank] done shard={args.shard} seconds={time.time()-started:.1f}",
          flush=True)


if __name__ == "__main__":
    main()
