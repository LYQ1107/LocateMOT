"""Materialize the four missing Refer-KITTI-V2 training records under L16.

Each record contains the fixed L10 Detic proposals, detector scores/classes,
CLIP ViT-B/32 crop embeddings, GT-to-candidate mappings for train-only
supervision, and a zero PBD slot.  Real crop PBD is stored separately by
``cache_l10_kitti_pbd.py`` under the L16 cache root and never faked here.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch

from build_l10_refer_kitti import (
    KITTI_IMGS,
    V2_ROOT,
    encode_clip,
    load_expressions,
    load_labels,
    match_dets,
)


DEFAULT_SEQS = ("0016", "0017", "0018", "0020")


def encode_text(model, texts, device):
    import clip

    result = {}
    for start in range(0, len(texts), 256):
        batch = texts[start:start + 256]
        with torch.no_grad():
            value = model.encode_text(clip.tokenize(batch).to(device)).float()
        value = value / value.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        for text, feature in zip(batch, value.cpu().numpy()):
            result[text] = feature.astype(np.float32)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seqs", default=",".join(DEFAULT_SEQS))
    ap.add_argument("--dets-root", default="outputs/l16/data/kitti_missing/dets")
    ap.add_argument("--out", default="outputs/l16/data/kitti_missing/records")
    args = ap.parse_args()

    seqs = tuple(x.strip() for x in args.seqs.split(",") if x.strip())
    if not seqs or any(x not in DEFAULT_SEQS for x in seqs):
        raise SystemExit(f"--seqs must be a subset of {DEFAULT_SEQS}")
    det_root = Path(args.dets_root).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    all_expressions = {seq: load_expressions(seq) for seq in seqs}
    texts = sorted({entry["sentence"] for entries in all_expressions.values()
                    for entry in entries.values()})
    text_features = encode_text(clip_model, texts, device)
    metadata = {}
    manifest = {
        "sequences": list(seqs),
        "proposal_source": "Detic-SwinB MASA checkpoint, L10 DLA protocol",
        "proposal_threshold": 0.05,
        "proposal_budget": 50,
        "clip_model": "ViT-B/32",
        "pbd": "external L16 crop cache; zeros in record are explicit slots",
        "records": {},
    }

    for seq in sorted(seqs):
        image_dir = KITTI_IMGS / seq
        image_paths = sorted(image_dir.glob("*.png"))
        labels = load_labels(seq)
        frames = []
        candidate_total = 0
        matched_total = 0
        image_size = None
        for frame_index, image_path in enumerate(image_paths):
            frame = int(image_path.stem)
            det_path = det_root / seq / f"{frame:06d}.pth"
            if not det_path.exists():
                raise FileNotFoundError(det_path)
            with det_path.open("rb") as handle:
                det = pickle.load(handle)
            boxes5 = det["det_bboxes"].numpy().astype(np.float32)
            det_labels = det["det_labels"].numpy().astype(np.int32)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"unreadable image: {image_path}")
            height, width = image.shape[:2]
            image_size = [width, height]
            gt_boxes = {}
            for _cls, track_id, x, y, box_w, box_h in labels.get(frame, []):
                gt_boxes[str(track_id)] = [
                    x * width, y * height,
                    (x + box_w) * width, (y + box_h) * height,
                ]
            boxes = boxes5[:, :4] if len(boxes5) else np.zeros((0, 4), np.float32)
            cand_gt = match_dets(boxes, gt_boxes) if len(boxes) else []
            crops = []
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                crops.append(image[y1:y2, x1:x2]
                             if x2 - x1 >= 2 and y2 - y1 >= 2
                             else np.zeros((2, 2, 3), np.uint8))
            crop_features = encode_clip(clip_model, crops, device) \
                if crops else np.zeros((0, 512), np.float16)
            frames.append({
                "frame": frame,
                "boxes": boxes.astype(np.float32),
                "gen": boxes5[:, 4].astype(np.float32)
                if len(boxes5) else np.zeros(0, np.float32),
                "label": det_labels,
                "clip": crop_features,
                "pbd": np.zeros((len(boxes), 2048), np.float16),
                "pbd_valid": np.zeros(len(boxes), np.bool_),
                "cand_gt": cand_gt,
                "gt_boxes": gt_boxes,
                "orig_idx": np.arange(len(boxes), dtype=np.int32),
                "proposal_source": "l16_detic_dla",
            })
            candidate_total += len(boxes)
            matched_total += sum(value is not None for value in cand_gt)
            if (frame_index + 1) % 100 == 0:
                print(f"[l16-kitti-build] {seq} "
                      f"{frame_index + 1}/{len(image_paths)}", flush=True)

        record = {
            "video_id": seq,
            "image_size": image_size,
            "frames": frames,
            "proposal_source": "l16_detic_dla",
            "clip_model": "ViT-B/32",
            "pbd_cache_root": "outputs/l16/data/kitti_missing/pbd",
        }
        with (out_root / f"{seq}.pkl").open("wb") as handle:
            pickle.dump(record, handle, protocol=pickle.HIGHEST_PROTOCOL)
        entries = []
        for entry in sorted(all_expressions[seq].values(),
                            key=lambda item: item["expression"]):
            entries.append({
                **entry,
                "spec": text_features[entry["sentence"]].tolist(),
                "split": "train_val" if seq == "0018" else "train",
            })
        metadata[seq] = entries
        manifest["records"][seq] = {
            "frames": len(frames),
            "candidates": candidate_total,
            "candidate_gt_matches": matched_total,
            "expressions": len(entries),
            "image_size": image_size,
        }
        print(f"[l16-kitti-build] {seq} frames={len(frames)} "
              f"candidates={candidate_total} expressions={len(entries)}",
              flush=True)

    (out_root / "expressions.json").write_text(json.dumps(metadata, indent=1))
    (out_root / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[l16-kitti-build] done out={out_root}", flush=True)


if __name__ == "__main__":
    main()

