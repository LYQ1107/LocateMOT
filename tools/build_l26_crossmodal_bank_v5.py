#!/usr/bin/env python3
"""Build the independent L26 DINOv2/word-token cross-modal bank.

Visual features are frozen DINOv2 patch tokens and text features are frozen
RoBERTa word-level hidden states.  GT is never used for sampling or fitting;
the existing candidate rows and frame pointers are copied from L19, while GT
records are consumed only later by the training/audit data loader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
RAW = ROOT / "data/kitti_tracking_training/image_02"
OLD = ROOT / "outputs/l19/dual_banks_features/kitti"
EXP1 = ROOT / "outputs/l11/data/rmot_kitti/expressions.json"
EXP2 = ROOT / "outputs/l16/data/kitti_missing/records/expressions.json"
DINO_HUB = Path("/home/lwr/.cache/torch/hub/facebookresearch_dinov2_main")
DINO_WEIGHTS = Path("/home/lwr/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth")
ROBERTA = Path("/home/lwr/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_expressions() -> list[dict]:
    by_key = {}
    for path in (EXP1, EXP2):
        data = json.loads(path.read_text())
        for video, rows in data.items():
            for row in rows:
                key = (str(video), str(row["expression"]))
                by_key[key] = {"video": str(video), **row}
    return [by_key[k] for k in sorted(by_key)]


def clean_state(state):
    if "teacher" in state and isinstance(state["teacher"], dict):
        state = state["teacher"]
    if "student" in state and isinstance(state["student"], dict):
        state = state["student"]
    out = {}
    for key, value in state.items():
        k = key
        for prefix in ("teacher.backbone.", "student.backbone.", "backbone.", "module."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        out[k] = value
    return out


def resize_input(image: np.ndarray, device: torch.device) -> torch.Tensor:
    rgb = cv2.cvtColor(cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((x - mean) / std).to(device)


def sample_map(fmap: torch.Tensor, points: np.ndarray, width: int, height: int) -> np.ndarray:
    # fmap: [1,C,Hm,Wm], points are raw-image xy coordinates.
    norm = points.astype(np.float32).copy()
    norm[:, 0] = norm[:, 0] / max(1.0, width) * 2.0 - 1.0
    norm[:, 1] = norm[:, 1] / max(1.0, height) * 2.0 - 1.0
    grid = torch.from_numpy(norm).to(fmap.device).view(1, -1, 1, 2)
    with torch.inference_mode():
        val = F.grid_sample(fmap, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return val[0, :, :, 0].transpose(0, 1).float().cpu().numpy()


def clipped_box(box: np.ndarray, width: int, height: int, scale: float) -> np.ndarray:
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    w = (box[2] - box[0]) * scale
    h = (box[3] - box[1]) * scale
    return np.asarray([max(0, cx-w*0.5), max(0, cy-h*0.5), min(width, cx+w*0.5), min(height, cy+h*0.5)], np.float32)


def points_for(box: np.ndarray, width: int, height: int, scale: float) -> np.ndarray:
    b = clipped_box(box, width, height, scale)
    xs = np.linspace(b[0] + 0.2*(b[2]-b[0]), b[2] - 0.2*(b[2]-b[0]), 3)
    ys = np.linspace(b[1] + 0.2*(b[3]-b[1]), b[3] - 0.2*(b[3]-b[1]), 3)
    return np.asarray([[x, y] for y in ys for x in xs], np.float32)


def build_visual(args, videos: list[str], out: Path, summary: dict) -> None:
    device = torch.device(args.device)
    model = torch.hub.load(str(DINO_HUB), "dinov2_vitb14", source="local", pretrained=False)
    state = torch.load(DINO_WEIGHTS, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(clean_state(state), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"DINOv2 state mismatch missing={len(missing)} unexpected={len(unexpected)}")
    model.eval().to(device)
    for video in videos:
        start = time.time()
        old = torch.load(OLD / f"{video}.pt", map_location="cpu", weights_only=False)
        t = old["tensors"]
        frames = t["frame"].numpy().astype(np.int64)
        ptr = t["frame_ptr"].numpy().astype(np.int64)
        boxes = t["box"].float().numpy().astype(np.float32)
        tracks = t["track_id"].numpy().astype(np.int64)
        pools = t["pool_id"].numpy().astype(np.int64)
        frame_ids = t["frame_ids"].numpy().astype(np.int64)
        if not np.array_equal(frames[ptr[:-1]], frame_ids):
            raise AssertionError(f"frame pointer mismatch {video}")
        maps_dir = out / "dense_maps"
        map_paths = []
        maps = {}
        for frame in frame_ids.tolist():
            image_path = RAW / video / f"{int(frame):06d}.png"
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(image_path)
            with torch.inference_mode():
                feat = model.forward_features(resize_input(image, device))["x_norm_patchtokens"]
            side = int(round(feat.shape[1] ** 0.5))
            if side * side != feat.shape[1] or feat.shape[-1] != 768:
                raise AssertionError(f"unexpected DINO output {tuple(feat.shape)}")
            fmap = feat.reshape(1, side, side, 768).permute(0, 3, 1, 2).contiguous().half().cpu()
            if not bool(torch.isfinite(fmap.float()).all()):
                raise FloatingPointError(f"nonfinite map {video}/{frame}")
            path = maps_dir / f"{video}_{int(frame):06d}.pt"
            torch.save({"video_id": video, "frame_id": int(frame), "feature_map": fmap,
                        "shape": list(fmap.shape), "patch_stride": 14,
                        "raw_image": str(image_path), "raw_image_sha256": sha(image_path)}, str(path)+".tmp")
            os.replace(str(path)+".tmp", path)
            maps[int(frame)] = fmap.to(device).float()
            map_paths.append({"frame_id": int(frame), "path": str(path), "shape": list(fmap.shape), "patch_stride": 14})
        fields = {k: [] for k in ("dino_roi_tokens_v5", "dino_context_1p5_tokens_v5", "dino_context_3_tokens_v5",
                                  "dino_prev_roi_tokens_v5", "dino_roi_v5", "dino_context_1p5_v5", "dino_context_3_v5",
                                  "dino_prev_roi_v5", "roi_coords_v5", "context_1p5_coords_v5", "context_3_coords_v5")}
        prev = {}
        for fi, frame in enumerate(frame_ids.tolist()):
            begin, end = int(ptr[fi]), int(ptr[fi+1])
            image = cv2.imread(str(RAW / video / f"{int(frame):06d}.png"), cv2.IMREAD_COLOR)
            height, width = image.shape[:2]
            fmap = maps[int(frame)]
            cur = []
            for row in range(begin, end):
                roi_p = points_for(boxes[row], width, height, 1.0)
                c15_p = points_for(boxes[row], width, height, 1.5)
                c3_p = points_for(boxes[row], width, height, 3.0)
                roi = sample_map(fmap, roi_p, width, height)
                c15 = sample_map(fmap, c15_p, width, height)
                c3 = sample_map(fmap, c3_p, width, height)
                ns = (int(pools[row]), int(tracks[row]))
                previous = prev.get(ns, np.zeros((9, 768), np.float32))
                fields["dino_roi_tokens_v5"].append(roi.astype(np.float32))
                fields["dino_context_1p5_tokens_v5"].append(c15.astype(np.float32))
                fields["dino_context_3_tokens_v5"].append(c3.astype(np.float32))
                fields["dino_prev_roi_tokens_v5"].append(previous.astype(np.float32))
                fields["dino_roi_v5"].append(roi.mean(0).astype(np.float32))
                fields["dino_context_1p5_v5"].append(c15.mean(0).astype(np.float32))
                fields["dino_context_3_v5"].append(c3.mean(0).astype(np.float32))
                fields["dino_prev_roi_v5"].append(previous.mean(0).astype(np.float32))
                fields["roi_coords_v5"].append((roi_p / np.asarray([width, height], np.float32)).astype(np.float32))
                fields["context_1p5_coords_v5"].append((c15_p / np.asarray([width, height], np.float32)).astype(np.float32))
                fields["context_3_coords_v5"].append((c3_p / np.asarray([width, height], np.float32)).astype(np.float32))
                cur.append((ns, roi.astype(np.float32)))
            for ns, roi in cur:
                prev[ns] = roi
        tensors = {k: v.clone() for k, v in t.items()}
        for key, vals in fields.items():
            tensors[key] = torch.from_numpy(np.asarray(vals, dtype=np.float16))
        tensors["dense_map_frame_index_v5"] = torch.repeat_interleave(torch.arange(len(frame_ids)), torch.diff(t["frame_ptr"]))
        for key, value in tensors.items():
            if torch.is_floating_point(value) and not bool(torch.isfinite(value.float()).all()):
                raise FloatingPointError(f"nonfinite candidate field {video}/{key}")
        if not torch.equal(tensors["frame_ptr"], t["frame_ptr"]) or not torch.equal(tensors["frame_ids"], t["frame_ids"]):
            raise AssertionError(f"row/frame alignment changed {video}")
        metadata = dict(old["metadata"])
        metadata.update({
            "format": "locatemot-l26-crossmodal-candidate-bank-v5",
            "stage": "L26",
            "parent_bank": str(OLD / f"{video}.pt"),
            "parent_bank_sha256": sha(OLD / f"{video}.pt"),
            "manifest_sha256": summary["fast_manifest_sha256"],
            "visual_backbone": "frozen DINOv2 ViT-B/14",
            "dino_checkpoint": str(DINO_WEIGHTS),
            "dino_checkpoint_sha256": sha(DINO_WEIGHTS),
            "dense_map_shape": [1, 768, int(side), int(side)],
            "dense_map_patch_stride": 14,
            "dense_map_files": map_paths,
            "text_backbone": "frozen local RoBERTa-base word-level hidden states",
            "gt_used_for_features": False,
            "tracker_modified": False,
            "row_order_preserved": True,
            "frame_ptr_preserved": True,
            "candidate_sampling": "fixed bbox ROI/context points; no GT-driven selection",
            "new_feature_dims": {k: list(tensors[k].shape[1:]) for k in fields},
        })
        target = out / "kitti" / f"{video}.pt"
        torch.save({"metadata": metadata, "tensors": tensors}, str(target)+".tmp")
        os.replace(str(target)+".tmp", target)
        (target.with_suffix(".complete")).write_text("ok\n")
        summary["videos"][video] = {"rows": int(len(boxes)), "frames": int(len(frame_ids)), "map_files": len(map_paths), "elapsed_sec": time.time()-start, "path": str(target)}
        print(json.dumps(summary["videos"][video]), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_text(args, expressions: list[dict], out: Path, summary: dict) -> None:
    from transformers import AutoModel, AutoTokenizer
    device = torch.device(args.text_device)
    tokenizer = AutoTokenizer.from_pretrained(str(ROBERTA), local_files_only=True)
    model = AutoModel.from_pretrained(str(ROBERTA), local_files_only=True).eval().to(device)
    max_length = int(args.max_text_length)
    texts = [str(row.get("sentence", row["expression"])) for row in expressions]
    hidden, masks, token_ids = [], [], []
    with torch.inference_mode():
        for i in range(0, len(texts), 64):
            enc = tokenizer(texts[i:i+64], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").to(device)
            h = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
            hidden.append(h.cpu().half())
            masks.append(enc["attention_mask"].cpu().bool())
            token_ids.append(enc["input_ids"].cpu())
    text_tensor = torch.cat(hidden, 0).contiguous()
    mask_tensor = torch.cat(masks, 0).contiguous()
    if not bool(torch.isfinite(text_tensor.float()).all()):
        raise FloatingPointError("nonfinite text hidden states")
    torch.save({"input_ids": torch.cat(token_ids, 0).contiguous(), "token_hidden": text_tensor,
                "attention_mask": mask_tensor, "embedding_dim": text_tensor.shape[-1],
                "max_length": max_length, "backbone": "roberta-base"}, out / "text_tokens.pt")
    (out / "text_manifest.json").write_text(json.dumps({
        "format": "locatemot-l26-word-token-bank-v1", "count": len(expressions), "max_length": max_length,
        "hidden_shape": list(text_tensor.shape), "query_index_source": "stable sorted (video, expression)",
        "expressions": [{"query_index": i, "video": row["video"], "expression": row["expression"],
                         "sentence": row.get("sentence", row["expression"])} for i, row in enumerate(expressions)],
        "weights": str(ROBERTA), "weights_sha256": sha(ROBERTA / "model.safetensors"),
        "gt_used": False,
    }, indent=2) + "\n")
    summary["text"] = {"count": len(expressions), "shape": list(text_tensor.shape), "max_length": max_length, "weights": str(ROBERTA), "weights_sha256": sha(ROBERTA / "model.safetensors")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="outputs/l26/candidate_bank_v5_crossmodal")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--text-device", default="cuda:0")
    ap.add_argument("--max-text-length", type=int, default=64)
    args = ap.parse_args()
    out = Path(args.out_root)
    if not out.is_absolute(): out = ROOT / out
    if out.exists(): raise FileExistsError(out)
    out.mkdir(parents=True); (out / "kitti").mkdir(); (out / "dense_maps").mkdir()
    expressions = load_expressions()
    split = json.loads((ROOT / "outputs/l16/data/protocol/split_manifest.json").read_text())["kitti_v2"]
    videos = sorted(set(split["train"] + split["train_val"] + split["official_eval"]))
    old_videos = {p.stem for p in OLD.glob("*.pt")}
    if set(videos) != old_videos: raise AssertionError(f"video set mismatch split={len(videos)} bank={len(old_videos)}")
    manifest = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
    summary = {"format": "locatemot-l26-crossmodal-bank-v5-build-v1", "stage": "L26", "videos": {},
               "manifest": str(manifest), "fast_manifest_sha256": sha(manifest), "query_count": len(expressions),
               "train_query_count": sum(str(x["video"]) in set(split["train"]) for x in expressions),
               "train_val_query_count": sum(str(x["video"]) in set(split["train_val"]) for x in expressions),
               "official_eval_query_count": sum(str(x["video"]) in set(split["official_eval"]) for x in expressions),
               "gt_used_for_features": False, "tracker_modified": False, "dino_checkpoint_sha256": sha(DINO_WEIGHTS),
               "roberta_checkpoint_sha256": sha(ROBERTA / "model.safetensors"), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    try:
        build_visual(args, videos, out, summary)
        build_text(args, expressions, out, summary)
        summary["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        (out / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (out / "BUILD_COMPLETE").write_text("ok\n")
        print(json.dumps(summary, indent=2), flush=True)
    except Exception as exc:
        (out / "INCOMPLETE.md").write_text(f"# L26 v5 build incomplete\n\nFirst actionable error: `{type(exc).__name__}: {exc}`\n")
        raise


if __name__ == "__main__": main()
