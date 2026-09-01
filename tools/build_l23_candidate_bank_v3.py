"""Build the independent Stage L23 dense candidate bank.

This is a structure adaptation using the locally available frozen OpenAI
CLIP ViT-B/32 visual transformer.  It exposes the projected 7x7 patch tokens
before CLIP's global pooling and samples them using candidate boxes only.  It
does not read GT to choose crops or points and never modifies the L19/L22
banks, tracker, or evaluator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
VISUAL_KEYS = ("dense_roi", "dense_points", "dense_context_1p5",
               "dense_context_3", "dense_prev_roi")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def box_clip(box: np.ndarray, width: int, height: int, scale: float = 1.0) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    bw, bh = max(1.0, x2 - x1) * scale, max(1.0, y2 - y1) * scale
    return np.asarray((max(0.0, cx - bw * 0.5), max(0.0, cy - bh * 0.5),
                       min(float(width), cx + bw * 0.5),
                       min(float(height), cy + bh * 0.5)), np.float32)


def full_frame_preprocess(image: np.ndarray) -> np.ndarray:
    """Use a fixed full-frame resize so the 7x7 map covers every bbox."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_CUBIC).astype(np.float32)


@torch.inference_mode()
def dense_clip_map(model, image: np.ndarray, device: torch.device) -> torch.Tensor:
    """Return projected, per-patch-normalized CLIP tokens as [1,512,7,7]."""
    view = full_frame_preprocess(image)
    value = torch.from_numpy(view).permute(2, 0, 1).unsqueeze(0).to(device)
    value = value.to(dtype=model.dtype).div(255.0)
    mean = torch.as_tensor(MEAN, device=device, dtype=model.dtype)[None, :, None, None]
    std = torch.as_tensor(STD, device=device, dtype=model.dtype)[None, :, None, None]
    value = (value - mean) / std
    visual = model.visual
    tokens = visual.conv1(value)
    tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1).permute(0, 2, 1)
    cls = visual.class_embedding.to(tokens.dtype) + torch.zeros(
        tokens.shape[0], 1, tokens.shape[-1], device=device, dtype=tokens.dtype)
    tokens = torch.cat((cls, tokens), dim=1)
    tokens = tokens + visual.positional_embedding.to(tokens.dtype)
    tokens = visual.ln_pre(tokens)
    tokens = visual.transformer(tokens.permute(1, 0, 2)).permute(1, 0, 2)
    tokens = visual.ln_post(tokens[:, 1:, :])
    if visual.proj is not None:
        tokens = tokens @ visual.proj.to(tokens.dtype)
    grid = int(round(tokens.shape[1] ** 0.5))
    if grid * grid != tokens.shape[1]:
        raise ValueError(f"unexpected CLIP patch token count: {tokens.shape}")
    tokens = F.normalize(tokens, dim=-1)
    return tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], grid, grid)


def grid_sample_points(feature_map: torch.Tensor, points: np.ndarray) -> np.ndarray:
    """Sample [x,y] pixel points from a [1,C,H,W] map with fixed alignment."""
    if len(points) == 0:
        return np.zeros((0, int(feature_map.shape[1])), np.float32)
    _, _, height, width = feature_map.shape
    coords = np.asarray(points, np.float32)
    coords = coords.copy()
    coords[:, 0] = coords[:, 0].clip(0.0, 1.0) * 2.0 - 1.0
    coords[:, 1] = coords[:, 1].clip(0.0, 1.0) * 2.0 - 1.0
    grid = torch.from_numpy(coords).to(feature_map.device, dtype=feature_map.dtype)
    grid = grid.reshape(1, len(coords), 1, 2)
    sampled = F.grid_sample(feature_map, grid, mode="bilinear",
                            padding_mode="border", align_corners=True)
    return sampled[0, :, :, 0].transpose(0, 1).float().cpu().numpy()


def normalized_box(box: np.ndarray, width: int, height: int, scale: float = 1.0) -> np.ndarray:
    b = box_clip(box, width, height, scale)
    return np.asarray((b[0] / width, b[1] / height, b[2] / width, b[3] / height), np.float32)


def region_points(box: np.ndarray, width: int, height: int, scale: float,
                  count: int = 3) -> np.ndarray:
    x1, y1, x2, y2 = normalized_box(box, width, height, scale)
    xs = np.linspace(x1, x2, count + 2, dtype=np.float32)[1:-1]
    ys = np.linspace(y1, y2, count + 2, dtype=np.float32)[1:-1]
    return np.asarray([(x, y) for y in ys for x in xs], np.float32)


def fixed_point_set(box: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = normalized_box(box, width, height)
    dx, dy = (x2 - x1) * 0.20, (y2 - y1) * 0.20
    return np.asarray(((x1 + x2) * 0.5, (y1 + y2) * 0.5,
                       x1 + dx, y1 + dy, x2 - dx, y1 + dy,
                       x1 + dx, y2 - dy, x2 - dx, y2 - dy), np.float32).reshape(5, 2)


def row_key(video: str, frame: int, pool: int, track: int, candidate: int) -> str:
    return f"{video}:{frame}:{pool}:{track}:{candidate}"


def check_old_v2_alignment(old: dict, v2: dict, old_labels: list, v2_labels: list,
                           video: str) -> None:
    old_t, v2_t = old["tensors"], v2["tensors"]
    if len(old_labels) != len(v2_labels) or old_labels != v2_labels:
        raise AssertionError(f"label sidecar mismatch for {video}")
    for key in ("frame_ptr", "frame_ids", "frame", "box", "pool_id",
                "track_id", "candidate_index"):
        if not torch.equal(old_t[key], v2_t[key]):
            raise AssertionError(f"old/v2 alignment mismatch in {video}/{key}")
    n = len(v2_t["track_id"])
    if int(v2_t["frame_ptr"][-1]) != n:
        raise AssertionError(f"frame pointer does not end at rows for {video}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/l19/protocol/kitti_fast_eval_manifest.json")
    ap.add_argument("--old-bank-root", default="outputs/l19/dual_banks_features")
    ap.add_argument("--v2-bank-root", default="outputs/l22/candidate_bank_v2")
    ap.add_argument("--raw-root", default="data/kitti_tracking_training/image_02")
    ap.add_argument("--out-root", default="outputs/l23/candidate_bank_v3")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--map-dtype", choices=("float16", "float32"), default="float16")
    args = ap.parse_args()
    manifest = Path(args.manifest); old_root = Path(args.old_bank_root)
    v2_root = Path(args.v2_bank_root); raw_root = Path(args.raw_root); out_root = Path(args.out_root)
    paths = locals()
    for name in ("manifest", "old_root", "v2_root", "raw_root", "out_root"):
        value = paths[name]
        if not value.is_absolute():
            paths[name] = ROOT / value
    manifest, old_root, v2_root, raw_root, out_root = [paths[x] for x in
        ("manifest", "old_root", "v2_root", "raw_root", "out_root")]
    if out_root.exists():
        raise FileExistsError(f"refusing to overwrite v3 bank root: {out_root}")
    out_root.mkdir(parents=True, exist_ok=False)
    (out_root / "kitti").mkdir()
    (out_root / "dense_maps").mkdir()
    manifest_data = json.loads(manifest.read_text())
    queries = manifest_data["queries"]
    if len(queries) != 160:
        raise ValueError(f"fixed manifest must contain 160 queries, got {len(queries)}")
    videos = sorted({str(q["video"]) for q in queries})
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for dense feature extraction")
    import clip
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    map_dtype = torch.float16 if args.map_dtype == "float16" else torch.float32
    summary = {
        "format": "locatemot-l23-candidate-bank-v3-build-v1",
        "stage": "L23", "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest), "query_count": len(queries),
        "calibration_queries": sum(q["split"] == "calibration" for q in queries),
        "screening_queries": sum(q["split"] == "screening" for q in queries),
        "old_bank_root": str(old_root), "v2_bank_root": str(v2_root),
        "raw_root": str(raw_root), "device": str(device),
        "backbone": "frozen OpenAI CLIP ViT-B/32 projected visual patch tokens",
        "dense_map_shape": [512, 7, 7], "dense_map_stride_input_pixels": 32,
        "dense_input_policy": "full frame fixed resize 1238x374 -> 224x224",
        "normalization": "CLIP mean/std; projected patch tokens L2-normalized per token",
        "gt_used_for_features": False, "tracker_modified": False,
        "official_flexhook_reproduction": False,
        "structure_note": "FlexHook-style coordinate sampling/cross-modal-ready bank adaptation; ROPE-Swin unavailable/not used",
        "candidate_sampling": {"roi": "3x3 mean bilinear samples", "points": "center and four 20%-inset corners",
                                "context_1p5": "3x3 mean bilinear samples", "context_3": "3x3 mean bilinear samples",
                                "previous_roi": "causal previous same pool/track namespace ROI; zeros at lifecycle start"},
        "videos": videos, "banks": {}
    }
    try:
        for video in videos:
            started = time.time()
            old_path = old_root / "kitti" / f"{video}.pt"
            v2_path = v2_root / "kitti" / f"{video}.pt"
            old = torch.load(old_path, map_location="cpu", weights_only=False)
            v2 = torch.load(v2_path, map_location="cpu", weights_only=False)
            old_labels = json.loads(old_path.with_suffix(".labels.json").read_text())["candidate_gt"]
            v2_labels = json.loads(v2_path.with_suffix(".labels.json").read_text())["candidate_gt"]
            check_old_v2_alignment(old, v2, old_labels, v2_labels, video)
            t = v2["tensors"]
            frame_ids = t["frame_ids"].numpy().astype(np.int32)
            frame_ptr = t["frame_ptr"].numpy().astype(np.int64)
            boxes = t["box"].numpy().astype(np.float32)
            track_ids = t["track_id"].numpy().astype(np.int64)
            pool_ids = t["pool_id"].numpy().astype(np.int64)
            candidate_ids = t["candidate_index"].numpy().astype(np.int64)
            maps = {}
            map_meta = []
            for frame in frame_ids.tolist():
                image_path = raw_root / video / f"{int(frame):06d}.png"
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(image_path)
                h, w = image.shape[:2]
                fmap = dense_clip_map(model, image, device).cpu().to(map_dtype).contiguous()
                if not bool(torch.isfinite(fmap.float()).all()):
                    raise FloatingPointError(f"nonfinite dense map {video}/{frame}")
                maps[int(frame)] = fmap
                map_path = out_root / "dense_maps" / f"{video}_{int(frame):06d}.pt"
                torch.save({"video": video, "frame_id": int(frame), "feature_map": fmap,
                            "shape": list(fmap.shape), "stride_input_pixels": 32,
                            "normalization": summary["normalization"],
                            "raw_image": str(image_path), "raw_image_sha256": sha256_file(image_path)},
                           str(map_path) + ".tmp")
                os.replace(str(map_path) + ".tmp", map_path)
                map_meta.append({"frame_id": int(frame), "path": str(map_path), "raw_image": str(image_path),
                                 "raw_image_sha256": sha256_file(image_path), "shape": list(fmap.shape)})
            dense = {key: [] for key in VISUAL_KEYS}
            previous_roi = {}
            for fi, frame in enumerate(frame_ids.tolist()):
                begin, end = int(frame_ptr[fi]), int(frame_ptr[fi + 1])
                image_path = raw_root / video / f"{int(frame):06d}.png"
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                h, w = image.shape[:2]
                fmap = maps[int(frame)].float().to(device)
                current_rois = []
                for row in range(begin, end):
                    box = boxes[row]
                    roi = grid_sample_points(fmap, region_points(box, w, h, 1.0)).mean(axis=0)
                    points = grid_sample_points(fmap, fixed_point_set(box, w, h))
                    c15 = grid_sample_points(fmap, region_points(box, w, h, 1.5)).mean(axis=0)
                    c3 = grid_sample_points(fmap, region_points(box, w, h, 3.0)).mean(axis=0)
                    namespace = (int(pool_ids[row]), int(track_ids[row]))
                    previous = previous_roi.get(namespace, np.zeros(512, np.float32))
                    dense["dense_roi"].append(roi.astype(np.float32))
                    dense["dense_points"].append(points.astype(np.float32))
                    dense["dense_context_1p5"].append(c15.astype(np.float32))
                    dense["dense_context_3"].append(c3.astype(np.float32))
                    dense["dense_prev_roi"].append(previous.astype(np.float32))
                    current_rois.append((namespace, roi.astype(np.float32)))
                for namespace, roi in current_rois:
                    previous_roi[namespace] = roi
            tensors = {key: value.clone() for key, value in t.items()}
            for key in VISUAL_KEYS:
                value = np.asarray(dense[key], np.float16 if args.map_dtype == "float16" else np.float32)
                tensors[key] = torch.from_numpy(value)
            tensors["dense_v2_row_index"] = torch.arange(len(track_ids), dtype=torch.int64)
            tensors["dense_map_frame_index"] = torch.repeat_interleave(
                torch.arange(len(frame_ids), dtype=torch.int64),
                torch.diff(t["frame_ptr"]))
            for key, value in tensors.items():
                if torch.is_floating_point(value) and not bool(torch.isfinite(value.float()).all()):
                    raise FloatingPointError(f"nonfinite candidate field {video}/{key}")
            row_keys = [row_key(video, int(t["frame"][i]), int(pool_ids[i]), int(track_ids[i]), int(candidate_ids[i]))
                        for i in range(len(track_ids))]
            if len(set(row_keys)) != len(row_keys):
                raise AssertionError(f"duplicate v3 row keys for {video}")
            metadata = {**v2.get("metadata", {}),
                        "format": "locatemot-l23-candidate-bank-v3",
                        "stage": "L23", "old_bank_sha256": sha256_file(old_path),
                        "v2_bank_sha256": sha256_file(v2_path),
                        "v2_labels_sha256": sha256_file(v2_path.with_suffix(".labels.json")),
                        "manifest_sha256": summary["manifest_sha256"],
                        "raw_image_root": str(raw_root), "dense_backbone": summary["backbone"],
                        "dense_map_shape": [512, 7, 7], "dense_map_stride_input_pixels": 32,
                        "dense_map_dtype": args.map_dtype, "dense_map_files": map_meta,
                        "gt_used_for_features": False, "row_order_preserved": True,
                        "old_v2_row_alignment_verified": True, "causal_previous_track_feature": True,
                        "new_feature_dims": {key: list(tensors[key].shape[1:]) for key in VISUAL_KEYS}}
            bank = {"metadata": metadata, "tensors": tensors, "row_keys": row_keys}
            output = out_root / "kitti" / f"{video}.pt"
            torch.save(bank, str(output) + ".tmp")
            os.replace(str(output) + ".tmp", output)
            output.with_suffix(".labels.json").write_text(json.dumps({"candidate_gt": v2_labels}, separators=(",", ":")) + "\n")
            output.with_suffix(".audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
            output.with_suffix(".complete").write_text("ok\n")
            summary["banks"][video] = {"path": str(output), "rows": len(track_ids),
                                        "frames": len(frame_ids), "dense_map_files": len(map_meta),
                                        "old_bank_sha256": metadata["old_bank_sha256"],
                                        "v2_bank_sha256": metadata["v2_bank_sha256"],
                                        "elapsed_sec": time.time() - started}
            del old, v2, bank, tensors, maps
            if device.type == "cuda":
                torch.cuda.empty_cache()
        (out_root / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (out_root / "BUILD_COMPLETE").write_text("ok\n")
        print(json.dumps({"out_root": str(out_root), "videos": videos,
                          "banks": summary["banks"], "dense_map_shape": [512, 7, 7]}, indent=2))
    except Exception as exc:
        (out_root / "INCOMPLETE.md").write_text(
            f"# INCOMPLETE\n\nStage L23 v3 build stopped at first error: `{type(exc).__name__}: {exc}`\n")
        raise


if __name__ == "__main__":
    main()
