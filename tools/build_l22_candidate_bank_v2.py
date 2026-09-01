"""Build the independent Stage L22 fine-grained candidate bank.

The input bank, row order, frame pointers, labels, and old features are kept
unchanged.  New visual features are computed from raw KITTI images with the
already cached frozen CLIP ViT-B/32.  Motion/lifecycle/neighbour features use
only the old bank boxes and track namespace (no GT or expression labels).
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

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clip_box(box: np.ndarray, width: int, height: int, scale: float = 1.0) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1 + x2) * .5, (y1 + y2) * .5
    bw, bh = max(1.0, x2 - x1) * scale, max(1.0, y2 - y1) * scale
    return np.asarray([max(0., cx - bw * .5), max(0., cy - bh * .5),
                       min(float(width), cx + bw * .5),
                       min(float(height), cy + bh * .5)], np.float32)


def crop(image: np.ndarray, box: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    b = clip_box(box, width, height)
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    x1, y1 = max(0, min(width - 1, x1)), max(0, min(height - 1, y1))
    x2, y2 = max(x1 + 1, min(width, x2)), max(y1 + 1, min(height, y2))
    return image[y1:y2, x1:x2]


def preprocess(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 3 or arr.shape[0] < 2 or arr.shape[1] < 2:
        arr = np.zeros((2, 2, 3), np.uint8)
    # cv2 reads BGR; CLIP receives RGB.
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    h, w = arr.shape[:2]
    scale = 224.0 / min(h, w)
    nh, nw = max(224, int(round(h * scale))), max(224, int(round(w * scale)))
    resized = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    y, x = max(0, (nh - 224) // 2), max(0, (nw - 224) // 2)
    view = resized[y:y + 224, x:x + 224]
    if view.shape[:2] != (224, 224):
        view = cv2.resize(view, (224, 224), interpolation=cv2.INTER_CUBIC)
    return view.astype(np.float32)


@torch.inference_mode()
def encode(model, arrays: list[np.ndarray], device: torch.device,
           batch_size: int) -> np.ndarray:
    if not arrays:
        return np.zeros((0, 512), np.float16)
    output = np.zeros((len(arrays), 512), np.float16)
    mean = torch.as_tensor(MEAN, device=device)[None, :, None, None]
    std = torch.as_tensor(STD, device=device)[None, :, None, None]
    for begin in range(0, len(arrays), batch_size):
        views = np.stack([preprocess(x) for x in arrays[begin:begin + batch_size]])
        value = torch.from_numpy(views).permute(0, 3, 1, 2).to(device) / 255.0
        value = (value - mean) / std
        output[begin:begin + len(views)] = model.encode_image(value).float().cpu().numpy().astype(np.float16)
    return output


def pair_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    aa = max(0., float(a[2] - a[0])) * max(0., float(a[3] - a[1]))
    bb = max(0., float(b[2] - b[0])) * max(0., float(b[3] - b[1]))
    return inter / max(1e-8, aa + bb - inter)


def geometry_v2(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    w, h = max(1., float(width)), max(1., float(height))
    x1, y1, x2, y2 = boxes.T
    bw, bh = np.maximum(1., x2 - x1), np.maximum(1., y2 - y1)
    cx, cy = (x1 + x2) * .5, (y1 + y2) * .5
    return np.column_stack((x1 / w, y1 / h, x2 / w, y2 / h,
                            cx / w, cy / h, bw / w, bh / h,
                            (bw * bh) / (w * h),
                            np.clip(bw / bh, 0., 20.) / 20.)).astype(np.float32)


def neighbour_v2(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
    n = len(boxes)
    if not n:
        return np.zeros((0, 10), np.float32)
    w, h = max(1., float(width)), max(1., float(height))
    centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) * .5 / w,
                               (boxes[:, 1] + boxes[:, 3]) * .5 / h))
    delta = centers[:, None, :] - centers[None, :, :]
    distance = np.linalg.norm(delta, axis=-1)
    np.fill_diagonal(distance, np.inf)
    nearest = np.argmin(distance, axis=1) if n > 1 else np.zeros(n, np.int64)
    nd = centers[nearest] - centers if n > 1 else np.zeros((n, 2), np.float32)
    ndist = distance[np.arange(n), nearest] if n > 1 else np.ones(n, np.float32)
    niou = np.asarray([pair_iou(boxes[i], boxes[nearest[i]]) if n > 1 else 0.
                       for i in range(n)], np.float32)
    order_x = np.argsort(np.argsort(centers[:, 0], kind="stable"), kind="stable") / max(1, n - 1)
    order_y = np.argsort(np.argsort(centers[:, 1], kind="stable"), kind="stable") / max(1, n - 1)
    finite = np.where(np.isfinite(distance), distance, 0.)
    mean_distance = finite.sum(axis=1) / max(1, n - 1)
    k = min(3, n - 1)
    kth = np.partition(distance, k - 1, axis=1)[:, :k] if k else np.ones((n, 1), np.float32)
    mean_k = np.mean(kth, axis=1) if k else np.ones(n, np.float32)
    return np.column_stack((nd[:, 0], nd[:, 1], ndist, niou, order_x, order_y,
                            mean_distance, mean_k, np.full(n, np.log1p(n) / 6.),
                            np.clip(np.sum(np.isfinite(distance), axis=1) / 32., 0., 1.))).astype(np.float32)


def motion_v2(tensors: dict, frame_index: int, begin: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal adjacent-frame motion and lifecycle from bank track IDs only."""
    frame_ids = tensors["frame_ids"].numpy().astype(np.int32)
    frames = tensors["frame"].numpy().astype(np.int32)
    boxes = tensors["box"].numpy().astype(np.float32)
    track_ids = tensors["track_id"].numpy().astype(np.int64)
    image_w, image_h = 1238., 374.
    by_track: dict[int, list[int]] = {}
    for idx in range(len(track_ids)):
        by_track.setdefault(int(track_ids[idx]), []).append(idx)
    for idxs in by_track.values():
        idxs.sort(key=lambda i: (int(frames[i]), i))
    m = np.zeros((end - begin, 16), np.float32)
    life = np.zeros((end - begin, 6), np.float32)
    for local, idx in enumerate(range(begin, end)):
        history = by_track[int(track_ids[idx])]
        position = history.index(idx)
        previous = history[position - 1] if position else None
        previous2 = history[position - 2] if position > 1 else None
        current = boxes[idx]
        cw, ch = max(1., current[2] - current[0]), max(1., current[3] - current[1])
        if previous is None:
            velocity = np.zeros(2, np.float32); gap = 0.; prev_velocity = np.zeros(2, np.float32)
            old_w, old_h = cw, ch
        else:
            old = boxes[previous]; old_w, old_h = max(1., old[2] - old[0]), max(1., old[3] - old[1])
            cc = np.asarray([(current[0] + current[2]) * .5, (current[1] + current[3]) * .5])
            pc = np.asarray([(old[0] + old[2]) * .5, (old[1] + old[3]) * .5])
            gap = max(1., float(frames[idx] - frames[previous]))
            velocity = (cc - pc) / np.asarray([image_w, image_h]) / gap
            if previous2 is not None:
                old2 = boxes[previous2]
                c2 = np.asarray([(old2[0] + old2[2]) * .5, (old2[1] + old2[3]) * .5])
                gap2 = max(1., float(frames[previous] - frames[previous2]))
                prev_velocity = (pc - c2) / np.asarray([image_w, image_h]) / gap2
            else:
                prev_velocity = np.zeros(2, np.float32)
        speed = float(np.linalg.norm(velocity)); acceleration = velocity - prev_velocity
        size_delta = np.asarray([np.log(cw / old_w), np.log(ch / old_h)], np.float32)
        direction = velocity / max(speed, 1e-6)
        age = max(0., float(frames[idx] - frames[history[0]]))
        m[local] = np.asarray([velocity[0], velocity[1], speed, acceleration[0], acceleration[1],
                                float(np.linalg.norm(acceleration)), size_delta[0], size_delta[1],
                                float(np.linalg.norm(size_delta)), direction[0], direction[1],
                                float(previous is not None), float(gap / 10.),
                                min(age, 300.) / 300., min(len(history), 100) / 100.,
                                float(position == len(history) - 1)], np.float32)
        life[local] = np.asarray([float(previous is not None), min(gap, 10.) / 10.,
                                  min(age, 300.) / 300., min(len(history), 100) / 100.,
                                  float(position == 0), float(position == len(history) - 1)], np.float32)
    return m, life


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/l19/protocol/kitti_fast_eval_manifest.json")
    ap.add_argument("--old-bank-root", default="outputs/l19/dual_banks_features")
    ap.add_argument("--raw-root", default="data/kitti_tracking_training/image_02")
    ap.add_argument("--out-root", default="outputs/l22/candidate_bank_v2")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()
    manifest = Path(args.manifest); old_root = Path(args.old_bank_root)
    raw_root = Path(args.raw_root); out_root = Path(args.out_root)
    if not manifest.is_absolute(): manifest = ROOT / manifest
    if not old_root.is_absolute(): old_root = ROOT / old_root
    if not raw_root.is_absolute(): raw_root = ROOT / raw_root
    if not out_root.is_absolute(): out_root = ROOT / out_root
    if out_root.exists():
        raise FileExistsError(f"refusing to overwrite v2 bank root: {out_root}")
    out_root.mkdir(parents=True, exist_ok=False)
    queries = json.loads(manifest.read_text())["queries"]
    videos = sorted({str(q["video"]) for q in queries})
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen CLIP feature extraction")
    import clip
    model, _preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    summary = {"format": "locatemot-l22-candidate-bank-v2-build-v1",
               "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
               "old_bank_root": str(old_root), "raw_root": str(raw_root),
               "device": str(device), "backbone": "frozen OpenAI CLIP ViT-B/32",
               "gt_used_for_features": False, "videos": videos, "banks": {}}
    try:
        for video in videos:
            old_path = old_root / "kitti" / f"{video}.pt"
            old = torch.load(old_path, map_location="cpu", weights_only=False)
            old_t = old["tensors"]
            labels_path = old_path.with_suffix(".labels.json")
            labels = json.loads(labels_path.read_text())["candidate_gt"]
            n = len(old_t["track_id"]); frame_ids = old_t["frame_ids"].numpy().astype(np.int32)
            if len(labels) != n or int(old_t["frame_ptr"][-1]) != n:
                raise ValueError(f"invalid old bank lengths for {video}")
            new_fields = {key: [] for key in ("crop_tight", "crop_context_1p5", "crop_local_context",
                                               "crop_full_context", "geometry_v2", "neighbor_v2",
                                               "motion_v2", "lifecycle_v2")}
            row_keys = []
            frame_ptr = old_t["frame_ptr"].numpy().astype(np.int64)
            start_time = time.time()
            for fi, frame in enumerate(frame_ids.tolist()):
                begin, end = int(frame_ptr[fi]), int(frame_ptr[fi + 1])
                image_path = raw_root / video / f"{frame:06d}.png"
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise FileNotFoundError(image_path)
                height, width = image.shape[:2]
                boxes = old_t["box"][begin:end].numpy().astype(np.float32)
                tight = [crop(image, box) for box in boxes]
                context = [crop(image, clip_box(box, width, height, 1.5)) for box in boxes]
                local = [crop(image, clip_box(box, width, height, 3.0)) for box in boxes]
                full = [image]
                new_fields["crop_tight"].append(encode(model, tight, device, args.batch_size))
                new_fields["crop_context_1p5"].append(encode(model, context, device, args.batch_size))
                new_fields["crop_local_context"].append(encode(model, local, device, args.batch_size))
                full_feature = encode(model, full, device, args.batch_size)
                new_fields["crop_full_context"].append(np.repeat(full_feature, end - begin, axis=0))
                new_fields["geometry_v2"].append(geometry_v2(boxes, width, height))
                new_fields["neighbor_v2"].append(neighbour_v2(boxes, width, height))
                motion, lifecycle = motion_v2(old_t, fi, begin, end)
                new_fields["motion_v2"].append(motion); new_fields["lifecycle_v2"].append(lifecycle)
                for local_index in range(end - begin):
                    row = begin + local_index
                    row_keys.append(f"{video}:{frame}:{int(old_t['pool_id'][row])}:{int(old_t['track_id'][row])}:{int(old_t['candidate_index'][row])}")
            tensors = {key: value.clone() for key, value in old_t.items()}
            for key, pieces in new_fields.items():
                value = np.concatenate(pieces, axis=0).astype(np.float16 if value_is_visual(key) else np.float32)
                tensors[key] = torch.from_numpy(value)
            if not torch.equal(tensors["frame_ptr"], old_t["frame_ptr"]) or not torch.equal(tensors["frame_ids"], old_t["frame_ids"]):
                raise AssertionError(f"frame structure changed for {video}")
            for key, value in tensors.items():
                if torch.is_floating_point(value) and not bool(torch.isfinite(value.float()).all()):
                    raise FloatingPointError(f"nonfinite {video}/{key}")
            bank = {"metadata": {**old.get("metadata", {}),
                                  "format": "locatemot-l22-candidate-bank-v2",
                                  "stage": "L22", "old_bank_sha256": sha256_file(old_path),
                                  "old_labels_sha256": sha256_file(labels_path),
                                  "manifest_sha256": summary["manifest_sha256"],
                                  "raw_image_root": str(raw_root),
                                  "raw_image_size_observed": [1238, 374],
                                  "visual_backbone": "frozen OpenAI CLIP ViT-B/32",
                                  "gt_used_for_features": False,
                                  "new_feature_dims": {k: list(v.shape[1:]) for k, v in tensors.items() if k.endswith("_v2") or k.startswith("crop_")},
                                  "motion_is_causal_track_namespace_only": True,
                                  "row_order_preserved": True},
                    "tensors": tensors, "row_keys": row_keys}
            if len(row_keys) != n or len(set(row_keys)) != n:
                raise AssertionError(f"row key mismatch/duplicate for {video}")
            output = out_root / "kitti" / f"{video}.pt"
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(bank, output.with_suffix(".pt.tmp")); os.replace(output.with_suffix(".pt.tmp"), output)
            output.with_suffix(".labels.json").write_text(json.dumps({"candidate_gt": labels}, separators=(",", ":")) + "\n")
            output.with_suffix(".audit.json").write_text(json.dumps(bank["metadata"], indent=2) + "\n")
            output.with_suffix(".complete").write_text("ok\n")
            summary["banks"][video] = {"path": str(output), "rows": n, "frames": len(frame_ids),
                                        "old_bank_sha256": bank["metadata"]["old_bank_sha256"],
                                        "elapsed_sec": time.time() - start_time}
            del old, bank, tensors
            if device.type == "cuda": torch.cuda.empty_cache()
        (out_root / "build_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    except Exception as exc:
        (out_root / "INCOMPLETE.md").write_text(f"# INCOMPLETE\n\nBank v2 build stopped at first error: `{type(exc).__name__}: {exc}`\n")
        raise


def value_is_visual(key: str) -> bool:
    return key.startswith("crop_")


if __name__ == "__main__":
    main()
