"""Build a compact L18 main/reserve track bank.

The main bank is copied from the frozen, deduplicated L16 bank.  The reserve
uses query-independent GroundingDINO proposals with a fixed top-K budget and a
small causal IoU/CLIP tracklet linker in a disjoint ID namespace.  This tool
does not alter L11/L16 banks or ordinary MOT/OVMOT artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from locatemot.rmot.track_bank import _context, _geometry  # noqa: E402


KITTI_IMAGE_ROOT = ROOT / "data/kitti_tracking_training/image_02"
MAIN_ROOT = ROOT / "outputs/l16/track_banks_dedup/kitti"
TRAINVAL_DINO = ROOT / "outputs/l18/cache/dino_kitti_trainval.pkl"
OFFICIAL_DINO = ROOT / "outputs/l13/cache/dino_kitti_eval.pkl"
PROTOCOL = ROOT / "outputs/l16/data/protocol/split_manifest.json"

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def load_record(seq: str) -> dict:
    path = ROOT / "outputs/l11/data/rmot_kitti" / f"{seq}.pkl"
    if not path.exists():
        path = ROOT / "outputs/l16/data/kitti_missing/records" / f"{seq}.pkl"
    return pickle.load(path.open("rb"))


def iou(a, b) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(1e-6, aa + bb - inter)


def match_gt(boxes: np.ndarray, gt: dict, threshold: float = 0.5) -> list:
    pairs = []
    for index, box in enumerate(boxes):
        for gid, target in gt.items():
            value = iou(box, target)
            if value >= threshold:
                pairs.append((value, index, str(gid)))
    pairs.sort(reverse=True)
    used_boxes, used_gt = set(), set()
    labels = [None] * len(boxes)
    for _value, index, gid in pairs:
        if index in used_boxes or gid in used_gt:
            continue
        used_boxes.add(index)
        used_gt.add(gid)
        labels[index] = gid
    return labels


def encode_clip(model, crops: list[np.ndarray], device: torch.device) -> np.ndarray:
    if not crops:
        return np.zeros((0, 512), np.float16)
    outputs = np.zeros((len(crops), 512), np.float16)
    mean = torch.as_tensor(CLIP_MEAN, device=device)[None, :, None, None]
    std = torch.as_tensor(CLIP_STD, device=device)[None, :, None, None]
    for start in range(0, len(crops), 128):
        chunk = []
        for crop in crops[start:start + 128]:
            h, w = crop.shape[:2]
            if h < 2 or w < 2:
                crop = np.zeros((2, 2, 3), np.uint8)
                h = w = 2
            scale = 224.0 / min(h, w)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_CUBIC)
            y, x = max(0, (nh - 224) // 2), max(0, (nw - 224) // 2)
            view = resized[y:y + 224, x:x + 224]
            if view.shape[:2] != (224, 224):
                view = cv2.resize(view, (224, 224), interpolation=cv2.INTER_CUBIC)
            chunk.append(view.astype(np.float32))
        tensor = torch.from_numpy(np.stack(chunk)).permute(0, 3, 1, 2)
        tensor = (tensor.to(device) / 255.0 - mean) / std
        with torch.no_grad():
            value = model.encode_image(tensor).float().cpu().numpy()
        outputs[start:start + len(chunk)] = value.astype(np.float16)
    return outputs


def reserve_track_ids(boxes_by_frame: list[np.ndarray],
                      clips_by_frame: list[np.ndarray],
                      frame_ids: list[int]) -> list[np.ndarray]:
    """Causal, conservative reserve tracklets; no future frame is read."""
    active = []
    next_id = 1
    result = []
    for boxes, clips, frame in zip(boxes_by_frame, clips_by_frame, frame_ids):
        assignments = np.full(len(boxes), -1, np.int64)
        costs = []
        for current, box in enumerate(boxes):
            for old, item in enumerate(active):
                if int(frame) - int(item["frame"]) > 2:
                    continue
                overlap = iou(box, item["box"])
                appearance = float(np.dot(
                    clips[current] / max(1e-6, np.linalg.norm(clips[current])),
                    item["clip"] / max(1e-6, np.linalg.norm(item["clip"]))))
                if overlap >= 0.12 or appearance >= 0.78:
                    costs.append((current, old, 0.72 * overlap + 0.28 * appearance))
        used_current, used_old = set(), set()
        for current, old, _score in sorted(costs, key=lambda x: x[2], reverse=True):
            if current in used_current or old in used_old:
                continue
            assignments[current] = int(active[old]["id"])
            used_current.add(current)
            used_old.add(old)
        for current in range(len(boxes)):
            if assignments[current] < 0:
                assignments[current] = next_id
                next_id += 1
        active = [{"id": int(assignments[i]), "box": boxes[i].copy(),
                   "clip": clips[i].copy(), "frame": int(frame)}
                  for i in range(len(boxes))]
        result.append(assignments)
    return result


def reserve_numeric(boxes_by_frame: list[np.ndarray], clips_by_frame: list[np.ndarray],
                    track_ids_by_frame: list[np.ndarray], frame_ids: list[int],
                    image_size: list[int]) -> dict[str, list[np.ndarray]]:
    previous = {}
    stats = {}
    out = {key: [] for key in ("history_clip", "geometry", "motion", "context",
                               "lifecycle")}
    for boxes, clips, ids, frame in zip(
            boxes_by_frame, clips_by_frame, track_ids_by_frame, frame_ids):
        geometry = _geometry(boxes, image_size)
        context = _context(boxes, image_size)
        motion = np.zeros((len(boxes), 8), np.float32)
        lifecycle = np.zeros((len(boxes), 8), np.float32)
        history = np.zeros_like(clips, np.float16)
        for index, track_id in enumerate(ids.tolist()):
            old = previous.get(int(track_id))
            if old is None:
                ema, old_box, age, hits = clips[index].astype(np.float32), \
                    boxes[index], 1, 1
                delta = np.zeros(2, np.float32)
            else:
                ema = 0.8 * old["ema"] + 0.2 * clips[index]
                old_box = old["box"]
                age, hits = old["age"] + 1, old["hits"] + 1
                old_center = np.asarray([(old_box[0] + old_box[2]) * 0.5,
                                         (old_box[1] + old_box[3]) * 0.5])
                new_center = np.asarray([(boxes[index][0] + boxes[index][2]) * 0.5,
                                         (boxes[index][1] + boxes[index][3]) * 0.5])
                delta = (new_center - old_center) / np.asarray(
                    [max(1.0, image_size[0]), max(1.0, image_size[1])])
            history[index] = ema.astype(np.float16)
            width = max(1.0, float(boxes[index][2] - boxes[index][0]))
            height = max(1.0, float(boxes[index][3] - boxes[index][1]))
            old_width = max(1.0, float(old_box[2] - old_box[0]))
            old_height = max(1.0, float(old_box[3] - old_box[1]))
            motion[index] = np.asarray([
                delta[0], delta[1], np.log(width / old_width),
                np.log(height / old_height), float(np.linalg.norm(delta)),
                min(age, 300) / 300.0, hits / max(1, age), 0.0,
            ], np.float32)
            score = float(stats.get((int(frame), index), 0.0))
            lifecycle[index] = np.asarray([
                hits, age, 0.0, 1.0, 1.0, score,
                age, hits,
            ], np.float32)
            previous[int(track_id)] = {"ema": ema, "box": boxes[index].copy(),
                                       "age": age, "hits": hits}
        out["history_clip"].append(history)
        out["geometry"].append(geometry.astype(np.float32))
        out["motion"].append(motion)
        out["context"].append(context.astype(np.float32))
        out["lifecycle"].append(lifecycle)
    return out


def split_for(seq: str) -> str:
    manifest = json.loads(PROTOCOL.read_text())
    for split in ("train", "train_val", "official_eval"):
        if seq in manifest["kitti_v2"][split]:
            return split
    raise KeyError(seq)


def source_cache(seq: str, split: str, trainval: dict, official: dict) -> dict:
    return (official if split == "official_eval" else trainval).get(seq, {})


def build_one(seq: str, budget: int, trainval: dict, official: dict,
              clip_model, device: torch.device, out_root: Path) -> dict:
    split = split_for(seq)
    source = source_cache(seq, split, trainval, official)
    if not source:
        raise FileNotFoundError(f"no DINO cache for {seq} ({split})")
    main_path = MAIN_ROOT / f"{seq}.pt"
    main_bank = torch.load(main_path, map_location="cpu", weights_only=False)
    main_tensors = main_bank["tensors"]
    main_labels_path = main_path.with_suffix(".labels.json")
    main_labels = None
    if main_labels_path.exists():
        main_labels = json.loads(main_labels_path.read_text())["candidate_gt"]
    record = load_record(seq)
    gt_by_frame = {
        int(fr["frame"]): {str(k): v for k, v in fr.get("gt_boxes", {}).items()}
        for fr in record["frames"]
    }
    frame_ids = [int(value) for value in main_tensors["frame_ids"].tolist()]
    image_size = list(main_bank["metadata"]["image_size"])
    reserve_boxes, reserve_scores, reserve_clips, reserve_labels = [], [], [], []
    cross_duplicate_50 = 0
    main_rows = []
    for frame_index, frame in enumerate(frame_ids):
        start = int(main_tensors["frame_ptr"][frame_index])
        end = int(main_tensors["frame_ptr"][frame_index + 1])
        main_box = main_tensors["box"][start:end].numpy().astype(np.float32)
        main_rows.append((start, end, main_box))
        entry = source.get(frame, {})
        all_boxes = np.asarray(entry.get("boxes", []), np.float32).reshape(-1, 4)
        scores = np.asarray(entry.get("scores", []), np.float32).reshape(-1)
        order = np.argsort(-scores, kind="stable")[:budget]
        boxes = all_boxes[order] if len(all_boxes) else np.zeros((0, 4), np.float32)
        scores = scores[order] if len(scores) else np.zeros(0, np.float32)
        # Exact duplicates inside the reserve are removed; cross-pool overlaps
        # remain as separate identities so the repair/selector can audit them.
        keep, seen = [], set()
        for index, box in enumerate(boxes):
            key = box.tobytes()
            if key in seen:
                continue
            seen.add(key)
            keep.append(index)
        boxes = boxes[np.asarray(keep, np.int64)] if keep else np.zeros((0, 4), np.float32)
        scores = scores[np.asarray(keep, np.int64)] if keep else np.zeros(0, np.float32)
        cross_duplicate_50 += sum(any(iou(box, x) >= 0.50 for x in main_box)
                                  for box in boxes)
        image_path = KITTI_IMAGE_ROOT / seq / f"{frame:06d}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read {image_path}")
        crops = []
        height, width = image.shape[:2]
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            crops.append(image[y1:y2, x1:x2] if x2 > x1 and y2 > y1
                         else np.zeros((2, 2, 3), np.uint8))
        clips = encode_clip(clip_model, crops, device)
        reserve_boxes.append(boxes)
        reserve_scores.append(scores)
        reserve_clips.append(clips)
        reserve_labels.append(match_gt(boxes, gt_by_frame.get(frame, {})))
    reserve_ids = reserve_track_ids(reserve_boxes, reserve_clips, frame_ids)
    numeric = reserve_numeric(reserve_boxes, reserve_clips, reserve_ids,
                              frame_ids, image_size)
    arrays = {key: [] for key in (
        "frame", "candidate_index", "track_id", "box", "objectness", "clip",
        "history_clip", "pbd", "uidm_h", "uidm_ref_pbd", "uidm_anchor_pbd",
        "geometry", "motion", "context", "lifecycle", "pool_id", "source_score")}
    labels = []
    for frame_index, frame in enumerate(frame_ids):
        start, end, _main_box = main_rows[frame_index]
        for name in ("frame", "candidate_index", "track_id", "box", "objectness",
                     "clip", "history_clip", "pbd", "uidm_h", "uidm_ref_pbd",
                     "uidm_anchor_pbd", "geometry", "motion", "context",
                     "lifecycle"):
            value = main_tensors[name][start:end].numpy()
            arrays[name].append(value)
        count = end - start
        arrays["pool_id"].append(np.zeros(count, np.int64))
        arrays["source_score"].append(
            main_tensors["objectness"][start:end].numpy().astype(np.float32))
        if main_labels is not None:
            labels.extend(main_labels[start:end])
        boxes = reserve_boxes[frame_index]
        clips = reserve_clips[frame_index]
        count = len(boxes)
        arrays["frame"].append(np.full(count, frame, np.int32))
        arrays["candidate_index"].append(np.arange(count, dtype=np.int32))
        arrays["track_id"].append(reserve_ids[frame_index].astype(np.int32) + 1_000_000)
        arrays["box"].append(boxes.astype(np.float32))
        arrays["objectness"].append(reserve_scores[frame_index].astype(np.float32))
        arrays["clip"].append(clips.astype(np.float16))
        arrays["history_clip"].append(numeric["history_clip"][frame_index])
        arrays["pbd"].append(np.zeros((count, 2048), np.float16))
        arrays["uidm_h"].append(np.zeros((count, 384), np.float16))
        arrays["uidm_ref_pbd"].append(np.zeros((count, 2048), np.float16))
        arrays["uidm_anchor_pbd"].append(np.zeros((count, 2048), np.float16))
        arrays["geometry"].append(numeric["geometry"][frame_index])
        arrays["motion"].append(numeric["motion"][frame_index])
        arrays["context"].append(numeric["context"][frame_index])
        arrays["lifecycle"].append(numeric["lifecycle"][frame_index])
        arrays["pool_id"].append(np.ones(count, np.int64))
        arrays["source_score"].append(reserve_scores[frame_index].astype(np.float32))
        if main_labels is not None:
            labels.extend(reserve_labels[frame_index])
    tensors = {}
    integer_names = {"frame", "candidate_index", "track_id", "pool_id"}
    tails = {"frame": (), "candidate_index": (), "track_id": (), "box": (4,),
             "objectness": (), "clip": (512,), "history_clip": (512,),
             "pbd": (2048,), "uidm_h": (384,), "uidm_ref_pbd": (2048,),
             "uidm_anchor_pbd": (2048,), "geometry": (7,), "motion": (8,),
             "context": (8,), "lifecycle": (8,), "pool_id": (),
             "source_score": ()}
    frame_ptr = [0]
    for frame_index in range(len(frame_ids)):
        frame_count = len(arrays["frame"][2 * frame_index]) + \
            len(arrays["frame"][2 * frame_index + 1])
        frame_ptr.append(frame_ptr[-1] + frame_count)
    for name, values in arrays.items():
        if values:
            value = np.concatenate(values, axis=0)
        else:
            value = np.zeros((0,) + tails[name],
                             np.int64 if name in integer_names else np.float32)
        if name in integer_names:
            value = value.astype(np.int64 if name == "pool_id" else np.int32)
        tensors[name] = torch.from_numpy(value)
    tensors["frame_ptr"] = torch.as_tensor(frame_ptr, dtype=torch.int64)
    tensors["frame_ids"] = torch.as_tensor(frame_ids, dtype=torch.int32)
    metadata = {
        "format": "locatemot-l18-dual-track-bank-v1",
        "dataset": "kitti", "video_id": seq, "split": split,
        "image_size": image_size, "main_source": str(main_path),
        "main_shared_checkpoint_sha256": main_bank["metadata"].get(
            "shared_checkpoint_sha256"),
        "reserve_source": "GroundingDINO Swin-T query-independent road-user prompt",
        "reserve_cache_sha256": hashlib.sha256(
            (OFFICIAL_DINO if split == "official_eval" else TRAINVAL_DINO).read_bytes()
        ).hexdigest(),
        "reserve_budget": budget, "reserve_tracker": "causal IoU/CLIP greedy linker",
        "reserve_id_offset": 1_000_000,
        "main_observations": int(sum(end - start for start, end, _ in main_rows)),
        "reserve_observations": int(sum(len(x) for x in reserve_boxes)),
        "observations": int(len(tensors["track_id"])),
        "cross_pool_iou50_duplicates_retained": int(cross_duplicate_50),
        "causal": True, "query_independent": True,
        "rmot_only_reserve_namespace": True,
    }
    bank = {"metadata": metadata, "tensors": tensors}
    out = out_root / f"{seq}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, out.with_suffix(".pt.tmp"))
    os.replace(out.with_suffix(".pt.tmp"), out)
    out.with_suffix(".audit.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if main_labels is not None:
        out.with_suffix(".labels.json").write_text(
            json.dumps({"candidate_gt": labels}) + "\n")
    out.with_suffix(".complete").write_text("ok\n")
    del main_bank, record
    torch.cuda.empty_cache()
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--out", default="outputs/l18/dual_banks/kitti")
    parser.add_argument("--videos", nargs="*", default=None)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("L18 bank build requires CUDA for CLIP crop features")
    trainval = pickle.load(TRAINVAL_DINO.open("rb"))
    official = pickle.load(OFFICIAL_DINO.open("rb"))
    manifest = json.loads(PROTOCOL.read_text())
    videos = sorted(set(manifest["kitti_v2"]["train"] +
                        manifest["kitti_v2"]["train_val"] +
                        manifest["kitti_v2"]["official_eval"]))
    if args.videos:
        videos = [value for value in videos if value in set(args.videos)]
    import clip
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    out_root = (ROOT / args.out).resolve()
    rows = []
    for index, seq in enumerate(videos):
        out = out_root / f"{seq}.pt"
        if out.with_suffix(".complete").exists():
            existing = torch.load(out, map_location="cpu", weights_only=False)
            rows.append(existing["metadata"])
            print(f"[l18-bank] skip {seq}", flush=True)
            continue
        metadata = build_one(seq, args.budget, trainval, official,
                             clip_model, device, out_root)
        rows.append(metadata)
        print(f"[l18-bank] {seq} {index + 1}/{len(videos)} "
              f"main={metadata['main_observations']} "
              f"reserve={metadata['reserve_observations']}", flush=True)
    (out_root / "manifest.json").write_text(json.dumps({
        "format": "locatemot-l18-dual-bank-manifest-v1",
        "budget": args.budget, "videos": rows,
        "trainval_dino": str(TRAINVAL_DINO), "official_dino": str(OFFICIAL_DINO),
    }, indent=2) + "\n")
    print(f"[l18-bank] done videos={len(rows)} out={out_root}", flush=True)


if __name__ == "__main__":
    main()
