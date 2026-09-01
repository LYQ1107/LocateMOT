"""Small, auditable data helpers for the L40 streaming raw-image probe.

This module deliberately keeps image tensors transient.  It only keeps frozen
embeddings for the fragments requested by one smoke/evaluation in RAM; no
image or dense feature cache is written.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
BANK_ROOT = ROOT / "outputs/l19/dual_banks_features/kitti"
CACHE_ROOT = ROOT / "outputs/l28/track_sequence_bank_final"
RAW_ROOT = ROOT / "data/kitti_tracking_training/image_02"
WEIGHTS = Path("/home/lwr/.cache/clip/ViT-B-16.pt")
FIT_VIDEOS = ("0000", "0001", "0002", "0003", "0006", "0007", "0008", "0009", "0010", "0012", "0014", "0020")
HELDOUT_VIDEOS = ("0015", "0016", "0017")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gt_set(value) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(x) for x in value}
    return {str(value)}


def image_path(video: str, frame: int) -> Path:
    return RAW_ROOT / str(video) / f"{int(frame):06d}.png"


def crop_box(box, width: int, height: int, padding: float = 0.10):
    x1, y1, x2, y2 = [float(x) for x in box]
    bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
    x1 -= padding * bw; y1 -= padding * bh
    x2 += padding * bw; y2 += padding * bh
    return (max(0, int(np.floor(x1))), max(0, int(np.floor(y1))),
            min(width, int(np.ceil(x2))), min(height, int(np.ceil(y2))))


def _bank_rows_by_track(tensors):
    rows = defaultdict(list)
    for row, track in enumerate(tensors["track_id"].tolist()):
        rows[int(track)].append(row)
    for track in rows:
        rows[track].sort(key=lambda r: (int(tensors["frame"][r]), r))
    return rows


def load_fragments(videos, max_history: int = 8, require_alignment: bool = True, include_frozen_features: bool = False):
    """Load metadata and frozen numeric observations, never image pixels."""
    fragments = []
    alignment = []
    for video in videos:
        bank_path = BANK_ROOT / f"{video}.pt"
        cache_path = CACHE_ROOT / f"{video}.pt"
        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        bt = bank["tensors"]
        rows_by_track = _bank_rows_by_track(bt)
        ptr = cache["track_ptr"].tolist()
        cframes = cache["obs_frame"].tolist()
        labels = cache["obs_gt_ids"]
        required = ("geometry", "motion", "lifecycle", "objectness", "box", "frame", "pool_id")
        missing = [x for x in required if x not in bt]
        if missing:
            raise KeyError(f"{video}: missing frozen-bank fields {missing}")
        for ti, track in enumerate(cache["track_ids"].tolist()):
            begin, end = int(ptr[ti]), int(ptr[ti + 1])
            brows = rows_by_track.get(int(track), [])
            bframes = [int(bt["frame"][r]) for r in brows]
            if bframes != [int(x) for x in cframes[begin:end]]:
                msg = {"video": video, "track_id": int(track), "cache_frames": cframes[begin:end], "bank_frames": bframes}
                if require_alignment:
                    raise AssertionError(f"L40 bank/cache frame alignment failed: {msg}")
                continue
            if len(brows) < 1:
                continue
            ids = list(range(max(begin, end - max_history), end))
            # cache and bank rows are aligned by position in this track.
            chosen_bank = brows[-len(ids):]
            obs = []
            for ci, br in zip(ids, chosen_bank):
                numeric = torch.cat((bt["geometry"][br].float(), bt["motion"][br].float(),
                                     bt["lifecycle"][br].float(), bt["objectness"][br].float().reshape(1)))
                obs.append({"row": int(br), "frame": int(bt["frame"][br]), "box": bt["box"][br].float().tolist(),
                            "numeric": numeric, "source": int(bt["pool_id"][br]),
                            "gt": gt_set(labels[ci]), "history": cache["obs_features"][ci, 512:1024].float(),
                            "image": str(image_path(video, int(bt["frame"][br])))})
                if include_frozen_features:
                    obs[-1]["frozen_features"] = cache["obs_features"][ci].float()
            gids = set().union(*(x["gt"] for x in obs))
            fragments.append({"video": str(video), "track_id": int(track), "obs": obs,
                              "frames": {x["frame"] for x in obs}, "gids": gids,
                              "source": int(round(float(np.mean([x["source"] for x in obs])))),
                              "labelled": bool(gids)})
            alignment.append({"video": str(video), "track_id": int(track), "cache_rows": len(ids), "bank_rows": len(chosen_bank)})
    return fragments, alignment


def make_pairs(fragments, max_positive_per_gt: int = 64, max_hard_per_fragment: int = 8,
               max_inactive_per_fragment: int = 4):
    by_gt = defaultdict(list); by_frame = defaultdict(list)
    for i, f in enumerate(fragments):
        for g in f["gids"]:
            by_gt[(f["video"], g)].append(i)
        for frame in f["frames"]:
            by_frame[(f["video"], frame)].append(i)
    pairs, seen = [], set()
    for key, ids in sorted(by_gt.items()):
        count = 0
        ids = sorted(set(ids))
        for ai, a in enumerate(ids):
            for b in ids[ai + 1:]:
                if fragments[a]["track_id"] == fragments[b]["track_id"]:
                    continue
                if fragments[a]["gids"] & fragments[b]["gids"]:
                    k = (a, b, 1, "same_gt_fragment")
                    if k not in seen:
                        pairs.append({"a": a, "b": b, "label": 1, "kind": k[3],
                                      "frame": min(fragments[a]["frames"] | fragments[b]["frames"])})
                        seen.add(k); count += 1
                if count >= max_positive_per_gt:
                    break
            if count >= max_positive_per_gt:
                break
    for a, fa in enumerate(fragments):
        hard, inactive = [], []
        for frame in sorted(fa["frames"]):
            for b in by_frame[(fa["video"], frame)]:
                if a == b or fa["gids"] & fragments[b]["gids"]:
                    continue
                (hard if fragments[b]["labelled"] else inactive).append((b, frame))
        for b, frame in sorted(set(hard))[:max_hard_per_fragment]:
            k = (a, b, 0, "same_frame_different_gt_hard")
            if k not in seen:
                pairs.append({"a": a, "b": b, "label": 0, "kind": k[3], "frame": int(frame)}); seen.add(k)
        for b, frame in sorted(set(inactive))[:max_inactive_per_fragment]:
            k = (a, b, 0, "inactive")
            if k not in seen:
                pairs.append({"a": a, "b": b, "label": 0, "kind": k[3], "frame": int(frame)}); seen.add(k)
    if not any(x["label"] for x in pairs) or not any(not x["label"] for x in pairs):
        raise RuntimeError("L40 pair construction has no positive or negative samples")
    return pairs


def history_cosine(fa, fb):
    # L28 history CLIP is frozen diagnostic only, not an L40 input.
    def mean_hist(f):
        # The helper caller attaches frozen history vectors when needed.
        return f["history"].mean(0)
    x, y = mean_hist(fa), mean_hist(fb)
    return float(torch.dot(x, y) / x.norm().clamp_min(1e-6) / y.norm().clamp_min(1e-6))


class StreamingClipEncoder:
    """Frozen CLIP encoder; pixels are batched and immediately discarded."""
    def __init__(self, device="cuda:0", weights=WEIGHTS, batch_size=32):
        import clip
        self.device = torch.device(device)
        self.model, self.preprocess = clip.load(str(weights), device=self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.batch_size = int(batch_size)

    @torch.inference_mode()
    def encode(self, fragments, ids):
        result = []
        for frag_id in ids:
            f = fragments[int(frag_id)]
            values = []
            for ob in f["obs"]:
                path = Path(ob["image"])
                with Image.open(path) as image:
                    image = image.convert("RGB")
                    box = crop_box(ob["box"], image.width, image.height)
                    if box[2] <= box[0] or box[3] <= box[1]:
                        raise ValueError(f"empty L40 crop {path} box={ob['box']}")
                    values.append(self.preprocess(image.crop(box)))
            chunks = []
            for start in range(0, len(values), self.batch_size):
                pixel = torch.stack(values[start:start + self.batch_size]).to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                    z = self.model.encode_image(pixel).float().cpu()
                chunks.append(z)
                del pixel, z
            result.append(torch.cat(chunks, 0))
        return result
