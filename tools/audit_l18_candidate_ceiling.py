"""Fixed-budget, GT-only candidate ceiling audit for Stage L18.

This tool is an audit: it never writes a training bank and never uses official
evaluation annotations to select the reserve budget.  It measures candidate
coverage and duplicate/false-positive load for the frozen L16 main pool and a
query-independent GroundingDINO reserve cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))


def iou(a, b):
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(1e-6, aa + bb - inter)


def hits(boxes, gt, threshold):
    return {
        str(gid) for gid, target in gt.items()
        if any(iou(box, target) >= threshold for box in boxes)
    }


def load_record(seq: str) -> dict:
    path = ROOT / "outputs/l11/data/rmot_kitti" / f"{seq}.pkl"
    if not path.exists():
        path = ROOT / "outputs/l16/data/kitti_missing/records" / f"{seq}.pkl"
    return pickle.load(path.open("rb"))


def main_boxes(record: dict) -> dict[int, np.ndarray]:
    """Recover the accepted bank geometry without retaining torch storages.

    L16's dedup operation is exact-box deduplication and is independent of GT;
    applying the same operation to the source record gives the same geometry
    for this ceiling audit while keeping memory bounded.
    """
    out = {}
    for raw in record["frames"]:
        boxes = np.asarray(raw.get("boxes", []), np.float32).reshape(-1, 4)
        seen = set()
        keep = []
        for index, box in enumerate(boxes):
            key = box.tobytes()
            if key not in seen:
                seen.add(key)
                keep.append(index)
        out[int(raw["frame"])] = boxes[np.asarray(keep, np.int64)] \
            if keep else np.zeros((0, 4), np.float32)
    return out


def split_sequences(split: str) -> list[str]:
    manifest = json.loads(
        (ROOT / "outputs/l16/data/protocol/split_manifest.json").read_text())
    values = manifest["kitti_v2"]
    if split == "trainval":
        return sorted(values["train"] + values["train_val"])
    if split == "official":
        return sorted(values["official_eval"])
    raise ValueError(split)


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(split: str, dino_path: Path, budgets: list[int]) -> dict:
    payload = pickle.load(dino_path.open("rb"))
    sequences = split_sequences(split)
    video_data = {}
    for seq in sequences:
        record = load_record(seq)
        main = main_boxes(record)
        gt = {
            int(raw["frame"]): {
                str(key): value
                for key, value in raw.get("gt_boxes", {}).items()
            }
            for raw in record["frames"]
        }
        video_data[seq] = (main, gt)
    result = {
        "split": split, "sequences": sequences,
        "dino_cache": str(dino_path),
        "dino_cache_sha256": source_hash(dino_path),
        "budgets": {},
    }
    for budget in budgets:
        aggregate = {
            "frames": 0, "gt_instances": 0,
            "main_observations": 0, "reserve_observations": 0,
            "main_hit_0.50": 0, "reserve_hit_0.50": 0,
            "union_hit_0.50": 0, "main_hit_0.75": 0,
            "reserve_hit_0.75": 0, "union_hit_0.75": 0,
            "reserve_rescue_0.50": 0, "reserve_rescue_0.75": 0,
            "duplicate_iou_0.50": 0, "duplicate_iou_0.70": 0,
        }
        by_video = {}
        for seq in sequences:
            main, gt_by_frame = video_data[seq]
            video = {key: 0 for key in aggregate}
            for frame, gt in gt_by_frame.items():
                dino = payload.get(seq, {}).get(frame, {})
                dino_boxes = np.asarray(dino.get("boxes", []), np.float32)
                scores = np.asarray(dino.get("scores", []), np.float32)
                order = np.argsort(-scores, kind="stable")[:budget]
                reserve = dino_boxes[order] if len(dino_boxes) else \
                    np.zeros((0, 4), np.float32)
                primary = np.asarray(main.get(frame, np.zeros((0, 4))),
                                     np.float32)
                hm50 = hits(primary, gt, 0.50)
                hr50 = hits(reserve, gt, 0.50)
                hu50 = hm50 | hr50
                hm75 = hits(primary, gt, 0.75)
                hr75 = hits(reserve, gt, 0.75)
                hu75 = hm75 | hr75
                duplicate50 = sum(any(iou(x, y) >= 0.50 for y in primary)
                                  for x in reserve)
                duplicate70 = sum(any(iou(x, y) >= 0.70 for y in primary)
                                  for x in reserve)
                row = {
                    "frames": 1, "gt_instances": len(gt),
                    "main_observations": len(primary),
                    "reserve_observations": len(reserve),
                    "main_hit_0.50": len(hm50),
                    "reserve_hit_0.50": len(hr50),
                    "union_hit_0.50": len(hu50),
                    "main_hit_0.75": len(hm75),
                    "reserve_hit_0.75": len(hr75),
                    "union_hit_0.75": len(hu75),
                    "reserve_rescue_0.50": len(hr50 - hm50),
                    "reserve_rescue_0.75": len(hr75 - hm75),
                    "duplicate_iou_0.50": duplicate50,
                    "duplicate_iou_0.70": duplicate70,
                }
                for key, value in row.items():
                    aggregate[key] += value
                    video[key] += value
            by_video[seq] = video
        for key in ("main_hit_0.50", "reserve_hit_0.50", "union_hit_0.50",
                    "main_hit_0.75", "reserve_hit_0.75", "union_hit_0.75",
                    "reserve_rescue_0.50", "reserve_rescue_0.75"):
            denominator = aggregate["gt_instances"]
            aggregate[key.replace("hit_", "recall_").replace(
                "reserve_rescue_", "reserve_rescue_rate_")] = \
                aggregate[key] / max(1, denominator)
        aggregate["main_observations_per_frame"] = aggregate[
            "main_observations"] / max(1, aggregate["frames"])
        aggregate["reserve_observations_per_frame"] = aggregate[
            "reserve_observations"] / max(1, aggregate["frames"])
        aggregate["reserve_duplicate_fraction_0.50"] = aggregate[
            "duplicate_iou_0.50"] / max(1, aggregate["reserve_observations"])
        aggregate["reserve_duplicate_fraction_0.70"] = aggregate[
            "duplicate_iou_0.70"] / max(1, aggregate["reserve_observations"])
        result["budgets"][str(budget)] = {
            "aggregate": aggregate, "by_video": by_video,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("trainval", "official"),
                        default="trainval")
    parser.add_argument("--dino", required=True)
    parser.add_argument("--budgets", default="5,10,20,40,80")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = audit(args.split, Path(args.dino).resolve(),
                   [int(x) for x in args.budgets.split(",") if x])
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "split": args.split,
        "budgets": {key: value["aggregate"] for key, value in
                     result["budgets"].items()},
        "output": str(output),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
