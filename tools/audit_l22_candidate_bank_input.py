"""Stage L22 read-only audit of the frozen candidate bank and raw crop path.

This audit does not train, write a bank, alter labels, or invoke TrackEval.  It
uses the fixed L19 fast manifest and reports candidate coverage, missed target
instances, IoU/appearance/position buckets, and whether every bank row maps to
an existing raw KITTI frame and a non-empty clipped crop.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def iou(a, b) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(1e-9, aa + bb - inter)


def stats(values) -> dict:
    values = np.asarray(values, np.float64).reshape(-1)
    if not len(values):
        return {"count": 0}
    return {"count": int(len(values)), "mean": float(values.mean()),
            "std": float(values.std()), "median": float(np.median(values)),
            "q10": float(np.quantile(values, .10)),
            "q90": float(np.quantile(values, .90)),
            "min": float(values.min()), "max": float(values.max())}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / max(1e-6, np.linalg.norm(a) * np.linalg.norm(b)))


def load_metadata(root: Path) -> dict[tuple[str, str], dict]:
    out = {}
    for path in (root / "outputs/l11/data/rmot_kitti/expressions.json",
                 root / "outputs/l16/data/kitti_missing/records/expressions.json"):
        if not path.exists():
            continue
        for video, entries in json.loads(path.read_text()).items():
            for entry in entries:
                text = str(entry.get("expression", entry.get("sentence", "")))
                out[(str(video), text)] = entry
    return out


def load_record(root: Path, video: str) -> dict:
    p = root / "outputs/l11/data/rmot_kitti" / f"{video}.pkl"
    if not p.exists():
        p = root / "outputs/l16/data/kitti_missing/records" / f"{video}.pkl"
    return pickle.loads(p.read_bytes())


def bucket(value: float, edges: tuple[float, ...], names: tuple[str, ...]) -> str:
    for edge, name in zip(edges, names):
        if value < edge:
            return name
    return names[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="outputs/l19/protocol/kitti_fast_eval_manifest.json")
    ap.add_argument("--bank-root", default="outputs/l19/dual_banks_features")
    ap.add_argument("--raw-root", default="data/kitti_tracking_training/image_02")
    ap.add_argument("--out-root", default="outputs/l22/audit/candidate_bank_v2_input")
    args = ap.parse_args()
    manifest = Path(args.manifest); bank_root = Path(args.bank_root)
    raw_root = Path(args.raw_root); out_root = Path(args.out_root)
    if not manifest.is_absolute(): manifest = ROOT / manifest
    if not bank_root.is_absolute(): bank_root = ROOT / bank_root
    if not raw_root.is_absolute(): raw_root = ROOT / raw_root
    if not out_root.is_absolute(): out_root = ROOT / out_root
    if out_root.exists():
        raise FileExistsError(f"refusing to overwrite audit: {out_root}")
    out_root.mkdir(parents=True, exist_ok=False)

    manifest_data = json.loads(manifest.read_text())
    queries = sorted(manifest_data["queries"], key=lambda r: int(r["query_index"]))
    if len(queries) != 160:
        raise ValueError(f"fixed manifest must contain 160 queries, got {len(queries)}")
    metadata = load_metadata(ROOT)
    videos = sorted({str(q["video"]) for q in queries})
    banks, records, raw_info = {}, {}, {}
    total_rows = 0
    row_issues = []
    for video in videos:
        path = bank_root / "kitti" / f"{video}.pt"
        label_path = path.with_suffix(".labels.json")
        bank = torch.load(path, map_location="cpu", weights_only=False)
        t, labels = bank["tensors"], json.loads(label_path.read_text())["candidate_gt"]
        n = len(t["box"])
        if len(labels) != n or int(t["frame_ptr"][-1]) != n:
            raise ValueError(f"invalid row/label/frame_ptr lengths for {video}")
        for i, frame in enumerate(t["frame_ids"].tolist()):
            begin, end = int(t["frame_ptr"][i]), int(t["frame_ptr"][i + 1])
            if not np.all(t["frame"][begin:end].numpy() == int(frame)):
                row_issues.append({"video": video, "frame": int(frame), "issue": "frame_ptr_mismatch"})
        for name, value in t.items():
            if torch.is_floating_point(value) and not bool(torch.isfinite(value.float()).all()):
                row_issues.append({"video": video, "field": name, "issue": "nonfinite"})
        banks[video] = (bank, labels)
        records[video] = load_record(ROOT, video)
        frame_stats = {"frames": 0, "missing_images": [], "unreadable_images": [],
                       "invalid_crops": 0, "candidate_rows": n, "image_paths": []}
        dims = {}
        for i, frame in enumerate(t["frame_ids"].tolist()):
            image_path = raw_root / video / f"{int(frame):06d}.png"
            frame_stats["frames"] += 1
            frame_stats["image_paths"].append(str(image_path))
            if not image_path.exists():
                frame_stats["missing_images"].append(int(frame)); continue
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                frame_stats["unreadable_images"].append(int(frame)); continue
            h, w = image.shape[:2]; dims[int(frame)] = (w, h)
            begin, end = int(t["frame_ptr"][i]), int(t["frame_ptr"][i + 1])
            boxes = t["box"][begin:end].numpy()
            clipped = boxes.copy()
            clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, w)
            clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, h)
            frame_stats["invalid_crops"] += int(np.sum(
                (clipped[:, 2] <= clipped[:, 0]) | (clipped[:, 3] <= clipped[:, 1])))
        frame_stats["unique_image_sizes"] = sorted({tuple(v) for v in dims.values()})
        raw_info[video] = frame_stats
        total_rows += n

    values = {k: [] for k in ("positive_iou", "negative_iou", "positive_appearance",
                              "negative_appearance", "positive_position", "negative_position",
                              "hard_negative_objectness", "easy_negative_objectness")}
    bucket_counts = Counter(); source_counts = Counter(); state_counts = Counter()
    coverage = {"target_instances": 0, "covered_target_instances": 0,
                "missed_target_instances": 0, "query_frames": 0,
                "positive_rows": 0, "negative_rows": 0, "null_frames": 0}
    missed = []
    raw_candidates_checked = 0
    for q in queries:
        video, text = str(q["video"]), str(q["expression"])
        entry = metadata[(video, text)]
        bank, labels = banks[video]; t = bank["tensors"]
        record_by_frame = {int(f["frame"]): f for f in records[video]["frames"]}
        target_by_frame = {int(k): {str(x) for x in v}
                           for k, v in entry.get("label", {}).items()}
        for fi, frame in enumerate(t["frame_ids"].tolist()):
            frame = int(frame); begin, end = int(t["frame_ptr"][fi]), int(t["frame_ptr"][fi + 1])
            target_ids = target_by_frame.get(frame, set())
            gt_boxes = record_by_frame.get(frame, {}).get("gt_boxes", {})
            target_boxes = [np.asarray(gt_boxes[gid], np.float32) for gid in target_ids if gid in gt_boxes]
            pos = np.asarray([labels[i] is not None and str(labels[i]) in target_ids
                              for i in range(begin, end)], bool)
            boxes = t["box"][begin:end].numpy().astype(np.float32)
            clips = t["clip"][begin:end].float().numpy().astype(np.float32)
            objectness = t["objectness"][begin:end].float().numpy().reshape(-1)
            source = t["pool_id"][begin:end].numpy().astype(int)
            ious = np.asarray([max((iou(box, target) for target in target_boxes), default=0.0)
                               for box in boxes], np.float32)
            if target_boxes:
                centers = np.asarray([[(b[0]+b[2])*.5, (b[1]+b[3])*.5] for b in target_boxes])
                image_w, image_h = bank["metadata"].get("image_size", [1242, 375])
                diag = math.hypot(float(image_w), float(image_h))
                positions = np.asarray([max(0.0, 1.0 - min(np.linalg.norm(
                    np.asarray([(b[0]+b[2])*.5, (b[1]+b[3])*.5]) - c) for c in centers) / diag)
                    for b in boxes], np.float32)
            else:
                positions = np.zeros(len(boxes), np.float32)
            positive_indices = np.flatnonzero(pos)
            if len(positive_indices):
                norm = clips / np.maximum(np.linalg.norm(clips, axis=1, keepdims=True), 1e-6)
                appearance = np.max(norm @ norm[positive_indices].T, axis=1)
            else:
                appearance = np.full(len(boxes), np.nan, np.float32)
            coverage["query_frames"] += 1
            coverage["target_instances"] += len(target_ids)
            # ``positive_indices`` are relative to this frame segment; the
            # sidecar labels are bank-global and therefore need ``begin``.
            covered_ids = {str(labels[begin + i]) for i in positive_indices
                           if labels[begin + i] is not None}
            coverage["covered_target_instances"] += len(covered_ids)
            coverage["missed_target_instances"] += len(target_ids - covered_ids)
            if target_ids and target_ids - covered_ids:
                missed.append({"query_index": int(q["query_index"]), "video": video,
                               "frame": frame, "target_ids": sorted(target_ids - covered_ids)})
            null = not target_ids or not len(positive_indices)
            state_counts["NULL" if null else "COVERED"] += 1
            coverage["null_frames"] += int(null)
            coverage["positive_rows"] += int(pos.sum()); coverage["negative_rows"] += int((~pos).sum())
            for src in (0, 1):
                source_counts[("positive" if src == 0 else "reserve", "rows")] += int(np.sum(pos & (source == src)))
                source_counts[("positive" if src == 0 else "reserve", "frames")] += int(np.any(pos & (source == src)))
            neg = np.flatnonzero(~pos)
            order = neg[np.argsort(-objectness[neg], kind="stable")]
            hard = set(order[:min(12, len(order))].tolist())
            for i in range(len(boxes)):
                if pos[i]:
                    values["positive_iou"].append(float(ious[i])); values["positive_position"].append(float(positions[i]))
                    if np.isfinite(appearance[i]): values["positive_appearance"].append(float(appearance[i]))
                else:
                    values["negative_iou"].append(float(ious[i])); values["negative_position"].append(float(positions[i]))
                    if np.isfinite(appearance[i]): values["negative_appearance"].append(float(appearance[i]))
                    values["hard_negative_objectness" if i in hard else "easy_negative_objectness"].append(float(objectness[i]))
                    bucket_counts[("hard" if i in hard else "easy", bucket(float(ious[i]), (.30, .50, .70), ("iou_lt_30", "iou_30_50", "iou_50_70", "iou_ge_70")))] += 1
            raw_candidates_checked += len(boxes)
    coverage["coverage_rate"] = coverage["covered_target_instances"] / max(1, coverage["target_instances"])
    report = {
        "format": "locatemot-l22-candidate-bank-input-audit-v1",
        "manifest": str(manifest), "manifest_sha256": sha256_file(manifest),
        "query_count": len(queries), "calibration_queries": sum(q["split"] == "calibration" for q in queries),
        "screening_queries": sum(q["split"] == "screening" for q in queries),
        "videos": videos, "bank_root": str(bank_root), "raw_root": str(raw_root),
        "candidate_rows": total_rows, "query_frame_units": coverage["query_frames"],
        "coverage": coverage, "missed_target_count": len(missed), "missed_targets": missed[:200],
        "raw_crop_mapping": {"candidates_checked": raw_candidates_checked, "row_issues": row_issues,
                              "per_video": raw_info, "all_frames_present": all(not v["missing_images"] for v in raw_info.values()),
                              "all_frames_readable": all(not v["unreadable_images"] for v in raw_info.values()),
                              "all_crops_nonempty": all(v["invalid_crops"] == 0 for v in raw_info.values())},
        "similarity_and_hard_buckets": {"continuous": {k: stats(v) for k, v in values.items()},
                                         "hard_negative_buckets": {"|".join(k): int(v) for k, v in sorted(bucket_counts.items())}},
        "source_counts": {"%s/%s" % k: int(v) for k, v in sorted(source_counts.items())},
        "frame_state_counts": dict(state_counts),
        "protocol": {"gt_used": True, "gt_use": "audit-only sidecar/record labels; no bank construction decision",
                      "formal_iou_threshold": 0.5, "threshold_changed": False},
    }
    (out_root / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# Stage L22 candidate-bank v2 input audit", "", f"- Manifest: `{manifest}`",
             f"- Manifest SHA256: `{report['manifest_sha256']}`", f"- Queries: `{len(queries)}` (calibration {report['calibration_queries']}, screening {report['screening_queries']})",
             f"- Candidate rows: `{total_rows}`; query-frame units: `{coverage['query_frames']}`",
             f"- Target coverage: `{coverage['covered_target_instances']}/{coverage['target_instances']}` = `{coverage['coverage_rate']:.6f}`; missed target instances: `{coverage['missed_target_instances']}`",
             f"- Raw crop mapping: checked `{raw_candidates_checked}` rows; all frames present/readable = `{report['raw_crop_mapping']['all_frames_present']}/{report['raw_crop_mapping']['all_frames_readable']}`; all crops non-empty = `{report['raw_crop_mapping']['all_crops_nonempty']}`",
             "", "The audit is read-only and uses IoU>=0.5 labels only for coverage and diagnostics; it does not alter formal GT or bank construction.", "",
             "## Similarity summaries", ""]
    for k, v in report["similarity_and_hard_buckets"]["continuous"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Hard-negative buckets", ""]
    for k, v in report["similarity_and_hard_buckets"]["hard_negative_buckets"].items(): lines.append(f"- {k}: {v}")
    lines += ["", "## Decision", "", "Raw image and exact video/frame/bbox crop prerequisites are available; an independent v2 bank can be generated without changing the frozen L19 bank."]
    (out_root / "audit.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"audit": str(out_root / "audit.json"), "candidate_rows": total_rows,
                      "coverage": coverage, "raw_crop_mapping": report["raw_crop_mapping"]}, indent=2))


if __name__ == "__main__":
    main()
