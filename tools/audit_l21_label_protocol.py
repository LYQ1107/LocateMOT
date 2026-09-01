"""Offline F4 audit of candidate/GT label thresholds on the fixed fast set."""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from tools.train_rmot_candidate_scorer import load_bank, load_metadata  # noqa: E402


def iou_matrix(boxes: np.ndarray, gt_boxes: list[list[float]]) -> np.ndarray:
    if not len(boxes) or not gt_boxes:
        return np.zeros((len(boxes), len(gt_boxes)), np.float32)
    boxes = boxes.astype(np.float32)
    gt = np.asarray(gt_boxes, np.float32)
    left = np.maximum(boxes[:, None, 0], gt[None, :, 0])
    top = np.maximum(boxes[:, None, 1], gt[None, :, 1])
    right = np.minimum(boxes[:, None, 2], gt[None, :, 2])
    bottom = np.minimum(boxes[:, None, 3], gt[None, :, 3])
    intersection = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    gt_area = np.maximum(0, gt[:, 2] - gt[:, 0]) * np.maximum(0, gt[:, 3] - gt[:, 1])
    return intersection / np.maximum(1e-6, area[:, None] + gt_area[None, :] - intersection)


def load_gt(video: str) -> dict[int, dict[str, list[float]]]:
    path = ROOT / "outputs/l11/data/rmot_kitti" / f"{video}.pkl"
    if not path.exists():
        path = ROOT / "outputs/l16/data/kitti_missing/records" / f"{video}.pkl"
    record = pickle.loads(path.read_bytes())
    return {int(frame["frame"]): {str(k): v for k, v in frame.get("gt_boxes", {}).items()}
            for frame in record["frames"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="outputs/l19/protocol/kitti_fast_eval_manifest.json")
    parser.add_argument("--bank-root", default="outputs/l19/dual_banks_features")
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest); bank_root = Path(args.bank_root); out_root = Path(args.out_root)
    if not manifest_path.is_absolute(): manifest_path = ROOT / manifest_path
    if not bank_root.is_absolute(): bank_root = ROOT / bank_root
    if not out_root.is_absolute(): out_root = ROOT / out_root
    if out_root.exists(): raise FileExistsError(out_root)
    manifest = json.loads(manifest_path.read_text())
    rows = sorted(manifest["queries"], key=lambda row: int(row["query_index"]))
    if len(rows) != 160: raise ValueError("expected fixed 160-query manifest")
    metadata = load_metadata()
    videos = sorted({str(row["video"]) for row in rows})
    banks = {v: load_bank(bank_root / "kitti" / f"{v}.pt") for v in videos}
    gt = {v: load_gt(v) for v in videos}
    stats = {str(t): {"candidate_positive": 0, "candidate_total": 0,
                      "positive_frames": 0, "frames": 0, "multi_positive_frames": 0,
                      "target_ids": 0, "covered_target_ids": 0,
                      "state": {"ABSENT": 0, "MAIN_COVERED": 0,
                                "RESERVE_COVERED": 0, "PRESENT_UNCOVERED": 0}}
             for t in (0.5, 0.7)}
    sidecar = {"total": 0, "positive": 0, "sidecar_positive_below_05": 0,
               "sidecar_positive_below_07": 0, "frames": 0,
               "multi_positive_frames": 0}
    for row in rows:
        video, expression = str(row["video"]), str(row["expression"])
        entry = metadata[(video, expression)]
        bank = banks[video]; tensors = bank["tensors"]
        targets = {int(k): {str(v) for v in values}
                   for k, values in entry.get("label", {}).items()}
        ptr = tensors["frame_ptr"].tolist()
        frame_ids = tensors["frame_ids"].tolist()
        boxes = tensors["box"].float().numpy()
        pool = tensors["pool_id"].numpy()
        labels = bank["candidate_gt"]
        for frame_index, frame_id in enumerate(frame_ids):
            begin, end = int(ptr[frame_index]), int(ptr[frame_index + 1])
            target_ids = targets.get(int(frame_id), set())
            gt_frame = gt[video].get(int(frame_id), {})
            target_gt = [gt_frame[gid] for gid in target_ids if gid in gt_frame]
            matrix = iou_matrix(boxes[begin:end], target_gt)
            best = matrix.max(axis=1) if matrix.shape[1] else np.zeros(end - begin, np.float32)
            sidecar_pos = np.asarray([
                value is not None and str(value) in target_ids
                for value in labels[begin:end]], bool)
            sidecar["total"] += end - begin; sidecar["positive"] += int(sidecar_pos.sum())
            sidecar["sidecar_positive_below_05"] += int(np.count_nonzero(sidecar_pos & (best < .5)))
            sidecar["sidecar_positive_below_07"] += int(np.count_nonzero(sidecar_pos & (best < .7)))
            sidecar["frames"] += 1
            if sidecar_pos.any(): sidecar["multi_positive_frames"] += int(sidecar_pos.sum() > 1)
            for threshold, report in stats.items():
                threshold = float(threshold)
                positive = best >= threshold
                main = bool(np.any(positive & (pool[begin:end] == 0)))
                reserve = bool(np.any(positive & (pool[begin:end] == 1)))
                if not target_ids: state = "ABSENT"
                elif main: state = "MAIN_COVERED"
                elif reserve: state = "RESERVE_COVERED"
                else: state = "PRESENT_UNCOVERED"
                report["candidate_positive"] += int(positive.sum())
                report["candidate_total"] += end - begin
                report["frames"] += 1; report["positive_frames"] += int(positive.any())
                report["multi_positive_frames"] += int(positive.sum() > 1)
                report["target_ids"] += len(target_gt)
                report["covered_target_ids"] += int(np.count_nonzero(
                    matrix.max(axis=0) >= threshold)) if matrix.shape[1] else 0
                report["state"][state] += 1
    for threshold, report in stats.items():
        report["candidate_positive_rate"] = report["candidate_positive"] / max(1, report["candidate_total"])
        report["frame_positive_rate"] = report["positive_frames"] / max(1, report["frames"])
        report["multi_positive_rate"] = report["multi_positive_frames"] / max(1, report["frames"])
        report["union_recall"] = report["covered_target_ids"] / max(1, report["target_ids"])
    payload = {"format": "locatemot-l21-label-audit-v1", "manifest": str(manifest_path.resolve()),
               "manifest_sha256": __import__('hashlib').sha256(manifest_path.read_bytes()).hexdigest(),
               "query_count": len(rows), "official_eval_used": False, "stats": stats,
               "sidecar": sidecar, "gt_source": "offline KITTI train-val records only"}
    out_root.mkdir(parents=True, exist_ok=False)
    (out_root / "label_audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# F4 label threshold audit", "", "Offline only; formal GT and evaluator unchanged.", ""]
    for threshold, report in stats.items():
        lines += [f"## IoU >= {threshold}", "", json.dumps(report, indent=2), ""]
    lines += ["## Existing sidecar", "", json.dumps(sidecar, indent=2), ""]
    (out_root / "label_audit.md").write_text("\n".join(lines))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
