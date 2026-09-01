"""Train-validation oracle audit for the L18 dual KITTI bank.

This is a privileged diagnostic only.  It selects bank observations from the
training-validation GT labels and therefore must never be used for training,
threshold selection, or official evaluation.  The semantic oracle keeps the
selected bank track IDs; the association oracle gives the same selected boxes
GT-consistent persistent IDs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))

from tools.eval_l18_carr import (  # noqa: E402
    run_trackeval,
    trainval_queries,
    write_trainval_gt,
)


def read_gt(path: Path) -> dict[int, list[tuple[str, np.ndarray]]]:
    by_frame: dict[int, list[tuple[str, np.ndarray]]] = {}
    for line in path.read_text().splitlines():
        fields = line.strip().split(",")
        if len(fields) < 6:
            continue
        frame = int(float(fields[0]))
        gid = str(fields[1])
        x, y, w, h = map(float, fields[2:6])
        by_frame.setdefault(frame, []).append(
            (gid, np.asarray([x, y, x + w, y + h], np.float32)))
    return by_frame


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    bb = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    return inter / max(1e-6, aa + bb - inter)


def bank_frames(video: str) -> dict[int, list[tuple[str, np.ndarray, int]]]:
    bank_path = ROOT / "outputs/l18/dual_banks/kitti" / f"{video}.pt"
    label_path = bank_path.with_suffix(".labels.json")
    if not bank_path.exists() or not label_path.exists():
        raise FileNotFoundError(
            f"train-val dual bank labels required: {bank_path}")
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    labels = json.loads(label_path.read_text())["candidate_gt"]
    tensors = bank["tensors"]
    out: dict[int, list[tuple[str, np.ndarray, int]]] = {}
    for frame_index, frame in enumerate(tensors["frame_ids"].tolist()):
        begin = int(tensors["frame_ptr"][frame_index])
        end = int(tensors["frame_ptr"][frame_index + 1])
        values = []
        for index in range(begin, end):
            gid = labels[index]
            if gid is None:
                continue
            values.append((str(gid), tensors["box"][index].numpy().copy(),
                           int(tensors["track_id"][index])))
        out[int(frame)] = values
    return out


def selected_rows(video: str, expression: str, gt_root: Path,
                  mode: str,
                  candidate_cache: dict[str, dict[int, list[tuple[str, np.ndarray, int]]]]) -> list[list[float | int]]:
    gt = read_gt(gt_root / video / expression / "gt.txt")
    if video not in candidate_cache:
        candidate_cache[video] = bank_frames(video)
    candidates = candidate_cache[video]
    oracle_ids: dict[str, int] = {}
    next_id = 1
    rows = []
    for gt_frame, targets in sorted(gt.items()):
        raw_frame = int(gt_frame) - 1
        frame_candidates = candidates.get(raw_frame, [])
        used = set()
        for gid, target_box in targets:
            matches = [item for item in frame_candidates
                       if item[0] == gid and item[2] not in used]
            if not matches:
                continue
            _label, box, bank_id = max(matches, key=lambda item: iou(item[1], target_box))
            used.add(bank_id)
            if mode == "association":
                if gid not in oracle_ids:
                    oracle_ids[gid] = next_id
                    next_id += 1
                output_id = oracle_ids[gid]
            else:
                output_id = bank_id
            x1, y1, x2, y2 = [float(x) for x in box]
            rows.append([gt_frame, output_id, x1, y1, x2 - x1, y2 - y1,
                         1.0, -1, -1, -1])
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/l18/eval/dual_oracle_trainval")
    args = parser.parse_args()
    queries, gt_root, seqmap, sequences, _ = trainval_queries("trainval_kitti")
    write_trainval_gt("trainval_kitti", queries, gt_root)
    out_root = (ROOT / args.out).resolve()
    result_root = out_root / "uidm18"
    result_root.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    for mode in ("semantic", "association"):
        variant_root = out_root / mode
        candidate_cache = {}
        for video, expression, _spec in queries:
            destination = variant_root / "uidm18" / video / expression
            destination.mkdir(parents=True, exist_ok=True)
            source = gt_root / video / expression / "gt.txt"
            gt_destination = destination / "gt.txt"
            if not gt_destination.exists():
                gt_destination.symlink_to(source.resolve())
            rows = selected_rows(video, expression, gt_root, mode,
                                 candidate_cache)
            (destination / "predict.txt").write_text(
                "".join(",".join(
                    f"{value:.6f}" if isinstance(value, float) else str(value)
                    for value in row) + "\n" for row in rows))
        metrics, log = run_trackeval(
            "trainval_kitti", variant_root, seqmap, sequences,
            {(video, expression) for video, expression, _ in queries})
        all_metrics[mode] = {"metrics": metrics, "log": str(log),
                             "queries": len(queries)}
    output = out_root / "results.json"
    output.write_text(json.dumps({
        "protocol": "train_val only; GT-privileged diagnostic",
        "bank": str(ROOT / "outputs/l18/dual_banks/kitti"),
        "budget": 20, "results": all_metrics,
    }, indent=2) + "\n")
    print(json.dumps(all_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
