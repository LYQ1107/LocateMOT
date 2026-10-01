#!/usr/bin/env python3
"""Measure Swin-B proposal coverage on a small legal development slice.

This is a U1 diagnostic.  It uses only fit/calibration/validation videos,
never opens reserved official-test labels, and performs no training or
threshold fitting.  The model prediction for a selected frame is completed
before that frame's label file is opened.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MMDET_ROOT = PROJECT_ROOT / "third_party/mmdetection"
for path in (PROJECT_ROOT, MMDET_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from mmengine.config import Config  # noqa: E402
from mmdet.registry import MODELS  # noqa: E402
from mmdet.structures import DetDataSample  # noqa: E402
from mmdet.utils import register_all_modules  # noqa: E402

from locatemot.paths import V1_ROOT, V2_ROOT  # noqa: E402
from locatemot.rmot.l49_data import load_l49_queries  # noqa: E402
from locatemot.unified.data.legal_scope import assert_legal_path, assert_legal_video  # noqa: E402


def iou_xyxy(left: list[float], right: list[float]) -> float:
    x0 = max(float(left[0]), float(right[0]))
    y0 = max(float(left[1]), float(right[1]))
    x1 = min(float(left[2]), float(right[2]))
    y1 = min(float(left[3]), float(right[3]))
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_left = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    area_right = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = area_left + area_right - inter
    return inter / union if union > 0.0 else 0.0


def read_boxes(path: Path, width: int, height: int) -> dict[str, list[float]]:
    boxes: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 6:
            continue
        track_id = str(fields[1])
        cx, cy, box_w, box_h = (float(value) for value in fields[2:6])
        boxes[track_id] = [
            (cx - box_w / 2.0) * width,
            (cy - box_h / 2.0) * height,
            (cx + box_w / 2.0) * width,
            (cy + box_h / 2.0) * height,
        ]
    return boxes


def select_rows(per_split: dict[str, int]) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped_missing_target = 0
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        root = V1_ROOT if dataset == "refer_kitti_v1" else V2_ROOT
        for row in load_l49_queries(dataset):
            split = row["split"]
            wanted = int(per_split.get(f"{dataset}:{split}", 0))
            if wanted <= 0 or len([x for x in rows if x.get("dataset") == dataset and x.get("split") == split]) >= wanted:
                continue
            video = assert_legal_video(row["video"], dataset=dataset)
            valid: list[tuple[int, list[str]]] = []
            for frame, targets in sorted(row["target"].items()):
                if not targets:
                    continue
                label = root / "labels_with_ids" / "image_02" / video / f"{int(frame):06d}.txt"
                if not label.is_file():
                    continue
                available = {line.split()[1] for line in label.read_text(encoding="utf-8").splitlines() if len(line.split()) >= 2}
                if any(str(target) not in available for target in targets):
                    skipped_missing_target += 1
                    continue
                valid.append((int(frame), sorted(str(target) for target in targets)))
            if valid:
                frame, targets = valid[len(valid) // 2]
                selected = dict(row)
                selected.update(video=video, probe_frame=frame, probe_targets=targets)
                rows.append(selected)
    return rows, skipped_missing_target


def build_model(device: torch.device, max_per_img: int):
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(str(MMDET_ROOT / "configs/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco.py"))
    cfg.model.language_model.name = str((PROJECT_ROOT / "weights/bert-base-uncased").resolve())
    cfg.model.test_cfg.max_per_img = int(max_per_img)
    model = MODELS.build(cfg.model)
    checkpoint = PROJECT_ROOT / "weights/groundingdino_swinb_cogcoor_mmdet-55949c9c.pth"
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = loaded.get("state_dict", loaded.get("model", loaded))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval(), checkpoint, len(missing), len(unexpected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-per-img", type=int, default=300)
    parser.add_argument("--prompt", default=None, help="fixed custom-entity prompt for a class-agnostic proposal diagnostic")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs/unified/u1_proposal_gate.json")
    args = parser.parse_args()
    per_split = {
        "refer_kitti_v1:calibration": 4,
        "refer_kitti_v1:validation": 4,
        "refer_kitti_v2:calibration": 4,
        "refer_kitti_v2:validation": 4,
    }
    rows, skipped_missing_target = select_rows(per_split)
    if not rows:
        raise RuntimeError("no legal proposal-gate rows")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, checkpoint, missing_keys, unexpected_keys = build_model(device, args.max_per_img)
    measurements: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        image_path = assert_legal_path(
            PROJECT_ROOT / "data/kitti_tracking/training/image_02" / row["video"] / f"{int(row['probe_frame']):06d}.png"
        )
        # Match the repository's OpenCV/BGR loader; DetDataPreprocessor then
        # performs its configured BGR->RGB conversion exactly once.
        image = np.asarray(Image.open(image_path).convert("RGB"))[..., ::-1].copy()
        sample = DetDataSample()
        sample.set_metainfo(dict(img_id=0, img_shape=image.shape[:2], ori_shape=image.shape[:2], scale_factor=(1.0, 1.0)))
        sample.text = str(args.prompt if args.prompt is not None else row["sentence"])
        sample.custom_entities = True
        with torch.inference_mode():
            outputs = model.test_step({"inputs": [torch.from_numpy(image).permute(2, 0, 1).contiguous(),], "data_samples": [sample]})
        prediction = outputs[0].pred_instances
        boxes = prediction.bboxes.tensor if hasattr(prediction.bboxes, "tensor") else prediction.bboxes
        boxes = boxes.detach().float().cpu().tolist()
        scores = prediction.scores.detach().float().cpu().tolist()
        # Labels are opened only after the model prediction is complete.
        dataset_root = V1_ROOT if row["dataset"] == "refer_kitti_v1" else V2_ROOT
        label_path = dataset_root / "labels_with_ids" / "image_02" / row["video"] / f"{int(row['probe_frame']):06d}.txt"
        targets = read_boxes(label_path, image.shape[1], image.shape[0])
        target_metrics: dict[str, dict[str, float]] = {}
        for target_id in row["probe_targets"]:
            target_box = targets.get(str(target_id))
            if target_box is None:
                continue
            values = [iou_xyxy(candidate, target_box) for candidate in boxes]
            best_index = int(np.argmax(values)) if values else -1
            target_metrics[str(target_id)] = {
                "best_iou": max(values) if values else 0.0,
                "best_score": scores[best_index] if best_index >= 0 else 0.0,
            }
        measurements.append({
            "dataset": row["dataset"], "split": row["split"], "video": row["video"],
            "query_id": int(row["query_id"]), "frame_id": int(row["probe_frame"]),
            "sentence": row["sentence"], "target_count": len(target_metrics),
            "candidate_count": len(boxes), "target_metrics": target_metrics,
            "finite_boxes": bool(torch.isfinite(torch.tensor(boxes)).all()),
            "finite_scores": bool(torch.isfinite(torch.tensor(scores)).all()),
        })
        print(f"[{index + 1}/{len(rows)}] {row['dataset']} {row['video']} frame={row['probe_frame']}", flush=True)
    target_values = [metric for row in measurements for metric in row["target_metrics"].values()]
    aggregate = {
        "rows": len(measurements),
        "target_boxes": len(target_values),
        "iou_at_025": sum(value["best_iou"] >= 0.25 for value in target_values) / max(1, len(target_values)),
        "iou_at_050": sum(value["best_iou"] >= 0.50 for value in target_values) / max(1, len(target_values)),
        "mean_best_iou": sum(value["best_iou"] for value in target_values) / max(1, len(target_values)),
        "mean_best_score": sum(value["best_score"] for value in target_values) / max(1, len(target_values)),
        "candidate_count_distribution": dict(Counter(str(row["candidate_count"]) for row in measurements)),
        "skipped_missing_target_references": skipped_missing_target,
        "elapsed_seconds": time.perf_counter() - started,
    }
    payload = {
        "format": "locatemot-u-u1-proposal-gate-v1",
        "status": "diagnostic_complete",
        "device": str(device),
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "checkpoint_missing_keys": missing_keys,
        "checkpoint_unexpected_keys": unexpected_keys,
        "max_per_img": int(args.max_per_img),
        "prompt": args.prompt,
        "selection": per_split,
        "aggregate": aggregate,
        "rows": measurements,
        "prediction_before_label": True,
        "official_test_labels_read": False,
        "screening_gt_used": False,
        "training_launched": False,
    }
    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
