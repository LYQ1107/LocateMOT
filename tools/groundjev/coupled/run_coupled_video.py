#!/usr/bin/env python3
"""Bounded causal GroundJEV replay on a real MOT video window.

The default input is a small TRAIN engineering window and uses the registered
MOT annotations only to supply fixed candidate boxes.  It is explicitly marked
``GT_teacher_candidate_engineering_gate``; it is not an official metric and
does not train or tune the readout.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from locatemot.groundjev.coupled.g2_feature_bridge import G2FeatureBridge
from locatemot.groundjev.coupled.g2_shared_runtime import G2SharedRuntime, RuntimeConfig
from locatemot.groundjev.coupled.groundjev_model import GroundJEVCoupledModel


BASE = Path("/data2/user/LocateMOT_GroundJEV/models/GroundAnything-DLM")
SOURCE = Path("/data2/user/LocateMOT_GroundJEV/external/GroundAnything")
ADAPTER = Path("/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final/checkpoints/adapter-step011000.pt")
SEQ = Path("/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-03")


def read_gt(path: Path, frames: list[int], max_boxes: int) -> dict[int, list[list[float]]]:
    result = {frame: [] for frame in frames}
    with path.open() as handle:
        for row in csv.reader(handle):
            frame = int(row[0])
            if frame not in result or len(result[frame]) >= max_boxes:
                continue
            x, y, w, h = map(float, row[2:6])
            result[frame].append([x, y, x + w, y + h])
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, nargs="+", default=[83, 84])
    parser.add_argument("--max-boxes", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    boxes_by_frame = read_gt(SEQ / "gt/gt.txt", args.frames, args.max_boxes)
    bridge = G2FeatureBridge(checkpoint=BASE, device=args.device,
                             g2_adapter_checkpoint=ADAPTER, local_source_root=SOURCE)
    first_frame = args.frames[0]
    first_image = Image.open(SEQ / "img1" / f"{first_frame:06d}.jpg").convert("RGB")
    # The readout is the Phase-A untrained typed interface; only real G2 visual
    # evidence and causal memory are evaluated here.
    model = GroundJEVCoupledModel(bridge.feature_dimension, hidden_dim=256).to(args.device)
    model.eval()
    runtime = G2SharedRuntime(bridge, model, config=RuntimeConfig(memory_enabled=True))
    rows = []
    for frame in args.frames:
        image_path = SEQ / "img1" / f"{frame:06d}.jpg"
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        result = runtime.step(image=image, boxes=boxes_by_frame[frame], timestamp=float(frame),
                              frame_id=frame, image_key=str(image_path), task_query="person",
                              metadata={"dataset": "MOT20", "split": "TRAIN",
                                        "candidate_source": "GT_teacher_candidate_engineering_gate"})
        rows.append({"frame": frame, "boxes": boxes_by_frame[frame],
                     "assignment": result["assignment"],
                     "decisions": result["decisions"].to_dict(),
                     "memory": result["memory"], "lifecycle": result["lifecycle"]})
    receipt = {
        "status": "PHASE_A_CAUSAL_VIDEO_REPLAY_COMPLETE",
        "video": "MOT20-03 TRAIN engineering window",
        "frames": args.frames,
        "candidate_source": "GT_teacher_candidate_engineering_gate",
        "official_metric": False,
        "single_backbone": True,
        "second_vlm_loaded": False,
        "visual_forward_calls": bridge.forward_calls,
        "rows": rows,
        "persistent_memory": runtime.store.state_dict(),
        "quality_claim": False,
        "note": "Candidate boxes are fixed annotation inputs for Phase-A causal plumbing; no benchmark claim.",
    }
    (args.out / "causal_video_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
