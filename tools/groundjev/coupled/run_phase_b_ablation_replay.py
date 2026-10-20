#!/usr/bin/env python3
"""Replay B0/B1/B2 readouts on one fixed real MOT20 TRAIN window.

Ground-truth boxes are supplied only as a registered candidate engineering
gate, matching the Phase-A replay.  Each arm receives the identical shared
G2 state per frame and owns an independent causal memory store.  The output is
an engineering trace, not an official metric.
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
ADAPTER = Path(
    "/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/"
    "formal_clean_coverage_v2_final/checkpoints/adapter-step011000.pt"
)
SEQ = Path("/data2/user/LocateMOT/data/MOT20/MOT20/train/MOT20-03")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def load_model(path: Path, device: str, input_dim: int) -> GroundJEVCoupledModel:
    model = GroundJEVCoupledModel(input_dim, hidden_dim=256).to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(f"invalid readout checkpoint: {path}")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, nargs="+", default=list(range(83, 91)))
    parser.add_argument("--max-boxes", type=int, default=3)
    parser.add_argument("--checkpoints-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    boxes_by_frame = read_gt(SEQ / "gt/gt.txt", args.frames, args.max_boxes)
    bridge = G2FeatureBridge(checkpoint=BASE, device=args.device,
                             g2_adapter_checkpoint=ADAPTER, local_source_root=SOURCE)
    modes = ("B0_choice_only", "B1_choice_yesno", "B2_choice_yesno_lifecycle")
    models = {}
    runtimes = {}
    checkpoint_info = {}
    for mode in modes:
        checkpoint = args.checkpoints_root / mode / "readout_step_final.pt"
        models[mode] = load_model(checkpoint, args.device, bridge.feature_dimension)
        runtimes[mode] = G2SharedRuntime(
            bridge, models[mode], config=RuntimeConfig(memory_enabled=True))
        checkpoint_info[mode] = {"path": str(checkpoint), "sha256": sha256(checkpoint)}
    rows = {mode: [] for mode in modes}
    for frame in args.frames:
        image_path = SEQ / "img1" / f"{frame:06d}.jpg"
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        # One shared visual encoder call feeds all three readouts.
        state = bridge.encode(
            image, boxes_by_frame[frame], frame_id=frame, image_key=str(image_path),
            task_query="person", metadata={
                "dataset": "MOT20", "split": "TRAIN",
                "candidate_source": "GT_teacher_candidate_engineering_gate",
            })
        for mode in modes:
            result = runtimes[mode].step_from_state(state, timestamp=float(frame))
            rows[mode].append({
                "frame": frame,
                "boxes": boxes_by_frame[frame],
                "assignment": result["assignment"],
                "decisions": result["decisions"].to_dict(),
                "memory": result["memory"],
                "lifecycle": result["lifecycle"],
            })
    receipt = {
        "status": "PHASE_B_TRAINED_READOUT_CAUSAL_REPLAY_COMPLETE",
        "video": "MOT20-03 TRAIN engineering window",
        "frames": args.frames,
        "candidate_source": "GT_teacher_candidate_engineering_gate",
        "official_metric": False,
        "single_backbone": True,
        "second_vlm_loaded": False,
        "visual_forward_calls": bridge.forward_calls,
        "checkpoint_info": checkpoint_info,
        "arms": rows,
        "persistent_memory": {mode: runtimes[mode].store.state_dict() for mode in modes},
        "quality_claim": False,
        "note": "All arms share one G2 visual state per frame; fixed annotation boxes are engineering inputs, not a benchmark result.",
    }
    (args.out / "phase_b_replay_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

