"""Cache query-independent GroundingDINO proposals for the L13 KITTI ablation.

The detector is used only as an open-vocabulary proposal pool.  Language
conditioning is deliberately limited to a fixed road-user vocabulary; the
RMOT expression is applied later by ``eval_l13_rmot.py`` through proposal-box
overlap and the unchanged UIDM tracker.

The implementation uses the already available OpenMMLab GroundingDINO
checkout/weights outside this repository.  A cache can be built per GPU and
merged with ``--merge-inputs``.
"""
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
TTAOD_ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/TTAOD-F-main")
DEFAULT_CONFIG = TTAOD_ROOT / "configs/mm_grounding_dino/" \
    "grounding_dino_swin-t_pretrain_obj365.py"
DEFAULT_CHECKPOINT = TTAOD_ROOT / "download/" \
    "grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_" \
    "20231204_095047-b448804b.pth"
IMAGE_ROOT = ROOT / "data/kitti_tracking_training/image_02"
DEFAULT_SEQS = ("0005", "0011", "0013", "0019")
PROMPT = "car. truck. bus. pedestrian. person. bicycle. motorcycle."


def image_paths(seq: str):
    paths = sorted(IMAGE_ROOT.joinpath(seq).glob("*.png"))
    if not paths:
        paths = sorted(IMAGE_ROOT.joinpath(seq).glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"no KITTI images found for sequence {seq}")
    return paths


def write_pickle(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def merge(inputs, out):
    merged = {
        "__meta__": {
            "prompt": PROMPT,
            "source": "OpenMMLab GroundingDINO Swin-T",
            "inputs": [str(Path(p).resolve()) for p in inputs],
        }
    }
    for input_path in inputs:
        with Path(input_path).open("rb") as f:
            payload = pickle.load(f)
        for seq, frames in payload.items():
            if seq.startswith("__"):
                continue
            if seq in merged:
                overlap = set(merged[seq]).intersection(frames)
                if overlap:
                    raise RuntimeError(
                        f"duplicate cached frames for {seq}: {sorted(overlap)[:3]}")
            merged[seq] = frames
    write_pickle(out, merged)
    counts = {seq: len(frames) for seq, frames in merged.items()
              if not seq.startswith("__")}
    print(f"[l13-dino] merged={out} frames={sum(counts.values())} "
          f"sequences={counts}", flush=True)


def build(args):
    if Path.cwd().resolve() != TTAOD_ROOT.resolve() and \
            os.environ.get("LOCATEMOT_L13_DINO_REEXEC") != "1":
        env = dict(os.environ)
        env["LOCATEMOT_L13_DINO_REEXEC"] = "1"
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            cwd=str(TTAOD_ROOT), env=env, check=True)
        return
    if not args.out.is_absolute():
        args.out = ROOT / args.out
    args.out = args.out.resolve()
    args.config = args.config.resolve()
    args.checkpoint = args.checkpoint.resolve()
    os.chdir(TTAOD_ROOT)
    sys.path.insert(0, str(TTAOD_ROOT))
    import cv2
    from mmdet.apis import inference_detector, init_detector

    model = init_detector(str(args.config), str(args.checkpoint),
                          device=f"cuda:{args.gpu}")
    payload = {
        "__meta__": {
            "prompt": PROMPT,
            "box_threshold": args.box_threshold,
            "text_threshold": args.text_threshold,
            "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
        }
    }
    total = 0
    started = time.time()
    for seq in args.seqs:
        frames = {}
        paths = image_paths(seq)
        if args.max_frames:
            paths = paths[:args.max_frames]
        for image_path in paths:
            frame = int(image_path.stem)
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"failed to decode image: {image_path}")
            result = inference_detector(
                model, image, text_prompt=PROMPT,
                custom_entities=True)
            pred = result.pred_instances
            boxes = pred.bboxes.detach().cpu().numpy().astype("float32")
            scores = pred.scores.detach().cpu().numpy().astype("float32")
            frames[frame] = {"boxes": boxes, "scores": scores}
            total += 1
            if total % 25 == 0 or total == 1:
                elapsed = time.time() - started
                print(f"[l13-dino] {seq} frame={frame} "
                      f"progress={total} elapsed={elapsed:.0f}s", flush=True)
        payload[seq] = frames
    write_pickle(args.out, payload)
    counts = {seq: len(frames) for seq, frames in payload.items()
              if not seq.startswith("__")}
    print(f"[l13-dino] wrote={args.out} frames={total} sequences={counts}",
          flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seqs", nargs="+", default=list(DEFAULT_SEQS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.20)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--merge-inputs", nargs="+")
    args = parser.parse_args()
    if args.merge_inputs:
        if args.out is None:
            parser.error("--out is required with --merge-inputs")
        merge(args.merge_inputs, args.out)
        return
    if args.out is None:
        parser.error("--out is required")
    build(args)


if __name__ == "__main__":
    main()
