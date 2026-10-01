#!/usr/bin/env python3
"""Run the published TempRMOT checkpoint on a legal development subset.

The upstream inference entrypoint enumerates the reserved official videos.
This adapter calls the same published ``Detector`` implementation directly,
but supplies only videos from the frozen LocateMOT-U development split.  It
keeps the external repository and its environment isolated from LocateMOT-U.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locatemot.unified.data.legal_scope import LEGAL_SPLITS, assert_legal_video  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("refer_kitti_v1", "refer_kitti_v2"), default="refer_kitti_v1")
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--expressions-per-video", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args_cli = parse_args()
    videos = [assert_legal_video(video, dataset=args_cli.dataset) for video in args_cli.videos]
    allowed = set(sum((list(LEGAL_SPLITS[args_cli.dataset][part]) for part in ("fit", "calibration", "validation")), []))
    if not set(videos).issubset(allowed):
        raise RuntimeError(f"videos outside legal development split: {videos}")
    if args_cli.expressions_per_video < 1:
        raise ValueError("expressions-per-video must be positive")

    external_root = args_cli.external_root.resolve()
    sys.path.insert(0, str(external_root))
    from inference import Detector  # noqa: PLC0415
    from models import build_model  # noqa: PLC0415
    from util.tool import load_model  # noqa: PLC0415

    checkpoint = args_cli.checkpoint.resolve()
    checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model_args = checkpoint_payload["args"]
    # The released checkpoint was trained under an internal alias, while the
    # public inference code registers the same builder as ``temp_rmot``.
    model_args.meta_arch = "temp_rmot"
    model_args.dataset_file = "e2e_rmot"
    model_args.device = "cuda"
    model_args.rmot_path = str(args_cli.data_root.resolve())
    model_args.output_dir = str(args_cli.output_dir.resolve())
    model_args.exp_name = "legal_dev"
    model_args.distributed = False
    model_args.rank = 0
    model_args.world_size = 1
    model_args.gpu = 0
    model_args.num_workers = 0
    model_args.vis = False
    model_args.visualization = False
    model_args.use_checkpoint = False

    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    model, _, _ = build_model(model_args)
    model = load_model(model, str(checkpoint))
    model.eval().cuda()

    # Preserve the adapter filename (checkpoint0049.pth) when the actual
    # downloaded release is reached through a symlink named checkpoint_rk.pth.
    checkpoint_id = int(args_cli.checkpoint.stem.split("t")[-1])
    completed = []
    for video in videos:
        expression_root = args_cli.data_root / "expression" / video
        expression_files = sorted(expression_root.glob("*.json"))[: args_cli.expressions_per_video]
        if len(expression_files) < args_cli.expressions_per_video:
            raise FileNotFoundError(f"not enough expressions for legal video {video}")
        for expression_file in expression_files:
            detector = Detector(
                model_args,
                checkpoint_id=checkpoint_id,
                model=model,
                seq_num=[video, expression_file.name],
            )
            detector.detect()
            completed.append({"video": video, "expression": expression_file.name})

    metadata = {
        "format": "locatemot-u-external-temprmot-legal-v1",
        "status": "PREDICTION_COMPLETE",
        "external_repository": "TempRMOT",
        "external_commit": "6a65640d849fdee4a32bb055945ee34c3b0edeb1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset": args_cli.dataset,
        "videos": videos,
        "completed_sequences": completed,
        "output_dir": str(args_cli.output_dir.resolve()),
        "official_test_labels_read": False,
        "screening_gt_used": False,
        "training_launched": False,
    }
    (args_cli.output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
