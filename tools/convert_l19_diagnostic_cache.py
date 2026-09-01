"""Convert a full L19 diagnostic cache to the legacy evaluator score format.

This is a format-only operation.  It reads the already-computed L19
``diagnose_l19.py`` cache and writes the eight-column score rows expected by
``eval_l18_carr.py``; it never runs a model or changes scores.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def safe_expression(text: str) -> str:
    return str(text).replace("/", "_")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", default="trainval_kitti")
    parser.add_argument("--frame-offset", type=int, default=1)
    args = parser.parse_args()
    source_root = (ROOT / args.input_root).resolve() / "cache" / args.dataset
    destination_root = (ROOT / args.output_root).resolve() / args.dataset
    paths = sorted(source_root.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(source_root)
    total = 0
    for source in paths:
        with np.load(source, allow_pickle=False) as data:
            required = {"frame", "track_id", "box", "raw"}
            if not required.issubset(data.files):
                raise ValueError(f"missing diagnostic fields: {source}")
            frame = np.asarray(data["frame"], np.int64) + int(args.frame_offset)
            track_id = np.asarray(data["track_id"], np.int64)
            boxes = np.asarray(data["box"], np.float32).reshape(-1, 4)
            raw = np.asarray(data["raw"], np.float32)
        if not (len(frame) == len(track_id) == len(boxes) == len(raw)):
            raise ValueError(f"row length mismatch: {source}")
        rows = np.stack((
            frame.astype(np.float32), track_id.astype(np.float32),
            boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0],
            boxes[:, 3] - boxes[:, 1],
            1.0 / (1.0 + np.exp(-np.clip(raw, -40.0, 40.0))), raw,
        ), axis=1).astype(np.float32)
        relative = source.relative_to(source_root)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, rows=rows)
        temporary.replace(destination)
        total += len(rows)
    print({"files": len(paths), "rows": total,
           "output_root": str(destination_root)})


if __name__ == "__main__":
    main()
