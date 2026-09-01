"""Generate query-independent Detic proposals for missing Refer-KITTI-V2 videos.

The detector, score threshold, top-50 budget, resize, and checkpoint are exactly
the L10 DLA protocol.  Only the input sequence enumeration and output root are
L16-specific; no L10/L11 cache is overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from generate_l10_kitti_dets import KITTI_FRAMES, worker


DEFAULT_SEQS = ("0016", "0017", "0018", "0020")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--seqs", default=",".join(DEFAULT_SEQS))
    ap.add_argument("--out", default="outputs/l16/data/kitti_missing/dets")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    seqs = tuple(x.strip() for x in args.seqs.split(",") if x.strip())
    if not seqs or any(x not in DEFAULT_SEQS for x in seqs):
        raise SystemExit(f"--seqs must be a subset of {DEFAULT_SEQS}")
    items = []
    for seq in sorted(seqs):
        image_dir = KITTI_FRAMES / seq
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        for path in sorted(image_dir.glob("*.png")):
            items.append((seq, int(path.stem), path))
    if args.max_frames:
        items = items[:args.max_frames]

    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    gpus = [int(x) for x in args.gpus.split(",")]
    if not 1 <= len(gpus) <= 4:
        raise SystemExit("L16 permits one to four GPUs")
    shards = [[] for _ in gpus]
    for index, item in enumerate(items):
        shards[index % len(gpus)].append(item)
    print(f"[l16-kitti-dets] seqs={seqs} frames={len(items)} "
          f"gpus={gpus} out={out_root}", flush=True)

    procs = []
    for gpu, shard in zip(gpus, shards):
        proc = torch.multiprocessing.Process(
            target=worker, args=(gpu, shard, str(out_root)))
        proc.start()
        procs.append(proc)
    failures = []
    for proc in procs:
        proc.join()
        if proc.exitcode != 0:
            failures.append(proc.exitcode)
    if failures:
        raise SystemExit(f"detector workers failed: {failures}")
    print("[l16-kitti-dets] done", flush=True)


if __name__ == "__main__":
    main()

