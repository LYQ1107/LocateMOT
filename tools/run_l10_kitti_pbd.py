"""Launch crop-PBD cache workers for Refer-KITTI-V2 candidates.

Usage:
  python tools/run_l10_kitti_pbd.py --gpus 0,1,2,3 --workers-per-gpu 2
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--workers-per-gpu", type=int, default=1)
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--shard-end", type=int, default=-1)
    ap.add_argument("--num-shards", type=int, default=0)
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(",")]
    total = len(gpus) * args.workers_per_gpu
    num_shards = args.num_shards if args.num_shards > 0 else total
    shard_end = args.shard_end if args.shard_end > 0 else total
    procs = []
    shard = args.shard_start
    for gpu in gpus:
        for _ in range(args.workers_per_gpu):
            if shard >= shard_end:
                break
            cmd = [
                sys.executable, str(ROOT / "tools" / "cache_l10_kitti_pbd.py"),
                "--gpu", str(gpu), "--shard", str(shard),
                "--num-shards", str(num_shards),
            ]
            log = ROOT / "outputs" / "l10" / "logs" / \
                f"kitti_pbd_shard{shard}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "wb") as f:
                p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            procs.append(p)
            shard += 1
        if shard >= shard_end:
            break
    print(f"[l10kpbd] launched {len(procs)} workers (shards "
          f"{args.shard_start}-{shard}, num_shards={num_shards})")
    rc = 0
    for p in procs:
        rc |= p.wait()
    print(f"[l10kpbd] all workers done rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
