"""Launch crop-PBD cache workers for the L10 TAO-train stream.

Usage:
  python tools/run_l10_pbd_cache.py --gpus 0,1,2,3 --workers-per-gpu 2

Each worker runs tools/cache_l9_tao_pbd.py against
outputs/l10/data/tao_train and writes to outputs/l10/cache/tao_train_pbd.
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
    ap.add_argument("--workers-per-gpu", type=int, default=2)
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--shard-end", type=int, default=-1,
                    help="exclusive end; default = total shards")
    ap.add_argument("--num-shards", type=int, default=0,
                    help="total shard space; default = gpus*workers")
    ap.add_argument("--data-dir", default=str(ROOT / "outputs" / "l10"
                                              / "data" / "tao_train"))
    ap.add_argument("--cache-root", default=str(ROOT / "outputs" / "l10"
                                                / "cache" / "tao_train_pbd"))
    ap.add_argument("--gt-json",
                    default="/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/"
                            "TAO-download/TAO-Amodal/annotations/train.json")
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
                sys.executable, str(ROOT / "tools" / "cache_l9_tao_pbd.py"),
                "--gpu", str(gpu), "--shard", str(shard),
                "--num-shards", str(num_shards),
                "--cache-root", args.cache_root,
                "--data-dir", args.data_dir,
                "--gt-json", args.gt_json,
            ]
            log = ROOT / "outputs" / "l10" / "logs" / \
                f"tao_train_pbd_shard{shard}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with open(log, "wb") as f:
                p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
            procs.append(p)
            shard += 1
        if shard >= shard_end:
            break
    print(f"[l10pbd] launched {shard} workers on gpus {args.gpus}")
    rc = 0
    for p in procs:
        r = p.wait()
        rc |= r
    print(f"[l10pbd] all workers done rc={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
