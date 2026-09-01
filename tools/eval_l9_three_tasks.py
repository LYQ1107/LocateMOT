"""Stage L9: run ordinary + RMOT (+ optional OVMOT) evals for a checkpoint.

Runs ordinary MOT (4 domains) and RMOT (40 GT queries) in parallel on the
given GPUs, then optionally OVMOT (TAO val, full PBD cache) on one GPU.
All outputs are written under --out with per-script logs.

Usage:
  python tools/eval_l9_three_tasks.py --ckpt outputs/l9/checkpoints/... \
      --out outputs/l9/trackeval/l9_main --gpus 4,6 [--ovmot-gpu 7]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
PY = "/home/lwr/anaconda3/envs/locatemot/bin/python"


def run(cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as f:
        f.write(f"\n=== {time.ctime()} {' '.join(cmd)} ===\n".encode())
        f.flush()
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpus", default="4,6")
    ap.add_argument("--ovmot-gpu", type=int, default=-1)
    ap.add_argument("--ovmot-cache",
                    default=str(ROOT / "outputs" / "l9" / "cache"
                                / "tao_val_pbd"))
    ap.add_argument("--threshold-file", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gpus = [int(x) for x in args.gpus.split(",")]
    procs = []
    logs = []
    for i, gpu in enumerate(gpus):
        domains = ["dance,bdd", "mot17,mot20"][i % 2]
        log = out / f"ordinary_gpu{gpu}.log"
        p = run([PY, str(ROOT / "tools" / "eval_l8_ordinary.py"),
                 "--ckpt", args.ckpt, "--out",
                 str(out / f"ordinary_gpu{gpu}"), "--gpu", str(gpu),
                 "--domains", domains], log)
        procs.append(p)
        logs.append(log)
    # RMOT on the first GPU (sequential after its ordinary half)
    rmot_log = out / "rmot.log"
    thr = args.threshold_file or str(
        ROOT / "outputs" / "l9" / "calib" / "threshold_l9.json")
    p = run([PY, str(ROOT / "tools" / "eval_l8_rmot.py"),
             "--ckpt", args.ckpt, "--out", str(out / "rmot"),
             "--gpu", str(gpus[0]),
             "--threshold-file", thr],
            rmot_log)
    procs.append(p)
    for p in procs:
        p.wait()
    if args.ovmot_gpu >= 0:
        log = out / "ovmot_full.log"
        p = run([PY, str(ROOT / "tools" / "eval_l8_ovmot.py"),
                 "--ckpt", args.ckpt, "--out", str(out / "ovmot_full"),
                 "--gpu", str(args.ovmot_gpu),
                 "--pbd-cache", args.ovmot_cache], log)
        p.wait()
    print("[eval_l9] all done", flush=True)


if __name__ == "__main__":
    main()
