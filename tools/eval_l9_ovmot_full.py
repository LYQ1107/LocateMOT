"""Stage L9: full-observation TAO OVMOT eval for one or more checkpoints.

Runs eval_l8_ovmot in 4 shards across --gpus (one shard per GPU), merges
the predictions and runs the official TETA evaluator per checkpoint.

Usage:
  python tools/eval_l9_ovmot_full.py --gpus 1,2,5,7 \
      --ckpts outputs/l8/checkpoints/uidm_l8_v2/latest.pt,outputs/l8/checkpoints/uidm_l8_final/latest.pt \
      --out outputs/l9/trackeval/ovmot_full
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
PY = "/home/lwr/anaconda3/envs/locatemot/bin/python"
PBD_CACHE = ROOT / "outputs" / "l9" / "cache" / "tao_val_pbd"


def run(cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as f:
        f.write(f"\n=== {time.ctime()} {' '.join(cmd)} ===\n".encode())
        f.flush()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pbd-cache", default=str(PBD_CACHE))
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ci, ck in enumerate(args.ckpts.split(",")):
        ck = ck.strip()
        name = f"ckpt{ci}"
        procs = []
        for si, gpu in enumerate(gpus):
            log = out / f"{name}_shard{si}.log"
            p = run([PY, str(ROOT / "tools" / "eval_l8_ovmot.py"),
                     "--ckpt", ck, "--out", str(out / name),
                     "--gpu", str(gpu), "--shard", str(si),
                     "--num-shards", str(len(gpus)),
                     "--pbd-cache", args.pbd_cache], log)
            procs.append(p)
        for p in procs:
            p.wait()
        log = out / f"{name}_merge.log"
        p = run([PY, str(ROOT / "tools" / "eval_l8_ovmot.py"),
                 "--out", str(out / name), "--merge-only",
                 "--pbd-cache", args.pbd_cache], log)
        p.wait()
        print(f"[ovmot_full] {name} done: {ck}", flush=True)
    print("[ovmot_full] all done", flush=True)


if __name__ == "__main__":
    main()
