"""Stage L10: run all official evaluations for the final shared checkpoint.

1. Ordinary MOT (DanceTrack/BDD/MOT17/MOT20)
2. RMOT Refer-Dance (40 GT queries)
3. OVMOT TAO val (full-PBD TETA)
4. RMOT Refer-KITTI-V2 (861 official queries, 4 eval sequences)

Usage:
  python tools/eval_l10_all.py --ckpt outputs/l10/checkpoints/.../latest.pt \
      --out outputs/l10/trackeval/final --gpus 0,1,2,3
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
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--rmot-threshold-file", default=None)
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    procs = []
    for i, gpu in enumerate(gpus):
        domains = ["dance,bdd", "mot17,mot20"][i % 2]
        log = out / f"ordinary_gpu{gpu}.log"
        p = run([PY, str(ROOT / "tools" / "eval_l8_ordinary.py"),
                 "--ckpt", args.ckpt, "--out",
                 str(out / f"ordinary_gpu{gpu}"), "--gpu", str(gpu),
                 "--domains", domains], log)
        procs.append(p)
    rmot_thr = args.rmot_threshold_file or str(
        ROOT / "outputs" / "l9" / "calib" / "threshold_l9.json")
    p = run([PY, str(ROOT / "tools" / "eval_l8_rmot.py"),
             "--ckpt", args.ckpt, "--out", str(out / "rmot_dance"),
             "--gpu", str(gpus[0]), "--threshold-file", rmot_thr],
            out / "rmot_dance.log")
    procs.append(p)
    p = run([PY, str(ROOT / "tools" / "eval_l10_rmot_kitti.py"),
             "--ckpt", args.ckpt, "--out", str(out / "rmot_kitti"),
             "--gpu", str(gpus[1]), "--threshold-file", rmot_thr],
            out / "rmot_kitti.log")
    procs.append(p)
    for p in procs:
        p.wait()
    # OVMOT after the lighter evals
    p = run([PY, str(ROOT / "tools" / "eval_l8_ovmot.py"),
             "--ckpt", args.ckpt, "--out", str(out / "ovmot"),
             "--gpu", str(gpus[2]),
             "--pbd-cache",
             str(ROOT / "outputs" / "l9" / "cache" / "tao_val_pbd")],
            out / "ovmot.log")
    p.wait()
    print("[eval_l10] all done", flush=True)


if __name__ == "__main__":
    main()
