"""Stage L9: run all remaining official evals after the PBD cache.

Waits until the TAO val PBD cache is complete, then:
  1) OVMOT full-PBD TETA for L8-B2, L8-B1, L9-v5 (4 shards each);
  2) ordinary + RMOT eval-time ablations (identity/semantic) for L9-v5.

Usage:
  python tools/finalize_l9_evals.py --gpus 1,2,3,4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
PY = "/home/lwr/anaconda3/envs/locatemot/bin/python"
PBD_CACHE = ROOT / "outputs" / "l9" / "cache" / "tao_val_pbd"
EXPECTED_FRAMES = 36375
OUT = ROOT / "outputs" / "l9" / "trackeval" / "final"

CKPTS = {
    "l8_b2": "outputs/l8/checkpoints/uidm_l8_v2/latest.pt",
    "l8_b1": "outputs/l8/checkpoints/uidm_l8_final/latest.pt",
    "l9_v5": "outputs/l9/checkpoints/uidm_l9_main/latest.pt",
}


def complete_frames():
    return sum(1 for _ in PBD_CACHE.glob("*/*/*/*.complete"))


def run(cmd, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as f:
        f.write(f"\n=== {time.ctime()} {' '.join(cmd)} ===\n".encode())
        f.flush()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="1,2,3,4")
    ap.add_argument("--skip-ovmot", action="store_true")
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()
    gpus = [int(x) for x in args.gpus.split(",")]
    OUT.mkdir(parents=True, exist_ok=True)
    while complete_frames() < EXPECTED_FRAMES:
        print(f"[finalize] cache {complete_frames()}/{EXPECTED_FRAMES} "
              f"waiting...", flush=True)
        time.sleep(600)
    print("[finalize] cache complete", flush=True)

    if not args.skip_ovmot:
        ckpt_list = ",".join(str(ROOT / p) for p in CKPTS.values())
        p = run([PY, str(ROOT / "tools" / "eval_l9_ovmot_full.py"),
                 "--gpus", ",".join(map(str, gpus)),
                 "--ckpts", ckpt_list,
                 "--out", str(OUT / "ovmot_full")],
                OUT / "ovmot_full.log")
        p.wait()
    if not args.skip_ablation:
        ck = str(ROOT / CKPTS["l9_v5"])
        for ab in ["identity", "semantic"]:
            p = run([PY, str(ROOT / "tools" / "eval_l8_ordinary.py"),
                     "--ckpt", ck, "--out", str(OUT / f"ord_{ab}"),
                     "--gpu", str(gpus[0]), "--domains", "dance,bdd",
                     "--ablation", ab], OUT / f"ord_{ab}.log")
            p.wait()
            p = run([PY, str(ROOT / "tools" / "eval_l8_rmot.py"),
                     "--ckpt", ck, "--out", str(OUT / f"rmot_{ab}"),
                     "--gpu", str(gpus[1]), "--ablation", ab,
                     "--threshold-file",
                     str(ROOT / "outputs" / "l9" / "calib"
                         / "threshold_l9_v5.json")],
                    OUT / f"rmot_{ab}.log")
            p.wait()
    print("[finalize] all done", flush=True)


if __name__ == "__main__":
    main()
