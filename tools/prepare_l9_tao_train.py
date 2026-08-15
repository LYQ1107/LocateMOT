"""Stage L9: prepare TAO train OVMOT stream (DLA dets -> CLIP pkl -> PBD).

Sequential, resumable driver.  Waits for enough host RAM before the DLA
detection step so it never fights the TAO-val PBD cache workers.

Usage:
  python tools/prepare_l9_tao_train.py --gpus 4,6
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
PY_LM = "/home/lwr/anaconda3/envs/locatemot/bin/python"
PY_MASA = "/home/lwr/anaconda3/envs/masaenv/bin/python"
DETS = ROOT / "outputs" / "l9" / "data" / "tao_train_dets"
PKLS = ROOT / "outputs" / "l9" / "data" / "tao_train"
PBD = ROOT / "outputs" / "l9" / "cache" / "tao_train_pbd"
TRAIN_GT = ("/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/TAO-download/"
            "TAO-Amodal/annotations/train.json")
MIN_RAM_DLA_GB = 35


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def run(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as f:
        f.write(f"\n=== {time.ctime()} {' '.join(cmd)} ===\n".encode())
        f.flush()
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"step failed rc={rc}: {cmd}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="4,6")
    ap.add_argument("--skip-dets", action="store_true")
    ap.add_argument("--skip-pkls", action="store_true")
    ap.add_argument("--skip-pbd", action="store_true")
    args = ap.parse_args()
    log_dir = ROOT / "outputs" / "l9" / "cache"

    if not args.skip_dets:
        n_det = sum(1 for _ in DETS.glob("*/*/*/*.pth"))
        if n_det < 1000:
            while mem_available_gb() < MIN_RAM_DLA_GB:
                print(f"[l9prep] wait RAM {mem_available_gb():.0f}G < "
                      f"{MIN_RAM_DLA_GB}G", flush=True)
                time.sleep(300)
            run([PY_MASA, str(ROOT / "tools" / "generate_l7_tao_train_dets.py"),
                 "--gpus", args.gpus, "--out", str(DETS)],
                log_dir / "tao_train_dets.log")
        else:
            print(f"[l9prep] DLA dets already present ({n_det} files)",
                  flush=True)

    if not args.skip_pkls:
        index = PKLS / "index.json"
        if not index.exists():
            run([PY_LM, str(ROOT / "tools" / "build_l7_tao.py"),
                 "--split", "train", "--gpus", args.gpus,
                 "--out", str(PKLS), "--dets-root", str(DETS)],
                log_dir / "tao_train_pkls.log")
        else:
            print("[l9prep] TAO train pkls already present", flush=True)

    if not args.skip_pbd:
        n_pbd = sum(1 for _ in PBD.glob("*/*/*/*.complete"))
        if n_pbd < 1000:
            run([PY_LM, str(ROOT / "tools" / "cache_l9_tao_pbd.py"),
                 "--gpu", args.gpus.split(",")[0],
                 "--shard", "0", "--num-shards", "1",
                 "--gt-json", TRAIN_GT,
                 "--data-dir", str(PKLS),
                 "--cache-root", str(PBD)],
                log_dir / "tao_train_pbd.log")
        else:
            print(f"[l9prep] TAO train PBD cache present ({n_pbd} frames)",
                  flush=True)
    print("[l9prep] done", flush=True)


if __name__ == "__main__":
    main()
