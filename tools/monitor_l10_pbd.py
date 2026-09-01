"""Detached monitor for the L10 TAO-train crop-PBD cache.

Checks every --interval seconds: worker liveness, cache progress, shard
log errors, GPU memory.  Restarts dead workers (same shard/num-shards)
and appends a status line to outputs/l10/logs/pbd_monitor.log.
Exits when --target frames are cached.

Usage:
  python tools/monitor_l10_pbd.py --gpus 0,1,2,3 --workers 8 --target 18274
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
CACHE = ROOT / "outputs" / "l10" / "cache" / "tao_train_pbd"
LOG_DIR = ROOT / "outputs" / "l10" / "logs"
PY = "/home/lwr/anaconda3/envs/locatemot/bin/python"


def running_shards():
    out = set()
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(
                errors="ignore").replace("\0", " ")
        except Exception:
            continue
        if "cache_l9_tao_pbd.py" not in cmd:
            continue
        m = re.search(r"--shard (\d+)", cmd)
        if m:
            out.add(int(m.group(1)))
    return out


def complete():
    return sum(1 for _ in CACHE.glob("*/*/*/*.complete"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--target", type=int, default=18274)
    ap.add_argument("--interval", type=int, default=1200)
    args = ap.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_DIR / "pbd_monitor.log", "a")
    gpus = [int(x) for x in args.gpus.split(",")]
    while True:
        n = complete()
        shards = running_shards()
        missing = [s for s in range(args.workers) if s not in shards
                   and not (CACHE / f"shard{s}.done").exists()]
        # detect tracebacks in shard logs
        errs = []
        for s in range(args.workers):
            lp = LOG_DIR / f"tao_train_pbd_shard{s}.log"
            if lp.exists():
                txt = lp.read_text(errors="ignore")
                if "Traceback" in txt[-4000:]:
                    errs.append(s)
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
             "--format=csv,noheader"], capture_output=True, text=True)
        line = (f"{time.strftime('%H:%M:%S')} frames={n}/{args.target} "
                f"shards={sorted(shards)} missing={missing} errs={errs} "
                f"gpu={gpu.stdout.strip().splitlines()[:4]}")
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()
        if n >= args.target:
            print("[pbdmon] target reached", flush=True)
            return 0
        if missing or errs:
            print(f"[pbdmon] restarting missing/err shards: "
                  f"{missing + errs}", flush=True)
            for s in set(missing + errs):
                gpu = gpus[s % len(gpus)]
                cmd = [
                    PY, str(ROOT / "tools" / "cache_l9_tao_pbd.py"),
                    "--gpu", str(gpu), "--shard", str(s),
                    "--num-shards", str(args.workers),
                    "--data-dir", str(ROOT / "outputs" / "l10" / "data"
                                      / "tao_train"),
                    "--cache-root", str(CACHE),
                    "--gt-json",
                    "/data1/LWR/vranlee/SERVER_ONLY/avis/TAO/"
                    "TAO-download/TAO-Amodal/annotations/train.json",
                ]
                lp = LOG_DIR / f"tao_train_pbd_shard{s}.log"
                with open(lp, "ab") as f:
                    f.write(b"\n=== monitor restart ===\n")
                    subprocess.Popen(cmd, stdout=f,
                                     stderr=subprocess.STDOUT)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
