"""Detached monitor for the L10 4-GPU joint training.

Checks every --interval seconds: trainer process liveness, training log
errors, step progress from learning_curve.json.  Relaunches
tools/run_l10_train.sh (which resumes from the latest checkpoint) if the
trainer died before --target-steps.  Exits when the target is reached.

Usage:
  python tools/monitor_l10_train.py --target-steps 30000
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
OUT = ROOT / "outputs" / "l10" / "checkpoints" / "uidm_l10_main"
LOG = ROOT / "outputs" / "l10" / "logs" / "uidm_l10_main_train.log"
MON_LOG = ROOT / "outputs" / "l10" / "logs" / "train_monitor.log"


def running():
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().decode(
                errors="ignore").replace("\0", " ")
        except Exception:
            continue
        if "train_l9_uidm.py" in cmd:
            return True
    return False


def step(out_dir):
    try:
        ck = json.loads((out_dir / "learning_curve.json").read_text())
        return max(int(r.get("step", 0)) for r in ck) if ck else 0
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-steps", type=int, default=30000)
    ap.add_argument("--interval", type=int, default=1200)
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument("--log", default=str(LOG))
    ap.add_argument("--script",
                    default=str(ROOT / "tools" / "run_l10_train.sh"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    log_path = Path(args.log)
    MON_LOG.parent.mkdir(parents=True, exist_ok=True)
    f = open(MON_LOG, "a")
    while True:
        s = step(out_dir)
        alive = running()
        err = ""
        if log_path.exists():
            tail = log_path.read_text(errors="ignore")[-6000:]
            if "Traceback" in tail or "CUDA out of memory" in tail:
                err = "log-error"
        line = (f"{time.strftime('%H:%M:%S')} step={s} "
                f"alive={alive} err={err}")
        print(line, flush=True)
        f.write(line + "\n")
        f.flush()
        if s >= args.target_steps:
            print("[trainmon] target reached", flush=True)
            return 0
        if (not alive) or err:
            print("[trainmon] restarting training", flush=True)
            with open(log_path, "ab") as lf:
                lf.write(b"\n=== monitor restart ===\n")
                subprocess.Popen(
                    ["bash", str(args.script)],
                    stdout=lf, stderr=subprocess.STDOUT)
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
