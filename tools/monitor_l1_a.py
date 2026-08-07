#!/usr/bin/env python
"""Single blocking memory-guard monitor for Stage L1-A (AGENTS.md compliant).

This process is the ONE blocking command that waits for the L1-A pipeline. It
does not expose agent-level polling: all checks happen inside this process.

Protection rules:
- If MemAvailable < low_gb for `low_streak` consecutive checks, terminate our
  own resume-safe cache processes (D-LA shards + D-CTRL), freeing RAM so the
  system OOM-killer does not take down the pipeline.
- Wait until MemAvailable >= recover_gb, then relaunch every missing shard.
- If the pipeline process dies before producing final_status.json, restart it.
- Exit when final_status.json contains a decision (pipeline finished).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/home/lwr/anaconda3/envs/locatemot/bin/python"
OCPY = "/home/lwr/anaconda3/envs/OC-SORT/bin/python"
LOG = os.path.join(ROOT, "outputs", "l1_a", "monitor.log")


def log(msg):
    line = f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def mem_avail_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    except Exception:
        return 999.0
    return 999.0


def pids_for(fragment):
    r = subprocess.run(["pgrep", "-f", fragment], capture_output=True, text=True)
    return [p for p in r.stdout.split() if p]


def running(fragment):
    return bool(pids_for(fragment))


def launch(cmd):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] launch: {' '.join(cmd)}\n")
    return subprocess.Popen(cmd, cwd=ROOT, stdout=open(LOG, "a"), stderr=subprocess.STDOUT)


def kill_fragment(fragment):
    for pid in pids_for(fragment):
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception:
            pass


def main():
    low_gb = float(os.environ.get("L1A_LOW_MEM_GB", "25"))
    recover_gb = float(os.environ.get("L1A_RECOVER_MEM_GB", "45"))
    low_streak_limit = int(os.environ.get("L1A_LOW_STREAK", "3"))
    check_interval = int(os.environ.get("L1A_CHECK_INTERVAL", "60"))

    cache_cmds = []
    for si, gpu in enumerate([3, 4, 5, 6, 7, 8]):
        cache_cmds.append((
            f"train_shard{si}",
            f"cache_dancetrack_locateanything.py --split train --shard {si}",
            [PY, "tools/cache_dancetrack_locateanything.py", "--split", "train",
             "--gpu", str(gpu), "--shard", str(si), "--num-shards", "6",
             "--query-id", "d1", "--protocol", "person", "--out", "outputs/l1_a"],
        ))
    for si, gpu in enumerate([3, 4, 5, 6, 7, 8]):
        cache_cmds.append((
            f"val_shard{si}",
            f"cache_dancetrack_locateanything.py --split val --shard {si}",
            [PY, "tools/cache_dancetrack_locateanything.py", "--split", "val",
             "--gpu", str(gpu), "--shard", str(si), "--num-shards", "6",
             "--query-id", "d1", "--protocol", "person", "--out", "outputs/l1_a"],
        ))
    cache_cmds.append((
        "calibration",
        "cache_dancetrack_locateanything.py --split calibration",
        [PY, "tools/cache_dancetrack_locateanything.py", "--split", "calibration",
         "--gpu", "9", "--query-id", "d1", "--protocol", "person", "--out", "outputs/l1_a"],
    ))
    cache_cmds.append((
        "dctrl_calibration",
        "cache_dancetrack_yolox.py --split calibration",
        [OCPY, "tools/cache_dancetrack_yolox.py", "--split", "calibration", "--gpu", "2"],
    ))

    pipeline_cmd = [PY, "tools/run_l1_a_pipeline.py", "--gpu", "4", "--ctrl-gpu", "9",
                    "--cache-gpus", "3,4,5,6,7,8"]
    pipeline_frag = "run_l1_a_pipeline.py"
    final_status = os.path.join(ROOT, "outputs", "l1_a", "final_status.json")

    log(f"monitor start: low={low_gb}GB recover={recover_gb}GB interval={check_interval}s")
    low_streak = 0
    suspended = False

    while True:
        if os.path.exists(final_status):
            import json
            try:
                st = json.load(open(final_status))
                if st.get("decision"):
                    log(f"pipeline finished: {st['decision']}")
                    break
            except Exception:
                pass

        if not running(pipeline_frag):
            log("pipeline not running; restarting")
            launch(pipeline_cmd)

        avail = mem_avail_gb()
        if avail < low_gb:
            low_streak += 1
            log(f"low memory: {avail:.0f}GB (streak {low_streak}/{low_streak_limit})")
            if low_streak >= low_streak_limit and not suspended:
                log("suspending our cache processes to protect the pipeline")
                kill_fragment("cache_dancetrack_locateanything.py")
                kill_fragment("cache_dancetrack_yolox.py")
                suspended = True
                low_streak = 0
        else:
            low_streak = 0

        if suspended and avail >= recover_gb:
            log(f"memory recovered ({avail:.0f}GB); relaunching cache shards")
            suspended = False

        if not suspended and avail >= recover_gb:
            for label, frag, cmd in cache_cmds:
                if not running(frag):
                    log(f"relaunch {label}")
                    launch(cmd)

        time.sleep(check_interval)

    log("monitor exit")


if __name__ == "__main__":
    main()
