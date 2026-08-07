#!/usr/bin/env python
"""Single blocking memory-guard monitor for Stage L1-A (AGENTS.md compliant).

This process is the ONE blocking command that waits for the L1-A pipeline. It
does not expose agent-level polling: all checks happen inside this process.

Design:
- The monitor records the child PIDs it launches, so it can never launch a
  duplicate shard even if pgrep behaves unexpectedly.
- If MemAvailable < low_gb for `low_streak` consecutive checks, it terminates
  our own resume-safe cache processes to keep the system away from OOM.
- After memory recovers to recover_gb, it relaunches every shard that is not
  running. Val shards are only relaunched after the train cache is complete.
- If the pipeline dies before final_status.json exists, it is restarted.
- Exit when final_status.json contains a decision.
"""
from __future__ import annotations

import json
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


def pgrep_pids(fragment):
    r = subprocess.run(["pgrep", "-f", fragment], capture_output=True, text=True)
    return [p for p in r.stdout.split() if p]


def kill_all_ours():
    # bracket trick avoids matching this command's own cmdline
    subprocess.run(["pkill", "-f", "[c]ache_dancetrack_locateanything.py"],
                   capture_output=True)
    subprocess.run(["pkill", "-f", "[c]ache_dancetrack_yolox.py"],
                   capture_output=True)


def launch(cmd):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(f"[{datetime.now().strftime('%m-%d %H:%M:%S')}] launch: {' '.join(cmd)}\n")
    return subprocess.Popen(cmd, cwd=ROOT, stdout=open(LOG, "a"), stderr=subprocess.STDOUT)


def main():
    low_gb = float(os.environ.get("L1A_LOW_MEM_GB", "35"))
    recover_gb = float(os.environ.get("L1A_RECOVER_MEM_GB", "50"))
    low_streak_limit = int(os.environ.get("L1A_LOW_STREAK", "2"))
    check_interval = int(os.environ.get("L1A_CHECK_INTERVAL", "60"))

    train_cmds = []
    for si, gpu in enumerate([3, 4, 5, 6, 7, 8]):
        train_cmds.append((
            f"train_shard{si}",
            f"cache_dancetrack_locateanything.py --split train --shard {si}",
            [PY, "tools/cache_dancetrack_locateanything.py", "--split", "train",
             "--gpu", str(gpu), "--shard", str(si), "--num-shards", "6",
             "--query-id", "d1", "--protocol", "person", "--out", "outputs/l1_a"],
        ))
    val_cmds = []
    for si, gpu in enumerate([3, 4, 5, 6, 7, 8]):
        val_cmds.append((
            f"val_shard{si}",
            f"cache_dancetrack_locateanything.py --split val --shard {si}",
            [PY, "tools/cache_dancetrack_locateanything.py", "--split", "val",
             "--gpu", str(gpu), "--shard", str(si), "--num-shards", "6",
             "--query-id", "d1", "--protocol", "person", "--out", "outputs/l1_a"],
        ))
    calib_cmd = ("calibration",
                 "cache_dancetrack_locateanything.py --split calibration",
                 [PY, "tools/cache_dancetrack_locateanything.py", "--split", "calibration",
                  "--gpu", "9", "--query-id", "d1", "--protocol", "person",
                  "--out", "outputs/l1_a"])
    dctrl_cmd = ("dctrl_calibration",
                 "cache_dancetrack_yolox.py --split calibration",
                 [OCPY, "tools/cache_dancetrack_yolox.py", "--split", "calibration",
                  "--gpu", "2"])
    base_cmds = train_cmds + [calib_cmd, dctrl_cmd]

    pipeline_cmd = [PY, "tools/run_l1_a_pipeline.py", "--gpu", "4", "--ctrl-gpu", "9",
                    "--cache-gpus", "3,4,5,6,7,8"]
    pipeline_frag = "run_l1_a_pipeline.py"
    final_status = os.path.join(ROOT, "outputs", "l1_a", "final_status.json")
    train_done_target = None
    cfg = json.load(open(os.path.join(ROOT, "configs/data/l1_a_dancetrack_train.json")))
    train_done_target = sum(v["frames"] for v in cfg["videos"])

    children = {}
    log(f"monitor start: low={low_gb}GB recover={recover_gb}GB "
        f"streak={low_streak_limit} interval={check_interval}s")
    low_streak = 0
    suspended = False
    first_cycle = True

    def alive(label, frag):
        proc = children.get(label)
        if proc is not None and proc.poll() is None:
            return True
        return bool(pgrep_pids(frag))

    while True:
        if os.path.exists(final_status):
            try:
                st = json.load(open(final_status))
                if st.get("decision"):
                    log(f"pipeline finished: {st['decision']}")
                    break
            except Exception:
                pass

        if not pgrep_pids(pipeline_frag):
            log("pipeline not running; restarting")
            children["pipeline"] = launch(pipeline_cmd)

        avail = mem_avail_gb()
        if avail < low_gb:
            low_streak += 1
            log(f"low memory: {avail:.0f}GB (streak {low_streak}/{low_streak_limit})")
            if low_streak >= low_streak_limit and not suspended:
                log("suspending our cache processes to protect the pipeline")
                kill_all_ours()
                children = {}
                suspended = True
                low_streak = 0
        else:
            low_streak = 0

        if suspended and avail >= recover_gb:
            log(f"memory recovered ({avail:.0f}GB); relaunching cache shards")
            suspended = False

        if not suspended and avail >= recover_gb:
            for label, frag, cmd in base_cmds:
                if not alive(label, frag):
                    log(f"relaunch {label}")
                    children[label] = launch(cmd)
            train_done = 0
            tr_root = "/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1A/cache_dla/dancetrack"
            for vid in os.listdir(tr_root):
                vdir = os.path.join(tr_root, vid)
                if not os.path.isdir(vdir):
                    continue
                for fdir in os.listdir(vdir):
                    if os.path.exists(os.path.join(vdir, fdir, "person.complete")):
                        train_done += 1
            if train_done >= train_done_target:
                for label, frag, cmd in val_cmds:
                    if not alive(label, frag):
                        log(f"relaunch {label}")
                        children[label] = launch(cmd)

        time.sleep(check_interval)

    log("monitor exit")


if __name__ == "__main__":
    main()
