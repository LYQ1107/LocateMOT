#!/usr/bin/env python
"""Stage L1-A end-to-end pipeline: wait caches -> train -> track -> evaluate.

Run as one blocking command (AGENTS.md long-experiment policy):
  python tools/run_l1_a_pipeline.py --gpu 3 --ctrl-gpu 9
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = "/home/lwr/anaconda3/envs/locatemot/bin/python"
OCPY = "/home/lwr/anaconda3/envs/OC-SORT/bin/python"
DLA_CACHE = "/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1A/cache_dla"
CTRL_CACHE = "/data3/testdata/vranlee/.MOTSynth.partial/LocateMOT_L1A/detections_ctrl"


def expected_frames(split):
    cfg = json.load(open(os.path.join(ROOT, "configs", "data", f"l1_a_dancetrack_{split}.json")))
    return sum(v["frames"] for v in cfg["videos"])


def dla_done(split):
    import configparser
    n = 0
    cfg = json.load(open(os.path.join(ROOT, "configs", "data", f"l1_a_dancetrack_{split}.json")))
    for v in cfg["videos"]:
        vid = v["video_id"]
        d = os.path.join(DLA_CACHE, "dancetrack", vid)
        if not os.path.isdir(d):
            continue
        for fdir in os.listdir(d):
            if os.path.exists(os.path.join(d, fdir, "person.complete")):
                n += 1
    return n


def ctrl_done(split):
    n = 0
    cfg = json.load(open(os.path.join(ROOT, "configs", "data", f"l1_a_dancetrack_{split}.json")))
    for v in cfg["videos"]:
        d = os.path.join(CTRL_CACHE, v["video_id"])
        if os.path.isdir(d):
            n += len(os.listdir(d))
    return n


def ctrl_running(split):
    import subprocess
    r = subprocess.run(["pgrep", "-f", f"cache_dancetrack_yolox.py --split {split}"],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def wait_for(label, fn, expected, timeout_hours=48, stall_minutes=45, on_stall=None):
    t0 = time.time()
    last = -1
    last_t = time.time()
    while True:
        cur = fn()
        if cur >= expected:
            print(f"[pipeline] {label} complete: {cur}/{expected}", flush=True)
            return True
        if cur != last:
            last, last_t = cur, time.time()
        elif time.time() - last_t > stall_minutes * 60:
            print(f"[pipeline] WARNING {label} stalled at {cur}/{expected}", flush=True)
            if on_stall is not None:
                on_stall()
                on_stall = None
        elapsed = time.time() - t0
        if elapsed > timeout_hours * 3600:
            print(f"[pipeline] TIMEOUT {label}: {cur}/{expected}", flush=True)
            return False
        if int(elapsed) % 600 == 0:
            print(f"[pipeline] waiting {label}: {cur}/{expected} ({elapsed/3600:.1f}h)", flush=True)
        time.sleep(120)


def run(cmd):
    print(f"[pipeline] run: {cmd}", flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}")


def launch_dla_shards(split, gpus):
    nshards = min(len(gpus), 6)
    procs = []
    import subprocess as sp
    running = set()
    r = sp.run(["pgrep", "-f", f"cache_dancetrack_locateanything.py --split {split} --shard"],
               capture_output=True, text=True)
    for line in r.stdout.splitlines():
        try:
            cmdline = open(f"/proc/{line.strip()}/cmdline", "rb").read().replace(b"\0", b" ").decode()
            for part in cmdline.split():
                if part.startswith("--shard"):
                    continue
            # parse --shard N from cmdline
            import re
            m = re.search(r"--shard (\d+)", cmdline)
            if m:
                running.add(int(m.group(1)))
        except Exception:
            pass
    for si, gpu in enumerate(gpus[:nshards]):
        if si in running:
            continue
        p = subprocess.Popen([
            PY, "tools/cache_dancetrack_locateanything.py",
            "--split", split, "--gpu", str(gpu), "--shard", str(si),
            "--num-shards", str(nshards), "--query-id", "d1",
            "--protocol", "person", "--out", "outputs/l1_a",
        ], cwd=ROOT)
        procs.append(p)
    if not procs:
        print(f"[pipeline] all D-LA {split} shards already running; nothing to launch", flush=True)
    return procs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=3)
    ap.add_argument("--ctrl-gpu", type=int, default=9)
    ap.add_argument("--cache-gpus", default="3,4,5,6,7,8")
    ap.add_argument("--skip-cache-wait", action="store_true")
    args = ap.parse_args()
    cache_gpus = [int(x) for x in args.cache_gpus.split(",")]

    if not args.skip_cache_wait:
        # monitor_l1_a.py owns shard relaunch; the pipeline only waits.
        wait_for("D-LA cache train", lambda: dla_done("train"), expected_frames("train"))

    # launch val cache if not complete (train/calibration done by now)
    if dla_done("val") < expected_frames("val"):
        procs = launch_dla_shards("val", cache_gpus)
        wait_for("D-LA cache val", lambda: dla_done("val"), expected_frames("val"))
        for p in procs:
            p.wait()
        print("[pipeline] val cache launch finished", flush=True)

    # train temporal modules
    ckpt = "outputs/l1_a/checkpoints/temporal/best.pt"
    if not os.path.exists(os.path.join(ROOT, ckpt)):
        run([PY, "tools/train_l1_a_trajectory.py", "--gpu", str(args.gpu)])

    # val tracker runs (D-LA main path first)
    run([PY, "tools/run_l1_a_tracker.py", "--variants", "T0,T1,T2,T3,T4,T5,T6",
         "--split", "val", "--gpu", str(args.gpu), "--protocol", "dla",
         "--temporal-ckpt", ckpt])

    # official TrackEval + stratified analysis
    run([PY, "tools/run_l1_a_trackeval.py", "--protocol", "dla", "--split", "val"])
    run([PY, "tools/evaluate_l1_a.py", "--split", "val", "--protocols", "dla"])

    # D-CTRL completeness + evaluation (secondary protocol; resume-safe)
    for split in ("calibration", "train", "val"):
        exp = expected_frames(split)
        if ctrl_done(split) < exp:
            if ctrl_running(split):
                print(f"[pipeline] D-CTRL {split} already running; waiting", flush=True)
                while ctrl_done(split) < exp and ctrl_running(split):
                    time.sleep(120)
            if ctrl_done(split) < exp:
                run([OCPY, "tools/cache_dancetrack_yolox.py", "--split", split,
                     "--gpu", str(args.ctrl_gpu)])
    run([PY, "tools/run_l1_a_tracker.py", "--variants", "T0,T1",
         "--split", "val", "--gpu", str(args.gpu), "--protocol", "ctrl"])
    run([PY, "tools/run_l1_a_trackeval.py", "--protocol", "ctrl", "--split", "val"])
    run([PY, "tools/evaluate_l1_a.py", "--split", "val", "--protocols", "dla,ctrl"])

    # calibration tracker runs (best-effort; thresholds stay frozen defaults)
    if dla_done("calibration") >= expected_frames("calibration"):
        run([PY, "tools/run_l1_a_tracker.py", "--variants", "T0,T1,T2,T3,T4,T5,T6",
             "--split", "calibration", "--gpu", str(args.gpu), "--protocol", "dla",
             "--temporal-ckpt", ckpt])
        run([PY, "tools/run_l1_a_tracker.py", "--variants", "T0,T1",
             "--split", "calibration", "--gpu", str(args.gpu), "--protocol", "ctrl"])
    else:
        print("[pipeline] calibration D-LA cache incomplete; skipping calibration tracker runs "
              "(frozen default thresholds used)", flush=True)

    # reports
    run([PY, "tools/write_l1_a_reports.py"])
    print("[pipeline] complete", flush=True)


if __name__ == "__main__":
    main()
