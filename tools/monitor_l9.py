"""Stage L9 self-contained watcher (no agent-level polling).

Every CHECK_SECONDS:
  - inspect host RAM and GPU memory;
  - keep the desired number of TAO-PBD cache workers alive (auto-restart
    crashed workers, scale up when RAM frees);
  - write a status JSON for later reporting.

Launch once (detached):
  python tools/monitor_l9.py > outputs/l9/cache/monitor.log 2>&1 &
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
PY = "/home/lwr/anaconda3/envs/locatemot/bin/python"
CACHE_SCRIPT = ROOT / "tools" / "cache_l9_tao_pbd.py"
CACHE_ROOT = ROOT / "outputs" / "l9" / "cache" / "tao_val_pbd"
LOG_DIR = ROOT / "outputs" / "l9" / "cache"
STATUS = LOG_DIR / "cache_status.json"
NUM_SHARDS = 8
GPUS = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # re-checked at runtime
CHECK_SECONDS = 300
MIN_RAM_PER_WORKER = 12  # GB
MAX_GPUS = 4
BUSY_GPUS_FILE = LOG_DIR / "busy_gpus.json"


def mem_available_gb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 / 1024
    return 0.0


def gpu_mem_used():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20).stdout
    used = {}
    for line in out.strip().splitlines():
        idx, mu, mt = [int(x) for x in line.replace(" MiB", "").split(",")]
        used[idx] = (mu, mt)
    return used


def running_workers():
    out = subprocess.run(["pgrep", "-af", "cache_l9_tao_pbd.py"],
                         capture_output=True, text=True).stdout
    shards = {}
    for line in out.splitlines():
        if "--shard" not in line or "monitor_l9" in line:
            continue
        try:
            args = line.split("cache_l9_tao_pbd.py", 1)[1]
            shard = int(args.split("--shard")[1].split()[0])
            gpu = int(args.split("--gpu")[1].split()[0])
            shards[shard] = {"gpu": gpu, "pid": int(line.split()[0])}
        except (ValueError, IndexError):
            continue
    return shards


def launch_worker(shard, gpu):
    cmd = [PY, str(CACHE_SCRIPT), "--gpu", str(gpu), "--shard", str(shard),
           "--num-shards", str(NUM_SHARDS), "--cache-root", str(CACHE_ROOT)]
    log = LOG_DIR / f"tao_val_pbd_shard{shard}.log"
    with open(log, "a") as f:
        f.write(f"\n=== restart {time.ctime()} gpu={gpu} ===\n")
        f.flush()
    with open(log, "ab") as f:
        subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"[monitor] launched shard {shard} gpu {gpu}", flush=True)


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[monitor] start {time.ctime()}", flush=True)
    prev_avail = 999.0
    while True:
        avail = mem_available_gb()
        gpu_used = gpu_mem_used()
        workers = running_workers()
        busy_gpus = []
        if BUSY_GPUS_FILE.exists():
            try:
                busy_gpus = json.loads(BUSY_GPUS_FILE.read_text())
            except (ValueError, OSError):
                busy_gpus = []
        # hysteresis: pause only after two consecutive low-RAM checks;
        # (re)start only after two consecutive high-RAM checks
        low2 = avail < 6 and prev_avail < 10
        high2 = avail > 16 and prev_avail > 14
        if low2 and workers:
            # protect training / other tenants from OOM: pause cache
            for shard, info in list(workers.items()):
                subprocess.run(
                    ["pkill", "-f",
                     f"cache_l9_tao_pbd.py --gpu {info['gpu']} "
                     f"--shard {shard}"], capture_output=True)
                print(f"[monitor] paused cache shard {shard} "
                      f"(avail={avail:.0f}G)", flush=True)
            time.sleep(30)
            continue
        gpu_worker_count = {}
        for info in workers.values():
            gpu_worker_count[info["gpu"]] = \
                gpu_worker_count.get(info["gpu"], 0) + 1
        want = 0
        if high2 and avail > 55:
            want = 8
        elif high2 and avail > 28:
            want = 4
        elif high2 and avail > 20:
            want = 2
        elif high2 and avail > 16:
            want = 1
        workers_per_gpu = 1
        max_gpus = max(1, MAX_GPUS - len(busy_gpus))
        missing = [
            s for s in range(NUM_SHARDS)
            if s not in workers
            and not (CACHE_ROOT / f"shard{s}.done").exists()]
        our_gpus = set(gpu_worker_count.keys())
        candidate_gpus = [
            g for g in GPUS
            if g not in busy_gpus
            and (g in our_gpus
                 or gpu_used.get(g, (10 ** 9, 0))[0] < 500)]
        candidate_gpus.sort(key=lambda g: gpu_worker_count.get(g, 0))
        # restart up to `want` workers, at most `workers_per_gpu` per GPU,
        # using at most MAX_GPUS physical GPUs
        for shard in missing:
            if len(workers) >= want:
                break
            gpu = None
            for g in candidate_gpus:
                c = gpu_worker_count.get(g, 0)
                if c >= workers_per_gpu:
                    continue
                if c == 0 and len(our_gpus) >= max_gpus:
                    continue
                gpu = g
                break
            if gpu is None:
                break
            launch_worker(shard, gpu)
            gpu_worker_count[gpu] = gpu_worker_count.get(gpu, 0) + 1
            our_gpus.add(gpu)
            workers[shard] = {"gpu": gpu, "pid": -1}
        status = {
            "time": time.ctime(),
            "mem_available_gb": round(avail, 1),
            "gpu_mem": gpu_used,
            "workers": workers,
            "complete_frames": sum(1 for _ in
                                   CACHE_ROOT.glob("*/*/*/*.complete")),
        }
        STATUS.write_text(json.dumps(status, indent=2))
        print(f"[monitor] {time.ctime()} avail={avail:.0f}G "
              f"workers={len(workers)} complete={status['complete_frames']}",
              flush=True)
        prev_avail = avail
        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    main()
