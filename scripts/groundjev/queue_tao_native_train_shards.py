#!/usr/bin/env python3
"""Resumable native TAO TRAIN export owner.

The owner waits for the protected G2 Clean terminal receipt, then uses only
idle physical GPUs 4 and 6.  It starts an owned GroundAnything service pair
per shard, streams the shard with the GT-free TRAIN contract, stops the pair,
and finally invokes the strict shard merger.  Existing compute applications
are treated as unavailable; this owner never co-resides with or terminates a
foreign process.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = Path("/data2/user/LocateMOT_GroundJEV")
G2_ROOT = Path(
    "/data3/user/LocateMOT-runs/GroundAnythingG2/"
    "G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final"
)
SHARD_ROOT = RUNTIME_ROOT / "outputs/groundjev/training_data/tao_native_ovmot_train_shards_v2"
PARENT_MANIFEST = RUNTIME_ROOT / "outputs/groundjev/training_data/tao_native_ovmot_train_scope_v1/manifest.json"
MERGED_OUT = RUNTIME_ROOT / "outputs/groundjev/training_data/tao_native_ovmot_train_merged_v1"
G2_MANIFEST_SHA = "f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71"
G2_STEPS = 18227
GPU_POOL = (4, 6)
GROUND_PYTHON = RUNTIME_ROOT / "external/GroundAnything/.venv-serve/bin/python"
STREAM_PYTHON = RUNTIME_ROOT / ".venv-onejev-py310/bin/python"
GROUND_SERVICE = RUNTIME_ROOT / "tools/groundjev/serve_groundanything.py"
FEATURE_SERVICE = RUNTIME_ROOT / "tools/groundjev/serve_native_features.py"
STREAM_SCRIPT = RUNTIME_ROOT / "tools/groundjev/stream_tao_native_candidates.py"
MERGE_SCRIPT = PROJECT_ROOT / "scripts/groundjev/merge_tao_native_shards.py"
GROUND_TRAIN_PYTHON = RUNTIME_ROOT / ".venv-ground-train-py312/bin/python"
G2_EXPORT_SCRIPT = RUNTIME_ROOT / "tools/groundjev/g2/export_g2_checkpoint.py"
G2_NATIVE_MODEL = RUNTIME_ROOT / "models/GroundAnything-DLM-G2-CLEAN-NATIVE-TAO-V1"
MODEL = G2_NATIVE_MODEL
G2_NATIVE_EXPORT_RECEIPT = SHARD_ROOT / "g2_native_bundle_export_receipt.json"
G2_ADAPTER = G2_ROOT / "adapter_projector_final.pt"
BASE_PORT = 8300
STATE_LOCK = threading.Lock()


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
    print(json.dumps({k: state.get(k) for k in ("status", "current_stage", "completed_shards", "failed")}, ensure_ascii=False), flush=True)


def g2_status() -> tuple[str, dict | None]:
    receipt = read_json(G2_ROOT / "final_receipt.json")
    if receipt is None:
        return "waiting", None
    valid = (
        receipt.get("status") == "G2_EXACT_COVERAGE_TRAINING_COMPLETE"
        and int(receipt.get("steps", -1)) == G2_STEPS
        and receipt.get("manifest_sha256") == G2_MANIFEST_SHA
    )
    return ("complete", receipt) if valid else ("failed", receipt)


def gpu_probe(gpu: int) -> dict:
    apps = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    free = int(subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0])
    pids = [line.strip() for line in apps.splitlines() if line.strip()]
    return {"gpu": gpu, "compute_apps": pids, "free_memory_MiB": free,
            "ready": not pids and free >= 32768}


def stop_group(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass


def check_ports(ports: tuple[int, int]) -> None:
    for port in ports:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))


def wait_services(ground_process: subprocess.Popen, feature_process: subprocess.Popen,
                  ground_port: int, feature_receipt: Path) -> None:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if ground_process.poll() is not None or feature_process.poll() is not None:
            raise RuntimeError("owned native service exited before readiness")
        try:
            with urlopen(f"http://127.0.0.1:{ground_port}/v1/models", timeout=5) as response:
                models = json.load(response).get("data", [])
            if any(item.get("id") == "groundinganything" for item in models) and feature_receipt.is_file():
                return
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise TimeoutError("owned native service readiness timeout")


def run_shard(shard_index: int, gpu: int, shard_count: int) -> dict:
    shard_dir = SHARD_ROOT / f"shard_{shard_index:02d}_of_{shard_count:02d}"
    manifest = shard_dir / "manifest.json"
    output = shard_dir / "native_candidates"
    receipt = output / "receipt.json"
    if receipt.is_file():
        existing = read_json(receipt)
        if existing and existing.get("status") == "COMPLETE_LEGAL_NATIVE_VOCABULARY_AND_ROI_STREAM":
            return existing
    if output.exists():
        raise FileExistsError(f"refusing to overwrite partial shard output: {output}")
    ground_port = BASE_PORT + shard_index * 4
    feature_port = ground_port + 1
    check_ports((ground_port, feature_port))
    service_dir = shard_dir / f"services_gpu{gpu}"
    service_dir.mkdir()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    ground_log = (service_dir / "ground.log").open("w", encoding="utf-8")
    feature_log = (service_dir / "features.log").open("w", encoding="utf-8")
    ground_process = feature_process = None
    try:
        ground_process = subprocess.Popen([
            str(GROUND_PYTHON), str(GROUND_SERVICE), "--model", str(MODEL),
            "--port", str(ground_port), "--mem-fraction-static", ".35",
            "--max-queued-requests", "2", "--out", str(service_dir / "ground"),
        ], cwd=RUNTIME_ROOT, env=env, stdout=ground_log, stderr=subprocess.STDOUT, start_new_session=True)
        feature_process = subprocess.Popen([
            str(GROUND_PYTHON), str(FEATURE_SERVICE), "--port", str(feature_port),
            "--receipt", str(service_dir / "feature_receipt.json"),
        ], cwd=RUNTIME_ROOT, env=env, stdout=feature_log, stderr=subprocess.STDOUT, start_new_session=True)
        wait_services(ground_process, feature_process, ground_port, service_dir / "feature_receipt.json")
        command = [
            str(STREAM_PYTHON), str(STREAM_SCRIPT), "--manifest", str(manifest), "--out", str(output),
            "--allow-training-scope", "--grounding-url", f"http://127.0.0.1:{ground_port}/v1",
            "--native-feature-url", f"http://127.0.0.1:{feature_port}",
        ]
        log_path = shard_dir / "native_stream.log"
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=RUNTIME_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"native shard {shard_index} exited {result.returncode}; see {log_path}")
        result_receipt = read_json(receipt)
        if result_receipt is None or result_receipt.get("status") != "COMPLETE_LEGAL_NATIVE_VOCABULARY_AND_ROI_STREAM":
            raise RuntimeError(f"native shard {shard_index} did not produce a terminal receipt")
        return result_receipt
    finally:
        stop_group(feature_process)
        stop_group(ground_process)
        feature_log.close()
        ground_log.close()


def shard_count() -> int:
    summary = read_json(SHARD_ROOT / "shards_receipt.json")
    if summary is None:
        raise FileNotFoundError(SHARD_ROOT / "shards_receipt.json")
    return int(summary["num_shards"])


def completed_shards(count: int) -> dict[str, dict]:
    result = {}
    for index in range(count):
        receipt = SHARD_ROOT / f"shard_{index:02d}_of_{count:02d}" / "native_candidates" / "receipt.json"
        value = read_json(receipt)
        if value and value.get("status") == "COMPLETE_LEGAL_NATIVE_VOCABULARY_AND_ROI_STREAM":
            result[str(index)] = value
    return result


def prepare_g2_native_bundle(state: dict, owner: Path) -> dict:
    """Materialize a private G2 terminal bundle before any TAO inference.

    The formal G2 trainer stores only the trainable adapter.  The TAO stream
    therefore needs a merged deployment copy; using the published G0 bundle
    here would make the resulting candidates impossible to attribute to G2.
    This copy is separate from both the G2 clean deployment owner and the G1
    bundle, and the source model is never overwritten.
    """
    existing = read_json(G2_NATIVE_EXPORT_RECEIPT)
    if existing is not None:
        if (existing.get("status") != "G2_NATIVE_MERGED_BUNDLE_EXPORTED"
                or int(existing.get("step", -1)) != G2_STEPS
                or existing.get("manifest_sha256") != G2_MANIFEST_SHA
                or existing.get("output_bundle") != str(G2_NATIVE_MODEL.resolve())
                or existing.get("native_tensor_count") != 569
                or existing.get("source_bundle_overwritten")):
            raise RuntimeError("existing G2 native TAO export receipt failed its contract")
        if not (G2_NATIVE_MODEL / "model.safetensors").is_file():
            raise RuntimeError("G2 native TAO receipt exists but its model bundle is missing")
        state["g2_native_bundle"] = existing
        save(owner, state)
        return existing
    if G2_NATIVE_MODEL.exists():
        raise FileExistsError(f"refusing to overwrite partial G2 native TAO bundle: {G2_NATIVE_MODEL}")
    if not G2_ADAPTER.is_file():
        raise FileNotFoundError(f"G2 terminal adapter missing: {G2_ADAPTER}")
    state.update(status="EXPORTING_G2_NATIVE_TAO_BUNDLE", current_stage="EXPORTING_G2_NATIVE_TAO_BUNDLE")
    save(owner, state)
    export_log = SHARD_ROOT / "G2_NATIVE_BUNDLE_EXPORT.log"
    command = [
        str(GROUND_TRAIN_PYTHON), str(G2_EXPORT_SCRIPT),
        "--adapter", str(G2_ADAPTER),
        "--out", str(G2_NATIVE_MODEL),
        "--receipt", str(G2_NATIVE_EXPORT_RECEIPT),
        "--step", str(G2_STEPS),
        "--manifest-sha256", G2_MANIFEST_SHA,
    ]
    with export_log.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=RUNTIME_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"G2 native TAO bundle export exited {result.returncode}; see {export_log}")
    exported = read_json(G2_NATIVE_EXPORT_RECEIPT)
    if (exported is None or exported.get("status") != "G2_NATIVE_MERGED_BUNDLE_EXPORTED"
            or int(exported.get("step", -1)) != G2_STEPS
            or exported.get("manifest_sha256") != G2_MANIFEST_SHA
            or exported.get("native_tensor_count") != 569
            or exported.get("source_bundle_overwritten")):
        raise RuntimeError("G2 native TAO bundle export did not pass its terminal contract")
    state["g2_native_bundle"] = exported
    save(owner, state)
    return exported


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(GPU_POOL))
    parser.add_argument("--shard-root", type=Path, default=SHARD_ROOT)
    parser.add_argument("--parent-manifest", type=Path, default=PARENT_MANIFEST)
    parser.add_argument("--merged-out", type=Path, default=MERGED_OUT)
    args = parser.parse_args()
    if tuple(args.gpus) != GPU_POOL:
        raise ValueError(f"this owner is restricted to physical GPUs {GPU_POOL}")
    if args.shard_root.resolve() != SHARD_ROOT.resolve() or args.parent_manifest.resolve() != PARENT_MANIFEST.resolve():
        raise ValueError("custom roots are disabled for the registered TAO TRAIN owner")
    count = shard_count()
    owner = args.shard_root / "queue_owner_state.json"
    old = read_json(owner)
    if old and str(old.get("status", "")).endswith(("_COMPLETE", "_FAILED")):
        raise FileExistsError(f"terminal TAO owner already exists: {owner}")
    if old and old.get("owner_pid") and Path(f"/proc/{old['owner_pid']}").exists():
        raise FileExistsError(f"TAO owner already live: {old['owner_pid']}")
    state = {
        "status": "WAITING_G2_CLEAN_SUCCESS",
        "current_stage": "WAITING_G2_CLEAN_SUCCESS",
        "owner_pid": os.getpid(),
        "g2_root": str(G2_ROOT),
        "shard_root": str(args.shard_root),
        "parent_manifest": str(args.parent_manifest),
        "parent_manifest_sha256": read_json(args.shard_root / "shards_receipt.json").get("parent_manifest_sha256"),
        "physical_gpus": list(args.gpus),
        "shards": count,
        "completed_shards": completed_shards(count),
        "failed": None,
        "other_active_jobs_untouched": True,
        "native_ovmot_training_rows": False,
        "model": str(G2_NATIVE_MODEL),
        "g2_native_bundle_receipt": str(G2_NATIVE_EXPORT_RECEIPT),
    }
    save(owner, state)
    active: dict[int, tuple[int, Future]] = {}
    executor = ThreadPoolExecutor(max_workers=len(args.gpus))
    try:
        while True:
            g2, evidence = g2_status()
            if g2 == "failed":
                state.update(status="TAO_NATIVE_EXPORT_G2_DEPENDENCY_FAILED", current_stage="FAILED", failed=evidence)
                save(owner, state)
                return 2
            done = completed_shards(count)
            state["completed_shards"] = done
            for gpu, (index, future) in list(active.items()):
                if not future.done():
                    continue
                del active[gpu]
                try:
                    done[str(index)] = future.result()
                except Exception as error:
                    state.update(status="TAO_NATIVE_EXPORT_FAILED", current_stage="FAILED",
                                 failed={"shard": index, "gpu": gpu, "error": repr(error)})
                    save(owner, state)
                    return 3
            state["completed_shards"] = done
            if g2 != "complete":
                state.update(status="WAITING_G2_CLEAN_SUCCESS", current_stage="WAITING_G2_CLEAN_SUCCESS")
            elif g2 == "complete" and "g2_native_bundle" not in state:
                prepare_g2_native_bundle(state, owner)
            elif len(done) == count and not active:
                state.update(status="MERGING_TAO_NATIVE_SHARDS", current_stage="MERGING_TAO_NATIVE_SHARDS")
                save(owner, state)
                command = ["python3", str(MERGE_SCRIPT), "--shard-root", str(args.shard_root),
                           "--parent-manifest", str(args.parent_manifest), "--out", str(args.merged_out)]
                log_path = args.shard_root / "MERGE.log"
                with log_path.open("w", encoding="utf-8") as log:
                    result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
                if result.returncode != 0:
                    raise RuntimeError(f"TAO shard merge exited {result.returncode}; see {log_path}")
                state.update(status="TAO_NATIVE_TRAIN_STREAM_COMPLETE", current_stage="COMPLETE",
                             merged_receipt=str(args.merged_out / "receipt.json"), native_ovmot_training_rows=False)
                save(owner, state)
                return 0
            else:
                state.update(status="STREAMING_TAO_NATIVE_TRAIN_SHARDS", current_stage="STREAMING_TAO_NATIVE_TRAIN_SHARDS")
                for gpu in args.gpus:
                    if gpu in active:
                        continue
                    next_index = next((i for i in range(count) if str(i) not in done and all(i != item[0] for item in active.values())), None)
                    if next_index is None:
                        break
                    try:
                        probe = gpu_probe(gpu)
                    except Exception as error:
                        probe = {"gpu": gpu, "ready": False, "error": repr(error)}
                    state.setdefault("resource_probes", {})[str(gpu)] = probe
                    if probe.get("ready"):
                        active[gpu] = (next_index, executor.submit(run_shard, next_index, gpu, count))
            state["active"] = {str(gpu): index for gpu, (index, _future) in active.items()}
            save(owner, state)
            time.sleep(args.poll_seconds)
    except Exception as error:
        state.update(status="TAO_NATIVE_EXPORT_FAILED", current_stage="FAILED", failed={"error": repr(error)})
        save(owner, state)
        return 3
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    raise SystemExit(main())
