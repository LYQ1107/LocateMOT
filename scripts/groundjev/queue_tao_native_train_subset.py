#!/usr/bin/env python3
"""Stream a registered TAO TRAIN subset and build the native joint manifest.

The full TAO TRAIN scope is preserved separately.  This resumable owner is the
explicit 64-frame/64-video native subset used to unblock the real three-task
GroundJEV run without calling a partial cache a full-dataset result.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = Path("/data2/user/LocateMOT_GroundJEV")
SCOPE_ROOT = RUNTIME_ROOT / "outputs/groundjev/training_data/tao_native_ovmot_train_subset64_v1"
MERGED_ROOT = RUNTIME_ROOT / "outputs/groundjev/training_data/tao_native_ovmot_train_subset64_merged_v1"
JOIN_ROOT = RUNTIME_ROOT / "outputs/groundjev/training_data/tao_native_ovmot_train_subset64_join_v1"
JOINT_ROOT = RUNTIME_ROOT / "outputs/groundjev/training_data/real_three_task_joint_native_v1"
G2_ROOT = Path("/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final")
G2_MODEL = RUNTIME_ROOT / "models/GroundAnything-DLM-G2-CLEAN-NATIVE-TAO-V1"
G2_MANIFEST_SHA = "f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71"
G2_STEPS = 18227
G2_EXPORT_RECEIPT = RUNTIME_ROOT / "outputs/groundjev/training_data/tao_native_ovmot_train_shards_v2/g2_native_bundle_export_receipt.json"
GROUND_PYTHON = RUNTIME_ROOT / "external/GroundAnything/.venv-serve/bin/python"
STREAM_PYTHON = RUNTIME_ROOT / ".venv-onejev-py310/bin/python"
GROUND_SERVICE = RUNTIME_ROOT / "tools/groundjev/serve_groundanything.py"
FEATURE_SERVICE = RUNTIME_ROOT / "tools/groundjev/serve_native_features.py"
STREAM_SCRIPT = RUNTIME_ROOT / "tools/groundjev/stream_tao_native_candidates.py"
JOIN_SCRIPT = PROJECT_ROOT / "tools/groundjev/prepare_native_ovmot_train_join.py"
JOINT_SCRIPT = PROJECT_ROOT / "tools/groundjev/prepare_real_three_task_joint_manifest.py"
BASE_PORT = 8400
GPU_POOL = (4, 5, 6, 7)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
    print(json.dumps({k: state.get(k) for k in ("status", "current_stage", "completed_shards", "failed")}, ensure_ascii=False), flush=True)


def g2_ready() -> bool:
    receipt = read_json(G2_ROOT / "final_receipt.json")
    export = read_json(G2_EXPORT_RECEIPT)
    return bool(
        receipt and receipt.get("status") == "G2_EXACT_COVERAGE_TRAINING_COMPLETE"
        and int(receipt.get("steps", -1)) == G2_STEPS
        and receipt.get("manifest_sha256") == G2_MANIFEST_SHA
        and export and export.get("status") == "G2_NATIVE_MERGED_BUNDLE_EXPORTED"
        and (G2_MODEL / "model.safetensors").is_file()
    )


def gpu_ready(gpu: int) -> dict:
    apps = subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    free = int(subprocess.run(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0])
    return {"gpu": gpu, "compute_apps": [line for line in apps.splitlines() if line.strip()],
            "free_memory_MiB": free, "ready": free >= 32768,
            "co_resident_foreign_apps": bool(apps)}


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


def wait_services(ground: subprocess.Popen, feature: subprocess.Popen, port: int, receipt: Path) -> None:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if ground.poll() is not None or feature.poll() is not None:
            raise RuntimeError("owned native service exited before readiness")
        try:
            with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
                payload = json.load(response)
            if any(item.get("id") == "groundinganything" for item in payload.get("data", [])) and receipt.is_file():
                return
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise TimeoutError("owned native service readiness timeout")


def run_shard(index: int, gpu: int, count: int) -> dict:
    shard_dir = SCOPE_ROOT / f"shard_{index:02d}_of_{count:02d}"
    manifest = shard_dir / "manifest.json"
    output = shard_dir / "native_candidates"
    receipt = output / "receipt.json"
    existing = read_json(receipt)
    if existing and existing.get("status") == "COMPLETE_LEGAL_NATIVE_VOCABULARY_AND_ROI_STREAM":
        return existing
    if output.exists():
        raise FileExistsError(f"refusing to overwrite partial subset output: {output}")
    ground_port = BASE_PORT + index * 4
    feature_port = ground_port + 1
    for port in (ground_port, feature_port):
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(("127.0.0.1", port))
    service_dir = shard_dir / f"services_gpu{gpu}"
    service_dir.mkdir()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    ground_log = (service_dir / "ground.log").open("w", encoding="utf-8")
    feature_log = (service_dir / "features.log").open("w", encoding="utf-8")
    ground = feature = None
    try:
        ground = subprocess.Popen([
            str(GROUND_PYTHON), str(GROUND_SERVICE), "--model", str(G2_MODEL), "--port", str(ground_port),
            "--mem-fraction-static", ".35", "--max-queued-requests", "2", "--out", str(service_dir / "ground"),
        ], cwd=RUNTIME_ROOT, env=env, stdout=ground_log, stderr=subprocess.STDOUT, start_new_session=True)
        feature = subprocess.Popen([
            str(GROUND_PYTHON), str(FEATURE_SERVICE), "--port", str(feature_port),
            "--receipt", str(service_dir / "feature_receipt.json"),
        ], cwd=RUNTIME_ROOT, env=env, stdout=feature_log, stderr=subprocess.STDOUT, start_new_session=True)
        wait_services(ground, feature, ground_port, service_dir / "feature_receipt.json")
        stream_log = shard_dir / "native_stream.log"
        command = [
            str(STREAM_PYTHON), str(STREAM_SCRIPT), "--manifest", str(manifest), "--out", str(output),
            "--allow-training-scope", "--grounding-url", f"http://127.0.0.1:{ground_port}/v1",
            "--native-feature-url", f"http://127.0.0.1:{feature_port}",
        ]
        with stream_log.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=RUNTIME_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"subset shard {index} exited {result.returncode}; see {stream_log}")
        result_receipt = read_json(receipt)
        if result_receipt is None or result_receipt.get("status") != "COMPLETE_LEGAL_NATIVE_VOCABULARY_AND_ROI_STREAM":
            raise RuntimeError(f"subset shard {index} did not produce a terminal receipt")
        return result_receipt
    finally:
        stop_group(feature)
        stop_group(ground)
        feature_log.close()
        ground_log.close()


def completed_shards(count: int) -> dict[str, dict]:
    result = {}
    for index in range(count):
        receipt = SCOPE_ROOT / f"shard_{index:02d}_of_{count:02d}" / "native_candidates" / "receipt.json"
        value = read_json(receipt)
        if value and value.get("status") == "COMPLETE_LEGAL_NATIVE_VOCABULARY_AND_ROI_STREAM":
            result[str(index)] = value
    return result


def merge_subset(count: int) -> dict:
    if MERGED_ROOT.exists():
        raise FileExistsError(f"refusing to overwrite {MERGED_ROOT}")
    MERGED_ROOT.mkdir(parents=True)
    output = MERGED_ROOT / "frame_candidates.jsonl"
    total = 0
    images = []
    with output.open("w", encoding="utf-8") as merged:
        for index in range(count):
            shard = SCOPE_ROOT / f"shard_{index:02d}_of_{count:02d}"
            manifest = read_json(shard / "manifest.json")
            candidate = shard / "native_candidates" / "frame_candidates.jsonl"
            expected = manifest["images"]
            lines = candidate.read_text(encoding="utf-8").splitlines()
            if len(lines) != len(expected):
                raise ValueError(f"subset shard {index} frame count mismatch: {len(lines)} vs {len(expected)}")
            for line, path in zip(lines, expected):
                record = json.loads(line)
                if record.get("image_path") != path or record.get("perception", {}).get("GT_read") is not False:
                    raise ValueError(f"subset shard {index} failed GT-free frame contract")
                merged.write(line + "\n")
                images.append(path)
                total += 1
    subset_manifest = read_json(SCOPE_ROOT / "subset_manifest.json") or {}
    receipt = {
        "status": "COMPLETE_REGISTERED_NATIVE_TAO_TRAIN_SUBSET_STREAM",
        "scope_root": str(SCOPE_ROOT),
        "candidate_archive": str(output),
        "candidate_archive_sha256": sha256(output),
        "frames": total,
        "videos": len(subset_manifest.get("selected_videos", [])),
        "shards": count,
        "registered_train_subset": True,
        "full_train_image_scope": False,
        "all1203classes_requested_on_every_frame": True,
        "GT_or_Novel_labels_read_by_model": False,
        "end_to_end_tracking_FPS_claim": False,
    }
    (MERGED_ROOT / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--gpus", type=int, nargs="+", default=list(GPU_POOL))
    args = parser.parse_args()
    if tuple(args.gpus) != GPU_POOL:
        raise ValueError(f"registered subset owner requires physical GPUs {GPU_POOL}")
    owner = SCOPE_ROOT / "queue_owner_state.json"
    old = read_json(owner)
    if old and old.get("owner_pid") and Path(f"/proc/{old['owner_pid']}").exists():
        raise FileExistsError(f"subset owner is already live: {old['owner_pid']}")
    if old and str(old.get("status", "")).endswith(("_COMPLETE", "_FAILED")):
        raise FileExistsError(f"terminal subset owner already exists: {owner}")
    state = {
        "status": "WAITING_G2_NATIVE_BUNDLE_AND_GPU_POOL",
        "current_stage": "WAITING_G2_NATIVE_BUNDLE_AND_GPU_POOL",
        "owner_pid": os.getpid(),
        "scope_root": str(SCOPE_ROOT),
        "subset_manifest": str(SCOPE_ROOT / "subset_manifest.json"),
        "registered_subset_frames": 64,
        "registered_subset_videos": 64,
        "physical_gpus": list(args.gpus),
        "g2_steps": G2_STEPS,
        "g2_manifest_sha256": G2_MANIFEST_SHA,
        "completed_shards": completed_shards(4),
        "active": {},
        "failed": None,
        "native_ovmot_training_rows": False,
        "full_train_image_scope": False,
    }
    save(owner, state)
    try:
        while not g2_ready():
            state.update(status="WAITING_G2_NATIVE_BUNDLE_AND_GPU_POOL", current_stage="WAITING_G2_NATIVE_BUNDLE_AND_GPU_POOL")
            save(owner, state)
            time.sleep(args.poll_seconds)
        probes = {str(gpu): gpu_ready(gpu) for gpu in args.gpus}
        state["resource_probe"] = probes
        if not all(item["ready"] for item in probes.values()):
            state.update(status="WAITING_REGISTERED_GPU_MEMORY", current_stage="WAITING_REGISTERED_GPU_MEMORY")
            save(owner, state)
            while True:
                probes = {str(gpu): gpu_ready(gpu) for gpu in args.gpus}
                state["resource_probe"] = probes
                if all(item["ready"] for item in probes.values()):
                    break
                save(owner, state)
                time.sleep(args.poll_seconds)
        state.update(status="STREAMING_REGISTERED_TAO_TRAIN_SUBSET", current_stage="STREAMING_REGISTERED_TAO_TRAIN_SUBSET")
        save(owner, state)
        count = 4
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
            futures = {executor.submit(run_shard, index, gpu, count): (index, gpu) for index, gpu in enumerate(args.gpus)}
            for future, (index, gpu) in [(future, value) for future, value in futures.items()]:
                try:
                    future.result()
                except Exception as error:
                    state.update(status="TAO_NATIVE_SUBSET_EXPORT_FAILED", current_stage="FAILED", failed={"shard": index, "gpu": gpu, "error": repr(error)})
                    save(owner, state)
                    return 3
        state["completed_shards"] = completed_shards(count)
        state.update(status="MERGING_REGISTERED_TAO_TRAIN_SUBSET", current_stage="MERGING_REGISTERED_TAO_TRAIN_SUBSET")
        save(owner, state)
        merged = merge_subset(count)
        state["completed"] = {"native_subset_stream": merged}
        state.update(status="JOINING_REGISTERED_NATIVE_OVMOT_SUBSET", current_stage="JOINING_REGISTERED_NATIVE_OVMOT_SUBSET")
        save(owner, state)
        if JOIN_ROOT.exists() or JOINT_ROOT.exists():
            raise FileExistsError("refusing to overwrite a previous subset join or joint manifest")
        join_log = SCOPE_ROOT / "NATIVE_OVMOT_SUBSET_JOIN.log"
        command = [str(STREAM_PYTHON), str(JOIN_SCRIPT), "--candidates", str(MERGED_ROOT / "frame_candidates.jsonl"), "--out", str(JOIN_ROOT), "--max-frames", "64", "--registered-train-subset"]
        with join_log.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        join_manifest = read_json(JOIN_ROOT / "manifest.json")
        if result.returncode != 0 or join_manifest is None or join_manifest.get("native_ovmot_training_rows") is not True or join_manifest.get("scope", {}).get("registered_train_subset") is not True:
            raise RuntimeError(f"registered subset native join failed; returncode={result.returncode}")
        state["completed"]["native_ovmot_join"] = {"manifest": str(JOIN_ROOT / "manifest.json"), "rows": join_manifest.get("rows"), "manifest_sha256": sha256(JOIN_ROOT / "manifest.json")}
        state.update(status="BUILDING_REGISTERED_NATIVE_THREE_TASK_MANIFEST", current_stage="BUILDING_REGISTERED_NATIVE_THREE_TASK_MANIFEST")
        save(owner, state)
        joint_log = SCOPE_ROOT / "NATIVE_THREE_TASK_SUBSET_MANIFEST.log"
        command = [str(STREAM_PYTHON), str(JOINT_SCRIPT), "--out", str(JOINT_ROOT), "--ovmot-native-manifest", str(JOIN_ROOT / "manifest.json")]
        with joint_log.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        joint_manifest = read_json(JOINT_ROOT / "train.sources.json")
        joint_receipt = read_json(JOINT_ROOT / "manifest_receipt.json")
        if result.returncode != 0 or joint_manifest is None or joint_receipt is None or joint_manifest.get("native_ovmot_training_rows") is not True or joint_manifest.get("ovmot_training_mode") != "native":
            raise RuntimeError(f"registered subset three-task manifest failed; returncode={result.returncode}")
        state["completed"]["native_three_task_manifest"] = {"manifest": str(JOINT_ROOT / "train.sources.json"), "manifest_receipt": str(JOINT_ROOT / "manifest_receipt.json"), "manifest_sha256": joint_receipt.get("manifest_sha256"), "rows": joint_manifest.get("rows"), "rows_per_task": joint_manifest.get("rows_per_task")}
        state.update(status="REGISTERED_NATIVE_TAO_SUBSET_AND_THREE_TASK_MANIFEST_COMPLETE", current_stage="COMPLETE", native_ovmot_training_rows=True, optimizer_updates=0, full_train_image_scope=False)
        save(owner, state)
        return 0
    except Exception as error:
        state.update(status="TAO_NATIVE_SUBSET_EXPORT_FAILED", current_stage="FAILED", failed={"error": repr(error)})
        save(owner, state)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
