#!/usr/bin/env python3
"""Queue a real, video-disjoint G2/G0 held-out comparison.

The owner waits for the terminal Clean G2 bundle and an exclusive GPU7 window,
then evaluates the same RMOT and OVMOT manifests with the same parser,
decoder, tracker, and scorer for G0 and G2.  It is deliberately independent
of the OneJev training queue and preserves every partial output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = Path("/data2/user/LocateMOT_GroundJEV")
VENV = RUNTIME_ROOT / ".venv-onejev-py310/bin/python"
SERVE_PY = RUNTIME_ROOT / "tools/groundjev/serve_groundanything.py"
RUN_VIDEO = RUNTIME_ROOT / "tools/groundjev/run_video.py"
SCORER = RUNTIME_ROOT / "tools/groundjev/evaluate_multitask.py"
BASE_MODEL = RUNTIME_ROOT / "models/GroundAnything-DLM"
G2_BUNDLE = RUNTIME_ROOT / "models/GroundAnything-DLM-G2-CLEAN-V2"
G2_ROOT = Path(
    "/data3/user/LocateMOT-runs/GroundAnythingG2/"
    "G2_CLEAN_FULL_COVERAGE_V2/formal_clean_coverage_v2_final"
)
EXPORT_STATE = RUNTIME_ROOT / "outputs/groundjev/ground_adaptation/" \
    "g2_clean_bundle_export_v1/owner_state.json"
TRAIN_MANIFEST = Path(
    "/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/"
    "g2_clean_train_manifest.json"
)
TRAIN_ROWS = Path(
    "/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/"
    "g2_clean_train_rows.jsonl"
)
GPU = 7
PORT = 8217
G2_EXPORT_STATUS = "G2_CLEAN_DEPLOYMENT_BUNDLE_EXPORTED"
G2_TRAIN_STATUS = "G2_EXACT_COVERAGE_TRAINING_COMPLETE"
MANIFEST_SHA = "f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71"
DEFAULT_OUT = RUNTIME_ROOT / "outputs/groundjev/heldout/g2_clean_independent_gpu7_v1"
SCOPES = (
    ("rmot_single0007", "rmot", RUNTIME_ROOT / "outputs/groundjev/multitask_pilot_data/rmot_single_empty_full.json"),
    ("rmot_multi0007", "rmot", RUNTIME_ROOT / "outputs/groundjev/multitask_pilot_data/rmot_multi_empty_full.json"),
    ("ovmot_tao105", "ovmot", RUNTIME_ROOT / "outputs/groundjev/multitask_pilot_data/ovmot_tao_independent105_full.json"),
)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
    print(json.dumps({k: state.get(k) for k in ("status", "current_stage", "completed", "failed")}, ensure_ascii=False), flush=True)


def bundle_ready() -> tuple[bool, dict | None, str | None]:
    state = read_json(EXPORT_STATE)
    if state is None:
        return False, None, None
    if state.get("status") != G2_EXPORT_STATUS:
        if str(state.get("status", "")).endswith("FAILED"):
            return False, state, "G2 bundle export failed"
        return False, state, None
    receipt = read_json(G2_BUNDLE.parent.parent / "outputs/groundjev/ground_adaptation/g2_clean_bundle_export_v1/export_receipt.json")
    if receipt is None or receipt.get("status") != "G2_NATIVE_MERGED_BUNDLE_EXPORTED":
        return False, state, "bundle owner is terminal but export receipt is missing or invalid"
    train = read_json(G2_ROOT / "final_receipt.json")
    if train is None or train.get("status") != G2_TRAIN_STATUS:
        return False, state, "bundle export has no valid G2 terminal receipt"
    if int(train.get("steps", -1)) != 18227 or train.get("manifest_sha256") != MANIFEST_SHA:
        return False, state, "G2 terminal receipt violates the clean training contract"
    if not (G2_BUNDLE / "model.safetensors").is_file():
        return False, state, "G2 bundle model.safetensors is missing"
    return True, {"export": state, "export_receipt": receipt, "training": train}, None


def gpu_ready() -> tuple[bool, dict]:
    apps = subprocess.run(
        ["nvidia-smi", "-i", str(GPU), "--query-compute-apps=pid,used_gpu_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True,
    ).stdout.strip()
    memory = subprocess.run(
        ["nvidia-smi", "-i", str(GPU), "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    free = int(memory.splitlines()[0].strip())
    pids = [line.strip() for line in apps.splitlines() if line.strip()]
    return (not pids and free >= 32768), {"compute_apps": pids, "free_memory_MiB": free}


def normalized_paths(path: str) -> set[str]:
    value = os.path.normpath(path)
    aliases = {value}
    marker = "/TAO-Amodal/frames/"
    if marker in value:
        aliases.add("TAO-Amodal/frames/" + value.split(marker, 1)[1])
    return aliases


def overlap_audit() -> dict:
    scope_paths: dict[str, set[str]] = {}
    scope_metadata: dict[str, dict] = {}
    for name, task, manifest_path in SCOPES:
        manifest = read_json(manifest_path)
        if manifest is None:
            raise FileNotFoundError(manifest_path)
        if manifest.get("task") != task or manifest.get("training_on_this_scope") is not False:
            raise RuntimeError(f"invalid held-out manifest contract: {manifest_path}")
        paths = set()
        for image in manifest.get("images", []):
            paths.update(normalized_paths(str(image)))
        scope_paths[name] = paths
        scope_metadata[name] = {
            "task": task,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "frames": len(manifest.get("images", [])),
            "sequence": manifest.get("sequence"),
            "video_id": manifest.get("video_id"),
        }
    found = {name: set() for name, _, _ in SCOPES}
    if TRAIN_ROWS.is_file():
        for line in TRAIN_ROWS.open():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for image in row.get("images", []):
                row_paths = normalized_paths(str(image.get("path", ""))) if isinstance(image, dict) else normalized_paths(str(image))
                for name, paths in scope_paths.items():
                    if paths.intersection(row_paths):
                        found[name].update(paths.intersection(row_paths))
    result = {
        "status": "HELDOUT_SCOPE_OVERLAP_AUDIT_COMPLETE",
        "train_manifest": str(TRAIN_MANIFEST),
        "train_manifest_sha256": sha256(TRAIN_MANIFEST),
        "train_rows": str(TRAIN_ROWS),
        "scopes": scope_metadata,
        "exact_image_overlap": {name: sorted(values) for name, values in found.items()},
        "independent": {name: not values for name, values in found.items()},
    }
    return result


def ready(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
            payload = json.load(response)
        return any(item.get("id") == "groundinganything" for item in payload.get("data", []))
    except (OSError, ValueError):
        return False


def stop_group(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=60)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            pass


def run_logged(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}; see {log_path}")


def launch_service(model: Path, label: str, out: Path, env: dict[str, str]) -> subprocess.Popen:
    command = [
        "python3", str(RUNTIME_ROOT / "tools/groundjev/run_with_resources.py"),
        "--gpus", str(GPU), "--", str(RUNTIME_ROOT / "external/GroundAnything/.venv-serve/bin/python"),
        str(SERVE_PY), "--model", str(model), "--port", str(PORT),
        "--mem-fraction-static", ".35", "--max-queued-requests", "2",
        "--out", str(out / f"{label}_service"),
    ]
    log = (out / f"{label}_service.log").open("w", encoding="utf-8")
    process = subprocess.Popen(command, cwd=RUNTIME_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    process._groundjev_log = log  # type: ignore[attr-defined]
    process._groundjev_command = command  # type: ignore[attr-defined]
    return process


def close_service(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    stop_group(process)
    log = getattr(process, "_groundjev_log", None)
    if log is not None:
        log.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    owner = args.out / "owner_state.json"
    if args.out.exists():
        old = read_json(owner)
        if old and old.get("status", "").endswith(("COMPLETE", "FAILED")):
            raise FileExistsError(f"terminal held-out queue already exists: {owner}")
        if old and old.get("owner_pid") and Path(f"/proc/{old['owner_pid']}").exists():
            raise FileExistsError(f"held-out queue owner is already live: {old['owner_pid']}")
    args.out.mkdir(parents=True, exist_ok=True)
    state = {
        "status": "WAITING_G2_CLEAN_BUNDLE_AND_GPU7",
        "current_stage": "WAITING_G2_CLEAN_BUNDLE_AND_GPU7",
        "owner_pid": os.getpid(),
        "project_root": str(PROJECT_ROOT),
        "runtime_root": str(RUNTIME_ROOT),
        "g2_root": str(G2_ROOT),
        "export_state": str(EXPORT_STATE),
        "g2_bundle": str(G2_BUNDLE),
        "physical_gpu": GPU,
        "scopes": [name for name, _, _ in SCOPES],
        "completed": {},
        "failed": None,
        "same_input_parser_decoder_tracker_scorer": True,
        "feature_cache": False,
        "heldout_claim": "pending until exact overlap audit and official metrics complete",
    }
    save(owner, state)
    service = None
    try:
        audit = overlap_audit()
        state["overlap_audit"] = audit
        if not all(audit["independent"].values()):
            raise RuntimeError(f"held-out scope overlaps G2 TRAIN: {audit['exact_image_overlap']}")
        save(owner, state)
        while True:
            ok, evidence, error = bundle_ready()
            if error:
                state.update(status="G2_CLEAN_HELDOUT_DEPENDENCY_FAILED", current_stage="G2_DEPENDENCY_FAILED", failed={"reason": error, "evidence": evidence})
                save(owner, state)
                return 2
            if ok:
                state["completed"]["g2_dependency"] = evidence
                try:
                    free, resource = gpu_ready()
                except Exception as probe_error:
                    free, resource = False, {"error": repr(probe_error)}
                state["resource_probe"] = resource
                if free:
                    break
            time.sleep(args.poll_seconds)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            env.pop(key, None)
        state.update(status="RUNNING_G0_G2_INDEPENDENT_HELDOUT", current_stage="RUNNING_G0_G2_INDEPENDENT_HELDOUT")
        save(owner, state)
        for label, model in (("G0", BASE_MODEL), ("G2-CLEAN-V2", G2_BUNDLE)):
            if not (model / "model.safetensors").is_file():
                raise FileNotFoundError(model / "model.safetensors")
            state["current_model"] = label
            state["model_paths"] = state.get("model_paths", {}) | {label: str(model)}
            service = launch_service(model, label.replace("-", "_"), args.out, env)
            state["service_command"] = getattr(service, "_groundjev_command")
            state["service_pid"] = service.pid
            state["status"] = f"WAITING_{label.replace('-', '_')}_SERVICE"
            save(owner, state)
            deadline = time.monotonic() + 900
            while not ready(PORT):
                if service.poll() is not None:
                    raise RuntimeError(f"{label} service exited with {service.returncode}; see {args.out}/{label.replace('-', '_')}_service.log")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"{label} service readiness timeout")
                time.sleep(2)
            for name, task, manifest in SCOPES:
                output = args.out / label.replace("-", "_") / name
                command = [
                    str(VENV), str(RUN_VIDEO), "--task", task, "--video", str(manifest), "--backend", "simple",
                    "--grounding-url", f"http://127.0.0.1:{PORT}/v1", "--grounding-parser", "official_scoring",
                    "--grounding-decoder-policy", "tracking_dense", "--perception-failure-policy", "empty_and_record",
                    "--out", str(output),
                ]
                if task == "ovmot":
                    command += ["--grounding-vocabulary-chunk-size", "96", "--category-policy", "native_label"]
                state.update(status=f"RUNNING_{label.replace('-', '_')}_{name}", current_stage=f"RUNNING_{label.replace('-', '_')}_{name}", current_scope=name)
                save(owner, state)
                run_logged(command, RUNTIME_ROOT, env, args.out / f"{label.replace('-', '_')}_{name}_inference.log")
                scorer_command = [str(VENV), str(SCORER), "--video", str(manifest), "--run", str(output)]
                run_logged(scorer_command, RUNTIME_ROOT, env, args.out / f"{label.replace('-', '_')}_{name}_eval.log")
                result = read_json(output / "result.json")
                metrics = read_json(output / "official_metrics.json")
                if result is None or metrics is None:
                    raise RuntimeError(f"missing result or official metrics for {label}/{name}")
                expected = read_json(manifest)["frame_numbers"]
                observed = [json.loads(line)["frame"] for line in (output / "frame_decisions.jsonl").read_text().splitlines()]
                if observed != expected:
                    raise RuntimeError(f"frame contract drift for {label}/{name}: {len(observed)} vs {len(expected)}")
                state["completed"][f"{label}/{name}"] = {
                    "model": str(model), "task": task, "manifest": str(manifest),
                    "frames": len(observed), "official_metrics": metrics,
                    "result": result, "inference_command": command, "scorer_command": scorer_command,
                }
                save(owner, state)
            close_service(service)
            service = None
            state["service_stopped"] = label
            save(owner, state)
        state.update(status="G2_CLEAN_FULL_INDEPENDENT_HELDOUT_COMPLETE", current_stage="COMPLETE", heldout_claim=True)
        state["receipt"] = {"status": state["status"], "scopes": state["completed"], "overlap_audit": state["overlap_audit"]}
        save(owner, state)
        (args.out / "comparison.json").write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
        return 0
    except Exception as error:
        state.update(status="G2_CLEAN_FULL_INDEPENDENT_HELDOUT_FAILED", current_stage="FAILED", failed={"error": repr(error)})
        save(owner, state)
        (args.out / "failure_receipt.json").write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
        return 1
    finally:
        close_service(service)


if __name__ == "__main__":
    raise SystemExit(main())
