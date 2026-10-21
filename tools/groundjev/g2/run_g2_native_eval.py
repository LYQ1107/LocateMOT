#!/usr/bin/env python3
"""Run a small frozen G0/G2 native tracking control.

The G2 service is launched from this checkout so its model-root guard cannot
accidentally resolve the historical OneJev checkout.  The existing G0 service
is read through its registered API; it is never stopped or reconfigured.
This control is intentionally limited to the established MOT4/RMOT8/RMOT12
diagnostics and does not stand in for the later independent held-out gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

CODE_ROOT = Path(__file__).resolve().parents[3]
RUNTIME_ROOT = Path("/data2/user/LocateMOT_GroundJEV")
sys.path.insert(0, str(CODE_ROOT))
from tools.groundjev.service_process_group import stop_owned_service


VENV = RUNTIME_ROOT / ".venv-onejev-py310/bin/python"
G2_MANIFEST = Path(
    "/data3/user/LocateMOT-runs/GroundAnythingG2/G2_CLEAN_FULL_COVERAGE_V2/"
    "g2_clean_train_manifest.json"
)
G2_MANIFEST_SHA = "f682519a1cf9204ca76ea75a56dad36dbdeac26259f202551ab9f66fefd06e71"
G2_EXPORT_RECEIPT = RUNTIME_ROOT / "outputs/groundjev/ground_adaptation/g2_clean_bundle_export_v1/export_receipt.json"
G2_STEP = 18_227
SCOPES = (
    ("dense_mot4", "mot", RUNTIME_ROOT / "outputs/groundjev/pilot_data/mot17_first4.json"),
    ("refer_single8", "rmot", RUNTIME_ROOT / "outputs/groundjev/multitask_pilot_data/rmot_single_empty.json"),
    ("refer_multi12", "rmot", RUNTIME_ROOT / "outputs/groundjev/multitask_pilot_data/rmot_multi_empty.json"),
)


def ready(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
            payload = json.load(response)
        return any(item.get("id") == "groundinganything" for item in payload.get("data", []))
    except (OSError, ValueError):
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_identity(model: Path, receipt_path: Path, manifest_path: Path) -> dict[str, object]:
    """Validate the registered Clean V2 provenance before launching inference."""
    if model.name == "GroundAnything-DLM-G2-PL":
        # Keep the historical directory/label contract byte-for-byte
        # addressable; this branch is never reclassified as Clean V2.
        return {
            "experiment_identity": "G2-PL",
            "historical_or_non_clean_model": True,
            "clean_provenance_checked": False,
        }
    if model.name != "GroundAnything-DLM-G2-CLEAN-V2":
        return {
            "experiment_identity": model.name,
            "historical_or_non_clean_model": True,
            "clean_provenance_checked": False,
        }
    if not receipt_path.is_file():
        raise FileNotFoundError(f"Clean V2 export receipt is missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("status") != "G2_NATIVE_MERGED_BUNDLE_EXPORTED":
        raise ValueError(f"unexpected Clean V2 export receipt status: {receipt.get('status')!r}")
    if int(receipt.get("step", -1)) != G2_STEP:
        raise ValueError(f"Clean V2 adapter step mismatch: {receipt.get('step')} != {G2_STEP}")
    if receipt.get("manifest_sha256") != G2_MANIFEST_SHA:
        raise ValueError("Clean V2 export manifest SHA mismatch")
    if Path(str(receipt.get("output_bundle", ""))).resolve() != model.resolve():
        raise ValueError("Clean V2 evaluation model is not the bundle recorded by the export receipt")
    if not manifest_path.is_file() or sha256(manifest_path) != G2_MANIFEST_SHA:
        raise ValueError("registered Clean V2 training manifest is missing or has the wrong SHA")
    model_file = model / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(model_file)
    model_sha = sha256(model_file)
    if receipt.get("output_model_safetensors_sha256") != model_sha:
        raise ValueError("Clean V2 model.safetensors SHA differs from its export receipt")
    adapter = Path(str(receipt.get("adapter", "")))
    if not adapter.is_file():
        raise FileNotFoundError(f"Clean V2 adapter is missing: {adapter}")
    # The adapter is a small CPU checkpoint relative to the deployment bundle;
    # read only its trusted metadata to ensure the evaluated step is explicit.
    import torch
    payload = torch.load(adapter, map_location="cpu", weights_only=True)
    if int(payload.get("step", -1)) != G2_STEP or payload.get("manifest_sha256") != G2_MANIFEST_SHA:
        raise ValueError("Clean V2 adapter metadata does not match the registered terminal contract")
    return {
        "experiment_identity": "G2-CLEAN-V2",
        "historical_or_non_clean_model": False,
        "clean_provenance_checked": True,
        "model_sha256": model_sha,
        "train_manifest": str(manifest_path),
        "train_manifest_sha256": G2_MANIFEST_SHA,
        "adapter": str(adapter),
        "adapter_sha256": receipt.get("adapter_sha256"),
        "adapter_step": G2_STEP,
        "export_receipt": str(receipt_path),
    }


def run_logged(command: list[str], log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=RUNTIME_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-port", type=int, default=8101)
    parser.add_argument("--adapted-port", type=int, default=8113)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--mem-fraction-static", default=".4")
    parser.add_argument("--g2-export-receipt", type=Path, default=G2_EXPORT_RECEIPT)
    parser.add_argument("--g2-manifest", type=Path, default=G2_MANIFEST)
    args = parser.parse_args()
    model = args.model.resolve(strict=True)
    if not model.is_relative_to(RUNTIME_ROOT / "models"):
        raise ValueError(f"G2 model must be under the GroundJEV runtime model root: {model}")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite an existing evaluation: {args.out}")
    args.out.mkdir(parents=True)
    identity = clean_identity(model, args.g2_export_receipt, args.g2_manifest)
    adapted_label = str(identity["experiment_identity"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)

    state: dict[str, object] = {
        "status": "STARTING_G2_NATIVE_SHORT_CONTROLS",
        "model": str(model),
        "experiment_identity": adapted_label,
        "provenance": identity,
        "reference_service": f"http://127.0.0.1:{args.reference_port}/v1",
        "adapted_service": f"http://127.0.0.1:{args.adapted_port}/v1",
        "physical_gpu": int(args.gpu),
        "scopes": [name for name, _, _ in SCOPES],
        "completed": {},
        "full_dataset_claim": False,
        "heldout_claim": False,
    }

    def save() -> None:
        (args.out / "owner_state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
        print(json.dumps({"status": state["status"], "completed": list(state["completed"])}, ensure_ascii=False), flush=True)

    server = None
    server_log = None
    try:
        if not ready(args.reference_port):
            raise RuntimeError(f"registered G0 reference service is not ready on port {args.reference_port}")
        launch_log = args.out / "g2_service.log"
        server_log = launch_log.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(RUNTIME_ROOT / "tools/groundjev/run_with_resources.py"),
            "--gpus", str(args.gpu),
            "--",
            str(RUNTIME_ROOT / "external/GroundAnything/.venv-serve/bin/python"),
            str(RUNTIME_ROOT / "tools/groundjev/serve_groundanything.py"),
            "--model", str(model),
            "--port", str(args.adapted_port),
            "--mem-fraction-static", str(args.mem_fraction_static),
            "--max-queued-requests", "2",
            "--out", str(args.out / "g2_service"),
        ]
        server = subprocess.Popen(command, cwd=RUNTIME_ROOT, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(status="WAITING_G2_SERVICE", owned_service_process_group=server.pid, launch_command=command)
        save()
        deadline = time.monotonic() + 600
        while not ready(args.adapted_port):
            if server.poll() is not None:
                raise RuntimeError(f"G2 service exited with code {server.returncode}; see {launch_log}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"G2 service readiness timeout; see {launch_log}")
            time.sleep(2)

        for name, task, manifest in SCOPES:
            state["status"] = f"RUNNING_{name}"
            save()
            for label, port in (("G0", args.reference_port), (adapted_label, args.adapted_port)):
                output = args.out / label / name
                command = [
                    str(VENV), str(RUNTIME_ROOT / "tools/groundjev/run_video.py"),
                    "--task", task, "--video", str(manifest), "--backend", "simple",
                    "--grounding-url", f"http://127.0.0.1:{port}/v1",
                    "--grounding-parser", "official_scoring",
                    "--grounding-decoder-policy", "tracking_dense",
                    "--perception-failure-policy", "empty_and_record",
                    "--out", str(output),
                ]
                run_logged(command, args.out / f"{label}_{name}_inference.log", env)
                scorer = RUNTIME_ROOT / ("tools/groundjev/evaluate_video.py" if task == "mot" else "tools/groundjev/evaluate_multitask.py")
                run_logged([str(VENV), scorer, "--video", manifest, "--run", str(output)], args.out / f"{label}_{name}_eval.log", env)
                expected = json.loads(Path(manifest).read_text())["frame_numbers"]
                observed = [json.loads(line)["frame"] for line in (output / "frame_decisions.jsonl").read_text().splitlines()]
                if observed != expected:
                    raise RuntimeError(f"frame contract drift for {label}/{name}: {len(observed)} vs {len(expected)}")
                metrics = json.loads((output / "official_metrics.json").read_text())
                state["completed"][f"{label}/{name}"] = metrics
                save()
        state["status"] = "G2_NATIVE_SHORT_CONTROLS_COMPLETE"
        save()
        (args.out / "comparison.json").write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
        return 0
    except Exception as error:
        state.update(status="G2_NATIVE_SHORT_CONTROLS_FAILED", error=repr(error))
        save()
        raise
    finally:
        if server is not None:
            stop_owned_service(server)
            state["owned_service_stopped"] = True
            save()
        if server_log is not None:
            server_log.close()
        if state.get("status") == "G2_NATIVE_SHORT_CONTROLS_FAILED":
            (args.out / "comparison.json").write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
