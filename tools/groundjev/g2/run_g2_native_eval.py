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
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tools.groundjev.service_process_group import stop_owned_service


ONEJEV = Path("/data2/user/LocateMOT_OneJev_CVPR")
VENV = Path("/data2/user/LocateMOT_GroundJEV/.venv-onejev-py310/bin/python")
SCOPES = (
    ("dense_mot4", "mot", "outputs/groundjev/pilot_data/mot17_first4.json"),
    ("refer_single8", "rmot", "outputs/groundjev/multitask_pilot_data/rmot_single_empty.json"),
    ("refer_multi12", "rmot", "outputs/groundjev/multitask_pilot_data/rmot_multi_empty.json"),
)


def ready(port: int) -> bool:
    try:
        with urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as response:
            payload = json.load(response)
        return any(item.get("id") == "groundinganything" for item in payload.get("data", []))
    except (OSError, ValueError):
        return False


def run_logged(command: list[str], log: Path, env: dict[str, str]) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        subprocess.run(command, cwd=ONEJEV, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-port", type=int, default=8101)
    parser.add_argument(
        "--reference-model",
        type=Path,
        default=None,
        help="start an owned fresh reference service instead of trusting a stale port",
    )
    parser.add_argument("--reference-gpu", type=int, default=6)
    parser.add_argument("--adapted-port", type=int, default=8113)
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--mem-fraction-static", default=".4")
    args = parser.parse_args()
    model = args.model.resolve(strict=True)
    if not model.is_relative_to(ROOT / "models"):
        raise ValueError(f"G2 model must be under the GroundJEV model root: {model}")
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite an existing evaluation: {args.out}")
    args.out.mkdir(parents=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)

    state: dict[str, object] = {
        "status": "STARTING_G2_NATIVE_SHORT_CONTROLS",
        "model": str(model),
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
    reference_server = None
    reference_log = None
    try:
        if args.reference_model is not None:
            reference_model = args.reference_model.resolve(strict=True)
            if not (reference_model / "model.safetensors").is_file():
                raise FileNotFoundError(reference_model / "model.safetensors")
            if ready(args.reference_port):
                raise RuntimeError(
                    f"refusing to replace an existing service on reference port {args.reference_port}; "
                    "choose a free port for the independent reference"
                )
            reference_log = (args.out / "g0_reference_service.log").open("w", encoding="utf-8")
            reference_command = [
                sys.executable,
                str(ROOT / "tools/groundjev/run_with_resources.py"),
                "--gpus", str(args.reference_gpu),
                "--",
                str(ROOT / "external/GroundAnything/.venv-serve/bin/python"),
                str(ROOT / "tools/groundjev/serve_groundanything.py"),
                "--model", str(reference_model),
                "--port", str(args.reference_port),
                "--mem-fraction-static", str(args.mem_fraction_static),
                "--max-queued-requests", "2",
                "--out", str(args.out / "g0_reference_service"),
            ]
            reference_server = subprocess.Popen(
                reference_command,
                cwd=ROOT,
                stdout=reference_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            state.update(
                status="WAITING_G0_REFERENCE_SERVICE",
                owned_reference_service_process_group=reference_server.pid,
                reference_launch_command=reference_command,
            )
            save()
            deadline = time.monotonic() + 600
            while not ready(args.reference_port):
                if reference_server.poll() is not None:
                    raise RuntimeError(
                        f"G0 reference service exited with code {reference_server.returncode}; see {reference_log.name}"
                    )
                if time.monotonic() > deadline:
                    raise TimeoutError(f"G0 reference service readiness timeout; see {reference_log.name}")
                time.sleep(2)
        elif not ready(args.reference_port):
            raise RuntimeError(f"registered G0 reference service is not ready on port {args.reference_port}")
        launch_log = args.out / "g2_service.log"
        server_log = launch_log.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(ROOT / "tools/groundjev/run_with_resources.py"),
            "--gpus", str(args.gpu),
            "--",
            str(ROOT / "external/GroundAnything/.venv-serve/bin/python"),
            str(ROOT / "tools/groundjev/serve_groundanything.py"),
            "--model", str(model),
            "--port", str(args.adapted_port),
            "--mem-fraction-static", str(args.mem_fraction_static),
            "--max-queued-requests", "2",
            "--out", str(args.out / "g2_service"),
        ]
        server = subprocess.Popen(command, cwd=ROOT, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
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
            for label, port in (("G0", args.reference_port), ("G2-PL", args.adapted_port)):
                output = args.out / label / name
                command = [
                    str(VENV), "tools/groundjev/run_video.py",
                    "--task", task, "--video", manifest, "--backend", "simple",
                    "--grounding-url", f"http://127.0.0.1:{port}/v1",
                    "--grounding-parser", "official_scoring",
                    "--grounding-decoder-policy", "tracking_dense",
                    "--perception-failure-policy", "empty_and_record",
                    "--out", str(output),
                ]
                run_logged(command, args.out / f"{label}_{name}_inference.log", env)
                scorer = "tools/groundjev/evaluate_video.py" if task == "mot" else "tools/groundjev/evaluate_multitask.py"
                run_logged([str(VENV), scorer, "--video", manifest, "--run", str(output)], args.out / f"{label}_{name}_eval.log", env)
                expected = json.loads((ONEJEV / manifest).read_text())["frame_numbers"]
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
        if reference_server is not None:
            stop_owned_service(reference_server)
            state["owned_reference_service_stopped"] = True
            save()
        if reference_log is not None:
            reference_log.close()
        if state.get("status") == "G2_NATIVE_SHORT_CONTROLS_FAILED":
            (args.out / "comparison.json").write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
