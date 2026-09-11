#!/usr/bin/env python3
"""One-frame, label-free native R1 GroundingDINO contract smoke.

This intentionally builds the request from the label-free request manifest and
never opens an L69 label sidecar.  It verifies one frozen native visual pass,
one text-conditioned replay, and the fixed-reference Z1/Z4 states before a
large cache build is authorized.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.rmot.r1_aligned_cache import R1BankStore, R1GroundingRuntime, write_json  # noqa: E402
from tools.r1_common import SEED, THREAD, check_manifest, file_meta, standard_flags, unit_key  # noqa: E402


def _first_manifest_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("unique_query_records", {})
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        if values.get(dataset):
            record = next(iter(values[dataset].values()))
            # Categories are sampler metadata only and are deliberately not
            # passed into feature construction.
            return {key: value for key, value in record.items() if key != "category"}
    raise AssertionError("R1 request manifest has no query records")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-manifest", type=Path, default=Path("outputs/r1/request_manifest_attempt1/request_manifest.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else WORK_ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R1 native contract output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([str(sys.executable), *sys.argv])
    started = time.perf_counter()
    status: dict[str, Any] = {
        "format": "locatemot-r1-native-contract-v1", "status": "running", "command": command,
        "cwd": str(Path.cwd().resolve()), "thread": THREAD, "seed": SEED,
        "failure_root_cause": None, "next_action": "build label-free R1 aligned cache only after this contract passes",
        **standard_flags(training_run=False),
    }
    runtime: R1GroundingRuntime | None = None
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R1 native-contract cwd: {Path.cwd()}")
        manifest_sha = check_manifest()
        request_path = (args.request_manifest if args.request_manifest.is_absolute() else WORK_ROOT / args.request_manifest).resolve()
        record = _first_manifest_record(request_path)
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("R1 native contract requires CUDA")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        store = R1BankStore()
        batch = store.build_frame(record["dataset"], record["video"], int(record["query_id"]),
                                  int(record["frame_id"]), str(record["sentence"]))
        runtime = R1GroundingRuntime(device)
        payload = runtime.extract_group([batch], query_batch_size=1)
        if int(payload["candidate_count"]) != batch.candidate_count:
            raise AssertionError("R1 native candidate count drift")
        for name in ("z0", "z1", "z4"):
            value = payload[name]
            if value.shape != (1, batch.candidate_count, 256) or not bool(torch.isfinite(value).all()):
                raise AssertionError(f"R1 native {name} shape/finite drift: {tuple(value.shape)}")
        if not all(not parameter.requires_grad for parameter in runtime.model.parameters()):
            raise AssertionError("R1 native detector is not frozen")
        if int(runtime.native_visual_forward_count) != 1 or int(runtime.replay_count) != 1:
            raise AssertionError("R1 native/replay count contract drift")
        audit = {
            **status, "status": "complete", "elapsed_seconds": float(time.perf_counter() - started),
            "manifest_sha256": manifest_sha, "request_manifest": file_meta(request_path),
            "unit_key": unit_key(record), "candidate_count": int(batch.candidate_count),
            "row_offsets": list(batch.row_offsets), "candidate_indices": list(batch.candidate_indices),
            "row_keys": [list(value) for value in batch.row_keys],
            "native_visual_forward_count": int(runtime.native_visual_forward_count),
            "text_replay_count": int(runtime.replay_count),
            "payload_shapes": {key: list(value.shape) for key, value in payload.items() if torch.is_tensor(value)},
            "query_audits": payload["query_audits"],
            "model_info": runtime.model_info,
            "detector_parameters_frozen": True,
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "features_written_to_disk": False, "labels_read": False,
            "inputs": {"request_manifest": str(request_path), "record": record},
            "outputs": {"contract": str(out / "contract.json")},
        }
        write_json(out / "contract.json", audit)
        write_json(out / "provenance.json", {
            "format": "locatemot-r1-native-contract-provenance-v1", "status": "complete", "command": command,
            "cwd": str(Path.cwd().resolve()), "thread": THREAD, "seed": SEED,
            "manifest_sha256": manifest_sha, "inputs": audit["inputs"], "outputs": audit["outputs"],
            "labels_read": False, "features_written_to_disk": False,
            "failure_root_cause": None, "next_action": status["next_action"], **standard_flags(),
        })
        write_json(out / "status.json", {**status, "status": "complete", "elapsed_seconds": audit["elapsed_seconds"],
                                          "manifest_sha256": manifest_sha, "outputs": audit["outputs"]})
        return 0
    except Exception as exc:
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        status.update({"status": "incomplete", "elapsed_seconds": float(time.perf_counter() - started),
                       "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback": error,
                       "outputs": {"attempt": str(out)}})
        write_json(out / "status.json", status)
        (out / "INCOMPLETE.md").write_text(
            "# R1 native contract incomplete\n\nFirst actionable traceback:\n\n```text\n" + error +
            "```\n\nNo labels or feature cache were written by this attempt.\n", encoding="utf-8")
        raise
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
