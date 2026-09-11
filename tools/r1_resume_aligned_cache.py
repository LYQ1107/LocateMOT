#!/usr/bin/env python3
"""Resume an interrupted R1 aligned cache without rewriting prior payloads.

The original builder is intentionally non-resumable.  This helper creates a
new attempt manifest, validates the durable prefix from an interrupted attempt,
references those existing tensor payloads in place, and computes only the
remaining frame groups.  It never opens GT/candidate-label sidecars.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.rmot.r1_aligned_cache import R1BankStore, R1GroundingRuntime, row_digest, write_json  # noqa: E402
from tools.build_r1_aligned_cache import _group_requests, _public_requests, _safe_tensor_path  # noqa: E402
from tools.r1_common import MANIFEST_SHA, SEED, THREAD, check_manifest, file_meta, standard_flags, unit_key  # noqa: E402


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_prior(prior_out: Path, ordered_groups: list[tuple[str, str, int, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    path = prior_out / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("interrupted R1 attempt has no durable manifest rows")
    expected = [f"{dataset}|{video}|{frame}" for dataset, video, frame, _ in ordered_groups]
    keys = [str(row.get("group_key")) for row in rows]
    if keys != expected[: len(keys)]:
        raise AssertionError("prior R1 manifest is not an ordered durable prefix")
    if len(set(keys)) != len(keys):
        raise AssertionError("prior R1 manifest has duplicate groups")
    for row in rows:
        if row.get("status") != "complete" or row.get("candidate_deletion") or row.get("candidate_truncation"):
            raise AssertionError(f"invalid durable R1 row: {row.get('group_key')}")
        tensor_path = Path(str(row.get("path", "")))
        if not tensor_path.is_file():
            raise FileNotFoundError(f"missing durable R1 tensor: {tensor_path}")
        if row.get("labels_in_cache") is not False or row.get("query_independent") is not False:
            raise AssertionError(f"invalid R1 label/cache flags: {row.get('group_key')}")
    return rows


def _append_group(
    *,
    manifest_handle: Any,
    status: dict[str, Any],
    store: R1BankStore,
    runtime: R1GroundingRuntime,
    dataset: str,
    video: str,
    frame_id: int,
    records_for_frame: list[dict[str, Any]],
    tensor_root: Path,
    query_batch_size: int,
    group_index: int,
    started: float,
    out: Path,
) -> None:
    batches = []
    for record in records_for_frame:
        batches.append(store.build_frame(str(record["dataset"]), video, int(record["query_id"]), frame_id,
                                          str(record["sentence"])))
    first = batches[0]
    for batch in batches[1:]:
        if (batch.row_offsets != first.row_offsets or batch.candidate_indices != first.candidate_indices or
                batch.track_ids != first.track_ids or batch.pool_ids != first.pool_ids or
                not torch.equal(batch.boxes, first.boxes)):
            raise AssertionError(f"R1 same-frame candidate row drift: {video}:{frame_id}")
    expected_keys = [unit_key(record) for record in records_for_frame]
    actual_keys = [f"{batch.dataset}|{batch.video}|{batch.query_id}|{batch.frame_id}" for batch in batches]
    if actual_keys != expected_keys:
        raise AssertionError(f"R1 request/batch order drift: {video}:{frame_id}")
    payload = runtime.extract_group(batches, query_batch_size=int(query_batch_size))
    if list(payload["query_unit_keys"]) != expected_keys:
        raise AssertionError(f"R1 cache query key order drift: {video}:{frame_id}")
    if int(payload["candidate_count"]) != first.candidate_count or list(payload["row_offsets"]) != first.row_offsets:
        raise AssertionError(f"R1 cache candidate row drift: {video}:{frame_id}")
    if str(payload.get("persistent_payload", "")).find("no labels/GT/scores") < 0:
        raise AssertionError("R1 cache payload is not explicitly label-free")
    tensor_path = _safe_tensor_path(tensor_root, records_for_frame[0])
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tensor_path.with_suffix(".tmp.pt")
    if temporary.exists() or tensor_path.exists():
        raise FileExistsError(f"R1 resumed tensor collision: {tensor_path}")
    torch.save(payload, temporary)
    os.replace(temporary, tensor_path)
    row = {
        "format": "locatemot-r1-aligned-cache-manifest-row-v1", "status": "complete",
        "group_key": f"{dataset}|{video}|{frame_id}", "dataset": dataset, "video": video,
        "frame_id": int(frame_id), "path": str(tensor_path),
        "query_unit_keys": list(payload["query_unit_keys"]),
        "query_ids": [int(value["query_id"]) for value in records_for_frame],
        "candidate_count": int(first.candidate_count), "row_offsets": list(first.row_offsets),
        "row_key_digest": row_digest(first.row_keys), "candidate_deletion": False,
        "candidate_truncation": False, "labels_in_cache": False, "query_independent": False,
    }
    manifest_handle.write(json.dumps(row, sort_keys=True) + "\n")
    manifest_handle.flush()
    status["groups_completed"] = int(group_index + 1)
    status["query_records_completed"] = int(status["query_records_completed"] + len(records_for_frame))
    if (group_index + 1) % 25 == 0:
        write_json(out / "progress.json", {**status, "elapsed_seconds": time.perf_counter() - started})
    del payload, batches
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-manifest", type=Path, required=True)
    parser.add_argument("--prior-out", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tensor-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--max-new-groups", type=int, default=0)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else WORK_ROOT / args.out).resolve()
    prior_out = (args.prior_out if args.prior_out.is_absolute() else WORK_ROOT / args.prior_out).resolve()
    request_path = (args.request_manifest if args.request_manifest.is_absolute() else WORK_ROOT / args.request_manifest).resolve()
    tensor_root = args.tensor_root.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R1 resume output: {out}")
    if tensor_root.exists() and any(tensor_root.iterdir()):
        raise FileExistsError(f"refusing nonempty R1 resume tensor root: {tensor_root}")
    if args.max_new_groups < 0:
        raise ValueError("max-new-groups must be nonnegative")
    out.mkdir(parents=True, exist_ok=True)
    tensor_root.mkdir(parents=True, exist_ok=True)
    command = " ".join([str(sys.executable), *sys.argv])
    started = time.perf_counter()
    status: dict[str, Any] = {
        "format": "locatemot-r1-aligned-cache-resume-status-v1", "status": "running", "command": command,
        "cwd": str(Path.cwd().resolve()), "thread": THREAD, "seed": SEED,
        "request_manifest": str(request_path), "prior_attempt": str(prior_out), "cache_root": str(tensor_root),
        "groups_total": None, "groups_reused": 0, "groups_completed": 0,
        "query_records_total": None, "query_records_reused": 0, "query_records_completed": 0,
        "labels_read": False, "candidate_deletion": False, "candidate_truncation": False,
        "failure_root_cause": None, "next_action": "run R1 feature/anchor contract after aligned cache completes",
        **standard_flags(training_run=False),
    }
    manifest_handle = None
    runtime: R1GroundingRuntime | None = None
    store = R1BankStore()
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R1 resume cwd: {Path.cwd()}")
        if check_manifest() != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        records = _public_requests(request_path)
        groups = _group_requests(records)
        ordered_groups = [(dataset, video, frame, groups[(dataset, video, frame)])
                          for dataset, video, frame in sorted(groups)]
        prior_rows = _read_prior(prior_out, ordered_groups)
        status["groups_total"] = len(ordered_groups)
        status["groups_reused"] = len(prior_rows)
        status["groups_completed"] = len(prior_rows)
        status["query_records_total"] = sum(len(values) for _, _, _, values in ordered_groups)
        status["query_records_reused"] = sum(len(row.get("query_unit_keys", [])) for row in prior_rows)
        status["query_records_completed"] = status["query_records_reused"]
        manifest_handle = (out / "manifest.jsonl").open("w", encoding="utf-8")
        for row in prior_rows:
            manifest_handle.write(json.dumps(row, sort_keys=True) + "\n")
        manifest_handle.flush()
        remaining = ordered_groups[len(prior_rows):]
        if args.max_new_groups:
            remaining = remaining[: int(args.max_new_groups)]
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("R1 resumed cache requires CUDA")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        runtime = R1GroundingRuntime(device)
        for local_index, (dataset, video, frame_id, records_for_frame) in enumerate(remaining):
            _append_group(manifest_handle=manifest_handle, status=status, store=store, runtime=runtime,
                          dataset=dataset, video=video, frame_id=frame_id, records_for_frame=records_for_frame,
                          tensor_root=tensor_root, query_batch_size=int(args.query_batch_size),
                          group_index=len(prior_rows) + local_index, started=started, out=out)
        if len(remaining) != len(ordered_groups) - len(prior_rows):
            status["status"] = "partial"
            status["next_action"] = "resume this durable prefix in a new R1 attempt"
            status["elapsed_seconds"] = float(time.perf_counter() - started)
            status["manifest_sha256"] = check_manifest()
            write_json(out / "status.json", status)
            write_json(out / "provenance.json", {
                "format": "locatemot-r1-aligned-cache-resume-provenance-v1", **status,
                "request_manifest_sha256": sha256_file(request_path),
                "inputs": {"request_manifest": file_meta(request_path), "prior_manifest": file_meta(prior_out / "manifest.jsonl")},
                "outputs": {"manifest": str(out / "manifest.jsonl"), "tensor_root": str(tensor_root)},
                **standard_flags(training_run=False),
            })
            return 0
        manifest_handle.close(); manifest_handle = None
        elapsed = float(time.perf_counter() - started)
        final = {**status, "status": "complete", "elapsed_seconds": elapsed,
                 "manifest_sha256": check_manifest(), "request_manifest_sha256": sha256_file(request_path),
                 "query_records_completed": int(status["query_records_completed"]),
                 "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                 "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                 "outputs": {"manifest": str(out / "manifest.jsonl"), "tensor_root": str(tensor_root),
                             "status": str(out / "status.json")}}
        write_json(out / "provenance.json", {
            "format": "locatemot-r1-aligned-cache-resume-provenance-v1", **final,
            "inputs": {"request_manifest": file_meta(request_path), "prior_manifest": file_meta(prior_out / "manifest.jsonl")},
            "outputs": final["outputs"], "tensor_root_owner": "Luna R1", "persistent_payload": "FP16 Z0/Z1/Z4 and frame summaries; no labels/GT/scores",
            "query_category_metadata_used_for_features": False, "labels_read": False,
            "failure_root_cause": None, "next_action": status["next_action"], **standard_flags(training_run=False),
        })
        write_json(out / "status.json", final)
        return 0
    except Exception as exc:
        if manifest_handle is not None:
            manifest_handle.close()
        error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        status.update({"status": "incomplete", "elapsed_seconds": float(time.perf_counter() - started),
                       "failure_root_cause": f"{type(exc).__name__}: {exc}", "traceback": error,
                       "outputs": {"attempt": str(out), "tensor_root": str(tensor_root)}})
        write_json(out / "status.json", status)
        (out / "INCOMPLETE.md").write_text(
            "# R1 aligned cache resume incomplete\n\nFirst actionable traceback:\n\n```text\n" + error +
            "```\n\nPrior attempt payloads were referenced, not overwritten. Preserve this attempt.\n",
            encoding="utf-8")
        raise
    finally:
        store._blob = None
        if runtime is not None:
            runtime.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
