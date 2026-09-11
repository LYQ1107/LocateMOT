#!/usr/bin/env python3
"""Build the label-free, query-conditioned R1 Z0/Z1/Z4 cache.

The request manifest is created before this script and contains only public
query/frame metadata.  This builder never opens a target or candidate-GT
sidecar.  GroundingDINO runs one native visual forward per frame and replays
the requested expressions over that frozen visual result.  Tensor payloads
are written to the explicitly approved /data2 scratch volume; only the small
manifest and audit metadata live in the R1 worktree.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

WORK_ROOT = Path(__file__).resolve().parents[1]
if str(WORK_ROOT) not in sys.path:
    sys.path.insert(0, str(WORK_ROOT))

from locatemot.rmot.r1_aligned_cache import R1BankStore, R1GroundingRuntime, row_digest, write_json  # noqa: E402
from tools.r1_common import (  # noqa: E402
    MANIFEST_SHA, SEED, THREAD, check_manifest, file_meta, standard_flags, unit_key,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _public_requests(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    accepted_formats = {
        "locatemot-r1-request-manifest-v1",
        "locatemot-r1-eval-request-manifest-v1",
    }
    if payload.get("format") not in accepted_formats or payload.get("status") != "complete":
        raise AssertionError("invalid R1 request manifest")
    if int(payload.get("seed", -1)) != SEED:
        raise AssertionError("R1 request seed/epoch contract drift")
    if payload.get("format") == "locatemot-r1-request-manifest-v1" and int(payload.get("epochs", -1)) != 6:
        raise AssertionError("R1 fit request epoch contract drift")
    result: list[dict[str, Any]] = []
    forbidden = {"target_ids", "positive_indices", "labels", "candidate_gt", "track_id", "state_key"}
    for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
        for value in payload.get("unique_query_records", {}).get(dataset, {}).values():
            if forbidden.intersection(value):
                raise AssertionError(f"forbidden label/identity field in request: {unit_key(value)}")
            result.append({key: item for key, item in value.items() if key != "category"})
    if len({unit_key(value) for value in result}) != len(result):
        raise AssertionError("duplicate R1 request unit key")
    if not result:
        raise AssertionError("empty R1 request manifest")
    return result


def _group_requests(records: list[dict[str, Any]]) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (str(record["dataset"]), str(record["video"]), int(record["frame_id"]))
        groups[key].append(record)
    for (dataset, video, frame), values in groups.items():
        values.sort(key=lambda value: (int(value["query_id"]), str(value["unit_key"])))
        if any(str(value["dataset"]) != dataset or str(value["video"]) != video or int(value["frame_id"]) != frame for value in values):
            raise AssertionError("R1 frame group identity drift")
    return groups


def _safe_tensor_path(tensor_root: Path, first: dict[str, Any]) -> Path:
    return tensor_root / str(first["dataset"]) / str(first["video"]) / f"{int(first['frame_id']):06d}.pt"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-manifest", type=Path, default=Path("outputs/r1/request_manifest_attempt1/request_manifest.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tensor-root", type=Path, required=True,
                        help="explicit /data2 root for compact FP16 tensor payloads")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--max-groups", type=int, default=0,
                        help="contract-only bound; zero means all requested groups")
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else WORK_ROOT / args.out).resolve()
    tensor_root = args.tensor_root.resolve()
    request_path = (args.request_manifest if args.request_manifest.is_absolute() else WORK_ROOT / args.request_manifest).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty R1 cache output: {out}")
    if tensor_root.exists() and any(tensor_root.iterdir()):
        raise FileExistsError(f"refusing nonempty R1 tensor root: {tensor_root}")
    out.mkdir(parents=True, exist_ok=True)
    tensor_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.jsonl"
    command = " ".join([str(sys.executable), *sys.argv])
    started = time.perf_counter()
    status: dict[str, Any] = {
        "format": "locatemot-r1-aligned-cache-status-v1", "status": "running", "command": command,
        "cwd": str(Path.cwd().resolve()), "thread": THREAD, "seed": SEED,
        "request_manifest": str(request_path), "cache_root": str(tensor_root),
        "groups_total": None, "groups_completed": 0, "query_records_total": None,
        "labels_read": False, "candidate_deletion": False, "candidate_truncation": False,
        "failure_root_cause": None, "next_action": "run R1 feature/anchor contract after aligned cache completes",
        **standard_flags(training_run=False),
    }
    runtime: R1GroundingRuntime | None = None
    store = R1BankStore()
    manifest_handle = None
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong R1 cache-builder cwd: {Path.cwd()}")
        manifest_sha = check_manifest()
        if manifest_sha != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        records = _public_requests(request_path)
        groups = _group_requests(records)
        ordered_groups = [(dataset, video, frame, values) for dataset, video, frame in sorted(groups)
                          for values in [groups[(dataset, video, frame)]]]
        if args.max_groups:
            if int(args.max_groups) < 1:
                raise ValueError("max-groups must be positive when provided")
            ordered_groups = ordered_groups[: int(args.max_groups)]
        status["groups_total"] = len(ordered_groups)
        status["query_records_total"] = sum(len(values) for _dataset, _video, _frame, values in ordered_groups)
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("R1 aligned cache requires CUDA")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        runtime = R1GroundingRuntime(device)
        manifest_handle = manifest_path.open("w", encoding="utf-8")
        for group_index, (dataset, video, frame_id, records_for_frame) in enumerate(ordered_groups):
            batches = []
            for record in records_for_frame:
                # Request categories are sampler metadata and are not passed
                # to the frame builder or stored in the feature payload.
                batches.append(store.build_frame(str(record["dataset"]), video, int(record["query_id"]),
                                                 frame_id, str(record["sentence"])))
            first = batches[0]
            for batch in batches[1:]:
                if (batch.row_offsets != first.row_offsets or batch.candidate_indices != first.candidate_indices or
                        batch.track_ids != first.track_ids or batch.pool_ids != first.pool_ids or
                        not torch.equal(batch.boxes, first.boxes)):
                    raise AssertionError(f"R1 same-frame candidate row drift: {video}:{frame_id}")
            expected_keys = [unit_key(record) for record in records_for_frame]
            if [f"{batch.dataset}|{batch.video}|{batch.query_id}|{batch.frame_id}" for batch in batches] != expected_keys:
                raise AssertionError(f"R1 request/batch order drift: {video}:{frame_id}")
            payload = runtime.extract_group(batches, query_batch_size=int(args.query_batch_size))
            query_keys = list(payload["query_unit_keys"])
            if query_keys != expected_keys:
                raise AssertionError(f"R1 cache query key order drift: {video}:{frame_id}")
            if int(payload["candidate_count"]) != first.candidate_count or list(payload["row_offsets"]) != first.row_offsets:
                raise AssertionError(f"R1 cache candidate row drift: {video}:{frame_id}")
            if str(payload.get("persistent_payload", "")).find("no labels/GT/scores") < 0:
                raise AssertionError("R1 cache payload is not explicitly label-free")
            tensor_path = _safe_tensor_path(tensor_root, records_for_frame[0])
            tensor_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = tensor_path.with_suffix(".tmp.pt")
            if temporary.exists() or tensor_path.exists():
                raise FileExistsError(f"R1 cache tensor collision: {tensor_path}")
            torch.save(payload, temporary)
            os.replace(temporary, tensor_path)
            manifest_row = {
                "format": "locatemot-r1-aligned-cache-manifest-row-v1", "status": "complete",
                "group_key": f"{dataset}|{video}|{frame_id}",
                "dataset": dataset, "video": video, "frame_id": int(frame_id),
                "path": str(tensor_path), "query_unit_keys": query_keys,
                "query_ids": [int(value["query_id"]) for value in records_for_frame],
                "candidate_count": int(first.candidate_count), "row_offsets": list(first.row_offsets),
                "row_key_digest": row_digest(first.row_keys), "candidate_deletion": False,
                "candidate_truncation": False, "labels_in_cache": False, "query_independent": False,
            }
            manifest_handle.write(json.dumps(manifest_row, sort_keys=True) + "\n")
            manifest_handle.flush()
            status["groups_completed"] = group_index + 1
            if (group_index + 1) % 25 == 0:
                write_json(out / "progress.json", {**status, "elapsed_seconds": time.perf_counter() - started})
            del payload, batches
            # Keep the immutable L69 bank resident while iterating frames of
            # one video.  Clearing only ``_blob`` here would leave the
            # matching ``_video`` sentinel set and make the next frame skip
            # reload, which is a contract error.  ``load_video`` replaces the
            # previous bank when the sorted video key changes; the finalizer
            # releases the last one.
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        manifest_handle.close(); manifest_handle = None
        elapsed = float(time.perf_counter() - started)
        final = {
            **status, "status": "complete", "elapsed_seconds": elapsed,
            "manifest_sha256": manifest_sha, "request_manifest_sha256": sha256_file(request_path),
            "groups_completed": int(status["groups_completed"]),
            "query_records_completed": int(sum(len(values) for _dataset, _video, _frame, values in ordered_groups)),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "outputs": {"manifest": str(manifest_path), "tensor_root": str(tensor_root), "status": str(out / "status.json")},
        }
        write_json(out / "provenance.json", {
            "format": "locatemot-r1-aligned-cache-provenance-v1", "status": "complete", "command": command,
            "cwd": str(Path.cwd().resolve()), "thread": THREAD, "seed": SEED,
            "manifest_sha256": manifest_sha, "inputs": {"request_manifest": file_meta(request_path)},
            "outputs": final["outputs"], "tensor_root": str(tensor_root), "tensor_root_owner": "Luna R1",
            "query_category_metadata_used_for_features": False, "labels_read": False,
            "persistent_payload": "FP16 Z0/Z1/Z4 and frame summaries; no labels/GT/scores",
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
            "# R1 aligned cache incomplete\n\nFirst actionable traceback:\n\n```text\n" + error +
            "```\n\nNo target/GT sidecar was opened. Preserve this attempt and repair only the first root cause.\n",
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
