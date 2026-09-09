#!/usr/bin/env python3
"""Build only the audited missing compact Z1 groups on a label-free basis."""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from l89d_fullvideo_common import (
    ASSET_ROOT,
    BASE_Z1_CACHE,
    L80BankStore,
    MANIFEST_SHA,
    THREAD,
    WORK_ROOT,
    capture_group_z1_batched,
    command_line,
    load_video_scopes,
    manifest_assertion,
    native_groups,
    sha256_file,
    standard_flags,
    validate_native_batch,
    write_json,
)


FORMAT = "locatemot-l89d-dense-z1-supplement-v1"
FORBIDDEN = {
    "target_ids", "positive_indices", "positive_count", "category", "labels",
    "candidate_gt", "candidate_scores", "coverage_mask", "declared_category",
}


def _read_missing(audit: Path, scope: str) -> list[dict[str, Any]]:
    payload = json.loads((audit.resolve() / "coverage.json").read_text(encoding="utf-8"))
    if payload.get("scope") != scope or payload.get("format") != "locatemot-l89d-dense-coverage-v1":
        raise AssertionError("coverage audit scope/format mismatch")
    if payload.get("labels_read") or payload.get("screening_gt_used") or payload.get("official_test_labels_read"):
        raise AssertionError("coverage audit crossed label boundary")
    missing = payload.get("missing_groups", [])
    if not isinstance(missing, list):
        raise AssertionError("coverage audit missing_groups is not a list")
    return sorted((dict(item) for item in missing), key=lambda item: str(item["group_key"]))


def _valid_existing(path: Path, group: dict[str, Any], batch: Any) -> bool:
    try:
        item = torch.load(path, map_location="cpu", weights_only=False)
        if FORBIDDEN.intersection(item) or item.get("labels_in_cache"):
            return False
        if item.get("candidate_deletion") or item.get("candidate_truncation"):
            return False
        if str(item.get("group_key")) != str(group["group_key"]):
            return False
        expected_qids = [int(row["query_id"]) for row in group["queries"]]
        if [int(x) for x in item.get("query_ids", [])] != expected_qids:
            return False
        if [str(x) for x in item.get("sentences", [])] != [str(row["sentence"]) for row in group["queries"]]:
            return False
        if int(item.get("candidate_count", -1)) != int(batch.candidate_count):
            return False
        if [int(x) for x in item.get("row_offsets", [])] != [int(x) for x in batch.row_offsets]:
            return False
        z1, text, frame = item.get("z1"), item.get("text_global"), item.get("frame_global")
        if not (torch.is_tensor(z1) and torch.is_tensor(text) and torch.is_tensor(frame)):
            return False
        if tuple(z1.shape) != (len(expected_qids), batch.candidate_count, 256):
            return False
        if tuple(text.shape) != (len(expected_qids), 256) or tuple(frame.shape) != (len(expected_qids), 256):
            return False
        return all(bool(torch.isfinite(value.float()).all()) for value in (z1, text, frame))
    except Exception:
        return False
    finally:
        gc.collect()


def _run_nvidia_smi() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.STDOUT, timeout=30,
        ).strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    command = command_line()
    started = time.perf_counter()
    store: L80BankStore | None = None
    runtime: Any | None = None
    shard_name = f"shard_{int(args.shard_index):03d}_of_{int(args.num_shards):03d}"
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong L89D cwd: {Path.cwd()}")
        if args.scope not in {"dev", "internal"}:
            raise ValueError(args.scope)
        if sha256_file(ASSET_ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json") != MANIFEST_SHA:
            raise AssertionError("fixed manifest SHA drift")
        if not args.audit.resolve().is_dir():
            raise FileNotFoundError(args.audit)
        missing_records = _read_missing(args.audit, args.scope)
        missing_keys = [str(item["group_key"]) for item in missing_records]
        if len(missing_keys) != len(set(missing_keys)):
            raise AssertionError("duplicate missing group keys in audit")
        if int(args.num_shards) < 1 or not 0 <= int(args.shard_index) < int(args.num_shards):
            raise ValueError("invalid shard selection")
        assigned = missing_keys[int(args.shard_index)::int(args.num_shards)]
        if args.max_groups > 0:
            assigned = assigned[: int(args.max_groups)]
        assigned_set = set(assigned)
        values = {
            "format": FORMAT, "status": "running", "scope": args.scope, "shard": shard_name,
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "audit": str(args.audit.resolve()), "audit_sha256": sha256_file(args.audit.resolve() / "coverage.json"),
            "base_z1_cache": str(args.base_z1_cache.resolve()), "assigned_group_count": len(assigned),
            "max_groups": int(args.max_groups), "device": str(args.device), "query_batch_size_requested": int(args.query_batch_size),
            "gpu_snapshot_before": _run_nvidia_smi(), "query_independent": True,
            "labels_in_cache": False, "raw_pixels_in_cache": False, "dense_detector_maps_in_cache": False,
            "candidate_deletion": False, "candidate_truncation": False, **standard_flags(hota_trackeval_run=False),
        }
        write_json(out / f"{shard_name}.running.json", values)
        scopes = load_video_scopes(args.scope)
        assigned_by_scope = {
            video_scope.scope_key: [
                key for key in assigned
                if key.startswith(f"{video_scope.dataset}|{video_scope.video}|")
            ]
            for video_scope in scopes
        }
        store = L80BankStore(max_history=8)
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("dense Z1 supplement requires the verified CUDA runtime")
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        from locatemot.rmot.l82_grounding_runtime import GroundingCandidateReferenceRuntime  # noqa: E402
        runtime = GroundingCandidateReferenceRuntime(device)
        item_root = out / "items"
        completed = 0; reused = 0; bytes_written = 0
        effective_batch = int(args.query_batch_size)
        status_rows: list[dict[str, Any]] = []
        processed_keys: set[str] = set()
        # Keep only one video's lightweight group metadata in memory.  The
        # previous implementation materialized every frame/query dictionary
        # for the whole scope before constructing the runtime; on this shared
        # host that was enough to trigger an exit-137 under low available RAM.
        for video_scope in scopes:
            scope_keys = assigned_by_scope.get(video_scope.scope_key, [])
            if not scope_keys:
                continue
            groups_by_key = {
                str(group["group_key"]): group
                for group in native_groups(video_scope, store)
            }
            missing_scope = set(scope_keys) - set(groups_by_key)
            if missing_scope:
                raise AssertionError(f"audit groups not in native scope: {sorted(missing_scope)[:5]}")
            for index, key in enumerate(scope_keys):
                group = groups_by_key[key]
                first = store.build_unit(dict(group["queries"][0]))
                validate_native_batch(first)
                dataset_dir = item_root / str(group["dataset"])
                dataset_dir.mkdir(parents=True, exist_ok=True)
                path = dataset_dir / f"{group['dataset']}__{group['video']}__{int(group['frame_id']):06d}.pt"
                if path.exists():
                    if not _valid_existing(path, group, first):
                        raise AssertionError(f"invalid existing supplement item: {path}")
                    reused += 1
                    status_rows.append({"group_key": key, "path": str(path), "status": "reused", "bytes": path.stat().st_size})
                    processed_keys.add(key)
                    del first
                    continue
                item = capture_group_z1_batched(group, device, runtime=runtime, bank_store=store,
                                                query_batch_size=effective_batch)
                if [int(x) for x in item["query_ids"]] != [int(row["query_id"]) for row in group["queries"]]:
                    raise AssertionError(f"query order drift: {key}")
                if int(item["candidate_count"]) != int(first.candidate_count) or [int(x) for x in item["row_offsets"]] != [int(x) for x in first.row_offsets]:
                    raise AssertionError(f"candidate contract drift: {key}")
                if FORBIDDEN.intersection(item) or item.get("labels_in_cache"):
                    raise AssertionError(f"labels in captured item: {key}")
                item.update({
                    "partition": args.scope, "capture_source": "L85 capture_group_z1_batched",
                    "audit_coverage_sha256": sha256_file(args.audit.resolve() / "coverage.json"),
                    "labels_in_cache": False, "candidate_deletion": False, "candidate_truncation": False,
                    "supplement_partition": args.scope, "zero_training": True,
                })
                if not all(bool(torch.isfinite(item[name].float()).all()) for name in ("z1", "text_global", "frame_global")):
                    raise FloatingPointError(f"nonfinite supplement item: {key}")
                tmp = path.with_suffix(path.suffix + ".tmp")
                torch.save(item, tmp)
                tmp.replace(path)
                completed += 1; bytes_written += path.stat().st_size
                processed_keys.add(key)
                status_rows.append({"group_key": key, "path": str(path), "status": "completed", "bytes": path.stat().st_size})
                del item, first
                gc.collect(); torch.cuda.empty_cache()
                if index % 10 == 0:
                    print(f"[l89d-z1] scope={args.scope} shard={shard_name} group={index + 1}/{len(scope_keys)}", flush=True)
            del groups_by_key
            gc.collect()
        if processed_keys != assigned_set:
            raise AssertionError(f"processed group set drift: missing={sorted(assigned_set - processed_keys)[:5]}")
        summary = {
            "format": FORMAT, "status": "complete", "scope": args.scope, "shard": shard_name,
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "audit": str(args.audit.resolve()), "audit_sha256": sha256_file(args.audit.resolve() / "coverage.json"),
            "assigned_group_count": len(assigned), "completed_group_count": completed, "reused_group_count": reused,
            "assigned_group_keys": assigned, "status_rows": status_rows, "bytes_written": bytes_written,
            "effective_query_batch_size": effective_batch, "device": str(device),
            "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "gpu_snapshot_before": values["gpu_snapshot_before"],
            "output_root": str(out), "item_root": str(item_root),
            "query_independent": True, "labels_in_cache": False, "raw_pixels_in_cache": False,
            "dense_detector_maps_in_cache": False, "candidate_deletion": False, "candidate_truncation": False,
            **standard_flags(hota_trackeval_run=False), "failure_root_cause": None,
            "next_action": "finalize supplement and rerun dense coverage audit",
            "wall_seconds": time.perf_counter() - started,
        }
        write_json(out / f"{shard_name}.json", summary)
        (out / f"{shard_name}.running.json").unlink(missing_ok=True)
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L89D dense Z1 supplement — INCOMPLETE\n\n" + trace, encoding="utf-8")
        write_json(out / "status.json", {
            "format": FORMAT, "status": "incomplete", "scope": args.scope, "command": command,
            "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "next_action": "repair first actionable supplement error and retry in a new output/attempt",
            **standard_flags(hota_trackeval_run=False),
        })
        raise
    finally:
        if runtime is not None:
            runtime.close()
        if store is not None:
            store._store._bank = None; store._store._text_cache = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("dev", "internal"), required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-z1-cache", type=Path, default=BASE_Z1_CACHE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-groups", type=int, default=0)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
