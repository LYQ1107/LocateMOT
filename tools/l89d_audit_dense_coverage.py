#!/usr/bin/env python3
"""Audit native full-video coverage of the compact, label-free Z1 cache."""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

from l89d_fullvideo_common import (
    ASSET_ROOT,
    BASE_Z1_CACHE,
    DEFAULT_LANGUAGE_CACHE,
    L80BankStore,
    L89LanguageTokenCache,
    MANIFEST_SHA,
    THREAD,
    WORK_ROOT,
    DenseZ1Resolver,
    Z1CacheIndex,
    command_line,
    expected_timeline_descriptor,
    file_meta,
    load_video_scopes,
    manifest_assertion,
    native_groups,
    sha256_file,
    standard_flags,
    validate_native_batch,
    write_json,
)


FORMAT = "locatemot-l89d-dense-coverage-v1"


def _write_failure(out: Path, command: str, exc: BaseException) -> None:
    out.mkdir(parents=True, exist_ok=True)
    trace = traceback.format_exc()
    (out / "INCOMPLETE.md").write_text(
        "# L89D dense coverage — INCOMPLETE\n\n" + trace, encoding="utf-8"
    )
    payload = {
        "format": FORMAT, "status": "incomplete", "command": command,
        "cwd": str(WORK_ROOT), "luna_thread": THREAD,
        "failure_root_cause": f"{type(exc).__name__}: {exc}",
        "next_action": "repair the first cache/timeline contract error and rerun in a new attempt",
        **standard_flags(hota_trackeval_run=False),
    }
    write_json(out / "status.json", payload)


def audit_scope(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty coverage output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = command_line()
    started = time.perf_counter()
    store: L80BankStore | None = None
    group_records_path = out / "group_records.jsonl"
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"L89D tools must run from the L89D worktree: {Path.cwd()}")
        manifest = manifest_assertion()
        if manifest["sha256"] != MANIFEST_SHA:
            raise AssertionError("manifest assertion drift")
        base_root = args.base_z1_cache.resolve()
        supplement_root = args.supplement_z1_cache.resolve() if args.supplement_z1_cache else None
        base = Z1CacheIndex(base_root)
        supplement = Z1CacheIndex(supplement_root) if supplement_root is not None and supplement_root.is_dir() else None
        resolver = DenseZ1Resolver(base_root, supplement_root)
        language = L89LanguageTokenCache(args.language_cache.resolve())
        store = L80BankStore(max_history=8)
        scopes = load_video_scopes(args.scope)
        timeline = expected_timeline_descriptor(args.scope, store)

        # Validate each legal query exactly once per unique cache key.  Query
        # IDs are lookup metadata only; language tensors never enter this
        # label-free coverage audit's candidate records.
        language_keys: set[tuple[str, str, int, str]] = set()
        language_records: list[dict[str, Any]] = []
        for video_scope in scopes:
            for query in video_scope.queries:
                key = (video_scope.dataset, video_scope.video, int(query["query_id"]), str(query["sentence"]))
                if key in language_keys:
                    continue
                item = language.get(video_scope.dataset, video_scope.video, int(query["query_id"]), str(query["sentence"]))
                language_keys.add(key)
                language_records.append({
                    "dataset": video_scope.dataset, "video": video_scope.video,
                    "query_id": int(query["query_id"]), "token_shape": list(item.tokens.shape),
                    "valid_tokens": int(item.mask.sum()), "finite": True,
                })
                del item

        missing: list[dict[str, Any]] = []
        resolved = 0
        expected_pairs = 0
        covered_pairs = 0
        timeline_videos: list[dict[str, Any]] = []
        with group_records_path.open("w", encoding="utf-8") as records_handle:
            for video_scope in scopes:
                groups = native_groups(video_scope, store)
                native_frame_ids = [int(group["frame_id"]) for group in groups]
                video_expected = len(native_frame_ids) * len(video_scope.queries)
                video_covered = 0
                video_resolved = 0
                for group in groups:
                    first = store.build_unit(dict(group["queries"][0]))
                    batch_audit = validate_native_batch(first)
                    key = str(group["group_key"])
                    expected_pairs += len(video_scope.queries)
                    source = None
                    contract: dict[str, Any] | None = None
                    error = None
                    try:
                        result = resolver.resolve(group, first)
                        source = str(result["source"])
                        contract = result["contract"]
                        video_covered += len(video_scope.queries)
                        video_resolved += 1
                        resolved += 1
                        del result
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        missing.append({
                            "group_key": key, "dataset": video_scope.dataset, "video": video_scope.video,
                            "frame_id": int(group["frame_id"]), "query_count": len(video_scope.queries),
                            "base_present": base.has(key), "supplement_present": supplement.has(key) if supplement else False,
                            "error": error,
                        })
                    row = {
                        "format": FORMAT, "status": "complete", "scope": args.scope,
                        "group_key": key, "dataset": video_scope.dataset, "video": video_scope.video,
                        "frame_id": int(group["frame_id"]), "query_count": len(video_scope.queries),
                        "expected_query_pairs": len(video_scope.queries), "covered_query_pairs": 0 if error else len(video_scope.queries),
                        "cache_source": source, "cache_contract": contract, "error": error,
                        "native": batch_audit, "labels_read": False,
                        "candidate_rows_retained": True, "candidate_deletion": False,
                        "candidate_truncation": False,
                    }
                    records_handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
                    del first, batch_audit
                    gc.collect()
                covered_pairs += video_covered
                timeline_videos.append({
                    "dataset": video_scope.dataset, "video": video_scope.video,
                    "native_frame_count": len(native_frame_ids), "legal_query_count": len(video_scope.queries),
                    "expected_query_frame_pairs": video_expected, "covered_query_frame_pairs": video_covered,
                    "missing_group_count": len(groups) - video_resolved,
                    "native_frame_ids_sha256": next(item["native_frame_ids_sha256"] for item in timeline["videos"] if item["video"] == video_scope.video and item["dataset"] == video_scope.dataset),
                })

        summary = {
            "format": FORMAT, "status": "complete", "scope": args.scope,
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD,
            "timeline": {"scope": args.scope, "videos": timeline_videos,
                          "expected_query_frame_pairs": expected_pairs,
                          "covered_query_frame_pairs": covered_pairs,
                          "missing_query_frame_pairs": expected_pairs - covered_pairs},
            "cache": {"base": base.summary_descriptor(),
                      "supplement": supplement.summary_descriptor() if supplement else None,
                      "resolver_precedence": ["supplement", "base"],
                      "base_invalid_files": base.invalid,
                      "supplement_invalid_files": supplement.invalid if supplement else {}},
            "frame_group_count": int(sum(len(native_groups(video_scope, store)) for video_scope in scopes)),
            "resolved_group_count": resolved, "missing_group_count": len(missing),
            "missing_groups": missing,
            "language": {"cache_root": str(args.language_cache.resolve()), "entry_count": language.entry_count,
                         "unique_legal_queries_checked": len(language_records), "records": language_records,
                         "complete": True},
            "dense_ready": not missing and expected_pairs == covered_pairs,
            "required_audit": "CPU/read-only; no labels, training, TrackEval, or dense feature construction",
            "inputs": {"manifest": manifest, "base_z1_cache": str(base_root),
                       "base_summary": file_meta(base_root / "summary.json"),
                       "supplement_z1_cache": str(supplement_root) if supplement_root else None,
                       "language_cache": file_meta(args.language_cache)},
            "outputs": {"coverage": str((out / "coverage.json").resolve()),
                        "group_records": str(group_records_path.resolve()),
                        "missing_groups": str((out / "missing_groups.jsonl").resolve())},
            "cache_labels_included": False, "labels_read": False,
            "candidate_deletion": False, "candidate_truncation": False,
            **standard_flags(hota_trackeval_run=False),
            "failure_root_cause": None if not missing else "native full-video groups missing or invalid in layered Z1 cache",
            "next_action": "run L89D dense supplement builder for missing groups" if missing else "run L89D true full-video dev smoke",
            "wall_seconds": time.perf_counter() - started,
        }
        with (out / "missing_groups.jsonl").open("w", encoding="utf-8") as handle:
            for item in missing:
                handle.write(json.dumps(item, sort_keys=True, ensure_ascii=False) + "\n")
        # Avoid serializing the redundant frame ID list into the coverage
        # artifact while retaining its hash and counts.
        summary["timeline"]["native_frame_ids_omitted_from_output"] = True
        write_json(out / "coverage.json", summary)
        provenance = dict(summary)
        provenance["provenance_note"] = "All cache and language checks are label-free; labels are not loaded by this tool."
        write_json(out / "provenance.json", provenance)
        write_json(out / "status.json", {key: summary[key] for key in (
            "format", "status", "scope", "dense_ready", "resolved_group_count", "missing_group_count",
            "timeline", "failure_root_cause", "next_action", "command", "cwd", "luna_thread",
        )} | standard_flags(hota_trackeval_run=False))
        return 0
    except Exception as exc:
        _write_failure(out, command, exc)
        raise
    finally:
        if store is not None:
            store._store._bank = None; store._store._text_cache = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("dev", "internal"), required=True)
    parser.add_argument("--base-z1-cache", type=Path, default=BASE_Z1_CACHE)
    parser.add_argument("--supplement-z1-cache", type=Path, default=None)
    parser.add_argument("--language-cache", type=Path, default=DEFAULT_LANGUAGE_CACHE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return audit_scope(args)


if __name__ == "__main__":
    raise SystemExit(main())
