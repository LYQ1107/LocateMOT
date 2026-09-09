#!/usr/bin/env python3
"""L89D inference over every native L69 frame/query pair.

This is a new timeline loop.  It deliberately does not call the sparse L89
scope helper.  The frozen L89 model and corrected L89C emission equation are
reused only as-is; this file adds no model, tracker, or threshold logic.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from l89d_fullvideo_common import (
    ASSET_ROOT,
    BASE_Z1_CACHE,
    DEFAULT_LANGUAGE_CACHE,
    L80BankStore,
    L89LanguageTokenCache,
    MANIFEST_SHA,
    RULES,
    SEED,
    THREAD,
    WORK_ROOT,
    DenseZ1Resolver,
    command_line,
    expected_timeline_descriptor,
    file_meta,
    load_video_scopes,
    manifest_assertion,
    native_groups,
    native_row_key,
    sha256_file,
    standard_flags,
    validate_native_batch,
    write_json,
)

# These functions only read legal fit/dev or internal validation labels after
# all prediction rows for the frozen strategy have been written.  The sparse
# scope loop itself is not reused.
from l89_infer_fullvideo import (  # noqa: E402
    emission_descriptor,
    materialize_gt,
    prepare_strategy,
    sequence_id,
    sigmoid,
)
from l88c_eval_metrics import corrected_emission_mask  # noqa: E402
from locatemot.models.l89_full_rmot import L89Config, L89FullRMOT  # noqa: E402


FORMAT = "locatemot-l89d-true-fullvideo-v1"
AUDIT_FORMAT = "locatemot-l89d-true-fullvideo-audit-v1"
DEV_SHORTLIST = Path(
    "/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89C/outputs/l89c/dev/final_selection_attempt1/checkpoint_selection.json"
).resolve()
L89C_EPOCH2_SHA = "f8b175597ece8aad1f0ec3ae9d05c70e0759a0f7480dff9af6ea2619a2e3f08b"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def _load_model(checkpoint_info: dict[str, Any], device: torch.device) -> tuple[L89FullRMOT, dict[str, Any]]:
    path = Path(str(checkpoint_info["path"])).resolve()
    observed = sha256_file(path)
    if observed != str(checkpoint_info["sha256"]):
        raise AssertionError(f"checkpoint SHA drift: {path}")
    package = torch.load(path, map_location="cpu", weights_only=False)
    if package.get("format") != "locatemot-l89-checkpoint-v1" or int(package.get("seed", -1)) != SEED:
        raise AssertionError(f"invalid L89 checkpoint package: {path}")
    if str(package.get("manifest_sha256")) != MANIFEST_SHA:
        raise AssertionError("checkpoint manifest SHA drift")
    model = L89FullRMOT(L89Config(**package["model_config"])).to(device=device, dtype=torch.float32)
    loaded = model.load_state_dict(package["model_state_dict"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise AssertionError(f"strict L89 reload failed: {loaded}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, {
        "path": str(path), "sha256": observed, "epoch": int(package["epoch"]),
        "optimizer_step": int(package["optimizer_step"]), "phase": str(package["phase"]),
        "model_config": package["model_config"], "strict_reload": True,
    }


def _validate_rule_fits(candidate: dict[str, Any], rules: tuple[str, ...]) -> None:
    fit_map = candidate.get("rule_fits")
    if not isinstance(fit_map, dict):
        raise AssertionError("candidate rule_fits missing")
    for rule in rules:
        item = fit_map.get(rule)
        if not isinstance(item, dict):
            raise AssertionError(f"candidate rule missing: {rule}")
        for name in ("candidate_threshold", "presence_threshold", "null_margin"):
            value = float(item[name])
            if not np.isfinite(value):
                raise AssertionError(f"nonfinite frozen threshold: {rule}/{name}")


def _load_dev_source(path: Path) -> list[dict[str, Any]]:
    source = _load_json(path)
    if source.get("status") != "complete" or not isinstance(source.get("shortlist"), list):
        raise AssertionError("L89C dev shortlist source incomplete")
    if str(source.get("evaluation_contract")) != "candidate_energy_vs_null":
        raise AssertionError("L89C source evaluation contract drift")
    if str(source.get("protocol_repair_stage")) != "L89C":
        raise AssertionError("L89C source protocol-repair stage drift")
    if len(source["shortlist"]) < 1:
        raise AssertionError("empty L89C shortlist")
    selected: list[dict[str, Any]] = []
    for index, raw in enumerate(source["shortlist"]):
        candidate = dict(raw)
        candidate["shortlist_index"] = int(candidate.get("shortlist_index", index))
        checkpoint = dict(candidate.get("checkpoint_info") or {})
        path_value = Path(str(checkpoint["path"])).resolve()
        if not path_value.is_file() or sha256_file(path_value) != str(checkpoint["sha256"]):
            raise AssertionError(f"L89C shortlist checkpoint drift: {path_value}")
        if int(checkpoint.get("epoch", -1)) <= 0:
            raise AssertionError("invalid shortlist epoch")
        candidate["checkpoint_info"] = checkpoint
        _validate_rule_fits(candidate, RULES)
        selected.append(candidate)
    # The chosen L89C epoch-2 source is explicitly checked as provenance; the
    # shortlist itself remains the complete dev matrix input.
    if not any(str(item["checkpoint_info"]["sha256"]) == L89C_EPOCH2_SHA for item in selected):
        raise AssertionError("L89C source shortlist lost epoch-2 checkpoint")
    return selected


def _load_internal_source(path: Path) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    source = _load_json(path)
    if source.get("status") != "complete":
        raise AssertionError("L89D internal selection incomplete")
    if str(source.get("protocol_repair_stage")) != "L89D":
        raise AssertionError("internal selection is not an L89D selection")
    if not bool(source.get("true_fullvideo_timeline")):
        raise AssertionError("internal selection lacks true-full-video timeline proof")
    final = source.get("final_selection")
    if not isinstance(final, dict):
        raise AssertionError("L89D final selection missing")
    rule = str(final.get("rule"))
    if rule not in RULES:
        raise AssertionError(f"invalid L89D internal rule: {rule}")
    candidate = {
        "shortlist_index": 0, "reason": "frozen L89D dev TrackEval selection",
        "checkpoint_info": dict(final["checkpoint_info"]),
        "rule_fits": {rule: dict(final["rule_object"])},
        "selection_source": str(path.resolve()), "selection_source_sha256": sha256_file(path.resolve()),
        "selection_frozen_before_fixed_validation": True,
    }
    _validate_rule_fits(candidate, (rule,))
    return [candidate], (rule,)


def _read_coverage(audit: Path, scope: str) -> dict[str, Any]:
    payload = _load_json(audit / "coverage.json")
    if payload.get("format") != "locatemot-l89d-dense-coverage-v1" or payload.get("scope") != scope:
        raise AssertionError("dense coverage audit format/scope mismatch")
    if not bool(payload.get("dense_ready")):
        raise AssertionError("required dense coverage audit is not ready")
    if payload.get("labels_read") or payload.get("screening_gt_used") or payload.get("official_test_labels_read"):
        raise AssertionError("dense coverage audit crossed label boundary")
    timeline = payload.get("timeline") or {}
    if int(timeline.get("missing_query_frame_pairs", -1)) != 0:
        raise AssertionError("required coverage audit has missing native pairs")
    return payload


def _load_language_once(cache: L89LanguageTokenCache, scopes: list[Any]) -> dict[tuple[str, str, int], tuple[torch.Tensor, torch.Tensor]]:
    result: dict[tuple[str, str, int], tuple[torch.Tensor, torch.Tensor]] = {}
    for video_scope in scopes:
        for row in video_scope.queries:
            key = (str(video_scope.dataset), str(video_scope.video), int(row["query_id"]))
            if key in result:
                continue
            item = cache.get(video_scope.dataset, video_scope.video, int(row["query_id"]), str(row["sentence"]))
            result[key] = (item.tokens.float().clone(), item.mask.bool().clone())
            del item
    return result


def _language_batch(
    language: dict[tuple[str, str, int], tuple[torch.Tensor, torch.Tensor]],
    dataset: str, video: str, rows: list[dict[str, Any]], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    items = [language[(dataset, video, int(row["query_id"]))] for row in rows]
    length = max(int(tokens.shape[0]) for tokens, _mask in items)
    tokens = torch.zeros((len(items), length, 256), dtype=torch.float32)
    mask = torch.zeros((len(items), length), dtype=torch.bool)
    for index, (value, valid) in enumerate(items):
        size = int(value.shape[0])
        tokens[index, :size] = value
        mask[index, :size] = valid
    return tokens.to(device=device), mask.to(device=device)


def _prepare_debug_groups(
    scopes: list[Any], store: L80BankStore, video_filter: str | None, max_frames: int, max_queries: int,
) -> tuple[list[Any], dict[str, list[dict[str, Any]]], bool]:
    selected_scopes: list[Any] = []
    groups_by_scope: dict[str, list[dict[str, Any]]] = {}
    debug = bool(video_filter or max_frames > 0 or max_queries > 0)
    for video_scope in scopes:
        if video_filter and video_scope.video != str(video_filter):
            continue
        queries = list(video_scope.queries)
        if max_queries > 0:
            queries = queries[: int(max_queries)]
        if not queries:
            continue
        scoped = type(video_scope)(video_scope.dataset, video_scope.video, tuple(queries))
        groups = native_groups(scoped, store)
        if max_frames > 0:
            groups = groups[: int(max_frames)]
        selected_scopes.append(scoped)
        groups_by_scope[scoped.scope_key] = groups
    if not selected_scopes:
        raise AssertionError("debug/full scope selected no legal videos")
    return selected_scopes, groups_by_scope, debug


def _query_map_from_scopes(scopes: list[Any]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Aggregate key-only queries without overwriting same-dataset videos."""
    result: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for scope in scopes:
        if scope.dataset in result and scope.video in result[scope.dataset]:
            raise AssertionError(f"duplicate selected video scope: {scope.scope_key}")
        result[str(scope.dataset)][str(scope.video)] = [dict(row) for row in scope.queries]
    return {dataset: {video: values for video, values in sorted(videos.items())}
            for dataset, videos in sorted(result.items())}


def _write_prediction(
    path: Path, frame_id: int, track_id: int, box: torch.Tensor, score: float,
) -> None:
    x1, y1, x2, y2 = [float(value) for value in box.tolist()]
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{int(frame_id) + 1},{int(track_id)},{x1:.6f},{y1:.6f},{x2 - x1:.6f},{y2 - y1:.6f},"
            f"{sigmoid(score):.8f},1,1,1\n"
        )


def _infer_candidate(
    candidate: dict[str, Any], rules: tuple[str, ...], args: argparse.Namespace,
    coverage: dict[str, Any], out: Path,
) -> dict[str, Any]:
    device = torch.device(args.device)
    scopes = load_video_scopes(args.scope)
    store = L80BankStore(max_history=8)
    selected_scopes, groups_by_scope, debug = _prepare_debug_groups(
        scopes, store, args.video, int(args.max_frames), int(args.max_queries)
    )
    timeline = expected_timeline_descriptor(args.scope, store)
    legal_scope_keys = {scope.scope_key for scope in scopes}
    selected_scope_keys = {scope.scope_key for scope in selected_scopes}
    if not debug and selected_scope_keys != legal_scope_keys:
        raise AssertionError("formal scope did not include every legal video")
    language_cache = L89LanguageTokenCache(args.language_cache.resolve())
    language = _load_language_once(language_cache, selected_scopes)
    resolver = DenseZ1Resolver(args.base_z1_cache.resolve(), args.supplement_z1_cache.resolve() if args.supplement_z1_cache else None)
    model, loaded = _load_model(candidate["checkpoint_info"], device)
    epoch = int(loaded["epoch"])
    candidate_root = out / f"candidate_epoch{epoch:03d}_shortlist{int(candidate.get('shortlist_index', 0)):02d}"
    query_map = _query_map_from_scopes(selected_scopes)
    strategy_paths = {rule: prepare_strategy(candidate_root / rule, sorted(query_map), query_map) for rule in rules}
    audit_path = candidate_root / "prediction_audits.jsonl"
    audit_handle = audit_path.open("w", encoding="utf-8")
    counters: dict[str, dict[str, Any]] = {}
    for rule in rules:
        counters[rule] = {
            "native_frame_visits": 0, "query_frame_pairs_scored": 0,
            "candidate_rows_scored": 0, "selected_rows": 0,
            "candidate_rows_retained": True, "candidate_deletion": False,
            "candidate_truncation": False, "key_digest": hashlib.sha256(),
            "per_video": {},
        }
    video_pair_actual: dict[str, int] = defaultdict(int)
    video_frame_visits: dict[str, int] = defaultdict(int)
    video_candidate_rows: dict[str, int] = defaultdict(int)
    video_selected_rows: dict[tuple[str, str], int] = defaultdict(int)
    actual_pairs = 0; actual_rows = 0
    group_keys_seen: set[str] = set()
    started = time.perf_counter()
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for scope in selected_scopes:
            groups = groups_by_scope[scope.scope_key]
            for group_index, group in enumerate(groups, 1):
                rows = [dict(row) for row in group["queries"]]
                first = store.build_unit(rows[0])
                native = validate_native_batch(first)
                resolved = resolver.resolve(group, first)
                expected_qids = [int(row["query_id"]) for row in rows]
                if [int(x) for x in resolved["query_ids"]] != expected_qids:
                    raise AssertionError(f"resolver query order drift: {group['group_key']}")
                n = int(first.candidate_count)
                z1_cpu = resolved["z1"]; tg_cpu = resolved["text_global"]; fg_cpu = resolved["frame_global"]
                if tuple(z1_cpu.shape) != (len(rows), n, 256):
                    raise AssertionError(f"resolved Z1 shape drift: {group['group_key']}")
                current = first.observations.float().clone().to(device)
                history = first.history_observations.float().clone().to(device)
                history_mask = first.history_mask.bool().clone().to(device)
                history_frames = first.history_frame_ids.long().clone().to(device)
                if int((history_frames > int(first.frame_id)).sum()) != 0:
                    raise AssertionError(f"future history at {group['group_key']}")
                for begin in range(0, len(rows), int(args.query_tile)):
                    chunk = rows[begin:begin + int(args.query_tile)]
                    stop = begin + len(chunk)
                    text_tokens, text_mask = _language_batch(language, scope.dataset, scope.video, chunk, device)
                    z1 = z1_cpu[begin:stop].float().clone().to(device)
                    text_global = tg_cpu[begin:stop].float().clone().to(device)
                    frame_global = fg_cpu[begin:stop].float().clone().to(device)
                    with torch.inference_mode():
                        output = model(
                            z1, text_tokens, text_mask, text_global, frame_global,
                            current, history, history_mask, history_frames, int(first.frame_id), temporal_enabled=True,
                        )
                    scores = output["candidate_energy"].float().detach().cpu().numpy()
                    presence = output["presence_logit"].float().detach().cpu().numpy()
                    null = output["null_logit"].float().detach().cpu().numpy()
                    if scores.shape != (len(chunk), n) or not np.isfinite(scores).all():
                        raise AssertionError(f"score shape/finite drift: {group['group_key']}")
                    if not np.isfinite(presence).all() or not np.isfinite(null).all():
                        raise FloatingPointError(f"presence/null nonfinite: {group['group_key']}")
                    for local, row in enumerate(chunk):
                        qid = int(row["query_id"])
                        keys = native_row_key(first, qid)
                        if len(keys) != n or [int(value[-1]) for value in keys] != [int(value) for value in first.row_offsets]:
                            raise AssertionError(f"row-key order drift: {row['unit_key']}")
                        for rule in rules:
                            rule_object = candidate["rule_fits"][rule]
                            c_threshold = float(rule_object["candidate_threshold"])
                            p_threshold = float(rule_object["presence_threshold"])
                            margin = float(rule_object["null_margin"])
                            selected_mask = corrected_emission_mask(
                                scores[local], float(presence[local]), float(null[local]),
                                c_threshold, p_threshold, margin,
                            )
                            selected = np.flatnonzero(selected_mask)
                            path = strategy_paths[rule][scope.dataset]["tracker_data"] / f"{sequence_id(scope.video, qid)}.txt"
                            for index in selected.tolist():
                                _write_prediction(path, int(first.frame_id), int(first.track_ids[index]), first.boxes[index], float(scores[local, index]))
                            counter = counters[rule]
                            counter["query_frame_pairs_scored"] += 1
                            counter["candidate_rows_scored"] += n
                            counter["selected_rows"] += int(selected.size)
                            counter["key_digest"].update(json.dumps({"unit_key": row["unit_key"], "row_keys": keys}, sort_keys=False).encode("utf-8"))
                            counter["per_video"].setdefault(scope.scope_key, {"native_frame_visits": 0, "query_frame_pairs_scored": 0, "candidate_rows_scored": 0, "selected_rows": 0})
                            counter["per_video"][scope.scope_key]["query_frame_pairs_scored"] += 1
                            counter["per_video"][scope.scope_key]["candidate_rows_scored"] += n
                            counter["per_video"][scope.scope_key]["selected_rows"] += int(selected.size)
                            video_selected_rows[(rule, scope.scope_key)] += int(selected.size)
                            audit_handle.write(json.dumps({
                                "format": AUDIT_FORMAT, "scope": args.scope, "dataset": scope.dataset,
                                "video": scope.video, "query_id": qid, "frame_id": int(first.frame_id),
                                "unit_key": str(row["unit_key"]), "candidate_rows_scored": n,
                                "selected_rows": int(selected.size), "candidate_rows_retained": True,
                                "candidate_deletion": False, "candidate_truncation": False,
                                "corrected_candidate_vs_null_applied": True,
                                "candidate_threshold": c_threshold, "presence_threshold": p_threshold,
                                "null_margin": margin, "emission_contract": "candidate>=threshold & candidate-null>=margin & presence>=threshold",
                                "row_offsets": [int(x) for x in first.row_offsets], "row_keys": keys,
                                "z1_source": resolved["source"], "labels_attached": False,
                                "future_history_count": int((first.history_frame_ids > int(first.frame_id)).sum()),
                            }, ensure_ascii=False) + "\n")
                        actual_pairs += 1; actual_rows += n
                        video_pair_actual[scope.scope_key] += 1
                        video_candidate_rows[scope.scope_key] += n
                for rule in rules:
                    counters[rule]["native_frame_visits"] += 1
                    counters[rule]["per_video"].setdefault(scope.scope_key, {"native_frame_visits": 0, "query_frame_pairs_scored": 0, "candidate_rows_scored": 0, "selected_rows": 0})
                    counters[rule]["per_video"][scope.scope_key]["native_frame_visits"] += 1
                video_frame_visits[scope.scope_key] += 1
                group_keys_seen.add(str(group["group_key"]))
                del resolved, first, native, z1_cpu, tg_cpu, fg_cpu, current, history, history_mask, history_frames
                gc.collect()
                if device.type == "cuda" and group_index % 20 == 0:
                    torch.cuda.empty_cache()
            print(f"[l89d-infer] scope={args.scope} {scope.scope_key} frames={len(groups)} queries={len(scope.queries)} elapsed={time.perf_counter()-started:.1f}s", flush=True)
    finally:
        audit_handle.close()
    # Every legal query has an explicit tracker file, including empty files.
    for rule in rules:
        for dataset, videos in query_map.items():
            for video, queries in videos.items():
                tracker_dir = strategy_paths[rule][dataset]["tracker_data"]
                for row in queries:
                    (tracker_dir / f"{sequence_id(video, int(row['query_id']))}.txt").touch(exist_ok=True)

    # Debug mode records its reduced expected universe and cannot claim
    # full_video; formal mode has one group per native frame and all legal qids.
    selected_timeline = {
        f"{scope.dataset}|{scope.video}": next(item for item in timeline["videos"] if item["dataset"] == scope.dataset and item["video"] == scope.video)
        for scope in selected_scopes
    }
    expected_pairs = int(sum(len(groups_by_scope[scope.scope_key]) * len(scope.queries) for scope in selected_scopes))
    timeline_videos: list[dict[str, Any]] = []
    for scope in selected_scopes:
        descriptor = selected_timeline[scope.scope_key]
        expected_video = len(groups_by_scope[scope.scope_key]) * len(scope.queries)
        timeline_videos.append({
            "dataset": scope.dataset, "video": scope.video,
            "native_frame_count": int(descriptor["native_frame_count"]),
            "visited_frame_count": int(video_frame_visits[scope.scope_key]),
            "legal_query_count": len(scope.queries), "expected_query_frame_pairs": expected_video,
            "actual_query_frame_pairs": int(video_pair_actual[scope.scope_key]),
            "native_frame_ids_sha256": str(descriptor["native_frame_ids_sha256"]),
            "timeline_pair_complete": int(video_pair_actual[scope.scope_key]) == expected_video,
        })
    actual_pair_complete = actual_pairs == expected_pairs and all(item["timeline_pair_complete"] for item in timeline_videos)
    full_video = (not debug and actual_pair_complete and {item["video"] for item in timeline_videos} == {scope.video for scope in scopes})
    rules_summary: dict[str, Any] = {}
    for rule in rules:
        counter = dict(counters[rule])
        counter["key_digest"] = counter["key_digest"].hexdigest()
        counter["per_video"] = {
            key: dict(value) for key, value in counter["per_video"].items()
        }
        rules_summary[rule] = {
            "counters": counter,
            "strategy_paths": {dataset: {name: str(path.resolve()) for name, path in paths.items()}
                               for dataset, paths in strategy_paths[rule].items()},
            "prediction_audits": str(audit_path.resolve()),
            "candidate_rows_retained": True, "candidate_deletion": False, "candidate_truncation": False,
        }

    # GT is materialized only after every candidate/rule has been scored and
    # timeline counters have been checked.  Debug smoke deliberately skips GT.
    # Ground truth is deliberately not materialized here.  The outer run()
    # first completes prediction/timeline work for every dev candidate.  Only
    # after all frozen predictions have been written does it attach legal GT
    # and compute the compact emission descriptors.
    labels_attached = False
    for rule in rules:
        rules_summary[rule]["labels_attached_after_predictions"] = labels_attached
    model_del = model
    del model_del, model, language_cache, language, resolver, store
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    summary = {
        "format": FORMAT, "status": "complete", "scope_key": args.scope,
        "scope": "true native-timeline dev fit/dev selection" if args.scope == "dev" else "true native-timeline internal validation",
        "full_video": bool(full_video), "debug_mode": bool(debug),
        "debug_limits": {"video": args.video, "max_frames": int(args.max_frames), "max_queries": int(args.max_queries)},
        "command": command_line(), "cwd": str(WORK_ROOT), "luna_thread": THREAD, "seed": SEED,
        "datasets": sorted(query_map), "videos": {dataset: sorted(values) for dataset, values in {
            dataset: [scope.video for scope in selected_scopes if scope.dataset == dataset] for dataset in sorted(query_map)
        }.items()},
        "group_count": len(group_keys_seen), "query_count": sum(len(scope.queries) for scope in selected_scopes),
        "expected_query_frame_pairs": expected_pairs, "actual_query_frame_pairs": actual_pairs,
        "expected_candidate_rows": int(sum(video_candidate_rows[key] for key in video_candidate_rows)),
        "candidate_rows_scored": actual_rows, "timeline": {"videos": timeline_videos, "pair_complete": actual_pair_complete},
        "native_frame_source": "L69 native frame_ids via L80BankStore",
        "legal_query_source": "L86 query_rows_for_video over key-only fit/dev or validation rows",
        "strategy_source": str(candidate.get("selection_source") or candidate.get("shortlist_source") or "L89D frozen candidate"),
        "candidate": {"shortlist_index": int(candidate.get("shortlist_index", 0)), "checkpoint": loaded,
                      "rules": rules_summary},
        "z1_cache": str(args.base_z1_cache.resolve()), "supplement_z1_cache": str(args.supplement_z1_cache.resolve()) if args.supplement_z1_cache else None,
        "coverage_audit": str(args.coverage_audit.resolve()), "coverage_audit_sha256": sha256_file(args.coverage_audit.resolve() / "coverage.json"),
        "all_candidate_rows_scored": True, "candidate_deletion": False, "candidate_truncation": False,
        "prediction_strategy_frozen_before_gt": True, "labels_attached_after_predictions": labels_attached,
        "fixed_validation_labels_used_for_selection": False,
        "manifest_sha256": MANIFEST_SHA, "persistent_dense_cache_written": False,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "wall_seconds": time.perf_counter() - started,
        **standard_flags(hota_trackeval_run=False), "failure_root_cause": None,
        "next_action": "run L89D true full-video TrackEval wrapper" if full_video else "formal run required before TrackEval",
        "timeline_contract_proof": {
            "native_timeline": True, "all_legal_videos": not debug,
            "all_native_frames": not debug and all(item["visited_frame_count"] == item["native_frame_count"] for item in timeline_videos),
            "all_legal_queries": not debug,
            "actual_equals_expected_pairs": actual_pair_complete,
            "native_frame_sha256_recorded": True,
        },
    }
    write_json(candidate_root / "summary.json", summary)
    write_json(candidate_root / "provenance.json", summary)
    write_json(candidate_root / "status.json", {"format": FORMAT, "status": "complete", "scope": args.scope,
                                                 "full_video": bool(full_video), "expected_query_frame_pairs": expected_pairs,
                                                 "actual_query_frame_pairs": actual_pairs, "candidate_deletion": False,
                                                 "candidate_truncation": False, "zero_training": True,
                                                 "screening_gt_used": False, "official_test_labels_read": False,
                                                 "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
    return summary


def _attach_gt_after_all_predictions(
    summaries: list[dict[str, Any]], args: argparse.Namespace, out: Path,
) -> None:
    """Attach legal GT only after every candidate's native prediction pass."""
    if any(bool(summary.get("debug_mode")) for summary in summaries):
        return
    scopes = load_video_scopes(args.scope)
    query_map = _query_map_from_scopes(scopes)
    store = L80BankStore(max_history=8)
    try:
        for summary in summaries:
            candidate = summary["candidate"]
            epoch = int(candidate["checkpoint"]["epoch"])
            shortlist_index = int(candidate.get("shortlist_index", 0))
            candidate_root = out / f"candidate_epoch{epoch:03d}_shortlist{shortlist_index:02d}"
            for rule, rule_summary in candidate["rules"].items():
                path_map: dict[str, dict[str, Path]] = {}
                for dataset, raw_paths in rule_summary["strategy_paths"].items():
                    path_map[dataset] = {name: Path(str(value)) for name, value in raw_paths.items()}
                eval_summary: dict[str, Any] = {}
                emission: dict[str, Any] = {}
                for dataset in sorted(query_map):
                    eval_summary[dataset] = materialize_gt(
                        args.scope, dataset, query_map[dataset], path_map[dataset], store,
                    )
                    flat = [row for video in sorted(query_map[dataset]) for row in query_map[dataset][video]]
                    emission[dataset] = emission_descriptor(path_map[dataset], flat)
                rule_summary["eval_summary"] = eval_summary
                rule_summary["emission_descriptor"] = emission
                rule_summary["labels_attached_after_predictions"] = True
            summary["labels_attached_after_predictions"] = True
            summary["candidate"]["labels_attached_after_predictions"] = True
            write_json(candidate_root / "summary.json", summary)
            write_json(candidate_root / "provenance.json", summary)
            write_json(candidate_root / "status.json", {
                "format": FORMAT, "status": "complete", "scope": args.scope,
                "full_video": bool(summary["full_video"]), "labels_attached_after_predictions": True,
                "prediction_strategy_frozen_before_gt": True, "zero_training": True,
                "screening_gt_used": False, "official_test_labels_read": False,
                "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
            })
    finally:
        store._store._bank = None; store._store._text_cache = None
        del store
        gc.collect()


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L89D inference output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = command_line()
    try:
        if Path.cwd().resolve() != WORK_ROOT:
            raise RuntimeError(f"wrong L89D cwd: {Path.cwd()}")
        manifest_assertion()
        if args.scope == "dev":
            source_path = args.shortlist_source.resolve()
            candidates = _load_dev_source(source_path)
            rules = RULES
        else:
            source_path = args.selection.resolve()
            candidates, rules = _load_internal_source(source_path)
        coverage = _read_coverage(args.coverage_audit.resolve(), args.scope)
        summaries = []
        for candidate in candidates:
            candidate["selection_source"] = str(source_path)
            summary = _infer_candidate(candidate, rules, args, coverage, out)
            summaries.append(summary)
        # Labels are outside every candidate's score pass.  The fixed source
        # selection and native timeline are now complete before this call.
        _attach_gt_after_all_predictions(summaries, args, out)
        all_full = bool(summaries) and all(bool(item.get("full_video")) for item in summaries)
        expected = sum(int(item["expected_query_frame_pairs"]) for item in summaries)
        actual = sum(int(item["actual_query_frame_pairs"]) for item in summaries)
        payload = {
            "format": FORMAT, "status": "complete", "scope_key": args.scope,
            "scope": "L89D true native-timeline replay", "full_video": all_full,
            "command": command, "cwd": str(WORK_ROOT), "luna_thread": THREAD, "seed": SEED,
            "source_strategy": str(source_path), "source_strategy_sha256": sha256_file(source_path),
            "candidates": [{"candidate_index": index, "summary": summary["candidate"],
                            "summary_path": str((out / f"candidate_epoch{int(summary['candidate']['checkpoint']['epoch']):03d}_shortlist{int(summary['candidate'].get('shortlist_index', 0)):02d}" / "summary.json").resolve()),
                            "full_video": bool(summary["full_video"]),
                            "expected_query_frame_pairs": int(summary["expected_query_frame_pairs"]),
                            "actual_query_frame_pairs": int(summary["actual_query_frame_pairs"])}
                           for index, summary in enumerate(summaries)],
            "candidate_count": len(summaries), "expected_query_frame_pairs_all_candidates": expected,
            "actual_query_frame_pairs_all_candidates": actual,
            "timeline_contract": {"all_candidates_full_video": all_full, "all_candidates_pair_complete": expected == actual},
            "native_frame_source": "L69 native frame_ids/frame_ptr",
            "candidate_bank": "immutable L69 budget-40; native frame rows",
            "coverage_audit": str(args.coverage_audit.resolve()), "coverage_audit_sha256": sha256_file(args.coverage_audit.resolve() / "coverage.json"),
            "all_candidate_rows_scored": True, "candidate_deletion": False, "candidate_truncation": False,
            "prediction_strategy_frozen_before_gt": True,
            "fixed_validation_labels_used_for_selection": False,
            "manifest_sha256": MANIFEST_SHA, "persistent_dense_cache_written": False,
            **standard_flags(hota_trackeval_run=False), "failure_root_cause": None,
            "next_action": "run L89D TrackEval matrix" if all_full else "repair incomplete true timeline before TrackEval",
        }
        write_json(out / "summary.json", payload)
        write_json(out / "provenance.json", payload)
        write_json(out / "status.json", {"format": FORMAT, "status": "complete", "scope": args.scope,
                                          "full_video": all_full, "candidate_count": len(summaries),
                                          "expected_query_frame_pairs": expected, "actual_query_frame_pairs": actual,
                                          "screening_gt_used": False, "official_test_labels_read": False,
                                          "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                                          "zero_training": True})
        return 0
    except Exception as exc:
        (out / "INCOMPLETE.md").write_text("# L89D true full-video inference — INCOMPLETE\n\n" + traceback.format_exc(), encoding="utf-8")
        write_json(out / "status.json", {"format": FORMAT, "status": "incomplete", "command": command,
                                          "cwd": str(WORK_ROOT), "luna_thread": THREAD,
                                          "failure_root_cause": f"{type(exc).__name__}: {exc}",
                                          "next_action": "repair first actionable true-timeline error and use a new attempt",
                                          **standard_flags(hota_trackeval_run=False)})
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("dev", "internal"), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--shortlist-source", type=Path)
    source.add_argument("--selection", type=Path)
    parser.add_argument("--base-z1-cache", type=Path, default=BASE_Z1_CACHE)
    parser.add_argument("--supplement-z1-cache", type=Path, default=None)
    parser.add_argument("--language-cache", type=Path, default=DEFAULT_LANGUAGE_CACHE)
    parser.add_argument("--coverage-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-tile", type=int, default=8)
    parser.add_argument("--candidate-index", type=int, default=-1)
    parser.add_argument("--video", default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=0)
    args = parser.parse_args()
    if int(args.query_tile) < 1:
        parser.error("--query-tile must be positive")
    if args.scope == "dev" and args.selection is not None:
        parser.error("--selection is internal-only")
    if args.scope == "internal" and args.shortlist_source is not None:
        parser.error("--shortlist-source is dev-only")
    if args.candidate_index >= 0 and args.scope != "dev":
        parser.error("--candidate-index is dev-only")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
