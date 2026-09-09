#!/usr/bin/env python3
"""Phase-consistent true-native-timeline inference for the frozen L89 model."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
WORK_ROOT = Path(__file__).resolve().parents[1]
for value in (WORK_ROOT, WORK_ROOT / "tools", ROOT, ROOT / "tools"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from l89d_fullvideo_common import (  # noqa: E402
    BASE_Z1_CACHE,
    DenseZ1Resolver,
    L80BankStore,
    L89LanguageTokenCache,
    MANIFEST_SHA,
    RULES,
    SEED,
    THREAD,
    WORK_ROOT as COMMON_WORK_ROOT,
    command_line,
    expected_timeline_descriptor,
    load_video_scopes,
    manifest_assertion,
    native_row_key,
    sha256_file,
    standard_flags,
    validate_native_batch,
    write_json,
)
from l89d_infer_true_fullvideo import (  # noqa: E402
    _language_batch,
    _load_language_once,
    _prepare_debug_groups,
    _query_map_from_scopes,
    _read_coverage,
    _write_prediction,
    emission_descriptor,
    materialize_gt,
    prepare_strategy,
    sequence_id,
)
from l88c_eval_metrics import corrected_emission_mask  # noqa: E402
from locatemot.models.l89_full_rmot import L89Config, L89FullRMOT  # noqa: E402
from l89e_phase_policy import history_for_batch, phase_policy_for_epoch  # noqa: E402


FORMAT = "locatemot-l89e-phase-consistent-true-fullvideo-v1"
AUDIT_FORMAT = "locatemot-l89e-phase-consistent-audit-v1"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def _load_candidate_model(
    checkpoint_info: dict[str, Any], device: torch.device,
) -> tuple[L89FullRMOT, dict[str, Any], Any]:
    path = Path(str(checkpoint_info["path"])).resolve()
    observed = sha256_file(path)
    if observed != str(checkpoint_info["sha256"]):
        raise AssertionError(f"checkpoint SHA drift: {path}")
    package = torch.load(path, map_location="cpu", weights_only=False)
    if package.get("format") != "locatemot-l89-checkpoint-v1":
        raise AssertionError(f"invalid L89 checkpoint package: {path}")
    if int(package.get("seed", -1)) != SEED or str(package.get("manifest_sha256")) != MANIFEST_SHA:
        raise AssertionError(f"checkpoint seed/manifest drift: {path}")
    policy = phase_policy_for_epoch(int(package["epoch"]), str(package["phase"]))
    model = L89FullRMOT(L89Config(**package["model_config"])).to(device=device, dtype=torch.float32)
    loaded = model.load_state_dict(package["model_state_dict"], strict=True)
    if loaded.missing_keys or loaded.unexpected_keys:
        raise AssertionError(f"strict L89 reload failed: {path}: {loaded}")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    info = dict(checkpoint_info)
    info.update({
        "path": str(path), "sha256": observed, "epoch": int(package["epoch"]),
        "optimizer_step": int(package["optimizer_step"]), "phase": str(package["phase"]),
        "model_config": package["model_config"], "strict_package": True,
        "phase_policy": policy.to_dict(), "phase_consistent_temporal": True,
    })
    return model, info, policy


def _validate_rule_fits(candidate: dict[str, Any], rules: tuple[str, ...]) -> None:
    fit_map = candidate.get("rule_fits")
    if not isinstance(fit_map, dict):
        raise AssertionError("L89E candidate rule fits missing")
    for rule in rules:
        item = fit_map.get(rule)
        if not isinstance(item, dict):
            raise AssertionError(f"L89E candidate rule missing: {rule}")
        for name in ("candidate_threshold", "presence_threshold", "null_margin"):
            if not np.isfinite(float(item[name])):
                raise AssertionError(f"nonfinite L89E threshold: {rule}/{name}")


def _load_dev_shortlist(path: Path) -> list[dict[str, Any]]:
    source = _load_json(path)
    if source.get("status") != "complete" or source.get("format") != "locatemot-l89e-phase-consistent-shortlist-v1":
        raise AssertionError("L89E dev shortlist is incomplete or wrong format")
    if not bool(source.get("phase_consistent_temporal")) or not bool(source.get("stage_s_rescored")):
        raise AssertionError("L89E shortlist lacks phase-consistent Stage-S provenance")
    values = source.get("shortlist")
    if not isinstance(values, list) or not values or len(values) > 5:
        raise AssertionError("invalid L89E shortlist size")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(values):
        candidate = dict(raw)
        candidate["shortlist_index"] = int(candidate.get("shortlist_index", index))
        checkpoint = dict(candidate.get("checkpoint_info") or {})
        checkpoint_path = Path(str(checkpoint["path"])).resolve()
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != str(checkpoint["sha256"]):
            raise AssertionError(f"L89E shortlist checkpoint drift: {checkpoint_path}")
        key = str(checkpoint_path)
        if key in seen:
            raise AssertionError("duplicate L89E shortlist checkpoint")
        seen.add(key)
        candidate["checkpoint_info"] = checkpoint
        _validate_rule_fits(candidate, RULES)
        result.append(candidate)
    return result


def _load_internal_selection(path: Path) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    source = _load_json(path)
    if source.get("status") != "complete" or source.get("format") != "locatemot-l89e-checkpoint-selection-v1":
        raise AssertionError("L89E internal selection is incomplete or wrong format")
    if not bool(source.get("phase_consistent_temporal")) or not bool(source.get("true_fullvideo_timeline")):
        raise AssertionError("L89E internal selection lacks phase/timeline proof")
    final = source.get("final_selection")
    if not isinstance(final, dict) or str(final.get("rule")) not in RULES:
        raise AssertionError("L89E internal final selection missing")
    checkpoint = dict(final["checkpoint_info"])
    rule = str(final["rule"])
    candidate = {
        "shortlist_index": 0,
        "reason": "frozen L89E dev TrackEval selection",
        "checkpoint_info": checkpoint,
        "rule_fits": {rule: dict(final["rule_object"])},
        "selection_source": str(path.resolve()),
        "selection_source_sha256": sha256_file(path.resolve()),
        "selection_frozen_before_fixed_validation": True,
    }
    _validate_rule_fits(candidate, (rule,))
    checkpoint_path = Path(str(checkpoint["path"])).resolve()
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != str(checkpoint["sha256"]):
        raise AssertionError(f"L89E internal checkpoint drift: {checkpoint_path}")
    return [candidate], (rule,)


def _attach_gt_after_all_predictions(
    summaries: list[dict[str, Any]], args: argparse.Namespace,
) -> None:
    """Materialize legal GT only after every frozen candidate/rule is scored."""
    if any(bool(item.get("debug_mode")) for item in summaries):
        return
    scopes = load_video_scopes(args.scope)
    query_map = _query_map_from_scopes(scopes)
    store = L80BankStore(max_history=8)
    try:
        for summary in summaries:
            candidate = summary["candidate"]
            epoch = int(candidate["checkpoint"]["epoch"])
            shortlist_index = int(candidate.get("shortlist_index", 0))
            candidate_root = Path(summary["candidate_root"]).resolve()
            for rule, rule_summary in candidate["rules"].items():
                path_map = {
                    dataset: {name: Path(str(value)) for name, value in raw_paths.items()}
                    for dataset, raw_paths in rule_summary["strategy_paths"].items()
                }
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
                "full_video": bool(summary["full_video"]),
                "phase_consistent_temporal": True,
                "labels_attached_after_predictions": True,
                "prediction_strategy_frozen_before_gt": True,
                "zero_training": True, "screening_gt_used": False,
                "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
                "hota_trackeval_run": False,
            })
    finally:
        store._store._bank = None
        store._store._text_cache = None
        del store
        gc.collect()


def _infer_candidate(
    candidate: dict[str, Any], rules: tuple[str, ...], args: argparse.Namespace,
    coverage: dict[str, Any], out: Path,
) -> dict[str, Any]:
    device = torch.device(args.device)
    scopes = load_video_scopes(args.scope)
    store = L80BankStore(max_history=8)
    selected_scopes, groups_by_scope, debug = _prepare_debug_groups(
        scopes, store, args.video, int(args.max_frames), int(args.max_queries),
    )
    timeline = expected_timeline_descriptor(args.scope, store)
    if not debug and {item.scope_key for item in selected_scopes} != {item.scope_key for item in scopes}:
        raise AssertionError("formal L89E scope did not include every legal video")
    language_cache = L89LanguageTokenCache(args.language_cache.resolve())
    language = _load_language_once(language_cache, selected_scopes)
    resolver = DenseZ1Resolver(
        args.base_z1_cache.resolve(),
        args.supplement_z1_cache.resolve() if args.supplement_z1_cache else None,
    )
    model, loaded, policy = _load_candidate_model(candidate["checkpoint_info"], device)
    epoch = int(loaded["epoch"])
    candidate_root = out / f"candidate_epoch{epoch:03d}_shortlist{int(candidate.get('shortlist_index', 0)):02d}"
    candidate_root.mkdir(parents=True, exist_ok=True)
    query_map = _query_map_from_scopes(selected_scopes)
    strategy_paths = {rule: prepare_strategy(candidate_root / rule, sorted(query_map), query_map) for rule in rules}
    audit_path = candidate_root / "prediction_audits.jsonl"
    counters: dict[str, dict[str, Any]] = {
        rule: {
            "native_frame_visits": 0, "query_frame_pairs_scored": 0,
            "candidate_rows_scored": 0, "selected_rows": 0,
            "candidate_rows_retained": True, "candidate_deletion": False,
            "candidate_truncation": False, "key_digest": hashlib.sha256(),
            "per_video": {},
        }
        for rule in rules
    }
    video_pair_actual: dict[str, int] = defaultdict(int)
    video_frame_visits: dict[str, int] = defaultdict(int)
    actual_pairs = 0
    actual_rows = 0
    group_keys_seen: set[str] = set()
    started = time.perf_counter()
    with audit_path.open("w", encoding="utf-8") as audit_handle:
        try:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for scope in selected_scopes:
                groups = groups_by_scope[scope.scope_key]
                for group_index, group in enumerate(groups, 1):
                    rows = [dict(row) for row in group["queries"]]
                    first = store.build_unit(rows[0])
                    validate_native_batch(first)
                    resolved = resolver.resolve(group, first)
                    expected_qids = [int(row["query_id"]) for row in rows]
                    if [int(value) for value in resolved["query_ids"]] != expected_qids:
                        raise AssertionError(f"resolver query order drift: {group['group_key']}")
                    n = int(first.candidate_count)
                    z1_cpu = resolved["z1"]
                    text_global_cpu = resolved["text_global"]
                    frame_global_cpu = resolved["frame_global"]
                    if tuple(z1_cpu.shape) != (len(rows), n, 256):
                        raise AssertionError(f"resolved Z1 shape drift: {group['group_key']}")
                    current = first.observations.float().clone().to(device)
                    history_cpu, history_mask_cpu, history_frames_cpu = history_for_batch(first, policy)
                    if policy.phase == "S" and int(history_mask_cpu.sum()) != 0:
                        raise AssertionError(f"Stage-S history is not zero: {group['group_key']}")
                    history = history_cpu.float().clone().to(device)
                    history_mask = history_mask_cpu.bool().clone().to(device)
                    history_frames = history_frames_cpu.long().clone().to(device)
                    if int((history_frames > int(first.frame_id)).sum()) != 0:
                        raise AssertionError(f"future history at {group['group_key']}")
                    for begin in range(0, len(rows), int(args.query_tile)):
                        chunk = rows[begin:begin + int(args.query_tile)]
                        stop = begin + len(chunk)
                        text_tokens, text_mask = _language_batch(
                            language, scope.dataset, scope.video, chunk, device,
                        )
                        z1 = z1_cpu[begin:stop].float().clone().to(device)
                        text_global = text_global_cpu[begin:stop].float().clone().to(device)
                        frame_global = frame_global_cpu[begin:stop].float().clone().to(device)
                        with torch.inference_mode():
                            output = model(
                                z1, text_tokens, text_mask, text_global, frame_global,
                                current, history, history_mask, history_frames,
                                int(first.frame_id), temporal_enabled=policy.temporal_enabled,
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
                                candidate_threshold = float(rule_object["candidate_threshold"])
                                presence_threshold = float(rule_object["presence_threshold"])
                                null_margin = float(rule_object["null_margin"])
                                selected_mask = corrected_emission_mask(
                                    scores[local], float(presence[local]), float(null[local]),
                                    candidate_threshold, presence_threshold, null_margin,
                                )
                                selected = np.flatnonzero(selected_mask)
                                path = (
                                    strategy_paths[rule][scope.dataset]["tracker_data"]
                                    / f"{sequence_id(scope.video, qid)}.txt"
                                )
                                for index in selected.tolist():
                                    _write_prediction(
                                        path, int(first.frame_id), int(first.track_ids[index]),
                                        first.boxes[index], float(scores[local, index]),
                                    )
                                counter = counters[rule]
                                counter["query_frame_pairs_scored"] += 1
                                counter["candidate_rows_scored"] += n
                                counter["selected_rows"] += int(selected.size)
                                counter["key_digest"].update(json.dumps({"unit_key": row["unit_key"], "row_keys": keys}, sort_keys=False).encode("utf-8"))
                                per_video = counter["per_video"].setdefault(
                                    scope.scope_key,
                                    {"native_frame_visits": 0, "query_frame_pairs_scored": 0, "candidate_rows_scored": 0, "selected_rows": 0},
                                )
                                per_video["query_frame_pairs_scored"] += 1
                                per_video["candidate_rows_scored"] += n
                                per_video["selected_rows"] += int(selected.size)
                                audit_handle.write(json.dumps({
                                    "format": AUDIT_FORMAT,
                                    "scope": args.scope,
                                    "dataset": scope.dataset,
                                    "video": scope.video,
                                    "query_id": qid,
                                    "frame_id": int(first.frame_id),
                                    "unit_key": str(row["unit_key"]),
                                    "candidate_rows_scored": n,
                                    "selected_rows": int(selected.size),
                                    "candidate_rows_retained": True,
                                    "candidate_deletion": False,
                                    "candidate_truncation": False,
                                    "corrected_candidate_vs_null_applied": True,
                                    "candidate_threshold": candidate_threshold,
                                    "presence_threshold": presence_threshold,
                                    "null_margin": null_margin,
                                    "emission_contract": "candidate>=threshold & candidate-null>=margin & presence>=threshold",
                                    "row_offsets": [int(x) for x in first.row_offsets],
                                    "row_keys": keys,
                                    "z1_source": resolved["source"],
                                    "labels_attached": False,
                                    "future_history_count": int((history_frames_cpu > int(first.frame_id)).sum()),
                                    "checkpoint_phase": policy.phase,
                                    "phase_consistent_temporal": True,
                                    "temporal_enabled": policy.temporal_enabled,
                                    "history_mode": policy.history_mode,
                                    "history_length": policy.history_length,
                                    "history_valid_count": int(history_mask_cpu.sum().item()),
                                }, ensure_ascii=False) + "\n")
                            actual_pairs += 1
                            actual_rows += n
                            video_pair_actual[scope.scope_key] += 1
                    for rule in rules:
                        counters[rule]["native_frame_visits"] += 1
                        counters[rule]["per_video"].setdefault(
                            scope.scope_key,
                            {"native_frame_visits": 0, "query_frame_pairs_scored": 0, "candidate_rows_scored": 0, "selected_rows": 0},
                        )["native_frame_visits"] += 1
                    video_frame_visits[scope.scope_key] += 1
                    group_keys_seen.add(str(group["group_key"]))
                    del resolved, first, z1_cpu, text_global_cpu, frame_global_cpu
                    del current, history_cpu, history_mask_cpu, history_frames_cpu
                    del history, history_mask, history_frames
                    gc.collect()
                    if device.type == "cuda" and group_index % 20 == 0:
                        torch.cuda.empty_cache()
                print(
                    f"[l89e-infer] scope={args.scope} {scope.scope_key} frames={len(groups)} "
                    f"queries={len(scope.queries)} elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        finally:
            audit_handle.flush()
    for rule in rules:
        for dataset, videos in query_map.items():
            tracker_dir = strategy_paths[rule][dataset]["tracker_data"]
            for video, queries in videos.items():
                for row in queries:
                    (tracker_dir / f"{sequence_id(video, int(row['query_id']))}.txt").touch(exist_ok=True)
    selected_timeline = {
        f"{scope.dataset}|{scope.video}": next(
            item for item in timeline["videos"]
            if item["dataset"] == scope.dataset and item["video"] == scope.video
        )
        for scope in selected_scopes
    }
    expected_pairs = int(sum(len(groups_by_scope[scope.scope_key]) * len(scope.queries) for scope in selected_scopes))
    timeline_videos = []
    for scope in selected_scopes:
        descriptor = selected_timeline[scope.scope_key]
        expected_video = len(groups_by_scope[scope.scope_key]) * len(scope.queries)
        timeline_videos.append({
            "dataset": scope.dataset,
            "video": scope.video,
            "native_frame_count": int(descriptor["native_frame_count"]),
            "visited_frame_count": int(video_frame_visits[scope.scope_key]),
            "legal_query_count": len(scope.queries),
            "expected_query_frame_pairs": expected_video,
            "actual_query_frame_pairs": int(video_pair_actual[scope.scope_key]),
            "native_frame_ids_sha256": str(descriptor["native_frame_ids_sha256"]),
            "timeline_pair_complete": int(video_pair_actual[scope.scope_key]) == expected_video,
        })
    pair_complete = actual_pairs == expected_pairs and all(item["timeline_pair_complete"] for item in timeline_videos)
    full_video = not debug and pair_complete and {item["video"] for item in timeline_videos} == {scope.video for scope in scopes}
    rules_summary: dict[str, Any] = {}
    for rule in rules:
        counter = dict(counters[rule])
        counter["key_digest"] = counter["key_digest"].hexdigest()
        rules_summary[rule] = {
            "counters": counter,
            "strategy_paths": {
                dataset: {name: str(path.resolve()) for name, path in paths.items()}
                for dataset, paths in strategy_paths[rule].items()
            },
            "prediction_audits": str(audit_path.resolve()),
            "candidate_rows_retained": True,
            "candidate_deletion": False,
            "candidate_truncation": False,
            "phase_consistent_temporal": True,
            "phase_policy": policy.to_dict(),
            "labels_attached_after_predictions": False,
        }
    summary = {
        "format": FORMAT,
        "status": "complete",
        "scope_key": args.scope,
        "scope": "L89E true native-timeline dev fit/dev selection" if args.scope == "dev" else "L89E true native-timeline internal validation",
        "full_video": bool(full_video),
        "debug_mode": bool(debug),
        "debug_limits": {"video": args.video, "max_frames": int(args.max_frames), "max_queries": int(args.max_queries)},
        "command": command_line(),
        "cwd": str(COMMON_WORK_ROOT),
        "luna_thread": THREAD,
        "seed": SEED,
        "datasets": sorted(query_map),
        "videos": {
            dataset: sorted(values)
            for dataset, values in {
                dataset: [scope.video for scope in selected_scopes if scope.dataset == dataset]
                for dataset in sorted(query_map)
            }.items()
        },
        "group_count": len(group_keys_seen),
        "query_count": sum(len(scope.queries) for scope in selected_scopes),
        "expected_query_frame_pairs": expected_pairs,
        "actual_query_frame_pairs": actual_pairs,
        "expected_candidate_rows": actual_rows,
        "candidate_rows_scored": actual_rows,
        "timeline": {"videos": timeline_videos, "pair_complete": pair_complete},
        "native_frame_source": "L69 native frame_ids via L80BankStore",
        "legal_query_source": "L86 query_rows_for_video over key-only fit/dev or validation rows",
        "strategy_source": str(candidate.get("selection_source") or candidate.get("source_strategy") or "L89E frozen candidate"),
        "candidate_root": str(candidate_root.resolve()),
        "candidate": {
            "shortlist_index": int(candidate.get("shortlist_index", 0)),
            "checkpoint": loaded,
            "rules": rules_summary,
            "phase_policy": policy.to_dict(),
            "phase_consistent_temporal": True,
            "candidate_root": str(candidate_root.resolve()),
        },
        "z1_cache": str(args.base_z1_cache.resolve()),
        "supplement_z1_cache": str(args.supplement_z1_cache.resolve()) if args.supplement_z1_cache else None,
        "coverage_audit": str(args.coverage_audit.resolve()),
        "coverage_audit_sha256": sha256_file(args.coverage_audit.resolve() / "coverage.json"),
        "all_candidate_rows_scored": True,
        "candidate_deletion": False,
        "candidate_truncation": False,
        "phase_consistent_temporal": True,
        "phase_consistency_contract": "S: temporal off + zero history; T/J: temporal on + last4 causal history",
        "phase_consistency_passed": True,
        "prediction_strategy_frozen_before_gt": True,
        "labels_attached_after_predictions": False,
        "fixed_validation_labels_used_for_selection": False,
        "manifest_sha256": MANIFEST_SHA,
        "persistent_dense_cache_written": False,
        "peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "wall_seconds": time.perf_counter() - started,
        **standard_flags(hota_trackeval_run=False),
        "failure_root_cause": None,
        "next_action": "attach legal GT after all predictions, then run L89E TrackEval wrapper" if full_video else "formal full-video run required before TrackEval",
    }
    write_json(candidate_root / "summary.json", summary)
    write_json(candidate_root / "provenance.json", summary)
    write_json(candidate_root / "status.json", {
        "format": FORMAT, "status": "complete", "scope": args.scope,
        "full_video": bool(full_video), "phase_consistent_temporal": True,
        "phase_policy": policy.to_dict(), "expected_query_frame_pairs": expected_pairs,
        "actual_query_frame_pairs": actual_pairs, "candidate_deletion": False,
        "candidate_truncation": False, "labels_attached_after_predictions": False,
        "zero_training": True, "screening_gt_used": False,
        "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
        "hota_trackeval_run": False,
    })
    del model, language_cache, language, resolver, store
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def run(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L89E inference output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = command_line()
    try:
        if Path.cwd().resolve() != COMMON_WORK_ROOT:
            raise RuntimeError(f"wrong L89E worktree cwd: {Path.cwd()}")
        manifest_assertion()
        if args.scope == "dev":
            source_path = args.shortlist_source.resolve()
            candidates = _load_dev_shortlist(source_path)
            rules = RULES
        else:
            source_path = args.selection.resolve()
            candidates, rules = _load_internal_selection(source_path)
        coverage = _read_coverage(args.coverage_audit.resolve(), args.scope)
        summaries: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate["selection_source"] = str(source_path)
            summaries.append(_infer_candidate(candidate, rules, args, coverage, out))
        _attach_gt_after_all_predictions(summaries, args)
        all_full = bool(summaries) and all(bool(item.get("full_video")) for item in summaries)
        expected = int(sum(item["expected_query_frame_pairs"] for item in summaries))
        actual = int(sum(item["actual_query_frame_pairs"] for item in summaries))
        payload = {
            "format": FORMAT,
            "status": "complete",
            "scope_key": args.scope,
            "scope": "L89E true native-timeline replay",
            "full_video": all_full,
            "command": command,
            "cwd": str(COMMON_WORK_ROOT),
            "luna_thread": THREAD,
            "seed": SEED,
            "source_strategy": str(source_path),
            "source_strategy_sha256": sha256_file(source_path),
            "candidates": [
                {
                    "candidate_index": index,
                    "summary": summary["candidate"],
                    "summary_path": str((Path(summary["candidate_root"]) / "summary.json").resolve()),
                    "full_video": bool(summary["full_video"]),
                    "expected_query_frame_pairs": int(summary["expected_query_frame_pairs"]),
                    "actual_query_frame_pairs": int(summary["actual_query_frame_pairs"]),
                    "phase_consistent_temporal": True,
                }
                for index, summary in enumerate(summaries)
            ],
            "candidate_count": len(summaries),
            "expected_query_frame_pairs_all_candidates": expected,
            "actual_query_frame_pairs_all_candidates": actual,
            "timeline_contract": {
                "all_candidates_full_video": all_full,
                "all_candidates_pair_complete": expected == actual,
                "native_frame_source": "L69 native frame_ids/frame_ptr",
            },
            "phase_consistency_contract": "S: temporal off + zero history; T/J: temporal on + last4 causal history",
            "phase_consistency_passed": True,
            "phase_consistent_temporal": True,
            "candidate_bank": "immutable L69 budget-40; native frame rows",
            "coverage_audit": str(args.coverage_audit.resolve()),
            "coverage_audit_sha256": sha256_file(args.coverage_audit.resolve() / "coverage.json"),
            "all_candidate_rows_scored": True,
            "candidate_deletion": False,
            "candidate_truncation": False,
            "prediction_strategy_frozen_before_gt": True,
            "fixed_validation_labels_used_for_selection": False,
            "labels_attached_after_predictions": True,
            "manifest_sha256": MANIFEST_SHA,
            "persistent_dense_cache_written": False,
            "zero_training": True,
            "new_checkpoint_created": False,
            "checkpoint_weights_changed": False,
            "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED",
            **standard_flags(hota_trackeval_run=False),
            "failure_root_cause": None,
            "next_action": "run L89E TrackEval matrix" if all_full else "repair incomplete true-native timeline",
        }
        write_json(out / "summary.json", payload)
        write_json(out / "provenance.json", payload | {"format": "locatemot-l89e-true-fullvideo-provenance-v1"})
        write_json(out / "status.json", {
            "format": FORMAT, "status": "complete", "scope": args.scope,
            "full_video": all_full, "candidate_count": len(summaries),
            "expected_query_frame_pairs": expected, "actual_query_frame_pairs": actual,
            "phase_consistency_passed": True, "zero_training": True,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        })
        return 0
    except Exception as exc:
        trace = traceback.format_exc()
        (out / "INCOMPLETE.md").write_text("# L89E true full-video inference — INCOMPLETE\n\n" + trace, encoding="utf-8")
        write_json(out / "status.json", {
            "format": FORMAT, "status": "incomplete", "command": command,
            "cwd": str(COMMON_WORK_ROOT), "luna_thread": THREAD,
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "next_action": "repair the first L89E phase/timeline error and use a new output",
            **standard_flags(hota_trackeval_run=False),
        })
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
    parser.add_argument("--language-cache", type=Path, required=True)
    parser.add_argument("--coverage-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--query-tile", type=int, default=8)
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
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
