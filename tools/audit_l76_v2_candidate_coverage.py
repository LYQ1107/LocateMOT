#!/usr/bin/env python3
"""Read-only V2 candidate/proposal coverage ceiling audit for L76.

The audit rebuilds rows from the L69 budget-40 bank's native frame pointers.
The L49 ``begin/end`` and ``positive_indices`` fields are intentionally never
used to address the L69 bank.  Candidate sidecar labels are joined only after
the complete native row list for a unit has been constructed and are used for
post-hoc oracle/coverage accounting only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
DATA = ROOT / "outputs/l49/data"
FEATURE_ROOT = ROOT / "outputs/l69/attempt9/budget40_features/kitti"
L69_CONTRACT = ROOT / "outputs/l69/audit/budget40_bank_contract_attempt13_full/contract.json"
L67_BASELINE = ROOT / "outputs/l67/audit/v2_candidate_coverage_attempt7/coverage.json"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
FIXED_ROWS = ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
FULL_V2_VIDEOS = ("0016", "0017", "0020")
REQUIRED_FIELDS = (
    "frame", "candidate_index", "track_id", "box", "objectness", "clip",
    "history_clip", "pbd", "uidm_h", "uidm_ref_pbd", "uidm_anchor_pbd",
    "geometry", "motion", "context", "lifecycle", "pool_id", "source_score",
    "raw_rank", "frame_ptr", "frame_ids",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_meta(path: Path, include_hash: bool = True) -> dict[str, Any]:
    path = path.resolve()
    value: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "mtime_ns": path.stat().st_mtime_ns if path.exists() else None,
    }
    if include_hash and path.is_file():
        value["sha256"] = sha256_file(path)
    return value


def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(jsonable(row), sort_keys=True, separators=(",", ":")) + "\n")


def command_string() -> str:
    return " ".join([sys.executable] + sys.argv)


def stats(values: Iterable[float | int]) -> dict[str, Any]:
    values = [float(x) for x in values]
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def fixed_units(context: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    result: list[dict[str, Any]] = []
    keys: list[str] = []
    for row in load_jsonl(FIXED_ROWS):
        key = str(row["unit_key"])
        if key in keys:
            raise AssertionError(f"duplicate fixed unit key: {key}")
        if key not in context:
            raise AssertionError(f"fixed key absent from L49 metadata: {key}")
        keys.append(key)
        result.append(context[key])
    if len(result) != 40 or len(set(keys)) != 40:
        raise AssertionError(f"fixed slice must contain 40 unique units, got {len(result)}")
    return result, keys


def source_contract(bank: dict[str, Any], bank_path: Path) -> dict[str, Any]:
    metadata = bank.get("metadata", {})
    tensors = bank["tensors"]
    pool_values = sorted({int(x) for x in tensors["pool_id"].detach().cpu().tolist()})
    verified = (
        metadata.get("preserve_source_ids") is True
        and metadata.get("reserve_id_offset") is not None
        and set(pool_values) <= {0, 1}
        and metadata.get("main_source")
        and metadata.get("reserve_source")
    )
    return {
        "status": "verified" if verified else "source_split_unavailable",
        "pool_values_observed": pool_values,
        "mapping": {"0": "main", "1": "reserve"} if verified else None,
        "basis": "L69 metadata preserve_source_ids, reserve_id_offset and observed pool values {0,1}",
        "bank": str(bank_path.resolve()),
        "main_source": metadata.get("main_source"),
        "reserve_source": metadata.get("reserve_source"),
        "reserve_id_offset": metadata.get("reserve_id_offset"),
        "preserve_source_ids": metadata.get("preserve_source_ids"),
    }


def safe_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        value = value.item()
    return int(value)


def source_name(pool: int, verified: bool) -> str:
    if not verified:
        return "unknown"
    return {0: "main", 1: "reserve"}.get(int(pool), "unknown")


def make_row(
    unit: dict[str, Any],
    tensors: dict[str, torch.Tensor],
    row_offset: int,
    source_verified: bool,
) -> dict[str, Any]:
    pool = safe_int(tensors["pool_id"][row_offset])
    candidate_index = safe_int(tensors["candidate_index"][row_offset])
    track_id = safe_int(tensors["track_id"][row_offset])
    box = [float(x) for x in tensors["box"][row_offset].detach().cpu().tolist()]
    raw_rank = safe_int(tensors["raw_rank"][row_offset])
    return {
        "row_offset": int(row_offset),
        "candidate_index": candidate_index,
        "track_id": track_id,
        "pool_id": pool,
        "source": source_name(pool, source_verified),
        "raw_rank": raw_rank,
        "cross_pool_duplicate": bool(tensors["cross_pool_duplicate"][row_offset].item()) if "cross_pool_duplicate" in tensors else False,
        "box": box,
        "row_key": [
            str(unit["dataset"]), str(unit["video"]), int(unit["query_id"]),
            int(unit["frame_id"]), candidate_index, track_id, int(row_offset),
        ],
    }


def build_native_rows(unit: dict[str, Any], view: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tensors = view["tensors"]
    frame_to_index = view["frame_to_index"]
    frame_id = int(unit["frame_id"])
    if frame_id not in frame_to_index:
        raise AssertionError(f"frame not present in L69 bank: {unit['unit_key']}")
    frame_index = int(frame_to_index[frame_id])
    start = safe_int(tensors["frame_ptr"][frame_index])
    end = safe_int(tensors["frame_ptr"][frame_index + 1])
    if end < start:
        raise AssertionError(f"descending frame_ptr for {unit['unit_key']}")
    # This is the feature/row construction phase.  No unit target or sidecar
    # label is consulted until this complete native row list exists.
    rows = [make_row(unit, tensors, offset, view["source_contract"]["status"] == "verified")
            for offset in range(start, end)]
    for row in rows:
        if row["row_key"][-1] < start or row["row_key"][-1] >= end:
            raise AssertionError(f"row offset outside native frame range: {unit['unit_key']}")
    native = {
        "frame_index": frame_index,
        "frame_ptr_start": start,
        "frame_ptr_end": end,
        "candidate_count": len(rows),
        "legacy_unit_candidate_count": int(unit.get("candidate_count", -1)),
        "legacy_range_used": False,
        "legacy_positive_indices_used": False,
    }
    return rows, native


def target_ids_from_unit(unit: dict[str, Any]) -> list[str]:
    value = unit.get("target_ids")
    if value is None:
        raise AssertionError(f"missing target_ids: {unit['unit_key']}")
    return sorted({str(x) for x in value})


def annotate_rows(unit: dict[str, Any], rows: list[dict[str, Any]], candidate_gt: list[Any]) -> dict[str, Any]:
    targets = set(target_ids_from_unit(unit))
    if any(r["row_offset"] >= len(candidate_gt) for r in rows):
        raise AssertionError(f"sidecar cannot address native rows: {unit['unit_key']}")
    # Labels are joined only after build_native_rows returned all candidate rows.
    positive: list[dict[str, Any]] = []
    nonnull = 0
    for row in rows:
        raw_gt = candidate_gt[row["row_offset"]]
        gt = None if raw_gt is None else str(raw_gt)
        row["candidate_gt"] = gt
        row["positive_for_unit"] = bool(gt is not None and gt in targets)
        if gt is not None:
            nonnull += 1
        if row["positive_for_unit"]:
            positive.append(row)
    target_to_rows: dict[str, list[dict[str, Any]]] = {target: [] for target in targets}
    for row in positive:
        target_to_rows[str(row["candidate_gt"])].append(row)
    covered_targets = sorted(k for k, v in target_to_rows.items() if v)
    main_targets = sorted({str(r["candidate_gt"]) for r in positive if r["source"] == "main"})
    reserve_targets = sorted({str(r["candidate_gt"]) for r in positive if r["source"] == "reserve"})
    candidate_indices = [int(r["candidate_index"]) for r in rows]
    track_ids = [int(r["track_id"]) for r in rows]
    index_counts = Counter(candidate_indices)
    track_counts = Counter(track_ids)
    pool_counts = Counter(str(r["source"]) for r in rows)
    positive_pool_counts = Counter(str(r["source"]) for r in positive)
    positive_track_ids = sorted({int(r["track_id"]) for r in positive})
    declared = str(unit.get("category", "unknown"))
    if not targets:
        derived = "inactive"
    elif not positive:
        derived = "present_uncovered"
    elif len(positive) > 1:
        derived = "multi_positive"
    else:
        derived = "positive"
    return {
        "target_ids": sorted(targets),
        "target_id_count": len(targets),
        "target_present": bool(targets),
        "candidate_present": bool(positive),
        "candidate_covered": bool(positive),
        "covered_target_ids": covered_targets,
        "target_ids_covered": len(covered_targets),
        "target_id_coverage": len(covered_targets) / max(1, len(targets)),
        "positive_count": len(positive),
        "non_null_candidate_gt_rows": nonnull,
        "negative_count": len(rows) - len(positive),
        "declared_category": declared,
        "derived_category": derived,
        "category_consistent": declared == derived or (declared == "positive" and derived == "multi_positive"),
        "main_positive_target_ids": main_targets,
        "reserve_positive_target_ids": reserve_targets,
        "union_positive_target_ids": covered_targets,
        "positive_track_ids": positive_track_ids,
        "pool_counts": dict(pool_counts),
        "positive_pool_counts": dict(positive_pool_counts),
        "candidate_index_duplicate_count": int(sum(max(0, n - 1) for n in index_counts.values())),
        "candidate_index_duplicate_values": sorted(int(k) for k, n in index_counts.items() if n > 1)[:32],
        "duplicate_track_count": int(sum(max(0, n - 1) for n in track_counts.values())),
        "duplicate_track_values": sorted(int(k) for k, n in track_counts.items() if n > 1)[:32],
        "cross_pool_duplicate_rows": int(sum(1 for r in rows if r.get("cross_pool_duplicate", False))),
        "all_negative_fallback": True,
        "same_class_hard_negative_metadata": "unavailable",
        "same_frame_negative_rows": len(rows) - len(positive),
        "candidate_rows": rows,
        "positive_row_offsets": [int(r["row_offset"]) for r in positive],
        "positive_candidate_indices": [int(r["candidate_index"]) for r in positive],
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    present = [r for r in records if r["target_present"]]
    covered = [r for r in present if r["candidate_present"]]
    candidate_rows = sum(int(r["candidate_count"]) for r in records)
    positive_rows = sum(int(r["positive_count"]) for r in records)
    present_rows = sum(int(r["candidate_count"]) for r in present)
    present_positive_rows = sum(int(r["positive_count"]) for r in present)
    target_total = sum(int(r["target_id_count"]) for r in present)
    target_covered = sum(int(r["target_ids_covered"]) for r in present)
    unit_coverage = len(covered) / max(1, len(present))
    target_micro = target_covered / max(1, target_total)
    target_macro_values = [float(r["target_id_coverage"]) for r in present]
    oracle_accept_all_precision = positive_rows / max(1, candidate_rows)
    oracle_present_precision = present_positive_rows / max(1, present_rows)
    gt_privileged_recall = unit_coverage
    gt_privileged_f1 = 2 * gt_privileged_recall / max(1e-12, 1 + gt_privileged_recall)
    return {
        "units": len(records),
        "target_present_units": len(present),
        "inactive_units": sum(not r["target_present"] for r in records),
        "candidate_present_units": len(covered),
        "present_uncovered_units": sum(r["target_present"] and not r["candidate_present"] for r in records),
        "unit_coverage": unit_coverage,
        "target_ids": target_total,
        "covered_target_ids": target_covered,
        "target_level_micro_coverage": target_micro,
        "target_level_macro_coverage": float(np.mean(target_macro_values)) if target_macro_values else None,
        "candidate_rows": candidate_rows,
        "positive_rows": positive_rows,
        "negative_rows": candidate_rows - positive_rows,
        "candidate_count": stats([r["candidate_count"] for r in records]),
        "positive_count": stats([r["positive_count"] for r in records]),
        "reserve_rows": sum(int(r["pool_counts"].get("reserve", 0)) for r in records),
        "main_rows": sum(int(r["pool_counts"].get("main", 0)) for r in records),
        "cross_pool_duplicate_rows": sum(int(r["cross_pool_duplicate_rows"]) for r in records),
        "candidate_index_duplicate_rows": sum(int(r["candidate_index_duplicate_count"]) for r in records),
        "duplicate_track_rows": sum(int(r["duplicate_track_count"]) for r in records),
        "empty_candidate_units": sum(int(r["candidate_count"]) == 0 for r in records),
        "oracle_proposal": {
            "label": "GT_PRIVILEGED_ORACLE",
            "criterion": "candidate_gt is in this unit's target_ids; labels joined after native row construction",
            "accept_all_candidate_rows_precision_all_units": oracle_accept_all_precision,
            "accept_all_candidate_rows_precision_target_present_only": oracle_present_precision,
            "accept_all_candidate_rows_frame_recall": unit_coverage,
            "accept_all_candidate_rows_target_level_micro_recall": target_micro,
            "gt_overlap_only_acceptance_precision": 1.0 if positive_rows else None,
            "gt_overlap_only_acceptance_frame_recall": gt_privileged_recall,
            "gt_overlap_only_acceptance_target_level_micro_recall": target_micro,
            "gt_overlap_only_membership_f1_upper_bound": gt_privileged_f1,
            "candidate_rows_if_gt_overlap_only": positive_rows,
        },
        "geometric_iou": {
            "available": False,
            "iou_0.50": None,
            "iou_0.75": None,
            "reason": "L69 sidecar/unit metadata provide candidate_gt membership but no per-unit GT boxes; exact geometric IoU cannot be recomputed without adding another GT source.",
            "registered_membership_used": True,
        },
        "source_coverage": source_coverage(records),
        "missing_target_reason_counts": missing_reason_counts(records),
    }


def source_coverage(records: list[dict[str, Any]]) -> dict[str, Any]:
    present = [r for r in records if r["target_present"]]
    total_targets = sum(int(r["target_id_count"]) for r in present)
    result: dict[str, Any] = {}
    for source, field in (("main", "main_positive_target_ids"), ("reserve", "reserve_positive_target_ids"), ("union", "union_positive_target_ids")):
        covered_units = sum(bool(r[field]) for r in present)
        covered_targets = sum(len(set(r[field])) for r in present)
        result[source] = {
            "target_present_units": len(present),
            "covered_units": covered_units,
            "unit_coverage": covered_units / max(1, len(present)),
            "target_ids": total_targets,
            "covered_target_ids": covered_targets,
            "target_level_micro_coverage": covered_targets / max(1, total_targets),
            "target_level_macro_coverage": float(np.mean([
                len(set(r[field])) / max(1, int(r["target_id_count"])) for r in present
            ])) if present else None,
        }
    return result


def missing_reason_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for record in records:
        for item in record.get("missing_target_reasons", []):
            counts[item["reason"]] += 1
    return {"target_level_missing_cases": dict(counts), "unit_level_present_uncovered": sum(r["target_present"] and not r["candidate_present"] for r in records)}


def group_summary(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[field])].append(record)
    return {key: summarize(value) for key, value in sorted(groups.items())}


def target_global_index(view: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    tensors = view["tensors"]
    candidate_gt = view["candidate_gt"]
    frame_ids = tensors["frame"].detach().cpu().tolist()
    pool = tensors["pool_id"].detach().cpu().tolist()
    track = tensors["track_id"].detach().cpu().tolist()
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for offset, raw_gt in enumerate(candidate_gt):
        if raw_gt is None:
            continue
        result[str(raw_gt)].append({
            "row_offset": int(offset), "frame_id": int(frame_ids[offset]),
            "pool_id": int(pool[offset]), "track_id": int(track[offset]),
        })
    return result


def annotate_missing(record: dict[str, Any], global_index: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    current_frame = int(record["frame_id"])
    current_offsets = {int(row["row_offset"]) for row in record["candidate_rows"]}
    current_target_rows = defaultdict(list)
    for row in record["candidate_rows"]:
        if row.get("candidate_gt") is not None:
            current_target_rows[str(row["candidate_gt"])].append(row)
    result = []
    for target in record["target_ids"]:
        if target in record["covered_target_ids"]:
            continue
        locations = global_index.get(str(target), [])
        current_locations = [x for x in locations if x["row_offset"] in current_offsets]
        valid_current = [x for x in current_locations if x["track_id"] >= 0 and x["pool_id"] in (0, 1)]
        if current_locations and not valid_current:
            reason = "present_but_attached_to_unusable_row"
        elif locations:
            reason = "present_in_wrong_fragment_or_other_frame"
        else:
            reason = "absent_from_bank"
        result.append({
            "target_id": str(target),
            "reason": reason,
            "current_frame_id": current_frame,
            "current_candidate_rows_for_target": len(current_locations),
            "bank_locations_count": len(locations),
            "bank_location_examples": locations[:8],
        })
    return result


def target_identity_summary(view_by_video: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for video, view in sorted(view_by_video.items()):
        index = target_global_index(view)
        rows = []
        for target, locations in sorted(index.items()):
            frames = sorted({int(x["frame_id"]) for x in locations})
            pools = sorted({int(x["pool_id"]) for x in locations})
            tracks = sorted({int(x["track_id"]) for x in locations})
            rows.append({
                "video": video, "target_id": target, "frames": len(frames),
                "first_frame": frames[0] if frames else None,
                "last_frame": frames[-1] if frames else None,
                "pools": pools, "tracks": len(tracks),
                "cross_pool": len(pools) > 1,
                "fragment_count_proxy": len({(int(x["pool_id"]), int(x["track_id"])) for x in locations}),
                "observation_count": len(locations),
            })
        cross_frame = [x for x in rows if x["frames"] > 1]
        cross_pool = [x for x in rows if x["cross_pool"]]
        output[video] = {
            "targets_with_sidecar_observation": len(rows),
            "targets_cross_frame": len(cross_frame),
            "targets_cross_pool": len(cross_pool),
            "fragment_count_proxy": stats([x["fragment_count_proxy"] for x in rows]),
            "same_gt_cross_frame_fraction": len(cross_frame) / max(1, len(rows)),
            "cross_pool_fraction": len(cross_pool) / max(1, len(rows)),
            "rows": rows,
        }
    return output


def build_views(videos: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    views: dict[str, dict[str, Any]] = {}
    source_contracts: dict[str, Any] = {}
    for video in videos:
        bank_path = (FEATURE_ROOT / f"{video}.pt").resolve()
        label_path = bank_path.with_suffix(".labels.json")
        if not bank_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"missing L69 bank or sidecar for {video}")
        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        if not isinstance(bank, dict) or "tensors" not in bank:
            raise AssertionError(f"invalid L69 bank container {video}")
        tensors = bank["tensors"]
        missing = [field for field in REQUIRED_FIELDS if field not in tensors]
        if missing:
            raise AssertionError(f"{video} missing L69 fields: {missing}")
        total = int(tensors["track_id"].numel())
        sidecar = json.loads(label_path.read_text())
        candidate_gt = sidecar.get("candidate_gt") if isinstance(sidecar, dict) else None
        if not isinstance(candidate_gt, list) or len(candidate_gt) != total:
            raise AssertionError(f"{video} candidate_gt length mismatch")
        frame_ids = tensors["frame_ids"].detach().cpu().long()
        frame_ptr = tensors["frame_ptr"].detach().cpu().long()
        frame_rows = tensors["frame"].detach().cpu().long()
        if frame_ptr.numel() != frame_ids.numel() + 1 or int(frame_ptr[-1]) != total:
            raise AssertionError(f"{video} frame_ptr total mismatch")
        if len(set(int(x) for x in frame_ids.tolist())) != int(frame_ids.numel()):
            raise AssertionError(f"{video} duplicate frame_ids")
        for field in ("box", "objectness", "clip", "history_clip", "pbd", "uidm_h", "uidm_ref_pbd", "uidm_anchor_pbd", "geometry", "motion", "context", "lifecycle", "source_score"):
            if not bool(torch.isfinite(tensors[field].float()).all()):
                raise AssertionError(f"nonfinite L69 field {video}:{field}")
        starts = frame_ptr[:-1].tolist()
        ends = frame_ptr[1:].tolist()
        for frame_id, start, end in zip(frame_ids.tolist(), starts, ends):
            if end < start:
                raise AssertionError(f"descending pointer {video}:{frame_id}")
            if end > start and not bool((frame_rows[start:end] == int(frame_id)).all()):
                raise AssertionError(f"row/frame mismatch {video}:{frame_id}")
        contract = source_contract(bank, bank_path)
        source_contracts[video] = contract
        views[video] = {
            "bank": bank,
            "tensors": tensors,
            "candidate_gt": [None if x is None else str(x) for x in candidate_gt],
            "frame_to_index": {int(frame_id): int(i) for i, frame_id in enumerate(frame_ids.tolist())},
            "source_contract": contract,
            "bank_path": bank_path,
            "label_path": label_path,
            "total_rows": total,
        }
    return views, source_contracts


def add_common(payload: dict[str, Any], status: str, command: str, inputs: list[Any], outputs: list[str], next_action: str) -> dict[str, Any]:
    payload.update({
        "format": payload.get("format", "locatemot-l76-v2-candidate-coverage-v1"),
        "status": status,
        "command": command,
        "inputs": inputs,
        "outputs": outputs,
        "failure_root_cause": payload.get("failure_root_cause"),
        "next_action": next_action,
    })
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    command = command_string()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    next_action = "future RMOT-specific visual/query interface if coverage is adequate; otherwise proposal repair audit"
    base_inputs = [str(ROOT / "AGENTS.md"), str(DATA / "validation_units.jsonl"), str(DATA / "calibration_units.jsonl"), str(FIXED_ROWS), str(MANIFEST), str(L69_CONTRACT), str(L67_BASELINE)]
    output_names = ["coverage.json", "unit_records.jsonl", "case_records.jsonl", "provenance.json", "status.json"]
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        manifest_sha = sha256_file(MANIFEST)
        if manifest_sha != EXPECTED_MANIFEST:
            raise AssertionError(f"manifest SHA mismatch: {manifest_sha}")
        validation = [x for x in load_jsonl(DATA / "validation_units.jsonl")
                      if x.get("dataset") == "refer_kitti_v2" and x.get("split") == "validation"
                      and str(x.get("video")) in FULL_V2_VIDEOS]
        if len(validation) != 768:
            raise AssertionError(f"expected 768 full V2 validation units, got {len(validation)}")
        if {str(x["video"]) for x in validation} != set(FULL_V2_VIDEOS):
            raise AssertionError("full V2 video set mismatch")
        context_rows = load_jsonl(DATA / "calibration_units.jsonl") + load_jsonl(DATA / "validation_units.jsonl")
        context = {str(x["unit_key"]): x for x in context_rows}
        fixed, fixed_keys = fixed_units(context)
        fixed_v2 = [x for x in fixed if x.get("dataset") == "refer_kitti_v2"]
        if len(fixed_v2) != 20:
            raise AssertionError(f"expected 20 V2 units in fixed 40 slice, got {len(fixed_v2)}")
        all_units_by_key = {str(x["unit_key"]): x for x in validation}
        for unit in fixed_v2:
            all_units_by_key.setdefault(str(unit["unit_key"]), unit)
        videos = sorted({str(x["video"]) for x in all_units_by_key.values()})
        if videos != ["0015", "0016", "0017", "0020"]:
            raise AssertionError(f"unexpected audit videos: {videos}")
        views, source_contracts = build_views(videos)
        global_indices = {video: target_global_index(view) for video, view in views.items()}

        records_by_key: dict[str, dict[str, Any]] = {}
        for key, unit in sorted(all_units_by_key.items()):
            video = str(unit["video"])
            view = views[video]
            rows, native = build_native_rows(unit, view)
            annotation = annotate_rows(unit, rows, view["candidate_gt"])
            record = {
                "format": "locatemot-l76-v2-unit-record-v1",
                "status": "complete",
                "scope": "full_v2_validation" if key in {str(x["unit_key"]) for x in validation} else "fixed_40_v2_context",
                "dataset": str(unit["dataset"]), "video": video,
                "query_id": int(unit["query_id"]), "frame_id": int(unit["frame_id"]),
                "unit_key": key, "split": str(unit.get("split")),
                "expression": str(unit.get("expression", unit.get("sentence", ""))),
                "frame_key": [str(unit["video"]), int(unit["frame_id"])],
                "bank_path": str(view["bank_path"]), "label_path": str(view["label_path"]),
                "source_mapping": view["source_contract"],
                **native, **annotation,
                "failure_root_cause": None, "next_action": next_action,
            }
            record["missing_target_reasons"] = annotate_missing(record, global_indices[video])
            records_by_key[key] = record
        full_records = [records_by_key[str(x["unit_key"])] for x in validation]
        fixed_records = [records_by_key[str(x["unit_key"])] for x in fixed_v2]
        # Preserve the exact L62 order for the fixed V2 subset.
        fixed_v2_order = [str(x["unit_key"]) for x in fixed if x.get("dataset") == "refer_kitti_v2"]
        fixed_records = [dict(records_by_key[key], scope="fixed_40_v2_subset") for key in fixed_v2_order]

        # L69's accepted full coverage numbers are a reproduction check, not a
        # model-selection signal.
        l69_contract = json.loads(L69_CONTRACT.read_text())
        expected = l69_contract["v2_validation_coverage"]
        full_summary = summarize(full_records)
        expected_checks = {
            "units": (full_summary["units"], expected["units"]),
            "target_present_units": (full_summary["target_present_units"], expected["target_present_units"]),
            "candidate_present_units": (full_summary["candidate_present_units"], expected["covered_units"]),
            "unit_coverage": (full_summary["unit_coverage"], expected["unit_coverage"]),
            "target_ids": (full_summary["target_ids"], expected["target_ids"]),
            "covered_target_ids": (full_summary["covered_target_ids"], expected["covered_target_ids"]),
            "target_level_micro_coverage": (full_summary["target_level_micro_coverage"], expected["target_micro_coverage"]),
            "present_uncovered_units": (full_summary["present_uncovered_units"], expected["present_uncovered_units"]),
            "inactive_units": (full_summary["inactive_units"], expected["inactive_units"]),
        }
        baseline_match = all(
            (a == b if isinstance(a, int) and isinstance(b, int) else math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12))
            for a, b in expected_checks.values()
        )
        if not baseline_match:
            raise AssertionError(f"L69 coverage reproduction mismatch: {expected_checks}")

        scope_summaries = {
            "full_v2_validation_768": full_summary,
            "fixed_40_v2_subset_20": summarize(fixed_records),
        }
        groupings = {
            "full_v2_by_video": group_summary(full_records, "video"),
            "full_v2_by_category": group_summary(full_records, "declared_category"),
            "fixed_v2_by_video": group_summary(fixed_records, "video"),
            "fixed_v2_by_category": group_summary(fixed_records, "declared_category"),
        }
        # Add a stable query grouping after all per-unit records have been made.
        for record in full_records + fixed_records:
            record["query_group"] = f"{record['dataset']}|{record['video']}|{record['query_id']}"
        # Recompute groupings now that query_group is materialized.
        groupings["full_v2_by_query"] = group_summary(full_records, "query_group")
        identity = target_identity_summary(views)
        all_missing = [
            {"scope": record["scope"], "unit_key": record["unit_key"], "video": record["video"], **item}
            for record in full_records + fixed_records for item in record.get("missing_target_reasons", [])
        ]
        covered_cases = []
        for record in full_records:
            if record["candidate_present"] and len(covered_cases) < 32:
                covered_cases.append({
                    "case_type": "covered_target",
                    "unit_key": record["unit_key"], "video": record["video"],
                    "target_ids": record["target_ids"], "positive_row_offsets": record["positive_row_offsets"][:16],
                    "positive_sources": record["positive_pool_counts"],
                    "candidate_count": record["candidate_count"],
                })
        cases = (all_missing[:96] + covered_cases)[:128]
        for item in cases:
            item.update({"format": "locatemot-l76-case-v1", "status": "complete", "failure_root_cause": None, "next_action": next_action})

        source_statuses = sorted({x["status"] for x in source_contracts.values()})
        source_mapping_verified = source_statuses == ["verified"]
        decision = (
            "candidate_coverage_blocked"
            if full_summary["unit_coverage"] < 0.7233333 or full_summary["target_level_micro_coverage"] < 0.80
            else "coverage_adequate_correspondence_remains_bottleneck"
        )
        coverage = add_common({
            "format": "locatemot-l76-v2-candidate-coverage-v1",
            "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
            "scope": {
                "full_v2_validation": {"videos": list(FULL_V2_VIDEOS), "units": 768, "labels_posthoc_only": True},
                "fixed_40_v2_subset": {"units": 20, "order_source": str(FIXED_ROWS), "order": fixed_v2_order, "labels_posthoc_only": True},
            },
            "baseline_reproduction": {"source": str(L69_CONTRACT), "checks": expected_checks, "exact_match": baseline_match},
            "decision": {
                "label": decision,
                "full_unit_coverage": full_summary["unit_coverage"],
                "full_target_level_micro_coverage": full_summary["target_level_micro_coverage"],
                "target_level_missing_is_systematic": full_summary["target_level_micro_coverage"] < 0.80,
                "next_action": next_action,
            },
            "scope_summaries": scope_summaries,
            "groupings": groupings,
            "source_mapping_verified": source_mapping_verified,
            "source_contracts": source_contracts,
            "identity_fragment_oracle": {"label": "GT_PRIVILEGED_ORACLE", "by_video": identity},
            "missing_target_reason_definitions": {
                "absent_from_bank": "target_id appears in no L69 sidecar row in the video",
                "present_in_wrong_fragment_or_other_frame": "target_id appears in the video bank but not in this native frame's rows; this is not proof of a wrong tracker fragment",
                "present_but_attached_to_unusable_row": "target_id is attached to a current row that fails the valid pool/track contract",
            },
            "missing_target_reason_counts": {
                "full_v2_validation": full_summary["missing_target_reason_counts"],
                "fixed_40_v2_subset": summarize(fixed_records)["missing_target_reason_counts"],
            },
            "iou_membership_contract": full_summary["geometric_iou"],
            "l75_precision_interpretation": {
                "l75_scores_used_as_input": False,
                "conclusion": "coverage limits attainable recall on uncovered targets, but cannot by itself explain false-positive volume; the GT-privileged candidate-only oracle is separate from L75 correspondence scores, and covered units retain many non-target rows",
                "basis": "coverage/oracle accounting only; no L75 score, threshold, top-k, NMS, NULL or learned state read",
            },
            "flags": {
                "screening_gt_used": False, "official_test_labels_read": False,
                "ordinary_mot_ovmot_touched": False, "training_run": False,
                "hota_trackeval_run": False, "l75_scores_used": False,
                "candidate_deletion": False, "candidate_truncation": False,
                "raw_image_cache_written": False, "dense_feature_cache_written": False,
            },
            "elapsed_seconds": time.perf_counter() - started,
            "failure_root_cause": None,
        }, "complete", command, base_inputs, [str(out / x) for x in output_names], next_action)
        provenance = add_common({
            "format": "locatemot-l76-v2-coverage-provenance-v1",
            "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
            "python": sys.executable, "python_version": sys.version,
            "torch_version": torch.__version__, "numpy_version": np.__version__,
            "label_use": "post-hoc GT candidate_gt/target_ids coverage and oracle only",
            "addressing_contract": "L69 native frame_ptr/frame_ids; L49 begin/end and positive_indices not used",
            "fixed_slice_key_order": fixed_keys,
            "full_v2_videos": list(FULL_V2_VIDEOS), "fixed_v2_videos": sorted({str(x["video"]) for x in fixed_v2}),
            "inputs": [file_meta(Path(p)) for p in [DATA / "validation_units.jsonl", DATA / "calibration_units.jsonl", FIXED_ROWS, MANIFEST, L69_CONTRACT, L67_BASELINE]],
            "bank_inputs": [
                {"bank": file_meta(views[v]["bank_path"]), "sidecar": file_meta(views[v]["label_path"]), "source_contract": source_contracts[v]}
                for v in sorted(views)
            ],
            "l69_contract_expected": expected,
            "l69_baseline_match": baseline_match,
            "l75_scores_read": False,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "training_run": False,
            "hota_trackeval_run": False, "raw_or_dense_cache_written": False,
            "failure_root_cause": None,
        }, "complete", command, base_inputs, [str(out / x) for x in output_names], next_action)
        status = add_common({
            "format": "locatemot-l76-v2-coverage-status-v1",
            "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
            "decision": decision, "baseline_match": baseline_match,
            "units_full_v2": len(full_records), "units_fixed_v2": len(fixed_records),
            "row_count_full_v2": full_summary["candidate_rows"],
            "failure_root_cause": None,
        }, "complete", command, base_inputs, [str(out / x) for x in output_names], next_action)
        write_json(out / "coverage.json", coverage)
        write_jsonl(out / "unit_records.jsonl", full_records + fixed_records)
        write_jsonl(out / "case_records.jsonl", cases)
        write_json(out / "provenance.json", provenance)
        write_json(out / "status.json", status)
        with (out / "unit_summary.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["scope", "unit_key", "video", "query_id", "frame_id", "category", "candidate_count", "positive_count", "candidate_present", "target_id_coverage", "main_rows", "reserve_rows", "candidate_index_duplicate_count", "duplicate_track_count"])
            for record in full_records + fixed_records:
                writer.writerow([record["scope"], record["unit_key"], record["video"], record["query_id"], record["frame_id"], record["declared_category"], record["candidate_count"], record["positive_count"], record["candidate_present"], record["target_id_coverage"], record["pool_counts"].get("main", 0), record["pool_counts"].get("reserve", 0), record["candidate_index_duplicate_count"], record["duplicate_track_count"]])
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        failure = {
            "format": "locatemot-l76-v2-candidate-coverage-v1",
            "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()),
            "failure_root_cause": f"{type(exc).__name__}: {exc}",
            "traceback": tb,
        }
        write_json(out / "status.json", add_common(failure, "invalid", command, base_inputs, [str(out / x) for x in output_names], "fix only the first audit contract error and rerun in a new attempt"))
        (out / "INCOMPLETE.md").write_text("# L76 incomplete audit\n\nFirst actionable error:\n\n```text\n" + tb + "```\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
