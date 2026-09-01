#!/usr/bin/env python3
"""Read-only Refer-KITTI-V2 L19 candidate/proposal coverage upper-bound audit."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
DATA = ROOT / "outputs/l49/data"
BANK_ROOT = ROOT / "outputs/l19/dual_banks_features/kitti"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
L66_ROWS = ROOT / "outputs/l62/eval/semantic_16cal24val_retry2/score_records.jsonl"
EXPECTED_MANIFEST = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
VALIDATION_VIDEOS = ("0016", "0017", "0020")
REQUIRED_TENSORS = (
    "frame_ids", "frame_ptr", "frame", "box", "track_id", "candidate_index",
    "pool_id", "clip", "history_clip", "geometry", "motion", "context",
    "lifecycle", "objectness", "source_score",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def unit_rows(path: Path, dataset: str, split: str, videos=None):
    result = []
    wanted = None if videos is None else set(str(x) for x in videos)
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("dataset") != dataset or row.get("split") != split:
            continue
        if wanted is not None and str(row.get("video")) not in wanted:
            continue
        result.append(row)
    return result


def fixed_slice_keys():
    return [json.loads(line)["unit_key"] for line in L66_ROWS.read_text().splitlines() if line.strip()]


def xyxy_iou(left, right):
    x1 = max(float(left[0]), float(right[0]))
    y1 = max(float(left[1]), float(right[1]))
    x2 = min(float(left[2]), float(right[2]))
    y2 = min(float(left[3]), float(right[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a = max(0.0, float(left[2] - left[0])) * max(0.0, float(left[3] - left[1]))
    b = max(0.0, float(right[2] - right[0])) * max(0.0, float(right[3] - right[1]))
    return inter / max(1e-8, a + b - inter)


def finite_tensor(tensor):
    return bool(torch.isfinite(tensor.float()).all())


def source_contract(bank, bank_path):
    metadata = bank.get("metadata", {})
    tensors = bank["tensors"]
    values = sorted(set(int(x) for x in tensors["pool_id"].tolist()))
    # This mapping is not inferred from the observed values alone.  It is
    # verified against the frozen L19 builder source and its metadata.
    builder = ROOT / "tools/build_l19_reserve_identity.py"
    source_text = builder.read_text()
    mapping_verified = (
        "main = source_pool == 0" in source_text
        and "reserve = source_pool == 1" in source_text
        and "reserve_id_offset" in metadata
        and metadata.get("preserve_source_ids") is True
        and set(values) <= {0, 1}
    )
    return {
        "status": "verified" if mapping_verified else "source_split_unavailable",
        "pool_values_observed": values,
        "mapping": {"0": "main", "1": "reserve"} if mapping_verified else None,
        "basis": [str(builder), "builder source contains explicit pool==0/main and pool==1/reserve",
                  "L19 metadata preserve_source_ids and reserve_id_offset"] if mapping_verified else [],
        "builder_sha256": sha256(builder),
        "bank_metadata_main_source": metadata.get("main_source"),
        "bank_metadata_reserve_source": metadata.get("reserve_source"),
        "bank_metadata_reserve_id_offset": metadata.get("reserve_id_offset"),
        "bank": str(bank_path),
    }


def row_key(unit, candidate_index, track_id, row_offset):
    return [str(unit["dataset"]), str(unit["video"]), int(unit["query_id"]),
            int(unit["frame_id"]), int(candidate_index), int(track_id), int(row_offset)]


def init_error_map():
    return {name: {"count": 0, "denominator": 0, "examples": []} for name in (
        "no_candidate_rows", "candidate_rows_but_no_target_id", "sidecar_mismatch",
        "target_metadata_missing")}


def note_error(errors, name, denominator, example):
    errors[name]["denominator"] += int(denominator)
    errors[name]["count"] += 1
    if len(errors[name]["examples"]) < 8:
        errors[name]["examples"].append(example)


def aggregate_stats(values):
    values = np.asarray(values, dtype=float)
    if not values.size:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"count": int(values.size), "mean": float(values.mean()),
            "median": float(np.median(values)), "min": float(values.min()),
            "max": float(values.max())}


def by_group(records, key_fn):
    groups = defaultdict(list)
    for row in records:
        groups[key_fn(row)].append(row)
    return groups


def compact_group_stats(records):
    if not records:
        return {"units": 0, "target_present_units": 0, "covered_units": 0,
                "unit_coverage": None, "target_ids": 0, "covered_target_ids": 0,
                "target_level_micro_coverage": None, "target_level_macro_coverage": None,
                "candidate_rows": 0, "positive_rows": 0, "candidate_count": {}}
    target_units = [x for x in records if x["target_present"]]
    covered = [x for x in target_units if x["candidate_covered"]]
    target_total = sum(len(x["target_ids"]) for x in target_units)
    target_covered = sum(x["target_ids_covered"] for x in target_units)
    candidate_counts = [x["candidate_count"] for x in records]
    return {
        "units": len(records), "target_present_units": len(target_units),
        "inactive_units": sum(not x["target_present"] for x in records),
        "covered_units": len(covered),
        "unit_coverage": len(covered) / max(1, len(target_units)),
        "target_ids": target_total, "covered_target_ids": target_covered,
        "target_level_micro_coverage": target_covered / max(1, target_total),
        "target_level_macro_coverage": float(np.mean([x["target_id_coverage"] for x in target_units])) if target_units else None,
        "candidate_rows": int(sum(x["candidate_count"] for x in records)),
        "positive_rows": int(sum(x["positive_count"] for x in records)),
        "candidate_count": aggregate_stats(candidate_counts),
        "positive_count": aggregate_stats([x["positive_count"] for x in records]),
        "present_uncovered_units": sum(x["category"] == "present_uncovered" for x in records),
    }


def source_coverage(records):
    target_present = [x for x in records if x["target_present"]]
    result = {}
    for source, field in (("main", "main_covered_target_ids"),
                          ("reserve", "reserve_covered_target_ids")):
        covered = [x for x in target_present if x[field]]
        total_targets = sum(len(x["target_ids"]) for x in target_present)
        covered_targets = sum(len(set(x[field])) for x in target_present)
        result[source] = {
            "target_present_units": len(target_present),
            "covered_units": len(covered),
            "unit_coverage": len(covered) / max(1, len(target_present)),
            "target_ids": total_targets,
            "covered_target_ids": covered_targets,
            "target_level_micro_coverage": covered_targets / max(1, total_targets),
            "target_level_macro_coverage": float(np.mean([
                len(set(x[field])) / max(1, len(x["target_ids"])) for x in target_present
            ])) if target_present else None,
        }
    union_covered = [set(x["main_covered_target_ids"]) | set(x["reserve_covered_target_ids"])
                     for x in target_present]
    total_targets = sum(len(x["target_ids"]) for x in target_present)
    result["union"] = {
        "target_present_units": len(target_present),
        "covered_units": sum(bool(x) for x in union_covered),
        "unit_coverage": sum(bool(x) for x in union_covered) / max(1, len(target_present)),
        "target_ids": total_targets,
        "covered_target_ids": sum(len(x) for x in union_covered),
        "target_level_micro_coverage": sum(len(x) for x in union_covered) / max(1, total_targets),
        "target_level_macro_coverage": float(np.mean([
            len(x) / max(1, len(row["target_ids"])) for x, row in zip(union_covered, target_present)
        ])) if target_present else None,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    command = " ".join([sys.executable] + sys.argv)
    status_path = out / "status.json"
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd: {Path.cwd()}")
        manifest_sha = sha256(MANIFEST)
        if manifest_sha != EXPECTED_MANIFEST:
            raise AssertionError(f"manifest SHA mismatch: {manifest_sha}")
        all_validation = unit_rows(DATA / "validation_units.jsonl", "refer_kitti_v2", "validation", VALIDATION_VIDEOS)
        if not all_validation:
            raise AssertionError("no V2 validation units")
        if set(str(x["video"]) for x in all_validation) != set(VALIDATION_VIDEOS):
            raise AssertionError("V2 validation video set mismatch")
        context_units = []
        for filename, split in (("calibration_units.jsonl", "calibration"),
                                ("validation_units.jsonl", "validation")):
            for dataset in ("refer_kitti_v1", "refer_kitti_v2"):
                context_units.extend(unit_rows(DATA / filename, dataset, split, None))
        fixed_keys = fixed_slice_keys()
        if len(fixed_keys) != 40 or len(set(fixed_keys)) != 40:
            raise AssertionError("L66 fixed slice is not 40 unique units")
        fixed_set = set(fixed_keys)
        records = []
        bank_summaries = {}
        source_summaries = {}
        errors = init_error_map()
        per_video_counts = Counter()
        per_category_counts = Counter()
        unit_lookup = {x["unit_key"]: x for x in all_validation}
        context_lookup = {x["unit_key"]: x for x in context_units}
        fixed_v2_missing = sorted(k for k in fixed_set if k.startswith("refer_kitti_v2|") and k not in context_lookup)
        if fixed_v2_missing:
            raise AssertionError(f"fixed L66 V2 keys missing from calibration/validation units: {fixed_v2_missing[:3]}")
        audit_units = list(all_validation)
        audit_keys = {x["unit_key"] for x in audit_units}
        for key in fixed_keys:
            if key not in audit_keys:
                if key not in context_lookup:
                    raise AssertionError(f"fixed L66 key missing from calibration/validation units: {key}")
                audit_units.append(context_lookup[key])
                audit_keys.add(key)
        audit_videos = sorted({str(x["video"]) for x in audit_units})
        all_source_temporal = {}

        for video in audit_videos:
            bank_path = (BANK_ROOT / f"{video}.pt").resolve()
            label_path = bank_path.with_suffix(".labels.json")
            if not bank_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(f"missing V2 bank/sidecar for {video}")
            bank = torch.load(bank_path, map_location="cpu", weights_only=False)
            tensors = bank["tensors"]
            missing_fields = [x for x in REQUIRED_TENSORS if x not in tensors]
            if missing_fields:
                raise AssertionError(f"{video} missing bank fields {missing_fields}")
            total = int(tensors["track_id"].numel())
            sidecar = json.loads(label_path.read_text())
            candidate_gt = sidecar.get("candidate_gt") if isinstance(sidecar, dict) else None
            if not isinstance(candidate_gt, list) or len(candidate_gt) != total:
                raise AssertionError(f"sidecar length/type mismatch {video}: {len(candidate_gt) if isinstance(candidate_gt,list) else None} vs {total}")
            candidate_gt = [None if x is None else str(x) for x in candidate_gt]
            frame_ids = tensors["frame_ids"].long()
            frame_ptr = tensors["frame_ptr"].long()
            frame = tensors["frame"].long()
            if frame_ptr.numel() != frame_ids.numel() + 1 or int(frame_ptr[-1]) != total:
                raise AssertionError(f"frame_ptr contract mismatch {video}")
            if not all(finite_tensor(tensors[x]) for x in ("box", "clip", "history_clip", "geometry", "motion", "context", "lifecycle", "objectness", "source_score")):
                raise AssertionError(f"nonfinite bank values {video}")
            source_contract_info = source_contract(bank, bank_path)
            source_summaries[video] = source_contract_info
            pool = tensors["pool_id"].long().tolist()
            track = tensors["track_id"].long().tolist()
            bank_frame = frame.tolist()
            # Bank-wide descriptive pool/GT transition audit, kept separate
            # from query-unit coverage metrics.
            gt_pool_frames = defaultdict(lambda: defaultdict(set))
            for row_index, gt in enumerate(candidate_gt):
                if gt is not None:
                    gt_pool_frames[gt][int(pool[row_index])].add(int(bank_frame[row_index]))
            main_to_reserve = []
            for gt, pools in gt_pool_frames.items():
                mains = pools.get(0, set()); reserves = pools.get(1, set())
                if mains and reserves:
                    ordered_main = sorted(mains)
                    main_to_reserve.append({"gt_id": gt, "main_frames": len(mains), "reserve_frames": len(reserves),
                                            "has_later_reserve": any(max(reserves) > m for m in ordered_main),
                                            "same_frame_overlap": bool(mains & reserves)})
            all_source_temporal[video] = main_to_reserve
            video_units = [x for x in audit_units if str(x["video"]) == video]
            for unit in video_units:
                begin, end = int(unit["begin"]), int(unit["end"])
                expected_n = end - begin
                if expected_n != int(unit["candidate_count"]):
                    raise AssertionError(f"candidate_count mismatch {unit['unit_key']}")
                fi = int(unit["frame_index"])
                if fi < 0 or fi + 1 >= frame_ptr.numel() or int(frame_ids[fi]) != int(unit["frame_id"]):
                    raise AssertionError(f"frame index/id mismatch {unit['unit_key']}")
                if int(frame_ptr[fi]) != begin or int(frame_ptr[fi + 1]) != end:
                    raise AssertionError(f"frame_ptr slice mismatch {unit['unit_key']}")
                if not all(int(x) == int(unit["frame_id"]) for x in frame[begin:end].tolist()):
                    raise AssertionError(f"row frame mismatch {unit['unit_key']}")
                unit_targets = {str(x) for x in unit.get("target_ids", [])}
                if unit.get("target_ids") is None:
                    note_error(errors, "target_metadata_missing", 1, {"unit_key": unit["unit_key"], "target_ids": None})
                    raise AssertionError(f"target metadata missing {unit['unit_key']}")
                rows_gt = candidate_gt[begin:end]
                positive = [x is not None and x in unit_targets for x in rows_gt]
                expected_positive = [int(x) for x in unit.get("positive_indices", [])]
                actual_positive = [i for i, value in enumerate(positive) if value]
                if expected_positive != actual_positive:
                    example = {"unit_key": unit["unit_key"], "expected_positive_indices": expected_positive,
                               "actual_positive_indices": actual_positive, "target_ids": sorted(unit_targets)}
                    note_error(errors, "sidecar_mismatch", 1, example)
                    raise AssertionError(f"sidecar/unit target mismatch {unit['unit_key']}")
                row_offsets = list(range(begin, end))
                candidate_indices = [int(x) for x in tensors["candidate_index"][begin:end].tolist()]
                track_ids = [int(x) for x in tensors["track_id"][begin:end].tolist()]
                keys = [row_key(unit, ci, tid, ro) for ci, tid, ro in zip(candidate_indices, track_ids, row_offsets)]
                if len(keys) != expected_n or len(set(tuple(x) for x in keys)) != expected_n:
                    raise AssertionError(f"duplicate immutable row key {unit['unit_key']}")
                pools = [int(x) for x in tensors["pool_id"][begin:end].tolist()]
                positive_ids = sorted({rows_gt[i] for i in actual_positive if rows_gt[i] is not None})
                covered_ids = sorted(unit_targets.intersection(set(x for x in rows_gt if x is not None)))
                target_present = bool(unit_targets)
                candidate_covered = bool(covered_ids)
                category = str(unit.get("category", "unknown"))
                if not target_present:
                    if category != "inactive":
                        raise AssertionError(f"non-inactive unit has no targets {unit['unit_key']}")
                elif not candidate_covered:
                    note_error(errors, "candidate_rows_but_no_target_id", 1, {"unit_key": unit["unit_key"],
                              "target_ids": sorted(unit_targets), "candidate_gt_ids": sorted(set(x for x in rows_gt if x is not None))})
                if expected_n == 0:
                    note_error(errors, "no_candidate_rows", 1, {"unit_key": unit["unit_key"], "target_ids": sorted(unit_targets)})
                boxes = tensors["box"][begin:end].float().tolist()
                pos_count = len(actual_positive)
                neg_count = expected_n - pos_count
                adjacent_positive_negative = 0
                for p in actual_positive:
                    for n in range(expected_n):
                        if n not in actual_positive and xyxy_iou(boxes[p], boxes[n]) >= 0.30:
                            adjacent_positive_negative += 1
                duplicate_candidate_count = expected_n - len(set(candidate_indices))
                duplicate_track_count = expected_n - len(set(track_ids))
                cross_pool_duplicates = int(sum(int(x) != 0 for x in tensors.get("cross_pool_duplicate", torch.zeros(total))[begin:end].tolist()))
                main_pos = sum(1 for i in actual_positive if pools[i] == 0)
                reserve_pos = sum(1 for i in actual_positive if pools[i] == 1)
                source_known = source_contract_info["status"] == "verified"
                same_unit_main_reserve = {
                    "target_ids_with_main_positive": sorted({rows_gt[i] for i in actual_positive if pools[i] == 0}),
                    "target_ids_with_reserve_positive": sorted({rows_gt[i] for i in actual_positive if pools[i] == 1}),
                }
                target_main = set(same_unit_main_reserve["target_ids_with_main_positive"])
                target_reserve = set(same_unit_main_reserve["target_ids_with_reserve_positive"])
                per_video_counts[video] += 1
                per_category_counts[(video, category)] += 1
                records.append({
                    "format": "locatemot-l67-v2-coverage-unit-v1", "status": "complete",
                    "unit_key": unit["unit_key"], "dataset": unit["dataset"], "video": str(unit["video"]),
                    "unit_split": str(unit["split"]),
                    "query_id": int(unit["query_id"]), "frame_id": int(unit["frame_id"]),
                    "category": category, "target_present": target_present,
                    "target_ids": sorted(unit_targets), "target_id_count": len(unit_targets),
                    "covered_target_ids": covered_ids, "target_ids_covered": len(covered_ids),
                    "target_id_coverage": len(covered_ids) / max(1, len(unit_targets)) if target_present else None,
                    "candidate_covered": candidate_covered, "candidate_count": expected_n,
                    "positive_count": pos_count, "positive_candidate_gt_ids": positive_ids,
                    "candidate_gt_nonempty_ids": sorted(set(x for x in rows_gt if x is not None)),
                    "candidate_rows_but_no_target_id": bool(target_present and not candidate_covered),
                    "empty_candidate": expected_n == 0, "candidate_indices_unique": duplicate_candidate_count == 0,
                    "duplicate_candidate_index_count": duplicate_candidate_count,
                    "duplicate_track_id_count": duplicate_track_count,
                    "cross_pool_duplicate_row_count": cross_pool_duplicates,
                    "cross_pool_duplicate_fraction": cross_pool_duplicates / max(1, expected_n),
                    "pool_values": sorted(set(pools)), "source_mapping_status": source_contract_info["status"],
                    "main_candidate_count": sum(x == 0 for x in pools) if source_known else None,
                    "reserve_candidate_count": sum(x == 1 for x in pools) if source_known else None,
                    "main_positive_count": main_pos if source_known else None,
                    "reserve_positive_count": reserve_pos if source_known else None,
                    "main_covered_target_ids": sorted(target_main) if source_known else [],
                    "reserve_covered_target_ids": sorted(target_reserve) if source_known else [],
                    "same_unit_main_to_reserve_target_recall": len(target_main & target_reserve) / max(1, len(target_main)) if source_known else None,
                    "same_frame_positive_negative_pairs_iou_ge_030": adjacent_positive_negative,
                    "same_class_hard_negative_metadata": "unavailable",
                    "descriptive_negative_count": neg_count,
                    "immutable_row_keys": keys,
                    "inputs": {"bank_path": str(bank_path), "label_path": str(label_path),
                               "begin": begin, "end": end, "frame_index": fi},
                    "fixed_l66_v2_slice": unit["unit_key"] in fixed_set,
                })
            bank_summaries[video] = {
                "bank_path": str(bank_path), "bank_sha256": sha256(bank_path),
                "label_path": str(label_path), "label_sha256": sha256(label_path),
                "rows": total, "frames": int(frame_ids.numel()),
                "metadata": {k: bank.get("metadata", {}).get(k) for k in (
                    "format", "video_id", "split", "image_size", "main_source", "reserve_source",
                    "main_observations", "reserve_observations", "observations", "cross_pool_duplicate_rows",
                    "reserve_id_offset", "preserve_source_ids", "causal", "query_independent")},
                "tensor_fields": sorted(tensors.keys()),
                "source_contract": source_contract_info,
                "bank_temporal_main_to_reserve_gt_count": len(all_source_temporal[video]),
                "bank_temporal_main_to_reserve_later_count": sum(x["has_later_reserve"] for x in all_source_temporal[video]),
            }
            del bank, tensors, candidate_gt
            print(f"[l67-coverage] completed video {video} ({len(video_units)} units)", flush=True)

        # The primary full-V2 aggregate remains in validation-file order;
        # fixed-slice rows are identified separately and never re-ordered into
        # the aggregate.
        all_audit_records = list(records)
        records_by_key = {x["unit_key"]: x for x in all_audit_records}
        records = [records_by_key[x["unit_key"]] for x in all_validation]
        if len(records) != len(all_validation) or len({x["unit_key"] for x in records}) != len(records):
            raise AssertionError("V2 validation unit coverage/key mismatch")
        group_all = compact_group_stats(records)
        group_video = {video: compact_group_stats([x for x in records if x["video"] == video]) for video in VALIDATION_VIDEOS}
        group_category = {f"{video}|{category}": compact_group_stats([x for x in records if x["video"] == video and x["category"] == category])
                          for video in VALIDATION_VIDEOS for category in ("positive", "multi_positive", "inactive", "present_uncovered")}
        fixed_v2_records = [x for x in all_audit_records if x["fixed_l66_v2_slice"] and x["dataset"] == "refer_kitti_v2"]
        fixed_v1_records = []
        fixed_all_records = []
        validation_order = [json.loads(line)["unit_key"] for line in L66_ROWS.read_text().splitlines() if line.strip()]
        # L66 slice rows are read from the same immutable 40-row source only
        # for context; V1 and V2 are never merged into the V2 aggregate.
        by_key = {x["unit_key"]: x for x in all_audit_records}
        for key in validation_order:
            if key in by_key:
                fixed_all_records.append(by_key[key])
        fixed_slice_summary = {
            "all_l66_fixed_rows": len(fixed_all_records),
            "v2_rows": len(fixed_v2_records),
            "v1_rows": len(fixed_all_records) - len(fixed_v2_records),
            "v2": compact_group_stats(fixed_v2_records),
            "v1_context_not_merged": True,
        }
        target_present = [x for x in records if x["target_present"]]
        covered = [x for x in target_present if x["candidate_covered"]]
        oracle_selected = sum(x["positive_count"] for x in records)
        oracle_candidate_rows = sum(x["candidate_count"] for x in records)
        source_mapping_status = "verified" if all(x["status"] == "verified" for x in source_summaries.values()) else "source_split_unavailable"
        temporal_summary = {}
        for video, items in all_source_temporal.items():
            temporal_summary[video] = {
                "gt_with_both_pools": len(items),
                "gt_with_later_reserve_after_main": sum(x["has_later_reserve"] for x in items),
                "gt_with_same_frame_main_reserve": sum(x["same_frame_overlap"] for x in items),
                "finite_sample_note": "bank-wide sidecar diagnostic; not query-conditioned model performance",
            }
        primary_errors = init_error_map()
        for primary in records:
            if primary["empty_candidate"]:
                note_error(primary_errors, "no_candidate_rows", 1,
                           {"unit_key": primary["unit_key"], "target_ids": primary["target_ids"]})
            if primary["candidate_rows_but_no_target_id"]:
                note_error(primary_errors, "candidate_rows_but_no_target_id", 1,
                           {"unit_key": primary["unit_key"], "target_ids": primary["target_ids"],
                            "candidate_gt_ids": primary["candidate_gt_nonempty_ids"]})
        coverage = {
            "format": "locatemot-l67-v2-candidate-coverage-v1", "status": "complete",
            "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()), "command": command,
            "inputs": {"validation_units": str(DATA / "validation_units.jsonl"), "l66_rows": str(L66_ROWS),
                        "banks": [str(BANK_ROOT / f"{v}.pt") for v in VALIDATION_VIDEOS],
                        "sidecars": [str(BANK_ROOT / f"{v}.labels.json") for v in VALIDATION_VIDEOS],
                        "manifest": str(MANIFEST), "manifest_sha256": manifest_sha},
            "outputs": [str(out / x) for x in ("coverage.json", "unit_records.jsonl", "provenance.json", "status.json")],
            "failure_root_cause": None, "next_action": None,
            "scope": {"dataset": "refer_kitti_v2", "split": "validation", "videos": list(VALIDATION_VIDEOS),
                      "units": len(records), "fixed_l66_v2_slice_separate": True},
            "coverage_ceiling": {
                "target_present_units": len(target_present), "inactive_units": sum(not x["target_present"] for x in records),
                "covered_target_present_units": len(covered),
                "target_present_unit_coverage": len(covered) / max(1, len(target_present)),
                "present_uncovered_units": sum(x["category"] == "present_uncovered" for x in records),
                "target_ids": sum(len(x["target_ids"]) for x in target_present),
                "covered_target_ids": sum(x["target_ids_covered"] for x in target_present),
                "target_level_micro_coverage": sum(x["target_ids_covered"] for x in target_present) / max(1, sum(len(x["target_ids"]) for x in target_present)),
                "target_level_macro_coverage": float(np.mean([x["target_id_coverage"] for x in target_present])) if target_present else None,
                "ideal_gt_oracle": {
                    "label": "ORACLE",
                    "selected_sidecar_positive_rows": oracle_selected,
                    "all_candidate_rows": oracle_candidate_rows,
                    "candidate_recall_on_covered_rows": 1.0 if oracle_selected else None,
                    "candidate_recall_over_target_present_units": len(covered) / max(1, len(target_present)),
                    "theoretical_candidate_precision": oracle_selected / max(1, oracle_selected),
                    "note": "GT-privileged sidecar-positive selection only; not a deployable score, model, or HOTA result",
                },
            },
            "candidate_statistics": {
                "candidate_count": aggregate_stats([x["candidate_count"] for x in records]),
                "positive_count": aggregate_stats([x["positive_count"] for x in records]),
                "empty_candidate_units": sum(x["empty_candidate"] for x in records),
                "duplicate_candidate_index_rows": sum(x["duplicate_candidate_index_count"] for x in records),
                "duplicate_track_rows": sum(x["duplicate_track_id_count"] for x in records),
                "cross_pool_duplicate_rows": sum(x["cross_pool_duplicate_row_count"] for x in records),
                "cross_pool_duplicate_fraction": sum(x["cross_pool_duplicate_row_count"] for x in records) / max(1, sum(x["candidate_count"] for x in records)),
            },
            "by_video": group_video, "by_video_category": group_category,
            "fixed_l66_slice": fixed_slice_summary,
            "source_mapping": {"status": source_mapping_status, "per_video": source_summaries,
                               "main_only_reserve_only_allowed": source_mapping_status == "verified",
                               "v2_validation_main_reserve_union": source_coverage(records) if source_mapping_status == "verified" else None,
                               "fixed_l66_v2_main_reserve_union": source_coverage(fixed_v2_records) if source_mapping_status == "verified" else None},
            "main_reserve_continuation": temporal_summary,
            "failure_reason_decomposition": primary_errors,
            "same_frame_descriptive_hard_negative": {
                "same_class_metadata": "unavailable",
                "proxy_definition": "candidate_gt not in current unit target_ids; positive-negative box IoU >= 0.30 counted descriptively",
                "negative_rows": sum(x["descriptive_negative_count"] for x in records),
                "positive_negative_iou_ge_030_pairs": sum(x["same_frame_positive_negative_pairs_iou_ge_030"] for x in records),
                "not_used_for_score_or_selection": True,
            },
            "decision_rules": {
                "candidate_coverage_blocked": "V2 target-present validation unit coverage < 0.7233333 OR systematic target-level missing",
                "candidate_coverage_adequate_representation_unknown": "union coverage >= 0.7233333 and target missing not primary, but source continuation/hard-negative composition unstable",
                "candidate_coverage_adequate_language_bottleneck": "union coverage adequate, target missing rare, and L66 hard-negative failure remains dominant",
                "thresholds_fixed_before_interpretation": True,
            },
            "labels": {"source": "candidate_gt sidecar intersected with unit target_ids; post-hoc oracle audit only",
                       "screening_gt_used": False, "official_test_labels_read": False},
            "training_run": False, "hota_trackeval_run": False,
            "ordinary_mot_ovmot_touched": False, "dense_or_raw_cache_written": False,
        }
        v2_coverage = float(coverage["coverage_ceiling"]["target_present_unit_coverage"])
        target_level = float(coverage["coverage_ceiling"]["target_level_micro_coverage"])
        systematic_missing = bool(coverage["coverage_ceiling"]["present_uncovered_units"] > 0 or target_level < 0.7233333)
        if v2_coverage < 0.7233333 or systematic_missing:
            decision = "candidate_coverage_blocked"
            next_action = "one candidate-generation/coverage repair design audit; no language decoder or LoRA long training"
        elif source_mapping_status != "verified" or any(x["bank_temporal_main_to_reserve_later_count"] == 0 for x in bank_summaries.values()):
            decision = "candidate_coverage_adequate_representation_unknown"
            next_action = "one independent fragment-association probe"
        else:
            decision = "candidate_coverage_adequate_language_bottleneck"
            next_action = "one new local visual-language/track correspondence probe, with no simultaneous changes"
        coverage["decision"] = {"label": decision, "v2_target_present_unit_coverage": v2_coverage,
                                 "target_level_micro_coverage": target_level, "next_action": next_action,
                                 "validation_labels_used_only_posthoc": True}
        coverage["next_action"] = next_action
        output_records = records + [x for x in all_audit_records if x["unit_key"] not in {r["unit_key"] for r in records}]
        for output_record in output_records:
            output_record["command"] = command
            output_record["outputs"] = [str(out / "unit_records.jsonl")]
            output_record["failure_root_cause"] = None
            output_record["next_action"] = next_action
        (out / "unit_records.jsonl").write_text("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in output_records))
        (out / "coverage.json").write_text(json.dumps(coverage, indent=2, default=jsonable) + "\n")
        provenance = {
            "format": "locatemot-l67-v2-coverage-provenance-v1", "status": "complete",
            "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()), "command": command,
            "inputs": coverage["inputs"], "outputs": coverage["outputs"],
            "bank_summaries": bank_summaries, "unit_count": len(records),
            "fixed_l66_v2_rows": len(fixed_v2_records), "immutable_row_key": "dataset,video,query_id,frame_id,candidate_index,track_id,row_offset",
            "candidate_labels_posthoc_only": True, "source_mapping_status": source_mapping_status,
            "screening_gt_used": False, "official_test_labels_read": False,
            "training_run": False, "hota_trackeval_run": False,
            "ordinary_mot_ovmot_touched": False, "dense_or_raw_cache_written": False,
            "failure_root_cause": None, "next_action": next_action,
        }
        (out / "provenance.json").write_text(json.dumps(provenance, indent=2, default=jsonable) + "\n")
        final_status = {"format": "locatemot-l67-v2-coverage-status-v1", "status": "complete",
                        "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()), "command": command,
                        "inputs": coverage["inputs"], "outputs": coverage["outputs"],
                        "failure_root_cause": None, "next_action": next_action,
                        "decision": decision, "units": len(records), "videos": list(VALIDATION_VIDEOS),
                        "elapsed_sec": time.time() - started, "training_run": False,
                        "hota_trackeval_run": False, "screening_gt_used": False,
                        "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False}
        status_path.write_text(json.dumps(final_status, indent=2) + "\n")
        print(json.dumps({"status": "complete", "decision": decision, "units": len(records),
                          "target_present_unit_coverage": v2_coverage,
                          "target_level_micro_coverage": target_level,
                          "output": str(out)}, indent=2), flush=True)
    except Exception as exc:
        failure = {"format": "locatemot-l67-v2-coverage-status-v1", "status": "INCOMPLETE",
                   "project_root": str(ROOT), "cwd": str(Path.cwd().resolve()), "command": command,
                   "inputs": {"validation_units": str(DATA / "validation_units.jsonl"), "manifest": str(MANIFEST)},
                   "outputs": [str(out / "INCOMPLETE.md")], "failure_root_cause": repr(exc),
                   "next_action": "inspect traceback and make one minimal targeted repair in a new attempt",
                   "training_run": False, "hota_trackeval_run": False,
                   "screening_gt_used": False, "official_test_labels_read": False,
                   "ordinary_mot_ovmot_touched": False}
        status_path.write_text(json.dumps(failure, indent=2) + "\n")
        (out / "INCOMPLETE.md").write_text("# L67 INCOMPLETE\n\n```text\n" + traceback.format_exc() + "\n```\n")
        raise


if __name__ == "__main__":
    main()
