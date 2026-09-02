#!/usr/bin/env python3
"""Build the L82 fit-only frame x expression supervision matrix.

This is an audit/data materialization step, not a model or a proposal step.
It reads only the L49 fit units and the immutable L69 candidate sidecars after
the native frame rows have been reconstructed.  No calibration, validation,
screening, or official-test unit file is opened by this script.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import re
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
DATA_ROOT = ROOT / "outputs/l49/data"
L69_ROOT = ROOT / "outputs/l69/attempt9/budget40_features/kitti"
MANIFEST = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
L69_CONTRACT = ROOT / "outputs/l69/audit/budget40_bank_contract_attempt13_full/contract.json"
EXPECTED_MANIFEST_SHA = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
FIT_VIDEOS = {
    "0000", "0001", "0002", "0003", "0006", "0007", "0008", "0009",
    "0010", "0012", "0014", "0015", "0020",
}
DATASETS = ("refer_kitti_v1", "refer_kitti_v2")
CATEGORIES = ("positive", "multi_positive", "inactive", "present_uncovered")
MATRIX_THRESHOLDS = {
    "eligible_frame_groups": 200,
    "eligible_v1_frame_groups": 40,
    "eligible_v2_frame_groups": 100,
    "exact_candidate_query_flip_triplets": 5000,
    "target_bag_query_flips": 1000,
    "fit_videos": 8,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_meta(path: Path, include_hash: bool = True) -> dict[str, Any]:
    path = path.resolve()
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "mtime_ns": path.stat().st_mtime_ns if path.is_file() else None,
    }
    if include_hash and path.is_file():
        result["sha256"] = sha256_file(path)
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def finite_tensor(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value.float()).all()):
        raise AssertionError(f"nonfinite {name}")


def load_bank(video: str) -> tuple[Path, dict[str, Any], list[Any], dict[int, tuple[int, int]]]:
    path = (L69_ROOT / f"{video}.pt").resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or not isinstance(blob.get("tensors"), dict):
        raise AssertionError(f"invalid L69 container: {path}")
    tensors = blob["tensors"]
    required = {"frame", "frame_ids", "frame_ptr", "candidate_index", "track_id", "pool_id", "box", "objectness", "raw_rank"}
    missing = required - set(tensors)
    if missing:
        raise AssertionError(f"{video}: missing fields {sorted(missing)}")
    frame_ids = tensors["frame_ids"].long()
    frame_ptr = tensors["frame_ptr"].long()
    frame_values = tensors["frame"].long()
    total = int(frame_values.numel())
    if frame_ptr.numel() != frame_ids.numel() + 1 or int(frame_ptr[-1]) != total:
        raise AssertionError(f"{video}: frame pointer total mismatch")
    if not bool(torch.all(frame_ptr[1:] >= frame_ptr[:-1])):
        raise AssertionError(f"{video}: nonmonotonic frame pointer")
    for name in ("frame", "candidate_index", "track_id", "pool_id", "box", "objectness", "raw_rank"):
        value = tensors[name]
        if name != "box" and value.numel() != total:
            raise AssertionError(f"{video}: {name} length mismatch")
        finite_tensor(value, f"{video}:{name}")
    label_path = path.with_suffix(".labels.json")
    sidecar = json.loads(label_path.read_text())
    candidate_gt = sidecar.get("candidate_gt")
    if not isinstance(candidate_gt, list) or len(candidate_gt) != total:
        raise AssertionError(f"{video}: candidate_gt length mismatch")
    frame_to_range: dict[int, tuple[int, int]] = {}
    for frame, begin, end in zip(frame_ids.tolist(), frame_ptr[:-1].tolist(), frame_ptr[1:].tolist()):
        frame = int(frame); begin = int(begin); end = int(end)
        if end > begin and not bool(torch.all(frame_values[begin:end] == frame)):
            raise AssertionError(f"{video}: frame rows disagree for frame {frame}")
        if frame in frame_to_range:
            raise AssertionError(f"{video}: duplicate frame id in frame_ids {frame}")
        frame_to_range[frame] = (begin, end)
    metadata = dict(blob.get("metadata") or {})
    metadata["label_path"] = str(label_path)
    metadata["candidate_gt_count"] = len(candidate_gt)
    return path, tensors, candidate_gt, frame_to_range


def category_for(targets: set[str], labels: list[bool]) -> str:
    if not targets:
        return "inactive"
    positive_count = int(sum(labels))
    if positive_count == 0:
        return "present_uncovered"
    if positive_count > 1:
        return "multi_positive"
    return "positive"


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def distribution(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None,
                "p05": None, "p25": None, "p50": None, "p75": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {"count": int(array.size), "min": float(array.min()), "max": float(array.max()),
            "mean": float(array.mean()), "std": float(array.std()),
            "p05": percentile(values, .05), "p25": percentile(values, .25),
            "p50": percentile(values, .50), "p75": percentile(values, .75),
            "p95": percentile(values, .95)}


def token_words(sentence: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", sentence.lower())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L82 matrix output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable] + sys.argv)
    started = time.perf_counter()
    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError(f"wrong cwd {Path.cwd()}")
        manifest_sha = sha256_file(MANIFEST)
        if manifest_sha != EXPECTED_MANIFEST_SHA:
            raise AssertionError(f"manifest SHA drift: {manifest_sha}")
        source = DATA_ROOT / "train_units.jsonl"
        units = read_jsonl(source)
        if len(units) != 5314:
            raise AssertionError(f"expected 5314 fit rows, got {len(units)}")
        seen_keys: set[str] = set()
        grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
        vocabulary = Counter()
        sentence_lengths: list[float] = []
        target_counts: list[float] = []
        for unit in units:
            if unit.get("split") != "fit" or unit.get("dataset") not in DATASETS:
                raise AssertionError(f"non-fit/non-V1V2 row encountered: {unit.get('unit_key')}")
            video = str(unit["video"])
            if video not in FIT_VIDEOS:
                raise AssertionError(f"fit video drift: {video}")
            key = str(unit["unit_key"])
            if key in seen_keys:
                raise AssertionError(f"duplicate fit unit key: {key}")
            seen_keys.add(key)
            sentence = str(unit.get("sentence") or unit.get("expression") or "")
            if not sentence:
                raise AssertionError(f"empty expression: {key}")
            words = token_words(sentence)
            vocabulary.update(words)
            sentence_lengths.append(float(len(words)))
            targets = unit.get("target_ids", [])
            if targets is None:
                targets = []
            if not isinstance(targets, list):
                raise AssertionError(f"target_ids is not a list: {key}")
            target_counts.append(float(len(targets)))
            grouped[(str(unit["dataset"]), video, int(unit["frame_id"]))].append(unit)
        if len(seen_keys) != len(units):
            raise AssertionError("fit key cardinality drift")
        if {str(x["video"]) for x in units} != FIT_VIDEOS:
            raise AssertionError("fit video set drift")

        matrix_rows: list[dict[str, Any]] = []
        video_summaries: dict[str, dict[str, Any]] = {}
        all_pair_stats = Counter()
        pair_categories = Counter()
        label_positive_counts: list[float] = []
        candidate_counts: list[float] = []
        duplicate_candidate_counts: list[float] = []
        duplicate_track_counts: list[float] = []
        group_flip_counts: list[float] = []
        eligible_keys: list[str] = []
        label_categories = Counter()
        label_domain_category = Counter()
        fit_unit_label_mismatches: list[dict[str, Any]] = []
        frame_group_count = Counter()
        target_bag_pair_records: list[dict[str, Any]] = []
        exact_pair_records: list[dict[str, Any]] = []
        frame_groups_by_video = Counter()

        # Each video is loaded and released serially.  The sidecar is read only
        # after the native frame range and immutable row offsets are known.
        for video in sorted(FIT_VIDEOS):
            path, tensors, candidate_gt, frame_to_range = load_bank(video)
            groups_for_video = sorted(
                (key, values) for key, values in grouped.items() if key[1] == video
            )
            per_video = {
                "video": video, "bank_path": str(path), "groups": 0, "units": 0,
                "candidate_rows": 0, "positive_rows": 0, "eligible_groups": 0,
                "candidate_query_flip_triplets": 0, "target_bag_query_flips": 0,
                "categories": Counter(), "dataset_counts": Counter(),
            }
            for (dataset, _video, frame_id), raw_group in groups_for_video:
                if frame_id not in frame_to_range:
                    raise KeyError(f"frame {frame_id} not in L69 {video}")
                begin, end = frame_to_range[frame_id]
                offsets = list(range(begin, end))
                if len(offsets) != end - begin:
                    raise AssertionError(f"native offset range drift: {video}:{frame_id}")
                candidate_indices = [int(x) for x in tensors["candidate_index"][offsets].tolist()]
                track_ids = [int(x) for x in tensors["track_id"][offsets].tolist()]
                pool_ids = [int(x) for x in tensors["pool_id"][offsets].tolist()]
                raw_ranks = [int(x) for x in tensors["raw_rank"][offsets].tolist()]
                boxes = tensors["box"][offsets].float().tolist()
                objectness = [float(x) for x in tensors["objectness"][offsets].float().tolist()]
                duplicate_candidate_count = len(candidate_indices) - len(set(candidate_indices))
                duplicate_track_count = len(track_ids) - len(set(track_ids))
                candidate_counts.append(float(len(offsets)))
                duplicate_candidate_counts.append(float(duplicate_candidate_count))
                duplicate_track_counts.append(float(duplicate_track_count))
                frame_key = f"{dataset}|{video}|{int(frame_id)}"
                queries: list[dict[str, Any]] = []
                for unit in sorted(raw_group, key=lambda x: (int(x["query_id"]), str(x["unit_key"]))):
                    targets = {str(x) for x in (unit.get("target_ids") or [])}
                    labels = [value is not None and str(value) in targets for value in (candidate_gt[o] for o in offsets)]
                    category = category_for(targets, labels)
                    declared = str(unit.get("category", "unknown"))
                    if declared != category:
                        fit_unit_label_mismatches.append({"unit_key": str(unit["unit_key"]), "declared": declared, "recomputed": category})
                    positive_indices = [i for i, value in enumerate(labels) if value]
                    label_categories[category] += 1
                    label_domain_category[(dataset, category)] += 1
                    label_positive_counts.append(float(len(positive_indices)))
                    per_video["units"] += 1
                    per_video["positive_rows"] += len(positive_indices)
                    per_video["categories"][category] += 1
                    per_video["dataset_counts"][dataset] += 1
                    queries.append({
                        "unit_key": str(unit["unit_key"]), "dataset": dataset,
                        "video": video, "query_id": int(unit["query_id"]),
                        "frame_id": int(frame_id), "sentence": str(unit.get("sentence") or unit.get("expression") or ""),
                        "target_ids": sorted(targets), "declared_category": declared,
                        "category": category, "target_present": bool(targets),
                        "candidate_present": bool(positive_indices),
                        "present_uncovered": bool(targets) and not bool(positive_indices),
                        "positive_count": len(positive_indices),
                        "positive_indices": positive_indices,
                        "label_vector": [int(value) for value in labels],
                    })
                queries.sort(key=lambda x: (int(x["query_id"]), str(x["unit_key"])))
                q_count = len(queries)
                same_frame_pairs = 0
                frame_flip_triplets = 0
                frame_bag_flips = 0
                for left in range(q_count):
                    for right in range(left + 1, q_count):
                        q_left, q_right = queries[left], queries[right]
                        y_left = np.asarray(q_left["label_vector"], dtype=np.int8)
                        y_right = np.asarray(q_right["label_vector"], dtype=np.int8)
                        if y_left.shape != y_right.shape:
                            raise AssertionError(f"matrix candidate axis drift: {frame_key}")
                        xor_count = int(np.count_nonzero(y_left != y_right))
                        bag_left, bag_right = set(q_left["target_ids"]), set(q_right["target_ids"])
                        intersection = bag_left & bag_right
                        if bag_left == bag_right:
                            relation = "exact_same_target_bag"
                        elif intersection:
                            relation = "partial_target_bag_overlap"
                        else:
                            relation = "disjoint_target_bag"
                        pair_categories[relation] += 1
                        all_pair_stats["same_frame_query_pairs"] += 1
                        if xor_count:
                            frame_flip_triplets += xor_count
                            all_pair_stats["candidate_query_flip_triplets"] += xor_count
                            exact_pair_records.append({
                                "group_key": frame_key,
                                "query_a": q_left["unit_key"], "query_b": q_right["unit_key"],
                                "relation": relation, "changed_candidate_count": xor_count,
                            })
                        if bag_left != bag_right and xor_count:
                            frame_bag_flips += 1
                            all_pair_stats["target_bag_query_flips"] += 1
                            target_bag_pair_records.append({
                                "group_key": frame_key,
                                "query_a": q_left["unit_key"], "query_b": q_right["unit_key"],
                                "relation": relation, "changed_candidate_count": xor_count,
                            })
                        same_frame_pairs += 1
                eligible = q_count >= 2 and frame_flip_triplets > 0
                if eligible:
                    eligible_keys.append(frame_key)
                    per_video["eligible_groups"] += 1
                per_video["groups"] += 1
                per_video["candidate_rows"] += len(offsets)
                frame_groups_by_video[video] += 1
                frame_group_count[(dataset, "all")] += 1
                frame_group_count[(dataset, "eligible")] += int(eligible)
                group_flip_counts.append(float(frame_flip_triplets))
                matrix_rows.append({
                    "format": "locatemot-l82-frame-query-matrix-row-v1",
                    "group_key": frame_key, "dataset": dataset, "video": video,
                    "frame_id": int(frame_id), "bank_path": str(path),
                    "begin": int(begin), "end": int(end), "candidate_count": len(offsets),
                    "row_offsets": offsets,
                    "row_keys": [[dataset, video, int(q["query_id"]), int(frame_id), str(path), int(offset)]
                                 for q in queries for offset in offsets],
                    "candidate_index": candidate_indices, "track_id": track_ids,
                    "pool_id": pool_ids, "raw_rank": raw_ranks, "boxes": boxes,
                    "objectness": objectness,
                    "duplicate_candidate_index_count": int(duplicate_candidate_count),
                    "duplicate_track_id_count": int(duplicate_track_count),
                    "queries": queries, "query_count": q_count,
                    "same_frame_query_pairs": same_frame_pairs,
                    "exact_candidate_query_flip_triplets": frame_flip_triplets,
                    "target_bag_query_flips": frame_bag_flips,
                    "eligible_for_exact_cross_query_supervision": bool(eligible),
                    "labels_are_fit_only_posthoc": True,
                    "candidate_deletion": False, "candidate_truncation": False,
                })
            per_video["categories"] = dict(per_video["categories"])
            per_video["dataset_counts"] = dict(per_video["dataset_counts"])
            video_summaries[video] = per_video
            del tensors, candidate_gt
            gc.collect()

        eligible_set = set(eligible_keys)
        eligible_v1 = sum(1 for row in matrix_rows if row["group_key"] in eligible_set and row["dataset"] == "refer_kitti_v1")
        eligible_v2 = sum(1 for row in matrix_rows if row["group_key"] in eligible_set and row["dataset"] == "refer_kitti_v2")
        fit_videos_seen = sorted({str(row["video"]) for row in units})
        threshold_checks = {
            "eligible_frame_groups": len(eligible_keys) >= MATRIX_THRESHOLDS["eligible_frame_groups"],
            "eligible_v1_frame_groups": eligible_v1 >= MATRIX_THRESHOLDS["eligible_v1_frame_groups"],
            "eligible_v2_frame_groups": eligible_v2 >= MATRIX_THRESHOLDS["eligible_v2_frame_groups"],
            "exact_candidate_query_flip_triplets": all_pair_stats["candidate_query_flip_triplets"] >= MATRIX_THRESHOLDS["exact_candidate_query_flip_triplets"],
            "target_bag_query_flips": all_pair_stats["target_bag_query_flips"] >= MATRIX_THRESHOLDS["target_bag_query_flips"],
            "fit_videos": len(fit_videos_seen) >= MATRIX_THRESHOLDS["fit_videos"],
        }
        matrix_status = "route_a_exact_cross_query_supervision" if all(threshold_checks.values()) else "matrix_supervision_insufficient"
        dataset_group_counts = Counter(str(row["dataset"]) for row in matrix_rows)
        dataset_unit_counts = Counter(str(unit["dataset"]) for unit in units)
        categories_by_dataset = {
            dataset: {category: int(label_domain_category[(dataset, category)]) for category in CATEGORIES}
            for dataset in DATASETS
        }
        pair_stats = {
            "same_frame_query_pairs": int(all_pair_stats["same_frame_query_pairs"]),
            "candidate_query_flip_triplets": int(all_pair_stats["candidate_query_flip_triplets"]),
            "target_bag_query_flips": int(all_pair_stats["target_bag_query_flips"]),
            "pair_relation_counts": dict(pair_categories),
            "exact_flip_pair_records": len(exact_pair_records),
            "target_bag_flip_pair_records": len(target_bag_pair_records),
            "flip_triplet_definition": "one candidate row and one unordered pair of distinct same-frame exact expressions whose binary membership labels differ",
            "target_bag_flip_definition": "one unordered same-frame query pair with unequal target-ID bags and at least one differing candidate membership label",
        }
        source_inputs = {
            "train_units": file_meta(DATA_ROOT / "train_units.jsonl"),
            "manifest": file_meta(MANIFEST),
            "l69_contract": file_meta(L69_CONTRACT),
            "l69_feature_root": str(L69_ROOT),
            "l69_videos": sorted(FIT_VIDEOS),
        }
        matrix_audit = {
            "format": "locatemot-l82-expression-matrix-audit-v1",
            "status": "complete", "stage": "phase_a_fit_only_expression_matrix",
            "command": command, "cwd": str(ROOT),
            "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745",
            "inputs": source_inputs,
            "outputs": {
                "frame_query_groups": str(out / "frame_query_groups.jsonl"),
                "eligible_group_keys": str(out / "eligible_group_keys.json"),
                "label_flip_statistics": str(out / "label_flip_statistics.json"),
                "vocabulary_statistics": str(out / "vocabulary_statistics.json"),
            },
            "fit_unit_count": len(units), "frame_group_count": len(matrix_rows),
            "dataset_unit_counts": dict(dataset_unit_counts),
            "dataset_frame_group_counts": dict(dataset_group_counts),
            "fit_videos": fit_videos_seen,
            "category_counts": dict(label_categories),
            "categories_by_dataset": categories_by_dataset,
            "candidate_count_distribution": distribution(candidate_counts),
            "positive_rows_distribution": distribution(label_positive_counts),
            "duplicate_candidate_index_distribution": distribution(duplicate_candidate_counts),
            "duplicate_track_id_distribution": distribution(duplicate_track_counts),
            "flip_triplets_per_frame_group": distribution(group_flip_counts),
            "pair_statistics": pair_stats,
            "eligible_groups": {"total": len(eligible_keys), "v1": eligible_v1, "v2": eligible_v2},
            "thresholds": MATRIX_THRESHOLDS,
            "threshold_checks": threshold_checks,
            "route_decision": matrix_status,
            "exact_join": "route-A native L69 frame-pointer join; no sparse/pseudo-label fallback used",
            "declared_category_mismatches": fit_unit_label_mismatches,
            "label_access_boundary": "candidate rows and immutable offsets were reconstructed first; L49 target_ids and L69 candidate_gt were used only for this fit-only post-hoc matrix audit",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "training_run": False,
            "hota_trackeval_run": False, "candidate_deletion": False,
            "candidate_truncation": False, "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED", "failure_root_cause": None,
            "next_action": "run Phase B duplicate/loss pathology only if route decision is route_a_exact_cross_query_supervision",
            "elapsed_sec": time.perf_counter() - started,
        }
        write_jsonl(out / "frame_query_groups.jsonl", matrix_rows)
        write_json(out / "eligible_group_keys.json", {
            "format": "locatemot-l82-eligible-group-keys-v1", "status": "complete",
            "keys": eligible_keys, "count": len(eligible_keys), "threshold_checks": threshold_checks,
            "route_decision": matrix_status, "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "training_run": False, "hota_trackeval_run": False,
        })
        write_json(out / "label_flip_statistics.json", {
            "format": "locatemot-l82-label-flip-statistics-v1", "status": "complete",
            "scope": "fit-only exact expression labels", "pair_statistics": pair_stats,
            "video_summaries": video_summaries, "thresholds": MATRIX_THRESHOLDS,
            "threshold_checks": threshold_checks, "route_decision": matrix_status,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "training_run": False,
            "hota_trackeval_run": False,
        })
        write_json(out / "vocabulary_statistics.json", {
            "format": "locatemot-l82-vocabulary-statistics-v1", "status": "complete",
            "scope": "fit-only sentence diagnostics; not semantic inputs",
            "sentence_count": len(units), "unique_word_count": len(vocabulary),
            "top_words": [[word, int(count)] for word, count in vocabulary.most_common(100)],
            "sentence_word_length": distribution(sentence_lengths),
            "target_count_per_unit": distribution(target_counts),
            "heuristic_expression_strata": "not used; no POS/static/motion alignment inferred",
            "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "training_run": False,
            "hota_trackeval_run": False,
        })
        write_json(out / "per_video_summary.json", {
            "format": "locatemot-l82-matrix-per-video-v1", "status": "complete",
            "videos": video_summaries, "screening_gt_used": False,
            "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False,
            "training_run": False, "hota_trackeval_run": False,
        })
        write_json(out / "expression_matrix_audit.json", matrix_audit)
        write_json(out / "provenance.json", {
            "format": "locatemot-l82-expression-matrix-provenance-v1", "status": "complete",
            "command": command, "cwd": str(ROOT), "luna_thread": "01a02014-fce8-7f51-8414-e7ed6ab44745",
            "inputs": source_inputs, "outputs": [str(out / name) for name in (
                "expression_matrix_audit.json", "frame_query_groups.jsonl", "eligible_group_keys.json",
                "label_flip_statistics.json", "vocabulary_statistics.json", "per_video_summary.json")],
            "labels": {"fit_only": True, "validation_opened": False, "calibration_opened": False,
                       "screening_opened": False, "official_test_opened": False,
                       "posthoc_after_native_row_join": True},
            "route_decision": matrix_status, "threshold_checks": threshold_checks,
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "training_run": False, "hota_trackeval_run": False,
            "candidate_deletion": False, "candidate_truncation": False,
            "failure_root_cause": None,
            "next_action": matrix_audit["next_action"],
        })
        write_json(out / "status.json", {
            "format": "locatemot-l82-status-v1", "status": "complete", "stage": "phase_a",
            "command": command, "inputs": source_inputs, "outputs": [str(out / "expression_matrix_audit.json")],
            "failure_root_cause": None, "next_action": matrix_audit["next_action"],
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "training_run": False, "hota_trackeval_run": False,
        })
        print(json.dumps(matrix_audit, indent=2, ensure_ascii=False), flush=True)
        return 0
    except Exception:
        (out / "INCOMPLETE.md").write_text(
            "# L82 expression matrix — INCOMPLETE\n\n" + traceback.format_exc() +
            "\nNo model, training, validation, screening, official-test, TrackEval/HOTA, MOT or OVMOT action was run.\n")
        write_json(out / "status.json", {
            "format": "locatemot-l82-status-v1", "status": "incomplete", "stage": "phase_a",
            "command": command, "failure_root_cause": "first actionable exception preserved in INCOMPLETE.md",
            "next_action": "fix only the first actionable matrix-contract error and rerun in a new attempt",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "training_run": False, "hota_trackeval_run": False,
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
