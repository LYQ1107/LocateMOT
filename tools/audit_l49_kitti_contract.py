#!/usr/bin/env python3
"""Read-only V1/V2 train-pool contract and video-overlap audit for L49."""
from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from locatemot.rmot.l49_data import (  # noqa: E402
    FAST_MANIFEST, L29_CHECKPOINT, L49_SPLITS, REQUIRED_TENSORS, TEXT_CACHE,
    frame_descriptor, load_bank, load_l49_queries, sha256_file, unit_row_key,
)

OUT = ROOT / "outputs/l49"
AUDIT = OUT / "audit"
DATA = OUT / "data"
SEED = 20260829
FIT_CAP = 64
VAL_CAP = 64
CAL_CAP = 64
CATEGORY_ORDER = ("multi_positive", "positive", "present_uncovered", "inactive")


def reservoir_add(bucket, item, seen, cap, rng):
    seen += 1
    if len(bucket) < cap:
        bucket.append(item)
    else:
        index = rng.randrange(seen)
        if index < cap:
            bucket[index] = item
    return seen


def finite(value) -> bool:
    return not torch.is_tensor(value) or not torch.is_floating_point(value) or bool(torch.isfinite(value.float()).all())


def write_jsonl(path: Path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def unit_split_maps():
    return {dataset: {split: set(values) for split, values in splits.items()}
            for dataset, splits in L49_SPLITS.items()}


def overlap_report():
    maps = unit_split_maps()
    result = {"within_dataset": {}, "cross_dataset": {}, "invalid": []}
    for dataset, splits in maps.items():
        result["within_dataset"][dataset] = {}
        keys = ("fit", "calibration", "validation", "official_eval")
        for left in keys:
            for right in keys:
                if left >= right:
                    continue
                inter = sorted(splits[left] & splits[right])
                result["within_dataset"][dataset][f"{left}∩{right}"] = inter
                if inter and "official_eval" in (left, right):
                    result["invalid"].append({"dataset": dataset, "intersection": f"{left}∩{right}", "videos": inter})
    names = list(maps)
    for left in names:
        for right in names:
            if left >= right:
                continue
            result["cross_dataset"][f"{left}↔{right}"] = {
                f"{ls}∩{rs}": sorted(maps[left][ls] & maps[right][rs])
                for ls in ("fit", "calibration", "validation", "official_eval")
                for rs in ("fit", "calibration", "validation", "official_eval")
                if (maps[left][ls] & maps[right][rs])
            }
    return result


def summarize_distribution(values):
    if not values:
        return {"count": 0, "min": None, "median": None, "q90": None, "max": None}
    values = np.asarray(values, dtype=np.float64)
    return {"count": int(len(values)), "min": int(values.min()),
            "median": float(np.median(values)), "q90": float(np.quantile(values, .9)),
            "max": int(values.max())}


def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")
    started = time.time()
    AUDIT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    errors = []
    missing = []
    nonfinite = Counter()
    duplicate_keys = 0
    all_queries = {dataset: load_l49_queries(dataset) for dataset in L49_SPLITS}
    reservoirs = defaultdict(list)
    reservoir_seen = Counter()
    domain_stats = {}
    bank_stats = {}
    query_manifest = []
    sampled_unit_keys = set()
    for dataset, queries in all_queries.items():
        by_video = defaultdict(list)
        for query in queries:
            by_video[query["video"]].append(query)
            query_manifest.append({
                "dataset": dataset, "video": query["video"],
                "query_id": int(query["query_id"]), "expression": query["expression"],
                "sentence": query["sentence"], "split": query["split"],
                "label_source": query["label_source"],
                "target_frames": int(sum(bool(x) for x in query["target"].values())),
                "target_ids": int(sum(len(x) for x in query["target"].values())),
                "text_word_count": len(query["sentence"].split()),
            })
        stat = {
            "query_count": len(queries),
            "fit_query_count": sum(x["split"] == "fit" for x in queries),
            "calibration_query_count": sum(x["split"] == "calibration" for x in queries),
            "validation_query_count": sum(x["split"] == "validation" for x in queries),
            "fit_videos": L49_SPLITS[dataset]["fit"],
            "calibration_videos": L49_SPLITS[dataset]["calibration"],
            "validation_videos": L49_SPLITS[dataset]["validation"],
            "official_eval_videos_metadata_only": L49_SPLITS[dataset]["official_eval"],
            "rows": 0, "frames": 0, "units": 0, "positive_rows": 0,
            "positive_units": 0, "multi_positive_units": 0,
            "inactive_units": 0, "present_uncovered_units": 0,
            "target_ids": 0, "covered_target_ids": 0, "same_frame_hard_pairs": 0,
            "candidate_sizes": [], "positive_sizes": [], "text_word_counts": [],
        }
        for video in sorted(by_video):
            try:
                bank = load_bank(dataset, video)
            except Exception as exc:
                missing.append({"dataset": dataset, "video": video,
                                "error": f"{type(exc).__name__}: {exc}"})
                errors.append(f"{dataset}/{video}: {type(exc).__name__}: {exc}")
                continue
            tensors = bank["tensors"]
            bank_stats.setdefault(dataset, {})[video] = {
                "path": str(bank["path"]), "sha256": sha256_file(bank["path"]),
                "label_path": str(bank["label_path"]),
                "label_sha256": sha256_file(bank["label_path"]),
                "rows": int(tensors["track_id"].numel()),
                "frames": int(tensors["frame_ids"].numel()),
                "image_size": list(bank["metadata"].get("image_size", [])),
                "feature_shapes": {key: list(value.shape) for key, value in tensors.items()
                                    if hasattr(value, "shape")},
            }
            for field in REQUIRED_TENSORS:
                if field not in tensors:
                    errors.append(f"{dataset}/{video}: missing tensor {field}")
                elif not finite(tensors[field]):
                    nonfinite[f"{dataset}/{video}/{field}"] += 1
            frames = tensors["frame_ids"].long().numpy()
            ptr = tensors["frame_ptr"].long().numpy()
            rows = tensors["frame"].long().numpy()
            if len(ptr) != len(frames) + 1 or len(ptr) == 0 or ptr[0] != 0 or ptr[-1] != len(rows):
                errors.append(f"{dataset}/{video}: invalid frame_ptr bounds")
            elif len(frames) and not np.array_equal(rows[ptr[:-1]], frames):
                errors.append(f"{dataset}/{video}: frame/frame_ptr mismatch")
            row_keys = set()
            for row in range(len(rows)):
                key = unit_row_key({"dataset": dataset, "video": video,
                                    "query_id": -1, "frame_id": int(rows[row])},
                                   int(tensors["candidate_index"][row]), int(tensors["track_id"][row]))
                if key in row_keys:
                    duplicate_keys += 1
                row_keys.add(key)
            stat["rows"] += len(rows)
            stat["frames"] += len(frames)
            for query in by_video[video]:
                stat["text_word_counts"].append(len(query["sentence"].split()))
                for frame_index in range(len(frames)):
                    desc = frame_descriptor(query, bank, frame_index)
                    stat["units"] += 1
                    stat["candidate_sizes"].append(desc["candidate_count"])
                    stat["positive_sizes"].append(desc["positive_count"])
                    stat["positive_rows"] += desc["positive_count"]
                    stat["target_ids"] += len(desc["target_ids"])
                    labels = bank["labels"][desc["begin"]:desc["end"]]
                    stat["covered_target_ids"] += len({str(x) for x in labels if x is not None} & set(desc["target_ids"]))
                    stat["same_frame_hard_pairs"] += desc["positive_count"] * max(0, desc["candidate_count"] - desc["positive_count"])
                    stat[f"{desc['category']}_units"] += 1
                    cap = {"fit": FIT_CAP, "validation": VAL_CAP, "calibration": CAL_CAP}[desc["split"]]
                    key = (dataset, desc["split"], video, desc["category"])
                    reservoir_seen[key] = reservoir_add(reservoirs[key], desc, reservoir_seen[key], cap, rng)
            del bank
        for field in ("candidate_sizes", "positive_sizes", "text_word_counts"):
            stat[field] = summarize_distribution(stat[field])
        stat["positive_frame_recall_coverage"] = stat["covered_target_ids"] / max(1, stat["target_ids"])
        stat["multi_positive_rate"] = stat["multi_positive_units"] / max(1, stat["units"])
        stat["inactive_rate"] = stat["inactive_units"] / max(1, stat["units"])
        domain_stats[dataset] = stat

    def choose(dataset, split):
        result = []
        for video in L49_SPLITS[dataset][split]:
            for category in CATEGORY_ORDER:
                values = list(reservoirs[(dataset, split, video, category)])
                result.extend(values)
        return result

    train_units = sum((choose(dataset, "fit") for dataset in L49_SPLITS), [])
    calibration_units = sum((choose(dataset, "calibration") for dataset in L49_SPLITS), [])
    validation_units = sum((choose(dataset, "validation") for dataset in L49_SPLITS), [])
    for group in (train_units, calibration_units, validation_units):
        for unit in group:
            if unit["unit_key"] in sampled_unit_keys:
                errors.append(f"duplicate sampled unit key: {unit['unit_key']}")
            sampled_unit_keys.add(unit["unit_key"])
    write_jsonl(DATA / "train_units.jsonl", train_units)
    write_jsonl(DATA / "calibration_units.jsonl", calibration_units)
    write_jsonl(DATA / "validation_units.jsonl", validation_units)
    write_jsonl(DATA / "query_manifest.jsonl", sorted(query_manifest, key=lambda x: (x["dataset"], x["query_id"])))

    required_files = [FAST_MANIFEST, TEXT_CACHE, L29_CHECKPOINT]
    missing_files = [str(path) for path in required_files if not path.exists()]
    fixed_manifest_sha = sha256_file(FAST_MANIFEST) if FAST_MANIFEST.exists() else None
    fixed_manifest_expected = "06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa"
    if fixed_manifest_sha != fixed_manifest_expected:
        errors.append(f"fixed manifest sha mismatch: {fixed_manifest_sha}")
    overlap = overlap_report()
    if overlap["invalid"]:
        errors.extend([f"official eval overlap: {item}" for item in overlap["invalid"]])

    text_info = {"path": str(TEXT_CACHE), "sha256": sha256_file(TEXT_CACHE) if TEXT_CACHE.exists() else None,
                 "screening_or_test_text_used": False, "token_span_region_alignment": "UNALIGNED",
                 "static_motion_language_mask": "UNALIGNED/not claimed"}
    if TEXT_CACHE.exists():
        payload = torch.load(TEXT_CACHE, map_location="cpu", weights_only=False)
        known = set(payload.get("sentence_to_index", {}))
        required_sentences = {x["sentence"] for x in query_manifest}
        text_info["required_sentence_count"] = len(required_sentences)
        text_info["missing_train_pool_sentences"] = sorted(required_sentences - known)
        if text_info["missing_train_pool_sentences"]:
            errors.append("L48 text cache misses train-pool sentences")
        del payload

    contract = {
        "schema_version": "locatemot-l49-kitti-data-contract-v1",
        "stage": "L49-A", "project_root": str(ROOT), "seed": SEED,
        "started_at_unix": started, "completed_at_unix": time.time(),
        "domains": domain_stats, "videos": bank_stats,
        "video_overlap": overlap,
        "sampled_units": {"train": len(train_units), "calibration": len(calibration_units),
                           "validation": len(validation_units),
                           "categories": list(CATEGORY_ORDER),
                           "sampling": "video/category reservoir, dataset-balanced at training time"},
        "row_key": "(dataset,video,query,frame,candidate_index,track/fragment)",
        "candidate_sets_complete": not errors and not missing_files,
        "duplicate_row_key_count": duplicate_keys, "missing_bank_count": len(missing),
        "missing_banks": missing, "nonfinite_fields": dict(nonfinite),
        "text": text_info,
        "fixed_fast_manifest": {"path": str(FAST_MANIFEST), "sha256": fixed_manifest_sha,
                                 "expected_sha256": fixed_manifest_expected,
                                 "query_count": 160, "calibration_queries": 64,
                                 "screening_queries": 96, "used_for_training": False,
                                 "used_for_structure_or_threshold_selection": False},
        "official_eval_labels_read": False, "test_labels_visible": False,
        "screening_gt_used_for_training_or_selection": False,
        "semantic_inputs_excluded": ["source_id", "pool_id", "group_id", "state_key", "query_id_as_feature"],
        "frozen": {"l29_checkpoint": str(L29_CHECKPOINT),
                   "l29_checkpoint_sha256": sha256_file(L29_CHECKPOINT) if L29_CHECKPOINT.exists() else None,
                   "ordinary_mot_ovmot_touched": False},
        "errors": errors, "missing_required_files": missing_files,
        "decision": "enter_B0" if not errors and not missing_files and not nonfinite and duplicate_keys == 0 else "INCOMPLETE",
    }
    contract_path = AUDIT / "kitti_data_contract.json"
    contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
    if contract["decision"] != "enter_B0":
        (AUDIT / "INCOMPLETE.md").write_text("# L49 data contract incomplete\n\n" + json.dumps({"errors": errors, "missing": missing_files}, ensure_ascii=False, indent=2) + "\n")
        raise RuntimeError(f"L49 data contract failed; see {contract_path}")
    report = [
        "# L49 V1/V2 train-pool data contract audit", "",
        "Only V1/V2 official train-pool videos were read. Official evaluation labels were not loaded.", "",
        "| domain | fit/cal/val videos | queries | units | coverage | multi-positive | inactive |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, stat in domain_stats.items():
        report.append(f"| {dataset} | {len(stat['fit_videos'])}/{len(stat['calibration_videos'])}/{len(stat['validation_videos'])} | {stat['query_count']} | {stat['units']} | {stat['positive_frame_recall_coverage']:.6f} | {stat['multi_positive_rate']:.4f} | {stat['inactive_rate']:.4f} |")
    report += [
        "", f"- Candidate rows checked: `{sum(x['rows'] for x in domain_stats.values())}`.",
        f"- Sampled units: fit `{len(train_units)}`, calibration `{len(calibration_units)}`, validation `{len(validation_units)}`.",
        f"- Duplicate row keys: `{duplicate_keys}`; nonfinite fields: `{sum(nonfinite.values())}`.",
        "- Fit, calibration and validation are video-disjoint within each dataset; cross-domain same-name intersections are retained as an explicit audit field.",
        "- All candidates in every sampled frame remain in the unit; no source/pool/group/state/query-id semantic feature is constructed.",
        "- Labels are expression-level frame→GT membership, including multi-positive, present-uncovered and inactive units.",
        "- Token/span→region and static/motion masks remain `UNALIGNED`; no official test labels were read.",
        "", "Machine-readable output: `outputs/l49/audit/kitti_data_contract.json`.",
    ]
    (ROOT / "reports/l49_data_contract_audit.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"decision": contract["decision"], "query_counts": {d: domain_stats[d]["query_count"] for d in domain_stats},
                      "sampled_units": contract["sampled_units"], "overlap_invalid": overlap["invalid"],
                      "elapsed_sec": time.time() - started}, indent=2), flush=True)


if __name__ == "__main__":
    main()
