#!/usr/bin/env python3
"""Read-only three-domain train/validation contract audit for L48."""
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
from locatemot.rmot.l48_data import (  # noqa: E402
    DOMAIN_ORDER, FAST, FEATURE_FIELDS, L29_CHECKPOINT, SPLIT, V5_TEXT,
    V5_TEXT_MANIFEST, frame_descriptor, load_bank, load_queries,
    query_summary, sha256_file, split_maps,
)

OUT = ROOT / "outputs/l48"
AUDIT = OUT / "audit"
DATA = OUT / "data"
RNG_SEED = 20260829
FIT_CAP = 240
VAL_CAP = 160


def reservoir_add(bucket, item, seen, cap, rng):
    seen += 1
    if len(bucket) < cap:
        bucket.append(item)
    else:
        index = rng.randrange(seen)
        if index < cap:
            bucket[index] = item
    return seen


def finite_field(value) -> bool:
    if not torch.is_tensor(value) or not torch.is_floating_point(value):
        return True
    return bool(torch.isfinite(value.float()).all())


def write_jsonl(path: Path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")
    started = time.time()
    AUDIT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RNG_SEED)
    maps = split_maps()
    all_queries = {domain: load_queries(domain) for domain in DOMAIN_ORDER}
    required_paths = [SPLIT, FAST, V5_TEXT, V5_TEXT_MANIFEST]
    missing_files = [str(p) for p in required_paths if not p.exists()]
    if not L29_CHECKPOINT.exists():
        missing_files.append(str(L29_CHECKPOINT))

    errors = []
    nonfinite = Counter()
    duplicate_keys = 0
    missing_bank = []
    bank_stats = {}
    domain_stats = {}
    query_summaries = []
    reservoirs = defaultdict(list)
    reservoir_seen = Counter()
    raw_samples = []

    for domain in DOMAIN_ORDER:
        dqueries = all_queries[domain]
        by_video = defaultdict(list)
        for query in dqueries:
            by_video[query["video"]].append(query)
            query_summaries.append(query_summary(query))
        dstat = {
            "query_count": len(dqueries),
            "fit_query_count": sum(q["split"] == "fit" for q in dqueries),
            "validation_query_count": sum(q["split"] == "validation" for q in dqueries),
            "fit_videos": maps[domain]["fit"],
            "validation_videos": maps[domain]["validation"],
            "official_eval_videos_metadata_only": maps[domain]["official_eval"],
            "rows": 0, "frames": 0, "units": 0, "positive_rows": 0,
            "positive_units": 0, "multi_positive_units": 0,
            "inactive_units": 0, "present_uncovered_units": 0,
            "target_ids": 0, "covered_target_ids": 0,
            "same_frame_hard_pairs": 0, "candidate_sizes": [],
            "positive_sizes": [], "text_word_counts": [],
        }
        for video in sorted(by_video):
            try:
                bank = load_bank(domain, video)
            except Exception as exc:
                missing_bank.append({"dataset": domain, "video": video,
                                     "error": f"{type(exc).__name__}: {exc}"})
                errors.append(f"{domain}/{video}: {type(exc).__name__}: {exc}")
                continue
            tensors = bank["tensors"]
            bstat = bank_stats.setdefault(domain, {})
            bstat[video] = {"path": str(bank["path"]),
                            "sha256": sha256_file(bank["path"]),
                            "label_path": str(bank["label_path"]),
                            "label_sha256": sha256_file(bank["label_path"]),
                            "rows": int(tensors["track_id"].numel()),
                            "frames": int(tensors["frame_ids"].numel()),
                            "image_size": list(bank["metadata"].get("image_size", [])),
                            "feature_shapes": {}}
            for field in FEATURE_FIELDS:
                if field not in tensors:
                    errors.append(f"{domain}/{video}: missing tensor {field}")
                    continue
                value = tensors[field]
                if hasattr(value, "shape"):
                    bstat[video]["feature_shapes"][field] = list(value.shape)
                if not finite_field(value):
                    nonfinite[f"{domain}/{video}/{field}"] += 1
            frames = tensors["frame_ids"].long().numpy()
            ptr = tensors["frame_ptr"].long().numpy()
            rows = tensors["frame"].long().numpy()
            if len(ptr) != len(frames) + 1 or ptr[0] != 0 or ptr[-1] != len(rows):
                errors.append(f"{domain}/{video}: invalid frame_ptr bounds")
            if len(frames) and not np.array_equal(rows[ptr[:-1]], frames):
                errors.append(f"{domain}/{video}: frame/frame_ptr mismatch")
            row_keys = set()
            for row in range(len(rows)):
                key = (int(rows[row]), int(tensors["candidate_index"][row]),
                       int(tensors["track_id"][row]))
                if key in row_keys:
                    duplicate_keys += 1
                row_keys.add(key)
            dstat["rows"] += len(rows)
            dstat["frames"] += len(frames)
            for query in by_video[video]:
                dstat["text_word_counts"].append(len(query["sentence"].split()))
                for fi in range(len(frames)):
                    desc = frame_descriptor(query, bank, fi)
                    pos = int(desc["positive_count"])
                    neg = int(desc["candidate_count"] - pos)
                    dstat["units"] += 1
                    dstat["candidate_sizes"].append(desc["candidate_count"])
                    dstat["positive_sizes"].append(pos)
                    dstat["positive_rows"] += pos
                    dstat["target_ids"] += len(desc["target_ids"])
                    dstat["covered_target_ids"] += min(pos, len(desc["target_ids"]))
                    dstat["same_frame_hard_pairs"] += pos * neg
                    category = desc["category"]
                    dstat[f"{category}_units"] += 1
                    cap = FIT_CAP if desc["split"] == "fit" else VAL_CAP
                    key = (domain, desc["split"], category)
                    reservoir_seen[key] = reservoir_add(
                        reservoirs[key], desc, reservoir_seen[key], cap, rng)
                    if len(raw_samples) < 12:
                        if domain == "refer_dance":
                            raw_image = ("/data1/LWR/vranlee/DATASETS/JDE/dancetrack/"
                                          "<train-or-val>/" + video + "/<frame>.jpg")
                        else:
                            raw_image = str(ROOT / "data/kitti_tracking_training/image_02" /
                                            video / f"{desc['frame_id']:06d}.png")
                        raw_samples.append({"dataset": domain, "video": video,
                                            "frame_id": desc["frame_id"],
                                            "box_field": "tensors.box",
                                            "bank": str(bank["path"]),
                                            "raw_image": raw_image})
            del bank
        domain_stats[domain] = dstat

    for domain, stat in domain_stats.items():
        for field in ("candidate_sizes", "positive_sizes", "text_word_counts"):
            values = stat[field]
            stat[field] = {
                "count": len(values),
                "min": int(min(values)) if values else None,
                "median": float(np.median(values)) if values else None,
                "q90": float(np.quantile(values, .9)) if values else None,
                "max": int(max(values)) if values else None,
            }
        stat["positive_frame_recall_coverage"] = (
            stat["covered_target_ids"] / max(1, stat["target_ids"])
        )
        stat["multi_positive_rate"] = stat["multi_positive_units"] / max(1, stat["units"])
        stat["inactive_rate"] = stat["inactive_units"] / max(1, stat["units"])

    query_summaries.sort(key=lambda x: (x["dataset"], x["query_id"]))
    write_jsonl(DATA / "query_manifest.jsonl", query_summaries)

    def selected_units(domain: str, split: str):
        values = []
        categories = ("multi_positive", "positive", "present_uncovered", "inactive")
        pools = {c: list(reservoirs[(domain, split, c)]) for c in categories}
        indices = {c: 0 for c in categories}
        target_count = 220 if split == "fit" else 100
        while len(values) < target_count and any(indices[c] < len(pools[c]) for c in categories):
            for category in categories:
                if indices[category] < len(pools[category]):
                    values.append(pools[category][indices[category]])
                    indices[category] += 1
                    if len(values) >= target_count:
                        break
        return values

    train_units, val_units, calibration_units = [], [], []
    for domain in DOMAIN_ORDER:
        fit_values = selected_units(domain, "fit")
        val_values = selected_units(domain, "validation")
        train_units.extend(fit_values)
        val_units.extend(val_values)
        calibration_units.extend(fit_values[:100])
    write_jsonl(DATA / "train_units.jsonl", train_units)
    write_jsonl(DATA / "val_units.jsonl", val_units)
    write_jsonl(DATA / "calibration_units.jsonl", calibration_units)

    contract = {
        "schema_version": "locatemot-l48-joint-data-contract-v1",
        "stage": "L48-A", "project_root": str(ROOT), "seed": RNG_SEED,
        "started_at_unix": started, "completed_at_unix": time.time(),
        "domains": domain_stats, "videos": bank_stats, "internal_split": maps,
        "sampled_units": {"train": len(train_units), "validation": len(val_units),
                          "calibration": len(calibration_units),
                          "categories": sorted({x["category"] for x in train_units + val_units})},
        "row_key": "(dataset,video,query,frame,candidate_index,track/fragment)",
        "candidate_sets_complete": len(errors) == 0,
        "duplicate_row_key_count": duplicate_keys, "missing_bank_count": len(missing_bank),
        "missing_banks": missing_bank, "nonfinite_fields": dict(nonfinite),
        "raw_image_provenance_samples": raw_samples,
        "text": {"word_level_expression_source": "expression sentence fields",
                  "frozen_v5_cache": str(V5_TEXT),
                  "frozen_v5_cache_sha256": sha256_file(V5_TEXT) if V5_TEXT.exists() else None,
                  "token_span_region_alignment": "UNALIGNED",
                  "static_motion_language_mask": "UNALIGNED/not claimed"},
        "teacher_control": {"l29_checkpoint": str(L29_CHECKPOINT),
                             "l29_checkpoint_sha256": sha256_file(L29_CHECKPOINT) if L29_CHECKPOINT.exists() else None,
                             "semantic_shortcut": False, "available_for_control": True},
        "fixed_fast_manifest": {"path": str(FAST), "sha256": sha256_file(FAST) if FAST.exists() else None,
                                 "query_count": 160, "calibration_queries": 64, "screening_queries": 96,
                                 "used_for_training": False, "used_for_structure_selection": False,
                                 "screening_gt_read": False},
        "official_eval_labels_read": False, "test_labels_visible": False,
        "screening_gt_used_for_training_or_selection": False,
        "semantic_inputs_excluded": ["source_id", "pool_id", "group_id", "state_key", "query_index_as_feature"],
        "errors": errors, "missing_required_files": missing_files,
        "decision": "enter_B0" if not errors and not missing_files and not nonfinite and duplicate_keys == 0 else "INCOMPLETE",
    }
    (AUDIT / "joint_data_contract.json").write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n")
    if contract["decision"] != "enter_B0":
        (AUDIT / "INCOMPLETE.md").write_text(
            "# L48 data contract incomplete\n\n"
            f"Errors: {json.dumps(errors, ensure_ascii=False)}\n"
        )
        raise RuntimeError("L48 data contract failed; see outputs/l48/audit/joint_data_contract.json")
    report = [
        "# L48 three-domain data contract audit", "",
        "The audit read only internal fit/validation videos. Official evaluation video names",
        "were retained as metadata; their labels were not loaded.", "",
        "| Domain | fit videos | val videos | queries | units | coverage | multi-positive | inactive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for domain in DOMAIN_ORDER:
        x = domain_stats[domain]
        report.append(
            f"| {domain} | {len(x['fit_videos'])} | {len(x['validation_videos'])} | "
            f"{x['query_count']} | {x['units']} | {x['positive_frame_recall_coverage']:.6f} | "
            f"{x['multi_positive_rate']:.4f} | {x['inactive_rate']:.4f} |"
        )
    report += [
        "", f"- Candidate rows checked: `{sum(x['rows'] for x in domain_stats.values())}`.",
        f"- Duplicate row keys: `{duplicate_keys}`; nonfinite fields: `{sum(nonfinite.values())}`.",
        f"- Sampled units: fit `{len(train_units)}`, validation `{len(val_units)}`.",
        "- Labels are expression-level frame→GT membership, including multi-positive and inactive units.",
        "- Token/span→region and static/motion masks remain `UNALIGNED`.",
        "- L29 is recorded as a frozen control only; source/pool/group/state are excluded from semantic inputs.",
        "- No screening/test labels were read for training, calibration, structure selection, or best-step selection.",
        "", "Machine-readable output: `outputs/l48/audit/joint_data_contract.json`.",
    ]
    (ROOT / "reports/l48_data_contract_audit.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"decision": contract["decision"],
                      "domain_query_counts": {d: domain_stats[d]["query_count"] for d in DOMAIN_ORDER},
                      "domain_units": {d: domain_stats[d]["units"] for d in DOMAIN_ORDER},
                      "train_units": len(train_units), "val_units": len(val_units),
                      "elapsed_sec": time.time() - started}, indent=2), flush=True)


if __name__ == "__main__":
    main()
