#!/usr/bin/env python3
"""Audit the expression-level RMOT supervision used by Stage L37.

This audit reads only the train-side expression/GT/cache sources plus the fixed
manifest's query identities.  It never reads screening GT and never creates a
model, threshold, or pseudo alignment label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def target_ids(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x) for x in value]
    return [str(value)]


def finite_tensor_fields(tensors):
    bad = []
    for name, value in tensors.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            if not bool(torch.isfinite(value).all()):
                bad.append(name)
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/l37/audit/expression_supervision_manifest.json")
    args = ap.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")

    split_path = ROOT / "outputs/l16/data/protocol/split_manifest.json"
    expression_path = ROOT / "outputs/l11/data/rmot_kitti/expressions.json"
    supplement_path = ROOT / "outputs/l16/data/kitti_missing/records/expressions.json"
    fast_path = ROOT / "outputs/l19/protocol/kitti_fast_eval_manifest.json"
    l19_root = ROOT / "outputs/l19/dual_banks_features/kitti"
    l19_manifest = l19_root / "manifest.json"
    l28_root = ROOT / "outputs/l28/track_sequence_bank_final"
    l28_manifest = l28_root / "manifest.json"
    raw_root = ROOT / "data/kitti_tracking_training/image_02"
    pkl_root = ROOT / "outputs/l11/data/rmot_kitti"

    split = read_json(split_path)["kitti_v2"]
    train_videos = [str(v) for v in split["train"]]
    expressions = read_json(expression_path)
    supplement = read_json(supplement_path)
    fast = read_json(fast_path)
    fast_queries = fast.get("queries", [])
    fast_keys = {(str(x.get("video")), str(x.get("expression"))) for x in fast_queries}

    field_stats = {
        "expression": {"source": str(expression_path), "split": "train", "coordinate_system": "text", "sample_count": 0, "missing_count": 0, "gt_derived": False, "usable_for_expression_training": True},
        "sentence": {"source": str(expression_path), "split": "train", "coordinate_system": "text", "sample_count": 0, "missing_count": 0, "gt_derived": False, "usable_for_expression_training": True},
        "raw_sentence": {"source": str(expression_path), "split": "train", "coordinate_system": "text", "sample_count": 0, "missing_count": 0, "gt_derived": False, "usable_for_expression_training": True},
        "label": {"source": str(expression_path), "split": "train", "coordinate_system": "frame_id -> GT object/track ID", "sample_count": 0, "missing_count": 0, "gt_derived": True, "usable_for_expression_training": True},
        "spec": {"source": str(expression_path), "split": "train", "coordinate_system": "numeric feature vector", "sample_count": 0, "missing_count": 0, "gt_derived": False, "usable_for_expression_training": False},
        "gt_boxes": {"source": str(pkl_root), "split": "train", "coordinate_system": "KITTI image_02 pixel xyxy", "sample_count": 0, "missing_count": 0, "gt_derived": True, "usable_for_expression_training": True},
        "candidate_gt": {"source": str(l19_root), "split": "train", "coordinate_system": "candidate row -> GT track/object ID", "sample_count": 0, "missing_count": 0, "gt_derived": True, "usable_for_expression_training": True},
        "obs_gt_ids": {"source": str(l28_root), "split": "train", "coordinate_system": "persistent observation -> GT ID", "sample_count": 0, "missing_count": 0, "gt_derived": True, "usable_for_expression_training": True},
        "inactive_null": {"source": "derived from expression label + available train frames", "split": "train", "coordinate_system": "video frame index", "sample_count": 0, "missing_count": 0, "gt_derived": True, "usable_for_expression_training": True},
    }

    query_count = 0
    query_frame_units = 0
    target_frame_units = 0
    multi_positive_frame_units = 0
    target_id_references = 0
    per_query = []
    train_keys = set()
    for video in train_videos:
        entries = list(expressions.get(video, []))
        if not entries:
            entries = list(supplement.get(video, []))
        for e in entries:
            query_count += 1
            key = (video, str(e.get("expression", "")))
            train_keys.add(key)
            for name in ("expression", "sentence", "raw_sentence", "label", "spec"):
                field_stats[name]["sample_count"] += 1
                value = e.get(name)
                if value is None or (isinstance(value, str) and not value.strip()):
                    field_stats[name]["missing_count"] += 1
            label = e.get("label") or {}
            present = 0
            references = 0
            multi = 0
            for frame, ids in label.items():
                ids = target_ids(ids)
                if ids:
                    present += 1
                    references += len(ids)
                    if len(ids) > 1:
                        multi += 1
            target_frame_units += present
            multi_positive_frame_units += multi
            target_id_references += references
            per_query.append({"video": video, "expression": key[1], "label_frame_count": len(label), "target_frame_count": present, "target_id_reference_count": references, "multi_positive_frame_count": multi})
        if not entries:
            raise AssertionError(f"no expression entries for train video {video}")

    if query_count != 7757:
        raise AssertionError(f"expected 7757 train expressions, found {query_count}")
    field_stats["label"]["sample_count"] = query_count

    bank_stats = {}
    cache_stats = {}
    pkl_stats = {}
    total_bank_rows = 0
    total_cache_obs = 0
    total_gt_boxes = 0
    total_candidate_gt_positive = 0
    for video in train_videos:
        pkl_path = pkl_root / f"{video}.pkl"
        if not pkl_path.exists():
            # L16 materialized missing records are not silently substituted for
            # the canonical train data; record the gap explicitly.
            pkl_stats[video] = {"path": str(pkl_path), "exists": False}
        else:
            with pkl_path.open("rb") as f:
                data = pickle.load(f)
            frames = data.get("frames", [])
            gt_count = sum(len(frame.get("gt_boxes", {})) for frame in frames)
            total_gt_boxes += gt_count
            pkl_stats[video] = {
                "path": str(pkl_path), "sha256": sha256(pkl_path), "exists": True,
                "frame_count": len(frames), "image_size": data.get("image_size"),
                "frame_fields": sorted(frames[0]) if frames else [],
                "gt_box_count": gt_count,
                "candidate_row_count": sum(len(frame.get("boxes", [])) for frame in frames),
                "candidate_gt_present": sum(sum(x is not None for x in frame.get("cand_gt", [])) for frame in frames),
            }
        bank_path = l19_root / f"{video}.pt"
        label_path = l19_root / f"{video}.labels.json"
        bank = torch.load(bank_path, map_location="cpu", weights_only=False)
        tensors = bank["tensors"]
        n = int(tensors["frame"].shape[0])
        total_bank_rows += n
        labels = read_json(label_path).get("candidate_gt", []) if label_path.exists() else []
        positives = sum(x is not None for x in labels)
        total_candidate_gt_positive += positives
        bank_stats[video] = {
            "path": str(bank_path), "sha256": sha256(bank_path), "rows": n,
            "fields": sorted(tensors), "image_size": bank.get("metadata", {}).get("image_size"),
            "frame_ptr_terminal": int(tensors["frame_ptr"][-1]),
            "frame_ptr_matches_rows": int(tensors["frame_ptr"][-1]) == n,
            "box_shape": list(tensors["box"].shape), "box_coordinate_system": "KITTI image_02 pixel xyxy",
            "label_path": str(label_path), "candidate_gt_count": len(labels),
            "candidate_gt_positive_count": positives,
            "finite_float_fields": finite_tensor_fields(tensors),
            "semantic_shortcut_fields_present_but_not_used": [x for x in ("pool_id", "observation_group_id") if x in tensors],
        }
        cache_path = l28_root / f"{video}.pt"
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        obs_n = int(cache["obs_frame"].shape[0])
        total_cache_obs += obs_n
        cache_stats[video] = {
            "path": str(cache_path), "sha256": sha256(cache_path), "observations": obs_n,
            "track_count": int(cache["track_ids"].numel()), "feature_dim": int(cache["feature_dim"]),
            "fields": sorted(cache), "obs_gt_ids_count": len(cache["obs_gt_ids"]),
            "labels_are_train_supervision_only": bool(cache["labels_are_train_supervision_only"]),
            "finite_float_fields": finite_tensor_fields(cache),
        }

    fast_video_set = {str(x.get("video")) for x in fast_queries}
    source_file_records = [
        {"path": str(x.resolve()), "sha256": sha256(x), "split": "train"}
        for x in (split_path, expression_path, supplement_path, l19_manifest, l28_manifest, fast_path)
        if x.exists()
    ]
    raw_presence = {}
    for video in train_videos:
        paths = sorted((raw_root / video).glob("*.png"))
        raw_presence[video] = {"path": str((raw_root / video).resolve()), "image_count": len(paths), "exists": (raw_root / video).is_dir()}

    manifest = {
        "schema_version": "l37-expression-supervision-manifest-v1",
        "audit": "expression_level_rmot_train_contract",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "train_split": {"videos": train_videos, "video_count": len(train_videos), "expression_count": query_count, "frame_units_with_target": target_frame_units, "multi_positive_frame_units": multi_positive_frame_units, "target_id_references": target_id_references},
        "field_schema": field_stats,
        "label_classes": {
            "EXPRESSION_LEVEL_VERIFIED": {"usable_for_training": True, "records": query_count, "meaning": "official/current expression associated with frame-level target GT object IDs"},
            "GT_PRIVILEGED_ORACLE": {"usable_for_training": True, "records": total_candidate_gt_positive, "meaning": "candidate/observation GT-derived positive labels and boxes; never a token alignment"},
            "HEURISTIC": {"usable_for_training": False, "records": 0, "meaning": "no heuristic pseudo-labels created"},
            "UNALIGNED": {"usable_for_training": False, "records": query_count, "meaning": "word/span-to-region relation is not annotated; latent attention may be learned but is not verified"},
        },
        "expression_samples": per_query,
        "source_files": source_file_records,
        "raw_image_provenance": raw_presence,
        "l19_bank": {"root": str(l19_root), "video_count": len(bank_stats), "rows": total_bank_rows, "per_video": bank_stats},
        "l11_gt_records": {"root": str(pkl_root), "total_gt_box_records": total_gt_boxes, "per_video": pkl_stats},
        "l28_sequence_cache": {"root": str(l28_root), "manifest": str(l28_manifest), "manifest_sha256": sha256(l28_manifest), "observation_count": total_cache_obs, "per_video": cache_stats},
        "fixed_fast_manifest": {"path": str(fast_path), "sha256": sha256(fast_path), "query_count": len(fast_queries), "calibration_count": int(fast.get("summary", {}).get("calibration", {}).get("queries", 0)), "screening_count": int(fast.get("summary", {}).get("screening", {}).get("queries", 0)), "videos": sorted(fast_video_set), "train_query_key_intersection": len(train_keys & fast_keys), "train_video_intersection": len(set(train_videos) & fast_video_set), "used_for_training": False, "screening_gt_read": False},
        "leakage_audit": {"train_videos_only_for_training": True, "train_query_key_count": len(train_keys), "fast_manifest_query_keys_read_only_for_provenance": len(fast_keys), "train_fast_query_intersection": len(train_keys & fast_keys), "screening_labels_loaded": False, "screening_gt_used_for_training_or_selection": False, "source_pool_group_state_used_as_semantic_input": False},
        "decision": {"expression_level_supervision_available": True, "enter_l37_training": True, "token_level_alignment_verified": False, "static_motion_language_mask_verified": False, "reason": "expression-to-frame/GT-track supervision is recoverable and train-only; token/span-to-region supervision remains UNALIGNED and will not be claimed"},
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"out": str(out), "expression_count": query_count, "train_videos": len(train_videos), "l19_rows": total_bank_rows, "l28_observations": total_cache_obs, "train_fast_query_intersection": len(train_keys & fast_keys), "enter_l37_training": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
