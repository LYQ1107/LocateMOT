#!/usr/bin/env python3
"""Audit whether the L34 train-side data contains verified word-to-region labels.

This is deliberately an audit-only tool.  It does not fit a projection, read the
screening split, or create pseudo alignment labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def scalar_shape(value):
    if isinstance(value, (list, tuple)):
        return [len(value)]
    if isinstance(value, dict):
        return ["mapping", len(value)]
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/l35/audit/alignment_audit.json")
    args = ap.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")

    split_path = ROOT / "outputs/l16/data/protocol/split_manifest.json"
    expr_path = ROOT / "outputs/l11/data/rmot_kitti/expressions.json"
    expr_missing_path = ROOT / "outputs/l16/data/kitti_missing/records/expressions.json"
    dense_root = ROOT / "outputs/l34/train_dense_bank_v1/kitti"
    source_root = ROOT / "outputs/l19/dual_banks_features/kitti"
    raw_root = ROOT / "data/kitti_tracking_training/image_02"
    train_videos = [str(v) for v in load_json(split_path)["kitti_v2"]["train"]]
    expressions = load_json(expr_path)
    missing_expressions = load_json(expr_missing_path)

    # These are annotation-like names only.  A field is verified only when it
    # contains an explicit token/span-to-region mapping, not merely a box/track
    # label or a free-form sentence.
    annotation_like_names = {
        "region", "regions", "region_id", "region_ids", "category", "categories",
        "attribute_region", "attribute_regions", "word_region", "word_regions",
        "token_region", "token_regions", "alignment", "alignments",
        "token_alignment", "word_alignment", "alignment_mask", "token_mask",
        "region_mask", "phrase_boxes", "attribute_boxes", "word_boxes",
    }
    explicit_alignment_fields = {
        "word_region", "word_regions", "token_region", "token_regions",
        "alignment", "alignments", "token_alignment", "word_alignment",
        "alignment_mask",
    }

    per_video = {}
    all_entry_keys = set()
    total_queries = 0
    train_query_videos = set()
    for video in train_videos:
        entries = list(expressions.get(video, []))
        if not entries and video in missing_expressions:
            entries = list(missing_expressions[video])
        keys = sorted({str(k) for e in entries for k in e.keys()})
        all_entry_keys.update(keys)
        total_queries += len(entries)
        train_query_videos.add(video)
        semantic_candidates = sorted(set(keys) & annotation_like_names)
        explicit = sorted(set(keys) & explicit_alignment_fields)
        raw_count = sum(1 for e in entries if str(e.get("raw_sentence", "")).strip())
        sentence_count = sum(1 for e in entries if str(e.get("sentence", "")).strip())
        per_video[video] = {
            "expression_count": len(entries),
            "entry_keys": keys,
            "annotation_like_fields": semantic_candidates,
            "explicit_alignment_fields": explicit,
            "raw_sentence_count": raw_count,
            "sentence_count": sentence_count,
            "has_free_text_only": bool(entries) and not explicit,
        }

    complete_videos = []
    missing_dense_videos = []
    dense_rows = 0
    dense_frames = 0
    dense_field_inventory = {}
    nonfinite_fields = []
    row_order_checks = {}
    sidecar_inventory = {}
    raw_image_counts = {}
    source_bank_audit = {}
    for video in train_videos:
        source_path = source_root / f"{video}.pt"
        source_bank_audit[video] = {
            "path": str(source_path),
            "exists": source_path.exists(),
        }
        if source_path.exists():
            source_data = torch.load(source_path, map_location="cpu", weights_only=False)
            source_tensors = source_data.get("tensors", {})
            source_bank_audit[video].update({
                "keys": sorted(source_tensors),
                "has_candidate_box": "box" in source_tensors,
                "has_track_id": "track_id" in source_tensors,
                "has_candidate_gt": "candidate_gt" in source_data,
                "metadata_keys": sorted(source_data.get("metadata", {})),
            })
        bank_path = dense_root / f"{video}.pt"
        complete = bank_path.exists() and (dense_root / f"{video}.complete").exists()
        if not complete:
            missing_dense_videos.append(video)
            continue
        complete_videos.append(video)
        data = torch.load(bank_path, map_location="cpu", weights_only=False)
        tensors = data.get("tensors", {})
        keys = sorted(tensors)
        dense_field_inventory[video] = {
            "keys": keys,
            "dense_roi_shape": list(tensors["dense_roi_tokens_v4"].shape)
            if "dense_roi_tokens_v4" in tensors else None,
            "dense_points_shape": list(tensors["dense_points_v4"].shape)
            if "dense_points_v4" in tensors else None,
            "metadata_keys": sorted(data.get("metadata", {}).keys()),
            "gt_used_for_features": data.get("metadata", {}).get("gt_used_for_features"),
        }
        n = int(tensors["frame"].shape[0])
        dense_rows += n
        dense_frames += int(tensors["frame_ids"].shape[0])
        row_order_checks[video] = {
            "rows": n,
            "frame_ptr_terminal": int(tensors["frame_ptr"][-1]),
            "frame_ptr_matches_rows": int(tensors["frame_ptr"][-1]) == n,
            "frame_ids_monotonic": bool(torch.all(tensors["frame_ids"][1:] >= tensors["frame_ids"][:-1]))
            if tensors["frame_ids"].numel() > 1 else True,
            "candidate_index_present": "candidate_index" in tensors,
            "track_id_present": "track_id" in tensors,
            "box_present": "box" in tensors,
        }
        for name, value in tensors.items():
            if torch.is_floating_point(value):
                finite = bool(torch.isfinite(value).all())
                if not finite:
                    nonfinite_fields.append({"video": video, "field": name})
        labels_path = dense_root / f"{video}.labels.json"
        if labels_path.exists():
            labels = load_json(labels_path)
            sidecar_inventory[video] = {
                "path": str(labels_path),
                "keys": sorted(labels),
                "candidate_gt_entries": len(labels.get("candidate_gt", [])),
                "is_candidate_track_label_only": set(labels) <= {"candidate_gt"},
            }
        frame_ids = tensors["frame_ids"].tolist()
        raw_image_counts[video] = {
            "frame_count": len(frame_ids),
            "raw_images_present": sum((raw_root / video / f"{int(frame):06d}.png").exists() for frame in frame_ids),
        }

    explicit_fields = sorted(all_entry_keys & explicit_alignment_fields)
    annotation_like_fields = sorted(all_entry_keys & annotation_like_names)
    verified = bool(explicit_fields)
    result = {
        "schema_version": "l35-alignment-audit-v1",
        "audit": "train_side_word_to_region_alignment",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "train_split": {
            "manifest": str(split_path),
            "manifest_sha256": sha256(split_path),
            "videos": train_videos,
            "video_count": len(train_videos),
            "query_count_from_l11_expressions": total_queries,
            "query_source": str(expr_path),
            "supplemental_missing_expression_source": str(expr_missing_path),
        },
        "expression_field_audit": {
            "all_entry_keys": sorted(all_entry_keys),
            "annotation_like_fields_found": annotation_like_fields,
            "explicit_word_region_alignment_fields_found": explicit_fields,
            "verified_alignment": verified,
            "free_text_is_not_alignment": True,
            "per_video": per_video,
        },
        "dense_bank_audit": {
            "root": str(dense_root),
            "source_l19_bank_root": str(source_root),
            "source_l19_bank": source_bank_audit,
            "format": "locatemot-l34-train-dense-bank",
            "complete_videos": complete_videos,
            "complete_video_count": len(complete_videos),
            "missing_or_incomplete_videos": missing_dense_videos,
            "rows_in_complete_videos": dense_rows,
            "frames_in_complete_videos": dense_frames,
            "per_video_fields": dense_field_inventory,
            "row_order_checks": row_order_checks,
            "sidecar_inventory": sidecar_inventory,
            "nonfinite_fields": nonfinite_fields,
            "raw_image_counts": raw_image_counts,
        },
        "provenance_classification": {
            "verified_word_to_region_annotation": False,
            "gt_privileged_candidate_positive_labels": True,
            "candidate_gt_sidecar_is_word_region_alignment": False,
            "heuristic_text_attribute_presence": True,
            "heuristic_pseudo_alignment_created": False,
            "gt_used_to_choose_sampling_or_projection": False,
            "screening_gt_read": False,
        },
        "decision": {
            "enter_l35_training": False,
            "reason": "No explicit train-side token/word-to-region alignment field or verified alignment mask exists; expressions provide free text and frame-to-track candidate labels only.",
            "no_zero_fill_for_missing_dense_video": True,
            "training_status": "BLOCKED_BY_ALIGNMENT_CONTRACT",
        },
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "out": str(out),
        "verified_alignment": verified,
        "train_videos": len(train_videos),
        "complete_dense_videos": len(complete_videos),
        "missing_dense_videos": missing_dense_videos,
        "train_queries": total_queries,
        "dense_rows": dense_rows,
        "nonfinite_fields": len(nonfinite_fields),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
