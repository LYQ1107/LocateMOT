#!/usr/bin/env python3
"""Materialize the L36 source/provenance audit without downloading data.

The audit intentionally keeps expression-level object supervision separate from
token/span-to-region supervision.  No screening file is read and no annotation
record is synthesized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/l36/audit/annotation_manifest.json")
    args = ap.parse_args()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")

    l35_audit = ROOT / "outputs/l35/audit/alignment_audit.json"
    l35 = json.loads(l35_audit.read_text())
    local_sources = [
        {
            "name": "Refer-KITTI local expressions",
            "path": str((ROOT / "outputs/l11/data/rmot_kitti/expressions.json").resolve()),
            "sha256": sha256(ROOT / "outputs/l11/data/rmot_kitti/expressions.json"),
            "observed_fields": l35["expression_field_audit"]["all_entry_keys"],
            "classification": "UNALIGNED",
            "reason": "free text and frame-to-GT-track labels; no token/span-to-box field",
        },
        {
            "name": "L19 candidate bank",
            "path": str((ROOT / "outputs/l19/dual_banks_features/kitti").resolve()),
            "classification": "GT_PRIVILEGED_ORACLE",
            "reason": "candidate boxes/tracks and candidate labels, but no language-region link",
        },
        {
            "name": "L34 train dense bank",
            "path": str((ROOT / "outputs/l34/train_dense_bank_v1/kitti").resolve()),
            "videos_complete": l35["dense_bank_audit"]["complete_videos"],
            "videos_missing_or_incomplete": l35["dense_bank_audit"]["missing_or_incomplete_videos"],
            "classification": "GT_PRIVILEGED_ORACLE",
            "reason": "dense candidate ROI sampling plus candidate_gt sidecar; no word-region annotation",
        },
    ]
    public_sources = [
        {
            "name": "RMOT / Refer-KITTI official implementation",
            "url": "https://github.com/wudongming97/RMOT",
            "version": "master",
            "commit": "d4fedb35538e79a743ff78ff946abc6c84453cab",
            "download_paths": [
                "https://github.com/wudongming97/RMOT/releases/download/v1.0/expression.zip",
                "https://github.com/wudongming97/RMOT/releases/download/v1.0/labels_with_ids.zip",
            ],
            "license": "MIT code license; dataset/image terms must follow upstream sources",
            "coordinate_system": "Refer-KITTI expression object IDs resolve to KITTI label boxes; no token/span offsets",
            "alignment_granularity": "expression_to_object_box",
            "verified_token_span_to_region": False,
            "usable_for_current_kitti_train_alignment": False,
            "reason": "official README documents expression-to-object-ID/box association, not word/span alignment",
        },
        {
            "name": "TempRMOT / Refer-KITTI-V2 official repository",
            "url": "https://github.com/zyn213/TempRMOT",
            "version": "main",
            "commit": "6a65640d849fdee4a32bb055945ee34c3b0edeb1",
            "download_paths": ["repository datasets/README.md"],
            "license": "repository license/underlying dataset terms require separate review",
            "coordinate_system": "Refer-KITTI/KITTI frame and label coordinates; no token-span boxes documented",
            "alignment_granularity": "expression_to_object_box",
            "verified_token_span_to_region": False,
            "usable_for_current_kitti_train_alignment": False,
            "reason": "manual/LLM expression expansion increases language coverage but does not provide verified token-region links",
        },
        {
            "name": "KITTI official benchmark",
            "url": "https://www.cvlibs.net/datasets/kitti/",
            "version": "official website accessed 2026-08-29",
            "commit": None,
            "download_paths": ["official tracking/object/raw-data download pages"],
            "license": "Creative Commons Attribution-NonCommercial-ShareAlike 3.0",
            "coordinate_system": "original KITTI image/benchmark coordinate systems",
            "alignment_granularity": "boxes/tracks only",
            "verified_token_span_to_region": False,
            "usable_for_current_kitti_train_alignment": False,
            "reason": "official images and tracking/object labels contain no language annotation",
        },
        {
            "name": "RefCOCO REFER API",
            "url": "https://github.com/lichengunc/refer",
            "version": "master",
            "commit": "e3bbaa30d2ca41cf0e5c0d3819d7e4ed9fd38fff",
            "download_paths": ["repository README/data instructions"],
            "license": "Apache-2.0 repository code; image/annotation terms require COCO/RefCOCO review",
            "coordinate_system": "COCO image pixel coordinates; referred bbox is [x,y,w,h]",
            "alignment_granularity": "sentence_tokens_to_referred_object_box (not explicit per-token boxes)",
            "verified_token_span_to_region": False,
            "usable_for_current_kitti_train_alignment": False,
            "reason": "has sentence token lists and referred-object boxes, but images are COCO, not current KITTI frames, and no explicit word-span box mapping",
        },
        {
            "name": "DKGTrack official code",
            "url": "https://github.com/acyddl/DKGTrack",
            "version": "main",
            "commit": "197f354443bd1e7b490d204456a7654b7d1e4ccd",
            "download_paths": ["repository; Refer-KITTI via TempRMOT instructions"],
            "license": "research use only (as stated by repository)",
            "coordinate_system": "RMOT/KITTI pipeline; no new annotation source",
            "alignment_granularity": "method code only",
            "verified_token_span_to_region": False,
            "usable_for_current_kitti_train_alignment": False,
            "reason": "SSE/MPA implementation does not supply current train-side alignment records",
        },
        {
            "name": "FlexHook official code",
            "url": "https://github.com/buptLwz/FlexHook",
            "version": "main",
            "commit": "bd1acc38634b28525d54dc6e0fcb38335f0029f9",
            "download_paths": ["repository pretrained instructions; dataset instructions"],
            "license": "MIT code license",
            "coordinate_system": "RMOT/KITTI pipeline; no new annotation source",
            "alignment_granularity": "method code only",
            "verified_token_span_to_region": False,
            "usable_for_current_kitti_train_alignment": False,
            "reason": "requires model/data setup but does not provide verified alignment for these expressions",
        },
        {
            "name": "iKUN official code",
            "url": "https://github.com/dyhBUPT/iKUN",
            "version": "master",
            "commit": "4db56bfaec703590e0fdfd1684d9769467a67e05",
            "download_paths": ["repository prepared files instructions"],
            "license": "MIT code license",
            "coordinate_system": "RMOT/KITTI pipeline; no new annotation source",
            "alignment_granularity": "method code only",
            "verified_token_span_to_region": False,
            "usable_for_current_kitti_train_alignment": False,
            "reason": "prepared text features and tracker integration are not token-region annotations",
        },
    ]
    manifest = {
        "schema_version": "l36-phrase-grounding-annotation-manifest-v1",
        "audit": "auditable_raw_image_phrase_grounding_supervision",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "source_audit": {
            "l35_alignment_audit": str(l35_audit.resolve()),
            "l35_alignment_audit_sha256": sha256(l35_audit),
            "screening_gt_read": False,
            "downloads_performed": False,
            "network_search_only": True,
        },
        "record_schema": {
            "required_fields": [
                "query", "token_span_offsets", "video_id", "frame_id", "region_box",
                "source", "confidence", "annotator_or_rule_version", "image_sha256",
            ],
            "allowed_status": ["VERIFIED_ALIGNMENT", "GT_PRIVILEGED_ORACLE", "HEURISTIC", "UNALIGNED"],
            "records": [],
            "record_count": 0,
        },
        "local_sources": local_sources,
        "public_sources": public_sources,
        "coverage": {
            "current_refer_kitti_train_verified_token_span_to_region": 0.0,
            "current_refer_kitti_train_verified_records": 0,
            "current_refer_kitti_train_expressions": l35["train_split"]["query_count_from_l11_expressions"],
            "current_train_videos": l35["train_split"]["video_count"],
            "dense_bank_complete_videos": l35["dense_bank_audit"]["complete_video_count"],
            "dense_bank_missing_videos": l35["dense_bank_audit"]["missing_or_incomplete_videos"],
        },
        "decision": {
            "verified_coverage_positive": False,
            "enter_l36_adapter_training": False,
            "status": "BLOCKED_NO_VERIFIED_ALIGNMENT",
            "reason": "No legally/provenance-verifiable token/span-to-region records for current Refer-KITTI train frames; external RefCOCO is a different image domain and only expression-level referred-box supervision.",
        },
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "out": str(out),
        "verified_records": 0,
        "verified_coverage": 0.0,
        "screening_gt_read": False,
        "downloads_performed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
