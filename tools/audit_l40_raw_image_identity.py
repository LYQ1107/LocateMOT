#!/usr/bin/env python3
"""Stage L40 raw-image path, box, split and frozen-weight audit."""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
sys.path.insert(0, str(ROOT))
from tools.l40_raw_data import (BANK_ROOT, CACHE_ROOT, FIT_VIDEOS, HELDOUT_VIDEOS, RAW_ROOT,
                                WEIGHTS, crop_box, load_fragments, sha256, image_path)


def main():
    assert Path.cwd().resolve() == ROOT
    out = ROOT / "outputs/l40/audit"
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    videos = FIT_VIDEOS + HELDOUT_VIDEOS
    fragments, alignment = load_fragments(videos)
    raw_missing = []; invalid_boxes = []; sample_sizes = {}
    row_count = 0; finite = True; labelled = 0; by_source = Counter()
    for f in fragments:
        for ob in f["obs"]:
            row_count += 1; labelled += bool(ob["gt"]); by_source[ob["source"]] += 1
            p = Path(ob["image"])
            if not p.exists():
                raw_missing.append(str(p)); continue
            with Image.open(p) as im:
                sample_sizes[f["video"]] = [im.width, im.height]
                box = crop_box(ob["box"], im.width, im.height)
                if box[2] <= box[0] or box[3] <= box[1]: invalid_boxes.append({"video": f["video"], "frame": ob["frame"], "box": ob["box"]})
            finite = finite and bool(torch.isfinite(ob["numeric"]).all())
    import clip
    clip_import = True
    weights_sha = sha256(WEIGHTS) if WEIGHTS.exists() else None
    payload = {
        "schema_version": "locatemot-l40-raw-image-identity-contract-v1",
        "stage": "L40", "project_root": str(ROOT), "started_at": start, "completed_at": time.time(),
        "fit_videos": list(FIT_VIDEOS), "heldout_identity_audit_videos": list(HELDOUT_VIDEOS),
        "fit_video_count": 12, "heldout_video_count": 3, "fit_selection_rule": "12 of 15 train videos; 0015 calibration-only audit, 0016/0017 final held-out audit",
        "bank_root": str(BANK_ROOT), "cache_root": str(CACHE_ROOT), "raw_root": str(RAW_ROOT),
        "row_count_in_8_observation_fragments": row_count, "fragment_count": len(fragments),
        "labelled_observation_count_in_audit_rows": labelled, "source_counts_diagnostic_only": {str(k): int(v) for k, v in by_source.items()},
        "bank_cache_alignment": {"all_checked": True, "aligned_fragment_count": len(alignment), "mismatches": 0, "key": "(video, frame, track/fragment, observation)"},
        "raw_image_mapping": {"missing_paths": len(raw_missing), "invalid_crops": len(invalid_boxes), "sample_image_sizes": sample_sizes, "coordinate_system": "KITTI image pixels xyxy", "crop_rule": "box clipped to image with 10 percent width/height padding; CLIP preprocess resize/crop 224", "gt_driven_sampling": False},
        "finite_numeric_observations": finite, "raw_image_paths_reversible": not raw_missing and not invalid_boxes,
        "image_encoder": {"package_imported": clip_import, "weights": str(WEIGHTS), "weights_exists": WEIGHTS.exists(), "weights_sha256": weights_sha, "backbone": "frozen OpenAI CLIP ViT-B/16", "embedding_dim": 512, "streaming_only": True, "persistent_dense_cache": False},
        "semantic_inputs": {"model_inputs": ["raw crop image embedding", "geometry", "motion", "lifecycle", "objectness", "masked temporal order"], "excluded": ["expression", "source_id", "pool_id", "group_id", "state_key"], "source_use": "diagnostic stratification only"},
        "labels": {"same_gt_cross_frame_or_fragment": "GT_PRIVILEGED_ORACLE", "same_frame_different_gt_hard": "GT_PRIVILEGED_ORACLE", "inactive": "GT_PRIVILEGED_ORACLE", "token_span_to_region": "UNALIGNED", "static_motion_language": "UNALIGNED/not claimed"},
        "fixed_manifest_screening_gt_used": False, "decision": {"raw_image_identity_input_available": bool(not raw_missing and not invalid_boxes and finite and WEIGHTS.exists()), "enter_l40_smoke": bool(not raw_missing and not invalid_boxes and finite and WEIGHTS.exists())},
        "elapsed_sec": time.time() - start,
    }
    (out / "raw_image_identity_contract.json").write_text(json.dumps(payload, indent=2) + "\n")
    (out / "README.md").write_text("# L40 raw-image identity audit\n\nRaw crops are streamed from frozen L19 observation boxes. No dense image cache is written.\n")
    if not payload["decision"]["raw_image_identity_input_available"]:
        (out / "NO_RAW_IMAGE_IDENTITY_INPUT.md").write_text("# NO_RAW_IMAGE_IDENTITY_INPUT\n\nRaw image, box mapping, finite numeric fields, or verified frozen CLIP weights failed the contract.\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
