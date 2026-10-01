#!/usr/bin/env python3
"""Compare MMDetection's default test pipeline with our shared wrapper."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locatemot.unified.data.legal_scope import assert_legal_path, assert_legal_video  # noqa: E402
from locatemot.unified.runtime.grounding_inference import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    load_grounding_runtime,
    prediction_arrays,
    run_grounding,
    runtime_metadata,
)
from mmdet.apis import inference_detector  # noqa: E402


def legal_images() -> list[Path]:
    # The universe is deterministic and development-only.  We inspect image
    # filenames, never any Refer-KITTI labels, to choose ten legal frames.
    videos = (
        ("refer_kitti_v1", "0004"), ("refer_kitti_v1", "0016"),
        ("refer_kitti_v1", "0018"), ("refer_kitti_v2", "0015"),
        ("refer_kitti_v2", "0016"), ("refer_kitti_v2", "0017"),
        ("refer_kitti_v2", "0020"),
    )
    paths: list[Path] = []
    for dataset, video in videos:
        video = assert_legal_video(video, dataset=dataset)
        root = PROJECT_ROOT / "data/kitti_tracking/training/image_02" / video
        frames = sorted(root.glob("*.png"))
        if not frames:
            raise FileNotFoundError(root)
        for index in (0, len(frames) // 2):
            path = assert_legal_path(frames[index])
            if path not in paths:
                paths.append(path)
    return paths[:10]


def main() -> int:
    prompt = "car . van . truck . bus . tram . pedestrian . person . cyclist . bicycle . motorcycle ."
    runtime = load_grounding_runtime(
        config_path=DEFAULT_CONFIG,
        checkpoint=DEFAULT_CHECKPOINT,
        training_contract=False,
        max_per_img=300,
    )
    rows = []
    for path in legal_images():
        official = inference_detector(
            runtime.model,
            str(path),
            text_prompt=prompt,
            custom_entities=True,
        )
        wrapped = run_grounding(runtime, path, prompt)
        official_boxes, official_scores = prediction_arrays(official)
        wrapped_boxes, wrapped_scores = prediction_arrays(wrapped)
        if official_boxes.shape != wrapped_boxes.shape:
            raise AssertionError(f"prediction count mismatch for {path}")
        box_diff = float((official_boxes - wrapped_boxes).abs().max().item()) if official_boxes.numel() else 0.0
        score_diff = float((official_scores - wrapped_scores).abs().max().item()) if official_scores.numel() else 0.0
        rows.append({
            "image": str(path.relative_to(PROJECT_ROOT)),
            "prediction_count": int(official_boxes.shape[0]),
            "max_box_abs_diff": box_diff,
            "max_score_abs_diff": score_diff,
            "finite": bool(official_boxes.isfinite().all() and official_scores.isfinite().all()),
        })
    max_box_diff = max(row["max_box_abs_diff"] for row in rows)
    max_score_diff = max(row["max_score_abs_diff"] for row in rows)
    payload = {
        "format": "locatemot-u-u1-preprocessing-equivalence-v1",
        "status": "PASS" if max_box_diff <= 1e-4 and max_score_diff <= 1e-5 else "FAIL",
        **runtime_metadata(runtime),
        "prompt": prompt,
        "image_count": len(rows),
        "max_box_abs_diff": max_box_diff,
        "max_score_abs_diff": max_score_diff,
        "box_tolerance": 1e-4,
        "score_tolerance": 1e-5,
        "rows": rows,
        "official_test_labels_read": False,
        "screening_gt_used": False,
        "training_launched": False,
    }
    out = PROJECT_ROOT / "outputs/unified/u1_preprocessing_equivalence.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
