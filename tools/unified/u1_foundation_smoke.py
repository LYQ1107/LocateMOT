#!/usr/bin/env python3
"""Run one label-free foundation smoke through the official test pipeline."""
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


def main() -> int:
    video = assert_legal_video("0004", dataset="refer_kitti_v1")
    image_path = assert_legal_path(
        PROJECT_ROOT / "data/kitti_tracking/training/image_02" / video / "000063.png"
    )
    runtime = load_grounding_runtime(
        config_path=DEFAULT_CONFIG,
        checkpoint=DEFAULT_CHECKPOINT,
        training_contract=False,
    )
    result = run_grounding(runtime, image_path, "car . pedestrian . cyclist .")
    boxes, scores = prediction_arrays(result)
    payload = {
        "format": "locatemot-u-u1-foundation-smoke-v2",
        "status": "pass",
        "model": "MMDetection GroundingDINO Swin-B CogCoOR",
        **runtime_metadata(runtime),
        "video": video,
        "image": str(image_path.relative_to(PROJECT_ROOT)),
        "image_shape": list(result.metainfo["ori_shape"])
        if "ori_shape" in result.metainfo
        else None,
        "prediction_count": int(boxes.shape[0]),
        "finite_boxes": bool(boxes.isfinite().all()),
        "finite_scores": bool(scores.isfinite().all()),
        "official_test_labels_read": False,
        "screening_gt_used": False,
    }
    out = PROJECT_ROOT / "outputs/unified/u1_foundation_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
