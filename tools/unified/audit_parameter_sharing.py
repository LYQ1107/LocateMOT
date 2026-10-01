#!/usr/bin/env python3
"""Report shared versus task-specific parameter ownership for UnifiedSpecTrack."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Keep the standalone audit runnable from a clean checkout without requiring
# callers to set PYTHONPATH or install the package first.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locatemot.unified.models.unified_spec_track import UnifiedSpecTrack


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/unified/parameter_sharing_smoke.json"))
    args = parser.parse_args()
    model = UnifiedSpecTrack()
    groups = {"task_heads": 0, "shared": 0, "foundation": 0, "track_core": 0, "specification": 0}
    for name, value in model.named_parameters():
        count = int(value.numel())
        if name.startswith("heads."):
            groups["task_heads"] += count
        elif name.startswith("foundation."):
            groups["foundation"] += count
        elif name.startswith("track_decoder."):
            groups["track_core"] += count
        elif name.startswith("spec_encoder.") or name.startswith("router."):
            groups["specification"] += count
        else:
            groups["shared"] += count
    total = sum(groups.values())
    payload = {
        "format": "locatemot-u-parameter-sharing-smoke-v1",
        "status": "skeleton_only",
        "total_parameters": total,
        "groups": groups,
        "shared_non_head_fraction": 1.0 - groups["task_heads"] / max(1, total),
        "checkpoint": None,
        "official_test_labels_read": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
