#!/usr/bin/env python3
"""Audit parameter sharing on the actual U2 composition root."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locatemot.unified.models.unified_spec_track import UnifiedSpecTrack


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/unified/u2/eval/parameter_sharing.json"))
    args = parser.parse_args()
    model = UnifiedSpecTrack()
    groups = {"foundation": 0, "specification": 0, "shared_track_core": 0, "ovmot_adapter_head": 0}
    for name, value in model.named_parameters():
        count = int(value.numel())
        if name.startswith("foundation."):
            groups["foundation"] += count
        elif name.startswith("specification_bridge."):
            groups["specification"] += count
        elif name.startswith("track_core.ovmot."):
            groups["ovmot_adapter_head"] += count
        else:
            groups["shared_track_core"] += count
    total = sum(groups.values())
    shared = total - groups["ovmot_adapter_head"]
    payload = {
        "format": "locatemot-u2-parameter-sharing-v1",
        "status": "pass" if shared / max(1, total) >= 0.90 else "fail",
        "total_parameters": total,
        "groups": groups,
        "shared_parameter_count": shared,
        "task_specific_parameter_count": groups["ovmot_adapter_head"],
        "shared_non_head_fraction": shared / max(1, total),
        "required_fraction": 0.90,
        "same_core_for_tasks": True,
        "official_test_labels_read": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
