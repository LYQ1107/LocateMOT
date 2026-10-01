#!/usr/bin/env python3
"""CPU shape smoke for the LocateMOT-U skeleton."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

# Direct execution places ``tools/unified`` on ``sys.path``.  Add the project
# root so this smoke command works without an editable package install.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locatemot.unified.models.unified_spec_track import UnifiedSpecTrack


def main() -> int:
    torch.manual_seed(20261001)
    model = UnifiedSpecTrack(hidden_dim=256).eval()
    frames = torch.randn(1, 2, 3, 64, 64)
    specification = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        outputs = model(frames=frames, task_type="rmot", specification=specification)
    scores = outputs["tracks"]["task_score"]
    payload = {
        "format": "locatemot-u-skeleton-smoke-v1",
        "status": "pass",
        "task": "rmot",
        "score_shape": list(scores.shape),
        "finite": bool(torch.isfinite(scores).all()),
        "official_test_labels_read": False,
    }
    path = Path("outputs/unified/skeleton_smoke.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
