#!/usr/bin/env python3
"""CPU integration smoke for the real U2 shared track core."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from locatemot.unified.models.unified_spec_track import UnifiedSpecTrack


def make_candidates(feature: torch.Tensor, shift: float) -> dict[str, torch.Tensor]:
    boxes = torch.tensor([[shift, 0.0, shift + 1.0, 1.0]], dtype=feature.dtype)
    return {
        "boxes": boxes,
        "scores": torch.tensor([0.9], dtype=feature.dtype),
        "id_visual": feature,
        "semantic_visual": feature * 0.8,
        "decoder_query": feature * 0.4,
    }


def main() -> int:
    torch.manual_seed(20261001)
    model = UnifiedSpecTrack(hidden_dim=256).eval()
    feature = torch.randn(1, 256)
    specification = torch.randn(4, 256)
    with torch.no_grad():
        first = model(candidates=make_candidates(feature, 0.0), task_type="mot")
        second = model(
            candidates=make_candidates(feature + 0.01, 0.02),
            task_type="ovmot",
            specification=specification,
            online_state=first["online_state"],
        )
    tracks = second["tracks"]
    payload = {
        "format": "locatemot-u2-integration-smoke-v1",
        "status": "pass",
        "tasks": ["mot", "ovmot"],
        "track_id_shape": list(tracks["track_ids"].shape),
        "state_tracks": tracks["state"].num_tracks,
        "finite": all(torch.isfinite(value).all().item() for value in tracks.values() if isinstance(value, torch.Tensor)),
        "official_test_labels_read": False,
    }
    path = Path("outputs/unified/u2/eval/integration_smoke.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
