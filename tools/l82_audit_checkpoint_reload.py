#!/usr/bin/env python3
"""Read-only strict reload check for the three L82-A probe packages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from locatemot.models.l82_rank_probe import L82FactorizedRankProbe, L82RankProbeConfig


ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
RUN = ROOT / "outputs/l82/train/frozen_rank_probe_retry3"
OUT = RUN / "reload_audit.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong cwd: {Path.cwd()}")
    torch.manual_seed(20260829)
    rows = []
    for path in sorted((RUN / "checkpoints").glob("*_checkpoint_epoch10.pt")):
        package = torch.load(path, map_location="cpu")
        config = L82RankProbeConfig(**package["model_config"])
        model = L82FactorizedRankProbe(config).eval()
        result = model.load_state_dict(package["model_state_dict"], strict=True)
        value = torch.randn(2, 7, config.input_dim)
        with torch.inference_mode():
            output = model(value)
        rows.append({
            "path": str(path), "sha256": digest(path),
            "representation": package.get("representation"),
            "strict_missing_keys": list(result.missing_keys),
            "strict_unexpected_keys": list(result.unexpected_keys),
            "input_shape": list(value.shape),
            "output_shapes": {key: list(item.shape) for key, item in output.items()},
            "finite": all(bool(torch.isfinite(item).all()) for item in output.values()),
            "parameter_count": int(package["model_parameter_count"]),
        })
    payload = {
        "format": "locatemot-l82-rank-probe-reload-audit-v1",
        "status": "complete" if len(rows) == 3 and all(
            not row["strict_missing_keys"] and not row["strict_unexpected_keys"] and row["finite"]
            for row in rows
        ) else "invalid",
        "stage": "phase_d_frozen_representation_rank_probe",
        "seed": 20260829, "checkpoint_count": len(rows), "records": rows,
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        "candidate_deletion": False, "candidate_truncation": False,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "path": str(OUT)}, sort_keys=True))
    return 0 if payload["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
