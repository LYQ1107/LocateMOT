#!/usr/bin/env python3
"""Freeze the preregistered MOT20 Top300 decision before any track training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.unified.build_u2_portability_artifact import read_measurement


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements-dir", type=Path, required=True)
    parser.add_argument("--foundation-sha256", required=True)
    parser.add_argument("--recipe", type=Path, default=ROOT / "outputs/unified/u2/protocol/perception_adaptation_recipe.json")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rule = json.loads(args.recipe.read_text())["mot20_crowded_candidate_rule"]
    if rule["default_k"] != 150 or rule["conditional_k"] != 300 or rule["top300_acceptance_min"] != .90:
        raise ValueError("Crowded candidate recipe differs from the preregistered 0.90 rule")
    measurement = read_measurement(args.measurements_dir, "mot20", args.foundation_sha256)
    if measurement is None:
        raise ValueError("MOT20 complete portability measurement is required")
    counts = measurement["counts"]
    top300 = counts["hits_top300"] / counts["gt_total"]
    if top300 < .90:
        raise ValueError(f"MOT20 Top300 recall {top300:.8f} is below the registered 0.90 gate")
    status = json.loads((ROOT / "outputs/unified/u2/final_status.json").read_text())
    if status["mot_only_baseline"] is not None or status["joint_mot_ovmot"] is not None:
        raise ValueError("Crowded candidate decision must precede learned track baselines")
    payload = {"format": "locatemot-u2-crowded-candidate-registration-v1", "dataset": "mot20",
               "foundation_sha256": args.foundation_sha256, "selected_top_k": 300, "minimum_recall": .90,
               "measured_recall": top300, "before_track_training": True,
               "measurement_sha256": digest(args.measurements_dir / "mot20.json"),
               "recipe_path": str(args.recipe.resolve()), "recipe_sha256": digest(args.recipe),
               "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    if args.out.exists() and json.loads(args.out.read_text()) != payload:
        raise ValueError("Existing crowded candidate registration is immutable")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
