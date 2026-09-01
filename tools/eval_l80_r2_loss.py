#!/usr/bin/env python3
"""Fixed semantic evaluation for L80-R2's loss-only checkpoint variant.

R2 has the exact R0 inference graph and frozen region interface; only the
fit-time loss changed.  The immutable L80 evaluator is therefore reused as
the scoring implementation, then the fresh output is explicitly annotated
with the R2 loss provenance.  No old output is opened for writing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
sys.path.insert(0, str(ROOT))
from tools import eval_l80_v12 as base  # noqa: E402


def annotate(out: Path) -> None:
    updates = {
        "semantic.json": ("format", "locatemot-l80-r2-semantic-evaluation-v1"),
        "gate_decision.json": ("format", "locatemot-l80-r2-semantic-gate-v1"),
        "provenance.json": ("format", "locatemot-l80-r2-evaluation-provenance-v1"),
        "config.json": ("format", "locatemot-l80-r2-eval-config-v1"),
        "status.json": ("format", "locatemot-l80-r2-status-v1"),
    }
    for filename, (field, value) in updates.items():
        path = out / filename
        payload = json.loads(path.read_text())
        payload[field] = value
        payload["stage"] = "R2 loss-only semantic evaluation"
        payload["r2_loss_variant"] = {
            "only_changed_factor": "train-only hard-negative/multi-positive loss",
            "hard_negative": "detached current membership logits with all-negative fallback",
            "positive_floor": True,
            "inference_region_interface": "unchanged L80-R0",
        }
        payload["screening_gt_used"] = False
        payload["official_test_labels_read"] = False
        payload["ordinary_mot_ovmot_touched"] = False
        payload["hota_trackeval_run"] = False
        path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    result = base.evaluate(args)
    annotate(out)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
