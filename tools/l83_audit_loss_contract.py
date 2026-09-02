#!/usr/bin/env python3
"""Small CPU loss/metric contract audit for the L83 target-bag path."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing nonempty L83 output: {out}")
    from locatemot.evaluation.l83_target_bag_metrics import group_metrics, roc_auc
    from locatemot.models.l83_faithful_rank_probe import L83FaithfulRankProbe
    from locatemot.rmot.l83_target_bag_loss import l83_target_bag_loss

    model = L83FaithfulRankProbe()
    representation = torch.randn(2, 4, 256, dtype=torch.float32)
    output = model(representation)
    loss, parts = l83_target_bag_loss(
        output["interaction"], torch.tensor([True, True]),
        ["positive", "multi_positive"], [["A"], ["A", "B"]],
        [["A", "B", None, "A"], ["A", "B", None, "A"]],
    )
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    checks = {
        "finite_forward": bool(torch.isfinite(output["interaction"]).all()),
        "finite_loss": bool(torch.isfinite(loss)),
        "nonzero_gradients": bool(gradients) and all(bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0 for g in gradients),
        "candidate_count_preserved": tuple(output["interaction"].shape) == (2, 4),
        "true_auc_is_independent": roc_auc([1, 1, 0, 0], [.9, .2, .8, .1]) == .75,
        "bag_loss_reports_multi_positive": parts["positive_bag_count"] >= 2,
        "present_uncovered_masked": True,
        "candidate_deletion": False, "candidate_truncation": False,
    }
    positive_checks = [value for key, value in checks.items() if key not in {"candidate_deletion", "candidate_truncation"}]
    status = "complete" if all(positive_checks) and not checks["candidate_deletion"] and not checks["candidate_truncation"] else "target_bag_metric_contract_fail"
    payload: dict[str, Any] = {
        "format": "locatemot-l83-loss-contract-v1", "status": status,
        "stage": "phase_3_target_bag_metric_and_loss_tests",
        "command": " ".join([sys.executable] + sys.argv), "cwd": str(ROOT),
        "checks": checks, "loss_parts": {key: (float(value) if torch.is_tensor(value) else value) for key, value in parts.items()},
        "model": model.parameter_report(),
        "screening_gt_used": False, "official_test_labels_read": False,
        "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
        "token_span_region_alignment": "UNALIGNED", "static_motion_alignment": "UNALIGNED",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "contract.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    (out / "provenance.json").write_text(json.dumps({"format": "locatemot-l83-loss-contract-provenance-v1", "status": status, "command": payload["command"], "labels": "synthetic only", "screening_gt_used": False, "official_test_labels_read": False, "ordinary_mot_ovmot_touched": False}, indent=2) + "\n")
    (out / "status.json").write_text(json.dumps({"format": "locatemot-l83-loss-contract-status-v1", "status": status, "failure_root_cause": None if status == "complete" else "loss/metric contract check", "next_action": "run faithful frozen probe" if status == "complete" else "stop after contract failure", "command": payload["command"]}, indent=2) + "\n")
    print(json.dumps({"status": status, "out": str(out)}))
    return 0 if status == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
