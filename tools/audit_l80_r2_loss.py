#!/usr/bin/env python3
"""Small R2 loss contract audit; no detector or evaluation labels are read."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT").resolve()
sys.path.insert(0, str(ROOT))
from locatemot.rmot.l80_r2_losses import l80_r2_loss  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def one(category: str, labels: list[int], covered: bool) -> dict[str, Any]:
    n = len(labels)
    def vector() -> torch.Tensor:
        return torch.linspace(-0.4, 0.5, n, requires_grad=True)
    output = {
        "candidate_logits": vector(), "track_logits": vector(),
        "continuation_logits": vector(), "quality_logits": vector(),
        "null_logit": torch.tensor(0.1, requires_grad=True),
        "cardinality_logit": torch.tensor(0.1, requires_grad=True),
    }
    observations = torch.zeros(n, 1432); observations[:, -1] = torch.arange(n, dtype=torch.float32)
    history_mask = torch.ones(n, 8, dtype=torch.bool)
    loss, parts = l80_r2_loss(output, torch.tensor(labels, dtype=torch.bool), covered,
                              observations, history_mask, category)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError(f"nonfinite R2 {category}")
    loss.backward()
    gradients = output["candidate_logits"].grad
    if gradients is not None and not bool(torch.isfinite(gradients).all()):
        raise AssertionError(f"nonfinite candidate gradient {category}")
    if covered and category != "present_uncovered" and gradients is None:
        raise AssertionError(f"missing candidate gradient {category}")
    if category == "multi_positive" and not bool((gradients[torch.tensor(labels, dtype=torch.bool)] != 0).all()):
        raise AssertionError("multi-positive candidate did not receive all-positive gradient")
    if category == "inactive" and not bool((gradients != 0).all()):
        raise AssertionError("inactive candidates did not receive negative gradient")
    gradient_nonzero = 0 if gradients is None else int((gradients != 0).sum())
    return {"category": category, "loss": float(loss.detach()), "parts": parts,
            "candidate_gradient_nonzero": gradient_nonzero,
            "candidate_count": n, "candidate_deletion": False, "candidate_truncation": False}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); out = (args.out if args.out.is_absolute() else ROOT / args.out).resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError(f"nonempty {out}")
    out.mkdir(parents=True, exist_ok=True)
    command = " ".join([sys.executable] + sys.argv)
    try:
        records = [one("multi_positive", [1, 1, 0, 0, 0], True),
                   one("positive", [1, 0, 0], True), one("inactive", [0, 0, 0, 0], True),
                   one("present_uncovered", [0, 0, 0, 0], False)]
        payload = {"format": "locatemot-l80-r2-loss-contract-v1", "status": "complete",
                   "records": records, "same_class_hard_negative_metadata": "unavailable; all-negative fallback",
                   "loss_only_change": True, "finite": True, "positive_and_negative_gradients": True,
                   "candidate_deletion": False, "candidate_truncation": False,
                   "screening_gt_used": False, "official_test_labels_read": False,
                   "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False,
                   "next_action": "run bounded R2 fit probe"}
        write_json(out / "contract.json", payload)
        write_json(out / "provenance.json", {"format": "locatemot-l80-r2-loss-provenance-v1", "status": "complete",
            "command": command, "project_root": str(ROOT), "synthetic_only": True,
            "inputs": {"loss_module": str(ROOT / "locatemot/rmot/l80_r2_losses.py")},
            "labels_from_dataset_read": False, "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        write_json(out / "status.json", {"format": "locatemot-l80-r2-status-v1", "status": "complete",
            "command": command, "failure_root_cause": None, "next_action": "run bounded R2 fit probe",
            "screening_gt_used": False, "official_test_labels_read": False,
            "ordinary_mot_ovmot_touched": False, "hota_trackeval_run": False})
        return 0
    except Exception:
        (out / "INCOMPLETE.md").write_text("# L80-R2 loss contract — INCOMPLETE\n\n" + __import__("traceback").format_exc() + "\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
