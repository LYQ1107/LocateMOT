#!/usr/bin/env python3
"""Synthetic gradient regression for the L47 grouped loss contract."""
from __future__ import annotations

import json
from pathlib import Path

import torch

ROOT = Path("/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT")
import sys
sys.path.insert(0, str(ROOT))

from locatemot.models.l47_teacher_anchored_output_contract import (  # noqa: E402
    L47TeacherAnchoredOutputContract,
    teacher_anchored_loss,
)


def run_case(labels, teacher_values, hard_values):
    torch.manual_seed(20260829)
    model = L47TeacherAnchoredOutputContract(hidden=128, heads=4, layers=2,
                                             residual_bound=0.05, dropout=0.0)
    model.train()
    region = torch.randn(len(labels), 512)
    history = torch.randn(len(labels), 512)
    numeric = torch.randn(len(labels), 32)
    query = torch.randn(7, 768)
    mask = torch.ones(7, dtype=torch.bool)
    teacher = torch.as_tensor(teacher_values, dtype=torch.float32)
    labels_tensor = torch.as_tensor(labels, dtype=torch.bool)
    hard = torch.as_tensor(hard_values, dtype=torch.long)
    pos = torch.nonzero(labels_tensor, as_tuple=False).flatten()
    pairs = torch.as_tensor([(int(p), int(h)) for p in pos.tolist() for h in hard.tolist()], dtype=torch.long)
    correct = pairs[(teacher[pairs[:, 0]] > teacher[pairs[:, 1]])] if len(pairs) else torch.empty((0, 2), dtype=torch.long)
    error = pairs[(teacher[pairs[:, 0]] <= teacher[pairs[:, 1]])] if len(pairs) else torch.empty((0, 2), dtype=torch.long)
    output = model(region, history, numeric, query, mask, teacher)
    output["final_score"].retain_grad()
    loss, parts = teacher_anchored_loss(output, labels_tensor, hard, correct, error)
    loss.backward()
    score_grad = output["final_score"].grad.detach().abs()
    parameter_grad = torch.cat([
        value.grad.detach().abs().flatten()
        for value in model.parameters() if value.grad is not None
    ])
    return {
        "labels": [int(x) for x in labels],
        "hard_indices": [int(x) for x in hard_values],
        "loss": float(loss.detach()),
        "loss_parts": parts,
        "positive_score_gradients": [float(x) for x in score_grad[pos]],
        "hard_score_gradients": [float(x) for x in score_grad[hard]],
        "all_positive_nonzero": bool(len(pos) == 0 or (score_grad[pos] > 1e-10).all()),
        "all_hard_nonzero": bool(len(hard) == 0 or (score_grad[hard] > 1e-10).all()),
        "parameter_grad_finite": bool(torch.isfinite(parameter_grad).all()),
        "parameter_grad_nonzero": bool((parameter_grad > 1e-10).any()),
    }


def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"wrong project root: {Path.cwd().resolve()}")
    two_positive = run_case(
        [True, True, False, False], [0.4, 0.3, 0.2, 0.1], [2, 3]
    )
    inactive = run_case(
        [False, False, False, False], [0.2, 0.1, 0.0, -0.1], []
    )
    payload = {
        "format": "locatemot-l47-model-gradient-regression-v1",
        "synthetic_only": True,
        "two_positive_two_negative": two_positive,
        "all_inactive": inactive,
        "passed": all([
            two_positive["all_positive_nonzero"],
            two_positive["all_hard_nonzero"],
            two_positive["parameter_grad_finite"],
            two_positive["parameter_grad_nonzero"],
            inactive["parameter_grad_finite"],
            inactive["parameter_grad_nonzero"],
        ]),
        "screening_gt_used": False,
    }
    out = ROOT / "outputs/l47/audit/model_gradient_regression.json"
    out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
