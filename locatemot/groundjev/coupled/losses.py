"""Joint typed losses with explicit validity masks."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F


def _masked_bce(logits: torch.Tensor, targets: Sequence[float], valid: Sequence[bool]) -> torch.Tensor | None:
    indices = [i for i, ok in enumerate(valid) if ok]
    if not indices:
        return None
    y = torch.tensor([float(targets[i]) for i in indices], device=logits.device)
    return F.binary_cross_entropy_with_logits(logits[indices].reshape(-1), y)


def structured_groundjev_losses(outputs: Mapping[str, Any], *, identity_targets: Sequence[int | None],
                                identity_valid: Sequence[bool], admit_targets: Sequence[float] | None = None,
                                admit_valid: Sequence[bool] | None = None,
                                yesno_targets: Sequence[float] | None = None,
                                yesno_valid: Sequence[bool] | None = None,
                                lifecycle_targets: Sequence[int | None] | None = None,
                                lifecycle_valid: Sequence[bool] | None = None,
                                memory_targets: Sequence[float] | None = None,
                                memory_valid: Sequence[bool] | None = None,
                                weights: Mapping[str, float] | None = None) -> dict[str, torch.Tensor | None]:
    weights = dict(weights or {})
    result: dict[str, torch.Tensor | None] = {}
    valid_ids = [i for i, ok in enumerate(identity_valid) if ok and identity_targets[i] is not None]
    if valid_ids:
        logits = outputs["identity_logits"][valid_ids]
        target = torch.tensor([int(identity_targets[i]) for i in valid_ids], device=logits.device)
        result["identity_choice"] = F.cross_entropy(logits, target)
        result["identity_brier"] = ((torch.softmax(logits, -1) - F.one_hot(target, logits.shape[-1]).float()) ** 2).sum(-1).mean()
    for name, key, targets, valid in (
        ("admit", "admit_logits", admit_targets, admit_valid),
        ("yesno", "yesno_logits", yesno_targets, yesno_valid),
        ("memory", "memory_logits", memory_targets, memory_valid),
    ):
        if targets is not None and valid is not None:
            value = _masked_bce(outputs[key], targets, valid)
            if value is not None:
                result[name] = value
    if lifecycle_targets is not None and lifecycle_valid is not None:
        indices = [i for i, ok in enumerate(lifecycle_valid) if ok and lifecycle_targets[i] is not None]
        if indices:
            logits = outputs["lifecycle_logits"][indices]
            target = torch.tensor([int(lifecycle_targets[i]) for i in indices], device=logits.device)
            result["lifecycle"] = F.cross_entropy(logits, target)
    total: torch.Tensor | None = None
    for name, value in result.items():
        if value is None or name.endswith("brier"):
            continue
        total = value * float(weights.get(name, 1.0)) if total is None else total + value * float(weights.get(name, 1.0))
    result["total"] = total
    return result
