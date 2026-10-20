"""Decode typed logits into auditable probabilities and actions."""
from __future__ import annotations

from typing import Any, Sequence

import torch

from .g2_jev_readout import LIFECYCLE_ACTIONS


def decode_typed_decisions(outputs: dict[str, Any], option_tokens: Sequence[str]) -> list[dict[str, Any]]:
    identity = torch.softmax(outputs["identity_logits"], dim=-1)
    admit = torch.sigmoid(outputs["admit_logits"])
    yesno = torch.sigmoid(outputs["yesno_logits"])
    life = torch.softmax(outputs["lifecycle_logits"], dim=-1)
    memory = torch.sigmoid(outputs["memory_logits"])
    result: list[dict[str, Any]] = []
    for i in range(identity.shape[0]):
        probs = {str(token): float(identity[i, j].detach().cpu()) for j, token in enumerate(option_tokens)}
        ranked = sorted(probs.items(), key=lambda pair: (-pair[1], pair[0]))
        best = ranked[0] if ranked else ("UNKNOWN", 0.0)
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        life_probs = {name: float(life[i, j].detach().cpu()) for j, name in enumerate(LIFECYCLE_ACTIONS)}
        result.append({
            "candidate_index": i,
            "identity": best[0],
            "identity_probability": best[1],
            "identity_margin": best[1] - second,
            "identity_probabilities": probs,
            "admit_probability": float(admit[i].detach().cpu()),
            "yes_probability": float(yesno[i].detach().cpu()),
            "lifecycle": max(life_probs, key=life_probs.get),
            "lifecycle_probabilities": life_probs,
            "memory_update_probability": float(memory[i].detach().cpu()),
        })
    return result
