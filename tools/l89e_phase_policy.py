#!/usr/bin/env python3
"""Single source of truth for the registered L89 phase/history contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from locatemot.rmot.l86_clip_data import _clip_history


@dataclass(frozen=True)
class L89PhasePolicy:
    epoch: int
    phase: str
    temporal_enabled: bool
    history_mode: str
    history_length: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def phase_policy_for_epoch(
    epoch: int,
    checkpoint_phase: str | None = None,
) -> L89PhasePolicy:
    epoch = int(epoch)
    if 1 <= epoch <= 8:
        expected_phase = "S"
        temporal_enabled = False
        history_mode = "zero_history"
        history_length = 0
    elif 9 <= epoch <= 20:
        expected_phase = "T"
        temporal_enabled = True
        history_mode = "last4_causal_history"
        history_length = 4
    elif 21 <= epoch <= 40:
        expected_phase = "J"
        temporal_enabled = True
        history_mode = "last4_causal_history"
        history_length = 4
    else:
        raise ValueError(f"L89 checkpoint epoch outside registered 1..40: {epoch}")
    if checkpoint_phase is not None and str(checkpoint_phase) != expected_phase:
        raise AssertionError(
            f"L89 checkpoint phase drift: epoch={epoch} package={checkpoint_phase} "
            f"expected={expected_phase}"
        )
    return L89PhasePolicy(
        epoch=epoch,
        phase=expected_phase,
        temporal_enabled=temporal_enabled,
        history_mode=history_mode,
        history_length=history_length,
    )


def history_for_batch(
    batch: Any,
    policy: L89PhasePolicy,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduce L86 packing while enforcing phase-specific invariants."""
    history, mask, frame_ids = _clip_history(
        batch,
        enabled=policy.temporal_enabled,
        length=4,
    )
    if history.shape != batch.history_observations.shape:
        raise AssertionError("history shape drift")
    if mask.shape != batch.history_mask.shape:
        raise AssertionError("history mask shape drift")
    if frame_ids.shape != batch.history_frame_ids.shape:
        raise AssertionError("history frame-id shape drift")
    if policy.phase == "S":
        if bool(mask.any()) or bool((frame_ids != -1).any()):
            raise AssertionError("Stage-S history is not empty")
        if not torch.equal(history, torch.zeros_like(history)):
            raise AssertionError("Stage-S history is not zero")
    else:
        valid_per_row = mask.sum(dim=1)
        if bool((valid_per_row > 4).any()):
            raise AssertionError("T/J history exceeded last4")
        if bool((frame_ids[mask] > int(batch.frame_id)).any()):
            raise AssertionError("future history entered L89E")
    if not bool(torch.isfinite(history.float()).all()):
        raise FloatingPointError("nonfinite L89E history")
    return history, mask, frame_ids


__all__ = ["L89PhasePolicy", "phase_policy_for_epoch", "history_for_batch"]
