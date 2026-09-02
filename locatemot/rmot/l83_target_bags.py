"""Duplicate-aware target-bag layout for the L83 RMOT probe.

Rows belonging to one non-null ``candidate_gt`` are one target bag.  A
background row is deliberately a singleton negative bag so that the strongest
false candidate cannot be hidden by pooling all background rows together.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


BagKey = tuple[str, str | int]


@dataclass(frozen=True)
class TargetBagLayout:
    target_to_rows: dict[str, torch.LongTensor]
    background_rows: torch.LongTensor
    unique_target_ids: tuple[str, ...]
    row_count: int

    def items(self) -> list[tuple[BagKey, torch.LongTensor]]:
        values: list[tuple[BagKey, torch.LongTensor]] = [
            (("target", target), rows) for target, rows in self.target_to_rows.items()
        ]
        values.extend(
            (("background", int(row)), torch.tensor([int(row)], dtype=torch.long))
            for row in self.background_rows.tolist()
        )
        return values


def build_target_bag_layout(candidate_gt: Iterable[object | None]) -> TargetBagLayout:
    values = [None if value is None else str(value) for value in candidate_gt]
    target_rows: dict[str, list[int]] = {}
    backgrounds: list[int] = []
    for row, target in enumerate(values):
        if target is None:
            backgrounds.append(row)
        else:
            target_rows.setdefault(target, []).append(row)
    ordered = {target: torch.tensor(rows, dtype=torch.long) for target, rows in sorted(target_rows.items())}
    background = torch.tensor(backgrounds, dtype=torch.long)
    return TargetBagLayout(
        target_to_rows=ordered,
        background_rows=background,
        unique_target_ids=tuple(ordered),
        row_count=len(values),
    )


def normalize_target_ids(target_ids: Iterable[object]) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in target_ids}))


def bag_values(
    row_scores: torch.Tensor,
    layout: TargetBagLayout,
    referred_target_ids: Iterable[object],
) -> tuple[list[BagKey], torch.Tensor, torch.Tensor]:
    """Return unique bag keys, scores and positive mask in deterministic order."""
    if row_scores.ndim != 1 or row_scores.numel() != layout.row_count:
        raise ValueError("row score/layout mismatch")
    if not bool(torch.isfinite(row_scores.float()).all()):
        raise FloatingPointError("nonfinite row scores")
    referred = set(normalize_target_ids(referred_target_ids))
    keys: list[BagKey] = []
    scores: list[torch.Tensor] = []
    positives: list[bool] = []
    for target, rows in layout.target_to_rows.items():
        keys.append(("target", target))
        scores.append(torch.amax(row_scores[rows.to(row_scores.device)]))
        positives.append(target in referred)
    for row in layout.background_rows.tolist():
        keys.append(("background", int(row)))
        scores.append(row_scores[int(row)])
        positives.append(False)
    if not scores:
        return keys, row_scores.new_empty((0,)), torch.empty(0, dtype=torch.bool, device=row_scores.device)
    return keys, torch.stack(scores), torch.tensor(positives, dtype=torch.bool, device=row_scores.device)


def positive_target_bags(
    layout: TargetBagLayout, referred_target_ids: Iterable[object]
) -> tuple[str, ...]:
    referred = set(normalize_target_ids(referred_target_ids))
    return tuple(target for target in layout.unique_target_ids if target in referred)


__all__ = [
    "BagKey", "TargetBagLayout", "bag_values", "build_target_bag_layout",
    "normalize_target_ids", "positive_target_bags",
]
