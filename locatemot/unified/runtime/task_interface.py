"""Stable public task names and model call contract."""
from __future__ import annotations

TASK_TYPES = ("mot", "ovmot", "rmot", "point", "box", "mask")


def validate_task_type(task_type: str) -> str:
    if task_type not in TASK_TYPES:
        raise ValueError(f"unsupported LocateMOT-U task: {task_type}")
    return task_type
