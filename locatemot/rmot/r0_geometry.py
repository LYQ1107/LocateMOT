"""Causal five-step box geometry for the R0 sidecar."""
from __future__ import annotations

from typing import Any

import torch


def _bank_tensors(store: Any) -> dict[str, torch.Tensor]:
    if hasattr(store, "tensors"):
        return store.tensors
    if hasattr(store, "bank") and isinstance(store.bank, dict):
        return store.bank["tensors"]
    if isinstance(store, dict) and "tensors" in store:
        return store["tensors"]
    raise TypeError("store must expose tensors or bank['tensors']")


def _track_rows(store: Any, tensors: dict[str, torch.Tensor]) -> dict[int, list[int]]:
    if hasattr(store, "track_rows"):
        return {int(k): [int(x) for x in v] for k, v in store.track_rows.items()}
    if hasattr(store, "_track_rows"):
        return {int(k): [int(x) for x in v] for k, v in store._track_rows.items()}
    result: dict[int, list[int]] = {}
    for offset, track in enumerate(tensors["track_id"].long().tolist()):
        result.setdefault(int(track), []).append(int(offset))
    return result


def _step(box: torch.Tensor, previous: torch.Tensor | None, frame: int, previous_frame: int | None,
          width: float, height: float) -> torch.Tensor:
    scale = box.new_tensor([width, height, width, height])
    normalized = box / scale
    center = (normalized[:2] + normalized[2:]) * 0.5
    size = (normalized[2:] - normalized[:2]).clamp_min(1e-6)
    if previous is None or previous_frame is None:
        delta = box.new_zeros(4)
        dt = box.new_zeros(1)
    else:
        old = previous / scale
        old_center = (old[:2] + old[2:]) * 0.5
        old_size = (old[2:] - old[:2]).clamp_min(1e-6)
        delta = torch.cat((center - old_center, torch.log(size / old_size)))
        dt = box.new_tensor([max(0, int(frame) - int(previous_frame))])
    return torch.cat((center, size, delta, dt, box.new_ones(1)))


def build_track_geometry(store: Any, batch: Any, image_hw: tuple[int, int] | list[int], history_length: int = 4) -> torch.Tensor:
    """Build ``[N,history_length+1,10]`` current-and-causal box features.

    ``image_hw`` is ``(height,width)``.  Track IDs are used only to look up
    bank observations and never appear in the returned tensor.
    """
    if int(history_length) < 0:
        raise ValueError("history_length must be nonnegative")
    tensors = _bank_tensors(store)
    if not hasattr(batch, "row_offsets") or not hasattr(batch, "frame_id"):
        raise TypeError("batch must expose row_offsets and frame_id")
    offsets = [int(x) for x in batch.row_offsets]
    boxes = tensors["box"].float()
    frames = tensors["frame"].long()
    tracks = tensors["track_id"].long()
    width, height = float(image_hw[1]), float(image_hw[0])
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid image_hw={image_hw}")
    rows_by_track = _track_rows(store, tensors)
    result = boxes.new_zeros((len(offsets), int(history_length) + 1, 10))
    current_frame = int(batch.frame_id)
    for out_row, offset in enumerate(offsets):
        if offset < 0 or offset >= int(boxes.shape[0]):
            raise IndexError(f"row offset out of range: {offset}")
        track = int(tracks[offset])
        eligible = [row for row in rows_by_track.get(track, [offset])
                    if int(frames[row]) <= current_frame]
        if offset not in eligible:
            eligible.append(offset)
        eligible = sorted(set(eligible), key=lambda row: (int(frames[row]), int(row)))
        chosen = eligible[-(int(history_length) + 1):]
        start = int(history_length) + 1 - len(chosen)
        previous_box: torch.Tensor | None = None
        previous_frame: int | None = None
        for local, row in enumerate(chosen, start=start):
            frame = int(frames[row])
            if frame > current_frame:
                raise AssertionError("future observation in R0 track geometry")
            result[out_row, local] = _step(boxes[row], previous_box, frame, previous_frame, width, height)
            previous_box = boxes[row]
            previous_frame = frame
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("nonfinite R0 geometry")
    return result


__all__ = ["build_track_geometry"]
