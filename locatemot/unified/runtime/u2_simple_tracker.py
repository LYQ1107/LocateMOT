"""Frozen causal IoU/motion/visual-cosine Hungarian baseline for U2."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F

from ..models.u2.shared_track_core import _iou_matrix


@dataclass(frozen=True)
class SimpleTrackerConfig:
    iou_weight: float = .25
    motion_iou_weight: float = .35
    visual_cosine_weight: float = .40
    minimum_match_score: float = .45
    maximum_center_distance: float = 2.
    minimum_birth_score: float = .10
    appearance_ema: float = .90
    max_age: int = 30


class U2SimpleTracker:
    """Consume every cached candidate; emit only current observed boxes.

    There is no learned parameter, class feature, or GT input. Low-score
    candidates may recover an existing ID; only births require score >= .10.
    All fixed parameters are registered before benchmark evaluation.
    """

    def __init__(self, config: SimpleTrackerConfig | None = None) -> None:
        self.config = config or SimpleTrackerConfig()
        self.tracks: list[dict[str, Any]] = []
        self.next_id = 0
        self.frame_index = -1

    def step(self, candidates: dict[str, torch.Tensor], *, frame_index: int | None = None) -> dict:
        frame = self.frame_index + 1 if frame_index is None else int(frame_index)
        if frame <= self.frame_index:
            raise ValueError("Simple tracker frames must advance causally")
        boxes = candidates["boxes"].detach().float().cpu()
        scores = candidates["scores"].detach().float().cpu()
        appearance = F.normalize(candidates["id_visual"].detach().float().cpu(), dim=-1)
        self.tracks = [track for track in self.tracks if frame - track["last_seen"] <= self.config.max_age]
        matched: list[tuple[int, int]] = []
        if self.tracks and len(boxes):
            last = torch.stack([track["box"] for track in self.tracks])
            velocity = torch.stack([track["velocity"] for track in self.tracks])
            gap = boxes.new_tensor([frame - track["last_seen"] for track in self.tracks])
            predicted = last + velocity * gap[:, None]
            identity = F.normalize(torch.stack([track["appearance"] for track in self.tracks]), dim=-1)
            cosine = identity @ appearance.T
            centers = (predicted[:, :2] + predicted[:, 2:]) * .5
            candidate_centers = (boxes[:, :2] + boxes[:, 2:]) * .5
            scale = (last[:, 2:] - last[:, :2]).norm(dim=-1).clamp_min(1.)
            distance = (centers[:, None] - candidate_centers[None]).norm(dim=-1) / scale[:, None]
            affinity = (self.config.iou_weight * _iou_matrix(last, boxes)
                        + self.config.motion_iou_weight * _iou_matrix(predicted, boxes)
                        + self.config.visual_cosine_weight * cosine)
            affinity = affinity.masked_fill(distance > self.config.maximum_center_distance, -1e6)
            # One dummy per track allows true no-match decisions inside the
            # optimization rather than filtering forced assignments afterward.
            no_match = affinity.new_full((len(self.tracks), len(self.tracks)),
                                         self.config.minimum_match_score)
            rows, cols = linear_sum_assignment(-torch.cat([affinity, no_match], dim=1).numpy())
            matched = [(int(row), int(col)) for row, col in zip(rows, cols)
                       if col < len(boxes) and affinity[row, col] > self.config.minimum_match_score]
        emitted: dict[int, int] = {}
        for row, col in matched:
            track = self.tracks[row]
            gap = frame - track["last_seen"]
            track["velocity"] = (boxes[col] - track["box"]) / gap
            track["box"] = boxes[col].clone()
            track["appearance"] = (self.config.appearance_ema * track["appearance"]
                                   + (1. - self.config.appearance_ema) * appearance[col])
            track["last_seen"] = frame
            emitted[col] = track["id"]
        for col in range(len(boxes)):
            if col not in emitted and scores[col] >= self.config.minimum_birth_score:
                self.tracks.append({"id": self.next_id, "box": boxes[col].clone(),
                                    "velocity": torch.zeros(4), "appearance": appearance[col].clone(),
                                    "last_seen": frame})
                emitted[col] = self.next_id
                self.next_id += 1
        self.frame_index = frame
        indices = torch.tensor(sorted(emitted), dtype=torch.long)
        return {"candidate_indices": indices,
                "track_ids": torch.tensor([emitted[index] for index in indices.tolist()], dtype=torch.long),
                "boxes": boxes[indices], "scores": scores[indices]}

    def state_dict(self) -> dict:
        return {"format": "locatemot-u2-simple-online-state-v1", "config": asdict(self.config),
                "tracks": [{key: value.clone() if isinstance(value, torch.Tensor) else value
                            for key, value in track.items()} for track in self.tracks],
                "next_id": self.next_id, "frame_index": self.frame_index}

    def load_state_dict(self, payload: dict) -> None:
        if payload["format"] != "locatemot-u2-simple-online-state-v1" or payload["config"] != asdict(self.config):
            raise ValueError("Simple tracker state/configuration differs")
        self.tracks = [{key: value.clone() if isinstance(value, torch.Tensor) else value
                        for key, value in track.items()} for track in payload["tracks"]]
        self.next_id, self.frame_index = payload["next_id"], payload["frame_index"]


__all__ = ["SimpleTrackerConfig", "U2SimpleTracker"]
