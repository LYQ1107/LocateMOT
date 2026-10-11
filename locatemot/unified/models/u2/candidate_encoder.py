"""Candidate projection with independent identity and semantic paths."""
from __future__ import annotations

import torch
from torch import nn


def _geometry(boxes: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    xy0, xy1 = boxes[:, :2], boxes[:, 2:]
    center = (xy0 + xy1) * 0.5
    size = (xy1 - xy0).clamp_min(1e-5)
    # Detector bridges normally provide normalized xyxy boxes.  Dividing by a
    # per-frame extent also keeps legal pixel-coordinate inputs numerically
    # stable without requiring image metadata in the state contract.
    if boxes.shape[0]:
        extent = torch.maximum(xy1.max(), boxes.new_tensor(1.0))
        center = center / extent
        size = size / extent
    return torch.cat([center, size, size.clamp_min(1e-5).log(), scores[:, None].clamp(0.0, 1.0)], dim=-1)


class CandidateEncoder(nn.Module):
    """Encode visual candidates; no class text enters the identity projection."""

    def __init__(self, hidden_dim: int = 256, visual_dim: int = 256) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.visual_dim = int(visual_dim)
        self.id_projection = nn.Sequential(
            nn.Linear(visual_dim + 7, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.semantic_projection = nn.Sequential(
            nn.Linear(visual_dim * 2 + 7, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        id_visual: torch.Tensor,
        boxes: torch.Tensor,
        scores: torch.Tensor,
        decoder_visual: torch.Tensor | None = None,
        semantic_visual: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if id_visual.ndim != 2 or boxes.ndim != 2 or boxes.shape[-1] != 4:
            raise ValueError("candidate inputs must be id_visual [N,D], boxes [N,4]")
        if semantic_visual is None:
            semantic_visual = id_visual
        if decoder_visual is None:
            decoder_visual = semantic_visual
        if id_visual.shape[0] != boxes.shape[0] or scores.shape != (boxes.shape[0],):
            raise ValueError("candidate batch lengths do not match")
        if id_visual.shape[-1] != self.visual_dim or semantic_visual.shape[-1] != self.visual_dim:
            raise ValueError("visual feature dimension mismatch")
        if decoder_visual.shape != semantic_visual.shape:
            raise ValueError("decoder_visual shape mismatch")
        geom = _geometry(boxes, scores)
        id_token = self.id_projection(torch.cat([id_visual, geom], dim=-1))
        sem_token = self.semantic_projection(torch.cat([semantic_visual, decoder_visual, geom], dim=-1))
        return {"candidate_id_token": id_token, "candidate_sem_token": sem_token, "geometry": geom}
