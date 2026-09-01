"""Minimal query-conditioned RMOT candidate scorer.

This module is intentionally smaller than the rejected L20 SINT-Set route.
It scores one current candidate at a time and has no grouping, membership
head, NULL head, source-acceptance branch, temporal GRU, or set pooling.
``source_embedding`` is accepted for a common caller interface but is checked
only for batch compatibility and is never consumed by a score path.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _Branch(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(torch.nan_to_num(value.float())).squeeze(-1)


def _batch(value: torch.Tensor, name: str, batch: int,
           width: int, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(value, device=device)
    if value.ndim == 1:
        if width != 1:
            value = value.reshape(1, -1)
        else:
            value = value.reshape(-1, 1)
    if value.ndim != 2 or value.shape[0] != batch or value.shape[1] != width:
        raise ValueError(
            f"{name} must have shape [{batch},{width}], got {tuple(value.shape)}")
    return torch.nan_to_num(value.float())


class RMOTCandidateScorer(nn.Module):
    """Static/motion/correspondence scorer for one candidate per row."""

    source_in_score = False
    uses_grouping = False
    uses_membership = False
    uses_null = False
    uses_temporal_gru = False
    uses_set_pooling = False

    def __init__(self, query_dim: int = 512, current_dim: int = 512,
                 history_dim: int = 512, geometry_dim: int = 7,
                 motion_dim: int = 8, frame_delta_dim: int = 1,
                 hidden: int = 256, dropout: float = 0.10,
                 lambda_motion: float = 0.25,
                 lambda_corr: float = 0.25):
        super().__init__()
        self.query_dim = int(query_dim)
        self.current_dim = int(current_dim)
        self.history_dim = int(history_dim)
        self.geometry_dim = int(geometry_dim)
        self.motion_dim = int(motion_dim)
        self.frame_delta_dim = int(frame_delta_dim)
        self.lambda_motion = float(lambda_motion)
        self.lambda_corr = float(lambda_corr)
        if self.frame_delta_dim < 1:
            raise ValueError("frame_delta_dim must be positive")

        # Static branch: current appearance, explicit query, geometry and
        # objectness. It has no motion/history state or source indicator.
        self.static_branch = _Branch(
            self.query_dim + self.current_dim + self.geometry_dim + 1,
            hidden, dropout)
        # Motion branch: motion query, frozen history appearance, motion and
        # frame delta. The history vector is a feature, not recurrent state.
        self.motion_branch = _Branch(
            self.query_dim + self.history_dim + self.motion_dim +
            self.frame_delta_dim, hidden, dropout)
        # Correspondence branch: a local current/history relation conditioned
        # on the query. The absolute difference makes the relation explicit.
        self.correspondence_branch = _Branch(
            self.query_dim + self.current_dim + self.history_dim +
            self.current_dim + self.frame_delta_dim, hidden, dropout)

    def forward(
            self,
            query_embedding: torch.Tensor,
            static_query_embedding: torch.Tensor | None,
            motion_query_embedding: torch.Tensor | None,
            candidate_current_feature: torch.Tensor,
            candidate_historical_feature: torch.Tensor,
            bbox_geometry: torch.Tensor,
            source_embedding: torch.Tensor | None = None,
            frame_delta: torch.Tensor | None = None,
            motion_feature: torch.Tensor | None = None,
            objectness: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Return independent logits for a batch of candidate observations.

        ``source_embedding`` is deliberately not concatenated into any branch.
        Passing two different source embeddings with all other inputs fixed
        must produce identical logits; a caller can audit this invariant.
        """
        query_embedding = torch.as_tensor(query_embedding)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.unsqueeze(0)
        if query_embedding.ndim != 2:
            raise ValueError("query_embedding must be [B, query_dim]")
        batch = int(query_embedding.shape[0])
        device = next(self.parameters()).device
        query = _batch(query_embedding, "query_embedding", batch,
                       self.query_dim, device)
        static_query = query if static_query_embedding is None else _batch(
            static_query_embedding, "static_query_embedding", batch,
            self.query_dim, device)
        motion_query = query if motion_query_embedding is None else _batch(
            motion_query_embedding, "motion_query_embedding", batch,
            self.query_dim, device)
        current = _batch(candidate_current_feature, "candidate_current_feature",
                         batch, self.current_dim, device)
        history = _batch(candidate_historical_feature,
                         "candidate_historical_feature", batch,
                         self.history_dim, device)
        geometry = _batch(bbox_geometry, "bbox_geometry", batch,
                          self.geometry_dim, device)
        if motion_feature is None:
            motion = torch.zeros((batch, self.motion_dim), device=device)
        else:
            motion = _batch(motion_feature, "motion_feature", batch,
                            self.motion_dim, device)
        if frame_delta is None:
            delta = torch.zeros((batch, self.frame_delta_dim), device=device)
        else:
            delta = _batch(frame_delta, "frame_delta", batch,
                           self.frame_delta_dim, device)
        if objectness is None:
            objectness_value = torch.zeros((batch, 1), device=device)
        else:
            objectness_value = _batch(objectness, "objectness", batch, 1,
                                      device)
        if source_embedding is not None:
            source_embedding = torch.as_tensor(source_embedding, device=device)
            if source_embedding.ndim == 1:
                source_embedding = source_embedding.unsqueeze(0)
            if source_embedding.ndim != 2 or source_embedding.shape[0] != batch:
                raise ValueError(
                    "source_embedding must have the same batch dimension")

        static_logit = self.static_branch(torch.cat(
            (static_query, current, geometry, objectness_value), dim=-1))
        motion_logit = self.motion_branch(torch.cat(
            (motion_query, history, motion, delta), dim=-1))
        relation = current - history
        correspondence_logit = self.correspondence_branch(torch.cat(
            (query, current, history, relation, delta), dim=-1))
        final_logit = (static_logit + self.lambda_motion * motion_logit +
                       self.lambda_corr * correspondence_logit)
        return {
            "static_logit": static_logit,
            "motion_logit": motion_logit,
            "correspondence_logit": correspondence_logit,
            "final_candidate_logit": final_logit,
        }


__all__ = ["RMOTCandidateScorer"]
