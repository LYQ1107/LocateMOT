"""Learned fragment-pair graph for the optional L20 Phase B."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class FragmentNode:
    node_id: int
    frame: int
    box: np.ndarray
    semantic: np.ndarray
    identity: np.ndarray
    motion: np.ndarray
    validity: float
    source: int


@dataclass
class FragmentEdge:
    left: int
    right: int
    score: float
    same_identity: bool | None = None


class L20FragmentPairGraph(nn.Module):
    """Source-blind edge scorer with an explicit learned no-match readout.

    Source is accepted by callers for provenance and adapter selection only;
    it is not concatenated to the edge feature.  The inference solver uses
    learned edge/no-match scores and enforces one predecessor/successor per
    node, preventing same-frame identity collisions.
    """

    def __init__(self, embedding_dim: int = 256, motion_dim: int = 8,
                 hidden: int = 256, dropout: float = 0.10):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.motion_dim = int(motion_dim)
        # ``extra`` is the caller-provided motion/geometry/time feature
        # vector; the current Phase-B interface uses ``motion_dim`` values.
        edge_dim = 6 * embedding_dim + motion_dim
        self.edge_head = nn.Sequential(
            nn.LayerNorm(edge_dim), nn.Linear(edge_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.no_match_head = nn.Sequential(
            nn.LayerNorm(embedding_dim + motion_dim + 1),
            nn.Linear(embedding_dim + motion_dim + 1, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def edge_features(self, semantic_a: torch.Tensor, semantic_b: torch.Tensor,
                      identity_a: torch.Tensor, identity_b: torch.Tensor,
                      extra: torch.Tensor) -> torch.Tensor:
        return torch.cat((
            semantic_a, semantic_b, (semantic_a - semantic_b).abs(),
            semantic_a * semantic_b,
            identity_a * identity_b, (identity_a - identity_b).abs(),
            extra,
        ), dim=-1)

    def forward(self, semantic_a: torch.Tensor, semantic_b: torch.Tensor,
                identity_a: torch.Tensor, identity_b: torch.Tensor,
                extra: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.edge_features(
            semantic_a, semantic_b, identity_a, identity_b, extra)
        return {"same_identity_logit": self.edge_head(features).squeeze(-1)}

    def no_match(self, semantic: torch.Tensor, motion: torch.Tensor,
                 validity: torch.Tensor) -> torch.Tensor:
        return self.no_match_head(torch.cat((semantic, motion,
                                             validity.reshape(-1, 1)), dim=-1)).squeeze(-1)

    @torch.no_grad()
    def infer(self, nodes: list[FragmentNode], edges: list[FragmentEdge],
              threshold: float = 0.0) -> list[FragmentEdge]:
        """Select learned high-score temporal edges with degree constraints."""
        chosen = []
        used_left, used_right = set(), set()
        for edge in sorted(edges, key=lambda value: value.score, reverse=True):
            if edge.score < float(threshold):
                continue
            if edge.left in used_left or edge.right in used_right:
                continue
            left = next(node for node in nodes if node.node_id == edge.left)
            right = next(node for node in nodes if node.node_id == edge.right)
            if left.frame >= right.frame:
                continue
            used_left.add(edge.left)
            used_right.add(edge.right)
            chosen.append(edge)
        return chosen
