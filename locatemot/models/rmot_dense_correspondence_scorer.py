"""Query-conditioned dense correspondence scorer for Stage L23.

The module is deliberately independent of tracker/source/grouping state.  The
``stage`` switches expose a small sequence of ablations that can be compared
without changing the frozen banks or protocol.
"""
from __future__ import annotations

import torch
from torch import nn


def _flat(value: torch.Tensor, batch: int, width: int, name: str, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(value, device=device).float()
    if value.ndim == 1:
        value = value.unsqueeze(0)
    value = value.reshape(batch, -1)
    if value.shape[1] != width:
        raise ValueError(f"{name}: expected flattened width {width}, got {value.shape}")
    return torch.nan_to_num(value)


class DenseQueryCorrespondenceScorer(nn.Module):
    """Dense ROI token scorer with optional context/history/C-Hook stages."""

    source_in_score = False
    uses_grouping = False
    uses_membership = False
    uses_source_acceptance = False
    uses_null_scalar = False
    uses_temporal_gru = False

    def __init__(self, stage: str = "D0", query_dim: int = 512, dense_dim: int = 512,
                 geometry_dim: int = 10, motion_dim: int = 16, hidden: int = 256,
                 heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if stage not in {"D0", "D1", "D2", "D3", "D4"}:
            raise ValueError(f"unknown correspondence stage {stage}")
        self.stage = stage
        self.query_dim, self.dense_dim = query_dim, dense_dim
        self.geometry_dim, self.motion_dim, self.hidden = geometry_dim, motion_dim, hidden
        self.query_proj = nn.Sequential(nn.LayerNorm(query_dim), nn.Linear(query_dim, hidden), nn.GELU())
        self.token_proj = nn.Sequential(nn.LayerNorm(dense_dim), nn.Linear(dense_dim, hidden), nn.GELU())
        self.cross_attention = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.token_norm = nn.LayerNorm(hidden)
        self.output_norm = nn.LayerNorm(hidden)
        if stage in {"D1", "D2", "D3", "D4"}:
            self.context_proj = nn.Sequential(nn.LayerNorm(dense_dim), nn.Linear(dense_dim, hidden), nn.GELU())
        if stage in {"D2", "D3", "D4"}:
            self.history_proj = nn.Sequential(nn.LayerNorm(dense_dim), nn.Linear(dense_dim, hidden), nn.GELU())
            self.relation_proj = nn.Sequential(nn.LayerNorm(dense_dim), nn.Linear(dense_dim, hidden), nn.GELU())
        if stage in {"D3", "D4"}:
            self.condition = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden * 2), nn.GELU(),
                                           nn.Linear(hidden * 2, hidden * 2))
        if stage == "D4":
            self.pair_decoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(hidden, heads, hidden * 4, dropout=dropout,
                                           batch_first=True, norm_first=True), num_layers=1)
        extra = geometry_dim + 1 + (motion_dim if stage in {"D2", "D3", "D4"} else 0)
        self.head = nn.Sequential(nn.LayerNorm(hidden * 3 + extra), nn.Linear(hidden * 3 + extra, hidden),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, query: torch.Tensor, dense_points: torch.Tensor,
                dense_roi: torch.Tensor, geometry: torch.Tensor,
                objectness: torch.Tensor, dense_context_1p5: torch.Tensor | None = None,
                dense_context_3: torch.Tensor | None = None,
                dense_prev_roi: torch.Tensor | None = None,
                motion: torch.Tensor | None = None) -> torch.Tensor:
        query = torch.as_tensor(query)
        if query.ndim == 1:
            query = query.unsqueeze(0)
        batch = query.shape[0]
        device = next(self.parameters()).device
        q = self.query_proj(_flat(query, batch, self.query_dim, "query", device)).unsqueeze(1)
        points = torch.as_tensor(dense_points, device=device).float().reshape(batch, -1, self.dense_dim)
        points = torch.nan_to_num(points)
        tokens = [self.token_proj(points), self.token_proj(_flat(dense_roi, batch, self.dense_dim, "dense_roi", device)).unsqueeze(1)]
        if self.stage in {"D1", "D2", "D3", "D4"}:
            if dense_context_1p5 is None or dense_context_3 is None:
                raise ValueError(f"{self.stage} requires context features")
            tokens.extend([self.context_proj(_flat(dense_context_1p5, batch, self.dense_dim, "context_1p5", device)).unsqueeze(1),
                           self.context_proj(_flat(dense_context_3, batch, self.dense_dim, "context_3", device)).unsqueeze(1)])
        if self.stage in {"D2", "D3", "D4"}:
            if dense_prev_roi is None:
                raise ValueError(f"{self.stage} requires previous ROI features")
            previous_raw = _flat(dense_prev_roi, batch, self.dense_dim, "prev_roi", device)
            current_raw = _flat(dense_roi, batch, self.dense_dim, "dense_roi", device)
            history = self.history_proj(previous_raw)
            tokens.extend([history.unsqueeze(1), self.relation_proj(current_raw - previous_raw).unsqueeze(1)])
        token_set = self.token_norm(torch.cat(tokens, dim=1))
        if self.stage in {"D3", "D4"}:
            gamma, beta = self.condition(q.squeeze(1)).chunk(2, dim=-1)
            token_set = token_set * (1.0 + 0.1 * torch.tanh(gamma).unsqueeze(1)) + 0.1 * beta.unsqueeze(1)
        if self.stage == "D4":
            token_set = self.pair_decoder(token_set)
        attended, _ = self.cross_attention(q, token_set, token_set, need_weights=False)
        pooled = token_set.mean(dim=1)
        fused = self.output_norm(attended.squeeze(1) + q.squeeze(1))
        roi = self.token_proj(_flat(dense_roi, batch, self.dense_dim, "dense_roi", device))
        extra_values = [_flat(geometry, batch, self.geometry_dim, "geometry", device),
                        _flat(objectness, batch, 1, "objectness", device)]
        if self.stage in {"D2", "D3", "D4"}:
            if motion is None:
                motion = torch.zeros(batch, self.motion_dim, device=device)
            extra_values.append(_flat(motion, batch, self.motion_dim, "motion", device))
        extra = torch.cat(extra_values, dim=-1)
        return self.head(torch.cat((fused, pooled, roi, extra), dim=-1)).squeeze(-1)


__all__ = ["DenseQueryCorrespondenceScorer"]
