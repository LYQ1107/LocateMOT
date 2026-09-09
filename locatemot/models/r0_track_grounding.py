"""Track-Centric Grounding Hook (TCGH) model for the isolated R0 branch."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class R0Config:
    dim: int = 256
    heads: int = 8
    visual_token_layers: int = 2
    pair_layers: int = 2
    set_layers: int = 2
    ffn_dim: int = 1024
    dropout: float = 0.10
    visual_levels: int = 4
    visual_grid_size: int = 3
    visual_region_types: int = 2
    geometry_input_dim: int = 10
    geometry_steps: int = 5


def _finite(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value.float()).all():
        raise FloatingPointError(f"nonfinite R0 {name}")


class R0CandidateVisualEncoder(nn.Module):
    """Encode each candidate's inner/context multi-scale tokens independently."""

    def __init__(self, config: R0Config) -> None:
        super().__init__()
        d = int(config.dim)
        token_count = int(config.visual_levels) * int(config.visual_grid_size) ** 2
        self.token_count = token_count
        self.level_embed = nn.Embedding(int(config.visual_levels), d)
        self.region_embed = nn.Embedding(int(config.visual_region_types), d)
        self.grid_embed = nn.Embedding(int(config.visual_grid_size) ** 2, d)
        self.obs_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.obs_token, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=int(config.heads), dim_feedforward=int(config.ffn_dim),
            dropout=float(config.dropout), activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(config.visual_token_layers))

    def _position(self, device: torch.device) -> torch.Tensor:
        level = torch.arange(self.token_count, device=device) // (int(self.token_count / 4))
        grid = torch.arange(self.token_count, device=device) % (int(self.token_count / 4))
        return self.level_embed(level) + self.grid_embed(grid)

    def forward(self, inner_tokens: torch.Tensor, context_tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inner_tokens.ndim != 3 or context_tokens.ndim != 3:
            raise ValueError("R0 visual inputs must be [N,T,D]")
        expected = self.token_count
        if inner_tokens.shape != (inner_tokens.shape[0], expected, self.obs_token.shape[-1]):
            raise ValueError(f"inner token shape drift: {tuple(inner_tokens.shape)}")
        if context_tokens.shape != inner_tokens.shape:
            raise ValueError("inner/context token shape mismatch")
        _finite(inner_tokens, "inner tokens"); _finite(context_tokens, "context tokens")
        n = int(inner_tokens.shape[0])
        position = self._position(inner_tokens.device)
        # The fixed order is inner level0..3 followed by context level0..3.
        inner = inner_tokens + position.unsqueeze(0) + self.region_embed.weight[0]
        context = context_tokens + position.unsqueeze(0) + self.region_embed.weight[1]
        tokens = torch.cat((self.obs_token.expand(n, -1, -1), inner, context), dim=1)
        encoded = self.encoder(tokens)
        visual_tokens = encoded[:, 1:]
        visual_summary = encoded[:, 0]
        _finite(visual_tokens, "encoded visual tokens"); _finite(visual_summary, "visual summary")
        if visual_tokens.shape != (n, 2 * expected, self.obs_token.shape[-1]):
            raise AssertionError("R0 encoded visual shape drift")
        return visual_tokens, visual_summary


class R0TrackGeometryEncoder(nn.Module):
    def __init__(self, config: R0Config) -> None:
        super().__init__()
        d = int(config.dim)
        self.input_proj = nn.Sequential(
            nn.Linear(int(config.geometry_input_dim), 128), nn.GELU(), nn.Linear(128, d)
        )
        self.gru = nn.GRU(d, d, num_layers=1, batch_first=True)

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        expected = int(self.gru.input_size)
        if geometry.ndim != 3 or geometry.shape[-1] != 10:
            raise ValueError(f"R0 geometry must be [N,5,10], got {tuple(geometry.shape)}")
        if geometry.shape[1] != 5:
            raise ValueError(f"R0 geometry history length drift: {tuple(geometry.shape)}")
        _finite(geometry, "geometry")
        projected = self.input_proj(geometry.float())
        _output, hidden = self.gru(projected)
        summary = hidden[-1]
        if summary.shape[-1] != expected:
            raise AssertionError("R0 geometry hidden size drift")
        _finite(summary, "geometry summary")
        return summary


class R0PairwiseCorrespondenceBlock(nn.Module):
    def __init__(self, config: R0Config) -> None:
        super().__init__()
        d = int(config.dim)
        h = int(config.heads)
        self.seed_norm = nn.LayerNorm(d)
        self.text_cross = nn.MultiheadAttention(d, h, dropout=float(config.dropout), batch_first=True)
        self.visual_cross = nn.MultiheadAttention(d, h, dropout=float(config.dropout), batch_first=True)
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, int(config.ffn_dim)), nn.GELU(), nn.Dropout(float(config.dropout)), nn.Linear(int(config.ffn_dim), d)
        )
        self.dropout = nn.Dropout(float(config.dropout))

    def forward(
        self,
        seed: torch.Tensor,
        visual_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        if seed.ndim != 3 or visual_tokens.ndim != 3 or text_tokens.ndim != 3:
            raise ValueError("R0 pair block rank mismatch")
        q_count, n_count, dim = seed.shape
        if dim != self.seed_norm.normalized_shape[0] or visual_tokens.shape[:2] != (n_count, 72):
            raise ValueError("R0 pair visual/seed shape mismatch")
        if text_tokens.shape[0] != q_count or text_tokens.shape[2] != dim:
            raise ValueError("R0 pair text shape mismatch")
        if text_mask.shape != text_tokens.shape[:2] or text_mask.dtype != torch.bool:
            raise ValueError("R0 text mask shape/type mismatch")
        _finite(seed, "pair seed"); _finite(visual_tokens, "pair visual tokens"); _finite(text_tokens, "pair text tokens")
        batch = q_count * n_count
        query = seed.reshape(batch, 1, dim)
        text = text_tokens[:, None].expand(q_count, n_count, -1, -1).reshape(batch, text_tokens.shape[1], dim)
        text_padding = (~text_mask[:, None].expand(q_count, n_count, -1)).reshape(batch, text_tokens.shape[1])
        attended, _ = self.text_cross(self.seed_norm(query), text, text, key_padding_mask=text_padding)
        state = query + self.dropout(attended)
        visual = visual_tokens[None].expand(q_count, -1, -1, -1).reshape(batch, 72, dim)
        visual_attended, _ = self.visual_cross(self.seed_norm(state), visual, visual)
        state = state + self.dropout(visual_attended)
        state = state + self.dropout(self.ffn(self.ffn_norm(state)))
        result = state[:, 0].reshape(q_count, n_count, dim)
        _finite(result, "pair state")
        return result


class R0CandidateSetReasoner(nn.Module):
    def __init__(self, config: R0Config) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=int(config.dim), nhead=int(config.heads), dim_feedforward=int(config.ffn_dim),
            dropout=float(config.dropout), activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(config.set_layers))

    def forward(self, pair_state: torch.Tensor, box_embed: torch.Tensor) -> torch.Tensor:
        if pair_state.ndim != 3 or box_embed.ndim != 2 or pair_state.shape[1:] != (box_embed.shape[0], box_embed.shape[1]):
            raise ValueError("R0 set input shape mismatch")
        result = self.encoder(pair_state + box_embed.unsqueeze(0))
        _finite(result, "set state")
        return result


class R0TrackGroundingHead(nn.Module):
    """Frozen-observation TCGH sidecar with complete candidate-set emission."""

    def __init__(self, config: R0Config | None = None) -> None:
        super().__init__()
        self.config = config or R0Config()
        d = int(self.config.dim)
        self.visual_encoder = R0CandidateVisualEncoder(self.config)
        self.geometry_encoder = R0TrackGeometryEncoder(self.config)
        self.box_proj = nn.Sequential(nn.Linear(4, 128), nn.GELU(), nn.Linear(128, d))
        self.pair_decoder = nn.ModuleList(
            [R0PairwiseCorrespondenceBlock(self.config) for _ in range(int(self.config.pair_layers))]
        )
        self.set_reasoner = R0CandidateSetReasoner(self.config)
        self.membership_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(0.05), nn.Linear(d, 1)
        )
        self.coverage_presence_head = nn.Sequential(
            nn.LayerNorm(3 * d), nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1)
        )

    def forward(
        self,
        inner_tokens: torch.Tensor,
        context_tokens: torch.Tensor,
        boxes_normalized: torch.Tensor,
        track_geometry: torch.Tensor,
        text_tokens: torch.Tensor,
        text_token_mask: torch.Tensor,
        text_global: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if inner_tokens.ndim != 3 or context_tokens.shape != inner_tokens.shape:
            raise ValueError("R0 visual input shape mismatch")
        n = int(inner_tokens.shape[0]); d = int(self.config.dim)
        if boxes_normalized.shape != (n, 4):
            raise ValueError("R0 box shape mismatch")
        if track_geometry.shape != (n, int(self.config.geometry_steps), int(self.config.geometry_input_dim)):
            raise ValueError("R0 geometry shape mismatch")
        if text_tokens.ndim != 3 or text_tokens.shape[-1] != d:
            raise ValueError("R0 text token shape mismatch")
        q_count, _length, _ = text_tokens.shape
        if text_token_mask.shape != (q_count, text_tokens.shape[1]) or text_token_mask.dtype != torch.bool:
            raise ValueError("R0 text mask shape mismatch")
        if text_global.shape != (q_count, d):
            raise ValueError("R0 text global shape mismatch")
        _finite(boxes_normalized, "boxes"); _finite(text_global, "text global")
        visual_tokens, visual_summary = self.visual_encoder(inner_tokens, context_tokens)
        geometry_summary = self.geometry_encoder(track_geometry)
        box_summary = self.box_proj(boxes_normalized.float())
        _finite(box_summary, "box summary")
        seed = (
            visual_summary.unsqueeze(0)
            + geometry_summary.unsqueeze(0)
            + box_summary.unsqueeze(0)
            + text_global.unsqueeze(1)
        )
        pair_state = seed
        for block in self.pair_decoder:
            pair_state = block(pair_state, visual_tokens, text_tokens, text_token_mask)
        set_state = self.set_reasoner(pair_state, box_summary)
        membership = self.membership_head(set_state).squeeze(-1)
        set_max = set_state.max(dim=1).values
        set_mean = set_state.mean(dim=1)
        presence_input = torch.cat((text_global, set_max, set_mean), dim=-1)
        coverage_presence = self.coverage_presence_head(presence_input).squeeze(-1)
        _finite(membership, "membership logits"); _finite(coverage_presence, "coverage logits")
        if membership.shape != (q_count, n):
            raise AssertionError("R0 membership output shape drift")
        return {
            "membership_logit": membership,
            "coverage_presence_logit": coverage_presence,
            "pair_state": pair_state,
            "set_state": set_state,
            "visual_summary": visual_summary,
            "geometry_summary": geometry_summary,
        }

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)


__all__ = [
    "R0CandidateSetReasoner", "R0CandidateVisualEncoder", "R0Config",
    "R0PairwiseCorrespondenceBlock", "R0TrackGeometryEncoder", "R0TrackGroundingHead",
]
