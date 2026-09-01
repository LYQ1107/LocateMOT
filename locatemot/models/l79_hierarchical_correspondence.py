"""L79 hierarchical RMOT-only query/region/track correspondence model.

The model consumes complete current-frame candidate sets from the L69 bank,
causal observation histories, frozen word-token features, and private CLIP
multi-scale feature maps.  IDs are used by the data contract to gather a
causal history but are never included in the neural feature tensors.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class L79Config:
    observation_dim: int = 1432
    text_dim: int = 768
    visual_dim: int = 768
    hidden: int = 384
    heads: int = 6
    history_length: int = 16
    query_layers: int = 4
    temporal_layers: int = 2
    set_layers: int = 4
    history_cross_layers: int = 4
    region_cross_layers: int = 2
    feedforward: int = 1536
    roi_grid: int = 4
    visual_taps: int = 3
    lora_rank: int = 32
    lora_alpha: float = 16.0


def sinusoidal_positions(length: int, dim: int) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    result = torch.zeros(length, dim, dtype=torch.float32)
    result[:, 0::2] = torch.sin(position * div)
    result[:, 1::2] = torch.cos(position * div[: result[:, 1::2].shape[1]])
    return result


class CrossAttentionBlock(nn.Module):
    def __init__(self, config: L79Config) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(config.hidden)
        self.key_norm = nn.LayerNorm(config.hidden)
        self.attn = nn.MultiheadAttention(config.hidden, config.heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(config.hidden)
        self.ff = nn.Sequential(
            nn.Linear(config.hidden, config.feedforward),
            nn.GELU(),
            nn.Linear(config.feedforward, config.hidden),
        )

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        normalized_query = self.query_norm(query)
        normalized_key = self.key_norm(key_value)
        value, _weights = self.attn(
            normalized_query, normalized_key, normalized_key,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        query = query + value
        return query + self.ff(self.ff_norm(query))


class L79HierarchicalCorrespondence(nn.Module):
    """Complete-set hierarchical RMOT scorer with no identity-ID shortcut."""

    def __init__(self, config: L79Config | None = None) -> None:
        super().__init__()
        self.config = config or L79Config()
        c = self.config
        if c.observation_dim != 1432 or c.hidden != 384:
            raise ValueError("L79 contract requires observation_dim=1432 and hidden=384")
        if c.history_length != 16 or c.heads != 6:
            raise ValueError("L79 contract requires history_length=16 and six attention heads")

        self.observation_norm = nn.LayerNorm(c.observation_dim)
        self.observation_projection = nn.Linear(c.observation_dim, c.hidden)
        self.register_buffer("time_positions", sinusoidal_positions(c.history_length, c.hidden), persistent=True)
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=c.hidden, nhead=c.heads, dim_feedforward=c.feedforward,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=c.temporal_layers)

        self.text_norm = nn.LayerNorm(c.text_dim)
        self.text_projection = nn.Linear(c.text_dim, c.hidden)
        query_layer = nn.TransformerEncoderLayer(
            d_model=c.hidden, nhead=c.heads, dim_feedforward=c.feedforward,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.query_encoder = nn.TransformerEncoder(query_layer, num_layers=c.query_layers)

        self.visual_projection = nn.ModuleList([nn.Linear(c.visual_dim, c.hidden) for _ in range(c.visual_taps)])
        self.context_projection = nn.Linear(c.visual_dim, c.hidden)
        self.box_projection = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, c.hidden))
        self.history_cross = nn.ModuleList([CrossAttentionBlock(c) for _ in range(c.history_cross_layers)])
        self.region_cross = nn.ModuleList([CrossAttentionBlock(c) for _ in range(c.region_cross_layers)])

        self.latent_slots = nn.Parameter(torch.zeros(4, c.hidden))
        nn.init.normal_(self.latent_slots, mean=0.0, std=0.02)
        self.latent_attention = nn.MultiheadAttention(c.hidden, c.heads, batch_first=True)
        self.latent_norm = nn.LayerNorm(c.hidden)

        set_layer = nn.TransformerEncoderLayer(
            d_model=c.hidden, nhead=c.heads, dim_feedforward=c.feedforward,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True,
        )
        self.track_set_encoder = nn.TransformerEncoder(set_layer, num_layers=c.set_layers)
        self.candidate_norm = nn.LayerNorm(c.hidden)
        self.frame_membership_head = nn.Linear(c.hidden, 1)
        self.track_relevance_head = nn.Linear(c.hidden, 1)
        self.observation_quality_head = nn.Linear(c.hidden, 1)
        self.continuation_head = nn.Linear(c.hidden, 1)
        self.null_head = nn.Sequential(
            nn.LayerNorm(c.hidden * 2),
            nn.Linear(c.hidden * 2, c.hidden),
            nn.GELU(),
            nn.Linear(c.hidden, 1),
        )

    @staticmethod
    def _masked_mean(value: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
        weight = mask.to(value.dtype)
        while weight.ndim < value.ndim:
            weight = weight.unsqueeze(-1)
        total = (value * weight).sum(dim=dim)
        denom = weight.sum(dim=dim).clamp_min(1.0)
        return total / denom

    @staticmethod
    def _box_grid(boxes: torch.Tensor, grid_size: int, context_scale: float = 1.0) -> torch.Tensor:
        # boxes are normalized xyxy in the full-frame coordinate system.  The
        # fixed lattice preserves every candidate; no proposal selection occurs.
        boxes = boxes.clamp(0.0, 1.0)
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        half_w, half_h = (x2 - x1) * 0.5 * context_scale, (y2 - y1) * 0.5 * context_scale
        left, right = (cx - half_w).clamp(0.0, 1.0), (cx + half_w).clamp(0.0, 1.0)
        top, bottom = (cy - half_h).clamp(0.0, 1.0), (cy + half_h).clamp(0.0, 1.0)
        frac = (torch.arange(grid_size, device=boxes.device, dtype=boxes.dtype) + 0.5) / grid_size
        x = left[:, None] + (right - left)[:, None] * frac[None, :]
        y = top[:, None] + (bottom - top)[:, None] * frac[None, :]
        # Expand the two per-candidate axes explicitly.  A generic
        # ``torch.meshgrid(y, x)`` would incorrectly mix the candidate axis
        # with the lattice axes when K != grid_size.
        xx = x[:, None, :].expand(-1, grid_size, -1)
        yy = y[:, :, None].expand(-1, -1, grid_size)
        return torch.stack((xx, yy), dim=-1).reshape(boxes.shape[0], grid_size * grid_size, 2)

    def _roi_tokens(self, vision_pyramid: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Pool ROI and outside-context tokens from all `[S,1,196,768]` taps."""
        if vision_pyramid.ndim != 4 or vision_pyramid.shape[0] != self.config.visual_taps or vision_pyramid.shape[2:] != (196, 768):
            raise ValueError(f"L79 visual pyramid must be [3,B,196,768], got {tuple(vision_pyramid.shape)}")
        if vision_pyramid.shape[1] != 1:
            raise ValueError("L79 unit forward expects one full-frame image")
        grid_size = self.config.roi_grid
        roi_grid = self._box_grid(boxes, grid_size, 1.0).reshape(-1, grid_size, grid_size, 2) * 2.0 - 1.0
        context_grid = self._box_grid(boxes, grid_size, 1.5).reshape(-1, grid_size, grid_size, 2) * 2.0 - 1.0
        projected_levels = []
        global_levels = []
        for level, projection in enumerate(self.visual_projection):
            fmap = vision_pyramid[level, 0].reshape(14, 14, 768).permute(2, 0, 1).unsqueeze(0)
            count = boxes.shape[0]
            fmap = fmap.expand(count, -1, -1, -1)
            inside = F.grid_sample(
                fmap,
                roi_grid.to(dtype=fmap.dtype),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            outside = F.grid_sample(
                fmap,
                context_grid.to(dtype=fmap.dtype),
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            inside = inside.permute(0, 2, 3, 1).reshape(count, grid_size * grid_size, 768)
            outside = outside.permute(0, 2, 3, 1).reshape(count, grid_size * grid_size, 768)
            projected_levels.append(projection(torch.cat((inside, outside), dim=1).float()))
            global_levels.append(fmap[:, :, :, :].mean(dim=(2, 3)).float())
        region_tokens = torch.cat(projected_levels, dim=1)
        global_context = self.context_projection(torch.stack(global_levels, dim=0).mean(dim=0))
        if not bool(torch.isfinite(region_tokens).all()) or not bool(torch.isfinite(global_context).all()):
            raise FloatingPointError("nonfinite L79 ROI/context representation")
        return region_tokens, global_context

    def forward(
        self,
        observations: torch.Tensor,
        history_observations: torch.Tensor,
        history_mask: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        boxes_norm: torch.Tensor,
        vision_pyramid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Score one complete `(video, query, frame)` candidate set."""
        if observations.ndim != 2 or observations.shape[-1] != self.config.observation_dim:
            raise ValueError(f"observations must be [N,1432], got {tuple(observations.shape)}")
        if history_observations.ndim != 3 or history_observations.shape[:2] != history_mask.shape:
            raise ValueError("history/mask shape mismatch")
        if history_observations.shape[1] != self.config.history_length:
            raise ValueError("history length drift")
        if text_tokens.ndim != 2 or text_tokens.shape[0] != text_mask.shape[0]:
            raise ValueError("text/mask shape mismatch")
        if boxes_norm.shape != (observations.shape[0], 4):
            raise ValueError("candidate/box shape mismatch")
        if observations.shape[0] == 0:
            raise ValueError("L79 never accepts an empty candidate set")
        if not bool(history_mask.any(dim=1).all()):
            raise ValueError("each L79 current row must have its current observation in history")
        if not bool(text_mask.any()):
            raise ValueError("L79 text mask has no valid token")

        count = observations.shape[0]
        obs = self.observation_projection(self.observation_norm(observations.float()))
        history = self.observation_projection(self.observation_norm(history_observations.float()))
        history = history + self.time_positions.to(history.device, history.dtype)[None, :, :]
        causal = torch.triu(torch.ones(self.config.history_length, self.config.history_length, device=history.device, dtype=torch.bool), diagonal=1)
        temporal = self.temporal_encoder(history, mask=causal, src_key_padding_mask=~history_mask.bool())
        track_base = self._masked_mean(temporal, history_mask.bool(), dim=1)

        query = self.text_projection(self.text_norm(text_tokens.float())).unsqueeze(0)
        query = self.query_encoder(query, src_key_padding_mask=~text_mask.bool().unsqueeze(0)).squeeze(0)
        query_expanded = query.unsqueeze(0).expand(count, -1, -1)
        history_query = query_expanded
        for block in self.history_cross:
            history_query = block(history_query, temporal, key_padding_mask=~history_mask.bool())
        history_query_vector = self._masked_mean(history_query, text_mask.bool()[None, :].expand(count, -1), dim=1)

        region_tokens, global_context = self._roi_tokens(vision_pyramid, boxes_norm.float())
        region_query = query_expanded
        for block in self.region_cross:
            region_query = block(region_query, region_tokens)
        region_query_vector = self._masked_mean(region_query, text_mask.bool()[None, :].expand(count, -1), dim=1)

        query_mean = self._masked_mean(query, text_mask.bool(), dim=0)
        slots = self.latent_slots.unsqueeze(0) + query_mean.unsqueeze(0).unsqueeze(1)
        slots, _slot_weights = self.latent_attention(
            self.latent_norm(slots), query.unsqueeze(0), query.unsqueeze(0),
            key_padding_mask=~text_mask.bool().unsqueeze(0), need_weights=False,
        )
        latent_context = slots.mean(dim=1).expand(count, -1)
        candidate = self.candidate_norm(
            track_base + history_query_vector + region_query_vector + global_context
            + self.box_projection(boxes_norm.float()) + latent_context
        )
        set_value = self.track_set_encoder(candidate.unsqueeze(0)).squeeze(0)
        frame_logits = self.frame_membership_head(set_value).squeeze(-1)
        track_logits = self.track_relevance_head(set_value).squeeze(-1)
        quality_logits = self.observation_quality_head(set_value).squeeze(-1)
        continuation_logits = self.continuation_head(set_value).squeeze(-1)
        set_mean = set_value.mean(dim=0)
        null_logit = self.null_head(torch.cat((query_mean, set_mean), dim=-1)).squeeze(-1)
        outputs = {
            "frame_membership_logits": frame_logits,
            "track_relevance_logits": track_logits,
            "observation_quality_logits": quality_logits,
            "continuation_logits": continuation_logits,
            "null_logit": null_logit,
            "query_vector": query_mean,
            "track_vector": set_value,
            "current_vector": obs,
            "region_tokens": region_tokens,
            "history_query_vector": history_query_vector,
            "region_query_vector": region_query_vector,
        }
        nonfinite = [name for name, value in outputs.items() if not bool(torch.isfinite(value.float()).all())]
        if nonfinite:
            raise FloatingPointError(f"nonfinite L79 output fields: {nonfinite}")
        return outputs

    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def trainable_parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad))


__all__ = ["L79Config", "L79HierarchicalCorrespondence"]
