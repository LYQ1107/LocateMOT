"""L80 raw-image regional correspondence model.

This module is intentionally independent from the L78/L79 scorers.  Frozen
CLIP provides spatial visual tokens; the trainable part performs
query-to-region cross-attention before one complete same-frame set operation.
Track IDs are used outside this module only to assemble causal history rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class L80Config:
    visual_dim: int = 768
    text_dim: int = 768
    observation_dim: int = 1432
    hidden: int = 256
    heads: int = 4
    region_scales: int = 3
    tokens_per_scale: int = 21
    history_length: int = 8
    dropout: float = 0.05


class L80RawRegionCorrespondence(nn.Module):
    """Query-conditioned regional/set head with causal observation memory."""

    def __init__(self, config: L80Config | None = None) -> None:
        super().__init__()
        self.config = config or L80Config()
        c = self.config
        if c.hidden % c.heads:
            raise ValueError("hidden must be divisible by heads")
        self.visual_adapters = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(c.visual_dim), nn.Linear(c.visual_dim, c.hidden))
            for _ in range(c.region_scales)
        ])
        self.text_proj = nn.Sequential(nn.LayerNorm(c.text_dim), nn.Linear(c.text_dim, c.hidden))
        self.observation_proj = nn.Sequential(nn.LayerNorm(c.observation_dim), nn.Linear(c.observation_dim, c.hidden))
        self.time_proj = nn.Linear(1, c.hidden)
        self.history_gru = nn.GRU(c.hidden, c.hidden, num_layers=1, batch_first=True)
        self.query_to_region = nn.MultiheadAttention(
            c.hidden, c.heads, dropout=c.dropout, batch_first=True)
        self.region_norm = nn.LayerNorm(c.hidden)
        self.fusion = nn.Sequential(
            nn.Linear(4 * c.hidden, 2 * c.hidden), nn.GELU(),
            nn.LayerNorm(2 * c.hidden), nn.Linear(2 * c.hidden, c.hidden), nn.GELU(),
        )
        set_layer = nn.TransformerEncoderLayer(
            d_model=c.hidden, nhead=c.heads, dim_feedforward=4 * c.hidden,
            dropout=c.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.same_frame_set = nn.TransformerEncoder(set_layer, num_layers=1)
        self.set_norm = nn.LayerNorm(c.hidden)
        self.membership_head = nn.Sequential(
            nn.Linear(2 * c.hidden, c.hidden), nn.GELU(), nn.Linear(c.hidden, 1))
        self.track_head = nn.Sequential(
            nn.Linear(2 * c.hidden, c.hidden // 2), nn.GELU(), nn.Linear(c.hidden // 2, 1))
        self.continuation_head = nn.Sequential(
            nn.Linear(c.hidden, c.hidden // 2), nn.GELU(), nn.Linear(c.hidden // 2, 1))
        self.quality_head = nn.Sequential(
            nn.Linear(c.hidden, c.hidden // 2), nn.GELU(), nn.Linear(c.hidden // 2, 1))
        self.null_head = nn.Sequential(
            nn.Linear(2 * c.hidden, c.hidden // 2), nn.GELU(), nn.Linear(c.hidden // 2, 1))
        self.cardinality_head = nn.Sequential(
            nn.Linear(2 * c.hidden, c.hidden // 2), nn.GELU(), nn.Linear(c.hidden // 2, 1))

    @staticmethod
    def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.bool().to(dtype=value.dtype).unsqueeze(-1)
        return (value * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    def _visual(self, region_tokens: torch.Tensor) -> torch.Tensor:
        c = self.config
        expected = c.region_scales * c.tokens_per_scale
        if region_tokens.ndim != 3 or region_tokens.shape[1:] != (expected, c.visual_dim):
            raise ValueError(f"expected visual tokens [N,{expected},{c.visual_dim}], got {tuple(region_tokens.shape)}")
        pieces = []
        for index, adapter in enumerate(self.visual_adapters):
            begin = index * c.tokens_per_scale
            end = begin + c.tokens_per_scale
            pieces.append(adapter(region_tokens[:, begin:end].float()))
        return self.region_norm(torch.cat(pieces, dim=1))

    def _history(self, observations: torch.Tensor, history_mask: torch.Tensor,
                 history_frame_ids: torch.Tensor, current_frame: int) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.config
        if observations.ndim != 3 or observations.shape[-1] != c.observation_dim:
            raise ValueError(f"expected observations [N,H,{c.observation_dim}], got {tuple(observations.shape)}")
        if history_mask.shape[:2] != observations.shape[:2] or history_frame_ids.shape != history_mask.shape:
            raise ValueError("history shape mismatch")
        mask = history_mask.bool()
        if bool((history_frame_ids[mask] > int(current_frame)).any()):
            raise ValueError("future history passed to L80 model")
        value = self.observation_proj(observations.float())
        denom = max(1.0, float(current_frame) + 1.0)
        time = (history_frame_ids.float().clamp_min(0.0) / denom).unsqueeze(-1)
        value = value + self.time_proj(time)
        value = value.masked_fill(~mask.unsqueeze(-1), 0.0)
        encoded, _ = self.history_gru(value)
        encoded = encoded.masked_fill(~mask.unsqueeze(-1), 0.0)
        lengths = mask.long().sum(dim=1).clamp_min(1) - 1
        selected = encoded[torch.arange(encoded.shape[0], device=encoded.device), lengths]
        return encoded, selected

    def forward(self, region_tokens: torch.Tensor, text_tokens: torch.Tensor,
                text_mask: torch.Tensor, observations: torch.Tensor,
                history_mask: torch.Tensor, history_frame_ids: torch.Tensor,
                current_frame: int) -> dict[str, torch.Tensor]:
        if region_tokens.shape[0] == 0:
            raise ValueError("L80 requires at least one complete candidate row")
        if text_tokens.ndim == 3:
            if text_tokens.shape[0] != 1:
                raise ValueError("one expression sequence is expected per unit")
            text_tokens = text_tokens[0]
        if text_tokens.ndim != 2 or text_tokens.shape[-1] != self.config.text_dim:
            raise ValueError(f"text shape mismatch {tuple(text_tokens.shape)}")
        text_mask = text_mask.bool().reshape(-1)
        if text_mask.numel() != text_tokens.shape[0] or not bool(text_mask.any()):
            raise ValueError("text mask mismatch/empty")
        visual = self._visual(region_tokens)
        qtok = self.text_proj(text_tokens.float()).unsqueeze(0)
        query = qtok.expand(region_tokens.shape[0], -1, -1)
        key_padding = None  # all regional tokens are retained and valid
        cross, attention = self.query_to_region(query, visual, visual,
                                                key_padding_mask=key_padding,
                                                need_weights=True, average_attn_weights=False)
        q_mask = text_mask.unsqueeze(0).expand(region_tokens.shape[0], -1)
        q_region = self.masked_mean(cross, q_mask)
        q_global = self.masked_mean(query, q_mask)
        region_mean = visual.mean(dim=1)
        _history_tokens, history = self._history(
            observations, history_mask, history_frame_ids, current_frame)
        fused = self.fusion(torch.cat((q_region, region_mean, history, q_global), dim=-1))
        set_value = self.same_frame_set(fused.unsqueeze(0))[0]
        set_value = self.set_norm(set_value)
        set_summary = set_value.mean(dim=0)
        q_for_candidate = q_global
        pair = torch.cat((set_value, q_for_candidate), dim=-1)
        frame_pair = torch.cat((set_summary, q_global[0]), dim=-1)
        candidate_logits = self.membership_head(pair).squeeze(-1)
        track_logits = self.track_head(pair).squeeze(-1)
        continuation_logits = self.continuation_head(set_value).squeeze(-1)
        quality_logits = self.quality_head(set_value).squeeze(-1)
        null_logit = self.null_head(frame_pair).squeeze()
        cardinality_logit = self.cardinality_head(frame_pair).squeeze()
        return {
            "candidate_logits": candidate_logits,
            "track_logits": track_logits,
            "continuation_logits": continuation_logits,
            "quality_logits": quality_logits,
            "null_logit": null_logit,
            "cardinality_logit": cardinality_logit,
            "query_vector": q_global[0],
            "candidate_vector": set_value,
            "cross_attention": attention,
        }

    def parameter_report(self) -> dict[str, Any]:
        trainable = [(name, int(parameter.numel()), str(parameter.dtype))
                     for name, parameter in self.named_parameters() if parameter.requires_grad]
        return {
            "config": self.config.__dict__,
            "trainable_parameter_count": int(sum(item[1] for item in trainable)),
            "total_parameter_count": int(sum(parameter.numel() for parameter in self.parameters())),
            "trainable_parameters": [item[0] for item in trainable],
            "trainable_parameter_dtypes": {item[0]: item[2] for item in trainable},
            "forbidden_semantic_inputs": ["source_id", "pool_id", "group_id", "query_id", "track_id", "state_key", "old_scores"],
            "history_ids_used_only_for_causal_row_assembly": True,
        }


__all__ = ["L80Config", "L80RawRegionCorrespondence"]
