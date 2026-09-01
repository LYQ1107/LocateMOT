"""Compact RMOT-only matcher for candidate-marked LocateAnything states."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn


class CandidateMarkedVLMMatcher(nn.Module):
    """Score a complete candidate batch from marked VLM expression states.

    The large visual/language model is intentionally not owned by this class;
    its frozen language states are supplied by the L75 runtime.  This makes
    the trainable state auditable and keeps the checkpoint adapter-only.
    """

    def __init__(self, visual_dim: int = 2048, language_dim: int = 2048,
                 hidden: int = 256, heads: int = 4, marker_std: float = 0.01):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.visual_dim = int(visual_dim)
        self.language_dim = int(language_dim)
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.marker_std = float(marker_std)
        self.region_marker = nn.Parameter(torch.empty(self.visual_dim, dtype=torch.float32))
        nn.init.normal_(self.region_marker, mean=0.0, std=self.marker_std)
        self.image_norm = nn.LayerNorm(self.visual_dim)
        self.image_proj = nn.Linear(self.visual_dim, self.hidden)
        self.text_norm = nn.LayerNorm(self.language_dim)
        self.text_proj = nn.Linear(self.language_dim, self.hidden)
        self.word_to_region = nn.MultiheadAttention(
            self.hidden, self.heads, dropout=0.0, batch_first=True
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.hidden * 3),
            nn.Linear(self.hidden * 3, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.match_head = nn.Sequential(
            nn.LayerNorm(self.hidden), nn.Linear(self.hidden, 1)
        )
        self.absent_head = nn.Sequential(
            nn.LayerNorm(self.hidden), nn.Linear(self.hidden, 1)
        )

    def forward(self, final_hidden: torch.Tensor, expression_positions: list[int],
                region_values: torch.Tensor, region_mask: torch.Tensor,
                return_audit: bool = False) -> dict[str, torch.Tensor]:
        if final_hidden.ndim != 3 or final_hidden.shape[-1] != self.language_dim:
            raise ValueError(f"final_hidden must be [B,S,{self.language_dim}]")
        if not expression_positions:
            raise ValueError("expression position list is empty")
        if region_values.ndim != 3 or region_values.shape[0] != final_hidden.shape[0]:
            raise ValueError("region values/final hidden batch mismatch")
        if region_values.shape[-1] != self.visual_dim:
            raise ValueError(f"region values must end in {self.visual_dim}")
        if region_mask.shape != region_values.shape[:2]:
            raise ValueError("region mask shape mismatch")
        text = final_hidden.index_select(
            1, torch.as_tensor(expression_positions, device=final_hidden.device, dtype=torch.long)
        ).float()
        region = region_values.float()
        text_h = self.text_proj(self.text_norm(text))
        region_h = self.image_proj(self.image_norm(region))
        valid = region_mask.bool()
        if not bool(valid.any(dim=1).all()):
            raise ValueError("region_mask contains an all-invalid candidate")
        attended, attention = self.word_to_region(
            text_h, region_h, region_h, key_padding_mask=~valid
        )
        text_pool = attended.mean(dim=1)
        region_pool = (region_h * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1).float()
        fused = self.fusion(torch.cat([text_pool, region_pool, text_pool * region_pool], dim=-1))
        match_logit = self.match_head(fused).squeeze(-1)
        # This is diagnostic only.  The L75 semantic gate must not use it as a
        # frame-wide suppression rule.
        absent_logit = self.absent_head(text_pool).squeeze(-1)
        result = {
            "match_logit": match_logit,
            "absent_logit": absent_logit,
            "query_state": text_pool,
            "region_state": region_pool,
        }
        if return_audit:
            result.update({
                "cross_attention": attention,
                "expression_hidden": text,
            })
        return result

    def parameter_contract(self) -> dict[str, Any]:
        return {
            "visual_dim": self.visual_dim,
            "language_dim": self.language_dim,
            "hidden": self.hidden,
            "heads": self.heads,
            "marker_shape": list(self.region_marker.shape),
            "marker_init_std": self.marker_std,
            "parameter_count": sum(parameter.numel() for parameter in self.parameters()),
            "trainable_parameter_count": sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad),
            "token_span_alignment": "UNALIGNED",
            "semantic_inputs": ["candidate_marked_visual_values", "final_expression_token_hidden"],
            "forbidden_inputs": ["source_id", "pool_id", "group_id", "query_id", "track_id", "old_scores", "GT_identity"],
        }
