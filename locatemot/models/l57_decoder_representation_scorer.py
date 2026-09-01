"""Small RMOT-only adapter for frozen GroundingDINO decoder representations."""
from __future__ import annotations

import torch
from torch import nn


class L57DecoderRepresentationScorer(nn.Module):
    """Candidate-set scorer; identifiers and GT-derived features are excluded."""

    def __init__(self, image_dim: int = 256, text_dim: int = 256,
                 entity_dim: int = 256, numeric_dim: int = 24, hidden: int = 128,
                 heads: int = 4) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.image_proj = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, hidden))
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden))
        self.entity_score_proj = nn.Sequential(nn.LayerNorm(entity_dim), nn.Linear(entity_dim, hidden))
        self.numeric_proj = nn.Sequential(nn.LayerNorm(numeric_dim), nn.Linear(numeric_dim, hidden), nn.GELU())
        self.query_to_text = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.set_competition = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.fuse_norm = nn.LayerNorm(hidden)
        self.relevance = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.null_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, decoder_rep: torch.Tensor, text_memory: torch.Tensor,
                text_mask: torch.Tensor, numeric: torch.Tensor,
                entity_scores: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run one complete frame candidate set.

        Args:
            decoder_rep: [N, 256], pooled from all decoder queries.
            text_memory: [1, T, 256] or [T, 256], frozen projected text memory.
            text_mask: [1, T] or [T], True for real text tokens.
            numeric: [N, 24], documented geometry/motion/lifecycle/objectness.
            entity_scores: [N, 256], pooled decoder entity/token scores.
        """
        if text_memory.dim() == 2:
            text_memory = text_memory.unsqueeze(0)
        if text_mask.dim() == 1:
            text_mask = text_mask.unsqueeze(0)
        x = (self.image_proj(decoder_rep) + self.entity_score_proj(entity_scores)
             + self.numeric_proj(numeric))
        text = self.text_proj(text_memory)
        attended, _ = self.query_to_text(
            x.unsqueeze(0), text, text,
            key_padding_mask=~text_mask.bool())
        x = self.fuse_norm(x + attended.squeeze(0))
        competed, _ = self.set_competition(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x = self.fuse_norm(x + competed.squeeze(0))
        score = self.relevance(x).squeeze(-1)
        text_valid = text_mask.bool().unsqueeze(-1)
        text_summary = (text * text_valid).sum(1) / text_valid.sum(1).clamp_min(1)
        null_input = x.mean(0) + text_summary.squeeze(0)
        return {
            "relevance_logit": score,
            "null_logit": self.null_head(null_input).reshape(()),
            "candidate_hidden": x,
        }
