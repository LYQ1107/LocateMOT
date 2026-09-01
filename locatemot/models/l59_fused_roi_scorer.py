"""Compact scorer for post-fusion GroundingDINO candidate ROI tokens.

The detector and its fused memory are frozen.  This module consumes complete
candidate sets and has no identifier/source features.
"""
from __future__ import annotations

import torch
from torch import nn


class L59FusedROIScorer(nn.Module):
    def __init__(self, image_dim=256, text_dim=256, numeric_dim=24,
                 hidden=128, heads=4):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.roi_proj = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, hidden))
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden))
        self.numeric_proj = nn.Sequential(nn.LayerNorm(numeric_dim), nn.Linear(numeric_dim, hidden), nn.GELU())
        self.roi_to_text = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.set_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2,
                                                    dropout=0.0, batch_first=True,
                                                    norm_first=True)
        self.set_norm = nn.LayerNorm(hidden)
        self.relevance = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.null_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, roi_tokens, text_memory, text_mask, numeric):
        if roi_tokens.dim() != 3 or text_memory.dim() not in (2, 3):
            raise ValueError("unexpected ROI/text rank")
        if text_memory.dim() == 2:
            text_memory = text_memory.unsqueeze(0)
        if text_mask.dim() == 1:
            text_mask = text_mask.unsqueeze(0)
        roi = self.roi_proj(roi_tokens)
        text = self.text_proj(text_memory).expand(roi.shape[0], -1, -1)
        text_pad = (~text_mask.bool()).expand(roi.shape[0], -1)
        attn, _ = self.roi_to_text(roi, text, text, key_padding_mask=text_pad)
        x = self.set_norm(roi.mean(1) + attn.mean(1) + self.numeric_proj(numeric))
        x = self.set_norm(x + self.set_layer(x.unsqueeze(0)).squeeze(0))
        score = self.relevance(x).squeeze(-1)
        valid = text_mask.bool().unsqueeze(-1)
        # The text tensor is expanded per candidate for cross-attention.  The
        # NULL state is query/frame-level, so summarize the unexpanded text
        # memory once; otherwise N>1 candidates produce an [N,H] input to a
        # scalar head and reshape fails.
        text_base = self.text_proj(text_memory)
        text_valid = text_mask.bool().unsqueeze(-1)
        text_summary = (text_base * text_valid).sum((0, 1)) / text_valid.sum().clamp_min(1)
        null_logit = self.null_head(x.mean(0) + text_summary).reshape(())
        return {"relevance_logit": score, "null_logit": null_logit,
                "candidate_hidden": x}
