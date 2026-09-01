"""Small frozen-crop/word-token expression-level probe for L63-C."""
from __future__ import annotations

import torch
from torch import nn


class L63RawRegionLanguageProbe(nn.Module):
    def __init__(self, image_dim=512, text_dim=768, hidden=128, heads=4):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must divide heads")
        self.image_proj = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, hidden))
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden))
        self.crop_to_words = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.set_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2,
                                                     dropout=0.0, batch_first=True,
                                                     norm_first=True)
        self.set_norm = nn.LayerNorm(hidden)
        self.relevance = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.null_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, crop, text, text_valid):
        if crop.dim() != 2 or text.dim() not in (2, 3):
            raise ValueError("crop must be [N,D], text [T,D] or [1,T,D]")
        if text.dim() == 2:
            text = text.unsqueeze(0)
        if text_valid.dim() == 1:
            text_valid = text_valid.unsqueeze(0)
        if text.shape[0] != 1 or text_valid.shape != text.shape[:2]:
            raise ValueError("text batch/mask shape mismatch")
        image = self.image_proj(crop).unsqueeze(1)
        words = self.text_proj(text[0]).unsqueeze(0).expand(crop.shape[0], -1, -1)
        padding = (~text_valid.bool()).expand(crop.shape[0], -1)
        attended, _ = self.crop_to_words(image, words, words, key_padding_mask=padding)
        candidate = image[:, 0] + attended[:, 0]
        competed = self.set_layer(candidate.unsqueeze(0)).squeeze(0)
        x = self.set_norm(candidate + competed)
        score = self.relevance(x).squeeze(-1)
        valid = text_valid[0].bool()
        summary = self.text_proj(text[0, valid]).mean(0) if bool(valid.any()) else self.text_proj(text[0]).mean(0)
        null = self.null_head(x.mean(0) + summary).reshape(())
        return {"relevance_logit": score, "null_logit": null, "candidate_hidden": x}
