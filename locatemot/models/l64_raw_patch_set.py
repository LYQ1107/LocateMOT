"""Small RMOT-only raw patch/text candidate-set probe for L64."""
from __future__ import annotations

import torch
from torch import nn


class L64RawPatchSet(nn.Module):
    def __init__(self, image_dim=768, text_dim=512, numeric_dim=32, hidden=128, heads=4):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must divide heads")
        self.image_proj = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, hidden))
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden))
        self.numeric_proj = nn.Sequential(nn.LayerNorm(numeric_dim), nn.Linear(numeric_dim, hidden), nn.GELU())
        self.patch_to_words = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.candidate_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout=0.0, batch_first=True, norm_first=True)
        self.set_norm = nn.LayerNorm(hidden)
        self.relevance = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.null_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, patches, text, text_valid, numeric):
        if patches.dim() != 3 or text.dim() != 2 or numeric.dim() != 2:
            raise ValueError("patches [N,P,D], text [T,D], numeric [N,F]")
        if patches.shape[0] != numeric.shape[0] or text.shape[-1] != self.text_proj[1].in_features:
            raise ValueError("candidate/text dimension mismatch")
        image = self.image_proj(patches)
        words = self.text_proj(text).unsqueeze(0).expand(patches.shape[0], -1, -1)
        mask = (~text_valid.bool()).unsqueeze(0).expand(patches.shape[0], -1)
        attended, _ = self.patch_to_words(image, words, words, key_padding_mask=mask)
        candidate = image.mean(1) + attended.mean(1) + self.numeric_proj(numeric)
        competed = self.candidate_layer(candidate.unsqueeze(0)).squeeze(0)
        x = self.set_norm(candidate + competed)
        score = self.relevance(x).squeeze(-1)
        valid_text = text_valid.bool()
        t = self.text_proj(text[valid_text]).mean(0) if bool(valid_text.any()) else self.text_proj(text).mean(0)
        null = self.null_head(torch.cat((x.mean(0), t), 0)).reshape(())
        return {"relevance_logit": score, "null_logit": null, "candidate_hidden": x}
