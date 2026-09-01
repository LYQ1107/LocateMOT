"""Small frozen-CLIP-joint-space RMOT candidate-set probe for L65."""
from __future__ import annotations

import torch
from torch import nn


class L65ClipJointSet(nn.Module):
    def __init__(self, image_dim=512, text_dim=512, numeric_dim=32, hidden=128, heads=4):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must divide heads")
        self.image_proj = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, hidden))
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden))
        self.numeric_proj = nn.Sequential(nn.LayerNorm(numeric_dim), nn.Linear(numeric_dim, hidden), nn.GELU())
        self.patch_to_words = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.set_layer = nn.TransformerEncoderLayer(hidden, heads, hidden * 2, dropout=0.0, batch_first=True, norm_first=True)
        self.set_norm = nn.LayerNorm(hidden)
        self.relevance = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.null_head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, patch_joint, text_joint, text_valid, numeric):
        if patch_joint.dim() != 3 or patch_joint.shape[-1] != 512:
            raise ValueError(f"patch_joint must be [N,P,512], got {tuple(patch_joint.shape)}")
        if text_joint.dim() != 2 or text_joint.shape[-1] != 512:
            raise ValueError(f"text_joint must be [T,512], got {tuple(text_joint.shape)}")
        if numeric.dim() != 2 or numeric.shape[0] != patch_joint.shape[0] or numeric.shape[-1] != 32:
            raise ValueError("numeric/candidate alignment mismatch")
        if text_valid.shape != text_joint.shape[:1]:
            raise ValueError("text mask mismatch")
        image = self.image_proj(patch_joint)
        words = self.text_proj(text_joint).unsqueeze(0).expand(patch_joint.shape[0], -1, -1)
        padding = (~text_valid.bool()).unsqueeze(0).expand(patch_joint.shape[0], -1)
        attended, _ = self.patch_to_words(image, words, words, key_padding_mask=padding)
        candidate = image.mean(1) + attended.mean(1) + self.numeric_proj(numeric)
        competed = self.set_layer(candidate.unsqueeze(0)).squeeze(0)
        x = self.set_norm(candidate + competed)
        score = self.relevance(x).squeeze(-1)
        t = self.text_proj(text_joint[text_valid.bool()]).mean(0) if bool(text_valid.bool().any()) else self.text_proj(text_joint).mean(0)
        null = self.null_head(torch.cat((x.mean(0), t), 0)).reshape(())
        return {"relevance_logit": score, "null_logit": null, "candidate_hidden": x}
