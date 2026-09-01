"""L52 direct query-conditioned region/set probe.

This branch emits a fresh candidate relevance logit.  It is intentionally not
an L29 residual, does not accept identifiers, and processes the complete
same-frame candidate set in one forward pass.
"""
from __future__ import annotations

import torch
from torch import nn


class L52QueryRegionSetProbe(nn.Module):
    def __init__(self, image_dim=768, text_dim=768, frozen_dim=512,
                 numeric_dim=36, hidden=128, heads=4, layers=2):
        super().__init__()
        if hidden % heads or not 1 <= layers <= 2:
            raise ValueError("L52 requires hidden divisible by heads and 1-2 layers")
        self.config = {
            "image_dim": image_dim, "text_dim": text_dim, "frozen_dim": frozen_dim,
            "numeric_dim": numeric_dim, "hidden": hidden, "heads": heads,
            "layers": layers, "direct_relevance": True,
            "candidate_set_is_complete": True, "token_span_region_alignment": "UNALIGNED",
            "static_motion_language_mask": "UNALIGNED/not claimed", "rmot_only": True,
        }
        self.image_proj = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, hidden), nn.GELU())
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden), nn.GELU())
        self.context_proj = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, hidden), nn.GELU())
        self.frozen_proj = nn.Sequential(nn.LayerNorm(frozen_dim), nn.Linear(frozen_dim, hidden), nn.GELU())
        self.numeric_proj = nn.Sequential(nn.LayerNorm(numeric_dim), nn.Linear(numeric_dim, hidden), nn.GELU())
        self.region_to_text = nn.ModuleList([
            nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True)
            for _ in range(layers)
        ])
        self.region_norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.set_attn = nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True)
        self.set_norm = nn.LayerNorm(hidden)
        self.fuse = nn.Sequential(nn.Linear(5 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.logit_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.quality_head = nn.Linear(hidden, 1)

    def forward(self, patch_tokens, context_tokens, text_tokens, text_mask,
                frozen_clip, numeric):
        if patch_tokens.ndim != 3 or context_tokens.ndim != 3 or text_tokens.ndim != 2:
            raise ValueError("expected patch [N,P,D], context [1,P,D], text [T,D]")
        n = patch_tokens.shape[0]
        if n < 1 or context_tokens.shape[0] != 1:
            raise ValueError("candidate set/context misaligned")
        if frozen_clip.shape != (n, 512) or numeric.shape != (n, 36):
            raise ValueError("frozen/numeric candidate streams misaligned")
        if text_mask.shape != (text_tokens.shape[0],):
            raise ValueError("text mask misaligned")
        image = self.image_proj(torch.nan_to_num(patch_tokens.float()))
        text = self.text_proj(torch.nan_to_num(text_tokens.float())).unsqueeze(0).expand(n, -1, -1)
        key_padding = ~text_mask.bool().unsqueeze(0).expand(n, -1)
        # Each candidate's spatial tokens query the complete word-token sequence.
        for attn, norm in zip(self.region_to_text, self.region_norm):
            z, _ = attn(image, text, text, key_padding_mask=key_padding, need_weights=False)
            image = norm(image + z)
        region = image.mean(1)
        context = self.context_proj(torch.nan_to_num(context_tokens.float())).mean(1).expand(n, -1)
        frozen = self.frozen_proj(torch.nan_to_num(frozen_clip.float()))
        numeric_h = self.numeric_proj(torch.nan_to_num(numeric.float()))
        text_summary = (text * text_mask.bool().to(text.dtype).unsqueeze(0).unsqueeze(-1)).sum(1)
        text_summary = text_summary / text_mask.bool().sum().clamp_min(1).to(text.dtype)
        fused = self.fuse(torch.cat((region, context, frozen, numeric_h, text_summary), -1))
        set_z, _ = self.set_attn(fused.unsqueeze(0), fused.unsqueeze(0), fused.unsqueeze(0), need_weights=False)
        fused = self.set_norm(fused + set_z[0])
        return {"relevance_logit": self.logit_head(fused).squeeze(-1),
                "quality_logit": self.quality_head(fused).squeeze(-1),
                "candidate_features": fused}
