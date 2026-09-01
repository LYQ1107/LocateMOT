"""LocateMOT L42 current-frame expression grounding model.

The module deliberately has no source/pool/group/state inputs.  It scores the
complete candidate set of one current frame and keeps the frozen L29 emission
outside the expression branch; the latter is passed only to the training-time
distillation term or to an explicit evaluation control.
"""
from __future__ import annotations

import torch
from torch import nn


class L42CurrentFrameGrounding(nn.Module):
    def __init__(self, image_dim: int = 768, text_dim: int = 768,
                 numeric_dim: int = 36, hidden: int = 128,
                 heads: int = 4, layers: int = 2):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.config = {"image_dim": image_dim, "text_dim": text_dim,
                       "numeric_dim": numeric_dim, "hidden": hidden,
                       "heads": heads, "layers": layers}
        self.image_proj = nn.Sequential(nn.LayerNorm(image_dim),
                                        nn.Linear(image_dim, hidden), nn.GELU())
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim),
                                       nn.Linear(text_dim, hidden), nn.GELU())
        self.cross_layers = nn.ModuleList(
            [nn.MultiheadAttention(hidden, heads, dropout=0.0,
                                   batch_first=True) for _ in range(layers)])
        self.cross_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.numeric_proj = nn.Sequential(nn.LayerNorm(numeric_dim),
                                          nn.Linear(numeric_dim, hidden), nn.GELU())
        self.candidate_fuse = nn.Sequential(nn.Linear(2 * hidden, hidden),
                                             nn.GELU(), nn.LayerNorm(hidden))
        self.set_layers = nn.ModuleList(
            [nn.MultiheadAttention(hidden, heads, dropout=0.0,
                                   batch_first=True) for _ in range(layers)])
        self.set_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.score_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(),
                                        nn.Linear(hidden // 2, 1))
        self.quality_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(),
                                          nn.Linear(hidden // 2, 1))

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=x.dtype).unsqueeze(-1)
        return (x * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def forward(self, patch_tokens: torch.Tensor, text_tokens: torch.Tensor,
                numeric: torch.Tensor, candidate_mask: torch.Tensor,
                text_mask: torch.Tensor, teacher: torch.Tensor | None = None):
        """Score a padded batch of complete current-frame candidate sets.

        Shapes are ``B,N,P,D_img``, ``B,T,D_txt``, ``B,N,F`` and masks
        ``B,N``/``B,T``.  No identifiers are accepted by this interface.
        """
        if patch_tokens.ndim != 4 or text_tokens.ndim != 3:
            raise ValueError("expected patch_tokens [B,N,P,D] and text_tokens [B,T,D]")
        b, n, p, _ = patch_tokens.shape
        if candidate_mask.shape != (b, n) or numeric.shape[:2] != (b, n):
            raise ValueError("candidate set dimensions are not aligned")
        if text_mask.shape[:2] != text_tokens.shape[:2]:
            raise ValueError("text mask is not aligned")

        image = self.image_proj(torch.nan_to_num(patch_tokens.float()))
        query = self.text_proj(torch.nan_to_num(text_tokens.float()))
        # One independent query-to-patch interaction per candidate.
        q = query[:, None].expand(b, n, query.shape[1], query.shape[2])
        q = q.reshape(b * n, query.shape[1], query.shape[2])
        kv = image.reshape(b * n, p, image.shape[-1])
        qmask = text_mask.bool()[:, None].expand(b, n, text_mask.shape[1]).reshape(b * n, -1)
        for attn, norm in zip(self.cross_layers, self.cross_norms):
            z, _ = attn(q, kv, kv, need_weights=False)
            q = norm(q + z)
        qmean = self.masked_mean(q, qmask).reshape(b, n, -1)
        numeric_h = self.numeric_proj(torch.nan_to_num(numeric.float()))
        candidates = self.candidate_fuse(torch.cat((qmean, numeric_h), dim=-1))

        set_values = candidates
        key_padding = ~candidate_mask.bool()
        for attn, norm in zip(self.set_layers, self.set_norms):
            z, _ = attn(set_values, set_values, set_values,
                        key_padding_mask=key_padding, need_weights=False)
            set_values = norm(set_values + z)
        # Padded candidates are kept at a harmless finite value for callers;
        # all losses and output materialization must use candidate_mask.
        scores = self.score_head(set_values).squeeze(-1)
        quality = self.quality_head(set_values).squeeze(-1)
        scores = scores.masked_fill(~candidate_mask.bool(), -20.0)
        quality = quality.masked_fill(~candidate_mask.bool(), -20.0)
        return {"s_expr": scores, "s_teacher": teacher,
                "q_conf": quality, "candidate_features": set_values}

