"""L51 RMOT-only streaming crop/token adapter.

The module has no identifier or tracker inputs.  L29 is the immutable base
emission; the trainable branch can only add a bounded residual.
"""
from __future__ import annotations

import torch
from torch import nn


class L51StreamingCropAdapter(nn.Module):
    def __init__(self, image_dim: int = 768, text_dim: int = 768,
                 frozen_dim: int = 512, numeric_dim: int = 36,
                 hidden: int = 128, heads: int = 4, layers: int = 2,
                 residual_bound: float = 0.05):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        if layers < 1 or layers > 2:
            raise ValueError("B0 permits one or two cross-attention layers")
        self.config = {
            "image_dim": image_dim, "text_dim": text_dim,
            "frozen_dim": frozen_dim, "numeric_dim": numeric_dim,
            "hidden": hidden, "heads": heads, "layers": layers,
            "residual_bound": float(residual_bound),
            "token_level_alignment": "UNALIGNED",
            "static_motion_language_mask": "UNALIGNED/not claimed",
            "rmot_only": True,
        }
        self.image_proj = nn.Sequential(nn.LayerNorm(image_dim),
                                        nn.Linear(image_dim, hidden), nn.GELU())
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim),
                                       nn.Linear(text_dim, hidden), nn.GELU())
        self.frozen_proj = nn.Sequential(nn.LayerNorm(frozen_dim),
                                         nn.Linear(frozen_dim, hidden), nn.GELU())
        self.numeric_proj = nn.Sequential(nn.LayerNorm(numeric_dim),
                                          nn.Linear(numeric_dim, hidden), nn.GELU())
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True)
            for _ in range(layers)
        ])
        self.cross_norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.candidate_fuse = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )
        self.set_attn = nn.MultiheadAttention(hidden, heads, dropout=0.0, batch_first=True)
        self.set_norm = nn.LayerNorm(hidden)
        self.residual_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1)
        )
        # Zero only the final projection: the initial residual is exactly zero,
        # while the preceding adapter remains capable of receiving gradients
        # after the first optimizer update. Zeroing every layer would reduce the
        # branch to a candidate-independent bias and block image/text gradients.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    @staticmethod
    def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        if mask.ndim == 1:
            mask = mask.unsqueeze(0).expand(tokens.shape[0], -1)
        weights = mask.to(tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def forward(self, patch_tokens: torch.Tensor, text_tokens: torch.Tensor,
                text_mask: torch.Tensor, frozen_clip: torch.Tensor,
                numeric: torch.Tensor, teacher: torch.Tensor) -> dict[str, torch.Tensor]:
        """Process one complete frame candidate set.

        Shapes: patch ``[N,P,768]``, text ``[T,768]``, frozen clip ``[N,512]``,
        numeric ``[N,36]``, teacher ``[N]``.  No semantic identifiers are
        accepted.
        """
        if patch_tokens.ndim != 3 or text_tokens.ndim != 2:
            raise ValueError("expected patch [N,P,D] and text [T,D]")
        n = patch_tokens.shape[0]
        if n < 1 or frozen_clip.shape[:1] != (n,) or numeric.shape[:1] != (n,):
            raise ValueError("candidate streams are not aligned")
        if teacher.shape != (n,) or text_mask.shape != (text_tokens.shape[0],):
            raise ValueError("teacher/text mask is not aligned")
        image = self.image_proj(torch.nan_to_num(patch_tokens.float()))
        text = self.text_proj(torch.nan_to_num(text_tokens.float()))
        q = text.unsqueeze(0).expand(n, -1, -1)
        for attn, norm in zip(self.cross_attn, self.cross_norm):
            z, _ = attn(q, image, image, need_weights=False)
            q = norm(q + z)
        q_summary = self.masked_mean(q, text_mask).to(image.dtype)
        frozen = self.frozen_proj(torch.nan_to_num(frozen_clip.float()))
        numeric_h = self.numeric_proj(torch.nan_to_num(numeric.float()))
        candidate = self.candidate_fuse(torch.cat((q_summary, frozen, numeric_h, image.mean(1)), -1))
        set_z, _ = self.set_attn(candidate.unsqueeze(0), candidate.unsqueeze(0),
                                 candidate.unsqueeze(0), need_weights=False)
        candidate = self.set_norm(candidate + set_z[0])
        raw_residual = self.residual_head(candidate).squeeze(-1)
        residual = float(self.config["residual_bound"]) * torch.tanh(raw_residual)
        teacher = teacher.float().detach()
        final = teacher + residual
        return {
            "teacher": teacher,
            "raw_residual": raw_residual,
            "residual": residual,
            "final_logit": final,
            "candidate_features": candidate,
        }
