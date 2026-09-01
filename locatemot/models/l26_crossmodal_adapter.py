"""Frozen-backbone DINOv2/word-token cross-modal adapter for L26."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class L26CrossModalAdapter(nn.Module):
    """Token-to-region matcher; source/pool/group/state are not inputs."""

    def __init__(self, shared_dim: int = 256, heads: int = 4, variant: str = "token_region"):
        super().__init__()
        if variant not in {"token_region", "projection", "attribute_mask"}:
            raise ValueError(variant)
        self.variant = variant
        self.shared_dim = shared_dim
        self.vision_proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, shared_dim, bias=False))
        self.text_proj = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, shared_dim, bias=False))
        self.coord_proj = nn.Linear(2, shared_dim, bias=False)
        self.cross = nn.MultiheadAttention(shared_dim, heads, batch_first=True)
        self.region_norm = nn.LayerNorm(shared_dim)
        self.query_norm = nn.LayerNorm(shared_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.0))

    def forward(self, text_tokens, text_mask, roi_tokens, roi_coords):
        # text_tokens [L,768], mask [L]; roi_tokens [N,K,768], coords [N,K,2].
        text = self.text_proj(torch.nan_to_num(text_tokens.float()))
        text = self.query_norm(text)
        visual = self.vision_proj(torch.nan_to_num(roi_tokens.float()))
        visual = self.region_norm(visual + self.coord_proj(torch.nan_to_num(roi_coords.float())))
        n = visual.shape[0]
        q = text.unsqueeze(0).expand(n, -1, -1)
        mask = text_mask.bool().reshape(1, -1).expand(n, -1)
        if self.variant == "attribute_mask" and self.training:
            keep = (torch.rand(mask.shape, device=mask.device) > .20) & mask
            keep[:, 0] = mask[:, 0]
            mask = keep
            q = q * keep.unsqueeze(-1).float()
        if self.variant == "projection":
            qpool = (q * mask.unsqueeze(-1).float()).sum(1) / mask.float().sum(1, keepdim=True).clamp_min(1.0)
            fpool = visual.mean(1)
            score = F.cosine_similarity(qpool, fpool, dim=-1) * self.logit_scale.exp().clamp(max=100.0)
            return {"score": score, "attention": None, "attention_entropy": qpool.new_tensor(1.0), "qpool": qpool, "fpool": fpool}
        fused, weights = self.cross(q, visual, visual, need_weights=True, average_attn_weights=True)
        valid = mask.unsqueeze(-1).float()
        qpool = (q * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        fpool = (fused * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        score = F.cosine_similarity(qpool, fpool, dim=-1) * self.logit_scale.exp().clamp(max=100.0)
        attn = weights.clamp_min(1e-8)
        denom = torch.log(torch.tensor(float(visual.shape[1]), device=weights.device)).clamp_min(1e-6)
        entropy = -(attn * attn.log()).sum(-1) / denom
        entropy = (entropy * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
        return {"score": score, "attention": weights, "attention_entropy": entropy, "qpool": qpool, "fpool": fpool}


class L26BoundedResidual(nn.Module):
    """A bounded residual over a frozen, already-trained token-region adapter."""

    def __init__(self, base: L26CrossModalAdapter, alpha: float = 0.1):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.alpha = alpha
        self.residual = nn.Sequential(nn.LayerNorm(base.shared_dim * 2), nn.Linear(base.shared_dim * 2, base.shared_dim), nn.GELU(), nn.Linear(base.shared_dim, 1))

    def forward(self, text_tokens, text_mask, roi_tokens, roi_coords):
        with torch.no_grad():
            teacher = self.base(text_tokens, text_mask, roi_tokens, roi_coords)
        qpool = teacher["qpool"].mean(0, keepdim=True).expand(roi_tokens.shape[0], -1)
        residual = self.residual(torch.cat((qpool, teacher["fpool"]), -1)).squeeze(-1)
        return {**teacher, "score": teacher["score"].detach() + self.alpha * torch.tanh(residual), "residual": residual}


__all__ = ["L26CrossModalAdapter", "L26BoundedResidual"]
