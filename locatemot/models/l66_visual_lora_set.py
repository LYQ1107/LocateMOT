"""L66 compact set head and a rank-8 LoRA wrapper for one CLIP visual layer."""
from __future__ import annotations

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Frozen linear plus zero-initialized low-rank update."""
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / max(1, self.rank)
        self.dropout = nn.Dropout(float(dropout))
        # Keep trainable LoRA state in FP32 even when the frozen CUDA CLIP is
        # loaded in FP16; otherwise AdamW's state/update can overflow on step 1.
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)

    def forward(self, x):
        base = self.base(x)
        update = (self.dropout(x.float()) @ self.lora_A.t()) @ self.lora_B.t()
        return base + (update * self.scaling).to(dtype=base.dtype)


def attach_visual_lora(clip_model, rank=8, alpha=16.0, dropout=0.0):
    """Attach exactly one LoRA to the verified final visual MLP c_proj."""
    block = clip_model.visual.transformer.resblocks[-1]
    target = block.mlp.c_proj
    if not isinstance(target, nn.Linear):
        raise TypeError(f"expected final visual mlp.c_proj Linear, got {type(target)}")
    if hasattr(target, "lora_A"):
        raise RuntimeError("LoRA already attached")
    wrapped = LoRALinear(target, rank=rank, alpha=alpha, dropout=dropout)
    block.mlp.c_proj = wrapped
    return "visual.transformer.resblocks[-1].mlp.c_proj", wrapped


class L66VisualLoraSet(nn.Module):
    """Same frozen-feature input/output contract as L65, under a new module."""
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
        if patch_joint.dim() != 3 or patch_joint.shape[-1] != 512 or patch_joint.shape[1] != 17:
            raise ValueError(f"patch_joint must be [N,17,512], got {tuple(patch_joint.shape)}")
        if text_joint.dim() != 2 or text_joint.shape[-1] != 512:
            raise ValueError(f"text_joint must be [T,512], got {tuple(text_joint.shape)}")
        if numeric.dim() != 2 or numeric.shape != (patch_joint.shape[0], 32):
            raise ValueError("numeric/candidate alignment mismatch")
        if text_valid.shape != (text_joint.shape[0],):
            raise ValueError("text mask mismatch")
        image = self.image_proj(patch_joint)
        words = self.text_proj(text_joint).unsqueeze(0).expand(patch_joint.shape[0], -1, -1)
        padding = (~text_valid.bool()).unsqueeze(0).expand(patch_joint.shape[0], -1)
        attended, _ = self.patch_to_words(image, words, words, key_padding_mask=padding)
        candidate = image.mean(1) + attended.mean(1) + self.numeric_proj(numeric)
        competed = self.set_layer(candidate.unsqueeze(0)).squeeze(0)
        x = self.set_norm(candidate + competed)
        score = self.relevance(x).squeeze(-1)
        valid_words = self.text_proj(text_joint[text_valid.bool()]) if bool(text_valid.any()) else self.text_proj(text_joint)
        null = self.null_head(torch.cat((x.mean(0), valid_words.mean(0)), 0)).reshape(())
        return {"relevance_logit": score, "null_logit": null, "candidate_hidden": x}


def lora_parameters(clip_model):
    return [p for name, p in clip_model.named_parameters() if "lora_A" in name or "lora_B" in name]
