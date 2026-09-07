"""Query-conditioned complete candidate-set decoder for L89."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class L89SetDecoderConfig:
    dim: int = 256
    num_heads: int = 8
    num_layers: int = 2
    ffn_dim: int = 1024
    dropout: float = 0.10


class L89SetDecoderBlock(nn.Module):
    def __init__(self, config: L89SetDecoderConfig) -> None:
        super().__init__()
        d = int(config.dim)
        self.candidate_norm = nn.LayerNorm(d)
        self.candidate_self_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=int(config.num_heads), dropout=float(config.dropout), batch_first=True
        )
        self.language_norm = nn.LayerNorm(d)
        self.language_cross_attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=int(config.num_heads), dropout=float(config.dropout), batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, int(config.ffn_dim)), nn.GELU(), nn.Dropout(float(config.dropout)), nn.Linear(int(config.ffn_dim), d)
        )
        self.residual_dropout = nn.Dropout(float(config.dropout))

    def forward(self, candidates: torch.Tensor, text_tokens: torch.Tensor, text_token_mask: torch.Tensor) -> torch.Tensor:
        if candidates.ndim != 3 or text_tokens.ndim != 3 or text_token_mask.ndim != 2:
            raise ValueError("L89 set block expects candidates [Q,N,D], tokens [Q,L,D], mask [Q,L]")
        if candidates.shape[0] != text_tokens.shape[0] or text_tokens.shape[:2] != text_token_mask.shape:
            raise ValueError("L89 set/query batch shape drift")
        if candidates.shape[-1] != 256 or text_tokens.shape[-1] != 256:
            raise ValueError("L89 set decoder dimension must be 256")
        valid = text_token_mask.bool()
        if not bool(valid.any(dim=1).all()):
            raise ValueError("each L89 query requires a valid language token")
        x = candidates
        y = self.candidate_norm(x)
        attended, _ = self.candidate_self_attn(y, y, y, need_weights=False)
        x = x + self.residual_dropout(attended)
        q = self.language_norm(x)
        lang, _ = self.language_cross_attn(q, text_tokens, text_tokens, key_padding_mask=~valid, need_weights=False)
        x = x + self.residual_dropout(lang)
        y = self.ffn_norm(x)
        x = x + self.residual_dropout(self.ffn(y))
        if not bool(torch.isfinite(x.float()).all()):
            raise FloatingPointError("nonfinite L89 set decoder block output")
        return x


class L89SetCorrespondenceDecoder(nn.Module):
    def __init__(self, config: L89SetDecoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or L89SetDecoderConfig()
        self.candidate_input_norm = nn.LayerNorm(256)
        self.text_input_norm = nn.LayerNorm(256)
        self.layers = nn.ModuleList([L89SetDecoderBlock(self.config) for _ in range(self.config.num_layers)])
        self.output_norm = nn.LayerNorm(256)

    def forward(self, candidate_state: torch.Tensor, text_tokens: torch.Tensor, text_token_mask: torch.Tensor) -> torch.Tensor:
        if candidate_state.ndim != 3 or candidate_state.shape[-1] != 256:
            raise ValueError(f"L89 candidate state must be [Q,N,256], got {tuple(candidate_state.shape)}")
        if text_tokens.ndim != 3 or text_tokens.shape[-1] != 256:
            raise ValueError(f"L89 text tokens must be [Q,L,256], got {tuple(text_tokens.shape)}")
        x = self.candidate_input_norm(candidate_state.float())
        t = self.text_input_norm(text_tokens.float())
        for layer in self.layers:
            x = layer(x, t, text_token_mask.bool())
        result = self.output_norm(x)
        if not bool(torch.isfinite(result.float()).all()):
            raise FloatingPointError("nonfinite L89 set decoder output")
        return result


__all__ = ["L89SetDecoderConfig", "L89SetDecoderBlock", "L89SetCorrespondenceDecoder"]
