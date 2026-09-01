"""L70 persistent-history/query-conditioned set decoder.

This is an RMOT-only adapter over the L69 budget-40 observation bank.  IDs are
used by the data loader to assemble a causal history, never as model inputs.
The module intentionally has no dependency on the L29/L64 score paths.
"""
from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn


class L70PersistentSetDecoder(nn.Module):
    def __init__(
        self,
        obs_dim: int = 1432,
        text_dim: int = 768,
        hidden: int = 192,
        heads: int = 4,
        layers: int = 2,
        max_history: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.obs_dim = int(obs_dim)
        self.text_dim = int(text_dim)
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.layers = int(layers)
        self.max_history = int(max_history)
        self.obs_proj = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.GELU(),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(1, self.hidden),
            nn.LayerNorm(self.hidden),
            nn.Tanh(),
        )
        temporal_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden,
            nhead=self.heads,
            dim_feedforward=self.hidden * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.track_temporal_encoder = nn.TransformerEncoder(temporal_layer, num_layers=self.layers)
        self.text_proj = nn.Sequential(
            nn.Linear(self.text_dim, self.hidden),
            nn.LayerNorm(self.hidden),
        )
        self.query_norm = nn.LayerNorm(self.hidden)
        self.history_norm = nn.LayerNorm(self.hidden)
        self.query_to_history = nn.MultiheadAttention(
            self.hidden, self.heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(self.hidden)
        set_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden,
            nhead=self.heads,
            dim_feedforward=self.hidden * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.current_frame_set_competition = nn.TransformerEncoder(set_layer, num_layers=self.layers)
        triple = self.hidden * 3
        self.membership_head = nn.Sequential(
            nn.LayerNorm(triple), nn.Linear(triple, self.hidden), nn.GELU(), nn.Linear(self.hidden, 1)
        )
        self.track_head = nn.Sequential(
            nn.LayerNorm(self.hidden * 2), nn.Linear(self.hidden * 2, self.hidden), nn.GELU(), nn.Linear(self.hidden, 1)
        )
        self.continuation_head = nn.Sequential(
            nn.LayerNorm(self.hidden * 2), nn.Linear(self.hidden * 2, self.hidden), nn.GELU(), nn.Linear(self.hidden, 1)
        )
        self.history_head = nn.Sequential(
            nn.LayerNorm(self.hidden * 2), nn.Linear(self.hidden * 2, self.hidden), nn.GELU(), nn.Linear(self.hidden, 1)
        )
        self.null_head = nn.Sequential(
            nn.LayerNorm(self.hidden * 2), nn.Linear(self.hidden * 2, self.hidden), nn.GELU(), nn.Linear(self.hidden, 1)
        )

    @staticmethod
    def _masked_mean(value: Tensor, mask: Tensor, dim: int) -> Tensor:
        weight = mask.to(value.dtype)
        while weight.ndim < value.ndim:
            weight = weight.unsqueeze(-1)
        denom = weight.sum(dim=dim).clamp_min(1.0)
        return (value * weight).sum(dim=dim) / denom

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        current = batch["current"]
        history = batch["history"]
        history_mask = batch["history_mask"].bool()
        history_time = batch["history_time"]
        text = batch["text"]
        text_mask = batch["text_mask"].bool()
        if current.ndim != 2 or current.shape[-1] != self.obs_dim:
            raise ValueError(f"current must be [N,{self.obs_dim}], got {tuple(current.shape)}")
        if history.ndim != 3 or history.shape[0] != current.shape[0] or history.shape[2] != self.obs_dim:
            raise ValueError(f"history/current shape mismatch: {tuple(history.shape)} / {tuple(current.shape)}")
        if history.shape[1] != self.max_history or history_mask.shape != history.shape[:2]:
            raise ValueError("history mask/length mismatch")
        if history_time.shape != history_mask.shape:
            raise ValueError("history time/mask mismatch")
        if text.ndim != 2 or text.shape[-1] != self.text_dim or text_mask.shape != text.shape[:1]:
            raise ValueError(f"text/mask shape mismatch: {tuple(text.shape)} / {tuple(text_mask.shape)}")
        if current.shape[0] == 0 or not history_mask.any(dim=1).all():
            raise ValueError("each current candidate must have at least one valid history observation")

        n, length = history.shape[:2]
        current_encoded = self.obs_proj(current)
        temporal = self.obs_proj(history.reshape(n * length, self.obs_dim)).reshape(n, length, self.hidden)
        temporal = temporal + self.time_proj(history_time.reshape(n * length, 1)).reshape(n, length, self.hidden)
        causal_mask = torch.triu(
            torch.ones((length, length), dtype=torch.bool, device=history.device), diagonal=1
        )
        temporal = self.track_temporal_encoder(
            temporal,
            mask=causal_mask,
            src_key_padding_mask=~history_mask,
        )
        temporal = self.history_norm(temporal)
        track_base = self._masked_mean(temporal, history_mask, dim=1)

        query_tokens = self.query_norm(self.text_proj(text)).unsqueeze(0).expand(n, -1, -1)
        query_output, _ = self.query_to_history(
            query_tokens,
            temporal,
            temporal,
            key_padding_mask=~history_mask,
            need_weights=False,
        )
        query_output = self.cross_norm(query_output + query_tokens)
        query_conditioned = self._masked_mean(query_output, text_mask.unsqueeze(0).expand(n, -1), dim=1)
        track_vector = track_base + query_conditioned
        current_slot = history_mask.long().sum(dim=1).clamp_min(1) - 1
        current_temporal = temporal[torch.arange(n, device=temporal.device), current_slot]

        set_input = track_vector + current_encoded + current_temporal
        set_vector = self.current_frame_set_competition(set_input.unsqueeze(0)).squeeze(0)
        membership_input = torch.cat((set_vector, track_vector, current_encoded), dim=-1)
        query_track = torch.cat((track_vector, query_conditioned), dim=-1)
        membership_logits = self.membership_head(membership_input).squeeze(-1)
        track_logits = self.track_head(query_track).squeeze(-1)
        continuation_logits = self.continuation_head(query_track).squeeze(-1)

        query_global = self._masked_mean(self.text_proj(text), text_mask, dim=0).expand(n, -1)
        history_input = torch.cat((temporal, query_global[:, None, :].expand(-1, length, -1)), dim=-1)
        history_logits = self.history_head(history_input).squeeze(-1)
        set_global = set_vector.mean(dim=0, keepdim=True).expand(n, -1)
        null_logits = self.null_head(torch.cat((query_global, set_global), dim=-1)).squeeze(-1)
        return {
            "membership_logits": membership_logits,
            "track_logits": track_logits,
            "continuation_logits": continuation_logits,
            "history_membership_logits": history_logits,
            "null_logit": null_logits[:1],
            "track_vector": track_vector,
            "set_vector": set_vector,
        }

    def parameter_summary(self) -> dict[str, int]:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        return {"total_parameters": int(total), "trainable_parameters": int(trainable)}
