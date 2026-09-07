"""L89 full RMOT model: set-aware correspondence over the frozen L84 Z1 bank."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from locatemot.rmot.l86_temporal import CausalHistoryEncoder
from locatemot.rmot.l89_set_decoder import L89SetCorrespondenceDecoder, L89SetDecoderConfig


@dataclass(frozen=True)
class L89Config:
    semantic_dim: int = 256
    obs_dim: int = 1432
    hidden: int = 256
    history_length: int = 8
    dropout: float = 0.05
    presence_input_dim: int = 512
    null_margin: float = 0.50
    temporal_margin: float = 0.20
    set_num_heads: int = 8
    set_num_layers: int = 2
    set_ffn_dim: int = 1024
    set_dropout: float = 0.10


class L89FullRMOT(nn.Module):
    def __init__(self, config: L89Config | None = None) -> None:
        super().__init__()
        self.config = config or L89Config()
        c = self.config
        if c.semantic_dim != 256 or c.hidden != 256 or c.history_length != 8:
            raise ValueError("L89 registered dimensions/history changed")
        self.set_decoder = L89SetCorrespondenceDecoder(L89SetDecoderConfig(
            dim=256, num_heads=8, num_layers=2, ffn_dim=1024, dropout=0.10
        ))
        self.membership_head = nn.Sequential(
            nn.LayerNorm(256), nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.05), nn.Linear(256, 1)
        )
        self.history = CausalHistoryEncoder(c.obs_dim, c.hidden, c.history_length)
        self.temporal_delta = nn.Sequential(
            nn.LayerNorm(c.semantic_dim * 3), nn.Linear(c.semantic_dim * 3, c.hidden), nn.GELU(),
            nn.Dropout(c.dropout), nn.Linear(c.hidden, c.semantic_dim)
        )
        self.temporal_gate_head = nn.Sequential(
            nn.LayerNorm(c.semantic_dim * 2), nn.Linear(c.semantic_dim * 2, c.hidden), nn.GELU(), nn.Linear(c.hidden, 1)
        )
        self.prior_obs = nn.Sequential(nn.LayerNorm(c.obs_dim), nn.Linear(c.obs_dim, c.hidden), nn.GELU())
        self.candidate_prior_head = nn.Sequential(
            nn.LayerNorm(c.hidden * 2), nn.Linear(c.hidden * 2, c.hidden), nn.GELU(), nn.Linear(c.hidden, 1)
        )
        self.presence_head = nn.Sequential(
            nn.LayerNorm(c.presence_input_dim), nn.Linear(c.presence_input_dim, c.hidden), nn.GELU(), nn.Linear(c.hidden, 1)
        )
        self.null_head = nn.Sequential(
            nn.LayerNorm(c.semantic_dim * 3), nn.Linear(c.semantic_dim * 3, c.hidden), nn.GELU(), nn.Linear(c.hidden, 1)
        )

    def parameter_report(self) -> dict[str, Any]:
        by_name = {name: int(value.numel()) for name, value in self.named_parameters() if value.requires_grad}
        return {"total": int(sum(value.numel() for value in self.parameters())),
                "trainable": int(sum(by_name.values())), "trainable_by_name": by_name, "config": asdict(self.config)}

    def forward(
        self,
        z1: torch.Tensor,
        text_tokens: torch.Tensor,
        text_token_mask: torch.Tensor,
        text_global: torch.Tensor,
        frame_global: torch.Tensor,
        current_observation: torch.Tensor,
        history_observations: torch.Tensor,
        history_mask: torch.Tensor,
        history_frame_ids: torch.Tensor | None = None,
        cutoff_frame: int | None = None,
        *,
        temporal_enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        c = self.config
        if z1.ndim != 3 or text_tokens.ndim != 3 or text_token_mask.ndim != 2:
            raise ValueError("L89 expects z1 [Q,N,256], text tokens [Q,L,256], mask [Q,L]")
        q, n, d = z1.shape
        if d != c.semantic_dim or text_tokens.shape[0] != q or text_tokens.shape[2] != 256:
            raise ValueError("L89 semantic/query shape drift")
        if text_tokens.shape[:2] != text_token_mask.shape:
            raise ValueError("L89 language mask shape drift")
        if text_global.shape != (q, c.semantic_dim) or frame_global.shape != (q, c.semantic_dim):
            raise ValueError("L89 global semantic shape drift")
        if current_observation.shape != (n, c.obs_dim):
            raise ValueError("L89 current observation shape drift")
        if history_observations.shape != (n, c.history_length, c.obs_dim) or history_mask.shape != (n, c.history_length):
            raise ValueError("L89 history shape drift")
        finite_inputs = (z1, text_tokens, text_global, frame_global, current_observation, history_observations)
        if not all(bool(torch.isfinite(value.float()).all()) for value in finite_inputs):
            raise FloatingPointError("nonfinite L89 input")
        z = z1.float()
        static_state = self.set_decoder(z, text_tokens.float(), text_token_mask.bool())
        r_static = self.membership_head(static_state).squeeze(-1)
        h = self.history(history_observations.float(), history_mask.bool(), history_frame_ids, cutoff_frame)
        obs = self.prior_obs(current_observation.float())
        a_raw = self.candidate_prior_head(torch.cat((obs, h), dim=-1)).squeeze(-1)
        a = 0.5 * torch.tanh(a_raw)
        a = a - a.mean()
        hq = h.unsqueeze(0).expand(q, -1, -1)
        temporal_input = torch.cat((static_state, hq, static_state * hq), dim=-1)
        delta = self.temporal_delta(temporal_input)
        gate_logits = self.temporal_gate_head(torch.cat((static_state, hq), dim=-1)).squeeze(-1)
        gate = torch.sigmoid(gate_logits)
        if not temporal_enabled:
            delta = delta * 0.0
            gate = gate * 0.0
        temporal_state = static_state + gate.unsqueeze(-1) * delta
        r_total = self.membership_head(temporal_state).squeeze(-1)
        candidate_energy = r_total + a.unsqueeze(0)
        presence_input = torch.cat((text_global.float(), frame_global.float()), dim=-1)
        presence_logit = self.presence_head(presence_input).squeeze(-1)
        weights = torch.softmax(r_total / 0.10, dim=-1)
        candidate_summary = (weights.unsqueeze(-1) * temporal_state).sum(dim=1)
        null_input = torch.cat((presence_input, candidate_summary), dim=-1)
        null_logit = self.null_head(null_input).squeeze(-1)
        output = {
            "r_static": r_static, "set_static_state": static_state, "temporal_state": temporal_state,
            "r_total": r_total, "candidate_prior": a, "candidate_energy": candidate_energy,
            "presence_logit": presence_logit, "null_logit": null_logit,
            "temporal_gate_logits": gate_logits, "temporal_gate": gate,
            "temporal_delta": delta, "history_state": h,
        }
        if not all(bool(value.isfinite().all()) for value in output.values()):
            raise FloatingPointError("nonfinite L89 output")
        if output["candidate_energy"].shape != (q, n):
            raise AssertionError("L89 candidate count changed")
        return output


__all__ = ["L89Config", "L89FullRMOT"]
