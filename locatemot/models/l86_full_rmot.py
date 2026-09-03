"""Faithful L86 factorized full-RMOT model.

L86 keeps the L84 Z1 representation and L69 observations frozen.  It fixes
the L85 objective/model contract by using a shared static/temporal semantic
head, a query-independent centered candidate prior, and independent presence
and candidate-evidence-aware NULL energies.  Track/source/pool/query IDs are
never model inputs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from locatemot.rmot.l86_temporal import CausalHistoryEncoder


@dataclass(frozen=True)
class L86Config:
    semantic_dim: int = 256
    obs_dim: int = 1432
    hidden: int = 256
    history_length: int = 8
    dropout: float = 0.05
    presence_input_dim: int = 512
    null_margin: float = 0.50
    temporal_margin: float = 0.20


class L86FullRMOT(nn.Module):
    def __init__(self, config: L86Config | None = None) -> None:
        super().__init__()
        self.config = config or L86Config()
        c = self.config
        self.semantic_head = nn.Sequential(
            nn.LayerNorm(c.semantic_dim),
            nn.Linear(c.semantic_dim, c.hidden),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden, 1),
        )
        self.history = CausalHistoryEncoder(c.obs_dim, c.hidden, c.history_length)
        self.temporal_delta = nn.Sequential(
            nn.LayerNorm(c.semantic_dim * 3),
            nn.Linear(c.semantic_dim * 3, c.hidden),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden, c.semantic_dim),
        )
        self.temporal_gate_head = nn.Sequential(
            nn.LayerNorm(c.semantic_dim * 2),
            nn.Linear(c.semantic_dim * 2, c.hidden),
            nn.GELU(),
            nn.Linear(c.hidden, 1),
        )
        self.prior_obs = nn.Sequential(
            nn.LayerNorm(c.obs_dim), nn.Linear(c.obs_dim, c.hidden), nn.GELU()
        )
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
        trainable = {name: int(value.numel()) for name, value in self.named_parameters() if value.requires_grad}
        return {
            "total": int(sum(value.numel() for value in self.parameters())),
            "trainable": int(sum(trainable.values())),
            "trainable_by_name": trainable,
            "config": asdict(self.config),
        }

    def forward(
        self,
        z1: torch.Tensor,
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
        if z1.ndim != 3 or text_global.ndim != 2 or frame_global.ndim != 2:
            raise ValueError("L86 expects z1 [Q,N,D] and text/frame globals [Q,D]")
        q, n, d = z1.shape
        c = self.config
        if d != c.semantic_dim or text_global.shape != (q, c.semantic_dim) or frame_global.shape != (q, c.semantic_dim):
            raise ValueError("L86 semantic input orientation/dimension drift")
        if current_observation.shape != (n, c.obs_dim):
            raise ValueError("L86 current observation shape drift")
        if history_observations.shape != (n, c.history_length, c.obs_dim):
            raise ValueError("L86 history observation shape drift")
        if history_mask.shape != (n, c.history_length):
            raise ValueError("L86 history mask shape drift")
        finite_inputs = (
            z1, text_global, frame_global, current_observation, history_observations
        )
        if not all(bool(torch.isfinite(value.float()).all()) for value in finite_inputs):
            raise FloatingPointError("nonfinite L86 model input")

        z = z1.float()
        r_static = self.semantic_head(z).squeeze(-1)
        h = self.history(history_observations.float(), history_mask.bool(), history_frame_ids, cutoff_frame)
        obs = self.prior_obs(current_observation.float())
        a_raw = self.candidate_prior_head(torch.cat((obs, h), dim=-1)).squeeze(-1)
        a = 0.5 * torch.tanh(a_raw)
        a = a - a.mean()
        hq = h.unsqueeze(0).expand(q, -1, -1)
        temporal_input = torch.cat((z, hq, z * hq), dim=-1)
        delta = self.temporal_delta(temporal_input)
        gate_logits = self.temporal_gate_head(torch.cat((z, hq), dim=-1)).squeeze(-1)
        gate = torch.sigmoid(gate_logits)
        if not temporal_enabled:
            # Keep every registered module in the graph for DDP; gradients are
            # intentionally zero in Stage S, without a history-length target.
            delta = delta * 0.0
            gate = gate * 0.0
        temporal_state = z + gate.unsqueeze(-1) * delta
        r_total = self.semantic_head(temporal_state).squeeze(-1)
        presence_input = torch.cat((text_global.float(), frame_global.float()), dim=-1)
        presence_logit = self.presence_head(presence_input).squeeze(-1)
        candidate_energy = r_total + a.unsqueeze(0)
        weights = torch.softmax(r_total / 0.10, dim=-1)
        candidate_summary = (weights.unsqueeze(-1) * temporal_state).sum(dim=1)
        null_input = torch.cat((presence_input, candidate_summary), dim=-1)
        null_logit = self.null_head(null_input).squeeze(-1)
        output = {
            "r_static": r_static,
            "temporal_state": temporal_state,
            "r_total": r_total,
            "candidate_prior": a,
            "candidate_energy": candidate_energy,
            "presence_logit": presence_logit,
            "null_logit": null_logit,
            "temporal_gate_logits": gate_logits,
            "temporal_gate": gate,
            "temporal_delta": delta,
            "history_state": h,
        }
        if not all(bool(torch.isfinite(value.float()).all()) for value in output.values()):
            raise FloatingPointError("nonfinite L86 model output")
        return output


__all__ = ["L86Config", "L86FullRMOT"]
