"""L85 factorized full-RMOT sidecar model.

The model consumes the selected L84 Z1 state and query-independent L69
observations.  Candidate rows remain a complete set; the only set operation
is a mean-centered prior, not selection or deletion.  Track IDs are never
passed as features.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn

from locatemot.rmot.l85_temporal import CausalHistoryEncoder


@dataclass(frozen=True)
class L85Config:
    semantic_dim: int = 256
    obs_dim: int = 1432
    hidden: int = 256
    history_length: int = 8
    dropout: float = 0.05
    presence_input_dim: int = 512
    lora_target: str = "Z1: last_two_fusion_encoder_blocks_plus_decoder_layer1 (not loaded in compact cache)"


class L85FullRMOT(nn.Module):
    def __init__(self, config: L85Config | None = None) -> None:
        super().__init__()
        self.config = config or L85Config()
        c = self.config
        self.static = nn.Sequential(
            nn.LayerNorm(c.semantic_dim), nn.Linear(c.semantic_dim, c.hidden), nn.GELU(),
            nn.Dropout(c.dropout), nn.Linear(c.hidden, 1),
        )
        self.history = CausalHistoryEncoder(c.obs_dim, c.hidden, c.history_length)
        self.temporal = nn.Sequential(nn.LayerNorm(c.hidden * 3), nn.Linear(c.hidden * 3, c.hidden), nn.GELU(), nn.Linear(c.hidden, 1))
        self.temporal_gate = nn.Sequential(nn.Linear(c.hidden * 2, c.hidden), nn.GELU(), nn.Linear(c.hidden, 1))
        self.obs_current = nn.Sequential(nn.LayerNorm(c.obs_dim), nn.Linear(c.obs_dim, c.hidden), nn.GELU())
        self.candidate_prior = nn.Sequential(nn.Linear(c.hidden * 2, c.hidden), nn.GELU(), nn.Linear(c.hidden, 1))
        self.presence = nn.Sequential(nn.LayerNorm(c.presence_input_dim), nn.Linear(c.presence_input_dim, c.hidden), nn.GELU(), nn.Linear(c.hidden, 1))
        self.null_bias = nn.Parameter(torch.zeros(()))

    def parameter_report(self) -> dict[str, Any]:
        trainable = {name: int(value.numel()) for name, value in self.named_parameters() if value.requires_grad}
        return {"total": int(sum(value.numel() for value in self.parameters())),
                "trainable": int(sum(trainable.values())), "trainable_by_name": trainable,
                "config": asdict(self.config)}

    @staticmethod
    def _obs_from_fields(fields: dict[str, torch.Tensor]) -> torch.Tensor:
        names = ("clip", "history_clip", "uidm_h", "geometry", "motion", "lifecycle", "objectness")
        chunks = []
        for name in names:
            value = fields[name].float()
            if value.ndim == 1:
                value = value[:, None]
            chunks.append(value)
        result = torch.cat(chunks, dim=-1)
        if result.shape[-1] != 1432:
            raise ValueError(f"observation dimension drift: {tuple(result.shape)}")
        return result

    def forward(self, z1: torch.Tensor, presence_input: torch.Tensor,
                current_observation: torch.Tensor, history_observations: torch.Tensor,
                history_mask: torch.Tensor, history_frame_ids: torch.Tensor | None = None,
                cutoff_frame: int | None = None, *, temporal_enabled: bool = True) -> dict[str, torch.Tensor]:
        if z1.ndim != 3 or presence_input.ndim != 2:
            raise ValueError("L85 expects z1 [Q,N,D] and presence [Q,2D]")
        q, n, d = z1.shape
        if d != self.config.semantic_dim or presence_input.shape != (q, self.config.presence_input_dim):
            raise ValueError("L85 input orientation/dimension drift")
        if current_observation.shape != (n, self.config.obs_dim):
            raise ValueError("current observation shape drift")
        if history_observations.shape[:2] != (n, self.config.history_length) or history_observations.shape[-1] != self.config.obs_dim:
            raise ValueError("history observation shape drift")
        if history_mask.shape != history_observations.shape[:2]:
            raise ValueError("history mask shape drift")
        if not bool(torch.isfinite(z1.float()).all() and torch.isfinite(presence_input.float()).all() and torch.isfinite(current_observation.float()).all() and torch.isfinite(history_observations.float()).all()):
            raise FloatingPointError("nonfinite L85 model input")

        r_static = self.static(z1.float()).squeeze(-1)
        h = self.history(history_observations.float(), history_mask.bool(), history_frame_ids, cutoff_frame)
        obs = self.obs_current(current_observation.float())
        a = self.candidate_prior(torch.cat((obs, h), dim=-1)).squeeze(-1)
        a = a - a.mean()
        hq = h.unsqueeze(0).expand(q, -1, -1)
        zq = z1.float()
        temporal_input = torch.cat((zq, hq, zq * hq), dim=-1)
        c = self.temporal(temporal_input).squeeze(-1)
        gate = torch.sigmoid(self.temporal_gate(torch.cat((zq, hq), dim=-1)).squeeze(-1))
        if not temporal_enabled:
            # Preserve the graph so DDP sees the registered temporal modules
            # in the warm-up stage; their gradients are intentionally zero.
            c = c * 0.0; gate = gate * 0.0
        r_total = r_static + gate * c
        b = self.presence(presence_input.float()).squeeze(-1)
        score = r_total + a.unsqueeze(0) + b.unsqueeze(-1)
        null = -b + self.null_bias
        output = {"r_static": r_static, "r_total": r_total, "candidate_prior": a,
                  "presence": b, "membership": score, "null_logit": null, "history": h,
                  "temporal_correction": c, "temporal_gate": gate}
        if not all(bool(torch.isfinite(value.float()).all()) for value in output.values()):
            raise FloatingPointError("nonfinite L85 output")
        return output


__all__ = ["L85Config", "L85FullRMOT"]
