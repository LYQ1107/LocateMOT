"""Stage L38: a bounded residual on top of frozen L29 current membership."""
from __future__ import annotations

import torch
from torch import nn


class L38BoundedEmissionResidual(nn.Module):
    """Query-conditioned, source-blind residual with L29 as an immutable anchor."""

    def __init__(self, feature_dim: int = 1432, text_dim: int = 768,
                 hidden: int = 96, history: int = 8, max_text: int = 64,
                 residual_bound: float = 0.05):
        super().__init__()
        self.hidden = int(hidden)
        self.history = int(history)
        self.residual_bound = float(residual_bound)
        if feature_dim != 1432:
            raise ValueError("L38 feature layout is the frozen L19/L28 1432-D layout")
        self.query_proj = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden))
        self.query_pos = nn.Parameter(torch.zeros(1, max_text, hidden))
        self.query_norm = nn.LayerNorm(hidden)
        self.visual_proj = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, hidden))
        self.identity_proj = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, hidden))
        self.numeric_proj = nn.Sequential(nn.LayerNorm(24), nn.Linear(24, hidden))
        self.residual_head = nn.Sequential(
            nn.Linear(3 * hidden + 1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.continuation_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        nn.init.normal_(self.query_pos, std=0.01)

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        w = mask.to(dtype=x.dtype).unsqueeze(-1)
        return (x * w).sum(1) / w.sum(1).clamp_min(1.0)

    def forward(self, observations: torch.Tensor, observation_mask: torch.Tensor,
                observation_time: torch.Tensor, query_tokens: torch.Tensor,
                query_mask: torch.Tensor, teacher_logits: torch.Tensor) -> dict[str, torch.Tensor]:
        # Frozen L19/L28 layout: clip+history_clip(1024), uidm_h(384),
        # geometry+motion+lifecycle+objectness(24).  No source/pool/group/state.
        if query_tokens.ndim == 2:
            query_tokens = query_tokens.unsqueeze(0)
        if query_mask.ndim == 1:
            query_mask = query_mask.unsqueeze(0)
        q = self.query_proj(torch.nan_to_num(query_tokens.float()))
        q = q[:, :self.query_pos.shape[1]] + self.query_pos[:, :q.shape[1]]
        qmask = query_mask[:, :q.shape[1]].bool()
        q = q.masked_fill(~qmask.unsqueeze(-1), 0.0)
        q_global = self.query_norm(self.masked_mean(q, qmask))
        x = torch.nan_to_num(observations.float())
        visual = self.visual_proj(x[..., :1024])
        identity = self.identity_proj(x[..., 1024:1408])
        numeric = self.numeric_proj(x[..., 1408:])
        mask = observation_mask.bool()
        hist = (visual + identity + numeric).masked_fill(~mask.unsqueeze(-1), 0.0)
        hist_global = self.masked_mean(hist, mask)
        latest = mask.long().sum(1).clamp_min(1) - 1
        q_track = q_global.expand(hist.shape[0], -1)
        teacher = teacher_logits.float().reshape(-1, 1)
        teacher_history = teacher[:, None, :].expand(-1, hist.shape[1], -1)
        q_history = q_track[:, None, :].expand(-1, hist.shape[1], -1)
        global_history = hist_global[:, None, :].expand(-1, hist.shape[1], -1)
        residual_raw_history = self.residual_head(
            torch.cat((hist, global_history, q_history, teacher_history), -1)).squeeze(-1)
        residual_history = self.residual_bound * torch.tanh(residual_raw_history)
        residual_history = residual_history.masked_fill(~mask, 0.0)
        residual = residual_history[torch.arange(hist.shape[0], device=hist.device), latest]
        final = teacher_logits.float() + residual
        continuation = self.continuation_head(torch.cat((hist_global, q_track), -1)).squeeze(-1)
        agreement = torch.sigmoid(-residual.abs() / max(self.residual_bound, 1e-6))
        return {
            "teacher_score": teacher_logits.float(),
            "residual_score": residual,
            "final_score": final,
            "residual_history": residual_history,
            "continuation_logit": continuation,
            "agreement": agreement,
        }
