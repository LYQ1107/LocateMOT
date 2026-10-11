"""Bounded semantic residual for OVMOT; identity logits remain primary."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class OVMOTAdapter(nn.Module):
    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1)
        )
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        # Favor identity association at initialization: sigmoid(-2) ~= 0.12.
        nn.init.constant_(self.gate[-1].bias, -2.0)
        self.text_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.temperature = nn.Parameter(torch.tensor(1.0))
        # Task conditioning belongs only to the bounded semantic gate. Zero
        # initialization preserves the original gate before any task training.
        # Supplying zero weights also preserves the original initialization
        # RNG stream for all subsequent shared-core modules.
        self.task_embeddings = nn.Embedding.from_pretrained(torch.zeros(2, hidden_dim), freeze=False)

    def class_logits(self, semantic: torch.Tensor, class_text: torch.Tensor) -> torch.Tensor:
        """Shared U1 text features; no learned per-category classifier table."""
        semantic = F.normalize(semantic.float(), dim=-1)
        text = F.normalize(self.text_projection(class_text).float(), dim=-1)
        return semantic @ text.T / self.temperature.float().clamp_min(0.1)

    def semantic_pair_residual(
        self, track_semantic: torch.Tensor, candidate_semantic: torch.Tensor, class_text: torch.Tensor | None,
        task_type: str = "ovmot",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        t, n = track_semantic.shape[0], candidate_semantic.shape[0]
        if t == 0 or n == 0:
            empty = track_semantic.new_empty((t, n))
            return empty, empty
        trk = F.normalize(track_semantic, dim=-1)
        cand = F.normalize(candidate_semantic, dim=-1)
        trk_exp = trk[:, None, :].expand(t, n, -1)
        cand_exp = cand[None, :, :].expand(t, n, -1)
        base = self.residual(torch.cat([trk_exp, cand_exp, cand_exp - trk_exp], dim=-1)).squeeze(-1)
        if class_text is not None and class_text.numel():
            text = F.normalize(self.text_projection(class_text), dim=-1)
            track_class = trk @ text.T
            cand_class = cand @ text.T
            class_agreement = track_class @ cand_class.T / text.shape[0]
            base = base + class_agreement / self.temperature.clamp_min(0.1)
        task = self.task_embeddings.weight[0 if task_type == "mot" else 1]
        gate = torch.sigmoid(self.gate(torch.cat([trk_exp, cand_exp + task], dim=-1)).squeeze(-1))
        return base, gate

    def forward(
        self,
        identity_pair_logits: torch.Tensor,
        track_semantic: torch.Tensor,
        candidate_semantic: torch.Tensor,
        *,
        task_type: str,
        class_text: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if task_type not in {"mot", "ovmot"}:
            raise ValueError("U2 adapter accepts only mot or ovmot")
        residual, learned_gate = self.semantic_pair_residual(track_semantic, candidate_semantic, class_text, task_type)
        if task_type == "mot":
            gate = torch.zeros_like(learned_gate)
        else:
            gate = learned_gate
        return {
            "identity_pair_logits": identity_pair_logits,
            "semantic_pair_residual": residual,
            "semantic_gate": gate,
            "pair_logits": identity_pair_logits + gate * residual,
        }
