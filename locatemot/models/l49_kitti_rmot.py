"""L49 RMOT-only semantic, persistent-track and NULL/sequence model.

The model consumes complete frozen candidate sets and expression-level word
tokens.  It has no tracker, source/pool/group/state feature, or ordinary
MOT/OVMOT import.  The semantic matcher is the L48 core; later branches are
bounded auxiliary evidence added only after the semantic warm-up.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from locatemot.models.l48_joint_rmot import L48SemanticMatcher


class L49KittiRMOT(nn.Module):
    def __init__(self, hidden: int = 256, heads: int = 4,
                 history_length: int = 8, dropout: float = 0.1):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = int(hidden)
        self.history_length = int(history_length)
        self.semantic = L48SemanticMatcher(hidden=hidden, heads=heads,
                                           dropout=dropout)
        self.history_input = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, hidden), nn.GELU())
        self.history_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.numeric = nn.Sequential(
            nn.LayerNorm(7 + 8 + 8 + 8 + 1 + 4), nn.Linear(7 + 8 + 8 + 8 + 1 + 4, hidden), nn.GELU())
        self.current_clip = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, hidden), nn.GELU())
        self.query_pool = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU())
        self.identity_fusion = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.identity_head = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.continuation_head = nn.Sequential(nn.Linear(2 * hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.sequence_head = nn.Sequential(nn.Linear(2 * hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.null_head = nn.Sequential(nn.Linear(2 * hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    @staticmethod
    def _unbatch(value: torch.Tensor) -> torch.Tensor:
        if value.ndim == 3 and value.shape[0] == 1:
            return value[0]
        return value

    @staticmethod
    def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        denom = mask.sum(-1, keepdim=True).clamp_min(1).to(tokens.dtype)
        return (tokens * mask.unsqueeze(-1).to(tokens.dtype)).sum(-2) / denom

    def _history_state(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sequence = self._unbatch(sequence.float())
        mask = self._unbatch(mask.bool())
        if sequence.ndim != 3 or sequence.shape[-1] != 512:
            raise ValueError(f"history_sequence must be [N,H,512], got {tuple(sequence.shape)}")
        encoded = self.history_input(torch.nan_to_num(sequence))
        output, _ = self.history_gru(encoded)
        lengths = mask.long().sum(-1).clamp_min(1)
        return output[torch.arange(len(output), device=output.device), lengths - 1]

    def forward(self, clip: torch.Tensor, history_clip: torch.Tensor,
                geometry: torch.Tensor, motion: torch.Tensor,
                context: torch.Tensor, lifecycle: torch.Tensor,
                objectness: torch.Tensor, text_tokens: torch.Tensor,
                text_mask: torch.Tensor, relation: torch.Tensor | None = None,
                history_sequence: torch.Tensor | None = None,
                history_mask: torch.Tensor | None = None,
                stage: str = "full") -> dict[str, torch.Tensor | dict]:
        clip = self._unbatch(clip.float())
        history_clip = self._unbatch(history_clip.float())
        geometry = self._unbatch(geometry.float())
        motion = self._unbatch(motion.float())
        context = self._unbatch(context.float())
        lifecycle = self._unbatch(lifecycle.float())
        objectness = self._unbatch(objectness.float()).reshape(-1)
        text_tokens = self._unbatch(text_tokens.float())
        text_mask = self._unbatch(text_mask.bool())
        if relation is None:
            relation = geometry.new_zeros((len(clip), 4))
        relation = self._unbatch(relation.float())
        n = len(clip)
        if n == 0 or any(value.shape[0] != n for value in
                         (history_clip, geometry, motion, context, lifecycle, objectness, relation)):
            raise ValueError("L49 candidate streams are not aligned or empty")
        if history_sequence is None:
            history_sequence = history_clip.unsqueeze(1)
            history_mask = torch.ones((n, 1), dtype=torch.bool, device=clip.device)
        else:
            history_sequence = history_sequence.to(clip.device)
            history_mask = history_mask.to(clip.device) if history_mask is not None else torch.ones(
                history_sequence.shape[:2], dtype=torch.bool, device=clip.device)
        sem = self.semantic(clip, history_clip, geometry, motion, context,
                            lifecycle, objectness, text_tokens, text_mask, relation)
        semantic_logit = sem["semantic_logit"]
        candidate_embedding = sem["candidate_embedding"]
        query_hidden = self.query_pool(torch.nan_to_num(text_tokens))
        query = self.masked_mean(query_hidden, text_mask).expand(n, -1)
        hist = self._history_state(history_sequence, history_mask)
        numeric_input = torch.cat((geometry, motion, context, lifecycle,
                                   objectness[:, None], relation), -1)
        numeric = self.numeric(torch.nan_to_num(numeric_input))
        current = self.current_clip(torch.nan_to_num(clip))
        identity_input = torch.cat((candidate_embedding, hist, current, query), -1)
        identity_state = self.identity_fusion(identity_input)
        identity_logit = self.identity_head(identity_state).squeeze(-1)
        continuation_input = torch.cat((hist, numeric), -1)
        continuation_logit = self.continuation_head(continuation_input).squeeze(-1)
        sequence_logit = self.sequence_head(torch.cat((identity_state, hist), -1)).squeeze(-1)
        set_summary = candidate_embedding.mean(0)
        null_logit = self.null_head(torch.cat((query[:1], set_summary[None]), -1)).squeeze(-1)
        if stage in ("semantic", "semantic_warmup"):
            identity_delta = torch.zeros_like(semantic_logit)
            continuation_delta = torch.zeros_like(semantic_logit)
            sequence_delta = torch.zeros_like(semantic_logit)
        else:
            # These bounded terms are auxiliary evidence; semantic_logit stays
            # the main expression-conditioned current-frame score.
            identity_delta = 0.20 * torch.tanh(identity_logit)
            continuation_delta = 0.10 * torch.tanh(continuation_logit)
            sequence_delta = 0.05 * torch.tanh(sequence_logit)
        final_logit = semantic_logit + identity_delta + continuation_delta + sequence_delta
        return {
            "semantic_logit": semantic_logit,
            "identity_logit": identity_logit,
            "continuation_logit": continuation_logit,
            "sequence_logit": sequence_logit,
            "null_logit": null_logit,
            "final_logit": final_logit,
            "candidate_embedding": candidate_embedding,
            "identity_state": identity_state,
            "stream_norms": sem["stream_norms"],
            "deltas": {"identity": identity_delta, "continuation": continuation_delta, "sequence": sequence_delta},
        }

    def semantic_parameters(self):
        return self.semantic.parameters()

    def auxiliary_parameters(self):
        semantic_ids = {id(x) for x in self.semantic.parameters()}
        return [x for x in self.parameters() if id(x) not in semantic_ids]

    def config(self) -> dict:
        return {
            "format": "locatemot-l49-kitti-rmot-v1", "hidden": self.hidden,
            "heads": 4, "history_length": self.history_length,
            "semantic_core": "L48SemanticMatcher", "attention_layers": {"semantic": 2, "history": "GRU"},
            "stages": ["semantic_warmup", "identity_continuation", "null_sequence"],
            "final_emission": "semantic_logit + bounded identity/continuation/sequence auxiliary",
            "bounded_auxiliary_scales": {"identity": 0.20, "continuation": 0.10, "sequence": 0.05},
            "inputs_excluded": ["source_id", "pool_id", "group_id", "state_key", "query_id_as_feature"],
            "token_span_region_alignment": "UNALIGNED",
            "static_motion_language_mask": "UNALIGNED/not claimed",
            "rmot_only": True, "ordinary_mot_ovmot_imported": False,
        }


def brier_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(torch.sigmoid(logits), target.float()) if len(logits) else logits.new_zeros(())
