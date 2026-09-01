"""Teacher-anchored, bounded current-frame score contract for Stage L47.

The module deliberately cannot emit an independent membership score.  Every
candidate score is the frozen L29 logit passed through one bounded global scale
and frame offset, plus a small query-conditioned residual.  The residual head
is zero-initialized so a fresh checkpoint is exactly the teacher control.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class L47TeacherAnchoredOutputContract(nn.Module):
    """A small full-frame residual probe with an explicit L29 score anchor."""

    def __init__(self, clip_dim: int = 512, text_dim: int = 768,
                 history_dim: int = 512, numeric_dim: int = 32,
                 hidden: int = 128, heads: int = 4, layers: int = 2,
                 residual_bound: float = 0.05, dropout: float = 0.0):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.config = {
            "clip_dim": int(clip_dim), "text_dim": int(text_dim),
            "history_dim": int(history_dim), "numeric_dim": int(numeric_dim),
            "hidden": int(hidden), "heads": int(heads), "layers": int(layers),
            "residual_bound": float(residual_bound), "dropout": float(dropout),
        }
        self.residual_bound = float(residual_bound)
        self.region_proj = nn.Sequential(
            nn.LayerNorm(clip_dim), nn.Linear(clip_dim, hidden), nn.GELU())
        self.history_proj = nn.Sequential(
            nn.LayerNorm(history_dim), nn.Linear(history_dim, hidden), nn.GELU())
        self.numeric_proj = nn.Sequential(
            nn.LayerNorm(numeric_dim), nn.Linear(numeric_dim, hidden), nn.GELU())
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden), nn.GELU())
        self.query_to_region = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True)
        self.query_norm = nn.LayerNorm(hidden)
        set_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=4 * hidden,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.set_competition = nn.TransformerEncoder(set_layer, num_layers=layers)
        self.candidate_norm = nn.LayerNorm(hidden)
        # Include the frozen teacher value as a conditioning scalar, not as a
        # replacement output.  Zero initialization keeps the initial residual
        # exactly zero while allowing the final head to learn a correction.
        self.residual_head = nn.Sequential(
            nn.Linear(hidden + 1, hidden), nn.GELU(),
            nn.Linear(hidden, 1))
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

        # raw_scale=0 -> a=1; raw_offset=0 -> b_frame=0.
        self.raw_scale = nn.Parameter(torch.zeros(()))
        self.raw_frame_offset = nn.Parameter(torch.zeros(()))

    def _score_map(self, teacher_score: torch.Tensor):
        scale = teacher_score.new_tensor(0.9) + teacher_score.new_tensor(0.2) * torch.sigmoid(self.raw_scale)
        frame_offset = teacher_score.new_tensor(self.residual_bound) * torch.tanh(self.raw_frame_offset)
        return scale, frame_offset, scale * teacher_score + frame_offset

    @staticmethod
    def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(dtype=tokens.dtype).unsqueeze(-1)
        return (tokens * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    def forward(self, region_tokens: torch.Tensor, history_tokens: torch.Tensor,
                numeric: torch.Tensor, query_tokens: torch.Tensor,
                query_mask: torch.Tensor, teacher_score: torch.Tensor):
        """Score one complete current-frame candidate set.

        Shapes are ``region/history [N,D]`` (one frozen observation token per
        candidate), ``numeric [N,F]``, ``query_tokens [T,D_text]`` and
        ``teacher_score [N]``.  A leading singleton batch dimension is also
        accepted for query tokens/masks.
        """
        region_tokens = torch.nan_to_num(region_tokens.float())
        history_tokens = torch.nan_to_num(history_tokens.float())
        numeric = torch.nan_to_num(numeric.float())
        teacher_score = torch.nan_to_num(teacher_score.float())
        if region_tokens.ndim != 2 or history_tokens.ndim != 2 or numeric.ndim != 2:
            raise ValueError("candidate inputs must be [N,D/F]")
        if query_tokens.ndim == 3:
            if query_tokens.shape[0] != 1:
                raise ValueError("only one expression query is supported per unit")
            query_tokens = query_tokens[0]
        if query_mask.ndim == 2:
            if query_mask.shape[0] != 1:
                raise ValueError("only one expression mask is supported per unit")
            query_mask = query_mask[0]
        if query_tokens.ndim != 2 or query_mask.ndim != 1:
            raise ValueError("query inputs must be [T,D] and [T]")
        if region_tokens.shape[0] != teacher_score.shape[0]:
            raise ValueError("candidate/teacher row count mismatch")

        n = region_tokens.shape[0]
        region = self.region_proj(region_tokens)
        history = self.history_proj(history_tokens)
        numeric_value = self.numeric_proj(numeric)
        text = self.text_proj(torch.nan_to_num(query_tokens.float()))
        text_mask = query_mask.bool()
        if not bool(text_mask.any()):
            text_mask = torch.ones_like(text_mask, dtype=torch.bool)
        text_batch = text.unsqueeze(0).expand(n, -1, -1)
        key_padding = ~text_mask.unsqueeze(0).expand(n, -1)
        attended, attention_weights = self.query_to_region(
            (region + history + numeric_value).unsqueeze(1),
            text_batch, text_batch, key_padding_mask=key_padding,
            need_weights=True,
        )
        candidate = self.candidate_norm(
            region + history + numeric_value + self.query_norm(attended[:, 0])
        )
        # Set competition is performed once over the complete current-frame
        # candidate set. It cannot observe source/pool/group/state identifiers.
        set_value = self.set_competition(candidate.unsqueeze(0))[0]
        fused = self.candidate_norm(candidate + set_value)
        scale, frame_offset, score_map = self._score_map(teacher_score)
        residual_input = torch.cat((fused, teacher_score.unsqueeze(-1)), dim=-1)
        residual = self.residual_bound * torch.tanh(self.residual_head(residual_input).squeeze(-1))
        final_score = score_map + residual
        return {
            "score": final_score,
            "final_score": final_score,
            "score_map": score_map,
            "teacher_score": teacher_score,
            "residual": residual,
            "scale": scale,
            "frame_offset": frame_offset,
            "candidate_features": fused,
            "attention_weights": attention_weights,
        }


def teacher_anchored_loss(output: dict, labels: torch.Tensor,
                          hard_indices: torch.Tensor,
                          teacher_correct_pairs: torch.Tensor | None = None,
                          teacher_error_pairs: torch.Tensor | None = None,
                          teacher_order_weight: float = 1.0,
                          teacher_error_weight: float = 0.5) -> tuple[torch.Tensor, dict]:
    """Compute the auditable L47 grouped losses for one frame unit.

    ``teacher_correct_pairs`` and ``teacher_error_pairs`` are ``[P,2]`` index
    tensors containing positive/negative local candidate indices.  Pair losses
    are never formed across frames or expressions.
    """
    score = output["final_score"]
    teacher = output["teacher_score"]
    labels = labels.bool()
    positive = torch.nonzero(labels, as_tuple=False).flatten()
    negative = torch.nonzero(~labels, as_tuple=False).flatten()
    zero = score.new_zeros(())
    hard = hard_indices.to(device=score.device, dtype=torch.long)
    if teacher_correct_pairs is None:
        teacher_correct_pairs = score.new_empty((0, 2), dtype=torch.long)
    else:
        teacher_correct_pairs = teacher_correct_pairs.to(device=score.device, dtype=torch.long)
    if teacher_error_pairs is None:
        teacher_error_pairs = score.new_empty((0, 2), dtype=torch.long)
    else:
        teacher_error_pairs = teacher_error_pairs.to(device=score.device, dtype=torch.long)

    bce = F.binary_cross_entropy_with_logits(score, labels.float()) if len(score) else zero
    if len(positive) and len(hard):
        pair_delta = score[positive, None] - score[hard][None, :]
        pairwise = F.softplus(0.1 - pair_delta).mean()
        listwise = torch.logsumexp(score, dim=0) - torch.logsumexp(score[positive], dim=0)
        min_positive = F.softplus(0.1 - pair_delta).mean(dim=1).mean()
    else:
        pairwise = listwise = min_positive = zero

    if len(teacher_correct_pairs):
        cp = score[teacher_correct_pairs[:, 0]] - score[teacher_correct_pairs[:, 1]]
        teacher_order = F.relu(-cp).mean()
        rank_flip_penalty = F.relu(-cp).mean()
    else:
        teacher_order = rank_flip_penalty = zero
    if len(teacher_error_pairs):
        ep = score[teacher_error_pairs[:, 0]] - score[teacher_error_pairs[:, 1]]
        teacher_error_correction = F.softplus(0.1 - ep).mean()
    else:
        teacher_error_correction = zero

    distillation = F.smooth_l1_loss(score, teacher, beta=1.0) if len(score) else zero
    scale_regularizer = (output["scale"] - 1.0).square()
    offset_regularizer = output["frame_offset"].square()
    residual_l2 = output["residual"].square().mean() if len(score) else zero
    frame_zero_drift = output["residual"].mean().square() if len(score) else zero
    inactive_aux = (
        F.binary_cross_entropy_with_logits(score, torch.zeros_like(score))
        if not len(positive) else zero
    )
    total = (
        bce + teacher_order_weight * teacher_order
        + teacher_error_weight * teacher_error_correction
        + pairwise + 0.5 * listwise + 0.5 * min_positive
        + 0.25 * distillation + 0.1 * inactive_aux
        + 0.1 * rank_flip_penalty + 0.05 * scale_regularizer
        + 0.05 * offset_regularizer + 0.1 * residual_l2
        + 0.1 * frame_zero_drift
    )
    part = {
        "total": float(total.detach()), "membership_bce": float(bce.detach()),
        "teacher_order": float(teacher_order.detach()),
        "teacher_error_correction": float(teacher_error_correction.detach()),
        "pairwise": float(pairwise.detach()), "listwise": float(listwise.detach()),
        "min_positive": float(min_positive.detach()),
        "distillation": float(distillation.detach()),
        "inactive_aux": float(inactive_aux.detach()),
        "rank_flip_penalty": float(rank_flip_penalty.detach()),
        "scale_regularizer": float(scale_regularizer.detach()),
        "offset_regularizer": float(offset_regularizer.detach()),
        "residual_l2": float(residual_l2.detach()),
        "frame_zero_drift": float(frame_zero_drift.detach()),
        "positive_count": int(positive.numel()), "negative_count": int(negative.numel()),
        "hard_count": int(hard.numel()),
    }
    return total, part
