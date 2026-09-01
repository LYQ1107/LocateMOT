"""RMOT-only end-to-end query/region/track decoder for Stage L46.

The module consumes already frozen L19 region features and L28 track-history
features.  It deliberately accepts no pool/source/group/state identifiers.
The current frame is the only candidate set dimension exposed to the
membership head; history and continuation are auxiliary sequence signals.
"""
from __future__ import annotations

import torch
from torch import nn


class L46EndToEndQueryRegionTrackDecoder(nn.Module):
    """Word-token to complete current-region set plus persistent-track model.

    Inputs are unbatched so a call corresponds to one complete
    ``(video, query, frame)`` unit:

    * ``region_tokens``: ``[N, P, region_dim]``;
    * ``text_tokens``/``text_mask``: ``[T, text_dim]``/``[T]``;
    * ``numeric``: ``[N, numeric_dim]``;
    * ``history``/masks/times: ``[N, H, track_dim]``/``[N, H]``;
    * ``candidate_mask``: ``[N]``.

    ``teacher`` is returned for diagnostics and distillation by the training
    entry point, but is not fed into the learned emission path.  This keeps
    the L29 score a frozen control/auxiliary target rather than a semantic
    shortcut.
    """

    def __init__(self, region_dim: int = 512, text_dim: int = 768,
                 numeric_dim: int = 36, track_dim: int = 1432,
                 hidden: int = 256, heads: int = 8, layers: int = 2,
                 history_len: int = 8, dropout: float = 0.0):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        if layers < 1 or history_len < 1:
            raise ValueError("layers and history_len must be positive")
        self.config = {
            "region_dim": int(region_dim), "text_dim": int(text_dim),
            "numeric_dim": int(numeric_dim), "track_dim": int(track_dim),
            "hidden": int(hidden), "heads": int(heads),
            "layers": int(layers), "history_len": int(history_len),
            "dropout": float(dropout),
        }

        self.region_proj = nn.Sequential(
            nn.LayerNorm(region_dim), nn.Linear(region_dim, hidden), nn.GELU())
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden), nn.GELU())
        self.coord_proj = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.region_text_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden, heads, dropout=dropout,
                                  batch_first=True)
            for _ in range(layers)
        ])
        self.region_text_norm = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(layers)])

        self.numeric_proj = nn.Sequential(
            nn.LayerNorm(numeric_dim), nn.Linear(numeric_dim, hidden), nn.GELU())
        self.current_fuse = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))

        self.set_attn = nn.ModuleList([
            nn.MultiheadAttention(hidden, heads, dropout=dropout,
                                  batch_first=True)
            for _ in range(layers)
        ])
        self.set_norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.set_ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, 4 * hidden), nn.GELU(),
                          nn.Linear(4 * hidden, hidden))
            for _ in range(layers)
        ])
        self.set_ffn_norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

        self.history_proj = nn.Sequential(
            nn.LayerNorm(track_dim), nn.Linear(track_dim, hidden), nn.GELU())
        self.history_time_proj = nn.Sequential(nn.Linear(1, hidden), nn.Tanh())
        self.history_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.history_membership_head = nn.Linear(hidden, 1)
        self.track_fuse = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))

        self.membership_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.sequence_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.continuation_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.null_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    @staticmethod
    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(dtype=values.dtype).unsqueeze(-1)
        return (values * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def default_coords(n: int, p: int, device, dtype) -> torch.Tensor:
        # Region tokens from L19 are pooled (P=1), but this deterministic
        # coordinate contract also supports future frozen patch-token inputs.
        if p == 1:
            return torch.full((n, 1, 2), 0.5, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, p, device=device, dtype=dtype)
        return torch.stack((x, torch.zeros_like(x)), dim=-1).unsqueeze(0).expand(n, -1, -1)

    def _check(self, region_tokens, text_tokens, numeric, history,
               history_mask, history_time, candidate_mask, text_mask,
               region_coords):
        if region_tokens.ndim != 3 or text_tokens.ndim != 2:
            raise ValueError("L46 expects unbatched [N,P,D] regions and [T,D] text")
        n, p, _ = region_tokens.shape
        if n < 1 or p < 1 or text_tokens.shape[0] < 1:
            raise ValueError("candidate and text sequences must be non-empty")
        if numeric.shape != (n, self.config["numeric_dim"]):
            raise ValueError("numeric is not aligned with region candidates")
        expected_history = (n, self.config["history_len"], self.config["track_dim"])
        if history.shape != expected_history:
            raise ValueError(f"history shape {tuple(history.shape)} != {expected_history}")
        expected_hmask = (n, self.config["history_len"])
        if history_mask.shape != expected_hmask or history_time.shape != expected_hmask:
            raise ValueError("history masks/times are not aligned")
        if candidate_mask.shape != (n,) or text_mask.shape != (text_tokens.shape[0],):
            raise ValueError("candidate/text masks are not aligned")
        if region_coords.shape != (n, p, 2):
            raise ValueError("region_coords must be [N,P,2]")

    def forward(self, region_tokens: torch.Tensor, text_tokens: torch.Tensor,
                numeric: torch.Tensor, history: torch.Tensor,
                history_mask: torch.Tensor, history_time: torch.Tensor,
                candidate_mask: torch.Tensor | None = None,
                text_mask: torch.Tensor | None = None,
                teacher: torch.Tensor | None = None,
                region_coords: torch.Tensor | None = None):
        n, p, _ = region_tokens.shape
        device = region_tokens.device
        if candidate_mask is None:
            candidate_mask = torch.ones(n, dtype=torch.bool, device=device)
        if text_mask is None:
            text_mask = torch.ones(text_tokens.shape[0], dtype=torch.bool, device=device)
        if region_coords is None:
            region_coords = self.default_coords(n, p, device, region_tokens.dtype)
        self._check(region_tokens, text_tokens, numeric, history, history_mask,
                    history_time, candidate_mask, text_mask, region_coords)
        valid = candidate_mask.bool()

        region = self.region_proj(torch.nan_to_num(region_tokens.float()))
        region = region + self.coord_proj(torch.nan_to_num(region_coords.float()))
        text = self.text_proj(torch.nan_to_num(text_tokens.float()))
        text_key_mask = ~text_mask.bool().unsqueeze(0).expand(n, -1)
        cross_entropies = []
        for attn, norm in zip(self.region_text_attn, self.region_text_norm):
            attended, weights = attn(
                region, text.unsqueeze(0).expand(n, -1, -1),
                text.unsqueeze(0).expand(n, -1, -1),
                key_padding_mask=text_key_mask, need_weights=True,
                average_attn_weights=False)
            region = norm(region + attended)
            # weights: [N, heads, P, T].  Entropy is diagnostic only.
            probs = weights.float().clamp_min(1e-8)
            cross_entropies.append(float((-(probs * probs.log()).sum(-1)).mean().detach()))
        text_pool = self.masked_mean(text.unsqueeze(0).expand(n, -1, -1),
                                     text_mask.unsqueeze(0).expand(n, -1))
        region_pool = region.mean(dim=1)
        numeric_h = self.numeric_proj(torch.nan_to_num(numeric.float()))
        current = self.current_fuse(torch.cat((region_pool, text_pool, numeric_h), dim=-1))
        current = current.masked_fill(~valid[:, None], 0.0)

        set_h = current
        padding = ~valid
        set_entropies = []
        for attn, norm, ffn, ffn_norm in zip(
                self.set_attn, self.set_norm, self.set_ffn, self.set_ffn_norm):
            attended, weights = attn(
                set_h.unsqueeze(0), set_h.unsqueeze(0), set_h.unsqueeze(0),
                key_padding_mask=padding.unsqueeze(0), need_weights=True,
                average_attn_weights=False)
            set_h = norm(set_h + attended[0])
            set_h = ffn_norm(set_h + ffn(set_h))
            set_h = set_h.masked_fill(~valid[:, None], 0.0)
            probs = weights.float().clamp_min(1e-8)
            set_entropies.append(float((-(probs * probs.log()).sum(-1)).mean().detach()))

        hist = self.history_proj(torch.nan_to_num(history.float()))
        hist = hist + self.history_time_proj(
            torch.nan_to_num(history_time.float()).unsqueeze(-1))
        hmask = history_mask.bool()
        hist = hist.masked_fill(~hmask[:, :, None], 0.0)
        hist_out, _ = self.history_gru(hist)
        last = hmask.long().sum(dim=1).clamp_min(1) - 1
        history_h = hist_out[torch.arange(n, device=device), last]
        history_h = history_h.masked_fill(~valid[:, None], 0.0)
        history_membership = self.history_membership_head(hist_out).squeeze(-1)
        history_membership = history_membership.masked_fill(~hmask, -20.0)

        track_h = self.track_fuse(torch.cat((set_h, history_h), dim=-1))
        track_h = track_h.masked_fill(~valid[:, None], 0.0)
        membership = self.membership_head(track_h).squeeze(-1).masked_fill(~valid, -20.0)
        sequence = self.sequence_head(track_h).squeeze(-1).masked_fill(~valid, -20.0)
        continuation = self.continuation_head(torch.cat((set_h, history_h), dim=-1))
        continuation = continuation.squeeze(-1).masked_fill(~valid, -20.0)
        set_pool = self.masked_mean(set_h, valid.unsqueeze(0))[:1]
        null = self.null_head(torch.cat((text_pool[:1], set_pool), dim=-1)).squeeze()
        return {
            "membership_logits": membership,
            "sequence_logits": sequence,
            "continuation_logits": continuation,
            "history_membership_logits": history_membership,
            "null_logit": null,
            "candidate_features": track_h,
            "current_features": current,
            "set_features": set_h,
            "region_text_features": region,
            "teacher_score": torch.nan_to_num(teacher.float()) if teacher is not None else None,
            "candidate_mask": valid,
            "text_region_attention_entropy": float(sum(cross_entropies) / max(1, len(cross_entropies))),
            "set_attention_entropy": float(sum(set_entropies) / max(1, len(set_entropies))),
        }
