"""Integrated query/region/track decoder for LocateMOT Stage L44.

The module is RMOT-only and intentionally accepts tensors, not identifiers.
It keeps word tokens, current region tokens, same-frame set context and a
causal persistent-track history in one forward pass.  The L29 score is an
explicit frozen anchor: the trainable branch produces a bounded residual
around it, while all auxiliary heads remain diagnostics/loss inputs rather
than source or pool shortcuts.
"""
from __future__ import annotations

import torch
from torch import nn


class L44IntegratedQueryRegionTrackDecoder(nn.Module):
    """Small integrated current-frame RMOT decoder.

    Unbatched input shapes used by the L44 smoke are:

    * ``patch_tokens``: ``[N, P, image_dim]``;
    * ``text_tokens``: ``[T, text_dim]`` and ``text_mask``: ``[T]``;
    * ``numeric``: ``[N, numeric_dim]``;
    * ``history``: ``[N, H, track_dim]`` with ``history_mask``/``history_time``
      shaped ``[N, H]``;
    * ``teacher`` and ``candidate_mask``: ``[N]``.

    A leading batch dimension is deliberately not accepted here.  Keeping a
    complete candidate set as one explicit unit makes the frame contract
    auditable and avoids accidental cross-query padding/aggregation.
    """

    def __init__(self, image_dim: int = 768, text_dim: int = 768,
                 numeric_dim: int = 36, track_dim: int = 1432,
                 hidden: int = 256, heads: int = 8, layers: int = 2,
                 history_len: int = 8, residual_bound: float = 0.5):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        if layers < 1 or history_len < 1:
            raise ValueError("layers and history_len must be positive")
        self.config = {
            "image_dim": image_dim, "text_dim": text_dim,
            "numeric_dim": numeric_dim, "track_dim": track_dim,
            "hidden": hidden, "heads": heads, "layers": layers,
            "history_len": history_len, "residual_bound": residual_bound,
        }
        self.residual_bound = float(residual_bound)

        self.image_proj = nn.Sequential(
            nn.LayerNorm(image_dim), nn.Linear(image_dim, hidden), nn.GELU())
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim), nn.Linear(text_dim, hidden), nn.GELU())
        self.region_coord = nn.Sequential(
            nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.query_region_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden, heads, dropout=0.0,
                                  batch_first=True)
            for _ in range(layers)
        ])
        self.query_region_norms = nn.ModuleList(
            [nn.LayerNorm(hidden) for _ in range(layers)])

        self.numeric_proj = nn.Sequential(
            nn.LayerNorm(numeric_dim), nn.Linear(numeric_dim, hidden), nn.GELU())
        self.current_fuse = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.set_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden, heads, dropout=0.0,
                                  batch_first=True)
            for _ in range(layers)
        ])
        self.set_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])

        self.history_proj = nn.Sequential(
            nn.LayerNorm(track_dim), nn.Linear(track_dim, hidden), nn.GELU())
        self.history_time = nn.Sequential(
            nn.Linear(1, hidden), nn.Tanh())
        # GRU is causal in the supplied temporal order.  Padded history is
        # zeroed before the recurrent pass and the last valid state is used.
        self.history_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.track_fuse = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))

        self.current_head = nn.Linear(hidden, 1)
        self.set_head = nn.Linear(hidden, 1)
        self.persistent_head = nn.Linear(hidden, 1)
        self.history_head = nn.Linear(hidden, 1)
        self.continuation_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self.stale_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

        # The final head is not a free scorer: it is a residual conditioned on
        # current query-region interaction, same-frame set context and track
        # history.  Zero initialization makes the initial model exactly the
        # frozen teacher when a teacher tensor is supplied.
        self.final_residual_head = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1))
        self.null_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))
        self._reset_final_head()

    def _reset_final_head(self):
        last = self.final_residual_head[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    @staticmethod
    def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(dtype=x.dtype).unsqueeze(-1)
        return (x * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _default_coords(count: int, device, dtype):
        side = max(1, int(round(float(count) ** 0.5)))
        if side * side != count:
            # Fixed order is still deterministic for non-square token counts.
            x = torch.linspace(0.0, 1.0, count, device=device, dtype=dtype)
            return torch.stack((x, torch.zeros_like(x)), dim=-1)
        x = torch.linspace(0.0, 1.0, side, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(x, x, indexing="ij")
        return torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)

    def _check_shapes(self, patch_tokens, text_tokens, numeric, history,
                      history_mask, history_time, candidate_mask, text_mask,
                      teacher, region_coords):
        if patch_tokens.ndim != 3 or text_tokens.ndim != 2:
            raise ValueError("L44 expects unbatched patch [N,P,D] and text [T,D]")
        n, p, _ = patch_tokens.shape
        if n < 1 or p < 1 or text_tokens.shape[0] < 1:
            raise ValueError("candidate and text sequences must be non-empty")
        if numeric.shape != (n, self.config["numeric_dim"]):
            raise ValueError(f"numeric shape {tuple(numeric.shape)} is not {(n, self.config['numeric_dim'])}")
        if history.shape != (n, self.config["history_len"], self.config["track_dim"]):
            raise ValueError("history is not aligned with the current candidate set")
        if history_mask.shape != history_time.shape != (n, self.config["history_len"]):
            raise ValueError("history masks/times are not aligned")
        if candidate_mask.shape != (n,) or teacher.shape != (n,):
            raise ValueError("teacher/candidate mask is not aligned with candidate rows")
        if text_mask.shape != (text_tokens.shape[0],):
            raise ValueError("text mask is not aligned with text tokens")
        if region_coords is not None and region_coords.shape != (n, p, 2):
            raise ValueError("region coordinates must be [N,P,2]")

    def forward(self, patch_tokens: torch.Tensor, text_tokens: torch.Tensor,
                numeric: torch.Tensor, history: torch.Tensor,
                history_mask: torch.Tensor, history_time: torch.Tensor,
                teacher: torch.Tensor, candidate_mask: torch.Tensor | None = None,
                text_mask: torch.Tensor | None = None,
                region_coords: torch.Tensor | None = None):
        n = patch_tokens.shape[0]
        device = patch_tokens.device
        if candidate_mask is None:
            candidate_mask = torch.ones(n, dtype=torch.bool, device=device)
        if text_mask is None:
            text_mask = torch.ones(text_tokens.shape[0], dtype=torch.bool, device=device)
        if region_coords is None:
            coords = self._default_coords(patch_tokens.shape[1], device,
                                          patch_tokens.dtype)
            region_coords = coords.unsqueeze(0).expand(n, -1, -1)
        self._check_shapes(patch_tokens, text_tokens, numeric, history,
                           history_mask, history_time, candidate_mask, text_mask,
                           teacher, region_coords)

        valid = candidate_mask.bool()
        patch = self.image_proj(torch.nan_to_num(patch_tokens.float()))
        patch = patch + self.region_coord(torch.nan_to_num(region_coords.float()))
        text = self.text_proj(torch.nan_to_num(text_tokens.float()))
        query = text.unsqueeze(0).expand(n, -1, -1)
        for attn, norm in zip(self.query_region_layers, self.query_region_norms):
            attended, _ = attn(query, patch, patch, need_weights=False)
            query = norm(query + attended)
        text_mask = text_mask.bool()
        query_pool = self.masked_mean(query, text_mask.unsqueeze(0).expand(n, -1))
        patch_pool = patch.mean(dim=1)
        numeric_h = self.numeric_proj(torch.nan_to_num(numeric.float()))
        current_h = self.current_fuse(torch.cat((query_pool, patch_pool, numeric_h), dim=-1))
        current_h = current_h.masked_fill(~valid[:, None], 0.0)

        set_h = current_h
        key_padding = ~valid
        for attn, norm in zip(self.set_layers, self.set_norms):
            attended, _ = attn(set_h.unsqueeze(0), set_h.unsqueeze(0),
                               set_h.unsqueeze(0),
                               key_padding_mask=key_padding.unsqueeze(0),
                               need_weights=False)
            set_h = norm(set_h + attended[0])
            set_h = set_h.masked_fill(~valid[:, None], 0.0)

        hist = self.history_proj(torch.nan_to_num(history.float()))
        hist = hist + self.history_time(torch.nan_to_num(history_time.float()).unsqueeze(-1))
        hmask = history_mask.bool()
        hist = hist.masked_fill(~hmask[:, :, None], 0.0)
        hist_out, _ = self.history_gru(hist)
        last = hmask.long().sum(dim=1).clamp_min(1) - 1
        history_h = hist_out[torch.arange(n, device=device), last]
        history_h = history_h.masked_fill(~valid[:, None], 0.0)

        track_h = self.track_fuse(torch.cat((set_h, history_h), dim=-1))
        track_h = track_h.masked_fill(~valid[:, None], 0.0)
        pair_context = torch.cat((current_h, set_h, track_h, query_pool), dim=-1)
        residual_raw = self.final_residual_head(pair_context).squeeze(-1)
        residual = self.residual_bound * torch.tanh(residual_raw)
        teacher = torch.nan_to_num(teacher.float())
        final = teacher + residual
        final = final.masked_fill(~valid, -20.0)

        history_query = hist + query_pool[:, None, :]
        history_logits = self.history_head(history_query).squeeze(-1)
        history_logits = history_logits.masked_fill(~hmask, -20.0)
        continuation = self.continuation_head(torch.cat((current_h, history_h), dim=-1)).squeeze(-1)
        stale = self.stale_head(torch.cat((current_h, history_h), dim=-1)).squeeze(-1)
        continuation = continuation.masked_fill(~valid, -20.0)
        stale = stale.masked_fill(~valid, -20.0)
        set_pool = self.masked_mean(set_h, valid.unsqueeze(0))[:1]
        null_logit = self.null_head(torch.cat((query_pool.mean(0, keepdim=True), set_pool), dim=-1)).squeeze()

        return {
            "final_membership_logits": final,
            "current_membership_logits": self.current_head(current_h).squeeze(-1).masked_fill(~valid, -20.0),
            "set_membership_logits": self.set_head(set_h).squeeze(-1).masked_fill(~valid, -20.0),
            "persistent_track_logits": self.persistent_head(track_h).squeeze(-1).masked_fill(~valid, -20.0),
            "teacher_score": teacher,
            "residual": residual.masked_fill(~valid, 0.0),
            "history_membership_logits": history_logits,
            "continuation_logits": continuation,
            "stale_logits": stale,
            "null_logit": null_logit,
            "candidate_features": track_h,
            "set_features": set_h,
            "query_region_features": query,
            "candidate_mask": valid,
        }
