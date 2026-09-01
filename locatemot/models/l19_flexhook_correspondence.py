"""Learned expression-to-tracklet correspondence for Stage L19.

This module ports the *structural* FlexHook recipe audited in the local
official repository (commit ``bd1acc3``):

* text-conditioned hook tokens predict sampling locations;
* a feature map is sampled at those locations with ``grid_sample``;
* C-Hook/PCD cross-attention fuses text, sampled features, and object
  features; and
* a causal memory supplies temporal correspondence.

The official model samples 2-D ROPE-Swin maps.  LocateMOT does not have the
official raw-image feature maps or checkpoint, so this controlled port uses
each tracklet's causal observation stack ``[K, hidden]`` as a 1-D feature map
(``[hidden, 1, K]``).  The LocalAnything main bank and GroundingDINO reserve
bank remain frozen inputs; no IoU/CLIP linker and no CARR coverage gate are
used here.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from locatemot.models.l16_track_selector import FAMILY_NAMES


class Projection(nn.Module):
    """Small normalized projection for one frozen bank feature view."""

    def __init__(self, input_dim: int, hidden: int, dropout: float = 0.10):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(torch.nan_to_num(value.float()))


class PCDCrossAttention(nn.Module):
    """FlexHook-style projected cross-attention.

    The official ``CAttention`` uses separate q/k/v projections and scaled
    dot-product attention.  This implementation keeps that contract while
    accepting a key-padding mask for cached text tokens.
    """

    def __init__(self, hidden: int, heads: int, dropout: float = 0.10):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.head_dim = hidden // heads
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)
        self.dropout = float(dropout)

    @staticmethod
    def _add_pos(value: torch.Tensor, position: torch.Tensor | None):
        return value if position is None else value + position

    def forward(self, query: torch.Tensor, memory: torch.Tensor,
                query_pos: torch.Tensor | None = None,
                memory_pos: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, query_len, _ = query.shape
        _, memory_len, _ = memory.shape
        q = self.q_proj(self._add_pos(query, query_pos))
        k = self.k_proj(self._add_pos(memory, memory_pos))
        v = self.v_proj(memory)
        q = q.view(batch, query_len, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, memory_len, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, memory_len, self.heads, self.head_dim).transpose(1, 2)
        attention_mask = None
        if key_padding_mask is not None:
            # scaled_dot_product_attention accepts a boolean allow-mask;
            # ``key_padding_mask`` follows MultiheadAttention's True=pad
            # convention, hence the inversion.
            attention_mask = (~key_padding_mask.bool()).unsqueeze(1).unsqueeze(1)
        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous().view(batch, query_len, -1)
        return self.out_proj(output)


class PCDTransformerBlock(nn.Module):
    """Pre-normalized C-Hook/PCD block matching FlexHook's residual layout."""

    def __init__(self, hidden: int, heads: int, dropout: float = 0.10):
        super().__init__()
        self.q_norm = nn.LayerNorm(hidden)
        self.kv_norm = nn.LayerNorm(hidden)
        self.attention = PCDCrossAttention(hidden, heads, dropout)
        self.ffn_norm = nn.LayerNorm(hidden)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden, 4 * hidden, bias=False), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(4 * hidden, hidden, bias=False),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor,
                query_pos: torch.Tensor | None = None,
                memory_pos: torch.Tensor | None = None,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attended = self.attention(
            self.q_norm(query), self.kv_norm(memory), query_pos, memory_pos,
            key_padding_mask)
        hidden = query + self.dropout(attended)
        return hidden + self.dropout(self.feed_forward(self.ffn_norm(hidden)))


class L19FlexHookCorrespondence(nn.Module):
    """Causal, learned expression-to-tracklet correspondence model."""

    def __init__(self, hidden: int = 256, heads: int = 4,
                 dropout: float = 0.10, token_dim: int = 512,
                 temporal_points: int = 8, hook_points: int = 10):
        super().__init__()
        self.hidden = int(hidden)
        self.temporal_points = int(temporal_points)
        self.hook_points = int(hook_points)
        self.token_dim = int(token_dim)
        if self.temporal_points < 2 or self.hook_points < 1:
            raise ValueError("temporal_points must be >=2 and hook_points positive")

        # Cached CLIP token states are the available text encoder output.  The
        # learned positional stream and PCD decoder retain token-level text,
        # rather than reducing the expression to a hand-authored taxonomy.
        self.text_projection = nn.Sequential(
            nn.LayerNorm(token_dim), nn.Linear(token_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.LayerNorm(hidden),
        )
        self.text_position = nn.Parameter(torch.zeros(1, 77, hidden))
        nn.init.normal_(self.text_position, std=0.02)
        self.text_sentence = Projection(token_dim, hidden, dropout)
        self.spec_projection = Projection(512, hidden, dropout)
        self.family_projection = Projection(len(FAMILY_NAMES), hidden, dropout)
        self.query_fusion = nn.Sequential(
            nn.Linear(3 * hidden, 2 * hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden, hidden),
            nn.LayerNorm(hidden),
        )

        # Official FlexHook starts from learned conditional positions, decodes
        # them against the text, and projects them to [-1, 1].  Here a hook
        # coordinate is one temporal axis instead of two image axes.
        self.conditional_pos_embed = nn.Parameter(
            torch.randn(1, self.hook_points, hidden) * 0.02)
        self.conditional_pos_decoder = PCDTransformerBlock(
            hidden, heads, dropout)
        self.conditional_pos_projector = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 1),
        )
        self.hook_norm = nn.LayerNorm(hidden)
        self.hook_projection = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
        )

        # Frozen LocalAnything/DINO views.  Reserve identity features are
        # consumed exactly as another observation view; they are not linked by
        # a separate heuristic.
        self.clip_current = Projection(512, hidden, dropout)
        self.clip_history = Projection(512, hidden, dropout)
        self.pbd = Projection(2048, hidden, dropout)
        self.identity = Projection(384, hidden, dropout)
        self.numeric = Projection(32, hidden, dropout)
        self.source_embedding = nn.Embedding(2, hidden)
        self.observation = nn.Sequential(
            nn.Linear(6 * hidden, 2 * hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.temporal_update = nn.GRUCell(hidden, hidden)
        self.temporal_position = nn.Parameter(
            torch.zeros(1, self.temporal_points, hidden))
        nn.init.normal_(self.temporal_position, std=0.02)
        self.register_buffer(
            "temporal_alpha",
            torch.linspace(0.0, 1.0, self.temporal_points).reshape(1, -1, 1),
        )

        # First aggregate the causal observation stack, then fuse text,
        # sampled hook features, and the full temporal map through PCD.
        self.temporal_pcd = PCDTransformerBlock(hidden, heads, dropout)
        self.output_pcd = PCDTransformerBlock(hidden, heads, dropout)
        self.correspondence_norm = nn.LayerNorm(hidden)
        self.association_projection = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
        )
        self.score_head = nn.Sequential(
            nn.Linear(5 * hidden + 1, 2 * hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # The track score learns expression↔tracklet compatibility.  The
        # observation score is a second learned readout for whether the
        # current sampled observation is the expression target; combining the
        # two keeps a temporally valid track from turning into a permanent
        # positive when its current box is absent.  This is not a source gate
        # and has no hand-written geometric/CLIP term.
        self.observation_score_head = nn.Sequential(
            nn.Linear(2 * hidden + 33, 2 * hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.observation_scale = nn.Parameter(torch.tensor(0.35))
        self.presence_head = nn.Sequential(
            nn.Linear(hidden + 32, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
        self.detach_state = True

    def query_context(self, query_tokens: torch.Tensor | None,
                      query: torch.Tensor, family: torch.Tensor,
                      query_mask: torch.Tensor | None = None) -> dict:
        if query_tokens is None:
            query_tokens = query.float().reshape(1, -1)
            query_mask = None
        tokens = torch.nan_to_num(query_tokens.float())
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
        if tokens.ndim != 3 or tokens.shape[0] != 1:
            raise ValueError(f"expected [T,D] or [1,T,D], got {tuple(tokens.shape)}")
        if query_mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool,
                              device=tokens.device)
        else:
            mask = query_mask.to(tokens.device).bool()
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if mask.shape != tokens.shape[:2]:
                raise ValueError("query token mask/state shape mismatch")
        if tokens.shape[1] > self.text_position.shape[1]:
            raise ValueError("cached token sequence is longer than text position table")
        projected = self.text_projection(tokens)
        projected = projected + self.text_position[:, :projected.shape[1]]
        count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(projected.dtype)
        pooled_raw = (tokens * mask.unsqueeze(-1)).sum(dim=1) / count
        pooled = self.text_sentence(pooled_raw)[0]
        spec = self.spec_projection(query.float().reshape(1, -1))[0]
        family_h = self.family_projection(family.float().reshape(1, -1))[0]
        holistic = F.normalize(
            self.query_fusion(torch.cat((pooled, spec, family_h), dim=-1)), dim=-1)

        hook_query = self.conditional_pos_embed + pooled.reshape(1, 1, -1)
        hook_query = self.conditional_pos_decoder(
            hook_query, projected, key_padding_mask=~mask)
        hook_coords = self.conditional_pos_projector(hook_query).squeeze(-1)
        hook_coords = hook_coords.sigmoid().mul(2.0).sub(1.0)[0]
        hooks = self.hook_projection(self.hook_norm(hook_query))[0]
        return {
            "tokens": projected[0], "mask": mask[0], "pooled": pooled,
            "spec": spec, "family": family_h, "holistic": holistic,
            "hooks": hooks, "hook_coords": hook_coords,
        }

    @staticmethod
    def _previous(track_ids: torch.Tensor, state: Dict[int, object],
                  hidden: int, temporal_points: int,
                  device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        zero = torch.zeros((temporal_points, hidden), device=device)
        memories, present = [], []
        for raw_id in track_ids.detach().cpu().tolist():
            value = state.get(int(raw_id))
            if isinstance(value, dict):
                value = value.get("memory")
            valid = (isinstance(value, torch.Tensor) and value.ndim == 2 and
                     value.shape[-1] == hidden and value.shape[0] >= 1)
            if valid:
                memory = value[-temporal_points:]
                if memory.shape[0] < temporal_points:
                    padding = zero[:temporal_points - memory.shape[0]]
                    memory = torch.cat((padding, memory), dim=0)
            else:
                memory = zero
            memories.append(memory)
            present.append(valid)
        if not memories:
            return (zero.new_zeros((0, temporal_points, hidden)),
                    torch.zeros(0, dtype=torch.bool, device=device))
        return torch.stack(memories, dim=0), torch.as_tensor(
            present, dtype=torch.bool, device=device)

    def _encode_tracks(self, features: dict, track_ids: torch.Tensor,
                       state: Dict[int, object]) -> tuple[torch.Tensor, dict]:
        numeric = torch.cat((
            features["geometry"], features["motion"], features["context"],
            features["lifecycle"], features["objectness"].reshape(-1, 1),
        ), -1)
        source = features.get("pool_id", torch.zeros(
            len(track_ids), dtype=torch.long, device=track_ids.device)).long()
        source = source.clamp(0, 1)
        history_seed = self.clip_history(features["history_clip"])
        base = self.observation(torch.cat((
            self.clip_current(features["clip"]),
            history_seed,
            self.pbd(features["pbd"]),
            self.identity(features["uidm_h"]),
            self.numeric(numeric), self.source_embedding(source),
        ), -1))
        previous, present = self._previous(
            track_ids, state, self.hidden, self.temporal_points,
            track_ids.device)
        previous_last = previous[:, -1] if len(track_ids) else \
            previous.new_zeros((0, self.hidden))
        current = self.temporal_update(base, previous_last)
        rolled = torch.cat((previous[:, 1:], current.unsqueeze(1)), dim=1)
        # A new track still has a non-degenerate temporal map: its cached
        # history view anchors the early positions and the current observation
        # anchors the last position.  This also lets hook coordinates receive
        # gradients before a track has survived K frames.
        alpha = self.temporal_alpha.to(current.dtype)
        initial = torch.lerp(
            history_seed.to(current.dtype).unsqueeze(1).expand(
                -1, self.temporal_points, -1),
            current.unsqueeze(1).expand(-1, self.temporal_points, -1), alpha,
        )
        support = torch.where(present[:, None, None], rolled, initial)
        return support, {
            "source": source, "numeric": numeric, "base": base,
            "current": current, "had_memory": present,
        }

    def forward_frame(self, features: dict, query: torch.Tensor,
                      family: torch.Tensor, track_ids: torch.Tensor,
                      state: Dict[int, object] | None = None,
                      query_tokens: torch.Tensor | None = None,
                      query_mask: torch.Tensor | None = None,
                      query_context: dict | None = None) -> dict:
        state = {} if state is None else state
        context = query_context if query_context is not None else \
            self.query_context(query_tokens, query, family, query_mask)
        n = int(track_ids.numel())
        if not n:
            empty = context["holistic"].new_zeros(0)
            return {
                "logits": empty, "membership_logits": empty,
                "observation_logits": empty, "presence_logits": empty,
                "state": state,
                "track_embedding": context["holistic"].new_zeros((0, self.hidden)),
                "association_embedding": context["holistic"].new_zeros((0, self.hidden)),
                "query_context": context,
            }

        support, aux = self._encode_tracks(features, track_ids, state)
        temporal = support + self.temporal_position[:, :self.temporal_points]
        current_query = aux["current"].unsqueeze(1)
        temporal_query = self.temporal_pcd(current_query, temporal)

        # Official FlexHook samples a spatial map per expression.  The
        # controlled port samples each candidate's temporal map with the same
        # text-conditioned hook coordinates using a true grid_sample call.
        feature_map = support.transpose(1, 2).unsqueeze(2)  # N,H,1,K
        grid = support.new_zeros((n, 1, self.hook_points, 2))
        grid[..., 0] = context["hook_coords"].reshape(1, 1, -1)
        sampled = F.grid_sample(
            feature_map, grid, mode="bilinear", padding_mode="zeros",
            align_corners=True,
        ).squeeze(2).transpose(1, 2)  # N,P,H
        sampled = self.hook_projection(self.hook_norm(sampled))

        text = context["tokens"].unsqueeze(0).expand(n, -1, -1)
        hooks = sampled
        track_map = temporal
        memory = torch.cat((text, hooks, track_map), dim=1)
        memory_pos = torch.cat((
            self.text_position[:, :text.shape[1]].expand(n, -1, -1),
            torch.zeros_like(hooks), self.temporal_position.expand(n, -1, -1),
        ), dim=1)
        key_padding = torch.cat((
            ~context["mask"].reshape(1, -1).expand(n, -1),
            torch.zeros((n, self.hook_points + self.temporal_points),
                        dtype=torch.bool, device=memory.device),
        ), dim=1)
        correspondence = self.output_pcd(
            temporal_query, memory, memory_pos=memory_pos,
            key_padding_mask=key_padding).squeeze(1)
        correspondence = self.correspondence_norm(correspondence)
        holistic = context["holistic"].unsqueeze(0).expand(n, -1)
        pair = torch.cat((
            correspondence, aux["current"], holistic,
            correspondence * holistic, (correspondence - holistic).abs(),
            aux["source"].float().reshape(-1, 1),
        ), dim=-1)
        membership = self.score_head(pair).squeeze(-1)
        observation = self.observation_score_head(torch.cat((
            correspondence, aux["current"], aux["numeric"],
            aux["source"].float().reshape(-1, 1),
        ), dim=-1)).squeeze(-1)
        final = membership + self.observation_scale * observation
        presence = self.presence_head(torch.cat(
            (aux["base"], aux["numeric"]), dim=-1)).squeeze(-1)
        association = F.normalize(self.association_projection(torch.cat(
            (aux["current"], correspondence), dim=-1)), dim=-1)

        new_state = dict(state)
        for index, raw_id in enumerate(track_ids.detach().cpu().tolist()):
            memory = support[index]
            if self.detach_state:
                memory = memory.detach()
            new_state[int(raw_id)] = {"memory": memory}
        return {
            "logits": final, "membership_logits": membership,
            "observation_logits": observation, "presence_logits": presence,
            "track_logits": membership, "state": new_state,
            "track_embedding": F.normalize(correspondence, dim=-1),
            "association_embedding": association,
            "query_context": context, "track_features": correspondence,
            "aux": aux, "hook_coords": context["hook_coords"],
            "sampled_hook_features": sampled,
        }

    def forward(self, features: dict, query: torch.Tensor,
                family: torch.Tensor, track_ids: torch.Tensor,
                state: Dict[int, object] | None = None,
                query_tokens: torch.Tensor | None = None,
                query_mask: torch.Tensor | None = None,
                query_context: dict | None = None) -> dict:
        return self.forward_frame(
            features, query, family, track_ids, state,
            query_tokens=query_tokens, query_mask=query_mask,
            query_context=query_context,
        )
