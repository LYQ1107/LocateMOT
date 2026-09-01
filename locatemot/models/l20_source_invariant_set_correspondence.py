"""Stage L20 source-invariant, null-aware temporal set correspondence.

L19 proved that text-conditioned temporal correspondence can recover reserve
ranking, but it also exposed a source shortcut and an extreme false-positive
rate.  This module keeps the L19 C-Hook/PCD temporal skeleton while changing
the candidate unit from a raw observation to an observation group and
removing source information from every final score head.

The two source adapters are allowed to normalize the different frozen-bank
feature distributions.  After adapter selection, all observations enter the
same shared semantic/temporal and scoring layers.  ``pool_id`` is never
concatenated to a final head; it is returned only as provenance for losses and
diagnostics.  This is an RMOT-only model and does not touch MOT/OVMOT paths.
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F

from locatemot.models.l16_track_selector import FAMILY_NAMES
from locatemot.models.l19_flexhook_correspondence import (
    PCDTransformerBlock,
    Projection,
)


FEATURE_DIMS = (512, 512, 2048, 384, 32)


class SourceAdapter(nn.Module):
    """Adapter for one frozen-bank source, followed by shared-space output."""

    def __init__(self, hidden: int, dropout: float):
        super().__init__()
        self.views = nn.ModuleList([
            Projection(dim, hidden, dropout) for dim in FEATURE_DIMS
        ])
        self.missing_projection = nn.Sequential(
            nn.Linear(len(FEATURE_DIMS), hidden // 2), nn.GELU(),
            nn.LayerNorm(hidden // 2),
        )
        self.fusion = nn.Sequential(
            nn.Linear(5 * hidden + hidden // 2, 2 * hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden, hidden),
            nn.LayerNorm(hidden),
        )

    @staticmethod
    def _missing(value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        finite = torch.isfinite(value).all(dim=-1)
        present = value.abs().sum(dim=-1) > 1e-8
        return (~(finite & present)).float()

    def forward(self, views: tuple[torch.Tensor, ...]) -> torch.Tensor:
        encoded = []
        missing = []
        for projection, value in zip(self.views, views):
            encoded.append(projection(value))
            missing.append(self._missing(value))
        missing_h = self.missing_projection(torch.stack(missing, dim=-1))
        return self.fusion(torch.cat((*encoded, missing_h), dim=-1))


class L20SourceInvariantSetCorrespondence(nn.Module):
    """Causal C-Hook/PCD correspondence over grouped dual-pool tracks."""

    def __init__(self, hidden: int = 256, heads: int = 4,
                 dropout: float = 0.10, token_dim: int = 512,
                 temporal_points: int = 8, hook_points: int = 10,
                 use_source_adapters: bool = True,
                 use_grouping: bool = True, use_null: bool = True):
        super().__init__()
        self.hidden = int(hidden)
        self.heads = int(heads)
        self.token_dim = int(token_dim)
        self.temporal_points = int(temporal_points)
        self.hook_points = int(hook_points)
        self.use_source_adapters = bool(use_source_adapters)
        self.use_grouping = bool(use_grouping)
        self.use_null = bool(use_null)
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        if temporal_points < 2 or hook_points < 1:
            raise ValueError("temporal_points >=2 and hook_points positive required")

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

        self.conditional_pos_embed = nn.Parameter(
            torch.randn(1, hook_points, hidden) * 0.02)
        self.conditional_pos_decoder = PCDTransformerBlock(
            hidden, heads, dropout)
        self.conditional_pos_projector = nn.Sequential(
            nn.LayerNorm(hidden), nn.Linear(hidden, 1),
        )
        self.hook_norm = nn.LayerNorm(hidden)
        self.hook_projection = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
        )

        self.main_adapter = SourceAdapter(hidden, dropout)
        self.reserve_adapter = SourceAdapter(hidden, dropout)
        self.shared_adapter = SourceAdapter(hidden, dropout)
        self.shared_space_norm = nn.LayerNorm(hidden)

        self.temporal_update = nn.GRUCell(hidden, hidden)
        self.temporal_position = nn.Parameter(
            torch.zeros(1, temporal_points, hidden))
        nn.init.normal_(self.temporal_position, std=0.02)
        self.register_buffer(
            "temporal_alpha",
            torch.linspace(0.0, 1.0, temporal_points).reshape(1, -1, 1),
        )
        self.temporal_pcd = PCDTransformerBlock(hidden, heads, dropout)
        self.output_pcd = PCDTransformerBlock(hidden, heads, dropout)
        self.correspondence_norm = nn.LayerNorm(hidden)

        # Group aggregation is source-blind.  Quality is query conditioned by
        # the row correspondence representation and is used only to select a
        # representative box after inference.
        self.view_quality_head = nn.Sequential(
            nn.Linear(2 * hidden + 32, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
        self.group_pool_norm = nn.LayerNorm(hidden)
        self.group_score_head = nn.Sequential(
            nn.Linear(5 * hidden, 2 * hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.group_observation_head = nn.Sequential(
            nn.Linear(2 * hidden + 32, 2 * hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.group_presence_head = nn.Sequential(
            nn.Linear(hidden + 32, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
        self.observation_scale = nn.Parameter(torch.tensor(0.35))
        self.null_head = nn.Sequential(
            nn.Linear(2 * hidden + 32, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, 1),
        )
        self.association_projection = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
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
            raise ValueError("cached token sequence exceeds position table")
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
                    memory = torch.cat((zero[:temporal_points - memory.shape[0]],
                                        memory), dim=0)
            else:
                memory = zero
            memories.append(memory)
            present.append(valid)
        if not memories:
            return (zero.new_zeros((0, temporal_points, hidden)),
                    torch.zeros(0, dtype=torch.bool, device=device))
        return torch.stack(memories, dim=0), torch.as_tensor(
            present, dtype=torch.bool, device=device)

    @staticmethod
    def _numeric(features: dict) -> torch.Tensor:
        return torch.cat((
            features["geometry"], features["motion"], features["context"],
            features["lifecycle"], features["objectness"].reshape(-1, 1),
        ), -1)

    def _adapt(self, features: dict, source: torch.Tensor) -> torch.Tensor:
        views = (
            features["clip"], features["history_clip"], features["pbd"],
            features["uidm_h"], self._numeric(features),
        )
        if not self.use_source_adapters:
            result = self.shared_adapter(views)
            return self.shared_space_norm(result)
        # Adapter projections explicitly sanitize/cast their frozen inputs;
        # keep the source-wise scatter in FP32 so CUDA autocast cannot make
        # an index_put Half/Float mismatch.
        result = torch.zeros((len(source), self.hidden), device=source.device,
                             dtype=torch.float32)
        for source_id, adapter in ((0, self.main_adapter),
                                   (1, self.reserve_adapter)):
            selected = source == source_id
            if selected.any():
                selected_views = tuple(value[selected] for value in views)
                result[selected] = adapter(selected_views).float()
        return self.shared_space_norm(result)

    def _encode_tracks(self, features: dict, track_ids: torch.Tensor,
                       state: Dict[int, object]) -> tuple[torch.Tensor, dict]:
        source = features.get("pool_id", torch.zeros(
            len(track_ids), dtype=torch.long, device=track_ids.device)).long()
        source = source.clamp(0, 1)
        numeric = self._numeric(features)
        base = self._adapt(features, source)
        previous, present = self._previous(
            track_ids, state, self.hidden, self.temporal_points,
            track_ids.device)
        previous_last = previous[:, -1] if len(track_ids) else \
            previous.new_zeros((0, self.hidden))
        current = self.temporal_update(base, previous_last)
        rolled = torch.cat((previous[:, 1:], current.unsqueeze(1)), dim=1)
        alpha = self.temporal_alpha.to(current.dtype)
        initial = current.unsqueeze(1).expand(-1, self.temporal_points, -1)
        support = torch.where(present[:, None, None], rolled, initial)
        return support, {
            "source": source, "numeric": numeric, "base": base,
            "current": current, "had_memory": present,
        }

    def _group_reduce(self, correspondence: torch.Tensor, current: torch.Tensor,
                      numeric: torch.Tensor, group_ids: torch.Tensor) -> dict:
        if not self.use_grouping:
            group_ids = torch.arange(len(group_ids), device=group_ids.device,
                                     dtype=torch.long)
        unique, inverse = torch.unique(group_ids, sorted=True,
                                       return_inverse=True)
        group_values, group_current, group_numeric = [], [], []
        selected_rows, quality_values, sizes = [], [], []
        member_rows, member_counts, raw_group_ids = [], [], []
        for group_index in range(len(unique)):
            indices = torch.nonzero(inverse == group_index,
                                    as_tuple=False).flatten()
            member_rows.append(indices)
            member_counts.append(int(len(indices)))
            raw_group_ids.append(unique[group_index])
            quality_input = torch.cat(
                (correspondence[indices], current[indices], numeric[indices]), -1)
            quality = self.view_quality_head(quality_input).squeeze(-1)
            weights = torch.softmax(quality, dim=0)
            group_values.append((weights[:, None] * correspondence[indices]).sum(0))
            group_current.append((weights[:, None] * current[indices]).sum(0))
            group_numeric.append((weights[:, None] * numeric[indices]).sum(0))
            selected = int(indices[torch.argmax(quality.detach())])
            selected_rows.append(selected)
            quality_values.append(quality)
            sizes.append(int(len(indices)))
        refined_ids = torch.arange(len(group_values), dtype=torch.long,
                                   device=group_ids.device)
        return {
            "group_ids": refined_ids,
            "raw_group_ids": torch.stack(raw_group_ids) if raw_group_ids else
                refined_ids.clone(),
            "inverse": inverse,
            "features": torch.stack(group_values, dim=0),
            "current": torch.stack(group_current, dim=0),
            "numeric": torch.stack(group_numeric, dim=0),
            "selected_rows": torch.as_tensor(selected_rows, dtype=torch.long,
                                               device=group_ids.device),
            "quality": quality_values,
            "sizes": torch.as_tensor(sizes, dtype=torch.long,
                                      device=group_ids.device),
            "member_rows": member_rows,
            "member_counts": torch.as_tensor(member_counts, dtype=torch.long,
                                               device=group_ids.device),
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
                "null_logit": context["holistic"].new_zeros(()),
                "state": state, "group_ids": torch.zeros(
                    0, dtype=torch.long, device=track_ids.device),
                "group_track_ids": torch.zeros(
                    0, dtype=torch.long, device=track_ids.device),
                "group_row_indices": torch.zeros(
                    0, dtype=torch.long, device=track_ids.device),
                "group_member_counts": torch.zeros(
                    0, dtype=torch.long, device=track_ids.device),
                "group_member_rows": [],
                "grouping_enabled": self.use_grouping,
                "track_features": context["holistic"].new_zeros((0, self.hidden)),
                "row_features": context["holistic"].new_zeros((0, self.hidden)),
                "row_quality_logits": empty, "query_context": context,
            }

        support, aux = self._encode_tracks(features, track_ids, state)
        temporal = support + self.temporal_position[:, :self.temporal_points]
        temporal_query = self.temporal_pcd(
            aux["current"].unsqueeze(1), temporal).squeeze(1)
        feature_map = support.transpose(1, 2).unsqueeze(2)
        grid = support.new_zeros((n, 1, self.hook_points, 2))
        grid[..., 0] = context["hook_coords"].reshape(1, 1, -1)
        sampled = F.grid_sample(
            feature_map, grid, mode="bilinear", padding_mode="zeros",
            align_corners=True,
        ).squeeze(2).transpose(1, 2)
        sampled = self.hook_projection(self.hook_norm(sampled))
        text = context["tokens"].unsqueeze(0).expand(n, -1, -1)
        memory = torch.cat((text, sampled, temporal), dim=1)
        memory_pos = torch.cat((
            self.text_position[:, :text.shape[1]].expand(n, -1, -1),
            torch.zeros_like(sampled),
            self.temporal_position.expand(n, -1, -1),
        ), dim=1)
        key_padding = torch.cat((
            ~context["mask"].reshape(1, -1).expand(n, -1),
            torch.zeros((n, self.hook_points + self.temporal_points),
                        dtype=torch.bool, device=memory.device),
        ), dim=1)
        row_corr = self.output_pcd(
            temporal_query.unsqueeze(1), memory, memory_pos=memory_pos,
            key_padding_mask=key_padding).squeeze(1)
        row_corr = self.correspondence_norm(row_corr)
        # ``l20_group_id`` is the GT-free strict one-to-one frame grouping.
        # Keep the historical broad ID only as an audit/provenance feature;
        # it must never drive the output aggregation.
        group_ids = features.get("l20_group_id",
                                 features.get("observation_group_id"))
        if group_ids is None:
            group_ids = torch.arange(n, dtype=torch.long, device=track_ids.device)
        group = self._group_reduce(row_corr, aux["current"], aux["numeric"],
                                   group_ids.long())
        group_corr = self.group_pool_norm(group["features"])
        group_current = group["current"]
        group_numeric = group["numeric"]
        holistic = context["holistic"].unsqueeze(0).expand(len(group_corr), -1)
        pair = torch.cat((
            group_corr, group_current, holistic,
            group_corr * holistic, (group_corr - holistic).abs(),
        ), dim=-1)
        membership = self.group_score_head(pair).squeeze(-1)
        observation = self.group_observation_head(torch.cat(
            (group_corr, group_current, group_numeric), dim=-1)).squeeze(-1)
        final = membership + self.observation_scale * observation
        presence = self.group_presence_head(torch.cat(
            (group_corr, group_numeric), dim=-1)).squeeze(-1)
        frame_summary = group_corr.mean(dim=0) if len(group_corr) else \
            context["holistic"].new_zeros(self.hidden)
        frame_numeric = group_numeric.mean(dim=0) if len(group_numeric) else \
            context["holistic"].new_zeros(32)
        null = self.null_head(torch.cat((context["holistic"], frame_summary,
                                         frame_numeric), dim=-1)).squeeze(-1)
        if not self.use_null:
            null = null.detach() * 0.0 - 100.0

        selected_rows = group["selected_rows"]
        row_source = aux["source"]
        group_source = []
        group_track_ids = []
        for group_index in range(len(group["group_ids"])):
            # ``inverse`` indexes raw unique IDs.  A raw group may have been
            # split into multiple refined singleton groups, so it is not a
            # valid lookup for the refined group index.
            indices = group["member_rows"][group_index]
            sources = set(int(value) for value in row_source[indices].detach().cpu())
            group_source.append(next(iter(sources)) if len(sources) == 1 else 2)
            group_track_ids.append(track_ids[selected_rows[group_index]])
        group_source = torch.as_tensor(group_source, dtype=torch.int8,
                                       device=track_ids.device)
        group_track_ids = torch.stack(group_track_ids) if group_track_ids else \
            track_ids.new_zeros(0)

        new_state = dict(state)
        for index, raw_id in enumerate(track_ids.detach().cpu().tolist()):
            memory_value = support[index]
            if self.detach_state:
                memory_value = memory_value.detach()
            new_state[int(raw_id)] = {"memory": memory_value}
        row_quality = self.view_quality_head(torch.cat((
            row_corr, aux["current"], aux["numeric"]), dim=-1)).squeeze(-1)
        association = F.normalize(self.association_projection(torch.cat(
            (group_corr, group_current), dim=-1)), dim=-1)
        return {
            # ``logits`` is the group-level final score for evaluator
            # compatibility.  It never contains source scalar/embedding.
            "logits": final, "membership_logits": membership,
            "observation_logits": observation, "presence_logits": presence,
            "null_logit": null, "state": new_state,
            "group_ids": group["group_ids"], "group_source": group_source,
            "raw_group_ids": group["raw_group_ids"],
            "grouping_enabled": self.use_grouping,
            "group_track_ids": group_track_ids,
            "group_row_indices": selected_rows,
            "group_sizes": group["sizes"],
            "group_member_counts": group["member_counts"],
            "group_member_rows": group["member_rows"],
            "track_features": F.normalize(group_corr, dim=-1),
            "row_features": F.normalize(row_corr, dim=-1),
            "row_track_ids": track_ids, "row_source": row_source,
            "row_group_ids": group_ids.long(), "row_quality_logits": row_quality,
            "association_embedding": association,
            "query_context": context, "aux": aux,
            "hook_coords": context["hook_coords"],
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
