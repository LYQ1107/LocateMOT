"""L81 RMOT-only hierarchical early-fusion correspondence head.

The detector/CLIP runtime is frozen outside this module.  This head keeps the
full spatial CLIP taps candidate-conditioned by a geometry marker, performs
candidate-local bidirectional fusion with four text slots, and only then lets
the complete current-frame candidate set compete.  IDs and historical row
identity are consumed by the data assembler, never by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class L81Config:
    visual_dim: int = 768
    text_dim: int = 768
    observation_dim: int = 1432
    hidden: int = 256
    heads: int = 4
    query_slots: int = 4
    history_length: int = 8
    text_layers: int = 2
    fusion_layers: int = 2
    set_layers: int = 2
    dropout: float = 0.05
    visual_taps: int = 3
    patch_tokens_per_tap: int = 196
    local_tokens_per_candidate: int = 63
    marker_feature_dim: int = 14
    pair_geometry_dim: int = 16


class ResidualBidirectionalFusion(nn.Module):
    """One candidate-local visual<->slot residual fusion block."""

    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.slot_from_visual = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True)
        self.visual_from_slot = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True)
        self.slot_norm1 = nn.LayerNorm(hidden)
        self.slot_norm2 = nn.LayerNorm(hidden)
        self.visual_norm1 = nn.LayerNorm(hidden)
        self.visual_norm2 = nn.LayerNorm(hidden)
        self.slot_ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden), nn.Dropout(dropout),
        )
        self.visual_ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden), nn.Dropout(dropout),
        )

    def forward(self, slots: torch.Tensor, visual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        slot_delta, _ = self.slot_from_visual(
            self.slot_norm1(slots), self.visual_norm1(visual), self.visual_norm1(visual),
            need_weights=False)
        slots = slots + slot_delta
        slots = slots + self.slot_ff(self.slot_norm2(slots))
        visual_delta, _ = self.visual_from_slot(
            self.visual_norm1(visual), self.slot_norm1(slots), self.slot_norm1(slots),
            need_weights=False)
        visual = visual + visual_delta
        visual = visual + self.visual_ff(self.visual_norm2(visual))
        return slots, visual


def _pair_geometry(boxes: torch.Tensor) -> torch.Tensor:
    """Return pairwise geometry with no order/ID features.

    The output is equivariant under any candidate-row permutation: swapping
    rows swaps both axes of the returned matrix.  It is used only as a set
    attention bias, never as a candidate deletion or ranking pre-filter.
    """
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError(f"expected normalized boxes [N,4], got {tuple(boxes.shape)}")
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1).clamp_min(1e-5)
    h = (y2 - y1).clamp_min(1e-5)
    centers = torch.stack((cx, cy), dim=-1)
    sizes = torch.stack((w, h), dim=-1)
    delta = centers[:, None, :] - centers[None, :, :]
    abs_delta = delta.abs()
    log_ratio = torch.log(sizes[:, None, :] / sizes[None, :,]).clamp(-8.0, 8.0)
    left = torch.maximum(x1[:, None], x1[None, :])
    top = torch.maximum(y1[:, None], y1[None, :])
    right = torch.minimum(x2[:, None], x2[None, :])
    bottom = torch.minimum(y2[:, None], y2[None, :])
    inter = (right - left).clamp_min(0.0) * (bottom - top).clamp_min(0.0)
    area = (w * h).clamp_min(1e-6)
    union = area[:, None] + area[None, :] - inter
    iou = inter / union.clamp_min(1e-6)
    center_dist = torch.sqrt((delta * delta).sum(-1).clamp_min(1e-10))
    overlap_x = (right - left).clamp_min(0.0) / torch.minimum(w[:, None], w[None, :]).clamp_min(1e-5)
    overlap_y = (bottom - top).clamp_min(0.0) / torch.minimum(h[:, None], h[None, :]).clamp_min(1e-5)
    contain_i = inter / area[:, None].clamp_min(1e-6)
    contain_j = inter / area[None, :].clamp_min(1e-6)
    return torch.cat((
        delta, abs_delta, log_ratio, iou[..., None], center_dist[..., None],
        overlap_x[..., None], overlap_y[..., None], contain_i[..., None], contain_j[..., None],
        centers[:, None, :].expand(-1, boxes.shape[0], -1),
        centers[None, :, :].expand(boxes.shape[0], -1, -1),
    ), dim=-1)


class RelationAwareSetLayer(nn.Module):
    """Permutation-equivariant self-attention with pairwise geometry bias."""

    def __init__(self, hidden: int, heads: int, pair_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.heads = heads
        self.head_dim = hidden // heads
        self.norm1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.pair_bias = nn.Sequential(
            nn.LayerNorm(pair_dim), nn.Linear(pair_dim, heads, bias=False))
        self.out = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * hidden, hidden), nn.Dropout(dropout),
        )

    def forward(self, value: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
        if value.ndim != 2 or value.shape[-1] != self.hidden:
            raise ValueError(f"set value shape mismatch: {tuple(value.shape)}")
        normalized = self.norm1(value)
        qkv = self.qkv(normalized).reshape(value.shape[0], 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=1)
        q = q.permute(1, 0, 2)
        k = k.permute(1, 0, 2)
        v = v.permute(1, 0, 2)
        logits = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        bias = self.pair_bias(_pair_geometry(boxes)).permute(2, 0, 1)
        weights = torch.softmax(logits + bias, dim=-1)
        attended = torch.matmul(weights, v).permute(1, 0, 2).reshape(value.shape[0], self.hidden)
        value = value + self.dropout(self.out(attended))
        value = value + self.ff(self.norm2(value))
        return value


class L81HierarchicalEarlyFusion(nn.Module):
    """Full-frame candidate-marked hierarchical correspondence head."""

    canonical_output_keys = (
        "candidate_logits", "track_logits", "continuation_logits",
        "quality_logits", "null_logit", "cardinality_logit",
    )

    def __init__(self, config: L81Config | None = None) -> None:
        super().__init__()
        self.config = config or L81Config()
        c = self.config
        if c.hidden % c.heads:
            raise ValueError("hidden must be divisible by heads")
        if c.visual_taps != 3 or c.patch_tokens_per_tap != 196:
            raise ValueError("L81 visual tap contract is fixed to three 14x14 taps")
        self.visual_adapters = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(c.visual_dim), nn.Linear(c.visual_dim, c.hidden))
            for _ in range(c.visual_taps)
        ])
        self.local_adapter = nn.Sequential(
            nn.LayerNorm(c.visual_dim), nn.Linear(c.visual_dim, c.hidden))
        self.text_projection = nn.Sequential(
            nn.LayerNorm(c.text_dim), nn.Linear(c.text_dim, c.hidden))
        text_layer = nn.TransformerEncoderLayer(
            d_model=c.hidden, nhead=c.heads, dim_feedforward=4 * c.hidden,
            dropout=c.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.text_encoder = nn.TransformerEncoder(text_layer, num_layers=c.text_layers)
        self.text_final_norm = nn.LayerNorm(c.hidden)
        self.query_slots = nn.Parameter(torch.empty(c.query_slots, c.hidden))
        nn.init.normal_(self.query_slots, mean=0.0, std=0.02)
        self.text_to_slots = nn.MultiheadAttention(
            c.hidden, c.heads, dropout=c.dropout, batch_first=True)
        self.slot_norm = nn.LayerNorm(c.hidden)
        self.marker_projection = nn.Sequential(
            nn.LayerNorm(c.marker_feature_dim), nn.Linear(c.marker_feature_dim, c.hidden),
            nn.GELU(), nn.Linear(c.hidden, c.hidden))
        self.region_marker = nn.Parameter(torch.zeros(c.hidden))
        self.tap_embedding = nn.Parameter(torch.zeros(c.visual_taps, c.hidden))
        nn.init.normal_(self.tap_embedding, mean=0.0, std=0.01)
        self.register_buffer("patch_centers", self._make_patch_centers(), persistent=False)
        self.fusion_blocks = nn.ModuleList([
            ResidualBidirectionalFusion(c.hidden, c.heads, c.dropout)
            for _ in range(c.fusion_layers)
        ])
        self.set_blocks = nn.ModuleList([
            RelationAwareSetLayer(c.hidden, c.heads, c.pair_geometry_dim, c.dropout)
            for _ in range(c.set_layers)
        ])
        self.slot_gate = nn.Linear(c.hidden, 1)
        self.observation_projection = nn.Sequential(
            nn.LayerNorm(c.observation_dim), nn.Linear(c.observation_dim, c.hidden))
        self.time_projection = nn.Linear(1, c.hidden)
        self.history_gru = nn.GRU(c.hidden, c.hidden, num_layers=1, batch_first=True)
        self.history_fusion = nn.Sequential(
            nn.LayerNorm(2 * c.hidden), nn.Linear(2 * c.hidden, c.hidden), nn.GELU())
        self.membership_head = nn.Sequential(
            nn.LayerNorm(2 * c.hidden), nn.Linear(2 * c.hidden, c.hidden), nn.GELU(),
            nn.Linear(c.hidden, 1))
        self.track_head = nn.Sequential(
            nn.LayerNorm(2 * c.hidden), nn.Linear(2 * c.hidden, c.hidden // 2), nn.GELU(),
            nn.Linear(c.hidden // 2, 1))
        self.continuation_head = nn.Sequential(
            nn.LayerNorm(c.hidden), nn.Linear(c.hidden, c.hidden // 2), nn.GELU(),
            nn.Linear(c.hidden // 2, 1))
        self.quality_head = nn.Sequential(
            nn.LayerNorm(c.hidden), nn.Linear(c.hidden, c.hidden // 2), nn.GELU(),
            nn.Linear(c.hidden // 2, 1))
        self.null_head = nn.Sequential(
            nn.LayerNorm(2 * c.hidden), nn.Linear(2 * c.hidden, c.hidden // 2), nn.GELU(),
            nn.Linear(c.hidden // 2, 1))
        self.cardinality_head = nn.Sequential(
            nn.LayerNorm(2 * c.hidden), nn.Linear(2 * c.hidden, c.hidden // 2), nn.GELU(),
            nn.Linear(c.hidden // 2, 1))
        self._marker_enabled = True

    @staticmethod
    def _make_patch_centers() -> torch.Tensor:
        fractions = (torch.arange(14, dtype=torch.float32) + 0.5) / 14.0
        gy, gx = torch.meshgrid(fractions, fractions, indexing="ij")
        return torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)

    def set_marker_enabled(self, enabled: bool) -> None:
        self._marker_enabled = bool(enabled)

    @property
    def marker_enabled(self) -> bool:
        return bool(self._marker_enabled)

    @staticmethod
    def masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.bool().to(dtype=value.dtype).unsqueeze(-1)
        return (value * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    def _relative_marker_features(self, boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        centers = self.patch_centers.to(device=boxes.device, dtype=boxes.dtype)
        x1, y1, x2, y2 = boxes.unbind(-1)
        width = (x2 - x1).clamp_min(1e-5)
        height = (y2 - y1).clamp_min(1e-5)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        dx = (centers[None, :, 0] - cx[:, None]) / width[:, None]
        dy = (centers[None, :, 1] - cy[:, None]) / height[:, None]
        left = (centers[None, :, 0] - x1[:, None]) / width[:, None]
        right = (x2[:, None] - centers[None, :, 0]) / width[:, None]
        top = (centers[None, :, 1] - y1[:, None]) / height[:, None]
        bottom = (y2[:, None] - centers[None, :, 1]) / height[:, None]
        inside = ((left >= 0) & (right >= 0) & (top >= 0) & (bottom >= 0)).to(boxes.dtype)
        outside = 1.0 - inside
        area = (width * height).clamp_min(1e-6)
        distance = torch.sqrt(dx.square() + dy.square() + 1e-8)
        features = torch.stack((
            dx, dy, dx.abs(), dy.abs(), left, right, top, bottom,
            inside, outside, width[:, None].expand_as(dx), height[:, None].expand_as(dy),
            area[:, None].expand_as(dx), distance,
        ), dim=-1)
        return features, inside.unsqueeze(-1)

    def _full_visual(self, visual_pyramid: torch.Tensor, local_tokens: torch.Tensor,
                     boxes_norm: torch.Tensor) -> torch.Tensor:
        c = self.config
        if visual_pyramid.ndim == 4:
            if tuple(visual_pyramid.shape) != (c.visual_taps, 1, c.patch_tokens_per_tap, c.visual_dim):
                raise ValueError(f"visual pyramid shape mismatch: {tuple(visual_pyramid.shape)}")
            pyramid = visual_pyramid[:, 0]
        elif visual_pyramid.ndim == 3:
            if tuple(visual_pyramid.shape) != (c.visual_taps, c.patch_tokens_per_tap, c.visual_dim):
                raise ValueError(f"visual pyramid shape mismatch: {tuple(visual_pyramid.shape)}")
            pyramid = visual_pyramid
        else:
            raise ValueError(f"visual pyramid rank mismatch: {tuple(visual_pyramid.shape)}")
        if local_tokens.ndim != 3 or tuple(local_tokens.shape[1:]) != (c.local_tokens_per_candidate, c.visual_dim):
            raise ValueError(f"local token shape mismatch: {tuple(local_tokens.shape)}")
        count = int(local_tokens.shape[0])
        if boxes_norm.shape != (count, 4) or not bool(torch.isfinite(boxes_norm).all()):
            raise ValueError("candidate box shape/finite mismatch")
        marker_features, inside = self._relative_marker_features(boxes_norm.float().clamp(0.0, 1.0))
        pieces = []
        for tap, adapter in enumerate(self.visual_adapters):
            base = adapter(pyramid[tap].float())
            candidate_base = base.unsqueeze(0).expand(count, -1, -1)
            if self._marker_enabled:
                marker = self.marker_projection(marker_features)
                marker = marker + inside * self.region_marker.view(1, 1, -1)
                candidate_base = candidate_base + marker
            candidate_base = candidate_base + self.tap_embedding[tap].view(1, 1, -1)
            pieces.append(candidate_base)
        local = self.local_adapter(local_tokens.float())
        return torch.cat((*pieces, local), dim=1)

    def _encode_text(self, text_tokens: torch.Tensor, text_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if text_tokens.ndim == 3:
            if text_tokens.shape[0] != 1:
                raise ValueError("one expression sequence is expected per unit")
            text_tokens = text_tokens[0]
        if text_tokens.ndim != 2 or text_tokens.shape[-1] != self.config.text_dim:
            raise ValueError(f"text shape mismatch: {tuple(text_tokens.shape)}")
        mask = text_mask.bool().reshape(-1)
        if mask.numel() != text_tokens.shape[0] or not bool(mask.any()):
            raise ValueError("text mask mismatch/empty")
        value = self.text_projection(text_tokens.float()).unsqueeze(0)
        encoded = self.text_encoder(value, src_key_padding_mask=~mask.unsqueeze(0))
        encoded = self.text_final_norm(encoded)
        query_summary = self.masked_mean(encoded, mask.unsqueeze(0))[0]
        return encoded, mask, query_summary

    def _fuse_candidates(self, visual: torch.Tensor, text: torch.Tensor,
                         text_mask: torch.Tensor, chunk_size: int | None) -> torch.Tensor:
        count = int(visual.shape[0])
        slots = self.query_slots.unsqueeze(0).expand(count, -1, -1)
        key_padding = ~text_mask.unsqueeze(0)
        result = []
        size = count if chunk_size is None else max(1, int(chunk_size))
        for begin in range(0, count, size):
            end = min(count, begin + size)
            current_slots = slots[begin:end]
            current_text = text.expand(end - begin, -1, -1)
            current_mask = key_padding.expand(end - begin, -1)
            current_slots, _ = self.text_to_slots(
                self.slot_norm(current_slots), current_text, current_text,
                key_padding_mask=current_mask, need_weights=False)
            current_slots = self.slot_norm(current_slots)
            current_visual = visual[begin:end]
            for block in self.fusion_blocks:
                current_slots, current_visual = block(current_slots, current_visual)
            result.append(current_slots)
        return torch.cat(result, dim=0)

    def _history(self, observations: torch.Tensor, history_mask: torch.Tensor,
                 history_frame_ids: torch.Tensor, current_frame: int) -> torch.Tensor:
        c = self.config
        if observations.ndim != 3 or observations.shape[-1] != c.observation_dim:
            raise ValueError(f"observation shape mismatch: {tuple(observations.shape)}")
        if history_mask.shape != observations.shape[:2] or history_frame_ids.shape != history_mask.shape:
            raise ValueError("history shape mismatch")
        mask = history_mask.bool()
        if bool((history_frame_ids[mask] > int(current_frame)).any()):
            raise ValueError("future history passed to L81")
        value = self.observation_projection(observations.float())
        denom = max(1.0, float(current_frame) + 1.0)
        time_value = history_frame_ids.float().clamp_min(0.0).unsqueeze(-1) / denom
        value = value + self.time_projection(time_value)
        value = value.masked_fill(~mask.unsqueeze(-1), 0.0)
        encoded, _ = self.history_gru(value)
        encoded = encoded.masked_fill(~mask.unsqueeze(-1), 0.0)
        lengths = mask.long().sum(dim=1).clamp_min(1) - 1
        row = torch.arange(encoded.shape[0], device=encoded.device)
        return encoded[row, lengths]

    def _set_competition(self, evidence: torch.Tensor, boxes_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        values = []
        for slot in range(self.config.query_slots):
            value = evidence[:, slot, :]
            for layer in self.set_blocks:
                value = layer(value, boxes_norm)
            values.append(value)
        slot_values = torch.stack(values, dim=1)
        gates = torch.softmax(self.slot_gate(slot_values).squeeze(-1), dim=1)
        candidate = (slot_values * gates.unsqueeze(-1)).sum(dim=1)
        return candidate, slot_values

    def forward(self, visual_pyramid: torch.Tensor, local_tokens: torch.Tensor,
                text_tokens: torch.Tensor, text_mask: torch.Tensor,
                observations: torch.Tensor, history_mask: torch.Tensor,
                history_frame_ids: torch.Tensor, current_frame: int,
                boxes_norm: torch.Tensor, candidate_chunk_size: int | None = None,
                return_audit: bool = False) -> dict[str, torch.Tensor]:
        count = int(local_tokens.shape[0])
        if count <= 0:
            raise ValueError("L81 requires a complete nonempty candidate set")
        visual = self._full_visual(visual_pyramid, local_tokens, boxes_norm)
        text, mask, query_summary = self._encode_text(text_tokens, text_mask)
        evidence = self._fuse_candidates(visual, text, mask, candidate_chunk_size)
        candidate_set, slot_values = self._set_competition(evidence, boxes_norm.float())
        history = self._history(observations, history_mask, history_frame_ids, current_frame)
        fused = self.history_fusion(torch.cat((candidate_set, history), dim=-1))
        pair = torch.cat((fused, candidate_set), dim=-1)
        set_summary = fused.mean(dim=0)
        frame_pair = torch.cat((set_summary, query_summary), dim=-1)
        output: dict[str, torch.Tensor] = {
            "candidate_logits": self.membership_head(pair).squeeze(-1),
            "track_logits": self.track_head(pair).squeeze(-1),
            "continuation_logits": self.continuation_head(fused).squeeze(-1),
            "quality_logits": self.quality_head(fused).squeeze(-1),
            "null_logit": self.null_head(frame_pair).squeeze(),
            "cardinality_logit": self.cardinality_head(frame_pair).squeeze(),
        }
        if return_audit:
            output.update({
                "query_tokens": text[0], "query_vector": query_summary,
                "candidate_evidence": candidate_set, "slot_evidence": slot_values,
                "candidate_visual_tokens": visual, "marker_enabled": torch.tensor(
                    float(self._marker_enabled), device=visual.device),
            })
        if set(output).intersection(self.canonical_output_keys) != set(self.canonical_output_keys):
            raise AssertionError("L81 canonical output contract drift")
        if any(not bool(torch.isfinite(value).all()) for key, value in output.items()
               if key in self.canonical_output_keys):
            raise FloatingPointError("nonfinite L81 canonical output")
        if output["candidate_logits"].shape != (count,):
            raise AssertionError("L81 candidate output count drift")
        return output

    def parameter_report(self) -> dict[str, Any]:
        trainable = [
            (name, int(parameter.numel()), str(parameter.dtype))
            for name, parameter in self.named_parameters() if parameter.requires_grad
        ]
        return {
            "config": self.config.__dict__,
            "trainable_parameter_count": int(sum(item[1] for item in trainable)),
            "total_parameter_count": int(sum(parameter.numel() for parameter in self.parameters())),
            "trainable_parameters": [item[0] for item in trainable],
            "trainable_parameter_dtypes": {item[0]: item[2] for item in trainable},
            "canonical_output_keys": list(self.canonical_output_keys),
            "forbidden_semantic_inputs": [
                "source_id", "pool_id", "group_id", "query_id", "track_id", "state_key",
                "candidate_id", "candidate_index", "old_scores", "gt_identity",
            ],
            "history_ids_used_only_for_causal_row_assembly": True,
            "token_span_region_alignment": "UNALIGNED",
            "static_motion_alignment": "UNALIGNED",
        }


__all__ = ["L81Config", "L81HierarchicalEarlyFusion", "RelationAwareSetLayer"]
