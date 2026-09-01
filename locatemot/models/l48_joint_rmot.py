"""Stage L48 semantic matcher for the RMOT-only branch.

This module intentionally contains no tracker, source/pool classifier, NULL
head, or sequence decoder.  It consumes a complete current-frame candidate set
and frozen observation streams, then returns expression-conditioned semantic
logits.  Identity and NULL are reserved for later stages only after the B1
semantic gate.
"""
from __future__ import annotations

import torch
from torch import nn


class L48SemanticMatcher(nn.Module):
    """Word-token to candidate-set semantic matcher.

    Inputs are unbatched tensors for one ``(dataset, video, query, frame)``
    unit.  The candidate dimension is retained through the set block; no
    candidate is removed or selected by this module.
    """

    def __init__(self, hidden: int = 256, heads: int = 4,
                 dropout: float = 0.1, relation_dim: int = 4):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = int(hidden)
        self.relation_dim = int(relation_dim)
        self.appearance = nn.Sequential(
            nn.LayerNorm(512), nn.Linear(512, hidden), nn.GELU())
        self.geometry = nn.Sequential(
            nn.LayerNorm(7 + relation_dim), nn.Linear(7 + relation_dim, hidden),
            nn.GELU())
        self.motion_identity = nn.Sequential(
            nn.LayerNorm(512 + 8 + 8 + 8 + 1), nn.Linear(512 + 8 + 8 + 8 + 1, hidden),
            nn.GELU())
        self.text = nn.Sequential(
            nn.LayerNorm(768), nn.Linear(768, hidden), nn.GELU())
        self.query_to_candidate = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden)
        set_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=4 * hidden,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.candidate_set = nn.TransformerEncoder(set_layer, num_layers=1)
        self.fusion = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.semantic_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1))

    @staticmethod
    def _unbatch(value: torch.Tensor, expected_last: int | None = None):
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if expected_last is not None and value.shape[-1] != expected_last:
            raise ValueError(f"expected last dim {expected_last}, got {tuple(value.shape)}")
        return value

    def forward(self, clip: torch.Tensor, history_clip: torch.Tensor,
                geometry: torch.Tensor, motion: torch.Tensor,
                context: torch.Tensor, lifecycle: torch.Tensor,
                objectness: torch.Tensor, text_tokens: torch.Tensor,
                text_mask: torch.Tensor, relation: torch.Tensor | None = None):
        clip = self._unbatch(clip.float(), 512)
        history_clip = self._unbatch(history_clip.float(), 512)
        geometry = self._unbatch(geometry.float(), 7)
        motion = self._unbatch(motion.float(), 8)
        context = self._unbatch(context.float(), 8)
        lifecycle = self._unbatch(lifecycle.float(), 8)
        objectness = self._unbatch(objectness.float())
        text_tokens = self._unbatch(text_tokens.float(), 768)
        text_mask = self._unbatch(text_mask.bool())
        if relation is None:
            relation = geometry.new_zeros((clip.shape[0], self.relation_dim))
        relation = self._unbatch(relation.float(), self.relation_dim)
        candidate_count = clip.shape[0]
        if any(value.shape[0] != candidate_count for value in
               (history_clip, geometry, motion, context, lifecycle, objectness, relation)):
            raise ValueError("candidate stream lengths are not aligned")
        if text_mask.numel() != text_tokens.shape[0]:
            raise ValueError("text token/mask lengths are not aligned")
        # The three streams remain explicit until after cross-attention so that
        # ablations can remove one stream without changing the row contract.
        appearance = self.appearance(torch.nan_to_num(clip))
        geo = self.geometry(torch.nan_to_num(torch.cat((geometry, relation), -1)))
        motion_input = torch.cat((history_clip, motion, lifecycle, context,
                                  objectness.unsqueeze(-1)), -1)
        motion_stream = self.motion_identity(torch.nan_to_num(motion_input))
        candidate_base = (appearance + geo + motion_stream) / 3.0
        query = self.text(torch.nan_to_num(text_tokens)).unsqueeze(0)
        candidate_query = candidate_base.unsqueeze(0)
        key_padding = ~text_mask.unsqueeze(0)
        cross, attention = self.query_to_candidate(
            candidate_query, query, query, key_padding_mask=key_padding,
            need_weights=True, average_attn_weights=False)
        cross = self.cross_norm(cross[0])
        fused = self.fusion(torch.cat((appearance, geo, motion_stream, cross), -1))
        # One full-frame set operation.  No mask is used because all rows in
        # the supplied unit are required current-frame candidates.
        set_features = self.candidate_set(fused.unsqueeze(0))[0]
        logits = self.semantic_head(set_features).squeeze(-1)
        with torch.no_grad():
            stream_norms = {
                "appearance": float(appearance.norm(dim=-1).mean().cpu()),
                "geometry_relation": float(geo.norm(dim=-1).mean().cpu()),
                "motion_identity": float(motion_stream.norm(dim=-1).mean().cpu()),
                "cross_attention": float(cross.norm(dim=-1).mean().cpu()),
            }
        return {
            "semantic_logit": logits,
            "candidate_embedding": set_features,
            "cross_attention": attention,
            "stream_norms": stream_norms,
        }

    def config(self) -> dict:
        return {"hidden": self.hidden, "heads": 4,
                "relation_dim": self.relation_dim,
                "attention_layers": {"query_to_candidate": 1, "candidate_set": 1},
                "semantic_only": True,
                "excluded_semantic_inputs": ["source_id", "pool_id", "group_id", "state_key", "query_id_as_feature"],
                "token_span_region_alignment": "UNALIGNED",
                "static_motion_language_mask": "UNALIGNED/not claimed"}
