"""Causal query-to-track set selector used by LocateMOT Stage L16."""
from __future__ import annotations

import re
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


FAMILY_NAMES = (
    "entity", "appearance", "absolute_space", "relation",
    "motion", "temporal", "compositional", "multi_object",
)


def expression_family_vector(text: str) -> torch.Tensor:
    """Deterministic content groups; no dataset identity is inspected."""
    value = " " + re.sub(r"[-_/]+", " ", text.lower()) + " "
    words = set(re.findall(r"[a-z]+", value))
    appearance = {
        "black", "white", "red", "blue", "green", "yellow", "silver",
        "gray", "grey", "brown", "pink", "orange", "purple", "dark",
        "bright", "shirt", "tshirt", "jacket", "dress", "hat", "wearing",
        "small", "large", "tall", "short",
    }
    absolute = {"left", "right", "front", "back", "middle", "center",
                "near", "far", "top", "bottom", "closest", "farthest"}
    relation = {"behind", "beside", "between", "among", "with", "following",
                "ahead", "adjacent", "next"}
    motion = {"moving", "running", "walking", "dancing", "parking", "parked",
              "standing", "driving", "turning", "riding", "stopping", "going"}
    temporal = {"currently", "then", "entering", "leaving", "appearing",
                "disappearing", "before", "after", "start", "end"}
    multi = {"people", "persons", "pedestrians", "automobiles", "cars",
             "vehicles", "dancers", "objects", "men", "women"}
    flags = [
        1.0,
        float(bool(words & appearance)),
        float(bool(words & absolute)),
        float(bool(words & relation)),
        float(bool(words & motion)),
        float(bool(words & temporal)),
        float(bool(words & {"and", "or", "while"}) or len(words) >= 8),
        float(bool(words & multi)),
    ]
    return torch.tensor(flags, dtype=torch.float32)


class Projection(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        )

    def forward(self, value):
        return self.net(torch.nan_to_num(value.float()))


class L16TrackSelector(nn.Module):
    """Per-track temporal GRU + cross-track interaction + semantic belief."""

    def __init__(self, hidden: int = 256, heads: int = 4,
                 dropout: float = 0.10):
        super().__init__()
        self.hidden = int(hidden)
        self.clip_current = Projection(512, hidden, dropout)
        self.clip_history = Projection(512, hidden, dropout)
        self.pbd = Projection(2048, hidden, dropout)
        self.identity = Projection(384, hidden, dropout)
        self.numeric = Projection(32, hidden, dropout)
        self.query = Projection(512, hidden, dropout)
        self.family = Projection(len(FAMILY_NAMES), hidden, dropout)
        self.observation = nn.Sequential(
            nn.Linear(5 * hidden, 2 * hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.temporal = nn.GRUCell(hidden, hidden)
        self.cross_track = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(hidden)
        self.query_fusion = nn.Sequential(
            nn.Linear(4 * hidden, 2 * hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.belief = nn.GRUCell(hidden, hidden)
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.cosine_scale = nn.Parameter(torch.tensor(3.0))
        self.history_cosine_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.tensor(-1.5))

    @staticmethod
    def _previous(track_ids, state: Dict[int, dict], key: str,
                  hidden: int, device) -> torch.Tensor:
        values = []
        zero = torch.zeros(hidden, device=device)
        for track_id in track_ids.detach().cpu().tolist():
            values.append(state.get(int(track_id), {}).get(key, zero))
        return torch.stack(values, dim=0) if values else zero.new_zeros((0, hidden))

    def forward_frame(self, features: dict, query: torch.Tensor,
                      family: torch.Tensor, track_ids: torch.Tensor,
                      state: Dict[int, dict] | None = None,
                      use_belief: bool = True,
                      use_cross_track: bool = True,
                      use_motion: bool = True) -> dict:
        """Process one frame. State contains only current/past track values."""
        state = {} if state is None else state
        n = int(track_ids.numel())
        if not n:
            return {"logits": query.new_zeros(0), "state": state,
                    "track_embedding": query.new_zeros((0, self.hidden)),
                    "query_embedding": self.query(query.reshape(1, -1))[0]}
        device = track_ids.device
        numeric_parts = [features["geometry"], features["motion"],
                         features["context"], features["lifecycle"],
                         features["objectness"].reshape(-1, 1)]
        if not use_motion:
            numeric_parts[1] = torch.zeros_like(numeric_parts[1])
        numeric = torch.cat(numeric_parts, dim=-1)
        base = self.observation(torch.cat((
            self.clip_current(features["clip"]),
            self.clip_history(features["history_clip"]),
            self.pbd(features["pbd"]),
            self.identity(features["uidm_h"]),
            self.numeric(numeric),
        ), dim=-1))
        previous_temporal = self._previous(
            track_ids, state, "temporal", self.hidden, device)
        temporal = self.temporal(base, previous_temporal)
        if use_cross_track:
            attended, _ = self.cross_track(
                temporal.unsqueeze(0), temporal.unsqueeze(0),
                temporal.unsqueeze(0), need_weights=False)
            track = self.cross_norm(temporal + attended[0])
        else:
            track = self.cross_norm(temporal)
        query_embedding = self.query(query.reshape(1, -1))[0] + \
            self.family(family.reshape(1, -1))[0]
        query_embedding = F.normalize(query_embedding, dim=-1)
        expanded = query_embedding.unsqueeze(0).expand(n, -1)
        fused = self.query_fusion(torch.cat((
            track, expanded, track * expanded, (track - expanded).abs(),
        ), dim=-1))
        previous_belief = self._previous(
            track_ids, state, "belief", self.hidden, device)
        belief = self.belief(fused, previous_belief) if use_belief else fused
        crop = F.normalize(torch.nan_to_num(features["clip"].float()), dim=-1)
        history = F.normalize(
            torch.nan_to_num(features["history_clip"].float()), dim=-1)
        query_raw = F.normalize(query.float().reshape(1, -1), dim=-1)[0]
        logits = self.readout(belief).squeeze(-1) + self.output_bias + \
            self.cosine_scale * (crop * query_raw).sum(dim=-1) + \
            self.history_cosine_scale * (history * query_raw).sum(dim=-1)
        new_state = dict(state)
        for index, track_id in enumerate(track_ids.detach().cpu().tolist()):
            new_state[int(track_id)] = {
                "temporal": temporal[index], "belief": belief[index],
            }
        return {
            "logits": logits, "state": new_state,
            "track_embedding": F.normalize(belief, dim=-1),
            "query_embedding": query_embedding,
        }

    def forward(self, features: dict, query: torch.Tensor,
                family: torch.Tensor, track_ids: torch.Tensor,
                state: Dict[int, dict] | None = None,
                use_belief: bool = True,
                use_cross_track: bool = True,
                use_motion: bool = True) -> dict:
        """DDP-compatible entry point for one causal frame."""
        return self.forward_frame(
            features, query, family, track_ids, state,
            use_belief=use_belief, use_cross_track=use_cross_track,
            use_motion=use_motion)
