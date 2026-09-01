"""Extract PBD hidden-state features from accepted coordinate box blocks."""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .types import GenerationBlockEvent


class PBDObjectExtractor:
    """Builds raw PBD features (last + penultimate hidden layers) for accepted
    coordinate box blocks recorded in generation events.
    """

    def __init__(self, keep_layers: Tuple[int, ...] = (-1, -2)):
        self.keep_layers = keep_layers

    def extract(
        self,
        events: List[GenerationBlockEvent],
        hidden_slices: List[Dict[int, torch.Tensor]],
    ) -> List[Dict[str, object]]:
        out = []
        for ev in events:
            if not ev.accepted or ev.block_type != "coord_box":
                continue
            feats = self._features_for_event(ev, hidden_slices)
            if feats is None:
                continue
            feats["event"] = ev
            out.append(feats)
        return out

    def _features_for_event(
        self,
        ev: GenerationBlockEvent,
        hidden_slices: List[Dict[int, torch.Tensor]],
    ) -> Dict[str, torch.Tensor] | None:
        if ev.decode_mode == "MTP/PBD accepted":
            step_hidden = hidden_slices[ev.generation_step - 1]
            feats = {}
            for li, h in step_hidden.items():
                h6 = h[0, -6:, :]
                feats[li] = {
                    "box_end": h6[5],
                    "coord_mean": h6[1:5].mean(dim=0),
                    "full_mean": h6.mean(dim=0),
                }
            return self._flatten_layer_dict(feats)

        if ev.decode_mode == "AR/NTP":
            token_steps = ev.extra.get("token_steps", [])
            hidden_rels = ev.extra.get("hidden_rel_positions", [])
            if len(token_steps) != len(ev.hidden_state_positions) or len(hidden_rels) != len(token_steps):
                return None
            layer_feats = {}
            for li in (hidden_slices[0].keys() if hidden_slices else []):
                per_layer = []
                for step_idx, rel_idx in zip(token_steps, hidden_rels):
                    step_hidden = hidden_slices[step_idx - 1]
                    if li not in step_hidden:
                        return None
                    per_layer.append(step_hidden[li][0, rel_idx : rel_idx + 1, :])
                stacked = torch.cat(per_layer, dim=0)
                layer_feats[li] = {
                    "box_end": stacked[-1],
                    "coord_mean": stacked[1:5].mean(dim=0),
                    "full_mean": stacked.mean(dim=0),
                }
            return self._flatten_layer_dict(layer_feats)

        return None

    def _flatten_layer_dict(self, layer_feats) -> Dict[str, torch.Tensor]:
        result = {}
        layer_names = sorted(layer_feats.keys())
        for li in layer_names:
            suffix = "" if li == max(layer_names) else "_penultimate"
            for k, v in layer_feats[li].items():
                result[f"{k}{suffix}"] = v
        return result
