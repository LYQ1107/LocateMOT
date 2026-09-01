"""Frame-aligned membership/set wrapper for LocateMOT Stage L29.

The frozen L28 decoder remains usable as the initialization/replay base, but
L29 makes the current observation the explicit emission contract.  Track-level
history relevance is retained only as an auxiliary state signal.
"""
from __future__ import annotations

import torch
from torch import nn

from locatemot.models.l28_track_set_decoder import L28TrackSetDecoder


class L29FrameMembershipSetDecoder(nn.Module):
    """Current-frame membership with same-frame set competition."""

    def __init__(self, feature_dim=1432, hidden=128, heads=4, layers=2,
                 dropout=0.1):
        super().__init__()
        self.base = L28TrackSetDecoder(feature_dim, hidden, heads, layers, dropout)
        self.stale_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden // 2), nn.GELU(), nn.Linear(hidden // 2, 1))

    @staticmethod
    def set_compete(logits, valid=None):
        """Return log-probabilities over all current tracks in one frame.

        The operation preserves ordering but makes the set denominator explicit;
        callers must supply the complete current candidate set and may retain
        multiple positives.
        """
        if valid is None:
            valid = torch.ones_like(logits, dtype=torch.bool)
        result = logits.new_full(logits.shape, -20.0)
        if valid.any():
            values = logits[valid]
            result[valid] = values - torch.logsumexp(values, dim=0)
        return result

    def encode_observations(self, observations, observation_mask, observation_time):
        return self.base.encode_observations(observations, observation_mask,
                                             observation_time)

    def forward_encoded(self, encoded, observation_mask, query_tokens, query_mask):
        output = self.base.forward_encoded(encoded, observation_mask, query_tokens,
                                           query_mask)
        obs, mask, track_base = encoded
        qtok = self.base.text_proj(torch.nan_to_num(query_tokens.float()))
        qmask = query_mask.bool().unsqueeze(0) if query_mask.ndim == 1 else query_mask.bool()
        q = self.base._masked_mean(qtok, qmask)
        if q.shape[0] != track_base.shape[0]:
            q = q[:1].expand(track_base.shape[0], -1)
        fused = torch.cat((track_base, q), dim=-1)
        stale = self.stale_head(fused).squeeze(-1)
        latest_index = mask.long().sum(dim=1).clamp_min(1) - 1
        current = output["membership_logits"].gather(1, latest_index[:, None]).squeeze(1)
        output.update({
            "current_membership_logits": current,
            "stale_logits": stale,
            "track_state_logits": output["track_logits"],
            "set_membership_logits": self.set_compete(current),
        })
        return output

    def forward(self, observations, observation_mask, observation_time,
                query_tokens, query_mask):
        encoded = self.encode_observations(observations, observation_mask,
                                            observation_time)
        return self.forward_encoded(encoded, observation_mask, query_tokens,
                                    query_mask)

    def load_l28_checkpoint(self, state_dict):
        """Load frozen L28 weights into the base for contract/replay control."""
        missing, unexpected = self.base.load_state_dict(state_dict, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"unexpected L28 load result: {missing}, {unexpected}")
