"""Stage L5 Route B: sequence-local dynamic identity prediction.

Scientific hypothesis (Route B): persistent association is better modelled as
an in-context identity prediction problem (MOTIP, CVPR 2025) than as pairwise
affinity ranking.  Within a clip/video, every GT identity gets a *sequence-
local slot*; the model predicts for each candidate its slot (or NEW).  The
slot vocabulary is shared by ALL and restricted views, so cross-spec identity
consistency is supervised directly (same GT -> same slot), without any
dataset-global ID vocabulary.

Clean reimplementation guided by audited official designs (no code copied):
- MOTIP IDDecoder / runtime tracker (CVPR 2025, Apache-2.0, commit ffc0e905):
  candidate -> ID class (K+1 with newborn), Hungarian on extended matrix,
  ID embedding + cross-attention between candidates and trajectory states.
- Our L5 Route A temporal encoder (same evidence projection).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from locatemot.models.l5_route_a import (
    CAND_FEATURES,
    PBD_DIM,
    TemporalIdentityEncoder,
)


class L5IdentityPredictor(nn.Module):
    """Candidate -> sequence-local identity slot classifier."""

    def __init__(self, d_model=256, temporal_layers=3, set_layers=3,
                 n_heads=8, ffn_dim=1024, dropout=0.1, n_spec=3,
                 d_spec=16, max_obs=16, max_slots=64, new_weight=3.0):
        super().__init__()
        self.d_model = d_model
        self.max_obs = max_obs
        self.max_slots = max_slots
        self.new_weight = new_weight
        self.n_spec = n_spec
        self.temporal_encoder = TemporalIdentityEncoder(
            d_model=d_model, n_layers=temporal_layers, n_heads=n_heads,
            ffn_dim=ffn_dim, dropout=dropout, max_obs=max_obs)
        self.cand_proj = nn.Sequential(
            nn.Linear(PBD_DIM + len(CAND_FEATURES), d_model),
            nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ffn_dim, dropout, batch_first=True,
            norm_first=True, activation="gelu")
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=set_layers)
        self.spec_embed = nn.Embedding(n_spec, d_spec)
        self.spec_proj = nn.Linear(d_spec, d_model)
        # identity slot logits (slots + NEW)
        self.slot_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, max_slots + 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch):
        obs_pbd = torch.nan_to_num(batch["obs_pbd"].float())
        obs_feat = torch.nan_to_num(batch["obs_feat"].float())
        obs_mask = batch.get("obs_mask")
        if obs_mask is None:
            obs_mask = torch.ones(obs_pbd.shape[:3], dtype=torch.bool,
                                  device=obs_pbd.device)
        cand_pbd = torch.nan_to_num(batch["cand_pbd"].float())
        cand_feat = torch.nan_to_num(batch["cand_feat"].float())
        B, N = cand_pbd.shape[:2]
        trk_mask = batch.get("trk_mask")
        cand_mask = batch.get("cand_mask")
        if trk_mask is None:
            trk_mask = torch.ones(B, obs_pbd.shape[1], dtype=torch.bool,
                                  device=obs_pbd.device)
        if cand_mask is None:
            cand_mask = torch.ones(B, N, dtype=torch.bool, device=cand_pbd.device)
        state = self.temporal_encoder(obs_pbd, obs_feat, obs_mask)
        cand_tok = self.cand_proj(torch.cat(
            [torch.nn.functional.normalize(cand_pbd), cand_feat], dim=-1))
        spec = batch.get("spec")
        if spec is None:
            spec = torch.zeros(B, dtype=torch.long, device=cand_pbd.device)
        spec_tok = self.spec_proj(self.spec_embed(spec))
        state = state + spec_tok.unsqueeze(1)
        cand_tok = cand_tok + spec_tok.unsqueeze(1)
        tokens = torch.cat([cand_tok, state], dim=1)
        mask = torch.cat([cand_mask, trk_mask], dim=1)
        out = self.set_encoder(tokens, src_key_padding_mask=~mask)
        cand_out = out[:, :N]
        slot_logits = self.slot_head(cand_out)  # [B,N,max_slots+1]
        # mask logits of slots >= G (per-sample vocab size)
        g = batch.get("n_slots")  # [B]
        if g is not None:
            idx = torch.arange(self.max_slots + 1, device=slot_logits.device)
            valid = idx[None, :] <= g[:, None]
            slot_logits = slot_logits.masked_fill(~valid.unsqueeze(1),
                                                  float("-inf"))
        return {"slot_logits": slot_logits}


def l5b_loss(batch, pred, new_weight=3.0):
    """CE on candidate identity slots (NEW = n_slots)."""
    logits = pred["slot_logits"]
    B, N, G1 = logits.shape
    target = batch.get("slot_target")  # [B,N], -1 = no supervision
    cand_mask = batch.get("cand_mask")
    if target is None:
        return {"loss": logits.new_zeros(()), "ce": logits.new_zeros(()),
                "n": 0}
    valid = (target >= 0) & (cand_mask if cand_mask is not None
                             else torch.ones(B, N, dtype=torch.bool,
                                             device=logits.device))
    if not valid.any():
        return {"loss": logits.new_zeros(()), "ce": logits.new_zeros(()),
                "n": 0}
    t = target[valid]
    g = batch.get("n_slots")  # [B]
    is_new = target == g[:, None] if g is not None else torch.zeros_like(target)
    weights = torch.where(
        is_new[valid], logits.new_tensor(new_weight), logits.new_tensor(1.0))
    ce = torch.nn.functional.cross_entropy(
        logits[valid], t, reduction="none")
    ce = (ce * weights).mean()
    return {"loss": ce, "ce": ce, "n": int(valid.sum())}
