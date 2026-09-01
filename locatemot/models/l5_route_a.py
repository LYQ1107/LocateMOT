"""Stage L5 Route A: GT-anchored temporal identity transformer.

Scientific hypothesis (Route A): cross-spec identity drift in U0 comes from
the absence of a persistent temporal identity state; a model that compresses
each track's observation history into a state and decodes associations with
set-level competition, supervised by GT trajectory identity (never by
tracker integer IDs), can learn specification-independent identity semantics
while keeping specification-dependent evidence.

Architecture:
  observation sequence (per track, causal)
      -> Temporal Identity Encoder (causal TransformerEncoder)
      -> persistent state h_i^t
  current candidates -> candidate tokens
      -> Set-level Track-Candidate Interaction (shared TransformerEncoder)
      -> pair head: delta on top of the frozen L1DK base affinity
      -> reliability gate (same pattern as L1D EGRA)

Clean reimplementation guided by audited official designs (no code copied):
- MOTIP IDDecoder / TrajectoryModeling (CVPR 2025, Apache-2.0,
  commit ffc0e905): trajectory features + relative-time interaction,
  track-candidate cross attention, sequence-local identity targets.
- CAMELTrack GAFFE set-level interaction + ranking training
  (commit 46a74bb).
- L1D EGRA (own project, previous stage): bounded residual + reliability
  gate + row/column ranking CE.
- SOTFormer (CVPR 2026, MIT, commit bb28e62): GT-primed initialization /
  constant-memory state update as conceptual reference only.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from locatemot.models.l1d_association import (
    CAND_FEATURES,
    PAIR_FEATURES,
    PBD_DIM,
    DELTA_SCALE,
)

OBS_NUM_FEATURES = 9  # box(4) + velocity(2) + gen(1) + log_n_cand(1) + gap(1)


def l2norm(x, dim=-1):
    return F.normalize(x.float(), dim=dim, eps=1e-6)


class TemporalIdentityEncoder(nn.Module):
    """Causal per-track transformer over observation sequences."""

    def __init__(self, d_model=256, n_layers=3, n_heads=8, ffn_dim=1024,
                 dropout=0.1, max_obs=16):
        super().__init__()
        self.d_model = d_model
        self.max_obs = max_obs
        self.obs_proj = nn.Sequential(
            nn.Linear(PBD_DIM + OBS_NUM_FEATURES, d_model),
            nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.pos_embed = nn.Parameter(torch.zeros(max_obs, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ffn_dim, dropout, batch_first=True,
            norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def causal_mask(self, K, device):
        return torch.triu(
            torch.full((K, K), float("-inf"), device=device), diagonal=1)

    def forward(self, pbd, feat, mask):
        """pbd [B,T,K,D], feat [B,T,K,F], mask [B,T,K] True=valid.
        Returns state [B,T,D] = encoder output at the last valid obs."""
        B, T, K, D = pbd.shape
        x = torch.cat([l2norm(pbd), feat], dim=-1)  # [B,T,K,D+F]
        x = self.obs_proj(x)
        pos = torch.arange(K, device=x.device)[None, None, :]
        pos_idx = (self.max_obs - 1 - pos).clamp(min=0)
        x = x + self.pos_embed[pos_idx.expand(B, T, K)]
        flat = x.reshape(B * T, K, self.d_model)
        flat_mask = mask.reshape(B * T, K)
        # zero out invalid rows so they cannot contribute (causal mask covers
        # future only; invalid obs are zeroed and therefore contribute
        # nothing to attention outputs).
        flat = flat * flat_mask.unsqueeze(-1).float()
        causal = self.causal_mask(K, x.device)  # [K,K]
        out = self.encoder(flat, mask=causal)  # [B*T,K,D]
        # state = last valid position (K-1 after right-aligned padding)
        state = out.reshape(B, T, K, self.d_model)[:, :, -1, :]
        return state


class L5TemporalAssociator(nn.Module):
    """Temporal identity encoder + set-level decoder + bounded residual."""

    def __init__(self, d_model=256, temporal_layers=3, set_layers=3,
                 n_heads=8, ffn_dim=1024, dropout=0.1, n_spec=3,
                 d_spec=16, max_obs=16, delta_scale=DELTA_SCALE):
        super().__init__()
        self.d_model = d_model
        self.max_obs = max_obs
        self.delta_scale = delta_scale
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
        self.pair_head = nn.Sequential(
            nn.Linear(2 * d_model + len(PAIR_FEATURES), 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1))
        self.reliability_head = nn.Sequential(
            nn.Linear(d_model + 7, 96),
            nn.LayerNorm(96), nn.GELU(),
            nn.Linear(96, 1))
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
        pair = torch.nan_to_num(batch["pair_feats"].float())
        base = torch.nan_to_num(batch["base"].float())
        B, T, N, Fp = pair.shape
        trk_mask = batch.get("trk_mask")
        cand_mask = batch.get("cand_mask")
        if trk_mask is None:
            trk_mask = torch.ones(B, T, dtype=torch.bool, device=pair.device)
        if cand_mask is None:
            cand_mask = torch.ones(B, N, dtype=torch.bool, device=pair.device)

        state = self.temporal_encoder(obs_pbd, obs_feat, obs_mask)  # [B,T,D]
        # zero states for invalid tracks (key padding mask excludes them)
        cand_tok = self.cand_proj(torch.cat([l2norm(cand_pbd), cand_feat], dim=-1))

        spec = batch.get("spec")
        if spec is None:
            spec = torch.zeros(B, dtype=torch.long, device=pair.device)
        spec_tok = self.spec_proj(self.spec_embed(spec))  # [B,D]
        state = state + spec_tok.unsqueeze(1)
        cand_tok = cand_tok + spec_tok.unsqueeze(1)

        tokens = torch.cat([cand_tok, state], dim=1)
        mask = torch.cat([cand_mask, trk_mask], dim=1)
        out = self.set_encoder(tokens, src_key_padding_mask=~mask)
        cand_out = out[:, :N]
        trk_out = out[:, N:]

        t_exp = trk_out.unsqueeze(2).expand(B, T, N, self.d_model)
        c_exp = cand_out.unsqueeze(1).expand(B, T, N, self.d_model)
        pair_in = torch.cat([t_exp, c_exp, pair], dim=-1)
        delta = self.delta_scale * torch.tanh(self.pair_head(pair_in).squeeze(-1))

        row_sel = torch.stack([
            batch["track_feats"][..., 9] if "track_feats" in batch
            else state.new_zeros(B, T),
            batch["track_feats"][..., 10] if "track_feats" in batch
            else state.new_zeros(B, T),
            batch["track_feats"][..., 11] if "track_feats" in batch
            else state.new_zeros(B, T),
            batch["track_feats"][..., 12] if "track_feats" in batch
            else state.new_zeros(B, T),
            batch["track_feats"][..., 13] if "track_feats" in batch
            else state.new_zeros(B, T),
            batch["track_feats"][..., 6] if "track_feats" in batch
            else state.new_zeros(B, T),
            pair.max(dim=2).values[..., 11],
        ], dim=-1)
        rel_in = torch.cat([trk_out, row_sel], dim=-1)
        rel_logit = self.reliability_head(rel_in).squeeze(-1)
        final = base + torch.sigmoid(rel_logit).unsqueeze(-1) * delta
        return {
            "final": final,
            "delta": delta,
            "reliability_logit": rel_logit,
            "reliability": torch.sigmoid(rel_logit),
            "trk_tok": trk_out,
            "cand_tok": cand_out,
            "state": state,
        }


def l5_association_loss(batch, pred, rel_weight=0.3, pres_weight=0.1,
                        pos_weight=9.0):
    """Row/col ranking CE + reliability + preservation (same as L1D)."""
    return l1d_loss_impl(batch, pred, rel_weight, pres_weight, pos_weight)


def l1d_loss_impl(batch, pred, rel_weight=0.3, pres_weight=0.1,
                  pos_weight=9.0):
    final = pred["final"]
    B, T, N = final.shape
    row_label = batch.get("row_label",
                          torch.full((B, T), -1, dtype=torch.long,
                                     device=final.device))
    col_label = batch.get("col_label",
                          torch.full((B, N), -1, dtype=torch.long,
                                     device=final.device))
    base_correct = batch.get("base_correct",
                             torch.ones(B, T, dtype=torch.bool,
                                        device=final.device))
    valid_row = row_label >= 0
    valid_col = col_label >= 0
    # mask padded tracks/candidates so CE cannot use padding
    final = final.clone()
    cand_mask = batch.get("cand_mask",
                          torch.ones(B, N, dtype=torch.bool,
                                     device=final.device))
    trk_mask = batch.get("trk_mask",
                         torch.ones(B, T, dtype=torch.bool,
                                    device=final.device))
    final = final.masked_fill(
        ~cand_mask.unsqueeze(1).expand(B, T, N), float("-inf"))
    final = final.masked_fill(
        ~trk_mask.unsqueeze(2).expand(B, T, N), float("-inf"))
    row_loss = final.new_zeros(())
    col_loss = final.new_zeros(())
    n_row = int(valid_row.sum())
    n_col = int(valid_col.sum())
    if n_row:
        row_loss = F.cross_entropy(
            final[valid_row].reshape(-1, N),
            row_label[valid_row].reshape(-1), reduction="mean")
    if n_col:
        col_loss = F.cross_entropy(
            final.transpose(1, 2)[valid_col].reshape(-1, T),
            col_label[valid_col].reshape(-1), reduction="mean")
    denom = (n_row > 0) + (n_col > 0)
    loss = final.new_zeros(())
    if denom:
        loss = (row_loss + col_loss) / denom
    if n_row:
        base_wrong = (~base_correct).float()
        rel = F.binary_cross_entropy_with_logits(
            pred["reliability_logit"][valid_row],
            base_wrong[valid_row], pos_weight=final.new_tensor(pos_weight))
        loss = loss + rel_weight * rel
    preserve = (base_correct & valid_row).unsqueeze(-1)
    if preserve.any():
        pres = (pred["delta"].abs() * preserve.float()).sum() / preserve.float().sum()
        loss = loss + pres_weight * pres
    else:
        pres = final.new_zeros(())
    return {
        "loss": loss,
        "row_ce": row_loss,
        "col_ce": col_loss,
        "n_row": n_row,
        "n_col": n_col,
        "pres": pres,
    }


def l5_relation_loss(batch, pred, max_pairs=64):
    """Same/different GT trajectory relation BCE on persistent states.

    The state is a temporal process, not a single-frame embedding; this loss
    supervises identity semantics directly from GT trajectory identity.
    """
    state = pred["state"]  # [B,T,D]
    B, T, D = state.shape
    trk_mask = batch.get("trk_mask", torch.ones(B, T, dtype=torch.bool,
                                                device=state.device))
    track_gt = batch.get("track_gt")  # [B,T] as int labels (-1 = none)
    if track_gt is None:
        return state.new_zeros(()), 0
    pos_pairs = 0
    neg_pairs = 0
    losses = []
    for b in range(B):
        idx = torch.nonzero(trk_mask[b]).squeeze(-1)
        if idx.numel() < 2:
            continue
        g = track_gt[b]
        same = (g[idx][:, None] == g[idx][None, :]) & (g[idx][None, :] >= 0)
        iu = torch.triu_indices(idx.numel(), idx.numel(), offset=1,
                                device=state.device)
        if iu.numel() == 0:
            continue
        si, sj = iu[0], iu[1]
        y = same[si, sj].float()
        keep = same[si, sj] | (g[idx][si] != g[idx][sj])
        keep &= (g[idx][si] >= 0) & (g[idx][sj] >= 0)
        keep &= ~(same[si, sj] & (g[idx][si] < 0))
        si, sj, y = si[keep], sj[keep], y[keep]
        if si.numel() == 0:
            continue
        if si.numel() > max_pairs:
            sel = torch.randperm(si.numel(), device=state.device)[:max_pairs]
            si, sj, y = si[sel], sj[sel], y[sel]
        h1 = state[b, idx[si]]
        h2 = state[b, idx[sj]]
        logit = (h1 * h2).sum(-1) * 0.1
        losses.append(F.binary_cross_entropy_with_logits(
            logit, y, reduction="mean"))
        pos_pairs += int((y > 0).sum())
        neg_pairs += int((y == 0).sum())
    if not losses:
        return state.new_zeros(()), 0
    return torch.stack(losses).mean(), (pos_pairs, neg_pairs)


def l5_cross_spec_relation_loss(groups, preds):
    """Relation-structure consistency between ALL and restricted views.

    groups: list of (batch_index_list) samples sharing (video, frame, source).
    preds: list of model outputs for each sample in the group.
    For common GT identities, cosine relation matrices R_ALL and R_SPEC
    should agree; both are separately GT-anchored, so this never forces
    restricted to imitate an erroneous ALL prediction.
    """
    losses = []
    n_common = 0
    for group in groups:
        if len(group) < 2:
            continue
        states = []
        gts = []
        for b, p, g in group:
            states.append(p["state"][b])
            gts.append(g)
        common = set(gts[0]) & set(gts[1])
        if len(common) < 2:
            continue
        mats = []
        for st, g in zip(states, gts):
            keep = [i for i, x in enumerate(g) if x in common]
            h = st[keep]
            h = F.normalize(h, dim=-1)
            mats.append(h @ h.T)
        loss = ((mats[0] - mats[1]) ** 2).mean()
        losses.append(loss)
        n_common += len(common)
    if not losses:
        return None
    return torch.stack(losses).mean(), n_common
