"""Unified Association Decoder (UA) for Stage L1-C.

Clean reimplementation based on audited official designs:
- MOTIP/FDTA: candidate self-attention + candidate->trajectory cross-attention
  with relative-time/geometry bias; assignment as K+1 (existing tracks + NEW).
- OVTR: persistent track queries updated by history self-attention; sequence
  -local IDs; miss-tolerance lifecycle handled by the shared tracker shell.
- COVTrack: multi-cue (appearance/geometry/score) fused inside one association
  head rather than hand-weighted costs.

No external code is copied; only the audited interface/design is followed.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

PBD_DIM = 2048
REGION_DIM = 4608
GEOM_DIM = 5  # x1,y1,x2,y2,area (normalized)


class FeatureProjector(nn.Module):
    def __init__(self, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.pbd_proj = nn.Sequential(
            nn.Linear(PBD_DIM, 96), nn.LayerNorm(96), nn.GELU(), nn.Dropout(dropout))
        self.region_proj = nn.Sequential(
            nn.Linear(REGION_DIM, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(dropout))
        self.aux_proj = nn.Sequential(
            nn.Linear(GEOM_DIM + 1, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout))
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, pbd, region, geom, gen, score=None):
        # pbd [*, 2048], region [*, 4608], geom [*,5], gen [*,1]
        pbd = F.normalize(torch.nan_to_num(pbd), dim=-1)
        region = F.normalize(torch.nan_to_num(region), dim=-1)
        geom = torch.nan_to_num(geom)
        gen = torch.nan_to_num(gen)
        h_pbd = self.pbd_proj(pbd)
        h_reg = self.region_proj(region)
        aux_in = torch.cat([geom, gen], dim=-1)
        h_aux = self.aux_proj(aux_in)
        return self.out_norm(torch.cat([h_pbd, h_reg, h_aux], dim=-1))


class TrackEncoder(nn.Module):
    """Causal 2-layer encoder over per-track K-frame history with missing mask."""

    def __init__(self, d_model: int = 256, nhead: int = 8,
                 ffn_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, ffn_dim, dropout, batch_first=True,
            norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.time_emb = nn.Embedding(32, d_model)

    def forward(self, tokens, times, mask):
        """tokens [B,T,K,D]; times [B,T,K]; mask [B,T,K] (True=valid)."""
        B, T, K, D = tokens.shape
        tok = tokens + self.time_emb(times.clamp(0, 31))
        flat = tok.reshape(B * T, K, D)
        flat_mask = mask.reshape(B * T, K)
        causal = torch.triu(
            torch.full((K, K), float("-inf"), device=tok.device), diagonal=1)
        out = self.encoder(flat, mask=causal,
                           src_key_padding_mask=~flat_mask)
        # mean-pool valid observations
        out = out.reshape(B, T, K, D)
        out = torch.nan_to_num(out)
        valid = mask.unsqueeze(-1).float()
        pooled = (out * valid).sum(dim=2) / (valid.sum(dim=2).clamp(min=1.0))
        return pooled  # [B,T,D]


class RelationBias(nn.Module):
    """Geometry/appearance/gap -> attention bias for candidate->track cross-attn."""

    def __init__(self, d_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, track_geom, cand_geom, gap, pbd_cos, region_cos, cand_gen):
        """track_geom [B,T,4] (last box normalized), cand_geom [B,N,4],
        gap [B,T], pbd_cos/region_cos [B,T,N], cand_gen [B,N]."""
        B, T, _ = track_geom.shape
        N = cand_geom.shape[1]
        tg = track_geom.unsqueeze(2)  # [B,T,1,4]
        cg = cand_geom.unsqueeze(1)   # [B,1,N,4]
        ix1 = torch.maximum(tg[..., 0], cg[..., 0])
        iy1 = torch.maximum(tg[..., 1], cg[..., 1])
        ix2 = torch.minimum(tg[..., 2], cg[..., 2])
        iy2 = torch.minimum(tg[..., 3], cg[..., 3])
        iw = torch.clamp(ix2 - ix1, min=0.0)
        ih = torch.clamp(iy2 - iy1, min=0.0)
        inter = iw * ih
        ar = torch.clamp((tg[..., 2] - tg[..., 0]) * (tg[..., 3] - tg[..., 1]), min=1e-6)
        ac = torch.clamp((cg[..., 2] - cg[..., 0]) * (cg[..., 3] - cg[..., 1]), min=1e-6)
        iou = inter / (ar + ac - inter + 1e-6)
        tc = (tg[..., :2] + tg[..., 2:]) / 2
        cc = (cg[..., :2] + cg[..., 2:]) / 2
        cdist = torch.sqrt(((tc - cc) ** 2).sum(-1) + 1e-6)
        lw = torch.log(torch.clamp((cg[..., 2] - cg[..., 0]) / ar.sqrt(), min=1e-4))
        lh = torch.log(torch.clamp((cg[..., 3] - cg[..., 1]) / ar.sqrt(), min=1e-4))
        larea = torch.log(ac / (ar + 1e-6))
        gap_f = torch.log1p(gap.unsqueeze(-1).float()).expand(B, T, N)
        gen_f = cand_gen.unsqueeze(1).expand(B, T, N)
        feats = torch.stack(
            [iou, cdist, lw, lh, larea, gap_f, pbd_cos, region_cos, gen_f,
             cand_geom.unsqueeze(1).expand(B, T, N, 4)[..., 3] / 2.0],
            dim=-1)
        bias = self.net(feats).squeeze(-1)  # [B,T,N]
        return bias.transpose(1, 2)  # [B,N,T] for query=candidate


class UADecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.cand_self = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.track_self = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.cross = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)
        self.norm = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model), nn.Dropout(dropout))
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, cand, track, cand_mask, track_mask, cross_bias):
        h = cand
        h2, _ = self.cand_self(h, h, h, key_padding_mask=None if cand_mask is None else ~cand_mask)
        h = self.norm[0](h + h2)
        t2, _ = self.track_self(track, track, track,
                                key_padding_mask=None if track_mask is None else ~track_mask)
        track = self.norm[1](track + torch.nan_to_num(t2))
        B, H, N, T = cross_bias.shape[0], self.cross.num_heads, h.shape[1], track.shape[1]
        kp = None if track_mask is None else ~track_mask
        bias = cross_bias.unsqueeze(1).expand(B, H, N, T).reshape(-1, N, T)
        if kp is not None:
            kp_f = kp.float().unsqueeze(1).unsqueeze(2).expand(B, H, N, T)
            bias = bias + (kp_f.reshape(-1, N, T) * -1e9)
            kp = None
        h2, _ = self.cross(h, track, track, key_padding_mask=kp, attn_mask=bias)
        h = self.norm[2](h + torch.nan_to_num(h2))
        h = self.norm_ffn(h + self.ffn(h))
        return h, track


class UnifiedAssociationDecoder(nn.Module):
    def __init__(self, d_model: int = 256, num_layers: int = 4,
                 num_heads: int = 8, ffn_dim: int = 1024, dropout: float = 0.1,
                 max_k: int = 8):
        super().__init__()
        self.d_model = d_model
        self.max_k = max_k
        self.proj = FeatureProjector(d_model, dropout)
        self.track_enc = TrackEncoder(d_model, num_heads, ffn_dim, dropout)
        self.rel_bias = RelationBias(128)
        self.layers = nn.ModuleList([
            UADecoderLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.track_key = nn.Linear(d_model, d_model)
        self.new_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Linear(128, 1))
        self.motion_head = nn.Sequential(
            nn.Linear(d_model * 2, 256), nn.GELU(),
            nn.Linear(256, 4))

    def project_obs(self, pbd, region, geom, gen):
        return self.proj(pbd, region, geom, gen)

    def _track_tokens(self, obs_pbd, obs_region, obs_geom, obs_gen, times, mask):
        """obs_* [B,T,K,...]"""
        B, T, K = mask.shape
        flat_pbd = obs_pbd.reshape(B * T * K, -1)
        flat_reg = obs_region.reshape(B * T * K, -1)
        flat_geom = obs_geom.reshape(B * T * K, -1)
        flat_gen = obs_gen.reshape(B * T * K, -1)
        tok = self.project_obs(flat_pbd, flat_reg, flat_geom, flat_gen)
        tok = tok.reshape(B, T, K, self.d_model)
        return self.track_enc(tok, times, mask)  # [B,T,D]

    def forward(self, batch):
        """batch keys:
        cur_pbd/cur_region/cur_geom/cur_gen: [B,N,...]
        trk_pbd/trk_region/trk_geom/trk_gen: [B,T,K,...]
        trk_times/trk_mask: [B,T,K]
        trk_last_geom: [B,T,4]; cur_norm_geom: [B,N,4]
        gap: [B,T]
        pbd_cos/region_cos: [B,T,N] (raw cosine)
        """
        cur = self.project_obs(batch["cur_pbd"], batch["cur_region"],
                               batch["cur_geom"], batch["cur_gen"])
        track = self._track_tokens(
            batch["trk_pbd"], batch["trk_region"], batch["trk_geom"],
            batch["trk_gen"], batch["trk_times"], batch["trk_mask"])
        cand_mask = batch.get("cur_mask")
        track_mask = batch.get("trk_valid")  # [B,T] True if track has valid history
        bias = self.rel_bias(
            batch["trk_last_geom"], batch["cur_norm_geom"], batch["gap"],
            batch["pbd_cos"], batch["region_cos"], batch["cur_gen"][..., 0])
        for layer in self.layers:
            cur, track = layer(cur, track, cand_mask, track_mask, bias)
        scores = torch.bmm(cur, self.track_key(track).transpose(1, 2)) / math.sqrt(self.d_model)
        new_logit = self.new_head(cur).squeeze(-1)  # [B,N]
        logits = torch.cat([scores, new_logit.unsqueeze(-1)], dim=-1)  # [B,N,K+1]
        return {
            "logits": logits,
            "scores": scores,
            "new_logits": new_logit,
            "cur_out": cur,
            "track_out": track,
            "motion_delta": (self.predict_motion(cur, track, batch["assign_targets"])
                             if "assign_targets" in batch else None),
        }

    def predict_motion(self, cur_out, track_out, track_idx):
        """cur_out [B,N,D], track_out [B,T,D], track_idx [B,N] -> [B,N,4]."""
        B, N, D = cur_out.shape
        idx = track_idx.clamp(0, track_out.shape[1] - 1)
        sel = torch.gather(track_out, 1, idx.unsqueeze(-1).expand(B, N, D))
        return self.motion_head(torch.cat([cur_out, sel], dim=-1))


def association_loss(batch, pred, new_weight: float = 0.2):
    """Per-candidate CE over K+1 labels; mask out invalid candidates."""
    logits = pred["logits"]
    B, N, K1 = logits.shape
    K = K1 - 1
    labels = batch["assign_targets"]  # [B,N] in [0,K], K = NEW
    valid = batch["assign_valid"].bool()
    weight = torch.ones(K1, dtype=logits.dtype, device=logits.device)
    weight[-1] = new_weight
    loss = torch.zeros((), device=logits.device, dtype=logits.dtype)
    count = 0
    for b in range(B):
        v = valid[b]
        if not v.any():
            continue
        loss = loss + torch.nn.functional.cross_entropy(
            logits[b][v], labels[b][v], weight=weight, reduction="sum")
        count += int(v.sum())
    loss = loss / max(1, count)
    return loss, count


def motion_loss(batch, pred):
    """SmoothL1 on matched candidates' delta-box; returns (loss, count)."""
    if "motion_target" not in batch or "motion_mask" not in batch:
        return torch.zeros((), device=pred["cur_out"].device), 0
    cur_out = pred["cur_out"]
    track_out = pred["track_out"]
    track_idx = batch["assign_targets"].clamp(max=track_out.shape[1] - 1)
    delta = pred.get("motion_delta")
    if delta is None:
        delta = cur_out.new_zeros((cur_out.shape[0], cur_out.shape[1], 4))
    mask = batch["motion_mask"].bool()
    loss = torch.zeros((), device=cur_out.device)
    count = 0
    for b in range(cur_out.shape[0]):
        m = mask[b]
        if not m.any():
            continue
        loss = loss + torch.nn.functional.smooth_l1_loss(
            delta[b][m], batch["motion_target"][b][m], reduction="sum")
        count += int(m.sum())
    return loss / max(1, count), count
