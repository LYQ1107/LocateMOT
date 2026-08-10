"""Stage L3: regime- and specification-conditioned shared association core.

U0 = L1DAssociator (shared dense, no conditioning) -- l1d_association.py.
U1 = L3Associator: same set-level association core conditioned by
     (a) a latent regime token z_regime from prediction-side state,
     (b) an object-specification token (ALL / category / text / visual).

Design evidence: docs/l3_reference_audit.md (TDLP/CAMELTrack set-level
competition; SAM3/GLEE spec interfaces; no verified equivalent latent-regime
conditional box-MOT association implementation found). Clean reimplementation.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from locatemot.models.l1d_association import (
    CAND_FEATURES,
    PAIR_FEATURES,
    TRACK_FEATURES,
    l1d_loss,
)

SPECS = [
    "ALL", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush", "rider", "other person",
    "other vehicle", "trailer", "OPEN",
]
SPEC2ID = {s: i for i, s in enumerate(SPECS)}


def regime_features_from_batch(batch):
    """Prediction-side regime statistics from a collated batch -> [B,48]."""
    pair = torch.nan_to_num(batch["pair_feats"].float())
    trk = torch.nan_to_num(batch["track_feats"].float())
    cand = torch.nan_to_num(batch["cand_feats"].float())
    base = torch.nan_to_num(batch["base"].float())
    B, T, N, Fp = pair.shape
    m = batch["trk_mask"].unsqueeze(-1).float()
    cm = batch["cand_mask"].unsqueeze(-1).float()
    t_count = m.sum(1).clamp(min=1).squeeze(-1)
    c_count = cm.sum(1).clamp(min=1).squeeze(-1)
    feat = []

    def add(x):
        feat.append(x.unsqueeze(-1) if x.dim() == 1 else x)

    add(torch.log1p(t_count))
    add(torch.log1p(c_count))
    pm = m.unsqueeze(-1) * cm.unsqueeze(1)
    for _, idx in [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
                   (6, 6), (7, 7), (11, 11), (14, 14)]:
        x = pair[..., idx]
        denom = pm[..., 0].sum((1, 2)).clamp(min=1)
        add((x * pm[..., 0]).sum((1, 2)) / denom)
        add((x * pm[..., 0]).max(dim=1).values.max(dim=1).values)
    for _, idx in [(4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9),
                   (10, 10), (11, 11), (12, 12)]:
        x = trk[..., idx]
        xm = x * m.squeeze(-1)
        add(xm.sum(1) / t_count)
        add(xm.max(1).values)
    for _, idx in [(4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]:
        x = cand[..., idx]
        xm = x * cm.squeeze(-1)
        add(xm.sum(1) / c_count)
        add(xm.max(1).values)
    bm = pm[..., 0]
    add((base * bm).sum((1, 2)) / bm.sum((1, 2)).clamp(min=1))
    add(((base >= 0.25).float() * bm).sum((1, 2)) / bm.sum((1, 2)).clamp(min=1))
    add(trk[..., 7].std(dim=1, unbiased=False))
    x = torch.cat(feat, dim=-1)
    if x.shape[-1] < 48:
        x = F.pad(x, (0, 48 - x.shape[-1]))
    return x[..., :48]


class RegimeEncoder(nn.Module):
    """Latent tracking regime encoder: prediction-side stats -> z_regime."""

    def __init__(self, in_dim=48, z_dim=32, hidden=128):
        super().__init__()
        self.z_dim = z_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, z_dim))

    def forward(self, batch):
        feats = regime_features_from_batch(batch)
        return self.mlp(feats)


class L3Associator(nn.Module):
    """Shared set-level association core with regime + spec conditioning."""

    def __init__(self, d_model=128, n_layers=2, n_heads=4, ffn_dim=512,
                 dropout=0.1, z_dim=32, use_spec=True, n_spec=len(SPECS)):
        super().__init__()
        self.d_model = d_model
        self.z_dim = z_dim
        self.use_spec = use_spec
        self.regime_enc = RegimeEncoder(in_dim=48, z_dim=z_dim)
        self.spec_embed = nn.Embedding(n_spec, d_model) if use_spec else None

        self.track_proj = nn.Sequential(
            nn.Linear(len(TRACK_FEATURES), d_model),
            nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.cand_proj = nn.Sequential(
            nn.Linear(len(CAND_FEATURES), d_model),
            nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ffn_dim, dropout, batch_first=True,
            norm_first=True, activation="gelu")
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.film_track = nn.Linear(z_dim, 2 * d_model)
        self.film_cand = nn.Linear(z_dim, 2 * d_model)
        self.film_enc = nn.Linear(z_dim, 2 * d_model)
        pair_in = 2 * d_model + len(PAIR_FEATURES) + z_dim
        if use_spec:
            pair_in += d_model
        self.pair_head = nn.Sequential(
            nn.Linear(pair_in, 192), nn.LayerNorm(192), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(192, 96), nn.GELU(),
            nn.Linear(96, 1))
        self.reliability_head = nn.Sequential(
            nn.Linear(d_model + 7 + z_dim, 96), nn.LayerNorm(96), nn.GELU(),
            nn.Linear(96, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch):
        pair = torch.nan_to_num(batch["pair_feats"].float())
        trk_f = torch.nan_to_num(batch["track_feats"].float())
        cand_f = torch.nan_to_num(batch["cand_feats"].float())
        base = torch.nan_to_num(batch["base"].float())
        B, T, N, Fp = pair.shape
        trk_mask = batch["trk_mask"]
        cand_mask = batch["cand_mask"]
        z = self.regime_enc(batch)
        spec = batch.get("spec")
        if spec is None:
            spec = torch.zeros(B, dtype=torch.long, device=pair.device)
        spec_tok = self.spec_embed(spec) if self.spec_embed is not None \
            else torch.zeros(B, self.d_model, device=pair.device)

        trk_tok = self.track_proj(trk_f)
        cand_tok = self.cand_proj(cand_f)
        if self.spec_embed is not None:
            trk_tok = trk_tok + spec_tok.unsqueeze(1)
            cand_tok = cand_tok + spec_tok.unsqueeze(1)
        g_t, b_t = self.film_track(z).chunk(2, dim=-1)
        g_c, b_c = self.film_cand(z).chunk(2, dim=-1)
        g_e, b_e = self.film_enc(z).chunk(2, dim=-1)
        trk_tok = trk_tok * g_t.unsqueeze(1) + b_t.unsqueeze(1)
        cand_tok = cand_tok * g_c.unsqueeze(1) + b_c.unsqueeze(1)
        tokens = torch.cat([cand_tok, trk_tok], dim=1)
        tokens = tokens * g_e.unsqueeze(1) + b_e.unsqueeze(1)
        mask = torch.cat([cand_mask, trk_mask], dim=1)
        out = self.set_encoder(tokens, src_key_padding_mask=~mask)
        cand_out = out[:, :N]
        trk_out = out[:, N:]
        t_exp = trk_out.unsqueeze(2).expand(B, T, N, self.d_model)
        c_exp = cand_out.unsqueeze(1).expand(B, T, N, self.d_model)
        z_exp = z.unsqueeze(1).unsqueeze(2).expand(B, T, N, self.z_dim)
        pair_in = [t_exp, c_exp, pair, z_exp]
        if self.spec_embed is not None:
            s_exp = spec_tok.unsqueeze(1).unsqueeze(2).expand(
                B, T, N, self.d_model)
            pair_in.append(s_exp)
        delta = 0.6 * torch.tanh(
            self.pair_head(torch.cat(pair_in, dim=-1)).squeeze(-1))

        row_sel = torch.stack([
            trk_f[..., 9], trk_f[..., 10], trk_f[..., 11], trk_f[..., 12],
            trk_f[..., 13], trk_f[..., 6],
            pair.max(dim=2).values[..., 11],
        ], dim=-1)
        rel_in = torch.cat([trk_out, row_sel,
                            z.unsqueeze(1).expand(B, T, self.z_dim)], dim=-1)
        rel_logit = self.reliability_head(rel_in).squeeze(-1)
        final = base + torch.sigmoid(rel_logit).unsqueeze(-1) * delta
        return {
            "final": final,
            "delta": delta,
            "reliability_logit": rel_logit,
            "reliability": torch.sigmoid(rel_logit),
            "z_regime": z,
        }


__all__ = ["L3Associator", "RegimeEncoder", "SPEC2ID", "SPECS", "l1d_loss"]
