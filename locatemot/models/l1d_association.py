"""Stage L1-D: Evidence-Gated Set-Level Residual Association (EGRA).

Design evidence: docs/l1_d_architecture_evidence.md,
reports/l1_d_structure_decision.md.

The strong base affinity (IoU + PBD cosine + constant-velocity motion IoU)
is preserved; a lightweight set-level transformer emits a bounded residual
and a track-level reliability gate.  Assignment is row/column ranking CE
without a NEW class; NEW is produced at inference by a shared threshold.

Clean reimplementation guided by audited official designs:
- CAMELTrack (GAFFE set-level interaction, InfoNCE/ranking training,
  Hungarian + threshold) -- commit 46a74bb;
- LLTrack / LG-Track (multi-cue cost fusion, confidence-gated cost).
No external code is copied.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PBD_DIM = 2048
DELTA_SCALE = 0.6

PAIR_FEATURES = [
    "iou", "iou_pred", "pbd_cos", "cd", "cd_pred", "log_scale",
    "gen", "gap", "log_n_cand", "iou_margin_row", "pbd_margin_row",
    "base_margin_row", "iou_margin_col", "pbd_margin_col",
    "base_margin_col", "cand_size", "track_age", "log_n_track",
    "pbd_anchor_cos",
]
TRACK_FEATURES = [
    "bx1", "by1", "bx2", "by2", "velx", "vely", "gap", "age", "hits",
    "iou_top1", "pbd_top1", "base_top1", "base_margin",
    "log_n_cand", "log_n_track", "anchor_cos_cur",
]
CAND_FEATURES = [
    "bx1", "by1", "bx2", "by2", "gen", "cand_size", "iou_top1_col",
    "pbd_top1_col", "base_top1_col", "base_margin_col",
    "log_n_cand", "log_n_track",
]


def _iou_matrix(refs, curs):
    r = np.asarray(refs, np.float64)[:, None, :]
    c = np.asarray(curs, np.float64)[None, :, :]
    ix1 = np.maximum(r[:, :, 0], c[:, :, 0])
    iy1 = np.maximum(r[:, :, 1], c[:, :, 1])
    ix2 = np.minimum(r[:, :, 2], c[:, :, 2])
    iy2 = np.minimum(r[:, :, 3], c[:, :, 3])
    iw = np.maximum(0.0, ix2 - ix1)
    ih = np.maximum(0.0, iy2 - iy1)
    inter = iw * ih
    ar = np.maximum(0.0, r[:, :, 2] - r[:, :, 0]) * np.maximum(0.0, r[:, :, 3] - r[:, :, 1])
    ac = np.maximum(0.0, c[:, :, 2] - c[:, :, 0]) * np.maximum(0.0, c[:, :, 3] - c[:, :, 1])
    union = ar + ac - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def _norm_pbd(pbd: np.ndarray) -> np.ndarray:
    pbd = np.nan_to_num(np.asarray(pbd, dtype=np.float32))
    n = np.linalg.norm(pbd, axis=-1, keepdims=True)
    return pbd / np.maximum(n, 1e-6)


def compute_affinity_features(
    track_boxes: np.ndarray,
    cand_boxes: np.ndarray,
    ref_pbd: np.ndarray,
    anchor_pbd: np.ndarray,
    cand_pbd: np.ndarray,
    cand_gen: np.ndarray,
    gaps: np.ndarray,
    ages: np.ndarray,
    hits: np.ndarray,
    prev_boxes: np.ndarray,
    weights,
    image_size=(1280, 720),
    motion_pred_boxes=None,
    app_dim=2048,
):
    """Compute prediction-side pair/track/candidate features and base affinity.

    All inputs are numpy, pixel coordinates.  Returns a dict with:
      pair_feats [T,N,F], track_feats [T,Ft], cand_feats [N,Fc],
      base [T,N], iou [T,N], iou_pred [T,N], pbd_cos [T,N].
    """
    T = len(track_boxes)
    N = len(cand_boxes)
    iw, ih = image_size
    diag = float(np.hypot(iw, ih)) + 1e-6
    wi, wp, wm = (float(weights[0]), float(weights[1]), float(weights[2]))

    tb = np.asarray(track_boxes, dtype=np.float64).reshape(T, 4)
    cb = np.asarray(cand_boxes, dtype=np.float64).reshape(N, 4)
    gaps = np.asarray(gaps, dtype=np.float32).reshape(T)
    ages = np.asarray(ages, dtype=np.float32).reshape(T)
    hits = np.asarray(hits, dtype=np.float32).reshape(T)
    gen = np.asarray(cand_gen, dtype=np.float32).reshape(N)
    if prev_boxes is None:
        pb = tb.copy()
    else:
        pb = np.asarray(prev_boxes, dtype=np.float64).reshape(T, 4)

    # motion prediction (constant velocity by default; Kalman boxes if given)
    vel = tb - pb
    if motion_pred_boxes is None:
        gap_f = np.maximum(gaps[:, None], 1.0)
        pred = tb + vel * gap_f
    else:
        pred = np.asarray(motion_pred_boxes, dtype=np.float64).reshape(T, 4)

    iou = _iou_matrix(tb, cb)
    iou_pred = _iou_matrix(pred, cb)

    rn = _norm_pbd(np.asarray(ref_pbd, dtype=np.float32).reshape(T, app_dim))
    if anchor_pbd is None:
        an = rn
    else:
        an = _norm_pbd(np.asarray(anchor_pbd, dtype=np.float32).reshape(T, app_dim))
    cn = _norm_pbd(np.asarray(cand_pbd, dtype=np.float32).reshape(N, app_dim))
    pbd_cos = (rn @ cn.T).astype(np.float32) if T and N else np.zeros((T, N), np.float32)
    pbd_anchor_cos = (an @ cn.T).astype(np.float32) if T and N else np.zeros((T, N), np.float32)
    anchor_cos_cur = (an * rn).sum(-1).astype(np.float32) if T else np.zeros(T, np.float32)

    tc = (tb[:, :2] + tb[:, 2:]) / 2.0
    pc = (pred[:, :2] + pred[:, 2:]) / 2.0
    cc = (cb[:, :2] + cb[:, 2:]) / 2.0
    cd = (np.sqrt(((tc[:, None, :] - cc[None, :, :]) ** 2).sum(-1)) / diag).astype(np.float32)
    cd_pred = (np.sqrt(((pc[:, None, :] - cc[None, :, :]) ** 2).sum(-1)) / diag).astype(np.float32)

    tw = tb[:, 2] - tb[:, 0]
    th = tb[:, 3] - tb[:, 1]
    cw = cb[:, 2] - cb[:, 0]
    ch = cb[:, 3] - cb[:, 1]
    log_scale = (np.log(np.maximum(cw, 1.0) / np.maximum(tw[:, None], 1.0) + 1e-6) ** 2
                 + np.log(np.maximum(ch, 1.0) / np.maximum(th[:, None], 1.0) + 1e-6) ** 2)
    log_scale = np.sqrt(log_scale).astype(np.float32)
    cand_size = (np.sqrt(np.maximum(cw, 0.0) * np.maximum(ch, 0.0)) / diag).astype(np.float32)
    log_n_cand = np.full((T, N), np.log1p(N), dtype=np.float32)
    log_n_track = np.full((T, N), np.log1p(T), dtype=np.float32)

    base = wi * iou + wp * pbd_cos + wm * iou_pred
    base = base.astype(np.float32)

    def margin_top2(mat):
        if mat.shape[1] < 2:
            return np.zeros(mat.shape[0], dtype=np.float32)
        s = np.sort(mat, axis=1)
        return (s[:, -1] - s[:, -2]).astype(np.float32)

    def margin_top2_col(mat):
        if mat.shape[0] < 2:
            return np.zeros(mat.shape[1], dtype=np.float32)
        s = np.sort(mat, axis=0)
        return (s[-1, :] - s[-2, :]).astype(np.float32)

    iou_mr = margin_top2(iou)
    pbd_mr = margin_top2(pbd_cos)
    base_mr = margin_top2(base)
    iou_mc = margin_top2_col(iou)
    pbd_mc = margin_top2_col(pbd_cos)
    base_mc = margin_top2_col(base)

    # [T,N,F]
    pair = np.stack([
        iou.astype(np.float32), iou_pred.astype(np.float32), pbd_cos,
        cd, cd_pred, log_scale,
        np.broadcast_to(gen[None, :], (T, N)),
        np.broadcast_to(gaps[:, None], (T, N)),
        log_n_cand, np.broadcast_to(iou_mr[:, None], (T, N)),
        np.broadcast_to(pbd_mr[:, None], (T, N)),
        np.broadcast_to(base_mr[:, None], (T, N)),
        np.broadcast_to(iou_mc[None, :], (T, N)),
        np.broadcast_to(pbd_mc[None, :], (T, N)),
        np.broadcast_to(base_mc[None, :], (T, N)),
        np.broadcast_to(cand_size[None, :], (T, N)),
        np.broadcast_to(ages[:, None], (T, N)),
        log_n_track,
        pbd_anchor_cos,
    ], axis=-1).astype(np.float32)

    def row_top1(mat):
        return np.max(mat, axis=1).astype(np.float32) if mat.shape[1] else np.zeros(T, np.float32)

    def col_top1(mat):
        return np.max(mat, axis=0).astype(np.float32) if mat.shape[0] else np.zeros(N, np.float32)

    trk = np.stack([
        tb[:, 0] / iw, tb[:, 1] / ih, tb[:, 2] / iw, tb[:, 3] / ih,
        vel[:, 0] / iw, vel[:, 1] / ih, gaps, ages, hits,
        row_top1(iou), row_top1(pbd_cos), row_top1(base), base_mr,
        np.full(T, np.log1p(N), np.float32),
        np.full(T, np.log1p(T), np.float32),
        anchor_cos_cur,
    ], axis=-1).astype(np.float32)

    cand = np.stack([
        cb[:, 0] / iw, cb[:, 1] / ih, cb[:, 2] / iw, cb[:, 3] / ih,
        gen, cand_size, col_top1(iou), col_top1(pbd_cos), col_top1(base),
        base_mc,
        np.full(N, np.log1p(N), np.float32),
        np.full(N, np.log1p(T), np.float32),
    ], axis=-1).astype(np.float32)

    return {
        "pair_feats": pair,
        "track_feats": trk,
        "cand_feats": cand,
        "base": base,
        "iou": iou.astype(np.float32),
        "iou_pred": iou_pred.astype(np.float32),
        "pbd_cos": pbd_cos,
        "pbd_anchor_cos": pbd_anchor_cos,
    }


class L1DAssociator(nn.Module):
    """Evidence-gated set-level residual associator."""

    def __init__(self, d_model: int = 128, n_layers: int = 2,
                 n_heads: int = 4, ffn_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.delta_scale = DELTA_SCALE

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

        self.pair_head = nn.Sequential(
            nn.Linear(2 * d_model + len(PAIR_FEATURES), 192),
            nn.LayerNorm(192), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(192, 96), nn.GELU(),
            nn.Linear(96, 1))
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
        """batch keys:
        pair_feats [B,T,N,Fp], track_feats [B,T,Ft], cand_feats [B,N,Fc],
        base [B,T,N], trk_mask [B,T] (True valid), cand_mask [B,N].
        """
        pair = torch.nan_to_num(batch["pair_feats"].float())
        trk_f = torch.nan_to_num(batch["track_feats"].float())
        cand_f = torch.nan_to_num(batch["cand_feats"].float())
        base = torch.nan_to_num(batch["base"].float())
        B, T, N, Fp = pair.shape
        trk_mask = batch.get("trk_mask")
        cand_mask = batch.get("cand_mask")
        if trk_mask is None:
            trk_mask = torch.ones(B, T, dtype=torch.bool, device=pair.device)
        if cand_mask is None:
            cand_mask = torch.ones(B, N, dtype=torch.bool, device=pair.device)

        trk_tok = self.track_proj(trk_f)  # [B,T,D]
        cand_tok = self.cand_proj(cand_f)  # [B,N,D]
        tokens = torch.cat([cand_tok, trk_tok], dim=1)  # [B,N+T,D]
        mask = torch.cat([cand_mask, trk_mask], dim=1)
        out = self.set_encoder(tokens, src_key_padding_mask=~mask)
        cand_out = out[:, :N]
        trk_out = out[:, N:]

        # [B,T,N,D]
        t_exp = trk_out.unsqueeze(2).expand(B, T, N, self.d_model)
        c_exp = cand_out.unsqueeze(1).expand(B, T, N, self.d_model)
        pair_in = torch.cat([t_exp, c_exp, pair], dim=-1)
        delta = self.delta_scale * torch.tanh(self.pair_head(pair_in).squeeze(-1))

        row_sel = torch.stack([
            trk_f[..., 9], trk_f[..., 10], trk_f[..., 11], trk_f[..., 12],
            trk_f[..., 13], trk_f[..., 6],
            pair.max(dim=2).values[..., 11],  # base_margin_row max == same as trk_f[...,12]
        ], dim=-1)
        rel_in = torch.cat([trk_out, row_sel], dim=-1)
        rel_logit = self.reliability_head(rel_in).squeeze(-1)  # [B,T]

        final = base + torch.sigmoid(rel_logit).unsqueeze(-1) * delta
        return {
            "final": final,
            "delta": delta,
            "reliability_logit": rel_logit,
            "reliability": torch.sigmoid(rel_logit),
            "trk_tok": trk_out,
            "cand_tok": cand_out,
        }


def l1d_loss(batch, pred, rel_weight: float = 0.3, pres_weight: float = 0.1,
             pos_weight: float = 9.0):
    """Main row/col assignment CE + reliability BCE + residual preservation."""
    final = pred["final"]
    B, T, N = final.shape
    row_label = batch.get("row_label", torch.full((B, T), -1, dtype=torch.long, device=final.device))
    col_label = batch.get("col_label", torch.full((B, N), -1, dtype=torch.long, device=final.device))
    base_correct = batch.get("base_correct", torch.ones(B, T, dtype=torch.bool, device=final.device))
    valid_row = (row_label >= 0)
    valid_col = (col_label >= 0)

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

    # reliability: base row argmax wrong (only supervise rows with label)
    if n_row:
        base_wrong = (~base_correct).float()
        rel = F.binary_cross_entropy_with_logits(
            pred["reliability_logit"][valid_row],
            base_wrong[valid_row], pos_weight=final.new_tensor(pos_weight))
        loss = loss + rel_weight * rel

    # preservation: keep residual small on rows where base was correct
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
