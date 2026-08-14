"""Stage L6 UIDM: Learned Causal Identity Dynamics Model.

Scientific claim (working name, subject to novelty audit):
heterogeneous MOT can be solved by one shared checkpoint that learns a
causal identity-dynamics process over a set of interacting trajectories.

Architecture (clean reimplementation; design evidence in
docs/l6_uidm_design.md and docs/l6_iclr_method_audit.md):
  per-track persistent memory h_i^t + permanent anchor a_i
  + per-frame set-of-sequences interaction (tracks + candidates)
  + identity transition decoder (continue / NEW / NO-MATCH)
  + learned lifecycle (alive logit, memory decay, birth, termination)
  + motion prediction head (Kalman/IoU/PBD remain *evidence inputs*)

References inspected (official repos, no code copied):
  - UniTrack ICLR 2026 (MIT, afdd986): tracking-level hinge criterion;
    our soft switch/FP loss follows the same scientific principle.
  - Samba ICLR 2025 (AGPL, f1c139a): synchronized set-of-sequences states;
    we implement synchronization with standard attention (license-safe).
  - MOTIP CVPR 2025 (Apache-2.0, ffc0e90): sequence-local identity prompts,
    NEW as explicit output, causal history attention.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from locatemot.models.l1d_association import (
    CAND_FEATURES,
    PAIR_FEATURES,
    TRACK_FEATURES,
    PBD_DIM,
    compute_affinity_features,
)


def l2norm(x, dim=-1):
    return F.normalize(x.float(), dim=dim, eps=1e-6)


class PBDEncoder(nn.Module):
    """Appearance-token evidence (PBD 2048 or frozen CLIP 512) -> d_model."""

    def __init__(self, d_model=320, dropout=0.1, in_dim=2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2 * d_model),
            nn.LayerNorm(2 * d_model), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model))

    def forward(self, pbd):
        return self.mlp(l2norm(pbd))


class EvidenceMLP(nn.Module):
    def __init__(self, in_dim, d_model, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.mlp(x)


class MemoryCell(nn.Module):
    """Gated recurrent identity-memory update (matched / decay / init)."""

    def __init__(self, d_model):
        super().__init__()
        self.gru_cell = nn.GRUCell(d_model * 2, d_model)
        self.decay_mlp = nn.Sequential(
            nn.Linear(d_model + 32, d_model), nn.GELU(),
            nn.Linear(d_model, d_model))
        self.init_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model))

    def update(self, h, obs, ctx):
        """h: [...,d]; obs/ctx: [...,d] -> new h (same shape)."""
        shape = h.shape
        inp = torch.cat([obs, ctx], dim=-1).reshape(-1, shape[-1] * 2)
        hf = h.reshape(-1, shape[-1])
        return self.gru_cell(inp, hf).reshape(shape)

    def decay(self, h, gap):
        """Gap: [...,1] float frame gap -> decayed state."""
        shape = h.shape
        gap_emb = self._gap_embed(gap)
        if h.dim() == 1:
            gap_emb = gap_emb.reshape(-1)
        out = h.reshape(-1, shape[-1]) + self.decay_mlp(
            torch.cat([h, gap_emb], dim=-1).reshape(-1, shape[-1] + 32))
        return out.reshape(shape)

    def init(self, obs):
        shape = obs.shape
        return self.init_mlp(obs.reshape(-1, shape[-1])).reshape(shape)

    def _gap_embed(self, gap):
        # 16-dim sinusoidal embedding; input [...,1]
        gap = gap.unsqueeze(-1) if gap.dim() == 1 else gap
        shape = gap.shape
        gapf = gap.reshape(-1, 1)
        device = gap.device
        i = torch.arange(16, device=device).float()
        ang = gapf * (1.0 / (10000 ** (i[None] / 16.0)))
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return emb.reshape(*shape, 32)


class UIDM(nn.Module):
    """Per-frame transition decoder; the rollout engine drives state."""

    def __init__(self, d_model=320, n_layers=4, n_heads=8, ffn_dim=1280,
                 dropout=0.1, max_obs_gap=30.0, no_interaction=False,
                 use_cue_rel=False, app_dim=2048):
        super().__init__()
        self.d_model = d_model
        self.app_dim = app_dim
        self.max_obs_gap = max_obs_gap
        self.no_interaction = no_interaction
        self.use_cue_rel = use_cue_rel
        self.pbd_encoder = PBDEncoder(d_model, dropout, app_dim)
        self.cand_mlp = EvidenceMLP(len(CAND_FEATURES), d_model, dropout)
        self.track_mlp = EvidenceMLP(len(TRACK_FEATURES), d_model, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ffn_dim, dropout, batch_first=True,
            norm_first=True, activation="gelu")
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pair_head = nn.Sequential(
            nn.Linear(2 * d_model + len(PAIR_FEATURES), 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1))
        if self.use_cue_rel:
            # Decision-level cue experts; indices index PAIR_FEATURES.
            self.cue_groups = {
                "motion": [1, 4, 7],            # iou_pred, cd_pred, gap
                "geometry": [0, 3, 5, 15],      # iou, cd, log_scale, cand_size
                "appearance": [2, 18],          # pbd_cos, pbd_anchor_cos
                "competition": [9, 10, 11, 12, 13, 14, 8, 17],
                "memory": [16],                 # + anchor_cos_cur, hits
            }
            self.cue_heads = nn.ModuleDict({
                name: nn.Sequential(
                    nn.Linear(2 * d_model + len(idxs) +
                              (2 if name == "memory" else 0), 128),
                    nn.GELU(), nn.Linear(128, 1))
                for name, idxs in self.cue_groups.items()
            })
            # router context: gap, track_age, log_n_cand, base row margin,
            # anchor_cos_cur, hits.
            self.reliability_head = nn.Sequential(
                nn.Linear(2 * d_model + 6, 128), nn.LayerNorm(128), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(128, len(self.cue_groups)))
        else:
            self.cue_groups = {}
            self.cue_heads = nn.ModuleDict()
            self.reliability_head = None
        self.no_match_head = nn.Sequential(
            nn.Linear(d_model + len(TRACK_FEATURES), 128),
            nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 1))
        self.new_head = nn.Sequential(
            nn.Linear(d_model + len(CAND_FEATURES), 128),
            nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 1))
        self.alive_head = nn.Sequential(
            nn.Linear(d_model + len(TRACK_FEATURES) + 1, 128),
            nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 1))
        self.motion_head = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(),
            nn.Linear(256, 4))
        self.memory = MemoryCell(d_model)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # motion head starts near zero (predict ~ no change)
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)
        # transition heads: start with mild matching preference so inference
        # does not fragment everything into NEW before training calibrates
        nn.init.zeros_(self.pair_head[-1].weight)
        nn.init.zeros_(self.pair_head[-1].bias)
        nn.init.zeros_(self.no_match_head[-1].weight)
        nn.init.zeros_(self.no_match_head[-1].bias)
        nn.init.constant_(self.no_match_head[-1].bias, -0.5)
        nn.init.zeros_(self.new_head[-1].weight)
        nn.init.constant_(self.new_head[-1].bias, -1.5)
        nn.init.zeros_(self.alive_head[-1].weight)
        nn.init.constant_(self.alive_head[-1].bias, 1.0)
        if self.use_cue_rel:
            for head in self.cue_heads.values():
                nn.init.xavier_uniform_(head[0].weight)
                nn.init.zeros_(head[0].bias)
                nn.init.zeros_(head[2].weight)
                nn.init.zeros_(head[2].bias)
            nn.init.xavier_uniform_(self.reliability_head[0].weight)
            nn.init.zeros_(self.reliability_head[0].bias)
            nn.init.zeros_(self.reliability_head[-1].weight)
            nn.init.zeros_(self.reliability_head[-1].bias)

    def forward_frame(self, frame):
        """One frame. frame: dict with torch tensors:
        cand_pbd [B,N,D], cand_feat [B,N,Fc], pair_feats [B,T,N,Fp],
        track_feats [B,T,Ft], cand_mask [B,N], trk_mask [B,T].
        Returns dict with pair_logits [B,T,N], no_match [B,T],
        new [B,N], alive_pre [B,T], pred_box [B,T,4] (normalized),
        trk_tok [B,T,d], cand_tok [B,N,d].
        """
        B, T, N = frame["pair_feats"].shape[:3]
        cand_pbd = torch.nan_to_num(frame["cand_pbd"].float())
        cand_feat = torch.nan_to_num(frame["cand_feat"].float())
        track_feat = torch.nan_to_num(frame["track_feats"].float())
        pair = torch.nan_to_num(frame["pair_feats"].float())
        trk_tok0 = frame["trk_tok"].float()  # [B,T,d]
        cand_tok0 = self.pbd_encoder(cand_pbd) + self.cand_mlp(cand_feat)
        trk_tok = trk_tok0 + self.track_mlp(track_feat)
        cand_mask = frame.get("cand_mask")
        if cand_mask is None:
            cand_mask = torch.ones(B, N, dtype=torch.bool, device=pair.device)
        trk_mask = frame.get("trk_mask")
        if trk_mask is None:
            trk_mask = torch.ones(B, T, dtype=torch.bool, device=pair.device)
        if self.no_interaction:
            trk_out = trk_tok
            cand_out = cand_tok0
        else:
            tokens = torch.cat([trk_tok, cand_tok0], dim=1)
            mask = torch.cat([trk_mask, cand_mask], dim=1)
            out = self.set_encoder(tokens, src_key_padding_mask=~mask)
            trk_out = out[:, :T]
            cand_out = out[:, T:]

        t_exp = trk_out.unsqueeze(2).expand(B, T, N, self.d_model)
        c_exp = cand_out.unsqueeze(1).expand(B, T, N, self.d_model)
        if self.use_cue_rel and getattr(self, "use_cue_mix", True):
            cue_scores = []
            for name, idxs in self.cue_groups.items():
                feats = pair[..., idxs]
                if name == "memory":
                    extra = torch.stack([
                        track_feat[:, :, 15:16].expand(B, T, N),
                        track_feat[:, :, 8:9].expand(B, T, N),
                    ], dim=-1)
                    feats = torch.cat([feats, extra], dim=-1)
                cue_scores.append(self.cue_heads[name](
                    torch.cat([t_exp, c_exp, feats], dim=-1)).squeeze(-1))
            cue_scores = torch.stack(cue_scores, dim=-1)  # [B,T,N,K]
            cue_scores = cue_scores.masked_fill(
                ~cand_mask.unsqueeze(1).unsqueeze(-1), -1e9)
            cue_scores = cue_scores.masked_fill(
                ~trk_mask.unsqueeze(2).unsqueeze(-1), -1e9)
            rel_ctx = torch.stack([
                pair[..., 7], pair[..., 16], pair[..., 8], pair[..., 11],
                track_feat[:, :, 15:16].expand(B, T, N),
                track_feat[:, :, 8:9].expand(B, T, N),
            ], dim=-1)
            rel_logits = self.reliability_head(
                torch.cat([t_exp, c_exp, rel_ctx], dim=-1))
            rel_w = torch.softmax(rel_logits, dim=-1)
            pair_mix = (rel_w * cue_scores).sum(-1)
            ctx = self.pair_head(
                torch.cat([t_exp, c_exp, pair], dim=-1)).squeeze(-1)
            pair_logits = pair_mix + ctx
        else:
            pair_in = torch.cat([t_exp, c_exp, pair], dim=-1)
            pair_logits = self.pair_head(pair_in).squeeze(-1)
        pair_logits = pair_logits.masked_fill(~cand_mask.unsqueeze(1), -1e9)
        pair_logits = pair_logits.masked_fill(~trk_mask.unsqueeze(2), -1e9)

        no_match = self.no_match_head(
            torch.cat([trk_out, track_feat], dim=-1)).squeeze(-1)
        no_match = no_match.masked_fill(~trk_mask, -1e9)
        new = self.new_head(
            torch.cat([cand_out, cand_feat], dim=-1)).squeeze(-1)
        new = new.masked_fill(~cand_mask, -1e9)
        alive_pre = self.alive_head(torch.cat(
            [trk_out, track_feat,
             torch.clamp(frame["gap"].float(), 0, self.max_obs_gap)
             .unsqueeze(-1)], dim=-1)).squeeze(-1)
        pred_box = self.motion_head(trk_out)  # normalized corners
        out = {
            "pair_logits": pair_logits,
            "no_match": no_match,
            "new": new,
            "alive_pre": alive_pre,
            "pred_box": pred_box,
            "trk_tok": trk_out,
            "cand_tok": cand_out,
        }
        if self.use_cue_rel and getattr(self, "use_cue_mix", True):
            out["cue_scores"] = cue_scores
            out["rel_logits"] = rel_logits
            out["cue_rel"] = rel_w
        return out

    def forward(self, frame):
        return self.forward_frame(frame)


def uidm_frame_loss(frame, pred, tgt):
    """Losses for one frame (all tensors on device).

    tgt keys: row_target [B,T] (0..N-1 or N), col_target [B,N] (0..T-1 or T),
    row_valid [B,T], col_valid [B,N], alive_target [B,T], match_box [B,T,4]
    (normalized, -1 if none), row_box_valid [B,T].
    """
    pair = pred["pair_logits"]
    no_match = pred["no_match"]
    new = pred["new"]
    B, T, N = pair.shape
    dev = pair.device
    loss_row = pair.new_zeros(())
    loss_col = pair.new_zeros(())
    n_row = 0
    n_col = 0

    # row CE over candidates + NO_MATCH
    row_logits = torch.cat([pair, no_match.unsqueeze(-1)], dim=-1)  # [B,T,N+1]
    rv = tgt["row_valid"]
    if rv.any():
        rt = tgt["row_target"].clamp(0, N)
        loss_row = F.cross_entropy(
            row_logits[rv], rt[rv], reduction="mean")
        n_row = int(rv.sum())

    # col CE over tracks + NEW
    col_logits = torch.cat([pair.transpose(1, 2), new.unsqueeze(-1)], dim=-1)
    cv = tgt["col_valid"]
    if cv.any():
        ct = tgt["col_target"].clamp(0, T)
        loss_col = F.cross_entropy(
            col_logits[cv], ct[cv], reduction="mean")
        n_col = int(cv.sum())

    # lifecycle BCE
    nm_t = tgt["no_match_target"]
    new_t = tgt["new_target"]
    alive_t = tgt["alive_target"]
    loss_nm = F.binary_cross_entropy_with_logits(
        no_match[rv], nm_t[rv].float()) if rv.any() else row_logits.new_zeros(())
    loss_new = F.binary_cross_entropy_with_logits(
        new[cv], new_t[cv].float()) if cv.any() else row_logits.new_zeros(())
    loss_alive = F.binary_cross_entropy_with_logits(
        pred["alive_pre"][rv], alive_t[rv].float()) if rv.any() \
        else row_logits.new_zeros(())

    # motion L1 (matched rows only)
    mb = tgt["match_box"]
    bv = tgt["row_box_valid"]
    loss_motion = (pred["pred_box"][bv] - mb[bv]).abs().mean() if bv.any() \
        else row_logits.new_zeros(())

    # decision-level cue reliability: on the GT-matched candidate row, the
    # router should upweight the cues whose own top-1 vote is the true target.
    # Soft-target CE (bounded, avoids logit divergence).
    loss_rel = row_logits.new_zeros(())
    if "cue_scores" in pred:
        cue_valid = rv & (tgt["row_target"] < N)
        if cue_valid.any():
            cue_scores = pred["cue_scores"][cue_valid]  # [Rv,N,K]
            gt_idx = tgt["row_target"][cue_valid]       # [Rv]
            top1 = cue_scores.argmax(dim=1)  # [Rv,K]
            vote = (top1 == gt_idx[:, None]).float()  # [Rv,K]
            n_votes = vote.sum(dim=1, keepdim=True).clamp(min=1.0)
            q = vote / n_votes  # uniform fallback when no cue votes correctly
            rel_logits_gt = torch.gather(
                pred["rel_logits"][cue_valid], 1,
                gt_idx[:, None, None].expand(-1, 1, q.shape[-1])
            ).squeeze(1)  # [Rv,K]
            loss_rel = F.cross_entropy(rel_logits_gt, q)

    # soft switch / FP margin (UniTrack-style, vectorised)
    col_probs = torch.softmax(col_logits, dim=-1)  # [B,N,T+1]
    correct = col_probs[torch.arange(B, device=dev)[:, None],
                        torch.arange(N, device=dev)[None, :],
                        tgt["col_target"].clamp(0, T)]
    incorrect = col_probs.clone()
    bad = torch.arange(T + 1, device=dev)[None, None, :].expand(B, N, T + 1)
    target_idx = tgt["col_target"].clamp(0, T).unsqueeze(-1)
    incorrect = incorrect.masked_fill(bad == target_idx, -1.0)
    if cv.any():
        margin = (incorrect.max(dim=-1).values - correct + 0.1).clamp(min=0)
        margin = margin * cv.float()
        loss_switch = margin.sum() / max(1, int(cv.sum()))
    else:
        loss_switch = row_logits.new_zeros(())

    return {
        "loss_row": loss_row, "loss_col": loss_col,
        "loss_nm": loss_nm, "loss_new": loss_new, "loss_alive": loss_alive,
        "loss_motion": loss_motion, "loss_switch": loss_switch,
        "loss_rel": loss_rel,
        "n_row": n_row, "n_col": n_col,
    }


def uidm_total_loss(l, w_life=0.3, w_motion=0.3, w_switch=0.5, w_rel=0.1):
    return (l["loss_row"] + l["loss_col"]
            + w_life * (l["loss_nm"] + l["loss_new"] + l["loss_alive"])
            + w_motion * l["loss_motion"] + w_switch * l["loss_switch"]
            + w_rel * l["loss_rel"])


def decode_lsa(pair, no_match, new):
    """Hard one-to-one decode from logits (numpy).

    pair [T,N], no_match [T], new [N].  Rows = candidates (N), columns =
    T tracks + N NEW dummies.  Returns (matches [(t,c,score)], births [c]).
    """
    from scipy.optimize import linear_sum_assignment
    T, N = pair.shape
    if T == 0:
        return [], list(range(N))
    cost = np.full((N, T + N), 1e6, dtype=np.float64)
    for j in range(N):
        for i in range(T):
            cost[j, i] = -float(pair[i, j])
        cost[j, T + j] = -float(new[j])
    rows, cols = linear_sum_assignment(cost)
    matches = []
    births = []
    for r, c in zip(rows, cols):
        if c < T:
            matches.append((int(c), int(r), float(pair[c, r])))
        else:
            births.append(int(r))
    return matches, births


def compute_frame_features(track_boxes, cand_boxes, ref_pbd, anchor_pbd,
                           cand_pbd, cand_gen, gaps, ages, hits, prev_boxes,
                           image_size, weights=(0.4, 0.2, 0.4),
                           motion_pred_boxes=None):
    """Numpy evidence features (same schema as L1DK)."""
    return compute_affinity_features(
        track_boxes, cand_boxes, ref_pbd, anchor_pbd, cand_pbd, cand_gen,
        gaps, ages, hits, prev_boxes, weights, image_size,
        motion_pred_boxes=motion_pred_boxes)


def compute_affinity_features_torch(
    tb, pb, cb, ref, anchor, cp, cg, gaps, ages, hits, image_size,
    weights=(0.4, 0.2, 0.4), motion_pred=None,
):
    """Batched, GPU-friendly version of compute_affinity_features.

    Shapes: tb/pb/cb [B,S,4]/[B,N,4], ref/anchor [B,S,2048],
    cp [B,N,2048], cg [B,N], gaps/ages/hits [B,S], image_size [B,2].
    Returns (pair_feats [B,S,N,19], track_feats [B,S,16],
             cand_feats [B,N,12], base [B,S,N]).
    """
    B, S, _ = tb.shape
    N = cp.shape[1]
    iw = image_size[:, 0].float().unsqueeze(-1)  # [B,1]
    ih = image_size[:, 1].float().unsqueeze(-1)
    diag1 = (torch.sqrt(iw ** 2 + ih ** 2) + 1e-6).unsqueeze(-1)  # [B,1,1]
    diag2 = diag1.squeeze(-1)  # [B,1]
    wi, wp, wm = weights

    gaps = gaps.float()
    ages = ages.float()
    hits = hits.float()
    cg = cg.float()
    if motion_pred is None:
        vel = tb - pb
        pred = tb + vel * torch.clamp(gaps, min=1.0).unsqueeze(-1)
    else:
        pred = motion_pred

    def iou_mat(a, b):
        # a [B,S,4], b [B,N,4]
        ix1 = torch.max(a[:, :, None, 0], b[:, None, :, 0])
        iy1 = torch.max(a[:, :, None, 1], b[:, None, :, 1])
        ix2 = torch.min(a[:, :, None, 2], b[:, None, :, 2])
        iy2 = torch.min(a[:, :, None, 3], b[:, None, :, 3])
        iw_ = torch.clamp(ix2 - ix1, min=0)
        ih_ = torch.clamp(iy2 - iy1, min=0)
        inter = iw_ * ih_
        ar = torch.clamp(a[:, :, 2] - a[:, :, 0], min=0) * \
            torch.clamp(a[:, :, 3] - a[:, :, 1], min=0)
        ac = torch.clamp(b[:, :, 2] - b[:, :, 0], min=0) * \
            torch.clamp(b[:, :, 3] - b[:, :, 1], min=0)
        union = ar[:, :, None] + ac[:, None, :] - inter
        return torch.where(union > 0, inter / union.clamp(min=1e-9),
                           torch.zeros_like(inter))

    iou = iou_mat(tb, cb)
    iou_pred = iou_mat(pred, cb)

    ref_n = l2norm(ref)
    anchor_n = l2norm(anchor)
    cand_n = l2norm(cp)
    pbd_cos = torch.bmm(ref_n, cand_n.transpose(1, 2))
    pbd_anchor_cos = torch.bmm(anchor_n, cand_n.transpose(1, 2))
    anchor_cos_cur = (anchor_n * ref_n).sum(-1)

    tc = (tb[:, :, :2] + tb[:, :, 2:]) / 2.0
    pc = (pred[:, :, :2] + pred[:, :, 2:]) / 2.0
    cc = (cb[:, :, :2] + cb[:, :, 2:]) / 2.0
    cd = torch.sqrt(((tc[:, :, None, :] - cc[:, None, :, :]) ** 2).sum(-1)) \
        / diag1
    cd_pred = torch.sqrt(((pc[:, :, None, :] - cc[:, None, :, :]) ** 2)
                         .sum(-1)) / diag1

    tw = tb[:, :, 2] - tb[:, :, 0]
    th = tb[:, :, 3] - tb[:, :, 1]
    cw = cb[:, :, 2] - cb[:, :, 0]
    ch = cb[:, :, 3] - cb[:, :, 1]
    log_scale = torch.sqrt(
        torch.log(torch.clamp(cw[:, None, :], min=1.0) / torch.clamp(tw[:, :, None],
                                                         min=1.0) + 1e-6) ** 2
        + torch.log(torch.clamp(ch[:, None, :], min=1.0) / torch.clamp(th[:, :, None],
                                                           min=1.0)
                    + 1e-6) ** 2)
    cand_size = torch.sqrt(torch.clamp(cw, min=0) * torch.clamp(ch, min=0)) \
        / diag2
    log_n_cand = torch.full((B, S, N), float(np.log1p(N)),
                            device=tb.device)
    log_n_track = torch.full((B, S, N), float(np.log1p(S)),
                             device=tb.device)

    base = wi * iou + wp * pbd_cos + wm * iou_pred

    def margin_row(mat):
        if mat.shape[2] < 2:
            return torch.zeros(B, S, device=mat.device)
        s = torch.topk(mat, 2, dim=2).values
        return (s[:, :, 0] - s[:, :, 1])

    def margin_col(mat):
        if mat.shape[1] < 2:
            return torch.zeros(B, N, device=mat.device)
        s = torch.topk(mat, 2, dim=1).values
        return (s[:, 0, :] - s[:, 1, :])

    iou_mr = margin_row(iou)
    pbd_mr = margin_row(pbd_cos)
    base_mr = margin_row(base)
    iou_mc = margin_col(iou)
    pbd_mc = margin_col(pbd_cos)
    base_mc = margin_col(base)

    pair = torch.stack([
        iou, iou_pred, pbd_cos, cd, cd_pred, log_scale,
        cg[:, None, :].expand(B, S, N),
        gaps[:, :, None].expand(B, S, N),
        log_n_cand,
        iou_mr[:, :, None].expand(B, S, N),
        pbd_mr[:, :, None].expand(B, S, N),
        base_mr[:, :, None].expand(B, S, N),
        iou_mc[:, None, :].expand(B, S, N),
        pbd_mc[:, None, :].expand(B, S, N),
        base_mc[:, None, :].expand(B, S, N),
        cand_size[:, None, :].expand(B, S, N),
        ages[:, :, None].expand(B, S, N),
        log_n_track,
        pbd_anchor_cos,
    ], dim=-1)

    def row_top1(mat):
        return mat.max(dim=2).values

    def col_top1(mat):
        return mat.max(dim=1).values

    tb_n = torch.stack([tb[:, :, 0] / iw, tb[:, :, 1] / ih,
                        tb[:, :, 2] / iw, tb[:, :, 3] / ih], dim=-1)
    vel_n = torch.stack([vel[:, :, 0] / iw,
                         vel[:, :, 1] / ih], dim=-1)
    trk = torch.stack([
        tb_n[:, :, 0], tb_n[:, :, 1], tb_n[:, :, 2], tb_n[:, :, 3],
        vel_n[:, :, 0], vel_n[:, :, 1],
        gaps, ages, hits,
        row_top1(iou), row_top1(pbd_cos), row_top1(base), base_mr,
        torch.full((B, S), float(np.log1p(N)), device=tb.device),
        torch.full((B, S), float(np.log1p(S)), device=tb.device),
        anchor_cos_cur,
    ], dim=-1)

    cb_n = torch.stack([cb[:, :, 0] / iw,
                        cb[:, :, 1] / ih,
                        cb[:, :, 2] / iw,
                        cb[:, :, 3] / ih], dim=-1)
    cand = torch.stack([
        cb_n[:, :, 0], cb_n[:, :, 1], cb_n[:, :, 2], cb_n[:, :, 3],
        cg, cand_size,
        col_top1(iou), col_top1(pbd_cos), col_top1(base), base_mc,
        torch.full((B, N), float(np.log1p(N)), device=tb.device),
        torch.full((B, N), float(np.log1p(S)), device=tb.device),
    ], dim=-1)
    return pair, trk, cand, base
